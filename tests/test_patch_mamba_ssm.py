"""CPU tests for scripts/patch_mamba_ssm.py.

The patcher edits mamba-ssm's installed source in place so NemotronH-family
checkpoints load in a triton-only venv (no `selective_scan_cuda` extension,
transformers 5.x). Because pip upgrades silently revert those edits, the patcher
has to be safe to re-run unconditionally from a preflight. These tests build a
fixture copy of the two upstream (mamba-ssm 2.2.5) files and prove:

  * a first run applies both patches,
  * a second run is a byte-exact no-op (idempotent),
  * --check reports the state without writing and exits 3 when unpatched,
  * --dry-run writes nothing,
  * the patched output is valid Python and still guards the right imports,
  * an unrecognized mamba-ssm version fails loudly instead of silently no-oping.

No GPU, no model, and no real mamba_ssm install required.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "patch_mamba_ssm.py"


def _load_patcher():
    spec = importlib.util.spec_from_file_location("_patch_mamba_ssm_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


patcher = _load_patcher()


# Verbatim upstream mamba-ssm 2.2.5 content for the two patched files, trimmed to the
# regions the patcher anchors on. The bodies below the anchors stand in for the rest of
# the module; the patcher must not touch them.
UPSTREAM_INIT = '''__version__ = "2.2.5"

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
'''

UPSTREAM_SCAN = '''# Copyright (c) 2023, Tri Dao, Albert Gu.

import torch
import torch.nn.functional as F
from mamba_ssm.utils.torch import custom_bwd, custom_fwd

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

from mamba_ssm.ops.triton.layer_norm import _layer_norm_fwd

import selective_scan_cuda


class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u):
        return selective_scan_cuda.fwd(u)
'''


@pytest.fixture
def pkg(tmp_path):
    """A pristine upstream-shaped mamba_ssm package tree."""
    root = tmp_path / "mamba_ssm"
    (root / "ops").mkdir(parents=True)
    (root / "__init__.py").write_text(UPSTREAM_INIT, encoding="utf-8")
    (root / "ops" / "selective_scan_interface.py").write_text(UPSTREAM_SCAN, encoding="utf-8")
    return root


def _snapshot(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): p.read_text(encoding="utf-8")
        for p in sorted(root.rglob("*.py"))
    }


def test_apply_then_reapply_is_idempotent(pkg, capsys):
    assert patcher.apply(pkg) == 0
    after_first = _snapshot(pkg)
    assert all(patcher.MARKER in text for text in after_first.values())

    # A second (and third) run must not change a single byte.
    assert patcher.apply(pkg) == 0
    assert _snapshot(pkg) == after_first
    assert patcher.apply(pkg) == 0
    assert _snapshot(pkg) == after_first

    out = capsys.readouterr().out
    assert "already applied" in out


def test_first_run_actually_changes_both_files(pkg):
    before = _snapshot(pkg)
    patcher.apply(pkg)
    after = _snapshot(pkg)
    assert set(before) == set(after)
    changed = [name for name in before if before[name] != after[name]]
    assert sorted(changed) == ["__init__.py", "ops/selective_scan_interface.py"]


def test_patched_output_is_valid_python_and_guards_the_imports(pkg):
    patcher.apply(pkg)

    init_src = (pkg / "__init__.py").read_text(encoding="utf-8")
    scan_src = (pkg / "ops" / "selective_scan_interface.py").read_text(encoding="utf-8")

    # Parses (the whole point of an in-place source edit is that it stays valid).
    init_tree = ast.parse(init_src)
    scan_tree = ast.parse(scan_src)

    # The four eager model imports are now inside a Try, not at module top level.
    top_level_imports = [n for n in init_tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert top_level_imports == []
    tries = [n for n in init_tree.body if isinstance(n, ast.Try)]
    assert len(tries) == 1
    guarded = [n for n in tries[0].body if isinstance(n, ast.ImportFrom)]
    assert len(guarded) == 4
    assert re.search(r"except Exception:\s*\n\s*pass", init_src)

    # selective_scan_cuda is imported under a try with a None fallback.
    assert "\nimport selective_scan_cuda\n" not in scan_src
    scan_tries = [n for n in scan_tree.body if isinstance(n, ast.Try)]
    names = [
        alias.name
        for node in scan_tries
        for stmt in node.body
        if isinstance(stmt, ast.Import)
        for alias in stmt.names
    ]
    assert "selective_scan_cuda" in names
    assert "selective_scan_cuda = None" in scan_src

    # Untouched body survives verbatim.
    assert "class SelectiveScanFn(torch.autograd.Function):" in scan_src
    assert '__version__ = "2.2.5"' in init_src


def test_check_mode_reports_and_writes_nothing(pkg, capsys):
    before = _snapshot(pkg)
    assert patcher.apply(pkg, check=True) == 3
    assert _snapshot(pkg) == before
    assert "MISSING" in capsys.readouterr().out

    patcher.apply(pkg)
    capsys.readouterr()
    after = _snapshot(pkg)
    assert patcher.apply(pkg, check=True) == 0
    assert _snapshot(pkg) == after


def test_dry_run_writes_nothing_but_reports_a_diff(pkg, capsys):
    before = _snapshot(pkg)
    assert patcher.apply(pkg, dry_run=True) == 0
    assert _snapshot(pkg) == before
    out = capsys.readouterr().out
    assert "would patch" in out
    assert "No files were written." in out
    assert "+try:" in out


def test_unknown_version_fails_loudly(pkg):
    # Anchor gone (an upstream release that already guards the import, say): the patcher
    # must raise rather than report a false success.
    (pkg / "ops" / "selective_scan_interface.py").write_text(
        UPSTREAM_SCAN.replace("\nimport selective_scan_cuda\n", "\nselective_scan_cuda = None\n"),
        encoding="utf-8",
    )
    with pytest.raises(patcher.PatchError, match="not the one this patcher was written for"):
        patcher.apply(pkg)


def test_missing_target_file_fails_loudly(pkg):
    (pkg / "ops" / "selective_scan_interface.py").unlink()
    with pytest.raises(patcher.PatchError, match="patch target missing"):
        patcher.apply(pkg)


def test_patch_text_helper_is_a_no_op_on_patched_input():
    patched, changed = patcher.patch_text(UPSTREAM_INIT, patcher.INIT_ANCHOR, patcher.INIT_REPLACEMENT)
    assert changed is True
    again, changed_again = patcher.patch_text(patched, patcher.INIT_ANCHOR, patcher.INIT_REPLACEMENT)
    assert changed_again is False
    assert again == patched


def test_cli_main_end_to_end(pkg, capsys):
    assert patcher.main(["--package-dir", str(pkg), "--check"]) == 3
    assert patcher.main(["--package-dir", str(pkg)]) == 0
    assert patcher.main(["--package-dir", str(pkg)]) == 0
    assert patcher.main(["--package-dir", str(pkg), "--check"]) == 0
    capsys.readouterr()
    # A bad target is a clean exit 2, not a traceback.
    assert patcher.main(["--package-dir", str(pkg / "does-not-exist")]) == 2
    assert "error:" in capsys.readouterr().err
