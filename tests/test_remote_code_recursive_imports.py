"""ensure_recursive_remote_code_imports: make transformers' dynamic-module
`check_imports` return the FULL recursive relative-import closure.

transformers stages a trust_remote_code modeling file plus the module names
`check_imports` returns (DIRECT relative imports only) into a per-hash cache dir,
but resolves the graph RECURSIVELY at import time. A 2-level custom-code chain
(Puzzle: modeling_nemotron_h_puzzle -> modeling_nemotron_h -> configuration_nemotron_h)
therefore leaves the transitively-imported base file un-staged and raises
FileNotFoundError. This test proves the patch closes that gap and is idempotent,
without loading any model or touching a GPU.
"""
from __future__ import annotations

import pytest

dmu = pytest.importorskip("transformers.dynamic_module_utils")

from nvfp4_lora.loader import ensure_recursive_remote_code_imports  # noqa: E402


def _write(p, text):
    p.write_text(text)
    return p


def test_check_imports_returns_recursive_closure(tmp_path):
    # main -> mid -> leaf, a 2-level relative-import chain (mirrors Puzzle's shape).
    _write(tmp_path / "main.py", "from .mid import thing\n")
    _write(tmp_path / "mid.py", "from .leaf import other\n")
    _write(tmp_path / "leaf.py", "VALUE = 1\n")

    # Unpatched, transformers stages only the DIRECT import of main (`mid`); the
    # transitively-needed `leaf` is missed. Capture that baseline first.
    if not getattr(dmu.check_imports, "_nvfp4_recursive", False):
        direct = set(dmu.check_imports(str(tmp_path / "main.py")))
        assert "leaf" not in direct  # this is exactly the gap the patch closes

    ensure_recursive_remote_code_imports()
    closure = set(dmu.check_imports(str(tmp_path / "main.py")))
    assert {"mid", "leaf"} <= closure


def test_patch_is_idempotent():
    ensure_recursive_remote_code_imports()
    first = dmu.check_imports
    ensure_recursive_remote_code_imports()
    assert dmu.check_imports is first
    assert getattr(dmu.check_imports, "_nvfp4_recursive", False) is True
