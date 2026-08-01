"""Idempotent source patcher for `mamba-ssm` in a triton-only training venv.

WHY THIS EXISTS
---------------
NemotronH-family checkpoints (`nemotron_h`, `nemotron_h_puzzle`, and the
NemotronH-Omni wrappers) import `mamba_ssm` for their Mamba2 mixer layers. On
DGX Spark / GB10 the practical install is triton-only: the `selective_scan_cuda`
C++/CUDA extension is a Mamba-1 kernel that is not built (and is not needed,
because these checkpoints run the Mamba2 triton path). Two upstream assumptions
in mamba-ssm 2.2.5 then make the package unimportable, so the model never loads:

  1. `mamba_ssm/ops/selective_scan_interface.py` does a bare, top-level
     `import selective_scan_cuda`. With no CUDA extension present this raises
     ImportError at import time.
  2. `mamba_ssm/__init__.py` eagerly imports the full Mamba-1/Mamba-2 model
     stack (`mamba_simple`, `mamba2`, `mixer_seq_simple`). That chain pulls in
     (1) above, and additionally reaches for transformers-4.x generation symbols
     that were removed in transformers 5.x. Either failure makes
     `import mamba_ssm` fail outright, even though every submodule the
     NemotronH families actually use (`mamba_ssm.ops.triton.*`) imports fine.

Both patches are pure guards: they wrap existing imports in try/except and set a
`None` fallback. Nothing that worked before stops working. On an install that
DOES have `selective_scan_cuda` built, both try-blocks succeed and the module
state is identical to upstream.

These edits live in installed site-packages, so **any `pip install -U mamba-ssm`
(or a venv rebuild) reverts them** and this script has to be re-run. Run it with
`--check` from a preflight to detect that state before paying for a model load.

USAGE
-----
    # patch the mamba_ssm installed in the current interpreter's venv
    python scripts/patch_mamba_ssm.py

    # show what would change, touch nothing
    python scripts/patch_mamba_ssm.py --dry-run

    # preflight: exit 0 if both patches are applied, 3 if not
    python scripts/patch_mamba_ssm.py --check

    # explicit target (e.g. patching another venv's site-packages)
    python scripts/patch_mamba_ssm.py --package-dir /path/to/site-packages/mamba_ssm

Exit codes: 0 success (or already patched), 2 target not found / anchor missing,
3 `--check` found an unpatched file.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

# Marker embedded in every patch. Its presence is what makes the patcher
# idempotent, and what `--check` looks for.
MARKER = "nvfp4-lora-spark: triton-only mamba-ssm guard"

# --------------------------------------------------------------------------------------
# Patch 1: mamba_ssm/ops/selective_scan_interface.py
# --------------------------------------------------------------------------------------

SCAN_REL_PATH = "ops/selective_scan_interface.py"

SCAN_ANCHOR = "\nimport selective_scan_cuda\n"

SCAN_REPLACEMENT = f"""
# {MARKER}
# selective_scan_cuda is the Mamba-1 CUDA extension and is absent from a triton-only
# install. NemotronH-family checkpoints use the Mamba2 triton path and never call into
# it, so degrade to None instead of making the whole module unimportable.
try:
    import selective_scan_cuda
except ImportError:
    selective_scan_cuda = None
"""

# --------------------------------------------------------------------------------------
# Patch 2: mamba_ssm/__init__.py
# --------------------------------------------------------------------------------------

INIT_REL_PATH = "__init__.py"

INIT_ANCHOR = """from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
"""

INIT_REPLACEMENT = f"""# {MARKER}
# These eager full-model imports pull the selective_scan_cuda extension (absent on a
# triton-only install) and transformers-4.x generation symbols that transformers 5.x
# removed. NemotronH-family checkpoints only need mamba_ssm.ops.triton.*, so keep the
# package importable and let those submodule imports succeed on their own.
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.modules.mamba2 import Mamba2
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except Exception:
    pass
"""

PATCHES = (
    ("selective_scan_cuda import guard", SCAN_REL_PATH, SCAN_ANCHOR, SCAN_REPLACEMENT),
    ("eager Mamba-1/Mamba-2 model import guard", INIT_REL_PATH, INIT_ANCHOR, INIT_REPLACEMENT),
)


class PatchError(RuntimeError):
    """A patch target exists but could not be patched safely."""


def find_package_dir(explicit: str | None = None) -> Path:
    """Locate the installed `mamba_ssm` package directory.

    With `--package-dir` this just validates the path. Otherwise it imports
    mamba_ssm's spec (without executing the package, which may currently be
    unimportable, which is the whole point) to find where it lives.
    """
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise PatchError(f"--package-dir is not a directory: {path}")
        return path

    import importlib.util

    try:
        spec = importlib.util.find_spec("mamba_ssm")
    except Exception as exc:  # a broken/partial install can raise here
        raise PatchError(f"could not locate mamba_ssm: {exc!r}") from exc
    if spec is None or not spec.submodule_search_locations:
        raise PatchError(
            "mamba_ssm is not installed in this interpreter "
            f"({sys.executable}); install it or pass --package-dir"
        )
    return Path(list(spec.submodule_search_locations)[0]).resolve()


def patch_text(text: str, anchor: str, replacement: str) -> tuple[str, bool]:
    """Return `(new_text, changed)`.

    Already-patched input (marker present) returns unchanged with
    `changed=False`. Raises PatchError if the file is neither patched nor
    matches the expected upstream anchor, so an unexpected mamba-ssm version
    fails loudly instead of being silently left broken.
    """
    if MARKER in text:
        return text, False
    count = text.count(anchor)
    if count != 1:
        raise PatchError(
            f"expected exactly 1 occurrence of the upstream anchor, found {count}. "
            "This mamba-ssm version is not the one this patcher was written for "
            "(2.2.5); re-derive the patch before using it."
        )
    return text.replace(anchor, replacement, 1), True


def apply(package_dir: Path, dry_run: bool = False, check: bool = False) -> int:
    """Apply (or check) every patch. Returns a process exit code."""
    unpatched = 0
    changed = 0

    for label, rel_path, anchor, replacement in PATCHES:
        target = package_dir / rel_path
        if not target.is_file():
            raise PatchError(f"patch target missing: {target}")

        original = target.read_text(encoding="utf-8")

        if check:
            if MARKER in original:
                print(f"ok        {rel_path}: {label} already applied")
            else:
                print(f"MISSING   {rel_path}: {label} NOT applied")
                unpatched += 1
            continue

        new_text, did_change = patch_text(original, anchor, replacement)
        if not did_change:
            print(f"unchanged {rel_path}: {label} already applied")
            continue

        if dry_run:
            print(f"would patch {rel_path}: {label}")
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{rel_path}",
                tofile=f"b/{rel_path}",
            )
            sys.stdout.writelines(diff)
        else:
            target.write_text(new_text, encoding="utf-8")
            print(f"patched   {rel_path}: {label}")
        changed += 1

    if check:
        if unpatched:
            print(
                f"\n{unpatched} patch(es) missing in {package_dir}. "
                "Run `python scripts/patch_mamba_ssm.py` (a pip upgrade of mamba-ssm "
                "reverts these in-place edits).",
                file=sys.stderr,
            )
            return 3
        print(f"\nAll patches present in {package_dir}.")
        return 0

    if changed == 0:
        print(f"\nNothing to do; {package_dir} is already patched.")
    elif dry_run:
        print(f"\n{changed} patch(es) would be applied to {package_dir}. No files were written.")
    else:
        print(f"\n{changed} patch(es) applied to {package_dir}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Idempotently guard mamba-ssm 2.2.5's CUDA-extension and eager model imports "
            "so NemotronH-family NVFP4 checkpoints load in a triton-only venv. "
            "Re-run after any pip upgrade of mamba-ssm, which reverts these edits."
        ),
    )
    ap.add_argument(
        "--package-dir",
        default=None,
        help="Path to the installed mamba_ssm package directory. Default: whatever "
             "the current interpreter would import.",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the diff that would be applied; write nothing.")
    ap.add_argument("--check", action="store_true",
                    help="Report whether the patches are applied and exit 3 if any is "
                         "missing. Writes nothing. Intended for a preflight.")
    args = ap.parse_args(argv)

    if args.dry_run and args.check:
        ap.error("--dry-run and --check are mutually exclusive")

    try:
        package_dir = find_package_dir(args.package_dir)
        print(f"target: {package_dir}")
        return apply(package_dir, dry_run=args.dry_run, check=args.check)
    except PatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
