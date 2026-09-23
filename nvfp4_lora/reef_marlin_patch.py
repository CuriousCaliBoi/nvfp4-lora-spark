"""Install and verify one source-pinned Marlin research patch without importing vLLM."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

MARLIN_SOURCE_PATH = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py"
MARLIN_MANIFEST_PATH = "/opt/nvfp4-marlin-order/manifest.json"
MARLIN_BASE_IMAGE_ID = "sha256:2b57e729b712509ed2eafbb49ca51d754a0f703e08b0b8b02bda8ab44f0a7925"
MARLIN_PATCH_MANIFEST = {
    "schema_version": 1,
    "patch_id": "marlin-moe-token-order",
    "upstream_url": "https://github.com/vllm-project/vllm/pull/52532",
    "upstream_commit": "e8a07dcccf8e48dd4b9b42a355fc6d5b9db59073",
    "upstream_patch_sha256": "9620420095ad7df41fca2dd922a48fcbd2e34bbd9e0be7ef8df701004ff94082",
    "source_path": MARLIN_SOURCE_PATH,
    "original_sha256": "7b2c444fd56462f98c16f25c4cfa4d7b95ea483b0529b15d0612f7dbc8b06b5c",
    "patched_sha256": "0d4f0d552cfb3b57ce327b748c4a4eece0c4228e0c7fc0f84fb389daa63d5ac1",
    "helper_sha256": "0a9fbd16915a968541a8ddb9a93a8302cfe40c8f0cd8e642d183da2c571568cf",
    "vllm_version": "0.27.1",
    "base_image_id": MARLIN_BASE_IMAGE_ID,
}


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


_MANIFEST_BYTES = _json_bytes(MARLIN_PATCH_MANIFEST)
MARLIN_PATCH_MANIFEST_SHA256 = hashlib.sha256(_MANIFEST_BYTES).hexdigest()


def _checked_patch() -> str:
    raw = (Path(__file__).parent / "vendor" / "marlin_order.patch").read_bytes()
    if hashlib.sha256(raw).hexdigest() != MARLIN_PATCH_MANIFEST["upstream_patch_sha256"]:
        raise ValueError("vendored Marlin patch differs from the reviewed upstream bytes")
    patch = raw.decode()
    helper_hunk = re.split(r"^@@[^\n]*\n", patch, flags=re.MULTILINE)[1]
    added = "\n".join(line[1:] for line in helper_hunk.splitlines() if line.startswith("+")) + "\n"
    function = ast.parse(added).body[0]
    helper = "\n".join(added.splitlines()[function.lineno - 1:function.end_lineno]) + "\n"
    if hashlib.sha256(helper.encode()).hexdigest() != MARLIN_PATCH_MANIFEST["helper_sha256"]:
        raise ValueError("Marlin canonicalization helper differs from the reviewed source")
    return patch


def patched_marlin_source(original: bytes) -> bytes:
    if hashlib.sha256(original).hexdigest() != MARLIN_PATCH_MANIFEST["original_sha256"]:
        raise ValueError("original Marlin source SHA256 mismatch; duplicate or unreviewed patch refused")
    result = original.decode()
    hunks = re.split(r"^@@[^\n]*\n", _checked_patch(), flags=re.MULTILINE)[1:]
    for hunk in hunks:
        lines = hunk.splitlines()
        before = "".join(line[1:] + "\n" for line in lines if line.startswith((" ", "-")))
        after = "".join(line[1:] + "\n" for line in lines if line.startswith((" ", "+")))
        if result.count(before) != 1:
            raise ValueError("Marlin patch must match exactly one original source location per hunk")
        result = result.replace(before, after, 1)
    patched = result.encode()
    if hashlib.sha256(patched).hexdigest() != MARLIN_PATCH_MANIFEST["patched_sha256"]:
        raise ValueError("patched Marlin source SHA256 mismatch")
    compile(patched, MARLIN_SOURCE_PATH, "exec")
    return patched


def _atomic_write(path: Path, content: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        staging = Path(temporary.name)
        try:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            os.chmod(staging, mode)
            os.replace(staging, path)
        finally:
            staging.unlink(missing_ok=True)


def apply_marlin_patch(source_path: str | Path, manifest_path: str | Path) -> dict:
    source, manifest = Path(source_path), Path(manifest_path)
    patched = patched_marlin_source(source.read_bytes())
    _atomic_write(source, patched, source.stat().st_mode & 0o777)
    _atomic_write(manifest, _MANIFEST_BYTES, 0o644)
    return verify_marlin_patch(source, manifest)


def verify_marlin_patch(source_path: str | Path, manifest_path: str | Path) -> dict:
    _checked_patch()
    raw_manifest = Path(manifest_path).read_bytes()
    if raw_manifest != _MANIFEST_BYTES:
        raise ValueError("installed Marlin patch manifest differs from the reviewed manifest")
    source_hash = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    if source_hash != MARLIN_PATCH_MANIFEST["patched_sha256"]:
        raise ValueError("installed Marlin source SHA256 differs from the reviewed patched source")
    return {
        "manifest": json.loads(raw_manifest),
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "installed_source_sha256": source_hash,
    }


def validate_actor_patch_contract(contract: dict) -> None:
    image_id = contract.get("image_id")
    if (not isinstance(image_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id)
            or image_id == MARLIN_BASE_IMAGE_ID
            or contract.get("base_image_id") != MARLIN_BASE_IMAGE_ID
            or _json_bytes(contract.get("marlin_patch")) != _MANIFEST_BYTES
            or contract.get("marlin_patch_manifest_sha256") != MARLIN_PATCH_MANIFEST_SHA256
            or contract.get("marlin_installed_source_sha256") != MARLIN_PATCH_MANIFEST["patched_sha256"]):
        raise ValueError("actor Marlin research-image provenance differs from the reviewed patch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "verify"))
    parser.add_argument("--source", default=MARLIN_SOURCE_PATH)
    parser.add_argument("--manifest", default=MARLIN_MANIFEST_PATH)
    args = parser.parse_args()
    action = apply_marlin_patch if args.action == "apply" else verify_marlin_patch
    print(json.dumps(action(args.source, args.manifest), sort_keys=True))


if __name__ == "__main__":
    main()
