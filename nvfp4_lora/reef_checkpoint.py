"""Immutable checkpoint validation without importing the training framework."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

MANIFEST = "nvfp4-checkpoint.json"
PAYLOAD_FILES = frozenset({
    "adapter/adapter_model.safetensors", "adapter/adapter_config.json",
    "native_adapter.safetensors", "optimizer.pt", "rng.pt", "metrics.json",
})


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def content_hash(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_hash(path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as destination:
            destination.write(canonical_json(value) + b"\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(name, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def sync_directory(path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def checkpoint_identity(value: dict) -> str:
    if value.get("training_job_id") is None and value.get("kind") != "train":
        descriptor = {"kind": "bootstrap", **{key: value[key] for key in (
            "base_model", "model_revision", "lora_rank", "lora_alpha", "config_sha256"
        )}}
    else:
        descriptor = {"kind": "train", **{key: value[key] for key in (
            "scenario", "parent_checkpoint_id", "batch_sha256", "config_sha256"
        )}}
    return content_hash(descriptor)


def _hash(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _safe_file(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")) or "\\" in name:
        raise ValueError("unsafe checkpoint path")
    candidate = root / name
    if not candidate.resolve().is_relative_to(root.resolve()) or not candidate.is_file():
        raise ValueError("checkpoint payload is missing or escapes its root")
    return candidate


def validate_checkpoint(path, *, expected_base_model=None, expected_model_revision=None,
                        expected_lora_rank=None, expected_lora_alpha=None,
                        expected_config_sha256=None) -> dict:
    path = Path(path).resolve()
    value = json.loads(_safe_file(path, MANIFEST).read_text())
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported checkpoint schema")
    for key in ("scenario", "base_model", "model_revision"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"missing checkpoint {key}")
    for key in ("checkpoint_id", "config_sha256", "adapter_sha256", "model_config_sha256", "frozen_tensor_sha256"):
        if not _hash(value.get(key)):
            raise ValueError(f"invalid checkpoint {key}")
    if type(value.get("optimizer_step")) is not int or value["optimizer_step"] < 0:
        raise ValueError("invalid checkpoint optimizer step")
    if type(value.get("lora_rank")) is not int or value["lora_rank"] <= 0:
        raise ValueError("invalid checkpoint rank")
    if type(value.get("lora_alpha")) not in (int, float) or value["lora_alpha"] <= 0:
        raise ValueError("invalid checkpoint alpha")
    if not isinstance(value.get("config"), dict) or content_hash(value["config"]) != value["config_sha256"]:
        raise ValueError("training config hash mismatch")
    for key in ("parent_checkpoint_id", "training_job_id", "batch_sha256"):
        if key not in value or (value[key] is not None and not _hash(value[key])):
            raise ValueError(f"invalid checkpoint {key}")
    if value["training_job_id"] is None:
        if value["parent_checkpoint_id"] is not None or value["batch_sha256"] is not None or value["optimizer_step"] != 0:
            raise ValueError("invalid bootstrap checkpoint")
    elif (value["training_job_id"] != value["checkpoint_id"] or value["parent_checkpoint_id"] is None
          or value["batch_sha256"] is None or value["optimizer_step"] < 1):
        raise ValueError("invalid training checkpoint ancestry")
    if checkpoint_identity(value) != value["checkpoint_id"]:
        raise ValueError("checkpoint identity mismatch")
    files = value.get("files")
    if not isinstance(files, dict) or set(files) != PAYLOAD_FILES:
        raise ValueError("checkpoint payload declaration mismatch")
    for name, checksum in files.items():
        if not _hash(checksum) or file_hash(_safe_file(path, name)) != checksum:
            raise ValueError(f"checkpoint hash mismatch: {name}")
    if value["adapter_sha256"] != files["adapter/adapter_model.safetensors"]:
        raise ValueError("adapter hash disagrees with payload")
    expected = {"base_model": expected_base_model, "model_revision": expected_model_revision,
                "lora_rank": expected_lora_rank, "lora_alpha": expected_lora_alpha,
                "config_sha256": expected_config_sha256}
    for key, wanted in expected.items():
        if wanted is not None and value[key] != wanted:
            raise ValueError(f"incompatible checkpoint {key}")
    return value


def publish_checkpoint(staging, destination, manifest: dict) -> Path:
    staging, destination = Path(staging), Path(destination)
    manifest = {**manifest, "files": {name: file_hash(staging / name) for name in PAYLOAD_FILES}}
    manifest["adapter_sha256"] = manifest["files"]["adapter/adapter_model.safetensors"]
    write_json(staging / MANIFEST, manifest)
    validate_checkpoint(staging)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if validate_checkpoint(destination) != manifest:
            raise ValueError("checkpoint identity already has different content")
        shutil.rmtree(staging)
    else:
        for name in PAYLOAD_FILES:
            with (staging / name).open("rb") as source:
                os.fsync(source.fileno())
        sync_directory(staging / "adapter")
        sync_directory(staging)
        os.rename(staging, destination)
        sync_directory(destination.parent)
    return destination


def materialize_checkpoint(artifact, checkpoint_root, *, base_checkpoint=None) -> Path:
    source = artifact.materialize().local_path
    if source is None:
        raise ValueError("a materialized checkpoint is required")
    source = Path(source).resolve()
    if not (source / MANIFEST).exists():
        contents = {entry.name for entry in source.iterdir()} - {".git", ".gitattributes", "reef-artifact.json"}
        if contents or base_checkpoint is None:
            raise ValueError("artifact has no NVFP4 checkpoint manifest")
        source = Path(base_checkpoint).resolve()
    manifest = validate_checkpoint(source)
    root = Path(checkpoint_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / manifest["checkpoint_id"]
    if destination.exists():
        if validate_checkpoint(destination) != manifest:
            raise ValueError("materialized checkpoint conflicts with existing identity")
        return destination
    temporary = Path(tempfile.mkdtemp(prefix=".materialize-", dir=root))
    try:
        for name in PAYLOAD_FILES:
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_safe_file(source, name), target)
        return publish_checkpoint(temporary, destination, manifest)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
