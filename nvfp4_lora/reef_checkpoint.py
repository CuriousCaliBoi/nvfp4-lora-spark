"""Immutable checkpoint validation without importing the training framework."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import tempfile

MANIFEST = "nvfp4-checkpoint.json"
PAYLOAD_FILES = frozenset({
    "adapter/adapter_model.safetensors", "adapter/adapter_config.json",
    "native_adapter.safetensors", "optimizer.pt", "rng.pt", "metrics.json",
})

# These fingerprints bind the audit to the reviewed Lightning checkpoint and
# training configuration; another architecture or configuration needs review.
_BOOTSTRAP_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
_BOOTSTRAP_REVISION = "bee7596271d1495f6992ae224aefde4410e816b8"
_BOOTSTRAP_MODEL_CONFIG = "f1d98b530846087dc08b574a219713a94f945bf6583dc7230a19ebf1e8c50933"
_BOOTSTRAP_FROZEN = "771893dc6b235fbb8e415089e0362ecf088aaf295bee62dcbfbe4fc4ff0cc9fb"
_BOOTSTRAP_TRAINING_CONFIG = "0b6b342997cacba1d6d40268cf119b020e78f1f097f4b3b4cba1ee7778aa7021"
_BOOTSTRAP_LAYERS = (5, 12, 19, 26, 33, 42)
_BOOTSTRAP_PROJECTIONS = {"q_proj": (2688, 4096), "k_proj": (2688, 256),
                          "v_proj": (2688, 256), "o_proj": (4096, 2688)}
_BOOTSTRAP_PEFT_CONFIG = {
    "base_model_name_or_path": _BOOTSTRAP_MODEL, "bias": "none", "fan_in_fan_out": False,
    "inference_mode": True, "init_lora_weights": True, "lora_alpha": 16,
    "lora_dropout": 0.0, "modules_to_save": None, "peft_type": "LORA", "r": 8,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "task_type": "CAUSAL_LM",
    "use_dora": False, "use_rslora": False,
}
_MAX_AUDIT_JSON_BYTES = 1024 * 1024


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


def _strict_json(data: bytes):
    def unique_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    if len(data) > _MAX_AUDIT_JSON_BYTES:
        raise ValueError("audit JSON exceeds size bound")
    return json.loads(data, object_pairs_hook=unique_pairs, parse_constant=reject_constant)


def _audit_safetensors(path: Path, expected_shapes: dict) -> tuple[dict, str]:
    payload_size = sum(rows * columns * 2 for rows, columns in expected_shapes.values())
    if not 8 < path.stat().st_size <= 8 + _MAX_AUDIT_JSON_BYTES + payload_size:
        raise ValueError("invalid safetensors file size")
    data = path.read_bytes()
    header_size = struct.unpack_from("<Q", data)[0]
    if not 0 < header_size <= min(_MAX_AUDIT_JSON_BYTES, len(data) - 8):
        raise ValueError("invalid safetensors header length")
    header_bytes = data[8:8 + header_size]
    if not header_bytes.startswith(b"{"):
        raise ValueError("invalid safetensors JSON header")
    header = _strict_json(header_bytes)
    if not isinstance(header, dict):
        raise ValueError("invalid safetensors header object")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                             for k, v in metadata.items()):
        raise ValueError("invalid safetensors metadata")
    if set(header) != set(expected_shapes):
        raise ValueError("safetensors tensor names differ from reviewed attention targets")
    payload = memoryview(data)[8 + header_size:]
    if len(payload) != payload_size:
        raise ValueError("safetensors payload size differs from reviewed layout")
    tensors, intervals = {}, []
    for name, shape in expected_shapes.items():
        entry = header[name]
        if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"invalid safetensors descriptor: {name}")
        if entry["dtype"] != "BF16":
            raise ValueError(f"safetensors dtype must be BF16: {name}")
        actual_shape = entry["shape"]
        if (not isinstance(actual_shape, list) or any(type(x) is not int for x in actual_shape)
                or actual_shape != list(shape)):
            raise ValueError(f"safetensors shape differs from reviewed layout: {name}")
        offsets = entry["data_offsets"]
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(x) is not int for x in offsets)):
            raise ValueError(f"invalid safetensors offsets: {name}")
        start, end = offsets
        if not 0 <= start < end <= len(payload) or end - start != shape[0] * shape[1] * 2:
            raise ValueError(f"safetensors offsets exceed bounds or tensor size: {name}")
        intervals.append((start, end))
        tensors[name] = payload[start:end]
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("safetensors payload has overlapping tensors or gaps")
        cursor = end
    if cursor != len(payload):
        raise ValueError("safetensors payload has unclaimed bytes")
    return tensors, hashlib.sha256(data).hexdigest()


def audit_bootstrap_checkpoint(path) -> dict:
    """Audit the reviewed zero-LoRA layout without Torch, NumPy, or pickle loads.

    Model/config/frozen fingerprints are compared with the pinned reviewed
    manifest values. This does not rehash the base model or deserialize Adam/RNG.
    """
    path = Path(path).resolve()
    manifest = validate_checkpoint(path, expected_base_model=_BOOTSTRAP_MODEL,
                                   expected_model_revision=_BOOTSTRAP_REVISION,
                                   expected_lora_rank=8, expected_lora_alpha=16,
                                   expected_config_sha256=_BOOTSTRAP_TRAINING_CONFIG)
    manifest_bytes = _safe_file(path, MANIFEST).read_bytes()
    if _strict_json(manifest_bytes) != manifest or type(manifest["schema_version"]) is not int:
        raise ValueError("invalid bootstrap manifest")
    if (manifest["optimizer_step"] != 0 or manifest["parent_checkpoint_id"] is not None
            or manifest["training_job_id"] is not None or manifest["batch_sha256"] is not None):
        raise ValueError("bootstrap audit requires an untrained checkpoint")
    for key, expected in (("model_config_sha256", _BOOTSTRAP_MODEL_CONFIG),
                          ("frozen_tensor_sha256", _BOOTSTRAP_FROZEN)):
        if manifest[key] != expected:
            raise ValueError(f"bootstrap {key} differs from reviewed fingerprint")
    config_bytes = _safe_file(path, "adapter/adapter_config.json").read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if config_sha256 != manifest["files"]["adapter/adapter_config.json"]:
        raise ValueError("adapter config changed during bootstrap audit")
    if canonical_json(_strict_json(config_bytes)) != canonical_json(_BOOTSTRAP_PEFT_CONFIG):
        raise ValueError("bootstrap PEFT configuration differs from reviewed configuration")
    shapes = {}
    targets = []
    for layer in _BOOTSTRAP_LAYERS:
        for projection, (inputs, outputs) in _BOOTSTRAP_PROJECTIONS.items():
            target = f"model.layers.{layer}.mixer.{projection}"
            targets.append(target)
            shapes[f"{target}.lora_A"] = (8, inputs)
            shapes[f"{target}.lora_B"] = (outputs, 8)
    export_names = {name: "base_model.model.backbone." + name.removeprefix("model.") + ".weight"
                    for name in shapes}
    native, native_sha256 = _audit_safetensors(_safe_file(path, "native_adapter.safetensors"), shapes)
    peft, peft_sha256 = _audit_safetensors(_safe_file(path, "adapter/adapter_model.safetensors"),
                                        {export_names[name]: shape for name, shape in shapes.items()})
    for name, checksum in (("native_adapter.safetensors", native_sha256),
                           ("adapter/adapter_model.safetensors", peft_sha256)):
        if checksum != manifest["files"][name]:
            raise ValueError(f"adapter changed during bootstrap audit: {name}")
    tensor_evidence = []
    for name, tensor in native.items():
        if name.endswith(".lora_B"):
            if any(tensor):
                raise ValueError(f"bootstrap B tensor is not byte-positive-zero: {name}")
        elif any(bits & 0x7f80 == 0x7f80 for (bits,) in struct.iter_unpack("<H", tensor)):
            raise ValueError(f"bootstrap A tensor contains nonfinite BF16 values: {name}")
        if tensor != peft[export_names[name]]:
            raise ValueError(f"native and PEFT tensor bytes differ: {name}")
        tensor_evidence.append({"native_name": name, "peft_name": export_names[name],
                                "shape": list(shapes[name]), "dtype": "BF16",
                                "sha256": hashlib.sha256(tensor).hexdigest()})
    return {"audit_schema_version": 1, "checkpoint_id": manifest["checkpoint_id"],
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "base_model": manifest["base_model"], "model_revision": manifest["model_revision"],
            "model_config_sha256": manifest["model_config_sha256"],
            "frozen_tensor_sha256": manifest["frozen_tensor_sha256"],
            "fingerprint_scope": "pinned_manifest_values", "config_sha256": manifest["config_sha256"],
            "adapter_config_sha256": config_sha256, "adapter_sha256": peft_sha256,
            "native_adapter_sha256": native_sha256, "lora_rank": 8, "lora_alpha": 16,
            "optimizer_step": 0, "parent_checkpoint_id": None, "attention_targets": targets,
            "target_count": len(targets), "tensor_count": len(shapes),
            "parameter_count": sum(rows * columns for rows, columns in shapes.values()),
            "positive_zero_b_count": len(targets), "finite_a_count": len(targets),
            "native_peft_exact_match": True, "tensor_mapping_sha256": content_hash(tensor_evidence)}


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
