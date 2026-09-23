import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from nvfp4_lora.reef_checkpoint import (
    MANIFEST, PAYLOAD_FILES, audit_bootstrap_checkpoint, canonical_json, checkpoint_identity,
    content_hash, file_hash, publish_checkpoint, write_json,
)
from nvfp4_lora.reef_data import normalized_config


MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
REVISION = "bee7596271d1495f6992ae224aefde4410e816b8"
NATIVE = "native_adapter.safetensors"
PEFT = "adapter/adapter_model.safetensors"
CONFIG = "adapter/adapter_config.json"
LAYERS = (5, 12, 19, 26, 33, 42)
PROJECTIONS = {"q_proj": (2688, 4096), "k_proj": (2688, 256),
               "v_proj": (2688, 256), "o_proj": (4096, 2688)}
FIRST_A = "model.layers.5.mixer.q_proj.lora_A"
FIRST_B = "model.layers.5.mixer.q_proj.lora_B"


def peft_name(name):
    return name.replace("model.layers.", "base_model.model.backbone.layers.", 1) + ".weight"


def tensor_file(header, payload):
    encoded = canonical_json(header)
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + payload


def read_tensor_file(path):
    raw = path.read_bytes()
    length = struct.unpack_from("<Q", raw)[0]
    return json.loads(raw[8:8 + length]), bytearray(raw[8 + length:])


@pytest.fixture
def bootstrap(tmp_path):
    staging = tmp_path / "staging"
    (staging / "adapter").mkdir(parents=True)
    native_header, peft_header, payload = {}, {}, bytearray()
    for layer in LAYERS:
        for projection, (inputs, outputs) in PROJECTIONS.items():
            for letter, shape in (("A", [8, inputs]), ("B", [outputs, 8])):
                name = f"model.layers.{layer}.mixer.{projection}.lora_{letter}"
                start = len(payload)
                payload.extend((b"\x80\x3f" if letter == "A" else b"\0\0") * (shape[0] * shape[1]))
                descriptor = {"dtype": "BF16", "shape": shape, "data_offsets": [start, len(payload)]}
                native_header[name] = descriptor
                peft_header[peft_name(name)] = descriptor
    (staging / NATIVE).write_bytes(tensor_file(native_header, payload))
    (staging / PEFT).write_bytes(tensor_file(peft_header, payload))
    write_json(staging / CONFIG, {
        "base_model_name_or_path": MODEL, "bias": "none", "fan_in_fan_out": False,
        "inference_mode": True, "init_lora_weights": True, "lora_alpha": 16,
        "lora_dropout": 0.0, "modules_to_save": None, "peft_type": "LORA", "r": 8,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "task_type": "CAUSAL_LM",
        "use_dora": False, "use_rslora": False,
    })
    (staging / "optimizer.pt").write_bytes(b"hashed opaque optimizer payload; never unpickle")
    (staging / "rng.pt").write_bytes(b"hashed opaque RNG payload; never unpickle")
    write_json(staging / "metrics.json", {"optimizer_step_after": 0})
    config = normalized_config()
    manifest = dict(
        schema_version=1, scenario="bootstrap-audit-test", base_model=MODEL, model_revision=REVISION,
        model_config_sha256="f1d98b530846087dc08b574a219713a94f945bf6583dc7230a19ebf1e8c50933",
        frozen_tensor_sha256="771893dc6b235fbb8e415089e0362ecf088aaf295bee62dcbfbe4fc4ff0cc9fb",
        lora_rank=8, lora_alpha=16, optimizer_step=0, parent_checkpoint_id=None,
        training_job_id=None, batch_sha256=None, config=config, config_sha256=content_hash(config),
    )
    manifest["checkpoint_id"] = checkpoint_identity(manifest)
    return publish_checkpoint(staging, tmp_path / manifest["checkpoint_id"], manifest)


def refresh_hashes(path):
    manifest = json.loads((path / MANIFEST).read_bytes())
    manifest["files"] = {name: file_hash(path / name) for name in PAYLOAD_FILES}
    manifest["adapter_sha256"] = manifest["files"][PEFT]
    write_json(path / MANIFEST, manifest)


def replace_tensor(path, name, value, *, both=True):
    for filename, tensor_name in ((NATIVE, name), (PEFT, peft_name(name))) if both else ((PEFT, peft_name(name)),):
        header, payload = read_tensor_file(path / filename)
        start = header[tensor_name]["data_offsets"][0]
        payload[start:start + 2] = struct.pack("<H", value)
        (path / filename).write_bytes(tensor_file(header, payload))
    refresh_hashes(path)


def test_bootstrap_exact_layout_and_evidence(bootstrap):
    evidence = audit_bootstrap_checkpoint(bootstrap)
    assert evidence["checkpoint_id"] == bootstrap.name
    assert evidence["manifest_sha256"] == file_hash(bootstrap / MANIFEST)
    assert evidence["native_adapter_sha256"] == file_hash(bootstrap / NATIVE)
    assert evidence["adapter_sha256"] == file_hash(bootstrap / PEFT)
    assert evidence["adapter_config_sha256"] == file_hash(bootstrap / CONFIG)
    assert evidence["target_count"] == evidence["positive_zero_b_count"] == evidence["finite_a_count"] == 24
    assert evidence["tensor_count"] == 48
    assert evidence["parameter_count"] == 933888
    assert evidence["native_peft_exact_match"] is True
    assert evidence["fingerprint_scope"] == "pinned_manifest_values"
    assert evidence["attention_targets"] == [
        f"model.layers.{layer}.mixer.{projection}" for layer in LAYERS for projection in PROJECTIONS
    ]
    assert evidence["optimizer_step"] == 0 and evidence["parent_checkpoint_id"] is None
    assert audit_bootstrap_checkpoint(bootstrap) == evidence


@pytest.mark.parametrize("bits", [0x8000, 0x3f80, 0x7fc0])
def test_bootstrap_b_requires_positive_zero_bytes(bootstrap, bits):
    replace_tensor(bootstrap, FIRST_B, bits)
    with pytest.raises(ValueError, match="byte-positive-zero"):
        audit_bootstrap_checkpoint(bootstrap)


@pytest.mark.parametrize("bits", [0x7f80, 0xff80, 0x7fc0, 0x7f81])
def test_bootstrap_a_must_be_finite(bootstrap, bits):
    replace_tensor(bootstrap, FIRST_A, bits)
    with pytest.raises(ValueError, match="nonfinite BF16"):
        audit_bootstrap_checkpoint(bootstrap)


@pytest.mark.parametrize("bits", [0x0000, 0x8000, 0x7f7f, 0xff7f, 0x0001])
def test_finite_bf16_a_edge_values_are_accepted(bootstrap, bits):
    replace_tensor(bootstrap, FIRST_A, bits)
    assert audit_bootstrap_checkpoint(bootstrap)["finite_a_count"] == 24


def test_peft_export_must_match_native_bytes(bootstrap):
    replace_tensor(bootstrap, FIRST_A, 0x4000, both=False)
    with pytest.raises(ValueError, match="tensor bytes differ"):
        audit_bootstrap_checkpoint(bootstrap)


@pytest.mark.parametrize("file", [NATIVE, PEFT])
@pytest.mark.parametrize("mutation", [
    "missing", "extra", "wrong_layer", "shape", "bool_shape", "dtype", "extra_descriptor",
    "descriptor_type", "negative_offset", "reverse_offset", "bool_offset", "float_offset",
    "out_of_bounds", "short_tensor", "overlap", "trailing_payload", "metadata",
])
def test_rehashed_malformed_safetensors_rejected(bootstrap, file, mutation):
    header, payload = read_tensor_file(bootstrap / file)
    name = FIRST_A if file == NATIVE else peft_name(FIRST_A)
    entry = header[name]
    if mutation == "missing":
        del header[name]
    elif mutation == "extra":
        header["unexpected_tensor"] = entry
    elif mutation == "wrong_layer":
        header[name.replace("layers.5.", "layers.6.")] = header.pop(name)
    elif mutation == "shape":
        entry["shape"].reverse()
    elif mutation == "bool_shape":
        entry["shape"][0] = True
    elif mutation == "dtype":
        entry["dtype"] = "F16"
    elif mutation == "extra_descriptor":
        entry["extra"] = 1
    elif mutation == "descriptor_type":
        header[name] = []
    elif mutation == "negative_offset":
        entry["data_offsets"][0] = -1
    elif mutation == "reverse_offset":
        entry["data_offsets"].reverse()
    elif mutation == "bool_offset":
        entry["data_offsets"][0] = False
    elif mutation == "float_offset":
        entry["data_offsets"][0] = 0.0
    elif mutation == "out_of_bounds":
        entry["data_offsets"][1] = len(payload) + 2
    elif mutation == "short_tensor":
        entry["data_offsets"][1] -= 2
    elif mutation == "overlap":
        header[name.replace("q_proj", "k_proj")]["data_offsets"] = list(entry["data_offsets"])
    elif mutation == "trailing_payload":
        payload.extend(b"\0\0")
    else:
        header["__metadata__"] = {"format": False}
    (bootstrap / file).write_bytes(tensor_file(header, payload))
    refresh_hashes(bootstrap)
    with pytest.raises(ValueError, match="safetensors"):
        audit_bootstrap_checkpoint(bootstrap)


@pytest.mark.parametrize("mutation", ["zero_length", "oversized_length", "truncated", "json", "duplicate"])
def test_invalid_safetensors_header_encoding(bootstrap, mutation):
    raw = (bootstrap / NATIVE).read_bytes()
    if mutation == "zero_length":
        raw = b"\0" * 8 + raw[8:]
    elif mutation == "oversized_length":
        raw = struct.pack("<Q", 2 ** 64 - 1) + raw[8:]
    elif mutation == "truncated":
        raw = raw[:6]
    elif mutation == "json":
        raw = struct.pack("<Q", 1) + b"{" + raw[9:]
    else:
        header, payload = read_tensor_file(bootstrap / NATIVE)
        encoded = canonical_json(header)
        encoded = b'{"' + FIRST_A.encode() + b'":null,' + encoded[1:]
        raw = struct.pack("<Q", len(encoded)) + encoded + payload
    (bootstrap / NATIVE).write_bytes(raw)
    refresh_hashes(bootstrap)
    with pytest.raises(ValueError):
        audit_bootstrap_checkpoint(bootstrap)


@pytest.mark.parametrize("key,value", [
    ("r", 16), ("lora_alpha", 32), ("target_modules", ["q_proj", "v_proj"]),
    ("lora_dropout", 0.1), ("bias", "all"), ("use_dora", True),
    ("fan_in_fan_out", 0), ("unexpected", None),
])
def test_peft_config_must_match_reviewed_config(bootstrap, key, value):
    config = json.loads((bootstrap / CONFIG).read_bytes())
    config[key] = value
    write_json(bootstrap / CONFIG, config)
    refresh_hashes(bootstrap)
    with pytest.raises(ValueError, match="PEFT configuration"):
        audit_bootstrap_checkpoint(bootstrap)


@pytest.mark.parametrize("key,value", [
    ("model_revision", "different-revision"), ("base_model", "different-model"),
    ("model_config_sha256", "c" * 64), ("frozen_tensor_sha256", "f" * 64),
    ("lora_rank", 16), ("lora_alpha", 32), ("schema_version", True),
])
def test_bootstrap_model_and_manifest_are_pinned(bootstrap, key, value):
    manifest = json.loads((bootstrap / MANIFEST).read_bytes())
    manifest[key] = value
    manifest["checkpoint_id"] = checkpoint_identity(manifest)
    write_json(bootstrap / MANIFEST, manifest)
    with pytest.raises(ValueError):
        audit_bootstrap_checkpoint(bootstrap)


def test_training_config_must_match_reviewed_hash(bootstrap):
    manifest = json.loads((bootstrap / MANIFEST).read_bytes())
    manifest["config"]["seed"] += 1
    manifest["config_sha256"] = content_hash(manifest["config"])
    manifest["checkpoint_id"] = checkpoint_identity(manifest)
    write_json(bootstrap / MANIFEST, manifest)
    with pytest.raises(ValueError, match="config_sha256"):
        audit_bootstrap_checkpoint(bootstrap)


def test_trained_checkpoint_is_not_a_bootstrap(bootstrap):
    manifest = json.loads((bootstrap / MANIFEST).read_bytes())
    manifest.update(optimizer_step=1, parent_checkpoint_id="a" * 64,
                    training_job_id="b" * 64, batch_sha256="c" * 64)
    manifest["checkpoint_id"] = checkpoint_identity(manifest)
    manifest["training_job_id"] = manifest["checkpoint_id"]
    write_json(bootstrap / MANIFEST, manifest)
    with pytest.raises(ValueError, match="untrained checkpoint"):
        audit_bootstrap_checkpoint(bootstrap)


@pytest.mark.parametrize("filename,key", [(MANIFEST, "schema_version"), (CONFIG, "r")])
def test_duplicate_manifest_and_config_keys_rejected(bootstrap, filename, key):
    raw = (bootstrap / filename).read_bytes()
    (bootstrap / filename).write_bytes(b'{"' + key.encode() + b'":123,' + raw[1:])
    if filename != MANIFEST:
        refresh_hashes(bootstrap)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        audit_bootstrap_checkpoint(bootstrap)


def test_existing_payload_hash_checks_remain_required(bootstrap):
    (bootstrap / "optimizer.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        audit_bootstrap_checkpoint(bootstrap)


def test_audit_uses_only_stdlib_without_loading_torch_or_numpy(bootstrap):
    root = str(Path(__file__).resolve().parents[1])
    script = f"""
import sys, importlib.abc
sys.path.insert(0, {root!r})
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'numpy', 'safetensors'):
            raise AssertionError('Non-stdlib dependency imported by bootstrap audit')
sys.meta_path.insert(0, Block())
from nvfp4_lora.reef_checkpoint import audit_bootstrap_checkpoint
assert audit_bootstrap_checkpoint({str(bootstrap)!r})['parameter_count'] == 933888
assert not {{'torch', 'numpy', 'safetensors'}}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-I", "-c", script], check=True)
