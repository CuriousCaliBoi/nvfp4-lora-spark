import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from nvfp4_lora.reef_checkpoint import (
    MANIFEST, PAYLOAD_FILES, checkpoint_identity, content_hash, materialize_checkpoint,
    publish_checkpoint, validate_checkpoint, write_json,
)
from nvfp4_lora.reef_data import SAMPLING, advantages, normalized_config, validate_capture, validate_rows


def make_checkpoint(root, *, parent=None, batch="b" * 64, step=0):
    config = normalized_config()
    manifest = dict(schema_version=1, scenario="test", base_model="test-model", model_revision="revision",
                    model_config_sha256="c" * 64, frozen_tensor_sha256="f" * 64, lora_rank=8,
                    lora_alpha=16, optimizer_step=step, config=config, config_sha256=content_hash(config),
                    parent_checkpoint_id=parent, batch_sha256=batch if parent else None,
                    training_job_id="pending" if parent else None)
    manifest["checkpoint_id"] = checkpoint_identity(manifest)
    if parent:
        manifest["training_job_id"] = manifest["checkpoint_id"]
    staging = root / "staging"
    staging.mkdir(parents=True)
    for name in PAYLOAD_FILES:
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"step": step}) if name.endswith(".json") else f"test state {step}")
    return publish_checkpoint(staging, root / manifest["checkpoint_id"], manifest)


def capture():
    return dict(tokens=[1, 2, 3], prompt_token_ids=[1, 2], completion_token_ids=[3],
                prompt_length=2, response_length=1, loss_mask=[1], rollout_log_probs=[-0.8],
                output_index=0, finish_reason="stop", runtime_load_id="runtime:one",
                adapter_sha256="a" * 64, adapter_config_sha256="b" * 64,
                base_model="test-model", model_revision="revision", sampling={**SAMPLING, "seed": 42})


def rows():
    return [dict(source_agent_record_id=f"source-{i}", report_agent_record_id=f"report-{i}",
                 cycle_id="cycle-one", group_id=0, rollout_id=i, group_size=2, batch_group_count=1,
                 dataset_id="openai/gsm8k", dataset_revision="dataset", dataset_split="train",
                 dataset_row_id=2, gold_answer="#### 3", response_text=f"#### {3 if i else 2}",
                 prompt_ids=[1, 2], completion_ids=[3], behavior_logprobs=[-0.8], reward=float(i),
                 **{key: value for key, value in capture().items() if key in (
                     "finish_reason", "runtime_load_id", "adapter_sha256", "adapter_config_sha256",
                     "base_model", "model_revision", "sampling")}) for i in range(2)]


def artifact(path):
    return SimpleNamespace(materialize=lambda: SimpleNamespace(local_path=path))


def test_checkpoint_materialization_and_empty_base(tmp_path):
    base = make_checkpoint(tmp_path / "original")
    value = validate_checkpoint(base, expected_lora_rank=8)
    copied = materialize_checkpoint(artifact(base), tmp_path / "actor")
    assert validate_checkpoint(copied) == value
    assert materialize_checkpoint(artifact(base), tmp_path / "actor") == copied
    empty = tmp_path / "empty"
    empty.mkdir()
    assert materialize_checkpoint(artifact(empty), tmp_path / "actor", base_checkpoint=base) == copied
    with pytest.raises(ValueError, match="manifest"):
        materialize_checkpoint(artifact(empty), tmp_path / "actor")


@pytest.mark.parametrize("mutation", ["corrupt", "traversal", "symlink", "config", "identity", "missing"])
def test_checkpoint_rejects_invalid_state(tmp_path, mutation):
    base = make_checkpoint(tmp_path)
    value = validate_checkpoint(base)
    if mutation == "corrupt":
        (base / "optimizer.pt").write_text("changed")
    elif mutation == "traversal":
        value["files"]["../outside"] = "a" * 64
    elif mutation == "symlink":
        outside = tmp_path / "outside"
        outside.write_text("state")
        (base / "optimizer.pt").unlink()
        (base / "optimizer.pt").symlink_to(outside)
    elif mutation == "config":
        value["config"]["seed"] = 99
    elif mutation == "identity":
        value["checkpoint_id"] = "e" * 64
    else:
        (base / "rng.pt").unlink()
    write_json(base / MANIFEST, value)
    with pytest.raises(ValueError):
        validate_checkpoint(base)


def test_incompatible_checkpoint_and_partial_state(tmp_path):
    base = make_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="incompatible"):
        validate_checkpoint(base, expected_model_revision="other")
    (base / MANIFEST).unlink()
    with pytest.raises(ValueError):
        validate_checkpoint(base)


def test_exact_capture_and_complete_group():
    validate_capture(capture(), "runtime:one")
    samples = validate_rows(rows())
    assert advantages(samples)[0] == -advantages(samples)[1]
    with pytest.raises(ValueError, match="incomplete"):
        validate_rows(samples[:1])


@pytest.mark.parametrize("key,value", [("rollout_log_probs", [float("nan")]), ("loss_mask", [0]),
                                      ("tokens", [1, 3]), ("runtime_load_id", "wrong"),
                                      ("completion_token_ids", [True]), ("finish_reason", "abort"),
                                      ("output_index", False), ("response_length", True),
                                      ("loss_mask", [True])])
def test_invalid_native_capture(key, value):
    native = capture()
    native[key] = value
    with pytest.raises(ValueError):
        validate_capture(native, "runtime:one")


def test_prefix_keeps_group_and_scores_zero():
    samples = rows()
    samples[1].update(finish_reason="length", reward=0.0)
    assert len(validate_rows(samples)) == 2
    assert advantages(samples) == (0.0, 0.0)
    samples[1]["reward"] = 1.0
    with pytest.raises(ValueError, match="verifier"):
        validate_rows(samples)


@pytest.mark.parametrize("mutation", ["duplicate", "mixed", "filter", "prompt"])
def test_invalid_rows(mutation):
    samples = copy.deepcopy(rows())
    if mutation == "duplicate":
        samples[1]["source_agent_record_id"] = samples[0]["source_agent_record_id"]
    elif mutation == "mixed":
        samples[1]["runtime_load_id"] = "runtime:two"
    elif mutation == "filter":
        samples[1]["sampling"]["top_p"] = 0.9
    else:
        samples[1]["prompt_ids"] = [4, 5]
    with pytest.raises(ValueError):
        validate_rows(samples)


def test_imports_do_not_load_torch():
    root = str(Path(__file__).resolve().parents[1])
    script = f"""
import sys, importlib.abc
sys.path.insert(0, {root!r})
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch imported in CPU plugin')
sys.meta_path.insert(0, Block())
import nvfp4_lora.reef_checkpoint, nvfp4_lora.reef_data
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-I", "-c", script], check=True)
