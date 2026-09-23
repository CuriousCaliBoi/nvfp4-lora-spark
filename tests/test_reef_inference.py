"""CPU HTTP doubles verify contracts; they are not native model evidence."""

import asyncio
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from reef.artifact.artifact import Artifact, LiveWeightArtifactRef
from reef.runtime.interfaces import ModelCandidate, TrainingRuntime, UpstreamStatusError
from reef.runtime.scheduler import RuntimeScheduler
from reef.train.backend import PreparedStep
from reef.train.runtime_backend import RuntimeCandidateBackend
from reef.surface.weights import WeightLoader

from nvfp4_lora.reef_checkpoint import (
    PAYLOAD_FILES, checkpoint_identity, content_hash, publish_checkpoint, write_json,
)
from nvfp4_lora.reef_data import normalized_config, validate_capture
from nvfp4_lora.reef_inference import VllmInferenceRuntime, chat_request, native_capture
from nvfp4_lora.reef_marlin_patch import (
    MARLIN_BASE_IMAGE_ID, MARLIN_PATCH_MANIFEST, MARLIN_PATCH_MANIFEST_SHA256,
)


def checkpoint(root, *, parent=None, step=0, batch="b" * 64):
    config = normalized_config()
    manifest = dict(schema_version=1, scenario="test", base_model="test-model", model_revision="revision",
                    model_config_sha256="c" * 64, frozen_tensor_sha256="f" * 64, lora_rank=8,
                    lora_alpha=16, optimizer_step=step, config=config, config_sha256=content_hash(config),
                    parent_checkpoint_id=parent, batch_sha256=batch if parent else None,
                    training_job_id="pending" if parent else None)
    manifest["checkpoint_id"] = checkpoint_identity(manifest)
    if parent:
        manifest["training_job_id"] = manifest["checkpoint_id"]
    staging = root / f"staging-{step}-{batch[:4]}"
    staging.mkdir(parents=True)
    for name in PAYLOAD_FILES:
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"step": step}))
    return publish_checkpoint(staging, root / manifest["checkpoint_id"], manifest)


def actor_contract(directory):
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "actor-contract.json", {
        "schema_version": 1, "actor_instance_id": "actor-owned-one", "container_id": "container-one",
        "image_id": "sha256:" + "a" * 64, "base_image_id": MARLIN_BASE_IMAGE_ID,
        "marlin_patch": MARLIN_PATCH_MANIFEST,
        "marlin_patch_manifest_sha256": MARLIN_PATCH_MANIFEST_SHA256,
        "marlin_installed_source_sha256": MARLIN_PATCH_MANIFEST["patched_sha256"],
        "base_model": "test-model", "model_revision": "revision", "vllm_version": "0.27.1",
        "logprobs_mode": "processed_logprobs", "generation_config": "vllm",
        "speculative_decoding": False, "exclusive_adapter_control": True, "async_scheduling": False,
        "max_num_seqs": 1, "kv_cache_dtype": "bfloat16", "attention_backend": "TRITON_ATTN",
        "mamba_cache_mode": "none", "cublas_workspace_config": ":4096:8",
        "command": ["vllm", "serve", "/hf", "--enable-lora", "--logprobs-mode", "processed_logprobs",
                    "--generation-config", "vllm", "--max-cpu-loras", "16", "--no-async-scheduling",
                    "--max-num-seqs", "1", "--kv-cache-dtype", "bfloat16",
                    "--attention-backend", "TRITON_ATTN", "--mamba-cache-mode", "none"],
    })


def response(model="adapter", *, finish="stop", text="#### 3"):
    return {
        "id": "chatcmpl-test", "model": model, "prompt_token_ids": [1, 2],
        "choices": [{"index": 0, "token_ids": [3, 4], "message": {"role": "assistant", "content": text},
                     "finish_reason": finish, "logprobs": {"content": [
                         {"token": "token_id:3", "logprob": -1.0}, {"token": "token_id:4", "logprob": -2.0}]}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
    }


class FakeActor:
    def __init__(self):
        self.registry = {"test-model": {"id": "test-model", "root": "/hf", "parent": None}}
        self.calls = []
        self.loads = []
        self.fail_load = False
        self.ignore_adapters = False
        self.reload_error = 0.0
        self.adapter_path_offset = 0.0
        self.probe_mutation = None
        self.chat_hook = None
        self.chat_mutation = None

    def request(self, method, path, payload=None):
        self.calls.append((method, path, copy.deepcopy(payload)))
        if path == "/version":
            return {"version": "0.27.1"}
        if path == "/v1/models":
            return {"data": list(self.registry.values())}
        if path == "/v1/load_lora_adapter":
            if self.fail_load:
                raise UpstreamStatusError("test load failure", status=500)
            name, location = payload["lora_name"], payload["lora_path"]
            assert name not in self.registry
            self.loads.append(name)
            self.registry[name] = {"id": name, "root": location, "parent": "test-model"}
            return "Success: adapter added"
        if path == "/v1/completions":
            model, tokens = payload["model"], payload["prompt"]
            step = 0
            if model != "test-model" and not self.ignore_adapters:
                step = json.loads((Path(self.registry[model]["root"]) / "adapter_model.safetensors").read_text())["step"]
            offset = self.reload_error if model.endswith("-reload") else 0.0
            if model != "test-model":
                offset += self.adapter_path_offset
            result = {"model": model, "choices": [{"index": 0, "prompt_token_ids": tokens,
                    "prompt_logprobs": [None] + [{str(token): {"logprob": -token / 10 + step / 100 + offset}}
                                                 for token in tokens[1:]]}]}
            if self.probe_mutation:
                self.probe_mutation(result)
            return result
        if path == "/v1/chat/completions":
            if self.chat_hook:
                hook, self.chat_hook = self.chat_hook, None
                hook()
            result = response(payload["model"])
            if self.chat_mutation:
                self.chat_mutation(result, payload)
            return result
        raise AssertionError((method, path, payload))


@pytest.fixture
def rig(tmp_path, monkeypatch):
    root, state = tmp_path / "checkpoints", tmp_path / "serving"
    base = checkpoint(root)
    actor_contract(state)
    actor = FakeActor()
    monkeypatch.setattr(VllmInferenceRuntime, "_http", lambda self, *args, **kwargs: actor.request(*args, **kwargs))
    options = dict(base_url="http://actor.invalid", actor_instance_id="actor-owned-one", base_model="test-model",
                   model_revision="revision", checkpoint_root=root, base_checkpoint=base, state_dir=state,
                   probe_token_ids=[1, 2, 3])
    runtime = VllmInferenceRuntime(**options)
    artifact = Artifact.local(base)
    runtime.activate_checkpoint(artifact)
    candidate_path = checkpoint(root, parent=base.name, step=1)
    candidate = ModelCandidate(candidate_id=candidate_path.name, training_job_id=candidate_path.name,
                               checkpoint_path=str(candidate_path), current_runtime_load_id=runtime.current_runtime_load_id())
    return SimpleNamespace(runtime=runtime, actor=actor, root=root, state=state, base=base,
                           artifact=artifact, candidate=candidate, candidate_path=candidate_path, options=options)


def test_exact_native_capture_preserves_provider_response(rig):
    result = asyncio.run(rig.runtime.inference_handler.inference(rig.artifact, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "One plus two?"}], "temperature": 1.2, "seed": 7,
    }))
    native = result["training"]
    validate_capture(native, rig.runtime.current_runtime_load_id())
    assert native["tokens"] == [1, 2, 3, 4]
    assert native["loss_mask"] == [1, 1]
    assert native["sampling"]["seed"] == 7
    assert result["choices"][0]["message"]["content"] == "#### 3"
    assert result["model"].startswith("reef-nvfp4-")
    request = next(body for _, path, body in reversed(rig.actor.calls) if path.endswith("chat/completions"))
    assert request["return_token_ids"] and request["return_tokens_as_token_ids"]
    assert request["logprobs"] and request["top_logprobs"] == 0
    assert rig.runtime.serving_adapter_name() is None


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("prompt_token_ids"),
    lambda r: r["choices"][0].pop("token_ids"),
    lambda r: r["choices"][0]["logprobs"]["content"].pop(),
    lambda r: r["choices"][0]["logprobs"]["content"][0].update(token="token_id:99"),
    lambda r: r["choices"][0]["logprobs"]["content"][0].update(logprob=float("nan")),
    lambda r: r["choices"][0]["token_ids"].__setitem__(0, True),
    lambda r: r["usage"].update(completion_tokens=99),
    lambda r: r["choices"][0].update(finish_reason="tool_calls"),
    lambda r: r["choices"][0]["message"].update(reasoning="hidden reasoning"),
    lambda r: r.update(model="another-adapter"),
])
def test_malformed_native_receipts_fail_closed(mutate):
    value = response()
    mutate(value)
    with pytest.raises(ValueError):
        native_capture(value, expected_model="adapter")


def test_length_limited_prefix_keeps_every_actual_token():
    value = native_capture(response(finish="length"), expected_model="adapter")
    assert value["completion_token_ids"] == [3, 4]
    assert value["finish_reason"] == "length"


@pytest.mark.parametrize("field,value", [
    ("stream", True), ("n", 2), ("temperature", 1), ("top_p", .9), ("top_k", 40), ("min_p", .1),
    ("presence_penalty", .1), ("frequency_penalty", .1), ("repetition_penalty", 1.1),
    ("logit_bias", {"1": 3}), ("tools", []), ("response_format", {"type": "json_object"}),
    ("allowed_token_ids", [1, 2]), ("bad_words", ["a"]), ("logits_processors", []),
    ("lora_path", "client-adapter"), ("model", "client-adapter"), ("seed", True),
    ("chat_template_kwargs", {"enable_thinking": True}), ("max_tokens", 257),
])
def test_unsupported_learning_requests(field, value):
    with pytest.raises(UpstreamStatusError):
        chat_request({"messages": [{"role": "user", "content": "q"}], field: value}, base_model="test-model")


def test_private_verification_and_evaluation_do_not_publish(rig):
    original = rig.runtime.current_runtime_load_id()
    proof = rig.runtime.verification(rig.candidate_path)
    assert proof["adapter_effect_max_delta"] > proof["reference_repeat_max_delta"]
    assert proof["reload_max_delta"] == 0
    assert len(rig.actor.loads) == 4
    results = rig.runtime.evaluate_adapter(rig.candidate_path, [{
        "messages": [{"role": "user", "content": "q"}], "temperature": 0.0,
    }])
    assert "training" not in results[0]
    assert rig.runtime.current_runtime_load_id() == original
    assert rig.runtime.serving_runtime_load_id() == original
    assert rig.runtime.pending_training_job_id is None


def test_failed_bootstrap_preserves_alias_repeats_and_interleaved_native_evidence(rig):
    rig.actor.adapter_path_offset = .125
    with pytest.raises(ValueError, match="bootstrap zero-delta adapter does not reproduce the base"):
        VllmInferenceRuntime(**rig.options)
    attempts = [json.loads(path.read_text()) for path in
                (rig.state / "native-probes" / rig.base.name).glob("*/evidence.json")]
    failed, = [value for value in attempts if value["status"] == "failed"]
    assert failed["bootstrap"] and not failed["native_verified"]
    assert failed["stage"] == "validation"
    assert failed["adapter_effect_max_delta"] == pytest.approx(.125)
    for field in ("reference_repeat_max_delta", "adapter_repeat_max_delta", "reload_repeat_max_delta",
                  "reload_max_delta", "reference_after_load_max_delta",
                  "reference_after_load_repeat_max_delta", "adapter_after_reference_max_delta"):
        assert failed[field] == 0
    assert set(failed["completed_probes"]) == {
        "reference_logprobs", "reference_repeat_logprobs", "adapter_logprobs", "adapter_repeat_logprobs",
        "reload_logprobs", "reload_repeat_logprobs", "reference_after_load_logprobs",
        "reference_after_load_repeat_logprobs", "adapter_after_reference_logprobs",
    }
    for key, location in failed["raw_probe_paths"].items():
        raw = json.loads(Path(location).read_text())
        assert raw["request"]["prompt"] == [1, 2, 3]
        assert raw["request"]["temperature"] == 1.0
        assert raw["response"]["model"] == raw["request"]["model"]
        assert raw["response"]["choices"][0]["prompt_token_ids"] == [1, 2, 3]
        assert failed[key] == [row[str(token)]["logprob"] for token, row in
                               zip([2, 3], raw["response"]["choices"][0]["prompt_logprobs"][1:])]
    assert failed["adapter_name"] in rig.actor.registry
    assert failed["reload_name"] in rig.actor.registry


def test_native_load_failure_preserves_completed_reference_probes(rig):
    rig.actor.fail_load = True
    with pytest.raises(UpstreamStatusError, match="test load failure"):
        rig.runtime.verification(rig.candidate_path)
    evidence_path, = (rig.state / "native-probes" / rig.candidate_path.name).glob("*/evidence.json")
    evidence = json.loads(evidence_path.read_text())
    assert evidence["status"] == "failed" and not evidence["native_verified"]
    assert evidence["stage"] == "load_primary"
    assert evidence["completed_probes"] == ["reference_logprobs", "reference_repeat_logprobs"]
    assert all(Path(path).is_file() for path in evidence["raw_probe_paths"].values())


def test_native_response_is_preserved_before_alignment_validation(rig):
    rig.actor.probe_mutation = lambda result: result["choices"][0].update(prompt_token_ids=[99])
    with pytest.raises(ValueError, match="changed native prompt IDs"):
        rig.runtime.verification(rig.candidate_path)
    evidence_path, = (rig.state / "native-probes" / rig.candidate_path.name).glob("*/evidence.json")
    evidence = json.loads(evidence_path.read_text())
    assert evidence["status"] == "failed"
    assert evidence["completed_probes"] == []
    raw = json.loads(Path(evidence["raw_probe_paths"]["reference_logprobs"]).read_text())
    assert raw["request"]["prompt"] == [1, 2, 3]
    assert raw["response"]["choices"][0]["prompt_token_ids"] == [99]


def offset_probe(rig, offsets):
    sequence = iter(offsets)

    def mutate(result):
        delta = next(sequence)
        for row in result["choices"][0]["prompt_logprobs"][1:]:
            for entry in row.values():
                entry["logprob"] += delta

    rig.actor.probe_mutation = mutate


@pytest.mark.parametrize("probe_index,metric", [
    (1, "reference_repeat_max_delta"), (3, "adapter_repeat_max_delta"),
    (5, "reload_repeat_max_delta"), (6, "reference_after_load_max_delta"),
    (7, "reference_after_load_repeat_max_delta"), (8, "adapter_after_reference_max_delta"),
])
def test_matching_reload_pair_cannot_hide_native_instability(rig, probe_index, metric):
    offsets = [0.0] * 9
    offsets[probe_index] = .001
    offset_probe(rig, offsets)
    with pytest.raises(ValueError, match=metric):
        rig.runtime.verification(rig.candidate_path)
    path, = (rig.state / "native-probes" / rig.candidate_path.name).glob("*/evidence.json")
    evidence = json.loads(path.read_text())
    assert evidence["status"] == "failed" and not evidence["native_verified"]
    assert evidence["reload_max_delta"] == 0
    assert evidence["adapter_effect_max_delta"] > evidence["reference_repeat_max_delta"]
    assert evidence["observed_null_max_delta"] == pytest.approx(.001)
    assert len(evidence["completed_probes"]) == 9
    assert all(Path(value).is_file() for value in evidence["raw_probe_paths"].values())
    assert not rig.runtime.inference_admission_status["open"]


def test_matching_zero_adapter_aliases_cannot_hide_unstable_bare_base(rig):
    offset_probe(rig, [0, .001, 0, 0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="reference_repeat_max_delta"):
        VllmInferenceRuntime(**rig.options)
    attempts = [json.loads(path.read_text()) for path in
                (rig.state / "native-probes" / rig.base.name).glob("*/evidence.json")]
    evidence, = [value for value in attempts if value["status"] == "failed"]
    assert evidence["reference_name"] == "test-model"
    assert evidence["adapter_effect_max_delta"] == evidence["reload_max_delta"] == 0
    assert len(evidence["completed_probes"]) == 9


def test_candidate_effect_must_exceed_alias_noise_even_below_stability_tolerance(rig):
    offset_probe(rig, [0, 0, -.01 + .000004, -.01 + .000004,
                       -.01 + .000009, -.01 + .000009, 0, 0, -.01 + .000004])
    with pytest.raises(ValueError, match="above observed null/repeat/alias noise"):
        rig.runtime.verification(rig.candidate_path)
    path, = (rig.state / "native-probes" / rig.candidate_path.name).glob("*/evidence.json")
    evidence = json.loads(path.read_text())
    assert evidence["reference_repeat_max_delta"] == 0
    assert evidence["adapter_effect_max_delta"] == pytest.approx(.000004)
    assert evidence["observed_null_max_delta"] == pytest.approx(.000005)
    assert evidence["observed_null_max_delta"] < 1e-5


@pytest.mark.parametrize("offsets,metric", [
    ([0, .000009, 0, 0, 0, 0, -.000009, -.000009, 0], "reference_observed_max_delta"),
    ([0, 0, 0, .000009, -.000009, -.000009, 0, 0, 0], "adapter_observed_max_delta"),
])
def test_stability_bounds_every_observed_pair_with_identical_weights(rig, offsets, metric):
    offset_probe(rig, offsets)
    with pytest.raises(ValueError, match=metric):
        rig.runtime.verification(rig.candidate_path)
    path, = (rig.state / "native-probes" / rig.candidate_path.name).glob("*/evidence.json")
    evidence = json.loads(path.read_text())
    assert evidence[metric] == pytest.approx(.000018)
    assert evidence["observed_null_max_delta"] == pytest.approx(.000018)


def test_candidate_requires_matching_durable_acknowledgement(rig):
    old = rig.runtime.current_runtime_load_id()
    selected = rig.runtime.activate_candidate(rig.candidate)
    assert selected.runtime_load_id != old
    assert rig.runtime.current_runtime_load_id() == old
    assert not rig.runtime.inference_admission_status["open"]
    rig.runtime.resume_admission()
    assert not rig.runtime.inference_admission_status["open"]
    with pytest.raises(Exception, match="another checkpoint"):
        rig.runtime.acknowledge_publication("wrong")
    artifact = Artifact.local(rig.candidate_path)
    assert rig.runtime.activate_checkpoint(artifact) == selected.runtime_load_id
    assert rig.runtime.current_runtime_load_id() == old
    rig.runtime.acknowledge_publication(rig.candidate.training_job_id)
    rig.runtime.mark_published()
    rig.runtime.resume_admission()
    assert rig.runtime.inference_admission_status["open"]
    assert rig.runtime.current_runtime_load_id() == selected.runtime_load_id


def test_request_pins_immutable_selector_across_head_change(rig):
    old_binding, old_version = rig.runtime.snapshot(rig.artifact)
    rig.actor.chat_hook = lambda: rig.runtime.activate_candidate(rig.candidate)
    result = asyncio.run(rig.runtime.inference_handler.inference(rig.artifact, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "q"}],
    }))
    assert result["model"] == old_binding.name
    assert result["training"]["runtime_load_id"] == old_version
    assert rig.runtime.serving_runtime_load_id() != old_version


def test_publication_drains_real_reef_admission(rig):
    async def run():
        lease = await rig.runtime.acquire_inference()
        activation = asyncio.create_task(asyncio.to_thread(rig.runtime.activate_candidate, rig.candidate))
        await asyncio.sleep(.02)
        assert not activation.done()
        lease.release()
        await activation
    asyncio.run(run())


class RejectOnlyTrainingRuntime(TrainingRuntime):
    def prepare_training_step(self, *args, **kwargs):
        raise AssertionError("test must not train")

    def train_candidate(self, payload):
        raise AssertionError("test must not train")

    def reject_candidate(self, candidate, decision):
        self.rejected = candidate.candidate_id


def test_native_load_failure_remains_fenced_through_actual_reef_abort(rig):
    training = RejectOnlyTrainingRuntime()
    backend = RuntimeCandidateBackend(training, "test", inference_runtime=rig.runtime)
    rig.actor.fail_load = True
    with pytest.raises(UpstreamStatusError):
        rig.runtime.activate_candidate(rig.candidate)
    backend.abort_step(PreparedStep.with_candidate(rig.candidate, state={}, metrics={}))
    assert training.rejected == rig.candidate.candidate_id
    assert not rig.runtime.inference_admission_status["open"]
    state = json.loads((rig.state / "serving-state.json").read_text())
    assert state["fenced"]
    rig.actor.fail_load = False
    rig.runtime.activate_checkpoint(rig.artifact)
    rig.runtime.mark_published()
    rig.runtime.resume_admission()
    assert rig.runtime.inference_admission_status["open"]


def test_same_runtime_initial_head_reload_reopens_after_evaluation_abort(rig):
    training = RejectOnlyTrainingRuntime()
    backend = RuntimeCandidateBackend(training, "test", inference_runtime=rig.runtime)
    backend.abort_step(PreparedStep.with_candidate(rig.candidate, state={}, metrics={}))
    reconciled = []

    def reconcile(artifact):
        assert not rig.runtime.inference_admission_status["open"]
        reconciled.append(artifact.ref.release_id)

    rig.runtime._on_checkpoint_activation = reconcile
    original = rig.runtime.current_runtime_load_id()
    loader = WeightLoader()
    assert loader.recover(rig.artifact.ref, rig.artifact.ref, rig.runtime) == rig.artifact.ref
    loader.activate(rig.artifact, rig.runtime)
    rebuilt = RuntimeScheduler(training, rig.runtime)
    rebuilt.recover_pending_step(0, committed_training_job_id=None)
    assert reconciled == [rig.artifact.ref.release_id]
    assert rig.runtime.current_runtime_load_id() == original
    assert rig.runtime.inference_admission_status["open"]
    result = asyncio.run(rig.runtime.inference_handler.inference(rig.artifact, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "A new request after recovery"}],
    }))
    assert result["training"]["runtime_load_id"] == original


@pytest.mark.parametrize("failure", ["native", "callback"])
def test_same_runtime_head_reload_failure_remains_closed(rig, failure):
    if failure == "native":
        binding, _ = rig.runtime.snapshot(rig.artifact)
        rig.actor.registry[binding.name]["root"] = "/wrong"
    else:
        def fail(artifact):
            raise ValueError("training reconciliation failed")
        rig.runtime._on_checkpoint_activation = fail
    with pytest.raises(ValueError):
        WeightLoader().activate(rig.artifact, rig.runtime)
    scheduler = RuntimeScheduler(RejectOnlyTrainingRuntime(), rig.runtime)
    scheduler.recover_pending_step(0, committed_training_job_id=None)
    rig.runtime.resume_admission()
    assert not rig.runtime.inference_admission_status["open"]
    assert json.loads((rig.state / "serving-state.json").read_text())["fenced"]


def test_explicit_rollback_of_same_checkpoint_waits_for_commit(rig):
    calls = []
    rig.runtime._on_checkpoint_activation = calls.append
    loader = WeightLoader()
    loader.load(rig.artifact, rig.runtime)
    republished = Artifact.local(rig.base)
    loader.activate(republished, rig.runtime)
    assert calls == [republished]
    assert not rig.runtime.inference_admission_status["open"]
    rig.runtime.mark_published()
    rig.runtime.resume_admission()
    assert rig.runtime.inference_admission_status["open"]


@pytest.mark.parametrize("failure", ["ignored", "reload", "capacity", "registry"])
def test_native_identity_failures_do_not_publish(rig, failure):
    before = rig.runtime.current_runtime_load_id()
    if failure == "ignored":
        rig.actor.ignore_adapters = True
    elif failure == "reload":
        rig.actor.reload_error = .1
    elif failure == "capacity":
        rig.runtime._capacity = 2
    else:
        binding, _ = rig.runtime.snapshot(rig.artifact)
        rig.actor.registry[binding.name]["root"] = "/wrong"
    with pytest.raises((ValueError, RuntimeError)):
        rig.runtime.verification(rig.candidate_path)
    assert rig.runtime.current_runtime_load_id() == before
    assert not rig.runtime.inference_admission_status["open"]


def test_restart_reverifies_existing_names_and_rollback_mints_new_version(rig):
    rig.runtime.activate_candidate(rig.candidate)
    candidate_artifact = Artifact.local(rig.candidate_path)
    rig.runtime.activate_checkpoint(candidate_artifact)
    rig.runtime.acknowledge_publication(rig.candidate.training_job_id)
    rig.runtime.mark_published()
    previous = rig.runtime.current_runtime_load_id()
    count = len(rig.actor.loads)
    restarted = VllmInferenceRuntime(**rig.options)
    assert not restarted.inference_admission_status["open"]
    restored = restarted.activate_checkpoint(candidate_artifact)
    assert restored != previous
    assert restarted.current_runtime_load_id() == restored
    assert len(rig.actor.loads) == count
    restarted.restore_checkpoint(rig.artifact)
    rollback = Artifact.local(rig.base)
    version = restarted.activate_checkpoint(rollback)
    assert version != restored
    assert not restarted.inference_admission_status["open"]
    restarted.mark_published()
    restarted.resume_admission()
    assert restarted.inference_admission_status["open"]
    state = json.loads((rig.state / "serving-state.json").read_text())
    assert state["active_checkpoint_id"] == rig.base.name


def test_failed_rollback_restart_reconciles_both_sides_before_admission(rig):
    rig.runtime.activate_candidate(rig.candidate)
    committed = Artifact.local(rig.candidate_path)
    rig.runtime.activate_checkpoint(committed)
    rig.runtime.acknowledge_publication(rig.candidate.training_job_id)
    rig.runtime.mark_published()
    learner = SimpleNamespace(incumbent=rig.candidate_path)
    old_version = rig.runtime.current_runtime_load_id()
    learner.incumbent = rig.base
    rig.runtime._verified.pop(rig.base.name)
    rig.actor.reload_error = .1
    with pytest.raises(ValueError, match="reload"):
        WeightLoader().load(rig.artifact, rig.runtime)
    assert learner.incumbent == rig.base
    assert not rig.runtime.inference_admission_status["open"]
    assert json.loads((rig.state / "serving-state.json").read_text())["fenced"]

    rig.actor.reload_error = 0
    callback_calls = []

    def reconcile(artifact):
        assert not restarted.inference_admission_status["open"]
        learner.incumbent = artifact.materialize().local_path
        callback_calls.append(learner.incumbent)

    restarted = VllmInferenceRuntime(**rig.options, on_checkpoint_activation=reconcile)
    old_live = LiveWeightArtifactRef("old-content", "old-live-release", None, old_version)
    loader = WeightLoader()
    recovered = loader.recover(old_live, committed.ref, restarted)
    assert recovered == committed.ref
    loader.activate(committed, restarted)
    assert learner.incumbent == rig.candidate_path
    assert callback_calls == [rig.candidate_path]
    assert restarted.inference_admission_status["open"]
    assert restarted.current_runtime_load_id() != old_version


def test_reconciliation_callback_failure_stays_fenced(rig):
    def fail(artifact):
        raise RuntimeError("learner restore failed")
    restarted = VllmInferenceRuntime(**rig.options, on_checkpoint_activation=fail)
    with pytest.raises(RuntimeError, match="learner restore failed"):
        WeightLoader().activate(rig.artifact, restarted)
    restarted.resume_admission()
    assert not restarted.inference_admission_status["open"]
    assert json.loads((rig.state / "serving-state.json").read_text())["fenced"]


def test_candidate_and_private_evaluation_never_restore_training_incumbent(rig):
    calls = []
    rig.runtime._on_checkpoint_activation = calls.append
    rig.runtime.evaluate_adapter(rig.candidate_path, [{"messages": [{"role": "user", "content": "q"}],
                                                    "temperature": 0.0}])
    rig.runtime.activate_candidate(rig.candidate)
    rig.runtime.activate_checkpoint(Artifact.local(rig.candidate_path))
    assert calls == []


def test_changed_actor_attestation_fences_requests(rig):
    path = rig.state / "actor-contract.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(Exception, match="actor identity changed"):
        rig.runtime.snapshot(rig.artifact)
    assert not rig.runtime.inference_admission_status["open"]


def test_incompatible_sampling_attestation_fails_before_native_load(rig):
    path = rig.state / "actor-contract.json"
    value = json.loads(path.read_text())
    value["logprobs_mode"] = "raw_logprobs"
    write_json(path, value)
    before = len(rig.actor.calls)
    with pytest.raises(ValueError, match="attestation"):
        VllmInferenceRuntime(**rig.options)
    assert len(rig.actor.calls) == before


@pytest.mark.parametrize("mutation", [
    lambda value: value.pop("image_id"),
    lambda value: value.update(image_id="research:latest"),
    lambda value: value.update(image_id=MARLIN_BASE_IMAGE_ID),
    lambda value: value.pop("base_image_id"),
    lambda value: value.update(base_image_id="sha256:" + "b" * 64),
    lambda value: value.pop("marlin_patch"),
    lambda value: value["marlin_patch"].update(upstream_commit="b" * 40),
    lambda value: value["marlin_patch"].update(schema_version=True),
    lambda value: value["marlin_patch"].update(patched_sha256="b" * 64),
    lambda value: value["marlin_patch"].update(helper_sha256="b" * 64),
    lambda value: value["marlin_patch"].update(source_path="/unreviewed/module.py"),
    lambda value: value.pop("marlin_patch_manifest_sha256"),
    lambda value: value.update(marlin_patch_manifest_sha256="b" * 64),
    lambda value: value.pop("marlin_installed_source_sha256"),
    lambda value: value.update(marlin_installed_source_sha256="b" * 64),
])
def test_research_image_provenance_fails_before_any_native_call(rig, mutation):
    path = rig.state / "actor-contract.json"
    contract = json.loads(path.read_text())
    mutation(contract)
    write_json(path, contract)
    before = len(rig.actor.calls)
    with pytest.raises(ValueError, match="research-image provenance"):
        VllmInferenceRuntime(**rig.options)
    assert len(rig.actor.calls) == before


@pytest.mark.parametrize("mutation", [
    lambda value: value.pop("async_scheduling"),
    lambda value: value.update(async_scheduling=True),
    lambda value: value.update(async_scheduling=0),
    lambda value: value["command"].remove("--no-async-scheduling"),
    lambda value: value["command"].append("--async-scheduling"),
    lambda value: value["command"].append("--async-scheduling=true"),
])
def test_async_scheduling_attestation_requires_false_and_unambiguous_inspected_flag(rig, mutation):
    path = rig.state / "actor-contract.json"
    value = json.loads(path.read_text())
    mutation(value)
    write_json(path, value)
    before = len(rig.actor.calls)
    with pytest.raises(ValueError, match="attestation|inspected actor command"):
        VllmInferenceRuntime(**rig.options)
    assert len(rig.actor.calls) == before


@pytest.mark.parametrize("field,value", [
    ("max_num_seqs", 32), ("max_num_seqs", True), ("kv_cache_dtype", "fp8"),
    ("attention_backend", "FLASHINFER"), ("mamba_cache_mode", "align"),
    ("cublas_workspace_config", ":16:8"), ("cublas_workspace_config", None),
])
def test_conservative_profile_attestation_rejects_missing_or_mismatched_settings(rig, field, value):
    path = rig.state / "actor-contract.json"
    contract = json.loads(path.read_text())
    if value is None:
        contract.pop(field)
    else:
        contract[field] = value
    write_json(path, contract)
    before = len(rig.actor.calls)
    with pytest.raises(ValueError, match="attestation"):
        VllmInferenceRuntime(**rig.options)
    assert len(rig.actor.calls) == before


@pytest.mark.parametrize("option,value", [
    ("--max-num-seqs", "32"), ("--kv-cache-dtype", "fp8"),
    ("--attention-backend", "FLASHINFER"), ("--mamba-cache-mode", "align"),
])
@pytest.mark.parametrize("mutation", ["change", "remove", "repeat"])
def test_conservative_profile_requires_matching_unambiguous_inspected_command(rig, option, value, mutation):
    path = rig.state / "actor-contract.json"
    contract = json.loads(path.read_text())
    command = contract["command"]
    index = command.index(option)
    if mutation == "change":
        command[index + 1] = value
    elif mutation == "remove":
        del command[index:index + 2]
    else:
        command.extend([option, value])
    write_json(path, contract)
    before = len(rig.actor.calls)
    with pytest.raises(ValueError, match="inspected actor command|repeats"):
        VllmInferenceRuntime(**rig.options)
    assert len(rig.actor.calls) == before


def test_streaming_is_explicitly_unsupported(rig):
    with pytest.raises(UpstreamStatusError, match="streaming"):
        asyncio.run(rig.runtime.inference_handler.inference_stream(rig.artifact, "/v1/chat/completions", {}))


def test_serving_and_evaluation_import_without_torch():
    source = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('CPU REEF imported torch')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import nvfp4_lora.reef_inference
import nvfp4_lora.reef_evaluation
"""
    subprocess.run([sys.executable, "-c", source], check=True, capture_output=True, text=True)
