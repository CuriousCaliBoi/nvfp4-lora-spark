from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from nvfp4_lora.reef_checkpoint import write_json
from nvfp4_lora.reef_evaluation import Gsm8kEvaluationFactory, read_holdout

from test_reef_inference import rig


@pytest.fixture
def evaluation(rig):
    path = rig.state / "heldout.json"
    write_json(path, {
        "schema_version": 1, "dataset": "openai/gsm8k", "dataset_revision": "pinned-test-revision",
        "split": "test", "seed": 43,
        "rows": [{"sample_id": f"test:{index}", "question": f"Question {index}", "answer": "#### 3"}
                 for index in range(16)],
    })
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    plugin = Gsm8kEvaluationFactory().build(
        {"holdout_path": str(path), "holdout_sha256": checksum},
        runtime=rig.runtime, training_runtime=SimpleNamespace(incumbent_checkpoint=rig.base), scenario="test", environ={},
    )
    candidate = replace(rig.candidate, training_metrics={
        "all_gradients_finite": True, "frozen_before_sha256": "f" * 64,
        "frozen_after_sha256": "f" * 64, "adapter_delta_norm": .2,
        "optimizer_step_before": 0, "optimizer_step_after": 1, "loss": .01, "gradient_norm": .1,
    })
    return SimpleNamespace(rig=rig, path=path, checksum=checksum, plugin=plugin, candidate=candidate)


def test_private_heldout_gate_preserves_head_and_contains_auditable_outputs(evaluation):
    r = evaluation.rig
    before = r.runtime.current_runtime_load_id()
    result = evaluation.plugin.evaluate(evaluation.candidate)
    decision = evaluation.plugin.decide(evaluation.candidate, result)
    assert decision.selected and decision.evaluation is result
    assert result.metrics["incumbent_correct"] == result.metrics["candidate_correct"] == 16
    assert len(result.metadata["examples"]) == 16
    assert result.metadata["holdout_sha256"] == evaluation.checksum
    assert result.metadata["dataset_revision"] == "pinned-test-revision"
    assert result.metadata["native_verification"]["native_verified"]
    assert r.runtime.current_runtime_load_id() == before == r.runtime.serving_runtime_load_id()
    assert r.runtime.pending_training_job_id is None
    requests = [payload for _, path, payload in r.actor.calls if path.endswith("chat/completions")]
    assert len(requests) == 32
    assert all(request["temperature"] == 0 and request["seed"] == 43 and request["max_tokens"] == 256 for request in requests)
    assert [request["messages"] for request in requests[:16]] == [request["messages"] for request in requests[16:]]


def test_regression_is_rejected_without_publication(evaluation):
    def wrong_candidate(response, request):
        if evaluation.rig.candidate_path.name in evaluation.rig.actor.registry[request["model"]]["root"]:
            response["choices"][0]["message"]["content"] = "#### 4"
    evaluation.rig.actor.chat_mutation = wrong_candidate
    result = evaluation.plugin.evaluate(evaluation.candidate)
    decision = evaluation.plugin.decide(evaluation.candidate, result)
    assert not decision.selected
    assert result.metrics["incumbent_correct"] == 16 and result.metrics["candidate_correct"] == 0
    assert evaluation.rig.runtime.pending_training_job_id is None


def test_truncated_final_marker_is_zero_reward_in_private_eval(evaluation):
    def truncated(response, request):
        if evaluation.rig.candidate_path.name in evaluation.rig.actor.registry[request["model"]]["root"]:
            response["choices"][0]["finish_reason"] = "length"
    evaluation.rig.actor.chat_mutation = truncated
    result = evaluation.plugin.evaluate(evaluation.candidate)
    assert result.metrics["candidate_correct"] == 0
    assert result.metrics["candidate_length_limited"] == 16
    assert not evaluation.plugin.decide(evaluation.candidate, result).selected


@pytest.mark.parametrize("key,value", [
    ("all_gradients_finite", False), ("adapter_delta_norm", 0), ("adapter_delta_norm", float("nan")),
    ("frozen_after_sha256", "a" * 64), ("optimizer_step_after", 2), ("loss", float("inf")),
])
def test_invalid_training_evidence_cannot_reach_private_sampling(evaluation, key, value):
    metrics = {**evaluation.candidate.training_metrics, key: value}
    candidate = replace(evaluation.candidate, training_metrics=metrics)
    before = len(evaluation.rig.actor.calls)
    with pytest.raises(ValueError):
        evaluation.plugin.evaluate(candidate)
    assert len(evaluation.rig.actor.calls) == before


def test_changed_holdout_fails_before_sampling(evaluation):
    evaluation.path.write_text(evaluation.path.read_text() + " ")
    before = len(evaluation.rig.actor.calls)
    with pytest.raises(ValueError, match="hash changed"):
        evaluation.plugin.evaluate(evaluation.candidate)
    assert len(evaluation.rig.actor.calls) == before


@pytest.mark.parametrize("mutation", ["duplicate", "train_split", "row_count", "bad_gold"])
def test_heldout_provenance_is_validated(evaluation, mutation):
    data = json.loads(evaluation.path.read_text())
    if mutation == "duplicate":
        data["rows"][1]["sample_id"] = data["rows"][0]["sample_id"]
    elif mutation == "train_split":
        data["split"] = "train"
    elif mutation == "row_count":
        data["rows"].pop()
    else:
        data["rows"][0]["answer"] = "The last number is 3"
    write_json(evaluation.path, data)
    with pytest.raises(ValueError):
        read_holdout(evaluation.path, hashlib.sha256(evaluation.path.read_bytes()).hexdigest())


def test_stale_source_cannot_be_evaluated_against_another_incumbent(evaluation):
    with pytest.raises(ValueError, match="source runtime is stale"):
        evaluation.plugin.evaluate(replace(evaluation.candidate, current_runtime_load_id="old:0"))
