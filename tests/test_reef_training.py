"""Actual REEF contracts with a fake subprocess testing transport, never model quality."""

import copy
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("reef")
from reef.core.records_types import AgentRecord, RequestType
from reef.core.evaluation import CandidateEvaluationPlugin, EvaluationResult, SelectionDecision
from reef.runtime.interfaces import ActivatedModel, InferenceRuntime, StaleCandidate
from reef.storage.sqlite import SQLiteRecordStore
from reef.train import Trainer
from reef.train.runtime_backend import RuntimeCandidateBackend
from reef.train.types import ProcessorContext

from nvfp4_lora.reef_checkpoint import MANIFEST, content_hash, file_hash, validate_checkpoint
from nvfp4_lora.reef_data import normalized_config
from nvfp4_lora.reef_recipe import GRPOStepPreparer, NVFP4Processor, NVFP4RolloutReport
from nvfp4_lora.reef_training import QuantizedTrainingRuntime
from nvfp4_lora.reef_worker import restore_training_state, save_training_state, update_from_rows, validate_job
from test_reef_checkpoint import artifact, capture, rows

PREPARER = "nvfp4_lora.reef_recipe:grpo_step_preparer"


def records(cycle="cycle-one", manifest=None, *, zero=False):
    pairs = []
    for index in range(2):
        native = capture()
        if manifest:
            native.update(adapter_sha256=manifest["adapter_sha256"],
                          adapter_config_sha256=manifest["files"]["adapter/adapter_config.json"])
        native["completion_token_ids"] = [3 if index else 2]
        native["tokens"] = native["prompt_token_ids"] + native["completion_token_ids"]
        score = 0.0 if zero else float(index)
        completion = f"#### {3 if score else 2}"
        inference = AgentRecord.create(scenario="test", request_type=RequestType.INFERENCE,
            agent_record_id=f"{cycle}-source-{index}", payload={
                "model": "test-model", "messages": [{"role": "user", "content": "one plus two"}],
                "runtime_load_id": native["runtime_load_id"],
                "response": {"training": native, "choices": [{"index": 0, "message": {
                    "role": "assistant", "content": completion}, "finish_reason": "stop"}]},
            })
        typed = NVFP4RolloutReport(score, cycle, 0, index, 2, 1, "openai/gsm8k", "dataset", "train", 2, "#### 3")
        report = AgentRecord.create(scenario="test", request_type=RequestType.REPORT,
            agent_record_id=f"{cycle}-report-{index}", payload=typed.to_dict(references=[inference.agent_record_id]))
        pairs.append((inference, report))
    return pairs


def processor():
    return NVFP4Processor(ProcessorContext("test", {"group_size": 2, "batch_group_count": 1},
                                          report_type=NVFP4RolloutReport))


def batch(manifest, cycle="cycle-one", **kwargs):
    value = processor()
    for inference, report in records(cycle, manifest, **kwargs):
        value.ingest(inference)
        value.ingest(report)
    assert value.ready()
    return value.build_batch()


def test_real_processor_barrier_references_and_skip():
    value = processor()
    pairs = records()
    value.ingest(pairs[0][0]); value.ingest(pairs[0][1])
    assert not value.ready()
    value.ingest(pairs[0][1])
    assert not value.ready()
    value.ingest(pairs[1][0]); value.ingest(pairs[1][1])
    assert value.ready()
    actual = value.build_batch()
    assert len(actual.items) == 2
    assert actual.items[0].source_agent_record_ids == (pairs[0][0].agent_record_id, pairs[0][1].agent_record_id)
    assert GRPOStepPreparer()(actual, {}).action == "train"
    assert GRPOStepPreparer()(batch(None, zero=True), {}).action == "skip"
    value.acknowledge(actual.batch_id)
    assert not value.ready()


def test_processor_rejects_mismatched_reward_and_mixed_version():
    value = processor()
    pairs = records()
    pairs[0][1].payload["score"] = 1.0
    value.ingest(pairs[0][0])
    with pytest.raises(ValueError, match="verifier"):
        value.ingest(pairs[0][1])
    value = processor()
    pairs = records()
    pairs[1][0].payload["runtime_load_id"] = "runtime:two"
    pairs[1][0].payload["response"]["training"]["runtime_load_id"] = "runtime:two"
    value.ingest(pairs[0][0]); value.ingest(pairs[0][1]); value.ingest(pairs[1][0])
    with pytest.raises(ValueError, match="mixed"):
        value.ingest(pairs[1][1])


@pytest.fixture
def runtime_factory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    worker = tmp_path / "fake_transport_worker.py"
    worker.write_text("""
import argparse, json, sys, tempfile
from pathlib import Path
sys.path.insert(0, ROOT)
from nvfp4_lora.reef_checkpoint import *
p=argparse.ArgumentParser(); p.add_argument('--job-file'); p.add_argument('--checkpoint-dir'); a=p.parse_args()
j=json.loads(Path(a.job_file).read_text()); d=Path(a.checkpoint_dir)
parent=validate_checkpoint(j['parent_checkpoint']) if j['parent_checkpoint'] else None
step=parent['optimizer_step']+1 if parent else 0
counter=Path(j['checkpoint_root']).parent/'worker_calls'
counter.write_text(str(int(counter.read_text())+1 if counter.exists() else 1))
t=Path(tempfile.mkdtemp(dir=d.parent))
for name in PAYLOAD_FILES:
    target=t/name; target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text('transport-only-state-'+str(step))
write_json(t/'metrics.json',{'optimizer_step_before':step-1 if step else 0,'optimizer_step_after':step})
m={k:j[k] for k in ('scenario','base_model','model_revision','lora_rank','lora_alpha','config','config_sha256','parent_checkpoint_id','training_job_id','batch_sha256')}
m.update(schema_version=1,checkpoint_id=checkpoint_identity(j),optimizer_step=step,model_config_sha256='c'*64,frozen_tensor_sha256='f'*64)
publish_checkpoint(t,d,m)
""".replace("ROOT", repr(str(root))))
    instances = []

    def create():
        runtime = QuantizedTrainingRuntime(state_root=tmp_path / "state", model_dir="/model/revision",
            base_model="test-model", model_revision="revision", worker_command=[sys.executable, str(worker)],
            training_config={}, scenario="test")
        instances.append(runtime)
        return runtime
    yield create
    for runtime in instances:
        runtime.shutdown()


def prepared(runtime, cycle="cycle-one", **kwargs):
    return runtime.prepare_training_step(batch(validate_checkpoint(runtime.incumbent_checkpoint), cycle, **kwargs),
        PREPARER, {}, 0, serving_runtime_load_id="runtime:one")


def test_candidate_commit_restart_second_cycle_rollback(runtime_factory):
    runtime = runtime_factory()
    base = runtime.base_checkpoint
    initial = validate_checkpoint(base)
    payload = prepared(runtime).payload
    candidate = runtime.train_candidate(payload)
    assert runtime.incumbent_checkpoint == base
    duplicate = runtime.train_candidate(payload)
    assert duplicate == candidate
    assert int((runtime.state_root / "worker_calls").read_text()) == 2
    runtime.commit_candidate(candidate.training_job_id)
    runtime.commit_candidate(candidate.training_job_id)
    assert runtime.train_candidate(payload) == candidate
    first = runtime.incumbent_checkpoint
    assert validate_checkpoint(first)["optimizer_step"] == 1
    runtime.shutdown()
    restarted = runtime_factory()
    assert restarted.incumbent_checkpoint == first
    second = restarted.train_candidate(prepared(restarted, "cycle-two").payload)
    second_manifest = validate_checkpoint(second.checkpoint_path)
    assert second_manifest["parent_checkpoint_id"] == candidate.candidate_id
    assert second_manifest["optimizer_step"] == 2
    restarted.commit_candidate(second.training_job_id)
    restarted.restore_checkpoint(artifact(first))
    assert restarted.incumbent_checkpoint == first
    assert file_hash(first / "optimizer.pt") == validate_checkpoint(first)["files"]["optimizer.pt"]
    restarted.restore_checkpoint(artifact(base))
    assert validate_checkpoint(restarted.incumbent_checkpoint) == initial
    with pytest.raises(ValueError, match="terminal"):
        restarted.commit_candidate(second.training_job_id)


def test_rejection_unknown_commit_and_zero_signal(runtime_factory):
    runtime = runtime_factory()
    base = runtime.incumbent_checkpoint
    assert prepared(runtime, zero=True).action == "skip"
    candidate = runtime.train_candidate(prepared(runtime).payload)
    runtime.reject_candidate(candidate, None)
    assert runtime.incumbent_checkpoint == base
    with pytest.raises(ValueError, match="terminal"):
        runtime.commit_candidate(candidate.training_job_id)
    with pytest.raises(ValueError):
        runtime.commit_candidate("d" * 64)
    runtime.shutdown()
    assert runtime_factory().incumbent_checkpoint == base


def test_stale_batch_and_single_writer(runtime_factory):
    runtime = runtime_factory()
    with pytest.raises(ValueError, match="owner"):
        runtime_factory()
    actual = batch(validate_checkpoint(runtime.incumbent_checkpoint))
    stale = runtime.prepare_training_step(actual, PREPARER, {}, 0, serving_runtime_load_id="runtime:two")
    with pytest.raises(StaleCandidate):
        runtime.train_candidate(stale.payload)
    payload = prepared(runtime).payload
    payload["parent_checkpoint_id"] = "e" * 64
    with pytest.raises(ValueError, match="identity"):
        runtime.train_candidate(payload)


def test_actual_candidate_backend_drops_stale_and_accepts_later_valid_work(runtime_factory):
    class Receiver(InferenceRuntime):
        version = "runtime:two"

        @property
        def inference_handler(self):
            return None

        def serving_runtime_load_id(self):
            return self.version

    runtime = runtime_factory()
    receiver = Receiver(base_url="http://unused")
    backend = RuntimeCandidateBackend(runtime, PREPARER, inference_runtime=receiver, scenario="test")
    actual = batch(validate_checkpoint(runtime.incumbent_checkpoint))
    dropped = backend.prepare_step(actual, {"batches": 3}, 3)
    assert dropped.outcome == "drop" and dropped.state == {"batches": 3}
    assert int((runtime.state_root / "worker_calls").read_text()) == 1
    receiver.version = "runtime:one"
    receiver.mark_published()
    valid = backend.prepare_step(actual, {"batches": 3}, 3)
    assert valid.outcome == "candidate"
    assert int((runtime.state_root / "worker_calls").read_text()) == 2
    assert runtime.incumbent_checkpoint == runtime.base_checkpoint


def test_actual_trainer_reload_reuses_candidate_after_evaluation_abort(runtime_factory):
    class Receiver(InferenceRuntime):
        version = "runtime:one"

        @property
        def inference_handler(self):
            return None

        def serving_runtime_load_id(self):
            return self.version

        def activate_candidate(self, candidate):
            self.version = "runtime:two"
            return ActivatedModel(candidate.candidate_id, self.version)

    class TransientEvaluator(CandidateEvaluationPlugin):
        def __init__(self):
            self.seen = []

        def evaluate(self, candidate):
            self.seen.append(candidate.candidate_id)
            if len(self.seen) == 1:
                raise RuntimeError("transient evaluation failure")
            return EvaluationResult("test", "1", {})

        def decide(self, candidate, evaluation):
            return SelectionDecision("select", "test", "1", "recovered evaluation", evaluation)

    runtime = runtime_factory()
    base = runtime.incumbent_checkpoint
    receiver = Receiver(base_url="http://unused")
    store = SQLiteRecordStore()
    for pair in records(manifest=validate_checkpoint(base)):
        for item in pair:
            store.append(item)
    evaluator = TransientEvaluator()

    def create_trainer():
        return Trainer.build("test", store,
            processor_factory=lambda context: NVFP4Processor(context.with_config({"group_size": 2, "batch_group_count": 1})),
            candidate_backend=RuntimeCandidateBackend(runtime, PREPARER, inference_runtime=receiver, scenario="test"),
            candidate_evaluator=evaluator, report_type=NVFP4RolloutReport)

    first = create_trainer()
    reserved = first.reserve_training_batch()
    with pytest.raises(RuntimeError, match="transient evaluation"):
        first.execute_reserved_step(0)
    identity = evaluator.seen[0]
    status = runtime.state_root / "jobs" / f"{identity}.status.json"
    assert json.loads(status.read_text())["status"] == "aborted"
    assert runtime.incumbent_checkpoint == base
    assert int((runtime.state_root / "worker_calls").read_text()) == 2
    first.close()
    reloaded = create_trainer()
    assert reloaded.reserve_training_batch().batch_id == reserved.batch_id
    result = reloaded.execute_reserved_step(0)
    assert result.outcome == "commit" and result.result.training_job_id == identity
    assert evaluator.seen == [identity, identity]
    assert json.loads(status.read_text())["status"] == "complete"
    assert runtime.incumbent_checkpoint == base
    assert int((runtime.state_root / "worker_calls").read_text()) == 2
    reloaded.close()


def test_normal_policy_rejection_replay_drops_without_worker(runtime_factory):
    class Receiver(InferenceRuntime):
        @property
        def inference_handler(self):
            return None

        def serving_runtime_load_id(self):
            return "runtime:one"

    runtime = runtime_factory()
    backend = RuntimeCandidateBackend(runtime, PREPARER, inference_runtime=Receiver(base_url="http://unused"), scenario="test")
    actual = batch(validate_checkpoint(runtime.incumbent_checkpoint))
    prepared_step = backend.prepare_step(actual, {}, 0)
    evaluation = EvaluationResult("test", "1", {})
    decision = SelectionDecision("reject", "strict_heldout_nonregression", "1", "candidate regression", evaluation)
    backend.settle_step(prepared_step, decision)
    backend.abort_step(prepared_step)
    replay = backend.prepare_step(actual, {}, 0)
    assert replay.outcome == "drop"
    assert replay.metrics["terminal_candidate_replay"] == 1
    assert runtime.incumbent_checkpoint == runtime.base_checkpoint
    assert int((runtime.state_root / "worker_calls").read_text()) == 2


def test_failed_rollback_reconciles_authoritative_head_before_learning(runtime_factory):
    runtime = runtime_factory()
    base = runtime.base_checkpoint
    candidate = runtime.train_candidate(prepared(runtime).payload)
    runtime.commit_candidate(candidate.training_job_id)
    durable_head = runtime.incumbent_checkpoint
    durable_optimizer_hash = file_hash(durable_head / "optimizer.pt")
    runtime.restore_checkpoint(artifact(base))
    runtime.shutdown()
    restarted = runtime_factory()
    assert restarted.incumbent_checkpoint == base
    current_batch = batch(validate_checkpoint(durable_head), "durable-head-cycle")
    with pytest.raises(ValueError, match="committed quantized policy"):
        restarted.prepare_training_step(current_batch, PREPARER, {}, 1, serving_runtime_load_id="runtime:one")
    # The serving recovery callback receives the authoritative artifact after native proof.
    restarted.restore_checkpoint(artifact(durable_head))
    assert file_hash(restarted.incumbent_checkpoint / "optimizer.pt") == durable_optimizer_hash
    assert restarted.prepare_training_step(current_batch, PREPARER, {}, 1,
                                          serving_runtime_load_id="runtime:one").action == "train"


def test_worker_job_validation_rejects_mutation_and_escaping_output(runtime_factory):
    runtime = runtime_factory()
    job = dict(prepared(runtime).payload)
    job.pop("source_runtime_load_id")
    source = runtime.state_root / "jobs" / "test.json"
    target = runtime.checkpoint_root / job["training_job_id"]
    validate_job(job, source, target)
    with pytest.raises(ValueError, match="destination"):
        validate_job(job, source, runtime.state_root / "escaped")
    changed = copy.deepcopy(job)
    changed["rows"][0]["behavior_logprobs"] = [-0.5]
    with pytest.raises(ValueError, match="identity"):
        validate_job(changed, source, target)
    with pytest.raises(ValueError, match="outside"):
        validate_job(job, runtime.state_root.parent / "outside.json", target)


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.embed.weight.requires_grad_(False)
        self.base = nn.Parameter(torch.randn(4, 8), requires_grad=False)
        self.lora_A = nn.Parameter(torch.randn(4, 2) * 0.1)
        self.lora_B = nn.Parameter(torch.zeros(2, 8))

    def forward(self, input_ids, attention_mask, use_cache):
        hidden = self.embed(input_ids)
        return SimpleNamespace(logits=hidden @ (self.base + self.lora_A @ self.lora_B))


def test_real_adam_and_rng_restart_matches_continuous_second_update(tmp_path):
    torch.manual_seed(42); random.seed(42)
    model = TinyPolicy()
    config = normalized_config()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0)
    samples = copy.deepcopy(rows())
    samples[0]["completion_ids"] = [2]
    frozen = model.base.clone()
    first = update_from_rows(model, optimizer, samples, config)
    assert first["all_gradients_finite"] and first["adapter_delta_norm"] > 0
    save_training_state(model, optimizer, tmp_path, 1)
    expected_random = (random.random(), torch.rand(3))
    update_from_rows(model, optimizer, samples, config)
    torch.manual_seed(42)
    reloaded = TinyPolicy()
    continued = torch.optim.AdamW([p for p in reloaded.parameters() if p.requires_grad], lr=1e-4, weight_decay=0)
    restore_training_state(reloaded, continued, tmp_path, 1)
    assert random.random() == expected_random[0]
    assert torch.equal(torch.rand(3), expected_random[1])
    update_from_rows(reloaded, continued, samples, config)
    assert torch.equal(model.lora_A, reloaded.lora_A)
    assert torch.equal(model.lora_B, reloaded.lora_B)
    assert torch.equal(model.base, frozen)
    for left, right in zip(optimizer.state.values(), continued.state.values(), strict=True):
        assert left["step"].item() == right["step"].item() == 2
        assert torch.equal(left["exp_avg"], right["exp_avg"])
        assert torch.equal(left["exp_avg_sq"], right["exp_avg_sq"])


def test_cpu_recipe_runtime_imports_forbid_torch():
    root = str(Path(__file__).resolve().parents[1])
    script = f"""
import sys, importlib.abc
sys.path.insert(0, {root!r})
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch imported by CPU REEF')
sys.meta_path.insert(0, Block())
import nvfp4_lora.reef_recipe, nvfp4_lora.reef_training
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-I", "-c", script], check=True)
