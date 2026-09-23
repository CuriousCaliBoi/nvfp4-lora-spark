"""Private, checksummed held-out evaluation for REEF's candidate gate."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.runtime.interfaces import ModelCandidate
from reef.train.evaluation import (
    CandidateEvaluationPlugin,
    CandidateEvaluationPluginFactory,
    EvaluationResult,
    SelectionDecision,
    UpdateCandidate,
)

from .reef_checkpoint import validate_checkpoint
from .reef_data import GSM8K_SYSTEM_PROMPT, gsm8k_reward
from .reef_inference import VllmInferenceRuntime


def read_holdout(path: Path, expected_sha256: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("held-out source hash changed")
    value = json.loads(raw)
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or value.get("dataset") != "openai/gsm8k" or value.get("split") != "test"
            or value.get("seed") != 43 or not isinstance(value.get("dataset_revision"), str)
            or not value["dataset_revision"]):
        raise ValueError("held-out file has incompatible dataset provenance")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 16:
        raise ValueError("held-out evaluation requires exactly 16 fixed test rows")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"sample_id", "question", "answer"} or any(
            not isinstance(row[key], str) or not row[key] for key in row
        ):
            raise ValueError("held-out rows require sample_id, question and answer strings")
        if gsm8k_reward(row["answer"], row["answer"]) != 1.0:
            raise ValueError("held-out answer lacks a valid final answer marker")
    if len({row["sample_id"] for row in rows}) != 16:
        raise ValueError("held-out sample IDs are not unique")
    return value


def _wilson(correct: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    midpoint = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [midpoint - radius, midpoint + radius]


def _training_gate(candidate: ModelCandidate, parent: dict, manifest: dict) -> None:
    metrics = candidate.training_metrics
    if metrics.get("all_gradients_finite") is not True:
        raise ValueError("candidate lacks finite-gradient evidence")
    frozen = metrics.get("frozen_before_sha256")
    if frozen != parent["frozen_tensor_sha256"] or frozen != metrics.get("frozen_after_sha256") or frozen != manifest["frozen_tensor_sha256"]:
        raise ValueError("candidate frozen-weight digest changed")
    delta = metrics.get("adapter_delta_norm")
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta <= 0:
        raise ValueError("candidate lacks a finite nonzero adapter update")
    if (metrics.get("optimizer_step_before") != parent["optimizer_step"]
            or metrics.get("optimizer_step_after") != manifest["optimizer_step"]
            or manifest["optimizer_step"] != parent["optimizer_step"] + 1):
        raise ValueError("candidate optimizer continuation does not match its parent")
    for name in ("loss", "gradient_norm"):
        if name in metrics and (isinstance(metrics[name], bool) or not isinstance(metrics[name], (int, float))
                                or not math.isfinite(metrics[name])):
            raise ValueError(f"candidate {name} is not finite")


def _scored(response: dict, answer: str) -> dict[str, Any]:
    choice = response["choices"][0]
    text = choice["message"]["content"]
    finish = choice["finish_reason"]
    return {
        "response_id": response.get("id"), "model": response["model"],
        "text": text, "finish_reason": finish,
        "score": 0.0 if finish == "length" else gsm8k_reward(text, answer),
        "prompt_token_ids": response["prompt_token_ids"], "completion_token_ids": choice["token_ids"],
    }


@dataclass(frozen=True)
class Gsm8kEvaluator(CandidateEvaluationPlugin):
    runtime: VllmInferenceRuntime
    training_runtime: Any
    scenario: str
    holdout_path: Path
    holdout_sha256: str
    max_tokens: int = 256
    seed: int = 43

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        if not isinstance(candidate, ModelCandidate):
            raise TypeError("quantized GSM8K evaluation requires ModelCandidate")
        data = read_holdout(self.holdout_path, self.holdout_sha256)
        incumbent_path = Path(self.training_runtime.incumbent_checkpoint)
        parent = validate_checkpoint(incumbent_path)
        manifest = validate_checkpoint(candidate.checkpoint_path)
        if (manifest["scenario"] != self.scenario or manifest["parent_checkpoint_id"] != parent["checkpoint_id"]
                or manifest["checkpoint_id"] != candidate.candidate_id
                or manifest["training_job_id"] != candidate.training_job_id):
            raise ValueError("candidate evaluation has a stale or mismatched checkpoint parent")
        _training_gate(candidate, parent, manifest)
        published = self.runtime.current_runtime_load_id()
        if candidate.current_runtime_load_id != published:
            raise ValueError("candidate evaluation source runtime is stale")
        requests = [{
            "messages": [{"role": "system", "content": GSM8K_SYSTEM_PROMPT},
                         {"role": "user", "content": row["question"]}],
            "temperature": 0.0, "n": 1, "seed": self.seed, "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        } for row in data["rows"]]
        before = self.runtime.evaluate_adapter(incumbent_path, requests)
        after = self.runtime.evaluate_adapter(candidate.checkpoint_path, requests)
        if len(before) != 16 or len(after) != 16:
            raise ValueError("held-out response count differs from the fixed test set")
        if self.runtime.current_runtime_load_id() != published:
            raise RuntimeError("private evaluation changed the published serving head")
        examples = [{"sample_id": row["sample_id"], "incumbent": _scored(old, row["answer"]),
                     "candidate": _scored(new, row["answer"])}
                    for row, old, new in zip(data["rows"], before, after)]
        old_correct = int(sum(example["incumbent"]["score"] for example in examples))
        new_correct = int(sum(example["candidate"]["score"] for example in examples))
        return EvaluationResult(
            evaluator="nvfp4_gsm8k_strict", evaluator_version="1",
            metrics={
                "sample_count": 16, "incumbent_correct": old_correct, "candidate_correct": new_correct,
                "incumbent_accuracy": old_correct / 16, "candidate_accuracy": new_correct / 16,
                "incumbent_wilson95": _wilson(old_correct, 16), "candidate_wilson95": _wilson(new_correct, 16),
                "incumbent_length_limited": sum(item["incumbent"]["finish_reason"] == "length" for item in examples),
                "candidate_length_limited": sum(item["candidate"]["finish_reason"] == "length" for item in examples),
            },
            metadata={
                "holdout_sha256": self.holdout_sha256, "dataset": data["dataset"],
                "dataset_revision": data["dataset_revision"], "split": data["split"],
                "seed": self.seed, "max_tokens": self.max_tokens, "temperature": 0.0,
                "incumbent_checkpoint_id": parent["checkpoint_id"], "candidate_checkpoint_id": manifest["checkpoint_id"],
                "source_runtime_load_id": published, "examples": examples,
                "native_verification": self.runtime.verification(candidate.checkpoint_path),
                "scope": "Sixteen fixed test rows give a bounded selection check, not a broad retention or quality claim.",
            },
        )

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        if evaluation.metadata.get("candidate_checkpoint_id") != candidate.candidate_id:
            raise ValueError("selection received another candidate's evaluation")
        before, after = evaluation.metrics["incumbent_correct"], evaluation.metrics["candidate_correct"]
        selected = after >= before
        return SelectionDecision(
            outcome="select" if selected else "reject", policy="strict_heldout_nonregression", policy_version="1",
            reason=f"strict held-out score {after}/16 {'meets' if selected else 'falls below'} incumbent {before}/16",
            evaluation=evaluation, metrics={"strict_heldout_nonregression": selected},
        )


class Gsm8kEvaluationFactory(CandidateEvaluationPluginFactory):
    def build(self, config, *, runtime, training_runtime, scenario, environ):
        del environ
        if not isinstance(runtime, VllmInferenceRuntime):
            raise TypeError("GSM8K evaluation requires the native quantized vLLM runtime")
        if set(config) - {"holdout_path", "holdout_sha256", "max_tokens", "seed"}:
            raise ValueError("unsupported GSM8K evaluation configuration")
        if config.get("max_tokens", 256) != 256 or config.get("seed", 43) != 43:
            raise ValueError("held-out protocol fixes max_tokens=256 and seed=43")
        path, checksum = Path(config["holdout_path"]).resolve(), config["holdout_sha256"]
        read_holdout(path, checksum)
        return Gsm8kEvaluator(runtime, training_runtime, scenario, path, checksum)
