#!/usr/bin/env python3
"""Bounded local GRPO on one frozen quantized Nemotron checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from nvfp4_lora.grpo import (Trajectory, behavior_metrics, collate, group_advantages,
                             gsm8k_reward, policy_logprobs, policy_loss, prompt_ids)
from nvfp4_lora.grpo_rollout import Rollout, reload_proof


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def append_json(path: Path, value) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resource_snapshot() -> dict:
    import psutil

    result = {"unix_time": time.time(), "process_rss_bytes": psutil.Process().memory_info().rss,
              "system_available_bytes": psutil.virtual_memory().available}
    if torch.cuda.is_initialized():
        result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                      cuda_reserved_bytes=torch.cuda.memory_reserved(),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    return result


def parameter_norm(parameters, *, gradient=False) -> float:
    values = [p.grad if gradient else p for p in parameters]
    return math.sqrt(sum(v.detach().float().square().sum().item() for v in values if v is not None))


def require_gradient(parameters) -> float:
    if any(p.grad is None for p in parameters):
        raise RuntimeError("some trainable adapter parameters did not receive gradients")
    norm = parameter_norm(parameters, gradient=True)
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError(f"expected a finite nonzero adapter gradient, got {norm}")
    return norm


def adapter_gradient_audit(model, expected_targets: list[str]) -> dict:
    norms = {}
    failures = []
    expected_names = {f"{name}.lora_{side}" for name in expected_targets for side in ("A", "B")}
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(trainable) != expected_names:
        failures.append("trainable parameter names do not match the attention adapter audit")
    for name, parameter in trainable.items():
        if parameter.grad is None:
            norms[name] = None
            failures.append(f"missing gradient: {name}")
            continue
        norm = parameter.grad.detach().float().norm().item()
        norms[name] = norm if math.isfinite(norm) else None
        if not math.isfinite(norm):
            failures.append(f"non-finite gradient: {name}")
        elif name.endswith(".lora_B") and norm == 0:
            failures.append(f"zero B gradient: {name}")
    return {"per_parameter_gradient_norms": norms, "gradient_failures": failures}


def gradient_gate(model, tokenizer, temperature: float) -> dict:
    from nvfp4_lora.learner import audit_quantized_learner, frozen_tensor_digest

    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    prompt = prompt_ids(tokenizer, "What is 17 + 25?")
    completion = tokenizer.encode("17 + 25 = 42.\n#### 42", add_special_tokens=False)
    batch = collate([Trajectory(prompt, completion, [0.0] * len(completion), "", 0.0, 0)],
                    tokenizer.pad_token_id, "cuda")
    before = frozen_tensor_digest(model)
    audit_before = audit_quantized_learner(model)
    old_adapter = [p.detach().clone() for p in parameters]
    model.zero_grad(set_to_none=True)
    logprobs = policy_logprobs(model, batch, temperature)
    loss = -logprobs[batch.completion_mask].mean()
    if not torch.isfinite(loss):
        raise RuntimeError("full-model gradient probe produced a non-finite loss")
    loss.backward()
    gradients = adapter_gradient_audit(model, audit_before["attention_targets"])
    norm = parameter_norm(parameters, gradient=True)
    after = frozen_tensor_digest(model)
    audit_after = audit_quantized_learner(model)
    if before != after or any(not torch.equal(p, old) for p, old in zip(parameters, old_adapter, strict=True)):
        raise RuntimeError("gradient-only probe mutated base weights or adapters")
    if audit_before["frozen_dtype_bytes"] != audit_after["frozen_dtype_bytes"]:
        raise RuntimeError("policy scoring changed persistent frozen tensor storage")
    result = {"passed": not gradients["gradient_failures"], "diagnostic_only_not_an_rl_update": True,
              "loss": loss.item(), "gradient_norm": norm if math.isfinite(norm) else None, "frozen_digest_before": before,
              "frozen_digest_after": after, "audit_before": audit_before, "audit_after": audit_after,
              "prompt_ids": prompt, "completion_ids": completion,
              "gradient_parameter_names": [n for n, p in model.named_parameters() if p.grad is not None],
              "resources": resource_snapshot(), **gradients}
    model.zero_grad(set_to_none=True)
    del old_adapter, logprobs, loss, batch
    torch.cuda.empty_cache()
    return result


def load_gsm8k(args, output: Path):
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download("openai/gsm8k", repo_type="dataset",
                                     revision=args.dataset_revision, local_files_only=args.offline))
    if snapshot.name != args.dataset_revision:
        raise RuntimeError("dataset snapshot does not match the pinned revision")
    files = {split: sorted((snapshot / "main").glob(f"{split}-*.parquet")) for split in ("train", "test")}
    if any(not paths for paths in files.values()):
        raise FileNotFoundError("both canonical GSM8K main train and test parquet files must be cached")
    dataset = load_dataset("parquet", data_files={s: [str(p) for p in ps] for s, ps in files.items()},
                           cache_dir=str(output / "dataset-cache"))
    provenance = {"dataset": "openai/gsm8k", "configuration": "main", "revision": snapshot.name,
                  "files": {s: [{"name": p.name, "sha256": file_hash(p)} for p in ps] for s, ps in files.items()},
                  "fingerprints": {s: dataset[s]._fingerprint for s in files},
                  "split_lengths": {s: len(dataset[s]) for s in files}}
    return dataset, provenance


def record_generations(rollout, tokenizer, rows: list[dict], *, n: int, temperature: float,
                       seed: int, args, path: Path, phase: str, step: int, attempt: int) -> list[Trajectory]:
    prompts = [prompt_ids(tokenizer, row["question"]) for row in rows]
    if any(len(p) + args.max_new_tokens > args.max_model_len for p in prompts):
        raise ValueError("prompt plus completion budget exceeds --max-model-len; prompts are never silently truncated")
    outputs = rollout.generate(prompts, n=n, max_tokens=args.max_new_tokens, temperature=temperature, seed=seed)
    trajectories = []
    for item in outputs:
        group = item["group_id"]
        row = rows[group]
        trajectory = Trajectory(prompt_ids=prompts[group], reward=gsm8k_reward(item["completion"], row["answer"]), **item)
        trajectories.append(trajectory)
        append_json(path, {**asdict(trajectory), "phase": phase, "step": step, "attempt": attempt,
                           "dataset_row": row["dataset_row"], "question": row["question"], "gold_answer": row["answer"],
                           "temperature": temperature, "seed": seed,
                           "sampling": {"top_p": 1.0, "top_k": -1, "min_p": 0.0},
                           "behavior_logprob_semantics": "vllm_processed_logprobs"})
    return trajectories


def evaluation(rollout, tokenizer, rows, args, output: Path, label: str) -> dict:
    samples = record_generations(rollout, tokenizer, rows, n=1, temperature=0.0, seed=args.seed,
                                 args=args, path=output / f"eval-{label}.jsonl", phase=f"eval-{label}", step=0, attempt=0)
    return {"count": len(samples), "correct": sum(t.reward for t in samples),
            "accuracy": sum(t.reward for t in samples) / len(samples),
            "length_limited": sum(t.finish_reason == "length" for t in samples),
            "dataset_rows": [r["dataset_row"] for r in rows]}


def update(model, tokenizer, samples, args, output: Path, step: int, optimizer) -> dict:
    from nvfp4_lora.learner import audit_quantized_learner, frozen_tensor_digest

    start_time = time.monotonic()
    model.train()
    batch = collate(samples, tokenizer.pad_token_id, "cuda")
    rewards = torch.tensor([t.reward for t in samples], device="cuda")
    groups = torch.tensor([t.group_id for t in samples], device="cuda")
    advantages = group_advantages(rewards, groups)
    before = frozen_tensor_digest(model)
    audit_before = audit_quantized_learner(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    adapter_before = [p.detach().clone() for p in parameters]
    old_parts = []
    with torch.no_grad():
        for start in range(0, len(samples), args.microbatch_size):
            old_parts.append(policy_logprobs(model, batch.select(start, start + args.microbatch_size), args.temperature))
    old = torch.cat(old_parts).detach()
    if not torch.isfinite(old[batch.completion_mask]).all():
        raise RuntimeError("pre-update learner log probabilities are not finite")
    for index, sample in enumerate(samples):
        append_json(output / "learner-logprobs.jsonl", {"step": step, "trajectory_index": index,
                    "temperature": args.temperature, "advantage": advantages[index].item(),
                    "old_logprobs": old[index][batch.completion_mask[index]].tolist()})
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    for start in range(0, len(samples), args.microbatch_size):
        end = min(start + args.microbatch_size, len(samples))
        micro = batch.select(start, end)
        current = policy_logprobs(model, micro, args.temperature)
        loss = policy_loss(current, old[start:end], micro.behavior_logprobs,
                           micro.completion_mask, advantages[start:end],
                           clip_epsilon=args.clip_epsilon, tis_min=args.tis_min, tis_max=args.tis_max)
        if not torch.isfinite(loss):
            raise RuntimeError("GRPO loss is not finite")
        # Uneven final microbatches must contribute their actual sample count.
        weight = (end - start) / len(samples)
        (loss * weight).backward()
        accumulated_loss += loss.item() * weight
    gradient_norm = require_gradient(parameters)
    torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    delta = math.sqrt(sum((p.detach().float() - old_p.float()).square().sum().item()
                          for p, old_p in zip(parameters, adapter_before, strict=True)))
    if not math.isfinite(delta) or delta <= 0 or any(not torch.isfinite(p).all() for p in parameters):
        raise RuntimeError("optimizer did not produce a finite nonzero adapter update")
    after = frozen_tensor_digest(model)
    audit_after = audit_quantized_learner(model)
    if before != after:
        raise RuntimeError("a frozen tensor changed during RL")
    if audit_before["frozen_dtype_bytes"] != audit_after["frozen_dtype_bytes"]:
        raise RuntimeError("persistent frozen storage changed during RL")
    result = {"step": step, "loss": accumulated_loss, "gradient_norm_before_clip": gradient_norm,
              "adapter_delta_norm": delta, "adapter_norm": parameter_norm(parameters),
              "reward_mean": rewards.mean().item(), "zero_variance_groups": sum(
                  len({s.reward for s in samples if s.group_id == group}) <= 1 for group in range(args.batch_size)),
              "length_limited_completions": sum(s.finish_reason == "length" for s in samples),
              "empty_completions": sum(not s.completion_ids for s in samples),
              "frozen_digest_before": before, "frozen_digest_after": after,
              "audit_before": audit_before, "audit_after": audit_after,
              "seconds": time.monotonic() - start_time, "resources": resource_snapshot(),
              **behavior_metrics(old, batch.behavior_logprobs, batch.completion_mask, args.tis_min, args.tis_max)}
    append_json(output / "metrics.jsonl", result)
    del batch, old, old_parts, adapter_before, current, loss
    torch.cuda.empty_cache()
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Local HF NVFP4 snapshot directory, shared by learner and actor")
    parser.add_argument("--model-revision", default="bee7596271d1495f6992ae224aefde4410e816b8")
    parser.add_argument("--dataset-revision", default="740312add88f781978c0658806c59bc2815b9866")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--forward-backward-only", action="store_true")
    for option, default in (("steps", 1), ("batch-size", 4), ("num-generations", 8), ("max-new-tokens", 256),
                            ("max-model-len", 1024), ("eval-size", 16), ("seed", 42), ("lora-rank", 8),
                            ("microbatch-size", 1), ("max-batch-attempts", 3)):
        parser.add_argument(f"--{option}", type=int, default=default)
    for option, default in (("temperature", 1.2), ("learning-rate", 1e-4), ("lora-alpha", 16),
                            ("clip-epsilon", 0.2), ("tis-min", 0.1), ("tis-max", 10),
                            ("max-grad-norm", 1.0), ("gpu-memory-utilization", 0.23)):
        parser.add_argument(f"--{option}", type=float, default=default)
    args = parser.parse_args(argv)
    for name in ("steps", "batch_size", "num_generations", "max_new_tokens", "max_model_len", "eval_size",
                 "lora_rank", "microbatch_size", "max_batch_attempts"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_generations < 2 or args.max_batch_attempts > 3:
        parser.error("GRPO needs >=2 generations per group and permits at most 3 candidate batches")
    for name in ("temperature", "learning_rate", "lora_alpha", "max_grad_norm", "tis_min", "tis_max"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not 0 < args.clip_epsilon < 1 or not 0 < args.gpu_memory_utilization < 1 or args.tis_min > args.tis_max:
        parser.error("invalid clipping, importance-weight, or GPU-memory bounds")
    return args


def run(args, output: Path) -> dict:
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, set_seed
    from nvfp4_lora.learner import build_quantized_learner, export_attention_adapter, audit_quantized_learner

    model_dir = Path(args.model_dir).resolve()
    if model_dir.name != args.model_revision:
        raise ValueError("--model-dir must resolve to the exact --model-revision snapshot directory")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "vllm", "datasets", "safetensors")}
    manifest = {"arguments": vars(args), "model_snapshot": str(model_dir), "model_revision": model_dir.name,
                "checkpoint_metadata_sha256": {p.name: file_hash(p) for p in
                    (model_dir / "config.json", model_dir / "model.safetensors.index.json") if p.exists()},
                "versions": versions, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                "objective": "one-epoch sample-mean clipped GRPO; detached token-clipped old-learner/behavior correction",
                "policy_temperature": args.temperature, "behavior_logprobs": "vllm_processed_logprobs",
                "reward": "flexible numeric GSM8K: final ####, boxed, or last number", "dropout": 0,
                "enable_thinking": False,
                "gradient_checkpointing": "non-reentrant", "initial_resources": resource_snapshot()}
    code_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    manifest["code_revision"] = code_revision.stdout.strip() if code_revision.returncode == 0 else None
    write_json(output / "manifest.json", manifest)
    print("Loading the quantized learner", flush=True)
    model = build_quantized_learner(str(model_dir), lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                                    device="cuda", dtype=torch.bfloat16)
    model.config.use_cache = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    manifest["learner_audit"] = audit_quantized_learner(model)
    write_json(output / "manifest.json", manifest)
    print("Running full-model gradient-only gate", flush=True)
    gate = gradient_gate(model, tokenizer, args.temperature)
    write_json(output / "gradient-gate.json", gate)
    if not gate["passed"]:
        raise RuntimeError("full-model adapter gradients failed; inspect gradient-gate.json")
    if args.forward_backward_only:
        return {"status": "gradient_gate_passed", "rl_updates": 0, "gate": gate}
    dataset, provenance = load_gsm8k(args, output)
    train_indices = list(range(len(dataset["train"]) - 512))
    random.Random(args.seed).shuffle(train_indices)
    required = args.steps * args.max_batch_attempts * args.batch_size
    if len(train_indices) < required or len(dataset["test"]) < args.eval_size:
        raise ValueError("insufficient disjoint training batches or held-out evaluation rows")
    candidates = [train_indices[i:i + args.batch_size] for i in range(0, required, args.batch_size)]
    eval_indices = list(range(len(dataset["test"])))
    random.Random(args.seed + 1).shuffle(eval_indices)
    eval_indices = eval_indices[:args.eval_size]
    eval_rows = [{**dataset["test"][i], "dataset_row": i} for i in eval_indices]
    manifest.update(dataset=provenance, candidate_train_batches=candidates, heldout_test_rows=eval_indices,
                    excluded_train_tail=512)
    write_json(output / "manifest.json", manifest)
    initial_adapter = export_attention_adapter(model, output / "adapters" / "initial", base_model_name=str(model_dir))
    print("Initializing vLLM actor from the same NVFP4 snapshot", flush=True)
    rollout = Rollout(str(model_dir), max_model_len=args.max_model_len, rank=args.lora_rank, seed=args.seed,
                      gpu_memory_utilization=args.gpu_memory_utilization)
    manifest["actor_engine_options"] = rollout.engine_options
    manifest["initial_adapter"] = initial_adapter
    write_json(output / "manifest.json", manifest)
    rollout.load_adapter(initial_adapter["directory"])
    fixed_ids = prompt_ids(tokenizer, "What is 17 + 25?") + tokenizer.encode("17 + 25 = 42.\n#### 42", add_special_tokens=False)
    fixed_before = rollout.fixed_sequence_logprobs(fixed_ids, seed=args.seed)
    baseline_repeat = rollout.fixed_sequence_logprobs(fixed_ids, seed=args.seed)
    write_json(output / "fixed-sequence-initial.json", {"token_ids": fixed_ids, "logprobs": fixed_before,
                "repeat_logprobs": baseline_repeat, "temperature": 1.0})
    print("Evaluating initial adapter on held-out test questions", flush=True)
    before_eval = evaluation(rollout, tokenizer, eval_rows, args, output, "before")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=0.0)
    completed = 0
    step_metrics = []
    for step in range(1, args.steps + 1):
        usable = None
        for attempt in range(args.max_batch_attempts):
            rows_ids = candidates[(step - 1) * args.max_batch_attempts + attempt]
            rows = [{**dataset["train"][i], "dataset_row": i} for i in rows_ids]
            print(f"Sampling RL update {step}, candidate batch {attempt + 1}", flush=True)
            samples = record_generations(rollout, tokenizer, rows, n=args.num_generations,
                        temperature=args.temperature, seed=args.seed + (step - 1) * args.max_batch_attempts + attempt,
                        args=args, path=output / "train-trajectories.jsonl", phase="train", step=step, attempt=attempt + 1)
            rewards = torch.tensor([s.reward for s in samples])
            advantages = group_advantages(rewards, torch.tensor([s.group_id for s in samples]))
            usable = samples if any(a != 0 and s.completion_ids for a, s in zip(advantages.tolist(), samples, strict=True)) else None
            append_json(output / "batch-attempts.jsonl", {"step": step, "attempt": attempt + 1, "dataset_rows": rows_ids,
                        "rewards": rewards.tolist(), "advantages": advantages.tolist(), "usable": usable is not None})
            if usable is not None:
                break
        if usable is None:
            return {"status": "inconclusive_zero_variance", "rl_updates": completed, "before_eval": before_eval,
                    "message": "All predeclared candidate batches lacked a usable group-relative gradient.", "metrics": step_metrics}
        print(f"Updating attention adapters with GRPO, update {step}", flush=True)
        metrics = update(model, tokenizer, usable, args, output, step, optimizer)
        step_metrics.append(metrics)
        completed += 1
        adapter = export_attention_adapter(model, output / "adapters" / f"update-{step:04d}", base_model_name=str(model_dir))
        native_path = output / "adapters" / f"update-{step:04d}" / "learner_adapter.safetensors"
        save_file({name: p.detach().cpu().contiguous() for name, p in model.named_parameters() if p.requires_grad}, str(native_path))
        adapter["native_sha256"] = file_hash(native_path)
        load_info = rollout.load_adapter(adapter["directory"])
        fixed_after = rollout.fixed_sequence_logprobs(fixed_ids, seed=args.seed)
        reload_info = rollout.load_adapter(adapter["directory"])
        repeated = rollout.fixed_sequence_logprobs(fixed_ids, seed=args.seed)
        proof = {**reload_proof(fixed_before, fixed_after, repeated, baseline_repeat), "step": step, "token_ids": fixed_ids,
                 "before_logprobs": fixed_before, "updated_logprobs": fixed_after, "reloaded_logprobs": repeated,
                 "baseline_repeat_logprobs": baseline_repeat,
                 "temperature": 1.0, "adapter": adapter, "load": load_info, "reload": reload_info}
        write_json(output / f"reload-proof-{step:04d}.json", proof)
        if not proof["passed"]:
            raise RuntimeError("updated adapter failed the fixed-sequence policy-change/reload proof")
        fixed_before = fixed_after
        baseline_repeat = repeated
    print("Evaluating updated adapter on the same held-out test questions", flush=True)
    after_eval = evaluation(rollout, tokenizer, eval_rows, args, output, "after")
    return {"status": "complete", "rl_updates": completed, "before_eval": before_eval,
            "after_eval": after_eval, "metrics": step_metrics,
            "interpretation": "Execution and synchronization smoke only; this sample does not establish a learning improvement."}


def main(argv=None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("--output-dir must be empty to preserve earlier experiment evidence")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    status = {"status": "failed", "rl_updates": 0}
    exit_code = 1
    try:
        status = run(args, output)
        exit_code = 0 if status["status"] in ("complete", "gradient_gate_passed") else 2
    except Exception as error:
        status.update(error=str(error), traceback=traceback.format_exc())
        metrics_path = output / "metrics.jsonl"
        if metrics_path.exists():
            status["rl_updates"] = len(metrics_path.read_text().splitlines())
        traceback.print_exc()
    finally:
        status.update(elapsed_seconds=time.monotonic() - started, exit_code=exit_code, resources=resource_snapshot())
        write_json(output / "result.json", status)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
