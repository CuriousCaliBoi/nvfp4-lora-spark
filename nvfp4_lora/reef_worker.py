"""One isolated quantized-learner job; serving is owned by REEF's actor runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
import tempfile

from .reef_checkpoint import (
    checkpoint_identity, content_hash, file_hash, publish_checkpoint, validate_checkpoint, write_json,
)
from .reef_data import advantages, normalized_config, validate_rows


def trainable_parameters(model):
    return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


def save_training_state(model, optimizer, directory, optimizer_step):
    import torch
    from safetensors.torch import save_file

    parameters = trainable_parameters(model)
    save_file({name: parameter.detach().cpu().contiguous() for name, parameter in parameters.items()},
              str(directory / "native_adapter.safetensors"))
    torch.save({"parameter_names": list(parameters), "optimizer_step": optimizer_step,
                "state_dict": optimizer.state_dict()}, directory / "optimizer.pt")
    torch.save({"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
               directory / "rng.pt")


def restore_training_state(model, optimizer, directory, expected_step):
    import torch
    from safetensors.torch import load_file

    parameters = trainable_parameters(model)
    native = load_file(str(directory / "native_adapter.safetensors"), device="cpu")
    if set(native) != set(parameters):
        raise ValueError("native adapter parameter names differ from learner")
    with torch.no_grad():
        for name, parameter in parameters.items():
            if native[name].shape != parameter.shape or native[name].dtype != parameter.dtype:
                raise ValueError(f"incompatible native adapter tensor {name}")
            parameter.copy_(native[name])
    saved = torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=True)
    if saved["parameter_names"] != list(parameters) or saved["optimizer_step"] != expected_step:
        raise ValueError("optimizer ordering/step differs from checkpoint")
    optimizer.load_state_dict(saved["state_dict"])
    for state in optimizer.state.values():
        if "step" in state and int(state["step"].item()) != expected_step:
            raise ValueError("AdamW state step differs from checkpoint")
    if expected_step > 0 and len(optimizer.state) != len(parameters):
        raise ValueError("trained checkpoint has incomplete AdamW state")
    rng = torch.load(directory / "rng.pt", map_location="cpu", weights_only=True)
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch_cpu"])
    if len(rng["torch_cuda"]) != torch.cuda.device_count():
        raise ValueError("checkpoint CUDA RNG topology differs from learner")
    if rng["torch_cuda"]:
        torch.cuda.set_rng_state_all(rng["torch_cuda"])


def update_from_rows(model, optimizer, rows, config, *, pad_token_id=0):
    import torch
    from .grpo import Trajectory, behavior_metrics, collate, group_advantages, policy_logprobs, policy_loss

    rows = validate_rows(rows)
    device = next(iter(trainable_parameters(model).values())).device
    batch = collate([Trajectory(row["prompt_ids"], row["completion_ids"], row["behavior_logprobs"],
                               row["response_text"], row["reward"], row["group_id"], row["finish_reason"])
                     for row in rows], pad_token_id, device)
    weights = group_advantages(torch.tensor([row["reward"] for row in rows], device=device),
                               torch.tensor([row["group_id"] for row in rows], device=device))
    if not weights.any():
        raise ValueError("zero-signal batch cannot execute an optimizer update")
    parameters = trainable_parameters(model)
    before = {name: parameter.detach().float().clone() for name, parameter in parameters.items()}
    micro_size = config["microbatch_size"]
    model.train()
    with torch.no_grad():
        old = torch.cat([policy_logprobs(model, batch.select(start, min(start + micro_size, len(rows))),
                                        config["temperature"])
                         for start in range(0, len(rows), micro_size)])
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for start in range(0, len(rows), micro_size):
        end = min(start + micro_size, len(rows))
        micro = batch.select(start, end)
        current = policy_logprobs(model, micro, config["temperature"])
        loss = policy_loss(current, old[start:end], micro.behavior_logprobs, micro.completion_mask,
                           weights[start:end], clip_epsilon=config["clip_epsilon"],
                           tis_min=config["tis_min"], tis_max=config["tis_max"])
        loss = loss * ((end - start) / len(rows))
        if not torch.isfinite(loss):
            raise ValueError("nonfinite GRPO loss")
        loss.backward()
        total_loss += loss.detach().item()
    norms = {}
    for name, parameter in parameters.items():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError(f"missing or nonfinite adapter gradient: {name}")
        norms[name] = parameter.grad.float().norm().item()
    norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), config["max_grad_norm"], error_if_nonfinite=True)
    if norm.item() <= 0:
        raise ValueError("optimizer update requires nonzero gradients")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    delta = sum((parameter.detach().float() - before[name]).square().sum() for name, parameter in parameters.items()).sqrt()
    if not torch.isfinite(delta) or delta.item() <= 0:
        raise ValueError("optimizer did not produce a finite nonzero adapter change")
    return {"loss": total_loss, "all_gradients_finite": True,
            "gradient_norm_before_clip": norm.item(), "parameter_gradient_norms": norms,
            "adapter_delta_norm": delta.item(),
            **behavior_metrics(old, batch.behavior_logprobs, batch.completion_mask, config["tis_min"], config["tis_max"])}


def validate_job(job, job_file, checkpoint_dir):
    if job.get("schema_version") != 1 or job.get("kind") not in ("bootstrap", "train"):
        raise ValueError("unsupported worker job schema/kind")
    root = Path(job["checkpoint_root"])
    if not root.is_absolute() or root.resolve() != root:
        raise ValueError("checkpoint root must be a canonical absolute path")
    if not Path(job_file).resolve().is_relative_to(root.parent):
        raise ValueError("worker job file is outside configured state root")
    identity = checkpoint_identity(job)
    if Path(checkpoint_dir).resolve() != root / identity:
        raise ValueError("worker destination disagrees with immutable job identity")
    config = normalized_config(job["config"])
    if config != job["config"] or content_hash(config) != job["config_sha256"]:
        raise ValueError("worker training configuration hash mismatch")
    if job["kind"] == "bootstrap":
        if any(job.get(key) is not None for key in ("parent_checkpoint", "parent_checkpoint_id", "training_job_id", "batch_sha256", "batch_id")) or job["rows"]:
            raise ValueError("bootstrap cannot contain training state")
    else:
        rows = validate_rows(job["rows"])
        if rows != job["rows"] or not any(advantages(rows)):
            raise ValueError("ordered nonzero-signal training rows required")
        if job["training_job_id"] != identity or content_hash({"batch_id": job["batch_id"], "rows": rows}) != job["batch_sha256"]:
            raise ValueError("worker batch identity mismatch")
        if Path(job["parent_checkpoint"]).resolve() != root / job["parent_checkpoint_id"]:
            raise ValueError("worker parent is outside checkpoint root or mismatched")
        if any(len(row["prompt_ids"]) + len(row["completion_ids"]) > config["max_model_len"] for row in rows):
            raise ValueError("native trajectory exceeds configured context")
    return identity, config


def run_job(job_file, checkpoint_dir):
    job_file, checkpoint_dir = Path(job_file), Path(checkpoint_dir)
    job = json.loads(job_file.read_text())
    identity, config = validate_job(job, job_file, checkpoint_dir)
    if checkpoint_dir.exists():
        result = validate_checkpoint(checkpoint_dir, expected_config_sha256=job["config_sha256"])
        if result["checkpoint_id"] != identity:
            raise ValueError("existing worker output has different identity")
        return result
    import torch
    from .learner import audit_quantized_learner, build_quantized_learner, export_attention_adapter, frozen_tensor_digest

    model_dir = Path(job["model_dir"])
    if model_dir.name != job["model_revision"]:
        raise ValueError("model_dir must identify the exact pinned snapshot revision")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    model = build_quantized_learner(model_dir, lora_rank=job["lora_rank"], lora_alpha=job["lora_alpha"])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    audit = audit_quantized_learner(model)
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(list(parameters.values()), lr=config["learning_rate"],
                                  betas=tuple(config["betas"]), eps=config["eps"], weight_decay=config["weight_decay"])
    frozen_before = frozen_tensor_digest(model)
    model_config_hash = file_hash(model_dir / "config.json")
    step = 0
    metrics = {"source_agent_record_ids": [], "optimizer_step_before": 0, "optimizer_step_after": 0,
               "frozen_before_sha256": frozen_before, "frozen_after_sha256": frozen_before,
               "audit": audit}
    if job["kind"] == "train":
        parent_path = Path(job["parent_checkpoint"])
        parent = validate_checkpoint(parent_path, expected_base_model=job["base_model"],
                                     expected_model_revision=job["model_revision"], expected_lora_rank=job["lora_rank"],
                                     expected_lora_alpha=job["lora_alpha"], expected_config_sha256=job["config_sha256"])
        if parent["frozen_tensor_sha256"] != frozen_before or parent["model_config_sha256"] != model_config_hash:
            raise ValueError("frozen quantized base differs from parent")
        for row in job["rows"]:
            if (row["adapter_sha256"] != parent["adapter_sha256"] or
                    row["adapter_config_sha256"] != parent["files"]["adapter/adapter_config.json"] or
                    row["model_revision"] != job["model_revision"] or row["base_model"] != job["base_model"]):
                raise ValueError("worker receipt provenance does not match parent")
        step = parent["optimizer_step"]
        restore_training_state(model, optimizer, parent_path, step)
        metrics.update(update_from_rows(model, optimizer, job["rows"], config,
                                        pad_token_id=model.config.pad_token_id or 0))
        metrics.update(optimizer_step_before=step, optimizer_step_after=step + 1,
                       parent_checkpoint_id=parent["checkpoint_id"],
                       parent_optimizer_sha256=parent["files"]["optimizer.pt"],
                       source_agent_record_ids=[row["source_agent_record_id"] for row in job["rows"]],
                       batch_sha256=job["batch_sha256"])
        step += 1
    frozen_after = frozen_tensor_digest(model)
    if frozen_before != frozen_after:
        raise ValueError("frozen quantized tensors changed")
    metrics["frozen_after_sha256"] = frozen_after
    metrics["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    staging = Path(tempfile.mkdtemp(prefix=".worker-", dir=checkpoint_dir.parent))
    try:
        export_attention_adapter(model, staging / "adapter", base_model_name=job["base_model"])
        save_training_state(model, optimizer, staging, step)
        write_json(staging / "metrics.json", metrics)
        manifest = {key: job[key] for key in ("scenario", "base_model", "model_revision", "lora_rank", "lora_alpha",
                    "config", "config_sha256", "parent_checkpoint_id", "training_job_id", "batch_sha256")}
        manifest.update(schema_version=1, checkpoint_id=identity, optimizer_step=step,
                        model_config_sha256=model_config_hash, frozen_tensor_sha256=frozen_after)
        publish_checkpoint(staging, checkpoint_dir, manifest)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return validate_checkpoint(checkpoint_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-file", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = run_job(args.job_file, args.checkpoint_dir)
    print(json.dumps({"checkpoint_id": manifest["checkpoint_id"], "optimizer_step": manifest["optimizer_step"]}), flush=True)


if __name__ == "__main__":
    main()
