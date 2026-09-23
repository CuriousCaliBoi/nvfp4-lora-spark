"""CPU-only grading and exact recorded policy-data validation."""

from __future__ import annotations

from decimal import Decimal
import math
import re

GSM8K_SYSTEM_PROMPT = "Solve with a concise calculation. End with a final line exactly in the form: #### <number>."
_NUMBER = r"[-+]?(?:(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?|\.[0-9]+)"
_FINAL = re.compile(r"^[ \t]*####[ \t]*(" + _NUMBER + r")\.?\s*\Z", re.MULTILINE)


def _final_number(text: str) -> Decimal | None:
    match = _FINAL.search(text)
    return Decimal(match.group(1).replace(",", "")) if match else None


def gsm8k_reward(completion: str, answer: str) -> float:
    gold = _final_number(answer)
    return float(gold is not None and _final_number(completion) == gold)


SAMPLING = {
    "temperature": 1.2, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
    "frequency_penalty": 0.0, "presence_penalty": 0.0,
    "repetition_penalty": 1.0, "logit_bias": {},
    "logprobs_mode": "processed_logprobs",
}
DEFAULT_CONFIG = {
    "seed": 42, "temperature": 1.2, "learning_rate": 1e-4,
    "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-8,
    "clip_epsilon": 0.2, "tis_min": 0.1, "tis_max": 10.0,
    "max_grad_norm": 1.0, "microbatch_size": 1, "updates_per_job": 1,
    "objective": "grpo_sample_mean_v1", "dropout": 0, "max_model_len": 1024,
}


def normalized_config(config=None) -> dict:
    overrides = dict(config or {})
    if set(overrides) - set(DEFAULT_CONFIG):
        raise ValueError("unknown training configuration fields")
    value = {**DEFAULT_CONFIG, **overrides}
    for key in ("seed", "microbatch_size", "updates_per_job", "max_model_len"):
        if type(value[key]) is not int or value[key] < (0 if key == "seed" else 1):
            raise ValueError(f"invalid {key}")
    for key in ("temperature", "learning_rate", "eps", "clip_epsilon", "tis_min", "tis_max", "max_grad_norm"):
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError(f"invalid {key}")
    if not 0 < value["clip_epsilon"] < 1 or value["tis_min"] > value["tis_max"]:
        raise ValueError("invalid policy clipping bounds")
    if value["temperature"] != 1.2 or value["dropout"] != 0 or value["updates_per_job"] != 1:
        raise ValueError("unsupported sampling/dropout/update configuration")
    if value["objective"] != DEFAULT_CONFIG["objective"] or value["weight_decay"] != 0:
        raise ValueError("unsupported objective or weight decay")
    if not isinstance(value["betas"], list) or len(value["betas"]) != 2 or any(
        type(v) not in (int, float) or not 0 <= v < 1 for v in value["betas"]
    ):
        raise ValueError("invalid AdamW betas")
    return value


def validate_sampling(value: dict) -> None:
    if not isinstance(value, dict) or set(value) != {*SAMPLING, "seed"}:
        raise ValueError("incomplete effective sampling metadata")
    for key, expected in SAMPLING.items():
        if value[key] != expected or isinstance(value[key], bool):
            raise ValueError(f"unsupported sampling {key}")
    if value["seed"] is not None and (type(value["seed"]) is not int or value["seed"] < 0):
        raise ValueError("invalid sampling seed")


def _tokens(value) -> bool:
    return isinstance(value, list) and bool(value) and all(type(x) is int and x >= 0 for x in value)


def validate_capture(capture: dict, runtime_load_id: str) -> None:
    if not isinstance(capture, dict):
        raise ValueError("missing native policy capture")
    prompt, completion = capture.get("prompt_token_ids"), capture.get("completion_token_ids")
    if not _tokens(prompt) or not _tokens(completion):
        raise ValueError("native prompt and completion token IDs required")
    if capture.get("tokens") != prompt + completion:
        raise ValueError("native concatenated token IDs disagree")
    if (type(capture.get("prompt_length")) is not int or type(capture.get("response_length")) is not int
            or capture["prompt_length"] != len(prompt) or capture["response_length"] != len(completion)):
        raise ValueError("native capture length mismatch")
    if capture.get("loss_mask") != [1] * len(completion) or any(type(value) is not int for value in capture["loss_mask"]):
        raise ValueError("completion-only loss mask required")
    logps = capture.get("rollout_log_probs")
    if not isinstance(logps, list) or len(logps) != len(completion) or any(
        type(x) not in (int, float) or not math.isfinite(x) or x > 1e-5 for x in logps
    ):
        raise ValueError("one finite native log probability per completion token required")
    if type(capture.get("output_index")) is not int or capture["output_index"] != 0 or capture.get("finish_reason") not in ("stop", "length"):
        raise ValueError("unsupported native output index or finish reason")
    if not runtime_load_id or capture.get("runtime_load_id") != runtime_load_id:
        raise ValueError("producing runtime identities disagree")
    for key in ("adapter_sha256", "adapter_config_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", capture.get(key, "")):
            raise ValueError(f"invalid {key}")
    for key in ("base_model", "model_revision"):
        if not isinstance(capture.get(key), str) or not capture[key]:
            raise ValueError(f"missing {key}")
    validate_sampling(capture.get("sampling"))


def validate_rows(rows: list[dict], *, complete=True) -> list[dict]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty recorded rows required")
    seen, sources, reports = set(), set(), set()
    first = rows[0]
    for row in rows:
        for key in ("group_id", "rollout_id", "group_size", "batch_group_count", "dataset_row_id"):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError(f"invalid {key}")
        if row["group_size"] < 2 or row["batch_group_count"] < 1:
            raise ValueError("invalid declared group grid")
        if row["group_id"] >= row["batch_group_count"] or row["rollout_id"] >= row["group_size"]:
            raise ValueError("rollout lies outside declared group grid")
        for key in ("source_agent_record_id", "report_agent_record_id", "cycle_id", "dataset_id", "dataset_revision", "dataset_split", "gold_answer", "response_text"):
            if not isinstance(row.get(key), str) or (key != "response_text" and not row[key]):
                raise ValueError(f"missing {key}")
        if row["dataset_id"] != "openai/gsm8k" or row["dataset_split"] != "train":
            raise ValueError("training requires GSM8K train provenance")
        if _final_number(row["gold_answer"]) is None:
            raise ValueError("gold answer requires a final numeric marker")
        slot = row["group_id"], row["rollout_id"]
        if slot in seen or row["source_agent_record_id"] in sources or row["report_agent_record_id"] in reports:
            raise ValueError("duplicate rollout, source or report")
        seen.add(slot); sources.add(row["source_agent_record_id"]); reports.add(row["report_agent_record_id"])
        capture = {
            "prompt_token_ids": row.get("prompt_ids"), "completion_token_ids": row.get("completion_ids"),
            "tokens": row.get("prompt_ids", []) + row.get("completion_ids", []),
            "prompt_length": len(row.get("prompt_ids", [])), "response_length": len(row.get("completion_ids", [])),
            "loss_mask": [1] * len(row.get("completion_ids", [])), "rollout_log_probs": row.get("behavior_logprobs"),
            "output_index": 0, **{k: row.get(k) for k in ("runtime_load_id", "finish_reason", "adapter_sha256", "adapter_config_sha256", "base_model", "model_revision", "sampling")},
        }
        validate_capture(capture, row.get("runtime_load_id"))
        expected = 0.0 if row["finish_reason"] == "length" else gsm8k_reward(row["response_text"], row["gold_answer"])
        if type(row.get("reward")) not in (int, float) or row["reward"] != expected:
            raise ValueError("reported reward disagrees with strict final-answer verifier")
        for key in ("cycle_id", "group_size", "batch_group_count", "runtime_load_id", "adapter_sha256", "adapter_config_sha256", "base_model", "model_revision", "dataset_id", "dataset_revision"):
            if row[key] != first[key]:
                raise ValueError(f"mixed batch {key}")
    for group in {row["group_id"] for row in rows}:
        members = [row for row in rows if row["group_id"] == group]
        for row in members[1:]:
            for key in ("prompt_ids", "dataset_row_id", "gold_answer"):
                if row[key] != members[0][key]:
                    raise ValueError(f"mixed comparison group {key}")
    if complete and len(rows) != first["group_size"] * first["batch_group_count"]:
        raise ValueError("incomplete declared rollout grid")
    return sorted(rows, key=lambda row: (row["group_id"], row["rollout_id"]))


def advantages(rows: list[dict]) -> tuple[float, ...]:
    result = []
    for row in rows:
        rewards = [member["reward"] for member in rows if member["group_id"] == row["group_id"]]
        mean = math.fsum(rewards) / len(rewards)
        std = math.sqrt(math.fsum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1))
        result.append((row["reward"] - mean) / (std + 1e-6))
    return tuple(result)
