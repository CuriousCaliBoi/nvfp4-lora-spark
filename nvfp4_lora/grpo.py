"""Completion-only GRPO with explicit, detached behavior-policy correction."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from decimal import Decimal

import torch
from torch import Tensor


@dataclass
class Trajectory:
    prompt_ids: list[int]
    completion_ids: list[int]
    behavior_logprobs: list[float]
    completion: str
    reward: float
    group_id: int
    finish_reason: str | None = None


@dataclass
class TokenBatch:
    input_ids: Tensor
    attention_mask: Tensor
    completion_mask: Tensor
    behavior_logprobs: Tensor

    def select(self, start: int, end: int) -> "TokenBatch":
        return TokenBatch(*(x[start:end] for x in (
            self.input_ids, self.attention_mask, self.completion_mask,
            self.behavior_logprobs,
        )))


def collate(trajectories: list[Trajectory], pad_token_id: int, device="cpu") -> TokenBatch:
    if not trajectories or any(not t.prompt_ids for t in trajectories):
        raise ValueError("a nonempty batch and at least one prompt token are required")
    for item in trajectories:
        if len(item.completion_ids) != len(item.behavior_logprobs):
            raise ValueError("one native behavior log probability per completion token is required")
        if not all(math.isfinite(x) for x in item.behavior_logprobs):
            raise ValueError("behavior log probabilities must be finite")
    width = max(len(t.prompt_ids) + len(t.completion_ids) for t in trajectories)
    ids = torch.full((len(trajectories), width), pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    behavior = torch.zeros_like(ids, dtype=torch.float32)
    for row, item in enumerate(trajectories):
        start = len(item.prompt_ids)
        end = start + len(item.completion_ids)
        ids[row, :end] = torch.tensor(item.prompt_ids + item.completion_ids, device=device)
        attention[row, :end] = 1
        mask[row, start:end] = True
        behavior[row, start:end] = torch.tensor(item.behavior_logprobs, device=device)
    return TokenBatch(ids, attention, mask, behavior)


def selected_logprobs(logits: Tensor, ids: Tensor, mask: Tensor, temperature: float) -> Tensor:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("learner temperature must be finite and positive")
    if logits.shape[:2] != ids.shape or ids.shape != mask.shape:
        raise ValueError("logits, tokens and mask must share batch and sequence dimensions")
    if mask[:, 0].any():
        raise ValueError("the first token cannot have a next-token loss")
    # Accumulating log-softmax in FP32 preserves small adapter changes.
    scores = logits[:, :-1].float() / temperature
    selected = scores.gather(-1, ids[:, 1:, None]).squeeze(-1) - scores.logsumexp(-1)
    selected = torch.cat((logits.new_zeros((len(ids), 1), dtype=torch.float32), selected), 1)
    return torch.where(mask, selected, 0.0)


def policy_logprobs(model, batch: TokenBatch, temperature: float) -> Tensor:
    result = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
    return selected_logprobs(result.logits, batch.input_ids, batch.completion_mask, temperature)


def group_advantages(rewards: Tensor, groups: Tensor) -> Tensor:
    if rewards.ndim != 1 or groups.shape != rewards.shape:
        raise ValueError("rewards and groups must have equal one-dimensional shapes")
    result = torch.zeros_like(rewards, dtype=torch.float32)
    for group in groups.unique():
        selected = groups == group
        values = rewards[selected].float()
        if len(values) > 1:
            result[selected] = (values - values.mean()) / (values.std(unbiased=True) + 1e-6)
    return result.detach()


def policy_loss(current: Tensor, old: Tensor, behavior: Tensor, mask: Tensor,
                advantages: Tensor, *, clip_epsilon=0.2, tis_min=0.1, tis_max=10.0) -> Tensor:
    if not (current.shape == old.shape == behavior.shape == mask.shape):
        raise ValueError("all token tensors must have the same shape")
    if advantages.shape != (len(current),):
        raise ValueError("one advantage per trajectory is required")
    if not 0 < clip_epsilon < 1 or not 0 < tis_min <= tis_max:
        raise ValueError("invalid policy clip or importance-weight bounds")
    old = old.detach()
    behavior = behavior.detach()
    log_ratio = torch.where(mask, current.float() - old.float(), 0.0)
    ratio = log_ratio.clamp(-20, 20).exp()
    correction = (old.float() - behavior.float()).clamp(math.log(tis_min), math.log(tis_max)).exp()
    correction = torch.where(mask, correction, 0.0).detach()
    advantage = advantages.detach()[:, None]
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage)
    per_sample = -(surrogate * correction).sum(-1) / mask.sum(-1).clamp_min(1)
    return per_sample.mean()


@torch.no_grad()
def behavior_metrics(old: Tensor, behavior: Tensor, mask: Tensor, tis_min=0.1, tis_max=10.0) -> dict:
    values = (old.float() - behavior.float())[mask]
    if not values.numel():
        return {"completion_tokens": 0}
    return {
        "completion_tokens": values.numel(),
        "old_minus_behavior_logprob_mean": values.mean().item(),
        "behavior_ratio_mean": values.clamp(-80, 80).exp().mean().item(),
        "behavior_ratio_min": values.min().clamp(-80, 80).exp().item(),
        "behavior_ratio_max": values.max().clamp(-80, 80).exp().item(),
        "behavior_correction_clipped_fraction": ((values < math.log(tis_min)) | (values > math.log(tis_max))).float().mean().item(),
    }


_NUMBER = r"[-+]?(?:(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?|\.[0-9]+)"
_FINAL = re.compile(r"^[ \t]*####[ \t]*(" + _NUMBER + r")\.?\s*\Z", re.MULTILINE)


def _final_number(text: str) -> Decimal | None:
    match = _FINAL.search(text)
    return Decimal(match.group(1).replace(",", "")) if match else None


def gsm8k_reward(completion: str, answer: str) -> float:
    gold = _final_number(answer)
    predicted = _final_number(completion)
    return float(gold is not None and predicted == gold)


def prompt_ids(tokenizer, question: str) -> list[int]:
    encoded = tokenizer.apply_chat_template([
        {"role": "system", "content": "Solve with a concise calculation. End with a final line exactly in the form: #### <number>."},
        {"role": "user", "content": question},
    ], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if isinstance(encoded, Tensor):
        encoded = encoded.flatten().tolist()
    return list(encoded)
