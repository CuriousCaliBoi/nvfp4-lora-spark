"""Single-GPU vLLM sampling and observable native LoRA synchronization."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch


def ensure_single_gpu_process_group() -> None:
    if torch.distributed.is_initialized():
        if torch.distributed.get_world_size() != 1:
            raise RuntimeError("this runner supports exactly one GPU process")
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)


def chosen_logprob(token: int, candidates) -> float:
    if candidates is None or token not in candidates:
        raise RuntimeError(f"vLLM omitted the log probability for native token {token}")
    value = float(candidates[token].logprob)
    if not math.isfinite(value):
        raise RuntimeError("vLLM returned a non-finite behavior log probability")
    return value


class Rollout:
    def __init__(self, model_dir: str, *, max_model_len: int, rank: int, seed: int,
                 gpu_memory_utilization: float = 0.23):
        from vllm import LLM

        ensure_single_gpu_process_group()
        self.model_dir = model_dir
        self.engine_options = dict(
            model=model_dir, tensor_parallel_size=1,
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len, max_num_batched_tokens=max_model_len,
            enforce_eager=True, logprobs_mode="processed_logprobs", seed=seed,
            trust_remote_code=False, enable_lora=True, max_loras=1,
            max_cpu_loras=1, max_lora_rank=rank, lora_dtype="bfloat16",
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            moe_backend="marlin", mamba_backend="flashinfer",
            mamba_cache_mode="align", kv_cache_dtype="fp8",
        )
        self.llm = LLM(**self.engine_options)
        self.request = None
        self.load_index = 0

    def load_adapter(self, directory: str | Path) -> dict:
        from vllm.lora.request import LoRARequest

        previous = self.request
        if previous is not None and not self.llm.llm_engine.remove_lora(previous.lora_int_id):
            raise RuntimeError("vLLM failed to remove its previous native adapter")
        self.load_index += 1
        request = LoRARequest(
            lora_name=f"grpo-{self.load_index}", lora_int_id=self.load_index,
            lora_path=str(Path(directory).resolve()), base_model_name=self.model_dir,
        )
        if not self.llm.llm_engine.add_lora(request):
            raise RuntimeError("vLLM failed to install the exported adapter")
        self.request = request
        self.llm.reset_prefix_cache()
        torch.cuda.empty_cache()
        return {"adapter_id": self.load_index, "directory": str(directory)}

    def generate(self, prompts: list[list[int]], *, n: int, max_tokens: int,
                 temperature: float, seed: int) -> list[dict]:
        from vllm import SamplingParams

        params = SamplingParams(n=n, max_tokens=max_tokens, temperature=temperature,
                                top_p=1.0, top_k=-1, min_p=0.0, logprobs=1, seed=seed)
        outputs = self.llm.generate([{"prompt_token_ids": ids} for ids in prompts], params,
                                   lora_request=self.request, use_tqdm=False)
        result = []
        for group, request in enumerate(outputs):
            if list(request.prompt_token_ids) != prompts[group]:
                raise RuntimeError("vLLM prompt IDs differ from the learner's prompt IDs")
            for item in request.outputs:
                tokens = list(item.token_ids)
                if item.logprobs is None or len(tokens) != len(item.logprobs):
                    raise RuntimeError("vLLM returned incomplete native behavior log probabilities")
                result.append(dict(group_id=group, completion_ids=tokens,
                                   behavior_logprobs=[chosen_logprob(t, p) for t, p in zip(tokens, item.logprobs, strict=True)],
                                   completion=item.text, finish_reason=item.finish_reason))
        if len(result) != n * len(prompts):
            raise RuntimeError("vLLM returned an unexpected number of completions")
        return result

    def fixed_sequence_logprobs(self, tokens: list[int], *, seed: int) -> list[float]:
        from vllm import SamplingParams

        # Prompt scoring compares identical tokens even if generation changes.
        self.llm.reset_prefix_cache()
        params = SamplingParams(n=1, max_tokens=1, temperature=1.0, top_p=1.0,
                                top_k=-1, min_p=0.0, prompt_logprobs=1, seed=seed)
        request = self.llm.generate([{"prompt_token_ids": tokens}], params,
                                    lora_request=self.request, use_tqdm=False)[0]
        if request.prompt_logprobs is None or len(request.prompt_logprobs) != len(tokens):
            raise RuntimeError("vLLM omitted fixed-sequence prompt log probabilities")
        return [chosen_logprob(t, p) for t, p in zip(tokens[1:], request.prompt_logprobs[1:], strict=True)]


def reload_proof(before: list[float], after: list[float], reloaded: list[float],
                 baseline_repeat: list[float], tolerance=1e-5) -> dict:
    if not before or not len(before) == len(after) == len(reloaded) == len(baseline_repeat):
        raise ValueError("reload proof requires equally sized, nonempty fixed sequences")
    if not all(math.isfinite(x) for sequence in (before, after, reloaded, baseline_repeat) for x in sequence):
        raise ValueError("reload proof contains non-finite log probabilities")
    change = max(abs(a - b) for a, b in zip(before, after, strict=True))
    repeat = max(abs(a - b) for a, b in zip(after, reloaded, strict=True))
    noise = max(abs(a - b) for a, b in zip(before, baseline_repeat, strict=True))
    threshold = 0.0 if noise == 0 else max(tolerance, 5 * noise)
    return {"max_absolute_update_delta": change, "max_absolute_reload_delta": repeat,
            "baseline_repeat_delta": noise, "update_detection_threshold": threshold,
            "reload_tolerance": tolerance, "policy_changed": change > threshold,
            "reload_consistent": repeat <= tolerance,
            "passed": change > threshold and repeat <= tolerance}
