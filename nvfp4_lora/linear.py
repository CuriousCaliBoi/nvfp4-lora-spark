"""NVFP4LoRALinear - frozen NVFP4 base weight + trainable bf16 LoRA delta.

- Base weight: NVFP4-quantized, stored as (qdata, scales, per_tensor_scale) - never trained.
- LoRA path: bf16 `lora_A` (r, in) and `lora_B` (out, r), trainable.
- Forward: y = (dequant(W) @ x.T).T + scale * (x @ A.T) @ B.T
  Equivalent: y = x @ dequant(W).T + scale * x @ A.T @ B.T
- Backward through W: standard F.linear backward auto-handles dx = dy @ W; we wrap
  the dequant in a custom autograd.Function so W_bf16 is recomputed in backward
  rather than saved, keeping memory ~weight-storage-only across the autograd graph.
- LoRA gradient: standard, since lora_A and lora_B are normal nn.Parameter.

This is `nn.Module` with a `Linear`-compatible interface, NOT a subclass of nn.Linear
(per the v6.2 PROPOSAL §5.3.1 lesson - nn.Linear assumes `.weight` is a trainable Parameter,
which is wrong here).
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dequant import dequantize_nvfp4_weight


# Process-wide budget for the eval-mode bf16 weight cache. Override via
# NVFP4_EVAL_CACHE_GB for memory-tight runs (e.g. NVFP4-attention models on
# GB10 UMA, where post-load headroom is ~50 GB and a 30 GB cache + eval
# activations can tip the box over).
import os as _os
_EVAL_CACHE_LIMIT_BYTES = int(float(_os.environ.get("NVFP4_EVAL_CACHE_GB", "30")) * 1024**3)
_EVAL_CACHE_BYTES_RESERVED = 0

# Process-wide budget for the OPT-IN train-mode bf16 weight cache (P1-3). Off by default (0):
# training uses the memory-efficient recompute path (_DequantLinear re-dequantizes every step).
# When enabled via --train-dequant-cache-gb / NVFP4_TRAIN_CACHE_GB (small/mid models with UMA
# headroom), the dequantized base weight is materialized ONCE per module and reused across steps
# with a plain F.linear -> near-bf16 step time. The base weight is frozen, so autograd still gives
# dx = dy @ W and no gradient to W: numerically identical to the recompute path, just faster. Once
# the budget is exhausted, remaining modules fall back to recompute (so a 120B stays memory-safe).
_TRAIN_CACHE_LIMIT_BYTES = int(float(_os.environ.get("NVFP4_TRAIN_CACHE_GB", "0")) * 1024**3)
_TRAIN_CACHE_BYTES_RESERVED = 0


def set_train_dequant_cache_gb(gb: float) -> None:
    """Set the process-wide train-time bf16 dequant cache budget in GB (0 disables it).

    Call once before the training loop. Resets the running reservation so a re-entry starts fresh.
    """
    global _TRAIN_CACHE_LIMIT_BYTES, _TRAIN_CACHE_BYTES_RESERVED
    _TRAIN_CACHE_LIMIT_BYTES = int(float(gb) * 1024**3)
    _TRAIN_CACHE_BYTES_RESERVED = 0


class _DequantLinear(torch.autograd.Function):
    """Custom autograd that dequants on every forward and recomputes on every backward.

    Does NOT save the dequantized bf16 weight across the graph. Saves only the small
    quantized buffers + scales (typically ~1/4 the size of bf16).

    The frozen base weight produces no gradient; only `x` gets a grad via `dx = dy @ W`.
    """

    @staticmethod
    def forward(ctx, x, weight_uint8, weight_scale_fp8, weight_scale_2_fp32, group_size: int, w_bf16_workspace, format: str):
        ctx.save_for_backward(weight_uint8, weight_scale_fp8, weight_scale_2_fp32)
        ctx.group_size = group_size
        ctx.w_bf16_workspace = None if w_bf16_workspace is None else w_bf16_workspace.detach()
        ctx.format = format
        W_bf16 = dequantize_nvfp4_weight(
            weight_uint8, weight_scale_fp8, weight_scale_2_fp32,
            group_size=group_size, out_dtype=x.dtype, out=ctx.w_bf16_workspace,
            format=format,
        )
        # F.linear computes x @ W.T (no bias handled here; caller adds it)
        return F.linear(x, W_bf16, bias=None)

    @staticmethod
    def backward(ctx, grad_output):
        weight_uint8, weight_scale_fp8, weight_scale_2_fp32 = ctx.saved_tensors
        group_size = ctx.group_size

        grad_x = None
        if ctx.needs_input_grad[0]:
            # Recompute W_bf16 inside backward - never saved
            W_bf16 = dequantize_nvfp4_weight(
                weight_uint8, weight_scale_fp8, weight_scale_2_fp32,
                group_size=group_size, out_dtype=grad_output.dtype, out=ctx.w_bf16_workspace,
                format=ctx.format,
            )
            # dx = dy @ W (since y = x @ W.T, dy/dx = W)
            grad_x = grad_output @ W_bf16

        # No grad for the frozen base weight, scales, or workspace
        return grad_x, None, None, None, None, None, None


class NVFP4LoRALinear(nn.Module):
    """nn.Linear-compatible interface with frozen NVFP4 base + trainable bf16 LoRA delta.

    Args:
        in_features, out_features: standard Linear dims (in_features is the unpacked dim).
        weight_uint8: (out, in/2) uint8 - the on-disk packed NVFP4 weight.
        weight_scale_fp8: (out, in/group_size) float8_e4m3fn.
        weight_scale_2_fp32: scalar float32.
        group_size: NVFP4 group size, default 16.
        bias: optional bias tensor, frozen by default.
        r: LoRA rank; if 0, LoRA path is disabled (frozen linear only).
        lora_alpha: LoRA scaling factor; effective scale = lora_alpha / r.
        lora_dropout: dropout applied to the LoRA path input (not the base path).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_uint8: torch.Tensor,
        weight_scale_fp8: torch.Tensor,
        weight_scale_2_fp32: torch.Tensor,
        group_size: int = 16,
        bias: Optional[torch.Tensor] = None,
        r: int = 0,
        lora_alpha: int = 0,
        lora_dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        copy_base_tensors: bool = True,
        lora_A_tensor: Optional[torch.Tensor] = None,
        lora_B_tensor: Optional[torch.Tensor] = None,
        format: str = "modelopt",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.nvfp4_format = format
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_scale = (lora_alpha / r) if r > 0 else 0.0
        self.dtype = dtype
        self.cache_dequant = True
        self.w_bf16_workspace: Optional[torch.Tensor] = None

        # Frozen NVFP4 base (register as buffers, not Parameters - never trained, never saved by optimizer)
        device = device or weight_uint8.device
        if copy_base_tensors:
            weight_uint8 = weight_uint8.to(device=device).contiguous()
            weight_scale_fp8 = weight_scale_fp8.to(device=device).contiguous()
            weight_scale_2_fp32 = weight_scale_2_fp32.to(device=device).contiguous()
        self.register_buffer("weight_uint8", weight_uint8)
        self.register_buffer("weight_scale_fp8", weight_scale_fp8)
        self.register_buffer("weight_scale_2_fp32", weight_scale_2_fp32)

        if bias is not None:
            self.bias = nn.Parameter(bias.to(device=device, dtype=dtype), requires_grad=False)
        else:
            self.bias = None

        # LoRA params
        if r > 0:
            _lora_A_supplied = lora_A_tensor is not None
            _lora_B_supplied = lora_B_tensor is not None
            if lora_A_tensor is None:
                lora_A_tensor = torch.empty(r, in_features, device=device, dtype=dtype)
            if lora_B_tensor is None:
                lora_B_tensor = torch.zeros(out_features, r, device=device, dtype=dtype)
            self.lora_A = nn.Parameter(lora_A_tensor)
            self.lora_B = nn.Parameter(lora_B_tensor)
            # Kaiming init for A; B starts at zero so LoRA delta starts at zero (standard PEFT init)
            if not _lora_A_supplied:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            with torch.no_grad():
                if not _lora_B_supplied:
                    self.lora_B.zero_()
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        else:
            self.lora_A = None
            self.lora_B = None
            self.lora_dropout = nn.Identity()

        # Eval-mode bf16 weight cache (populated lazily on first eval forward; cleared on train())
        # Materializes the dequantized base weight once for fast `F.linear` reuse. On Super-120B,
        # caching every module would silently build a huge bf16 shadow, so allocation is capped
        # per process and skipped once the heuristic budget is exhausted.
        self._eval_weight: Optional[torch.Tensor] = None
        self._eval_weight_bytes = 0
        self._eval_cache_warned = False
        # Opt-in train-mode bf16 weight cache (P1-3); persists across steps, budget-capped.
        self._train_weight: Optional[torch.Tensor] = None
        self._train_weight_bytes = 0
        self._train_cache_warned = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base path: training uses custom autograd (dequant recomputed in backward, no bf16 shadow saved);
        # eval uses a lazily-materialized bf16 weight cache for fast `F.linear`.
        if not self.cache_dequant:
            # Policy scoring also needs recomputation in eval mode: retaining a dense
            # weight here would accumulate a second base model across scoring passes.
            y = _DequantLinear.apply(
                x, self.weight_uint8, self.weight_scale_fp8, self.weight_scale_2_fp32,
                self.group_size, None, self.nvfp4_format,
            )
        elif self.training:
            cached = self._train_cached_weight(x.dtype)
            if cached is not None:
                # Frozen resident bf16 weight: autograd gives dx = dy @ W and no grad to W,
                # identical to _DequantLinear but without re-dequantizing each step.
                y = F.linear(x, cached, bias=None)
            else:
                if self.w_bf16_workspace is None:
                    raise RuntimeError(
                        "NVFP4LoRALinear.w_bf16_workspace is not set; "
                        "load the module through the NVFP4 loader so the dequant workspace pool is assigned."
                    )
                y = _DequantLinear.apply(
                    x, self.weight_uint8, self.weight_scale_fp8, self.weight_scale_2_fp32,
                    self.group_size, self.w_bf16_workspace, self.nvfp4_format,
                )
        else:
            if self._train_weight is not None and self._train_weight.dtype == x.dtype:
                # Reuse the train cache if it is resident, so eval during a train loop does not
                # build a second bf16 shadow of the same weight.
                base_weight = self._train_weight
            elif self._eval_weight is None or self._eval_weight.dtype != x.dtype:
                global _EVAL_CACHE_BYTES_RESERVED
                if self._eval_weight is not None:
                    _EVAL_CACHE_BYTES_RESERVED = max(0, _EVAL_CACHE_BYTES_RESERVED - self._eval_weight_bytes)
                    self._eval_weight = None
                    self._eval_weight_bytes = 0

                est_cache_bytes = self.out_features * self.in_features * 2
                if _EVAL_CACHE_BYTES_RESERVED + est_cache_bytes <= _EVAL_CACHE_LIMIT_BYTES:
                    self._eval_weight = dequantize_nvfp4_weight(
                        self.weight_uint8, self.weight_scale_fp8, self.weight_scale_2_fp32,
                        group_size=self.group_size, out_dtype=x.dtype,
                        format=self.nvfp4_format,
                    )
                    self._eval_weight_bytes = est_cache_bytes
                    _EVAL_CACHE_BYTES_RESERVED += est_cache_bytes
                    base_weight = self._eval_weight
                else:
                    if not self._eval_cache_warned:
                        warnings.warn(
                            "Skipping NVFP4LoRALinear eval bf16 cache because the estimated "
                            "process-wide cache would exceed 30 GB; recomputing this module "
                            "on each eval forward instead.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        self._eval_cache_warned = True
                    base_weight = dequantize_nvfp4_weight(
                        self.weight_uint8, self.weight_scale_fp8, self.weight_scale_2_fp32,
                        group_size=self.group_size, out_dtype=x.dtype,
                        format=self.nvfp4_format,
                    )
            else:
                base_weight = self._eval_weight
            y = F.linear(x, base_weight, bias=None)
        # LoRA delta
        if self.r > 0:
            lora_out = F.linear(self.lora_dropout(x), self.lora_A)  # (..., r)
            lora_out = F.linear(lora_out, self.lora_B)              # (..., out_features)
            y = y + self.lora_scale * lora_out
        if self.bias is not None:
            y = y + self.bias
        return y

    def train(self, mode: bool = True):
        # Clear the eval-mode bf16 cache when switching back to training so we don't leak the shadow.
        if mode:
            global _EVAL_CACHE_BYTES_RESERVED
            if self._eval_weight is not None:
                _EVAL_CACHE_BYTES_RESERVED = max(0, _EVAL_CACHE_BYTES_RESERVED - self._eval_weight_bytes)
            self._eval_weight = None
            self._eval_weight_bytes = 0
        return super().train(mode)

    def _train_cached_weight(self, dtype) -> Optional[torch.Tensor]:
        """Resident bf16 base weight for the training forward, or None to use recompute.

        Populated lazily under the process-wide budget (_TRAIN_CACHE_LIMIT_BYTES); returns None
        once the budget is exhausted so the remaining modules fall back to the memory-efficient
        recompute path. The cached weight is frozen (requires_grad_(False)); F.linear(x, w) then
        yields the same forward and the same dx = dy @ w as _DequantLinear, with no grad to w.
        """
        if _TRAIN_CACHE_LIMIT_BYTES <= 0:
            return None
        if self._train_weight is not None and self._train_weight.dtype == dtype:
            return self._train_weight
        global _TRAIN_CACHE_BYTES_RESERVED
        if self._train_weight is not None:  # dtype changed -> release and re-reserve
            _TRAIN_CACHE_BYTES_RESERVED = max(0, _TRAIN_CACHE_BYTES_RESERVED - self._train_weight_bytes)
            self._train_weight = None
            self._train_weight_bytes = 0
        est = self.out_features * self.in_features * 2
        if _TRAIN_CACHE_BYTES_RESERVED + est <= _TRAIN_CACHE_LIMIT_BYTES:
            w = dequantize_nvfp4_weight(
                self.weight_uint8, self.weight_scale_fp8, self.weight_scale_2_fp32,
                group_size=self.group_size, out_dtype=dtype, format=self.nvfp4_format,
            ).requires_grad_(False)
            self._train_weight = w
            self._train_weight_bytes = est
            _TRAIN_CACHE_BYTES_RESERVED += est
            return w
        if not self._train_cache_warned:
            warnings.warn(
                "train dequant cache budget exhausted; this module recomputes its dequant each "
                "step. Raise --train-dequant-cache-gb, or accept the slower recompute path.",
                RuntimeWarning, stacklevel=2,
            )
            self._train_cache_warned = True
        return None

    @property
    def weight(self) -> torch.Tensor:
        """Compatibility shim for code that queries `.weight.dtype` / `.shape` / `.device`.

        Returns a META tensor (zero memory) with the correct shape and dtype. Any code
        that tries to USE this for compute will fail loudly - which is correct, since
        the actual NVFP4 storage lives in `self.weight_uint8` + scales, dequanted lazily
        in forward via `_DequantLinear`.

        Specifically relevant for Nemotron-3's modeling_nemotron_h.py:855 which does
        `expert.down_proj.weight.dtype` for autocast-dtype detection.
        """
        # Meta tensor: correct metadata, zero memory
        return torch.empty(
            self.out_features, self.in_features,
            dtype=self.dtype, device="meta",
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, lora_alpha={self.lora_alpha}, group_size={self.group_size}, "
            f"bias={self.bias is not None}"
        )

    @classmethod
    def from_safetensors_record(
        cls,
        record: dict[str, torch.Tensor],
        prefix: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 0,
        lora_dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "NVFP4LoRALinear":
        """Build from a dict of tensors loaded from a safetensors shard.

        Expected keys (relative to `prefix`):
            {prefix}.weight (uint8)
            {prefix}.weight_scale (float8_e4m3fn)
            {prefix}.weight_scale_2 (float32 scalar)
            {prefix}.bias (optional)
        """
        bias = record.get(f"{prefix}.bias")
        return cls(
            in_features=in_features,
            out_features=out_features,
            weight_uint8=record[f"{prefix}.weight"],
            weight_scale_fp8=record[f"{prefix}.weight_scale"],
            weight_scale_2_fp32=record[f"{prefix}.weight_scale_2"],
            group_size=16,
            bias=bias,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            device=device,
            dtype=dtype,
        )


class _FP8DequantLinear(torch.autograd.Function):
    """Frozen FP8 (e4m3) base: dequant on forward, recompute on backward (no bf16 shadow).

    FP8 dequant is a cheap cast+scale (no nibble unpack / group scales), so unlike the
    NVFP4 path there is no eval-mode bf16 cache -- the weight is recomputed each forward.
    The frozen base produces no gradient; only `x` gets dx = dy @ W.
    """

    @staticmethod
    def forward(ctx, x, weight_fp8, weight_scale):
        ctx.save_for_backward(weight_fp8, weight_scale)
        W = _decode_fp8(weight_fp8, weight_scale, x.dtype)
        return F.linear(x, W, bias=None)

    @staticmethod
    def backward(ctx, grad_output):
        weight_fp8, weight_scale = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            W = _decode_fp8(weight_fp8, weight_scale, grad_output.dtype)
            grad_x = grad_output @ W
        return grad_x, None, None


def _decode_fp8(weight_fp8, weight_scale, dtype):
    # Rounding the stored FP32 scale to BF16 before multiplication changes the
    # reconstructed base; cast only after applying the original scale.
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    return (weight_fp8.to(compute_dtype) * weight_scale.to(compute_dtype)).to(dtype)


class FP8LoRALinear(nn.Module):
    """Frozen FP8-per-tensor base weight + trainable bf16 LoRA delta.

    The FP8 sibling of NVFP4LoRALinear, for FP8-quantized targets such as the canonical
    3.6's attention (q/k/v/o, GDN in/out proj: `.weight` fp8_e4m3 + `.weight_scale`). The
    base stays frozen and is dequantized `W = weight.to(bf16) * weight_scale` on each
    forward (recomputed in backward, no bf16 shadow); the only trainable params are the
    bf16 LoRA A/B. This is what lets a NATIVE run adapt FP8 attention instead of freezing
    it (MEASURED 2026-06-22: attention LoRA is ~61% of the ICH quality gain), without ever
    re-quantizing the base.

    `weight_scale` may be a per-tensor scalar or a per-output-channel vector; it is
    normalized to broadcast over the input dim. nn.Linear-compatible interface, NOT a
    subclass (`.weight` is not a trainable Parameter here) -- same rationale as
    NVFP4LoRALinear.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_fp8: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        r: int = 0,
        lora_alpha: int = 0,
        lora_dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        copy_base_tensors: bool = True,
        lora_A_tensor: Optional[torch.Tensor] = None,
        lora_B_tensor: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_scale = (lora_alpha / r) if r > 0 else 0.0
        self.dtype = dtype

        device = device or weight_fp8.device
        if copy_base_tensors:
            weight_fp8 = weight_fp8.to(device=device).contiguous()
            weight_scale = weight_scale.to(device=device)
        # Normalize the scale to broadcast over the input dim: a per-output-channel
        # (out,) vector becomes (out, 1); a scalar / (1,) broadcasts as-is.
        weight_scale = weight_scale.to(torch.float32)
        if weight_scale.ndim == 1 and weight_scale.shape[0] == out_features:
            weight_scale = weight_scale.reshape(out_features, 1)
        self.register_buffer("weight_fp8", weight_fp8)
        self.register_buffer("weight_scale", weight_scale)

        if bias is not None:
            self.bias = nn.Parameter(bias.to(device=device, dtype=dtype), requires_grad=False)
        else:
            self.bias = None

        if r > 0:
            _A_supplied = lora_A_tensor is not None
            _B_supplied = lora_B_tensor is not None
            if lora_A_tensor is None:
                lora_A_tensor = torch.empty(r, in_features, device=device, dtype=dtype)
            if lora_B_tensor is None:
                lora_B_tensor = torch.zeros(out_features, r, device=device, dtype=dtype)
            self.lora_A = nn.Parameter(lora_A_tensor)
            self.lora_B = nn.Parameter(lora_B_tensor)
            if not _A_supplied:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            if not _B_supplied:
                with torch.no_grad():
                    self.lora_B.zero_()
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        else:
            self.lora_A = None
            self.lora_B = None
            self.lora_dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _FP8DequantLinear.apply(x, self.weight_fp8, self.weight_scale)
        if self.r > 0:
            lora_out = F.linear(self.lora_dropout(x), self.lora_A)
            lora_out = F.linear(lora_out, self.lora_B)
            y = y + self.lora_scale * lora_out
        if self.bias is not None:
            y = y + self.bias
        return y

    @property
    def weight(self) -> torch.Tensor:
        """Meta tensor (zero memory), correct shape/dtype; see NVFP4LoRALinear.weight."""
        return torch.empty(self.out_features, self.in_features, dtype=self.dtype, device="meta")

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, lora_alpha={self.lora_alpha}, fp8_base=True, "
            f"bias={self.bias is not None}"
        )

    @classmethod
    def from_safetensors_record(
        cls,
        record: dict,
        prefix: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 0,
        lora_dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "FP8LoRALinear":
        """Build from safetensors tensors: {prefix}.weight (fp8_e4m3) + {prefix}.weight_scale
        (+ optional {prefix}.bias). `.input_scale` is an inference activation scale, unused
        for weight dequant."""
        return cls(
            in_features=in_features,
            out_features=out_features,
            weight_fp8=record[f"{prefix}.weight"],
            weight_scale=record[f"{prefix}.weight_scale"],
            bias=record.get(f"{prefix}.bias"),
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            device=device,
            dtype=dtype,
        )


class BF16LoRALinear(nn.Module):
    """Frozen BF16 base weight + trainable bf16 LoRA delta.

    Completes the set (NVFP4 / FP8 / BF16 LoRALinear): a frozen base of ANY storage plus a
    bf16 LoRA, all trainable through the same NATIVE path. For genuinely-BF16 target modules
    (e.g. attention a mixed-precision quantizer left in bf16) this lets them CO-TRAIN with
    NVFP4/FP8 targets in one native adapter, instead of being dropped (a native run cannot
    host PEFT). No dequant -- the base is already bf16 -- so forward is a plain frozen
    F.linear + the LoRA path. The frozen base is a buffer (never an optimizer Parameter).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        r: int = 0,
        lora_alpha: int = 0,
        lora_dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
        copy_base_tensors: bool = True,
        lora_A_tensor: Optional[torch.Tensor] = None,
        lora_B_tensor: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_scale = (lora_alpha / r) if r > 0 else 0.0
        self.dtype = dtype
        device = device or weight.device
        w = weight.to(device=device, dtype=dtype)
        if copy_base_tensors:
            w = w.contiguous()
        self.register_buffer("weight", w)  # frozen base: a buffer, never a trained Parameter
        if bias is not None:
            self.bias = nn.Parameter(bias.to(device=device, dtype=dtype), requires_grad=False)
        else:
            self.bias = None
        if r > 0:
            _A = lora_A_tensor is not None
            _B = lora_B_tensor is not None
            if lora_A_tensor is None:
                lora_A_tensor = torch.empty(r, in_features, device=device, dtype=dtype)
            if lora_B_tensor is None:
                lora_B_tensor = torch.zeros(out_features, r, device=device, dtype=dtype)
            self.lora_A = nn.Parameter(lora_A_tensor)
            self.lora_B = nn.Parameter(lora_B_tensor)
            if not _A:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            if not _B:
                with torch.no_grad():
                    self.lora_B.zero_()
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        else:
            self.lora_A = None
            self.lora_B = None
            self.lora_dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, bias=None)
        if self.r > 0:
            lora_out = F.linear(self.lora_dropout(x), self.lora_A)
            lora_out = F.linear(lora_out, self.lora_B)
            y = y + self.lora_scale * lora_out
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, lora_alpha={self.lora_alpha}, bf16_base=True, "
            f"bias={self.bias is not None}"
        )
