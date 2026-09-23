"""Frozen-checkpoint Nemotron learner with attention-only LoRA for policy gradients."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from types import MethodType

import safetensors
from safetensors.torch import save_file
import torch
from torch import nn

from .linear import BF16LoRALinear, FP8LoRALinear, NVFP4LoRALinear
from .loader import assert_no_meta_tensors

ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


class QuantizedNemotronExperts(nn.Module):
    """Non-gated Nemotron experts, preserving the native token routing order."""

    def __init__(self, experts: list[nn.Module], act_fn):
        super().__init__()
        if not experts:
            raise ValueError("Nemotron requires at least one expert")
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        self.act_fn = act_fn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states, dtype=top_k_weights.dtype)
        with torch.no_grad():
            mask = torch.nn.functional.one_hot(top_k_index, self.num_experts).permute(2, 1, 0)
            hits = (mask.sum(dim=(-1, -2)) > 0).nonzero().flatten()
        for expert_idx in hits.tolist():
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            expert = self.experts[expert_idx]
            value = expert.down_proj(self.act_fn(expert.up_proj(hidden_states[token_idx])))
            value = value * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, value.to(final.dtype))
        return final.to(hidden_states.dtype)


class _Checkpoint:
    def __init__(self, model_dir, stack):
        self.root = Path(model_dir)
        self.weight_map = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self.handles = {
            shard: stack.enter_context(safetensors.safe_open(str(self.root / shard), framework="pt", device="cpu"))
            for shard in set(self.weight_map.values())
        }
        self.used = set()

    def read(self, key, *, shape=None, dtype=None):
        if key not in self.weight_map:
            raise ValueError(f"Missing checkpoint tensor: {key}")
        tensor = self.handles[self.weight_map[key]].get_tensor(key)
        if shape is not None and tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"{key}: shape {tuple(tensor.shape)} != {tuple(shape)}")
        if dtype is not None and tensor.dtype != dtype:
            raise ValueError(f"{key}: dtype {tensor.dtype} != {dtype}")
        if tensor.is_floating_point() and not torch.isfinite(tensor.float()).all():
            raise ValueError(f"Non-finite checkpoint tensor: {key}")
        self.used.add(key)
        return tensor


def _load_linear(checkpoint, prefix, in_features, out_features, *, device, dtype, rank=0, alpha=0):
    weight_key = prefix + ".weight"
    scale_key = prefix + ".weight_scale"
    scale2_key = prefix + ".weight_scale_2"
    weight = checkpoint.read(weight_key)
    bias_key = prefix + ".bias"
    bias = checkpoint.read(bias_key, shape=(out_features,)) if bias_key in checkpoint.weight_map else None
    common = dict(in_features=in_features, out_features=out_features, bias=bias,
                  r=rank, lora_alpha=alpha, device=device, dtype=dtype)
    if weight.dtype == torch.uint8:
        if tuple(weight.shape) != (out_features, in_features // 2) or in_features % 16:
            raise ValueError(f"{prefix}: invalid NVFP4 packed weight shape")
        scale = checkpoint.read(scale_key, shape=(out_features, in_features // 16), dtype=torch.float8_e4m3fn)
        scale2 = checkpoint.read(scale2_key, dtype=torch.float32)
        if tuple(scale2.shape) not in ((), (1,)) or (scale2 <= 0).any() or (scale.float() < 0).any():
            raise ValueError(f"{prefix}: invalid NVFP4 scales")
        module = NVFP4LoRALinear(weight_uint8=weight, weight_scale_fp8=scale,
                                weight_scale_2_fp32=scale2, **common)
        module.cache_dequant = False
    elif weight.dtype == torch.float8_e4m3fn:
        if tuple(weight.shape) != (out_features, in_features):
            raise ValueError(f"{prefix}: invalid FP8 weight shape")
        scale = checkpoint.read(scale_key, dtype=torch.float32)
        if tuple(scale.shape) not in ((), (1,), (out_features,), (out_features, 1)) or (scale <= 0).any():
            raise ValueError(f"{prefix}: invalid FP8 scales")
        if scale2_key in checkpoint.weight_map:
            raise ValueError(f"{prefix}: FP8 weight with NVFP4 secondary scale")
        module = FP8LoRALinear(weight_fp8=weight, weight_scale=scale, **common)
    else:
        if tuple(weight.shape) != (out_features, in_features):
            raise ValueError(f"{prefix}: invalid dense weight shape")
        if scale_key in checkpoint.weight_map or scale2_key in checkpoint.weight_map:
            raise ValueError(f"{prefix}: dense weight with quantization scales")
        if rank:
            if weight.dtype != dtype:
                raise ValueError(f"{prefix}: attention weight dtype {weight.dtype} must match learner dtype {dtype}")
            module = BF16LoRALinear(weight=weight, **common)
        else:
            module = nn.Linear(in_features, out_features, bias=bias is not None, device="meta")
            module.weight = nn.Parameter(weight.to(device), requires_grad=False)
            if bias is not None:
                module.bias = nn.Parameter(bias.to(device), requires_grad=False)
    # Inference activation/KV scales are retained for provenance; learner arithmetic
    # intentionally uses higher-precision activations without an inference KV cache.
    for suffix in ("input_scale", "k_scale", "v_scale"):
        key = prefix + "." + suffix
        if key in checkpoint.weight_map:
            module.register_buffer("checkpoint_" + suffix, checkpoint.read(key).to(device))
    return module


def _set_module(model, name, value):
    parent_name, _, attr = name.rpartition(".")
    setattr(model.get_submodule(parent_name) if parent_name else model, attr, value)


def _unfused_mamba_train(self, mode=True):
    # The native fused training kernel reads out_proj.weight directly. The eval
    # branch calls the differentiable packed projection and still supports autograd.
    return nn.Module.train(self, False)


def _checkpoint_name(name):
    name = name.replace(".experts.experts.", ".experts.")
    return "backbone." + name[len("model."):] if name.startswith("model.") else name


def _materialize(model, checkpoint, *, device, dtype, rank, alpha):
    config = model.config
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ == "NemotronHExperts":
            experts = []
            for _ in range(module.num_experts):
                expert = nn.Module()
                expert.up_proj = nn.Linear(module.up_proj.shape[2], module.up_proj.shape[1], bias=False, device="meta")
                expert.down_proj = nn.Linear(module.down_proj.shape[2], module.down_proj.shape[1], bias=False, device="meta")
                experts.append(expert)
            _set_module(model, name, QuantizedNemotronExperts(experts, module.act_fn))

    prefix = "model" if hasattr(model, "model") else "backbone"
    blocks = config.layers_block_type
    expected = {
        f"{prefix}.layers.{i}.mixer.{projection}"
        for i, block in enumerate(blocks) if block in ("attention", "full_attention")
        for projection in ATTENTION_PROJECTIONS
    }
    actual = set()
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            is_target = name in expected
            replacement = _load_linear(checkpoint, _checkpoint_name(name), module.in_features, module.out_features,
                                       device=device, dtype=dtype, rank=rank if is_target else 0,
                                       alpha=alpha if is_target else 0)
            _set_module(model, name, replacement)
            if is_target:
                actual.add(name)
    if not expected or actual != expected:
        raise ValueError(f"Missing attention targets: {sorted(expected - actual)}")

    state = list(model.named_parameters()) + list(model.named_buffers())
    for name, original in state:
        checkpoint_key = _checkpoint_name(name)
        if checkpoint_key in checkpoint.used:
            continue
        if not original.is_meta and checkpoint_key not in checkpoint.weight_map:
            continue
        tensor = checkpoint.read(checkpoint_key, shape=original.shape)
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        tensor = tensor.to(device=device)
        if isinstance(original, nn.Parameter):
            setattr(parent, attr, nn.Parameter(tensor, requires_grad=False))
        else:
            parent._buffers[attr] = tensor

    unused = sorted(key for key in checkpoint.weight_map if key not in checkpoint.used and not key.startswith("mtp."))
    if unused:
        raise ValueError(f"Unloaded checkpoint tensors: {unused[:10]}")
    model._quantized_attention_targets = sorted(expected)
    for module in model.modules():
        if module.__class__.__name__ == "NemotronHMamba2Mixer":
            module.train = MethodType(_unfused_mamba_train, module)
            module.train(False)
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    return model


def build_quantized_learner(model_dir, *, lora_rank=8, lora_alpha=16, device="cuda", dtype=torch.bfloat16):
    """Load ModelOpt Nemotron weights without replacing them with a BF16 checkpoint.

    Requires native Transformers Nemotron-H support. Expert and FP8 bases remain
    packed; only q/k/v/o LoRA parameters are trainable. MTP is not in this graph.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    if lora_rank <= 0 or lora_alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=False, local_files_only=True)
    if config.model_type != "nemotron_h":
        raise ValueError("Quantized RL learner supports only nemotron_h checkpoints")
    config.use_cache = False
    config.attention_dropout = 0.0
    config.hidden_dropout = 0.0
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False, dtype=dtype)
    model.requires_grad_(False)
    with ExitStack() as stack:
        _materialize(model, _Checkpoint(model_dir, stack), device=torch.device(device), dtype=dtype,
                     rank=lora_rank, alpha=lora_alpha)
    model.eval()
    audit = audit_quantized_learner(model)
    if not audit["packed_nvfp4_modules"]:
        raise ValueError("Checkpoint contains no packed NVFP4 weights")
    return model


def _frozen_tensors(model):
    for name, tensor in sorted(list(model.named_parameters()) + list(model.named_buffers())):
        if not tensor.requires_grad:
            yield name, tensor


def frozen_tensor_digest(model):
    """Hash every frozen tensor, including scales, without a full CPU model copy."""
    digest = hashlib.sha256()
    for name, tensor in _frozen_tensors(model):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        flat = tensor.detach().reshape(-1)
        for start in range(0, flat.numel(), 8 * 1024 * 1024):
            chunk = flat[start:start + 8 * 1024 * 1024].contiguous().view(torch.uint8).cpu().numpy()
            digest.update(chunk.tobytes())
    return digest.hexdigest()


def audit_quantized_learner(model):
    """Validate adapter-only optimization and report persistent frozen storage."""
    assert_no_meta_tensors(model)
    targets = getattr(model, "_quantized_attention_targets", None)
    if not targets:
        raise ValueError("Model has no quantized learner target contract")
    expected = {f"{name}.lora_{side}" for name in targets for side in ("A", "B")}
    trainable = {name: tensor for name, tensor in model.named_parameters() if tensor.requires_grad}
    if set(trainable) != expected:
        raise ValueError(f"Unexpected trainable parameters: {sorted(set(trainable) ^ expected)}")
    dtype_bytes = {}
    for _, tensor in _frozen_tensors(model):
        key = str(tensor.dtype)
        dtype_bytes[key] = dtype_bytes.get(key, 0) + tensor.numel() * tensor.element_size()
    cache_bytes = 0
    nvfp4_count = fp8_count = 0
    for module in model.modules():
        if isinstance(module, NVFP4LoRALinear):
            nvfp4_count += 1
            for attr in ("_eval_weight", "_train_weight", "w_bf16_workspace"):
                tensor = getattr(module, attr, None)
                if tensor is not None:
                    cache_bytes += tensor.numel() * tensor.element_size()
            if module.cache_dequant:
                raise ValueError("Persistent NVFP4 dequantization caching must be disabled")
        if isinstance(module, FP8LoRALinear):
            fp8_count += 1
    if cache_bytes:
        raise ValueError(f"Unexpected persistent dequantization storage: {cache_bytes} bytes")
    return {
        "attention_targets": targets, "trainable_names": sorted(trainable),
        "trainable_parameters": sum(t.numel() for t in trainable.values()),
        "frozen_tensor_bytes": sum(dtype_bytes.values()), "frozen_dtype_bytes": dtype_bytes,
        "packed_nvfp4_modules": nvfp4_count, "packed_fp8_modules": fp8_count,
        "dequant_cache_bytes": cache_bytes, "meta_tensors": [],
    }


def export_attention_adapter(model, output_dir, *, base_model_name):
    """Export PEFT LoRA weights using vLLM's Nemotron backbone naming."""
    audit = audit_quantized_learner(model)
    state = {}
    settings = set()
    for name in audit["attention_targets"]:
        module = model.get_submodule(name)
        settings.add((module.r, module.lora_alpha))
        for side in ("A", "B"):
            tensor = getattr(module, f"lora_{side}").detach().cpu().contiguous()
            if not torch.isfinite(tensor.float()).all():
                raise ValueError(f"Non-finite adapter: {name}.lora_{side}")
            state[f"base_model.model.{_checkpoint_name(name)}.lora_{side}.weight"] = tensor
    if len(settings) != 1:
        raise ValueError("All attention adapters must have the same rank and alpha")
    rank, alpha = settings.pop()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output_dir / "adapter_model.safetensors"))
    config = {
        "base_model_name_or_path": base_model_name, "bias": "none", "fan_in_fan_out": False,
        "inference_mode": True, "init_lora_weights": True, "lora_alpha": alpha, "lora_dropout": 0.0,
        "modules_to_save": None, "peft_type": "LORA", "r": rank,
        "target_modules": list(ATTENTION_PROJECTIONS), "task_type": "CAUSAL_LM",
        "use_dora": False, "use_rslora": False,
    }
    (output_dir / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return {
        "directory": str(output_dir), "attention_targets": audit["attention_targets"],
        "tensor_names": sorted(state), "rank": rank, "alpha": alpha,
        "adapter_sha256": hashlib.sha256((output_dir / "adapter_model.safetensors").read_bytes()).hexdigest(),
    }
