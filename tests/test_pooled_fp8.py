"""Pooled-loader FP8 LoRA wiring (CPU-only).

The pooled loader (`replace_nvfp4_modules_pooled`, used when
`pooled_loader_buffers=True`) used to FAIL LOUD on any FP8 LoRA target: it pooled
only NVFP4 storage and froze every FP8 module. This suite proves the pooled path
now installs a trainable `FP8LoRALinear` over the frozen-FP8 base for FP8 LoRA
targets -- exactly like the non-pooled `replace_nvfp4_modules` path -- while:
  * still freezing FP8 modules that are NOT LoRA targets (dequant -> bf16 nn.Linear),
  * still pooling NVFP4 targets,
  * keeping the trainable params restricted to the LoRA adapters,
  * starting from the standard zero-delta PEFT init (lora_B == 0),
  * and pooling the FP8 base + scale + adapter (no per-FP8-module bf16 shadow).

Everything runs on CPU with tiny synthetic safetensors shards; no GPU, no real
model build (a tiny hand-shaped module matches the nemotron heuristic translator:
safetensors `backbone.X` -> in-memory child with `.layers`).
"""
from __future__ import annotations

import json
import types

import torch
import torch.nn as nn
from safetensors.torch import save_file

from nvfp4_lora.linear import FP8LoRALinear, NVFP4LoRALinear
from nvfp4_lora.loader import replace_nvfp4_modules_pooled

CPU = torch.device("cpu")


class _TinyBackbone(nn.Module):
    """safetensors `backbone.layers.0.attn.{q,k,v,o}_proj` -> same in-memory path.

    `config.model_type=None` forces make_key_translator down the nemotron heuristic
    (single safetensors prefix `backbone`, in-memory child `backbone` owning `.layers`)."""

    def __init__(self, in_f, out_f):
        super().__init__()
        layer = nn.Module()
        layer.attn = nn.Module()
        layer.attn.q_proj = nn.Linear(in_f, out_f, bias=False)   # NVFP4 LoRA target
        layer.attn.v_proj = nn.Linear(in_f, out_f, bias=False)   # FP8 LoRA target
        layer.attn.o_proj = nn.Linear(in_f, out_f, bias=False)   # FP8 frozen (non-target)
        layers = nn.ModuleList([layer])
        self.backbone = nn.Module()
        self.backbone.layers = layers
        self.config = types.SimpleNamespace(model_type=None)


def _nvfp4_storage(out_f, in_f, group_size=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "weight": torch.randint(0, 256, (out_f, in_f // 2), dtype=torch.uint8, generator=g),
        "weight_scale": (torch.rand(out_f, in_f // group_size, generator=g) * 1.5 + 0.5).to(torch.float8_e4m3fn),
        "weight_scale_2": torch.tensor(0.75, dtype=torch.float32),
    }


def _fp8_storage(out_f, in_f, per_channel=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(out_f, in_f, generator=g) * 0.5).to(torch.float8_e4m3fn)
    if per_channel:
        scale = (torch.rand(out_f, generator=g) * 0.75 + 0.25).to(torch.float32)
    else:
        scale = torch.tensor(0.5, dtype=torch.float32)
    return {"weight": w, "weight_scale": scale}


def _write_ckpt(tmp_path, in_f, out_f, fp8_per_channel=False, frozen_fp8_per_channel=False):
    shard = "model-00001-of-00001.safetensors"
    tensors = {}
    for k, v in _nvfp4_storage(out_f, in_f, seed=1).items():
        tensors[f"backbone.layers.0.attn.q_proj.{k}"] = v
    for k, v in _fp8_storage(out_f, in_f, per_channel=fp8_per_channel, seed=2).items():
        tensors[f"backbone.layers.0.attn.v_proj.{k}"] = v
    # o_proj is the FROZEN (non-LoRA-target) FP8 module. `frozen_fp8_per_channel` gives it a
    # per-output-channel (out,) scale -- the shape that trips the loader's frozen-dequant
    # .item() (Puzzle's lm_head + attn q/k/v/o_proj).
    for k, v in _fp8_storage(out_f, in_f, per_channel=frozen_fp8_per_channel, seed=3).items():
        tensors[f"backbone.layers.0.attn.o_proj.{k}"] = v
    save_file(tensors, str(tmp_path / shard))
    index = {"weight_map": {k: shard for k in tensors}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))


def _build(tmp_path, in_f=32, out_f=8, **kw):
    _write_ckpt(tmp_path, in_f, out_f, **kw)
    model = _TinyBackbone(in_f, out_f)
    counts = replace_nvfp4_modules_pooled(
        model, tmp_path, target_lora_suffixes=("q_proj", "v_proj"),
        r=4, lora_alpha=8, device=CPU, dtype=torch.float32,
    )
    return model, counts


def test_pooled_fp8_lora_target_becomes_trainable_fp8_lora_linear(tmp_path):
    model, counts = _build(tmp_path)
    q = model.backbone.layers[0].attn.q_proj
    v = model.backbone.layers[0].attn.v_proj
    o = model.backbone.layers[0].attn.o_proj

    # NVFP4 target pooled as NVFP4LoRALinear; FP8 target as a trainable FP8LoRALinear.
    assert isinstance(q, NVFP4LoRALinear)
    assert isinstance(v, FP8LoRALinear) and v.r == 4
    # Non-target FP8 is frozen: a plain nn.Linear with a non-trainable weight.
    assert isinstance(o, nn.Linear)
    assert o.weight.requires_grad is False

    # The FP8 base stays in fp8 (no bf16 shadow), scale stays fp32, adapter is bf16/fp32.
    assert v.weight_fp8.dtype == torch.float8_e4m3fn
    assert v.weight_scale.dtype == torch.float32
    assert v.lora_A.requires_grad and v.lora_B.requires_grad

    # Coherent counts: 1 NVFP4 LoRA + 1 FP8 LoRA = 2 trainable; 1 frozen FP8; no demotion.
    assert counts["lora"] == 2
    assert counts["lora_fp8"] == 1
    assert counts["frozen_fp8"] == 1
    assert "lora_demoted_fp8" not in counts


def test_pooled_fp8_lora_only_adapter_is_trainable(tmp_path):
    model, _ = _build(tmp_path)
    v = model.backbone.layers[0].attn.v_proj
    trainable = {n for n, p in v.named_parameters() if p.requires_grad}
    assert trainable == {"lora_A", "lora_B"}
    # The fp8 base + scale are buffers, never trained Parameters.
    assert not isinstance(v.weight_fp8, nn.Parameter)


def test_pooled_fp8_lora_starts_at_zero_delta(tmp_path):
    """Standard PEFT init must survive pooling: lora_B == 0 (zero delta at step 0),
    lora_A != 0 (Kaiming), so the first forward equals the frozen FP8 base."""
    model, _ = _build(tmp_path)
    v = model.backbone.layers[0].attn.v_proj
    assert torch.count_nonzero(v.lora_B) == 0
    assert torch.count_nonzero(v.lora_A) > 0

    model.eval()
    x = torch.randn(3, v.in_features)
    base = torch.nn.functional.linear(x, v.weight_fp8.to(torch.float32) * v.weight_scale)
    assert torch.allclose(v(x), base, atol=1e-5)


def test_pooled_fp8_lora_backward_trains_only_adapter(tmp_path):
    model, _ = _build(tmp_path)
    v = model.backbone.layers[0].attn.v_proj
    with torch.no_grad():
        v.lora_B.copy_(torch.randn_like(v.lora_B) * 0.1)  # non-zero so A gets grad
    x = torch.randn(2, v.in_features, requires_grad=True)
    v(x).sum().backward()
    assert v.lora_A.grad is not None and v.lora_A.grad.abs().sum() > 0
    assert v.lora_B.grad is not None and v.lora_B.grad.abs().sum() > 0
    assert x.grad is not None
    # Frozen base accumulates no grad.
    assert v.weight_fp8.grad is None


def test_pooled_fp8_lora_per_channel_scale(tmp_path):
    """A per-output-channel FP8 weight_scale must pool + broadcast correctly:
    FP8LoRALinear normalizes (out,) -> (out,1) over a view of the pooled scale."""
    model, _ = _build(tmp_path, fp8_per_channel=True)
    v = model.backbone.layers[0].attn.v_proj
    assert v.weight_scale.shape == (v.out_features, 1)

    model.eval()
    x = torch.randn(3, v.in_features)
    base = torch.nn.functional.linear(
        x, v.weight_fp8.to(torch.float32) * v.weight_scale  # (out,1) broadcast over in
    )
    assert torch.allclose(v(x), base, atol=1e-5)


def test_pooled_frozen_fp8_per_channel_scale(tmp_path):
    """FROZEN (non-LoRA-target) FP8 with a per-output-channel (out,) weight_scale must
    dequant without the old scalar `.item()` crash (Puzzle lm_head + q/k/v/o_proj) and
    broadcast the per-row scale correctly. Exercises replace_nvfp4_modules_pooled's
    frozen-FP8 branch."""
    model, counts = _build(tmp_path, frozen_fp8_per_channel=True)  # o_proj frozen, per-channel
    o = model.backbone.layers[0].attn.o_proj
    assert isinstance(o, nn.Linear) and o.weight.requires_grad is False
    assert counts["frozen_fp8"] == 1

    # Dequant equals w_fp8 * scale[:, None] (per-row), computed from the same seeded storage.
    st = _fp8_storage(o.out_features, o.in_features, per_channel=True, seed=3)
    expected = st["weight"].to(torch.float32) * st["weight_scale"].reshape(-1, 1)
    assert torch.allclose(o.weight.to(torch.float32), expected.to(o.weight.dtype).to(torch.float32), atol=1e-5)


def test_nonpooled_frozen_fp8_per_channel_scale(tmp_path):
    """Same per-channel frozen-FP8 dequant, but through the NON-pooled
    replace_nvfp4_modules path (the second patched site)."""
    from nvfp4_lora.loader import replace_nvfp4_modules

    in_f, out_f = 32, 8
    _write_ckpt(tmp_path, in_f, out_f, frozen_fp8_per_channel=True)
    model = _TinyBackbone(in_f, out_f)
    counts = replace_nvfp4_modules(
        model, tmp_path, target_lora_suffixes=("q_proj", "v_proj"),
        r=4, lora_alpha=8, device=CPU, dtype=torch.float32,
    )  # o_proj is frozen FP8 with a per-channel scale -> must not raise
    o = model.backbone.layers[0].attn.o_proj
    assert isinstance(o, nn.Linear) and o.weight.requires_grad is False
    st = _fp8_storage(out_f, in_f, per_channel=True, seed=3)
    expected = st["weight"].to(torch.float32) * st["weight_scale"].reshape(-1, 1)
    assert torch.allclose(o.weight.to(torch.float32), expected.to(torch.float32), atol=1e-5)


def test_pooled_fp8_lora_views_share_pool_storage(tmp_path):
    """Memory-efficiency intent: FP8 base + scale + adapter live in pooled flat
    allocations, so the module tensors VIEW the pools rather than owning copies.

    Builds TWO FP8-LoRA targets and asserts both modules' bases (and both adapters)
    point into a SINGLE shared storage per pool -- the defining property of pooling."""
    in_f, out_f, shard = 32, 8, "model-00001-of-00001.safetensors"
    tensors = {}
    for proj in ("q_proj", "k_proj"):  # two FP8 LoRA targets this time
        for k, v in _fp8_storage(out_f, in_f, seed=hash(proj) % 100).items():
            tensors[f"backbone.layers.0.attn.{proj}.{k}"] = v
    save_file(tensors, str(tmp_path / shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: shard for k in tensors}})
    )

    model = _TinyBackbone(in_f, out_f)
    # _TinyBackbone has q_proj/v_proj/o_proj; reuse q_proj and add k_proj.
    model.backbone.layers[0].attn.k_proj = nn.Linear(in_f, out_f, bias=False)
    del model.backbone.layers[0].attn.v_proj
    del model.backbone.layers[0].attn.o_proj
    replace_nvfp4_modules_pooled(
        model, tmp_path, target_lora_suffixes=("q_proj", "k_proj"),
        r=4, lora_alpha=8, device=CPU, dtype=torch.float32,
    )
    q = model.backbone.layers[0].attn.q_proj
    k = model.backbone.layers[0].attn.k_proj
    assert isinstance(q, FP8LoRALinear) and isinstance(k, FP8LoRALinear)

    def storage_id(t):
        return t.untyped_storage().data_ptr()

    # Both fp8 bases share ONE weight pool; both A's and both B's share their pools.
    assert storage_id(q.weight_fp8) == storage_id(k.weight_fp8)
    assert storage_id(q.weight_scale) == storage_id(k.weight_scale)
    assert storage_id(q.lora_A) == storage_id(k.lora_A)
    assert storage_id(q.lora_B) == storage_id(k.lora_B)
    # Distinct slices within the shared storage (not aliasing the same elements).
    assert q.lora_A.data_ptr() != k.lora_A.data_ptr()
