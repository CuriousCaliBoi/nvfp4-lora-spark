"""The policy learner must preserve frozen quantization through forward and backward."""
from contextlib import ExitStack
import json
from types import SimpleNamespace

import pytest
from safetensors.torch import load_file, save_file
import torch
from torch import nn
from torch.nn import functional as F

from nvfp4_lora.dequant import dequantize_nvfp4_weight
from nvfp4_lora.learner import (
    QuantizedNemotronExperts, _Checkpoint, _load_linear, _materialize,
    audit_quantized_learner, build_quantized_learner, export_attention_adapter, frozen_tensor_digest,
)
from nvfp4_lora.linear import BF16LoRALinear, FP8LoRALinear


def _checkpoint(tmp_path, tensors):
    save_file(tensors, str(tmp_path / "weights.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {key: "weights.safetensors" for key in tensors}
    }))


def _record(prefix, out_features=32, in_features=16):
    return {
        prefix + ".weight": torch.randint(0, 256, (out_features, in_features // 2), dtype=torch.uint8),
        prefix + ".weight_scale": torch.full((out_features, in_features // 16), 0.25).to(torch.float8_e4m3fn),
        prefix + ".weight_scale_2": torch.tensor(0.5),
    }


def test_experts_match_native_transformers_routing_and_input_gradients(tmp_path):
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHExperts

    torch.manual_seed(7)
    records = {}
    for i in range(4):
        records.update(_record(f"experts.{i}.up_proj"))
        records.update(_record(f"experts.{i}.down_proj", 16, 32))
    _checkpoint(tmp_path, records)
    with ExitStack() as stack:
        checkpoint = _Checkpoint(tmp_path, stack)
        experts = []
        for i in range(4):
            expert = nn.Module()
            expert.up_proj = _load_linear(checkpoint, f"experts.{i}.up_proj", 16, 32, device="cpu", dtype=torch.float32)
            expert.down_proj = _load_linear(checkpoint, f"experts.{i}.down_proj", 32, 16, device="cpu", dtype=torch.float32)
            experts.append(expert)
    config = SimpleNamespace(n_routed_experts=4, hidden_size=16, moe_intermediate_size=32,
                             moe_latent_size=None, mlp_hidden_act="relu2", _experts_implementation="eager")
    reference = NemotronHExperts(config)
    learner = QuantizedNemotronExperts(experts, reference.act_fn)
    with torch.no_grad():
        for i in range(4):
            for projection in ("up_proj", "down_proj"):
                prefix = f"experts.{i}.{projection}"
                getattr(reference, projection)[i].copy_(dequantize_nvfp4_weight(
                    records[prefix + ".weight"], records[prefix + ".weight_scale"],
                    records[prefix + ".weight_scale_2"], out_dtype=torch.float32))
    x = torch.randn(5, 16, requires_grad=True)
    x_reference = x.detach().clone().requires_grad_()
    routing = torch.tensor([[0, 2], [1, 0], [2, 1], [0, 1], [2, 0]])
    weights = torch.tensor([[0.1, 0.9], [0.7, 0.3], [0.2, 0.8], [0.4, 0.6], [0.6, 0.4]], requires_grad=True)
    weights_reference = weights.detach().clone().requires_grad_()
    for training in (False, True):
        learner.train(training)
        x.grad = x_reference.grad = weights.grad = weights_reference.grad = None
        result = learner(x, routing, weights)
        expected = reference(x_reference, routing, weights_reference)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
        probe = torch.randn_like(result)
        (result * probe).sum().backward()
        (expected * probe).sum().backward()
        torch.testing.assert_close(x.grad, x_reference.grad, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(weights.grad, weights_reference.grad, atol=1e-6, rtol=1e-6)
    for module in learner.modules():
        assert getattr(module, "_eval_weight", None) is None
        assert getattr(module, "_train_weight", None) is None
        assert getattr(module, "w_bf16_workspace", None) is None


@pytest.mark.parametrize("mutation", ["missing_weight", "missing_scale", "bad_shape", "nan_scale", "wrong_dtype", "negative_scale"])
def test_invalid_nvfp4_records_fail(tmp_path, mutation):
    tensors = _record("layer")
    if mutation == "missing_weight":
        del tensors["layer.weight"]
    elif mutation == "missing_scale":
        del tensors["layer.weight_scale_2"]
    elif mutation == "bad_shape":
        tensors["layer.weight"] = tensors["layer.weight"][:, :-1].contiguous()
    elif mutation == "nan_scale":
        tensors["layer.weight_scale_2"] = torch.tensor(float("nan"))
    elif mutation == "negative_scale":
        tensors["layer.weight_scale_2"] = torch.tensor(-1.0)
    elif mutation == "wrong_dtype":
        tensors["layer.weight_scale"] = tensors["layer.weight_scale"].float()
    _checkpoint(tmp_path, tensors)
    with ExitStack() as stack, pytest.raises(ValueError):
        _load_linear(_Checkpoint(tmp_path, stack), "layer", 16, 32, device="cpu", dtype=torch.float32)


def _attention_model():
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].mixer = nn.Module()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(model.model.layers[0].mixer, name, BF16LoRALinear(
            16, 16, torch.randn(16, 16), r=2, lora_alpha=4, dtype=torch.float32))
    model._quantized_attention_targets = [f"model.layers.0.mixer.{name}" for name in ("q_proj", "k_proj", "v_proj", "o_proj")]
    return model


def test_frozen_fp8_backward_storage_digest_and_adapter_export(tmp_path):
    model = _attention_model()
    model.fp8 = FP8LoRALinear(16, 16, torch.randn(16, 16).to(torch.float8_e4m3fn),
                              torch.tensor(0.25), r=0, dtype=torch.float32)
    model.register_buffer("A_log", torch.randn(16, dtype=torch.float32))
    before = audit_quantized_learner(model)
    before_hash = frozen_tensor_digest(model)
    original = export_attention_adapter(model, tmp_path / "original", base_model_name="test/nvfp4")
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    model.eval()
    x = torch.randn(3, 16)
    target = model.model.layers[0].mixer.q_proj
    model.fp8(target(x)).square().mean().backward()
    assert target.lora_B.grad.norm() > 0
    assert not any(p.requires_grad for p in model.fp8.parameters())
    optimizer.step()
    assert frozen_tensor_digest(model) == before_hash
    assert audit_quantized_learner(model) == before
    updated = export_attention_adapter(model, tmp_path / "updated", base_model_name="test/nvfp4")
    assert updated["adapter_sha256"] != original["adapter_sha256"]
    state = load_file(str(tmp_path / "updated" / "adapter_model.safetensors"))
    assert len(state) == 8
    key = "base_model.model.backbone.layers.0.mixer.q_proj."
    delta = F.linear(F.linear(x, state[key + "lora_A.weight"]), state[key + "lora_B.weight"]) * 2
    torch.testing.assert_close(target(x), F.linear(x, target.weight) + delta)
    model.fp8.weight_scale.add_(0.1)
    assert frozen_tensor_digest(model) != before_hash


def test_audit_rejects_additional_trainables():
    model = _attention_model()
    model.extra = nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="Unexpected trainable"):
        audit_quantized_learner(model)


def test_fp8_scale_is_applied_before_bf16_rounding():
    weight = torch.tensor([[1, 1.125, 1.25, 1.5, 1.75, 2, 3, 5, 7]]).to(torch.float8_e4m3fn)
    scale = torch.tensor(0.03137, dtype=torch.float32)
    module = FP8LoRALinear(9, 1, weight, scale, dtype=torch.bfloat16)
    decoded = (weight.float() * scale).to(torch.bfloat16)
    premature_rounding = weight.to(torch.bfloat16) * scale.to(torch.bfloat16)
    assert not torch.equal(decoded, premature_rounding)
    x = torch.eye(9, dtype=torch.bfloat16, requires_grad=True)
    result = module(x)
    torch.testing.assert_close(result, F.linear(x, decoded), atol=0, rtol=0)
    result.sum().backward()
    torch.testing.assert_close(x.grad, decoded.expand(9, -1), atol=0, rtol=0)


def test_materialize_keeps_fp32_state_and_checks_missing_targets(tmp_path):
    model = _attention_model()
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(model.model.layers[0].mixer, projection, nn.Linear(16, 16, bias=False, device="meta"))
    model.model.layers[0].mixer.register_parameter("A_log", nn.Parameter(torch.empty(16, device="meta")))
    model.config = SimpleNamespace(layers_block_type=["attention"])
    tensors = {f"backbone.layers.0.mixer.{name}.weight": torch.randn(16, 16, dtype=torch.bfloat16)
               for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
    tensors["backbone.layers.0.mixer.A_log"] = torch.randn(16, dtype=torch.float32)
    _checkpoint(tmp_path, tensors)
    with ExitStack() as stack:
        _materialize(model, _Checkpoint(tmp_path, stack), device="cpu", dtype=torch.bfloat16, rank=2, alpha=4)
    assert model.model.layers[0].mixer.A_log.dtype == torch.float32
    torch.testing.assert_close(model.model.layers[0].mixer.A_log, tensors["backbone.layers.0.mixer.A_log"])
    assert len(audit_quantized_learner(model)["attention_targets"]) == 4


def test_small_native_model_load_forward_checkpointed_backward(tmp_path):
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM

    config = NemotronHConfig(
        vocab_size=32, hidden_size=32, num_hidden_layers=3,
        layers_block_type=["attention", "mamba", "moe"],
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        mamba_num_heads=2, mamba_head_dim=16, expand=1, n_groups=1,
        ssm_state_size=4, chunk_size=8, conv_kernel=4,
        intermediate_size=32, moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=32, n_routed_experts=3,
        n_shared_experts=1, num_experts_per_tok=2, n_group=1, topk_group=1,
        moe_latent_size=None, mlp_hidden_act="relu2", use_cache=False,
        attention_dropout=0.3,
    )
    config._attn_implementation = "eager"
    dense = NemotronHForCausalLM(config).to(torch.bfloat16)
    records = {}
    for name, tensor in dense.state_dict().items():
        name = name.replace("model.", "backbone.", 1)
        if ".experts.up_proj" in name or ".experts.down_proj" in name:
            for i in range(3):
                prefix = name.replace(".experts.", f".experts.{i}.")
                records.update(_record(prefix, tensor.shape[1], tensor.shape[2]))
        elif name.endswith((".in_proj.weight", ".out_proj.weight")):
            records[name] = torch.randn(tensor.shape).to(torch.float8_e4m3fn)
            records[name.removesuffix("weight") + "weight_scale"] = torch.tensor(0.01)
        elif ".shared_experts." in name or name == "lm_head.weight":
            records.update(_record(name.removesuffix(".weight"), tensor.shape[0], tensor.shape[1]))
        else:
            records[name] = tensor.contiguous()
            if name.endswith((".A_log", ".D", ".dt_bias")):
                records[name] = tensor.float()
    records["backbone.layers.0.mixer.k_proj.k_scale"] = torch.tensor(0.25)
    records["backbone.layers.0.mixer.v_proj.v_scale"] = torch.tensor(0.5)
    config.save_pretrained(tmp_path)
    _checkpoint(tmp_path, records)
    learner = build_quantized_learner(tmp_path, lora_rank=2, lora_alpha=4, device="cpu")
    audit_before = audit_quantized_learner(learner)
    digest_before = frozen_tensor_digest(learner)
    learner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    learner.train()
    assert learner.training
    assert learner.model.layers[0].mixer.attention_dropout == 0.0
    assert not learner.model.layers[1].mixer.training
    inputs = torch.tensor([[1, 4, 9, 2]])
    result = learner(input_ids=inputs, use_cache=False, logits_to_keep=0)
    assert torch.isfinite(result.logits).all()
    result.logits.float().square().mean().backward()
    grads = [p.grad for p in learner.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.float().abs().sum() for g in grads) > 0
    assert frozen_tensor_digest(learner) == digest_before
    assert audit_quantized_learner(learner) == audit_before
    learner.eval()
    learner.train()
    assert not learner.model.layers[1].mixer.training
