"""Small numerical oracles for native-token policy scoring and GRPO."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nvfp4_lora.grpo import (Trajectory, collate, group_advantages, gsm8k_reward,
                             policy_logprobs, policy_loss, prompt_ids, selected_logprobs)
from nvfp4_lora.grpo_rollout import chosen_logprob, reload_proof


def trajectory(prompt, completion, behavior=None):
    return Trajectory(prompt, completion, [-1.0] * len(completion) if behavior is None else behavior, "", 0, 0)


def test_padding_eos_and_empty_completion_alignment():
    batch = collate([trajectory([2, 3], [4, 0]), trajectory([2], [0]), trajectory([2, 3], [])], 0)
    assert batch.input_ids.tolist() == [[2, 3, 4, 0], [2, 0, 0, 0], [2, 3, 0, 0]]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]]
    assert batch.completion_mask.tolist() == [[False, False, True, True], [False, True, False, False], [False] * 4]
    torch.manual_seed(42)
    logits = torch.randn(3, 4, 5, requires_grad=True)
    selected = selected_logprobs(logits, batch.input_ids, batch.completion_mask, 1.2)
    reference = (logits.float() / 1.2).log_softmax(-1)
    torch.testing.assert_close(selected[0, 2], reference[0, 1, 4])
    torch.testing.assert_close(selected[0, 3], reference[0, 2, 0])
    torch.testing.assert_close(selected[1, 1], reference[1, 0, 0])
    assert selected[~batch.completion_mask].eq(0).all()
    selected.sum().backward()
    assert logits.grad[0, 0].eq(0).all()
    assert logits.grad[0, 3].eq(0).all()
    assert logits.grad[1, 1:].eq(0).all()


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        selected_logprobs(torch.zeros(1, 2, 3), torch.ones(1, 2, dtype=torch.long), torch.tensor([[False, True]]), temperature)


def test_temperature_matches_behavior_distribution_and_gradient():
    logits = torch.tensor([[[1., 2., 4.], [0., 0., 0.]]], requires_grad=True)
    ids = torch.tensor([[0, 1]])
    mask = torch.tensor([[False, True]])
    for temperature in (0.5, 1.2, 2.0):
        selected = selected_logprobs(logits, ids, mask, temperature)[0, 1]
        expected = torch.log_softmax(logits[0, 0] / temperature, dim=0)[1]
        torch.testing.assert_close(selected, expected)
        grad = torch.autograd.grad(selected, logits, retain_graph=True)[0][0, 0]
        expected_grad = (torch.tensor([0., 1., 0.]) - torch.softmax(logits[0, 0] / temperature, dim=0)) / temperature
        torch.testing.assert_close(grad, expected_grad)


def test_zero_variance_and_singleton_groups_have_zero_advantage():
    rewards = torch.tensor([1., 1., 0., 1., 1.])
    result = group_advantages(rewards, torch.tensor([0, 0, 1, 1, 2]))
    assert result[[0, 1, 4]].eq(0).all()
    torch.testing.assert_close(result[2:4], torch.tensor([-2 ** -.5, 2 ** -.5]), atol=2e-6, rtol=1e-6)


def test_nonzero_gradient_at_ratio_one_and_detached_behavior_correction():
    current = torch.tensor([[-1., -2.], [-3., -4.]], requires_grad=True)
    old = current.detach().clone().requires_grad_()
    behavior = old.detach().clone().requires_grad_()
    mask = torch.ones_like(current, dtype=torch.bool)
    loss = policy_loss(current, old, behavior, mask, torch.tensor([1., -1.]))
    loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([[-.25, -.25], [.25, .25]]))
    assert old.grad is None and behavior.grad is None


def test_empty_completion_loss_is_zero_with_finite_zero_gradient():
    current = torch.zeros(2, 3, requires_grad=True)
    loss = policy_loss(current, current.detach(), current.detach(), torch.zeros(2, 3, dtype=torch.bool), torch.ones(2))
    loss.backward()
    assert loss.item() == 0 and current.grad.eq(0).all()


def test_single_prompt_token_without_completion_has_no_loss():
    logits = torch.zeros(1, 1, 5, requires_grad=True)
    batch = collate([trajectory([2], [])], 0)
    actual = selected_logprobs(logits, batch.input_ids, batch.completion_mask, 1.2)
    assert actual.shape == (1, 1) and actual.item() == 0
    actual.sum().backward()
    assert logits.grad.eq(0).all()


def test_ppo_clip_stops_gradient_only_in_improving_direction():
    current = torch.tensor([[1.], [-1.], [1.], [-1.]], requires_grad=True)
    old = torch.zeros_like(current)
    advantages = torch.tensor([1., -1., -1., 1.])
    policy_loss(current, old, old, torch.ones_like(old, dtype=torch.bool), advantages).backward()
    assert current.grad[0].item() == 0 and current.grad[1].item() == 0
    assert current.grad[2].item() > 0 and current.grad[3].item() < 0


def test_extreme_behavior_logprob_correction_is_bounded():
    current = torch.zeros(2, 1, requires_grad=True)
    old = torch.zeros(2, 1)
    behavior = torch.tensor([[-1000.], [1000.]])
    loss = policy_loss(current, old, behavior, torch.ones(2, 1, dtype=torch.bool), torch.ones(2))
    loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([[-5.], [-.05]]))
    assert torch.isfinite(loss)


def test_uneven_microbatch_weight_preserves_full_batch_gradient():
    torch.manual_seed(42)
    initial = torch.randn(5, 4)
    advantages = torch.tensor([1., -1., 0., 2., -2.])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]], dtype=torch.bool)
    full = initial.clone().requires_grad_()
    policy_loss(full, initial, initial - .4, mask, advantages).backward()
    micro = initial.clone().requires_grad_()
    for start in range(0, 5, 2):
        end = min(start + 2, 5)
        loss = policy_loss(micro[start:end], initial[start:end], initial[start:end] - .4, mask[start:end], advantages[start:end])
        (loss * (end - start) / 5).backward()
    torch.testing.assert_close(full.grad, micro.grad)


def test_policy_logprobs_disables_cache():
    class Model:
        def __call__(self, **kwargs):
            assert kwargs["use_cache"] is False
            return SimpleNamespace(logits=torch.ones(1, 2, 5))
    batch = collate([trajectory([2], [3])], 0)
    actual = policy_logprobs(Model(), batch, 1.2)
    torch.testing.assert_close(actual[0, 1], torch.tensor(-5.0).abs().log().neg())


def test_rejects_incomplete_native_behavior_probabilities():
    with pytest.raises(ValueError, match="one native"):
        collate([trajectory([2], [3, 4], [-1.0])], 0)
    with pytest.raises(ValueError, match="finite"):
        collate([trajectory([2], [3], [float("nan")])], 0)
    with pytest.raises(RuntimeError, match="omitted"):
        chosen_logprob(3, {4: SimpleNamespace(logprob=-1.0)})


def test_reload_proof_rejects_no_effect_noise_and_unstable_reload():
    assert reload_proof([-1., -2.], [-1.01, -2.], [-1.01, -2.], [-1., -2.])["passed"]
    assert not reload_proof([-1.], [-1.], [-1.], [-1.])["passed"]
    assert not reload_proof([-1.], [-1.00001], [-1.00001], [-1.0001])["passed"]
    assert not reload_proof([-1.], [-1.01], [-1.1], [-1.])["passed"]


@pytest.mark.parametrize(("completion", "reference", "expected"), [
    ("The calculation is 1,200.\n#### 1,200", "explanation\n#### 1200", 1),
    ("\\boxed{42}", "#### 42", 0), ("42", "#### 42", 0),
    ("#### -3.5", "#### -3.50", 1), ("", "#### 42", 0), ("#### 4", "#### 42", 0),
    ("#### 9.", "#### 9", 1), ("#### -3.5.", "#### -3.50", 1),
    ("#### 9..", "#### 9", 0),
    ("Calculation\n  #### +1,200.00 \t\n\n", "#### 1200", 1),
    ("#### -.5", "#### -0.50", 1),
    ("#### 0", "#### 0", 1),
    ("#### 1,20", "#### 120", 0),
    ("#### 42\nHere is more reasoning.", "#### 42", 0),
    ("#### 42 dollars", "#### 42", 0),
    ("My answer is #### 42", "#### 42", 0),
    ("####\n42", "#### 42", 0),
])
def test_gsm8k_numeric_verifier(completion, reference, expected):
    assert gsm8k_reward(completion, reference) == expected


def test_truncated_reasoning_with_correct_last_number_is_not_an_answer():
    assert gsm8k_reward("She has enough time to enter 3 races", "Solution\n#### 3") == 0
    assert gsm8k_reward("She has enough time to enter 3 races\n#### 3", "Solution\n#### 3") == 1


def test_prompt_disables_thinking():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            assert "####" in messages[0]["content"]
            return {"input_ids": [1, 2, 3]}
    assert prompt_ids(Tokenizer(), "test") == [1, 2, 3]


def load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_grpo.py"
    spec = importlib.util.spec_from_file_location("train_grpo_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_provenance_without_git(monkeypatch):
    module = load_runner()
    monkeypatch.delenv("NVFP4_SOURCE_REVISION", raising=False)
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: pytest.fail("must not execute unavailable git"))
    assert module.source_provenance() == {"code_revision": None, "code_revision_source": "unavailable"}
    monkeypatch.setenv("NVFP4_SOURCE_REVISION", "")
    assert module.source_provenance() == {"code_revision": None, "code_revision_source": "unavailable"}


def test_source_provenance_prefers_supervisor_revision(monkeypatch):
    module = load_runner()
    revision = "a" * 40
    monkeypatch.setenv("NVFP4_SOURCE_REVISION", revision)
    monkeypatch.setattr(module.shutil, "which", lambda name: pytest.fail("supervisor revision needs no git lookup"))
    assert module.source_provenance() == {"code_revision": revision, "code_revision_source": "supervisor_env"}
    monkeypatch.setenv("NVFP4_SOURCE_REVISION", "unknown")
    with pytest.raises(ValueError, match="full Git commit"):
        module.source_provenance()


def test_runner_import_and_default_contract_without_vllm():
    module = load_runner()
    args = module.parse_args(["--model-dir", "/unused", "--output-dir", "/unused-output"])
    assert (args.batch_size, args.num_generations, args.steps, args.lora_rank) == (4, 8, 1, 8)
    assert args.max_batch_attempts == 3 and args.eval_size == 16 and args.temperature == 1.2
    with pytest.raises(SystemExit):
        module.parse_args(["--model-dir", "/unused", "--output-dir", "/unused-output", "--max-batch-attempts", "4"])
    model = torch.nn.Module()
    model.add_module("projection", torch.nn.Module())
    model.projection.register_parameter("lora_A", torch.nn.Parameter(torch.zeros(1)))
    model.projection.register_parameter("lora_B", torch.nn.Parameter(torch.zeros(1)))
    model.projection.lora_A.grad = torch.zeros(1)
    model.projection.lora_B.grad = torch.ones(1)
    assert not module.adapter_gradient_audit(model, ["projection"])["gradient_failures"]
    model.projection.lora_B.grad = torch.zeros(1)
    assert module.adapter_gradient_audit(model, ["projection"])["gradient_failures"] == ["zero B gradient: projection.lora_B"]
    model.projection.lora_A.grad = None
    assert "missing gradient: projection.lora_A" in module.adapter_gradient_audit(model, ["projection"])["gradient_failures"]
