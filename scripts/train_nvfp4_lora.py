#!/usr/bin/env python3
"""Unified LoRA fine-tuning for NVFP4-quantized models on GB10 (DGX Spark).

One trainer for every supported model family. Supersedes the per-model
scripts (train_mistral_rh_nvfp4_lora_ich_smoke.py,
train_qwen3_5_122b_rh_nvfp4_lora_ich.py), which remain as frozen, proven
artifacts of their respective runs.

What is detected instead of hardcoded:
  * Model family (config.json model_type) -> auto class, expert-tensor key
    translation, PEFT scoping, submodules to freeze.
  * LoRA mechanism: each target trains through a frozen-base LoRALinear in the
    NATIVE path -- NVFP4 -> NVFP4LoRALinear, FP8 -> FP8LoRALinear, BF16 ->
    BF16LoRALinear -- so NVFP4/FP8/BF16 targets co-train in one adapter. Only an
    all-BF16 target set takes the standard PEFT path (family-scoped regex).

GB10 UMA lessons are applied via nvfp4_lora.gb10_prep:
  * expandable_segments alloc conf set before the first CUDA allocation
  * weight-sized buffers allocated directly on cuda (never CPU)
  * shard page cache dropped after assembly (posix_fadvise, no sudo)

Crash safety: atomic adapter saves (tmp+rename), checkpoint rotation,
best-by-val-loss tracking at <output_dir>/best/, and full resume via
--resume-from (adapter + optimizer + scheduler + RNG + deterministic
per-epoch data order).

Smoke example (Qwen3.5-122B):
    python -u scripts/train_nvfp4_lora.py \\
        --model-dir /path/to/RedHatAI-Qwen3.5-122B-A10B-NVFP4 \\
        --target-modules q_proj,k_proj,v_proj,o_proj \\
        --max-train-examples 8 --max-val-examples 4 --max-steps 3 \\
        --eval-every 0 --checkpoint-every 0 --output-dir /tmp/unified_smoke
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Must run before the first CUDA allocation.
from nvfp4_lora.gb10_prep import set_alloc_conf  # noqa: E402

set_alloc_conf()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.metadata  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from accelerate import init_empty_weights  # noqa: E402
from torch.nn.utils.rnn import pad_sequence  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from nvfp4_lora.experts import (  # noqa: E402
    NVFP4Experts3D,
    assemble_nvfp4_experts3d_batched,
    detect_moe_expert_storage,
    replace_moe_experts_with_nvfp4_3d,
)
from nvfp4_lora.families import FAMILIES, family_view, resolve_family  # noqa: E402, F401
from nvfp4_lora.gb10_prep import drop_shard_page_cache, memory_snapshot  # noqa: E402
from nvfp4_lora.chat_encode import encode_chat_example  # noqa: E402
from nvfp4_lora.linear import BF16LoRALinear, FP8LoRALinear, NVFP4LoRALinear  # noqa: E402
from nvfp4_lora.loader import (  # noqa: E402
    _assign_dequant_workspaces,
    assert_no_meta_tensors,
    decide_lora_mode,
    load_non_nvfp4_weights,
    replace_bf16_targets,
    replace_nvfp4_modules,
)
from nvfp4_lora.training_utils import (  # noqa: E402
    load_step_callback,
    resolve_callback_cadence,
    run_step_callback,
)

# The family registry (FAMILIES / resolve_family) lives in nvfp4_lora/families.py
# and is shared with the loader, the checkpoint inspector and the merge scripts.
# They are re-exported above so existing callers and tests keep working.


def detect_lora_mode(
    model_dir: Path,
    target_suffixes: list[str],
    allow_partial_targets: bool = False,
    family: dict | None = None,
) -> tuple[str, dict]:
    """'native' if the target modules are NVFP4-quantized in the checkpoint,
    'peft' if they are plain BF16. Returns (mode, coverage_report).

    Unlike the v1 suffix-set heuristic, this classifies EVERY matching module
    individually (via loader.decide_lora_mode), so partial quantization across
    layers and FP8-demoted targets are hard errors instead of silent gaps.

    `family` is the effective per-run view; when it is a vision view the coverage
    inventory is restricted to the tower + projector. `None` is the text default.
    """
    return decide_lora_mode(
        model_dir,
        target_suffixes,
        allow_partial_targets=allow_partial_targets,
        family=family,
    )


def _sha256_file(path):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return None


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_run_meta(args, coverage) -> dict:
    """Pure-ish run metadata bundle; no torch/transformers/peft imports."""
    arg_snapshot = dict(sorted(vars(args).items()))
    files = {
        "train_file": {
            "path": arg_snapshot.get("train_file"),
            "sha256": _sha256_file(arg_snapshot.get("train_file")),
        },
        "val_file": {
            "path": arg_snapshot.get("val_file"),
            "sha256": _sha256_file(arg_snapshot.get("val_file")),
        },
    }
    return {
        "args": arg_snapshot,
        "coverage": coverage,
        "files": files,
        "git_sha": _git_sha(),
        "versions": {
            "peft": _package_version("peft"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
    }


def _cuda_metrics():
    out = {
        "cuda_allocated_gb": None,
        "cuda_free_gb": None,
        "cuda_reserved_gb": None,
    }
    try:
        if not torch.cuda.is_available():
            return out
        out["cuda_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 4)
        out["cuda_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 4)
        free, _total = torch.cuda.mem_get_info()
        out["cuda_free_gb"] = round(free / 1e9, 4)
    except Exception:
        return {
            "cuda_allocated_gb": None,
            "cuda_free_gb": None,
            "cuda_reserved_gb": None,
        }
    return out


def _host_mem_available_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().available / 1e9, 4)
    except Exception:
        return None


def build_metrics_row(
    step,
    total_updates,
    window_supervised_tokens,
    wall_elapsed,
    recent_upd_s,
    loss_window_mean,
):
    updates_s = None
    if recent_upd_s is not None and math.isfinite(recent_upd_s) and recent_upd_s > 0:
        updates_s = 1.0 / recent_upd_s
    supervised_tokens_s = None
    if recent_upd_s is not None and math.isfinite(recent_upd_s) and recent_upd_s > 0:
        supervised_tokens_s = window_supervised_tokens / recent_upd_s
    eta_s = None
    if step > 0 and wall_elapsed is not None and math.isfinite(wall_elapsed):
        eta_s = max(0.0, (total_updates - step) * (wall_elapsed / step))
    row = {
        "eta_s": (round(eta_s, 1) if eta_s is not None else None),
        "loss_window_mean": (round(loss_window_mean, 4) if loss_window_mean is not None else None),
        "supervised_tokens_s": (round(supervised_tokens_s, 2) if supervised_tokens_s is not None else None),
        "updates_s": (round(updates_s, 4) if updates_s is not None else None),
        "window_supervised_tokens": int(window_supervised_tokens),
    }
    row.update(_cuda_metrics())
    row["host_mem_available_gb"] = _host_mem_available_gb()
    return row


# =========================================================================
# Dataset (messages -> input_ids/labels with assistant-only loss masking)
# =========================================================================
class ChatJsonlDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int, max_examples: int | None = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items: list[dict[str, torch.Tensor]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if max_examples is not None and len(self.items) >= max_examples:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                item = self._encode(obj["messages"])
                if item is not None:
                    self.items.append(item)

    def _tokenize(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _render(self, messages, add_generation_prompt: bool = False) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def _encode(self, messages):
        # mistral_common (tekken) models tokenize chat token-natively; the HF
        # text-render path below is wrong for them (broken LlamaTokenizerFast).
        if getattr(self.tokenizer, "is_mistral_common", False):
            return self.tokenizer.encode_chat(messages, self.max_length)
        encoded = encode_chat_example(messages, self.tokenizer, self.max_length)
        if encoded["dropped_reason"] is not None:
            return None
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "labels": torch.tensor(encoded["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_batch(batch, pad_token_id: int):
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# =========================================================================
# Model load
# =========================================================================
def _stage(tag: str) -> None:
    snap = memory_snapshot()
    print(f"[load-mem] {tag}: rss={snap['process_rss_gb']}GB cuda_free={snap['cuda_free_gb']}GB", flush=True)


def _force_attn_implementation(cfg, impl: str, _depth: int = 0) -> None:
    """Set `_attn_implementation` on a config AND its nested sub-configs (`*_config`).

    from_config's `attn_implementation` kwarg only reaches the top config; a composite
    multimodal config (llm_config / vision_config / sound_config) needs each sub-config
    set so a submodule that declares no FA2 support doesn't raise at construction.
    """
    if cfg is None or _depth > 4:
        return
    try:
        cfg._attn_implementation = impl
        cfg._attn_implementation_internal = impl
    except Exception:
        pass
    for key in list(getattr(cfg, "__dict__", {}).keys()):
        if "config" in key:
            sub = getattr(cfg, key, None)
            if hasattr(sub, "__dict__"):
                _force_attn_implementation(sub, impl, _depth + 1)


def load_model(
    model_dir: Path,
    family: dict,
    device: torch.device,
    dtype: torch.dtype,
    lora_mode: str,
    target_suffixes: list[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    strict: bool = True,
    expert_lora_r: int = 0,
    expert_lora_alpha: int = 0,
    expert_lora_dropout: float = 0.0,
):
    print("[load] building model on meta…", flush=True)
    # `family` here is the EFFECTIVE per-run view (families.family_view). In a vision
    # run (_train_target=="vision") the LLM backbone stays entirely frozen (no NVFP4
    # LoRA); only the bf16 tower + projector train, via replace_bf16_targets scoped to
    # the view's vision peft_scope. Text runs pass the registry entry verbatim (the
    # view is the identity), so nothing below changes for them.
    is_vision = family.get("_train_target") == "vision"
    is_both = family.get("_train_target") == "both"
    cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    # Some multimodal families (e.g. NemotronH-Omni) declare NO Flash-Attention-2 support and
    # default to it, so from_config raises unless we force eager. The family opts in via
    # `attn_implementation`; it must be set RECURSIVELY on nested sub-configs (llm/vision/sound),
    # because from_config's attn_implementation kwarg does not propagate to them.
    attn_impl = family.get("attn_implementation")
    if attn_impl:
        _force_attn_implementation(cfg, attn_impl)
    auto_cls = AutoModelForCausalLM if family["auto_class"] == "causal_lm" else AutoModelForImageTextToText
    with init_empty_weights():
        model = auto_cls.from_config(
            cfg, trust_remote_code=True,
            **({"attn_implementation": attn_impl} if attn_impl else {}),
        )
    _stage("post-meta-build")

    model_type = getattr(model.config, "model_type", None)
    if family.get("moe_experts_class"):
        # Probe the checkpoint's expert storage: ModelOpt vs compressed-tensors
        # key naming, and whether gate/up per-tensor scales differ (which
        # requires split storage with one global scale per projection).
        idx_obj = json.loads((model_dir / "model.safetensors.index.json").read_text())
        moe_storage = detect_moe_expert_storage(model_dir, idx_obj["weight_map"])
        if moe_storage is None:
            # Dense variant of a family whose model_type also covers MoE members:
            # mistral3 spans both the dense Mistral-Small-3.2-24B and the MoE
            # Mistral-Small-4-119B. No per-expert keys -> treat as a dense variant,
            # skip fused-3D replacement; replace_nvfp4_modules handles the dense MLP
            # (and the expert-assembly loop below is a no-op with no NVFP4Experts3D).
            print("[load] moe_experts_class declared but checkpoint has no per-expert "
                  "keys; treating as a dense variant of this family (skipping MoE "
                  "replacement)", flush=True)
        else:
            _ex_r = expert_lora_r if lora_mode == "native" else 0
            print(f"[load] replacing fused-3D MoE blocks with NVFP4Experts3D "
                  f"(format={moe_storage['quant_format']}, "
                  f"split_gate_up_scales={moe_storage['split_gate_up_scales']}, "
                  f"expert_lora_r={_ex_r})…", flush=True)
            # device= is load-bearing on GB10 UMA: the default (None) allocates the
            # packed expert buffers (~weight-sized) on CPU, permanently starving CUDA.
            replace_moe_experts_with_nvfp4_3d(
                model, model_family=model_type, device=device,
                quant_format=moe_storage["quant_format"],
                split_gate_up_scales=moe_storage["split_gate_up_scales"],
                lora_r=_ex_r, lora_alpha=expert_lora_alpha, lora_dropout=expert_lora_dropout,
                lora_dtype=dtype,
            )
            _stage("post-moe-replace")
    else:
        # Family stores routed experts as per-expert nn.Linear modules
        # (Nemotron); replace_nvfp4_modules handles them like any other linear.
        if expert_lora_r and lora_mode == "native":
            print("[load] NOTE: --expert-lora-r set but this family stores experts as per-module "
                  "linears (not fused-3D); expert adaptation falls under the normal target-module "
                  "LoRA, not the fused expert path.", flush=True)
        print("[load] no fused-3D MoE for this family; experts are per-module linears", flush=True)

    # Vision run: the NVFP4 LLM is pure frozen dequant-forward (native_targets=()),
    # so its buffers still load for the frozen graph but NO NVFP4 LoRA attaches. This
    # is what keeps the vision suffixes (gate_proj/down_proj also name the NVFP4 text
    # experts) from accidentally adapting the backbone -- the tower is reached only by
    # the vision-scoped replace_bf16_targets below.
    native_targets = () if is_vision else (tuple(target_suffixes) if lora_mode == "native" else ())
    print(f"[load] replacing NVFP4 nn.Linear (mode={lora_mode}, native targets={list(native_targets)})…", flush=True)
    replace_nvfp4_modules(
        model, model_dir,
        target_lora_suffixes=native_targets,
        r=lora_r if native_targets else 0,
        lora_alpha=lora_alpha if native_targets else 0,
        lora_dropout=lora_dropout if native_targets else 0.0,
        device=device, dtype=dtype, family=family,
    )
    _stage("post-linear-replace")

    if family.get("expert_prefix"):
        print("[load] assembling routed-expert NVFP4 buffers…", flush=True)
        mem_prefix, st_prefix = family["expert_prefix"]
        idx_obj = json.loads((model_dir / "model.safetensors.index.json").read_text())
        wm = idx_obj["weight_map"]
        for name, module in model.named_modules():
            if isinstance(module, NVFP4Experts3D):
                assert name.startswith(mem_prefix), f"unexpected expert path: {name!r}"
                st_name = st_prefix + name[len(mem_prefix):]
                assemble_nvfp4_experts3d_batched(module, st_name, model_dir, wm)
        _stage("post-expert-assembly")

    print("[load] loading non-NVFP4 weights (attention/embeddings/norms/lm_head)…", flush=True)
    load_non_nvfp4_weights(model, model_dir, device=device, dtype=dtype, strict=strict, family=family)
    _stage("post-non-nvfp4-load")

    # Co-train genuinely-BF16 targets (e.g. attention a mixed-precision quantizer left in
    # bf16) alongside the NVFP4/FP8 ones in this single native adapter. Runs AFTER the bf16
    # weights are loaded; family peft_scope keeps out-of-scope BF16 (MTP heads / vision
    # towers) frozen. No-op when there are no in-scope BF16 targets (e.g. the 3.6, whose
    # attention is FP8).
    if lora_mode == "native" and is_both:
        # `both`: TWO paired passes with disjoint (scope, suffixes). Pass A wraps text bf16
        # targets under the TEXT scope with projector_scopes=() (the projector belongs to
        # pass B, not the text pass). Pass B wraps the bf16 tower + projector under the
        # VISION scope. De-dup between passes is automatic: BF16LoRALinear is an nn.Module,
        # not an nn.Linear, and replace_bf16_targets only collects nn.Linear, so pass B
        # inherently skips anything pass A wrapped (no double-wrap).
        vision_suffixes = list(family.get("_vision_target_suffixes", ()))
        n_text_bf16 = replace_bf16_targets(
            model, target_suffixes, family.get("_text_peft_scope"),
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            device=device, dtype=dtype, projector_scopes=(),
        )
        n_vis_bf16 = replace_bf16_targets(
            model, vision_suffixes, family.get("_vision_peft_scope"),
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            device=device, dtype=dtype, projector_scopes=family.get("_projector_scopes", ()),
        )
        print(f"[load] both: wrapped {n_text_bf16} text-bf16 + {n_vis_bf16} tower/projector "
              f"BF16LoRALinear (co-trained natively)", flush=True)
        # Per-HALF wrap assert: a single total count can hide a zero-wrapped half. The tower
        # half must be > 0 here; the text half is asserted after freeze/enable (it may be all
        # NVFP4/FP8 native with zero bf16, which is legitimate).
        if n_vis_bf16 == 0:
            raise SystemExit(
                "--train-target both wrapped ZERO tower/projector Linears (pass B): check "
                "--vision-target-modules and that the tower fell in the vision scope")
        _stage("post-bf16-targets")
    elif lora_mode == "native":
        n_bf16 = replace_bf16_targets(
            model, target_suffixes, family.get("peft_scope"),
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            device=device, dtype=dtype,
            projector_scopes=family.get("_projector_scopes", ()),
        )
        if n_bf16:
            print(f"[load] wrapped {n_bf16} BF16 target Linears with BF16LoRALinear "
                  f"(co-trained natively)", flush=True)
        _stage("post-bf16-targets")

    _assign_dequant_workspaces(model, device=device, dtype=dtype)
    _stage("post-workspaces")

    # Catch any straggler CPU tensors (RoPE inv_freq etc. — should be tiny;
    # anything weight-sized here indicates a placement bug upstream).
    moved_bytes = 0
    for mod in model.modules():
        for nm, buf in list(mod.named_buffers(recurse=False)):
            if buf.device.type == "cpu":
                moved_bytes += buf.numel() * buf.element_size()
                mod._buffers[nm] = buf.to(device)
        for nm, par in list(mod.named_parameters(recurse=False)):
            if par.device.type == "cpu":
                moved_bytes += par.numel() * par.element_size()
                mod._parameters[nm] = torch.nn.Parameter(
                    par.data.to(device), requires_grad=par.requires_grad
                )
    if moved_bytes > 1e9:
        print(f"[load] WARNING: move-loop relocated {moved_bytes/1e9:.1f}GB from CPU; "
              f"a loader stage is allocating weight-sized buffers on the wrong device", flush=True)
    _stage("post-move-loop")

    before, after = drop_shard_page_cache(model_dir)
    print(f"[load] dropped shard page cache: cuda_free {before:.1f}GB -> {after:.1f}GB", flush=True)

    # Freeze multimodal towers (text-only training)
    inner = getattr(model, "model", model)
    for attr in family["freeze"]:
        sub = getattr(inner, attr, None)
        if sub is None:
            continue
        for p in sub.parameters():
            p.requires_grad = False

    # Tied embeddings never appear as a separate lm_head tensor on disk; re-tie
    # so lm_head does not stay on meta. No-op for untied checkpoints.
    if getattr(cfg, "tie_word_embeddings", False) and hasattr(model, "tie_weights"):
        model.tie_weights()

    # Everything still on meta at this point was never loaded and will explode
    # at first forward; only the family's frozen multimodal towers are allowed.
    meta_allowed = tuple(family.get("meta_allowed_prefixes", ()))
    if strict:
        assert_no_meta_tensors(model, allowed_prefixes=meta_allowed)
    else:
        try:
            assert_no_meta_tensors(model, allowed_prefixes=meta_allowed)
        except RuntimeError as e:
            print(f"[load] WARNING (--permissive-load): {e}", flush=True)

    return model


def attach_peft_lora(model, family: dict, target_suffixes: list[str],
                     lora_r: int, lora_alpha: int, lora_dropout: float):
    from peft import LoraConfig, get_peft_model
    # Scope to the text backbone so multimodal towers (whose weights may sit on
    # meta) can never match a bare suffix.
    target_regex = family["peft_scope"] + r".*\.(" + "|".join(target_suffixes) + r")$"
    peft_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        bias="none", target_modules=target_regex,
    )
    return get_peft_model(model, peft_cfg)


def freeze_all_then_enable_lora(model) -> int:
    """Vision-mode freeze mechanism: requires_grad=False on EVERY param, then True on
    exactly the LoRA A/B of the in-scope LoRALinears.

    This is the codex-corrected freeze: it structurally CANNOT use torch.no_grad() on
    the LLM forward (which would sever the autograd graph and silently zero every vision
    gradient -- the #1 footgun), and it catches embeddings / norms / lm_head / router
    gates without enumerating them. In a vision run the only r>0 LoRALinears are the
    vision-scoped BF16LoRALinears (the NVFP4 backbone was loaded with native_targets=()),
    so this enables precisely the tower + projector adapter. Returns the count of
    trainable LoRA tensors enabled.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for _, mod in model.named_modules():
        if isinstance(mod, (NVFP4LoRALinear, FP8LoRALinear, BF16LoRALinear)) and mod.r > 0:
            for pt in (mod.lora_A, mod.lora_B):
                if pt is not None and not (hasattr(pt, "is_meta") and pt.is_meta):
                    pt.requires_grad_(True)
                    n += 1
    return n


def _move_batch_to_device(batch: dict, device, compute_dtype: torch.dtype = torch.bfloat16):
    """Move a batch to `device`, tolerating non-tensor values.

    Text batches are all-integer tensors, so this is byte-for-byte the old
    `{k: v.to(device) ...}`; a vision batch may carry `pixel_values` / `image_sizes`
    as a list of per-image tensors (variable patch counts), which this moves
    element-wise. Non-tensors pass through untouched.

    FLOATING-point tensors are also cast to `compute_dtype` (the bf16 the model runs in):
    an image processor emits `pixel_values` as fp32, but a bf16 vision tower's first matmul
    needs bf16 inputs (some forwards, e.g. NemotronH-Omni, don't self-cast). Integer tensors
    (input_ids/labels/attention_mask/image_flags) keep their dtype; a no-op for text batches.
    """
    def _move(v):
        if isinstance(v, torch.Tensor):
            if compute_dtype is not None and v.is_floating_point():
                return v.to(device=device, dtype=compute_dtype)
            return v.to(device)
        if isinstance(v, (list, tuple)):
            return type(v)(_move(x) for x in v)
        return v
    return {k: _move(v) for k, v in batch.items()}


def assert_vision_grads_flow(model, log_fn, *, train_target: str = "vision",
                             is_vision_param=None) -> None:
    """First-(image-)backward gradient gate for a vision or both run (plan section 5.2).

    After the first backward on an image-containing batch, `lora_B` grads prove the autograd
    graph is intact through the frozen 4-bit LLM (dL/dB flows iff so). A severed graph (the
    no_grad-on-LLM footgun) shows up as zero/None grad on lora_B. lora_A can legitimately be
    zero on step 1 (B is zero-initialized, so dL/dA == 0), so its zero-grad is only a warning.

    `vision` (default, 2-arg call unchanged): EVERY trainable lora_B must be non-zero.
    `both`: the VISION half keeps ALL-nonzero (the tower is dense), but the TEXT half needs
    only >=1 non-zero lora_B -- a MoE LLM stores routed experts as per-expert Linears, so one
    batch exercises only the routed subset and an all-nonzero text check would hard-fail every
    healthy run. `is_vision_param(name)` splits the halves. lora_B/lora_A are matched by suffix
    (`.endswith`) so expert blocks' `lora_B_gate_up`/`lora_B_down` never leak into the check.
    """
    lora_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not lora_params:
        raise SystemExit(
            f"--train-target {train_target}: no trainable params after freeze/enable")

    def _has_grad(p) -> bool:
        return p.grad is not None and p.grad.abs().sum().item() > 0

    b_params = {n: p for n, p in lora_params.items() if n.endswith(".lora_B")}
    a_params = {n: p for n, p in lora_params.items() if n.endswith(".lora_A")}

    if train_target == "both":
        assert is_vision_param is not None, "both grad-gate needs an is_vision_param splitter"
        vis_b = {n: p for n, p in b_params.items() if is_vision_param(n)}
        txt_b = {n: p for n, p in b_params.items() if not is_vision_param(n)}
        # Vision half: ALL-nonzero (a severed graph / mis-scoped tower wrap must trip here).
        zero_vis = [n for n, p in vis_b.items() if not _has_grad(p)]
        if zero_vis:
            raise SystemExit(
                f"severed vision autograd graph: zero/None grad on {len(zero_vis)} tower/"
                f"projector lora_B on the first image batch (the no_grad-on-LLM footgun): "
                f"{zero_vis[:8]}")
        # Text half: >=1 nonzero (a MoE LLM routes only a subset of experts per batch).
        if txt_b and not any(_has_grad(p) for p in txt_b.values()):
            raise SystemExit(
                f"no gradient on ANY of {len(txt_b)} text/LLM lora_B on the first image batch; "
                f"the LLM adapter is not training (check --target-modules / the text scope)")
        log_fn("both_first_image_grad_check", result="pass",
               vision_lora_B=len(vis_b), text_lora_B=len(txt_b),
               text_lora_B_with_grad=sum(1 for p in txt_b.values() if _has_grad(p)))
        return

    # vision (and the 2-arg default): every lora_B non-zero.
    zero_b = [n for n, p in b_params.items() if not _has_grad(p)]
    zero_a = [n for n, p in a_params.items() if not _has_grad(p)]
    if zero_b:
        raise SystemExit(
            f"severed autograd graph: zero/None grad on {len(zero_b)} lora_B param(s) "
            f"(the no_grad-on-LLM footgun -- do NOT wrap the frozen LLM forward in "
            f"torch.no_grad/inference_mode): {zero_b[:8]}"
        )
    if zero_a:
        log_fn("warn_lora_A_zero_grad_step1", names=zero_a[:8])
    log_fn("first_backward_grad_check", result="pass",
           lora_params=len(lora_params), lora_B_checked=len(b_params) - len(zero_a))


def build_text_only_bypass_forward(orig_forward, language_model):
    """Forward wrapper for a `--train-target both` run on a VLM whose training forward MANDATES
    pixel_values (e.g. NemotronH-Omni runs the tower unconditionally; only generate() handles
    no-image). An IMAGE batch (pixel_values present) hits `orig_forward` UNCHANGED; a text-only
    batch runs `language_model` directly on token embeddings, so the tower stays out of the
    graph (no tower grad from text rows) with no wasted tower compute. Returns a plain function
    to assign to `model.forward`.
    """
    vocab = language_model.config.vocab_size

    def _forward(**kw):
        if kw.get("pixel_values", None) is not None:
            return orig_forward(**kw)                          # image row: real forward, untouched
        input_ids = kw["input_ids"]
        inputs_embeds = language_model.get_input_embeddings()(input_ids)  # embed hook -> grad
        out = language_model(inputs_embeds=inputs_embeds,
                             attention_mask=kw.get("attention_mask"),
                             use_cache=False, return_dict=True)
        logits = out.logits
        loss = None
        labels = kw.get("labels")
        if labels is not None:
            sl = logits[..., :-1, :].contiguous().view(-1, vocab)
            st = labels[..., 1:].contiguous().view(-1).to(sl.device)
            loss = torch.nn.functional.cross_entropy(sl, st)   # ignore_index=-100 default
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

    return _forward


# =========================================================================
# Training loop
# =========================================================================
@torch.no_grad()
def evaluate(model, loader, device, compute_dtype: torch.dtype = torch.bfloat16) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        batch = _move_batch_to_device(batch, device, compute_dtype=compute_dtype)
        out = model(**batch)
        # out.loss is CE over SHIFTED tokens (labels[:, 1:] vs logits[:, :-1]),
        # so the token-weight must be the shifted supervised-token count, not the
        # full unshifted (labels != -100) count. Using the unshifted count over-
        # weights batches by their first label position and biases model selection.
        n_tok = (batch["labels"][:, 1:] != -100).sum().item()
        # Skip batches with zero shifted supervised tokens: HF returns nan for an
        # all-ignored CE, and nan*0 would poison the aggregate. (A row can pass the
        # dataset's unshifted-count filter yet have its only label at position 0,
        # which the causal shift drops.)
        if n_tok == 0:
            continue
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok
    model.train()
    # No supervised eval tokens at all -> not a valid val loss; signal inf so
    # best/ selection never treats an empty eval as an improvement.
    return total_loss / total_tokens if total_tokens else float("inf")


def _save_adapter_atomic(model, tokenizer, dest_dir: Path, log_fn, *,
                         lora_mode: str, base_model_dir: str,
                         lora_r: int, lora_alpha: int, lora_dropout: float,
                         target_suffixes, both_meta: dict | None = None) -> None:
    """Atomic (tmp+rename) adapter save for either LoRA mechanism.

    `both_meta` (a `--train-target both` run) records the two-scope structure -- text vs
    vision suffixes and their scopes -- so the merge/serve splitter can route the unified
    adapter's keys (tower -> merge into base; LLM -> merge or runtime-LoRA) without
    re-deriving scope membership from key prefixes alone.
    """
    import shutil
    from safetensors.torch import save_file as safe_save_file

    dest_dir = Path(dest_dir)

    def _raise_if_meta_trainable_tensors(names, context: str) -> None:
        if not names:
            return
        sample = ", ".join(names[:8])
        if len(names) > 8:
            sample += f", ... (+{len(names) - 8} more)"
        raise RuntimeError(
            f"refusing to save {context} adapter to {dest_dir}: {len(names)} trainable "
            f"LoRA tensor(s) are still on meta ({sample}). Saving would publish a "
            "partial adapter."
        )

    tmp_dir = dest_dir.with_name(dest_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if lora_mode == "native":
        state = {}
        meta_trainable = []
        n_expert_blocks = 0
        expert_lora_r = 0
        expert_lora_alpha = 0
        for name, mod in model.named_modules():
            if isinstance(mod, (NVFP4LoRALinear, FP8LoRALinear, BF16LoRALinear)) and mod.r > 0:
                a, b = mod.lora_A, mod.lora_B
                a_meta = hasattr(a, "is_meta") and a.is_meta
                b_meta = hasattr(b, "is_meta") and b.is_meta
                if a_meta:
                    meta_trainable.append(f"{name}.lora_A")
                if b_meta:
                    meta_trainable.append(f"{name}.lora_B")
                if a_meta or b_meta:
                    continue
                state[f"base_model.model.{name}.lora_A.weight"] = a.detach().cpu().contiguous()
                state[f"base_model.model.{name}.lora_B.weight"] = b.detach().cpu().contiguous()
            elif isinstance(mod, NVFP4Experts3D) and getattr(mod, "lora_r", 0) > 0:
                # Per-expert LoRA: native train-side stacked layout, A (E, r, in) and
                # B (E, out, r) for gate_up and down. Mapping this to a serve engine's
                # fused-MoE LoRA format is a separate (GPU-gated) rekey step.
                tensors = {}
                meta = False
                for proj in ("gate_up", "down"):
                    A = getattr(mod, f"lora_A_{proj}"); B = getattr(mod, f"lora_B_{proj}")
                    if hasattr(A, "is_meta") and A.is_meta:
                        meta_trainable.append(f"{name}.experts.{proj}.lora_A")
                        meta = True
                    if hasattr(B, "is_meta") and B.is_meta:
                        meta_trainable.append(f"{name}.experts.{proj}.lora_B")
                        meta = True
                    if meta:
                        continue
                    tensors[f"base_model.model.{name}.experts.{proj}.lora_A"] = A.detach().cpu().contiguous()
                    tensors[f"base_model.model.{name}.experts.{proj}.lora_B"] = B.detach().cpu().contiguous()
                if meta:
                    continue
                state.update(tensors)
                n_expert_blocks += 1
                expert_lora_r = mod.lora_r
                expert_lora_alpha = mod.lora_alpha
        _raise_if_meta_trainable_tensors(meta_trainable, "native")
        if n_expert_blocks:
            log_fn("save_expert_lora", blocks=n_expert_blocks, r=expert_lora_r)
        safe_save_file(state, str(tmp_dir / "adapter_model.safetensors"))
        cfg = {
            "base_model_name_or_path": base_model_dir,
            "peft_type": "LORA", "task_type": "CAUSAL_LM",
            "r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout,
            "bias": "none", "target_modules": list(target_suffixes),
            "inference_mode": True, "fan_in_fan_out": False,
        }
        if n_expert_blocks:
            # Non-PEFT-standard extension: records the per-expert LoRA so resume and
            # the serve-time rekey can reconstruct it. Keys are stacked per-expert
            # (gate_up=w13, down=w2). vLLM does not read adapter_config; the rekey does.
            cfg["expert_lora"] = {
                "r": expert_lora_r, "lora_alpha": expert_lora_alpha,
                "lora_dropout": lora_dropout, "blocks": n_expert_blocks,
                "projections": ["gate_up", "down"],
                "key_format": "base_model.model.{block}.experts.{proj}.lora_{A|B}",
                # Actual saved tensor shapes: A is (E, r, in), B is (E, out, r).
                "tensor_shapes": {
                    "gate_up": {"lora_A": ["E", "r", "hidden"], "lora_B": ["E", "2*intermediate", "r"]},
                    "down": {"lora_A": ["E", "r", "intermediate"], "lora_B": ["E", "hidden", "r"]},
                },
                "experimental": True,
                "note": "train-side native stacked layout; runtime serving is GPU-gated (docs/plans/expert_lora_scope.md)",
            }
        if both_meta is not None:
            cfg["train_target"] = "both"
            # target_modules above is the TEXT half only; the full two-scope record lives here.
            cfg["both"] = both_meta
        (tmp_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    else:
        from peft.utils import get_peft_model_state_dict
        sd = get_peft_model_state_dict(model)
        meta_keys = [k for k, v in sd.items() if hasattr(v, "is_meta") and v.is_meta]
        _raise_if_meta_trainable_tensors(meta_keys, "PEFT")
        safe_save_file({k: v.detach().contiguous() for k, v in sd.items()},
                       str(tmp_dir / "adapter_model.safetensors"),
                       metadata={"format": "pt"})
        model.peft_config[model.active_adapter].save_pretrained(str(tmp_dir))

    tokenizer.save_pretrained(str(tmp_dir))
    # Move files into dest individually (os.replace is atomic per file).
    # Never rmtree dest_dir: the final save targets the OUTPUT ROOT, which
    # also holds best/, checkpoint_step_*/ and metrics.jsonl -- a whole-dir
    # swap deletes them all (this destroyed the best adapter of the Mistral
    # 119B v3.5 run).
    dest_dir.mkdir(parents=True, exist_ok=True)
    for item in tmp_dir.iterdir():
        os.replace(str(item), str(dest_dir / item.name))
    tmp_dir.rmdir()


def _validate_native_resume_coverage(
    *,
    adapter_dir: Path,
    expected_modules: int,
    loaded_modules: int,
    expected_expert_blocks: int,
    loaded_expert_blocks: int,
    expert_missing: int,
    log_fn,
) -> None:
    expected_total = expected_modules + expected_expert_blocks
    loaded_total = loaded_modules + loaded_expert_blocks
    if expected_total == 0:
        return
    if loaded_total == expected_total:
        return
    log_fn(
        "resume_adapter_mismatch",
        modules=f"{loaded_modules}/{expected_modules}",
        expert_blocks=f"{loaded_expert_blocks}/{expected_expert_blocks}",
        expert_missing=expert_missing,
        path=str(adapter_dir),
    )
    raise RuntimeError(
        f"resume adapter mismatch at {adapter_dir}: matched {loaded_total}/{expected_total} "
        f"expected LoRA target(s) (modules {loaded_modules}/{expected_modules}, expert "
        f"blocks {loaded_expert_blocks}/{expected_expert_blocks}). Refusing to resume "
        "with a partial or zero adapter while restoring optimizer/scheduler/RNG state."
    )


def _load_adapter_weights(model, adapter_dir: Path, lora_mode: str, log_fn) -> None:
    from safetensors.torch import load_file
    sd = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    if lora_mode == "native":
        expected = 0
        loaded = 0
        expected_expert_blocks = 0
        expert_loaded = 0
        expert_missing = 0
        for name, mod in model.named_modules():
            if isinstance(mod, (NVFP4LoRALinear, FP8LoRALinear, BF16LoRALinear)) and mod.r > 0:
                expected += 1
                k_a = f"base_model.model.{name}.lora_A.weight"
                k_b = f"base_model.model.{name}.lora_B.weight"
                if k_a in sd and k_b in sd:
                    mod.lora_A.data.copy_(sd[k_a].to(mod.lora_A.device, mod.lora_A.dtype))
                    mod.lora_B.data.copy_(sd[k_b].to(mod.lora_B.device, mod.lora_B.dtype))
                    loaded += 1
            elif isinstance(mod, NVFP4Experts3D) and getattr(mod, "lora_r", 0) > 0:
                expected_expert_blocks += 1
                keys = {proj: (f"base_model.model.{name}.experts.{proj}.lora_A",
                               f"base_model.model.{name}.experts.{proj}.lora_B")
                        for proj in ("gate_up", "down")}
                present = [k for ka_kb in keys.values() for k in ka_kb if k in sd]
                if len(present) == 0:
                    expert_missing += 1
                    continue
                if len(present) != 4:
                    # Partial block: copying some-but-not-all tensors leaves the rest
                    # silently zero-initialized -> a wrong adapter. Fail loud.
                    raise RuntimeError(
                        f"resume: expert LoRA block {name!r} has {len(present)}/4 tensors in the "
                        f"adapter; refusing a partial load. Adapter at {adapter_dir}.")
                for proj, (k_a, k_b) in keys.items():
                    pa, pb = getattr(mod, f"lora_A_{proj}"), getattr(mod, f"lora_B_{proj}")
                    if tuple(sd[k_a].shape) != tuple(pa.shape) or tuple(sd[k_b].shape) != tuple(pb.shape):
                        raise RuntimeError(
                            f"resume: expert LoRA shape mismatch at {name}.{proj}: "
                            f"adapter A{tuple(sd[k_a].shape)}/B{tuple(sd[k_b].shape)} vs "
                            f"model A{tuple(pa.shape)}/B{tuple(pb.shape)} (did --expert-lora-r change?)")
                    pa.data.copy_(sd[k_a].to(pa.device, pa.dtype))
                    pb.data.copy_(sd[k_b].to(pb.device, pb.dtype))
                expert_loaded += 1
        # If the adapter carries expert keys but the model has no expert-LoRA modules
        # (e.g. --expert-lora-r was not re-passed on resume), the saved expert delta
        # is silently dropped -> fail loud rather than train with frozen experts.
        has_expert_keys = any(".experts." in k for k in sd)
        if has_expert_keys and expert_loaded == 0:
            raise RuntimeError(
                f"resume: adapter at {adapter_dir} contains per-expert LoRA tensors but the model "
                f"was built without expert LoRA. Re-pass --expert-lora-r (and matching alpha) to resume.")
        _validate_native_resume_coverage(
            adapter_dir=Path(adapter_dir),
            expected_modules=expected,
            loaded_modules=loaded,
            expected_expert_blocks=expected_expert_blocks,
            loaded_expert_blocks=expert_loaded,
            expert_missing=expert_missing,
            log_fn=log_fn,
        )
        log_fn("resume_adapter_loaded", modules=loaded, expert_blocks=expert_loaded,
               expert_missing=expert_missing, path=str(adapter_dir))
    else:
        from peft.utils import set_peft_model_state_dict
        set_peft_model_state_dict(model, sd)
        log_fn("resume_adapter_loaded", keys=len(sd), path=str(adapter_dir))


def _save_train_state(dest_dir: Path, optim, sched, update_step: int, epoch: int) -> None:
    state = {
        "update_step": update_step,
        "epoch": epoch,
        "optimizer": optim.state_dict(),
        "scheduler": sched.state_dict(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "rng_python": random.getstate(),
    }
    tmp = Path(dest_dir) / "train_state.pt.tmp"
    torch.save(state, str(tmp))
    os.rename(str(tmp), str(Path(dest_dir) / "train_state.pt"))


def _load_train_state(resume_dir: Path, optim, sched, log_fn) -> int:
    state = torch.load(str(Path(resume_dir) / "train_state.pt"), map_location="cpu", weights_only=False)
    optim.load_state_dict(state["optimizer"])
    sched.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng_torch"])
    if state.get("rng_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["rng_cuda"])
    random.setstate(state["rng_python"])
    log_fn("resume_state_loaded", step=state["update_step"], epoch=state["epoch"])
    return state["update_step"]


def _rotate_checkpoints(output_dir: Path, keep: int = 2) -> None:
    import shutil
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return
    ckpts = sorted(
        (p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint_step_")),
        key=lambda p: int(p.name[len("checkpoint_step_"):]),
    )
    for p in ckpts[:-keep] if keep > 0 else ckpts:
        shutil.rmtree(p)


def _save_with_timeout(save_fn, dest, log_fn, *, timeout_s: float, label: str) -> bool:
    """Run save_fn(dest) in a worker thread, bounded by timeout_s.

    The final root save of a large MoE adapter has been observed to hang
    (process spinning at ~100% CPU) AFTER best/ was already written
    successfully -- on every GLM-4.5-Air expert-LoRA run -- costing hours of
    manual babysitting and a kill-after-best/. best/ (and the last rotated
    checkpoint) are the canonical artifacts, so the convenience root save must
    never be allowed to block the process forever. We time-box it: returns True
    if it completed, False if it overran (the daemon worker is then abandoned
    and the process hard-exits in main()). A real save of a ~1GB adapter is
    seconds; the default timeout is generous so the normal path never trips it.
    """
    import threading

    err: dict = {}

    def _run():
        try:
            save_fn(dest)
        except BaseException as e:  # noqa: BLE001 -- re-raised in the parent thread
            err["exc"] = e

    t = threading.Thread(target=_run, name=f"final-save-{label}", daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        log_fn("final_save_timeout", dest=str(dest), timeout_s=timeout_s,
               note="root save overran; best/ and the last checkpoint are the "
                    "canonical artifacts. Exiting without blocking on it.")
        return False
    if "exc" in err:
        raise err["exc"]
    return True


FINAL_SAVE_TIMEOUT_EXIT_CODE = 3


def _exit_after_final_save_timeout(output_dir: Path, log_fn) -> None:
    log_fn("final_save_timeout", path=str(output_dir), fatal=True,
           exit_code=FINAL_SAVE_TIMEOUT_EXIT_CODE,
           note="root adapter save did not complete; refusing to report success")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(FINAL_SAVE_TIMEOUT_EXIT_CODE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--train-file", required=False, default=None,
                    help="JSONL of {\"messages\": [...]} chat examples. Required "
                         "unless --dry-run is set.")
    ap.add_argument("--val-file", default=None,
                    help="Optional validation JSONL; enables evals + best tracking")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    # Per-expert LoRA on fused-3D routed MoE experts (gate_up + down). 0 = off
    # (experts stay frozen, current default). Only applies to native-mode fused-3D
    # MoE families (GLM-4.5-Air, Qwen3.5); a no-op on dense / per-module-expert
    # families. NOTE: runtime serving of an expert delta is GPU-gated on the marlin
    # NVFP4-MoE-LoRA path (see docs/plans/expert_lora_scope.md) -- training works
    # without it, but the adapter is not yet runtime-servable on this hardware.
    ap.add_argument("--expert-lora-r", type=int, default=0,
                    help="LoRA rank for fused-3D routed MoE experts (0=frozen experts)")
    ap.add_argument("--expert-lora-alpha", type=int, default=0,
                    help="LoRA alpha for expert LoRA; default 2*expert_lora_r when 0")
    ap.add_argument("--expert-lora-dropout", type=float, default=0.0)
    ap.add_argument("--target-modules", required=False, default=None,
                    help="Comma-separated projection suffixes. The LoRA mechanism "
                         "(native NVFP4 vs PEFT) is detected from whether these are "
                         "quantized in the checkpoint. Required in --train-target text; "
                         "in vision mode it defaults to the family's vision_target_suffixes "
                         "(or use --vision-target-modules).")
    ap.add_argument("--train-target", choices=("text", "vision", "both"), default="text",
                    help="What to fine-tune. 'text' (default) is byte-for-byte today's "
                         "behaviour: LoRA on the (NVFP4/FP8/BF16) text backbone, tower "
                         "frozen/skipped, no multimodal plumbing. 'vision' freezes the "
                         "text backbone and trains the bf16 vision tower + multimodal "
                         "projector via BF16LoRALinear on the multimodal data path "
                         "(needs an AutoProcessor + an images sidecar). 'both' trains the "
                         "LLM backbone (--target-modules) AND the tower/projector "
                         "(--vision-target-modules) in one run from a mixed image+text "
                         "dataset; requires --batch-size 1. Only families declaring a "
                         "vision scope support vision/both (mistral3/mistral4, llama4).")
    ap.add_argument("--vision-target-modules", default=None,
                    help="Comma-separated projection suffixes for --train-target vision "
                         "(overrides --target-modules and the family default). E.g. "
                         "'linear_1,linear_2' for a projector-only smoke, then "
                         "'q_proj,v_proj' to add Pixtral attention.")
    ap.add_argument("--include-projector", action=argparse.BooleanOptionalAction, default=True,
                    help="In --train-target vision, include the multimodal projector "
                         "(multi_modal_projector) as a LoRA target (default on). "
                         "--no-include-projector trains the tower only; the projector is "
                         "still materialized + frozen (never left on meta).")
    ap.add_argument("--max-image-tiles", type=int, default=None,
                    help="Cap the image processor's dynamic-tiling tile count (for VLMs like "
                         "NemotronH-Omni whose InternVL-style processor tiles by target "
                         "resolution up to a large max, exploding the sequence length "
                         "regardless of source image size). Set to 1-2 for short-answer VQA. "
                         "No-op if the processor has no `max_num_tiles`.")
    ap.add_argument("--allow-partial-targets", action="store_true",
                    help="DEPRECATED / no-op: a target suffix that is NVFP4 in some "
                         "layers and BF16 in others now co-trains both natively "
                         "(quantized via NVFP4LoRALinear, BF16 via BF16LoRALinear), so "
                         "no flag is needed. Accepted for backward compatibility.")
    ap.add_argument("--allow-unverified-family", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="For an unregistered model_type, synthesize a best-effort flat "
                         "causal-LM family instead of failing (default on). The run is "
                         "tagged UNVERIFIED and still gated by strict-load + coverage, which "
                         "fail fast on a real mismatch. Use --no-allow-unverified-family for "
                         "the strict fail-fast, or --family-config for an exact spec.")
    ap.add_argument("--family-config", default=None,
                    help="Path to a JSON family spec (auto_class, expert_prefix, peft_scope, "
                         "freeze, skip_st_prefixes, st_to_model, meta_allowed_prefixes, "
                         "moe_experts_class). Onboards a model without editing library source; "
                         "wins over the registry and the generic fallback.")
    ap.add_argument("--train-dequant-cache-gb", type=float, default=0.0,
                    help="Opt-in: cap (GB) for a train-time bf16 cache of dequantized base weights. "
                         "0 (default) = the memory-efficient recompute path. For small/mid models "
                         "with UMA headroom (<=~32B), a cache >= the bf16 weight size keeps the base "
                         "resident and trains at near-bf16 step time; numerically identical (frozen "
                         "weight). 120B: leave 0 (recompute) -- it will not fit.")
    ap.add_argument("--permissive-load", action="store_true",
                    help="Bring-up escape hatch: downgrade strict-load errors "
                         "(unmapped on-disk tensors, tensors left on the meta "
                         "device) to warnings. Never use for a real run.")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-subset", type=int, default=0,
                    help="If >0, in-flight evals run only over the first N val "
                         "examples (cheap). A subset eval that beats the current "
                         "best triggers a confirming FULL eval; only the full value "
                         "updates best tracking. 0 means every eval is full.")
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--callback-module", default=None,
                    help="Optional dotted module path OR path to a .py file exposing "
                         "on_step(step, model, tokenizer, output_dir). Called on the "
                         "callback cadence (see --callback-every) with the live model, so an "
                         "out-of-tree monitor (e.g. persona-vector drift) can read the "
                         "current weight-state. Any exception it raises is logged and "
                         "swallowed; it can never abort the run. Default: no callback.")
    ap.add_argument("--callback-every", type=int, default=0,
                    help="Cadence (in update steps) for --callback-module. 0 (default) "
                         "means follow --checkpoint-every, so the callback reads exactly the "
                         "weight-states that get checkpointed.")
    ap.add_argument("--final-save-timeout", type=float, default=900.0,
                    help="Seconds to allow the final root-dir adapter save before "
                         "abandoning it and hard-exiting (best/ and the last checkpoint "
                         "remain the canonical artifacts). Guards the observed "
                         "large-MoE final-save hang. A real save is seconds.")
    ap.add_argument("--max-train-examples", type=int, default=None)
    ap.add_argument("--max-val-examples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume-from", default=None,
                    help="checkpoint_step_N/ dir: loads adapter + optimizer/scheduler/RNG "
                         "and fast-forwards the deterministic data order.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preflight OOM probe: load the model exactly as a real run "
                         "would (load_model + LoRA + gradient checkpointing + optimizer), "
                         "run one synthetic forward+backward at (batch_size, max_length), "
                         "log a memory reading, and exit WITHOUT saving any adapter. Catches "
                         "out-of-memory in ~12 minutes instead of mid-run. --train-file is "
                         "not required in this mode.")
    ap.add_argument("--fused-linear-ce", action="store_true",
                    help="Bind Liger fused linear cross-entropy (lce_forward) onto the "
                         "causal-LM head: computes the loss without ever materializing the "
                         "full (seq x vocab) logits tensor or its fp32 upcast, cutting ~10 GB "
                         "of peak activation. Required for seq>=8192 on 121 GB UMA. Binds ONLY "
                         "lce_forward (NOT apply_liger_kernel_*, which would rewrite MoE MLPs "
                         "into dense SwiGLU and corrupt NVFP4 experts). GLM-family validated.")
    args = ap.parse_args()

    if not args.dry_run and not args.train_file:
        ap.error("--train-file is required unless --dry-run is set")
    if args.train_target == "text" and not args.target_modules:
        ap.error("--target-modules is required unless --train-target vision")
    if args.train_target == "both" and not args.target_modules:
        ap.error("--train-target both requires --target-modules (the LLM/text LoRA suffixes); "
                 "the vision suffixes come from --vision-target-modules or the family default")
    if args.train_target in ("vision", "both") and args.expert_lora_r:
        ap.error(
            f"--expert-lora-r is not supported with --train-target {args.train_target}: the "
            "routed experts are not in the trained set here, yet the trainer would still "
            "create + SAVE a zero-delta expert adapter (freeze_all_then_enable_lora does not "
            "re-enable NVFP4Experts3D params, but the save path emits any block with "
            "lora_r>0). Drop --expert-lora-r.")
    if args.train_target == "both" and args.batch_size > 1:
        ap.error(
            "--train-target both requires --batch-size 1: a batch mixing image and text-only "
            "rows drops the later rows' pixel_values (the collator derives image keys from the "
            "first row), silently corrupting training. bs>1 homogeneous bucketing is a future "
            "item.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.output_dir) / "metrics.jsonl"
    if not args.resume_from:
        metrics_path.unlink(missing_ok=True)

    def log(event: str, **kw):
        rec = {"ts": time.strftime("%H:%M:%S"), "event": event, **kw}
        print(f"[{rec['ts']}] {event}: {json.dumps(kw, sort_keys=True)}", flush=True)
        with open(metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    log("config", **vars(args))

    # Optional out-of-tree per-step callback (e.g. a persona-drift monitor). Loaded ONCE,
    # defensively: a broken or missing --callback-module is logged and disabled, never fatal.
    # It fires on --callback-every (0 => the --checkpoint-every cadence).
    callback_cadence = resolve_callback_cadence(args.callback_every, args.checkpoint_every)
    step_callback = None
    if args.callback_module:
        try:
            step_callback = load_step_callback(args.callback_module)
            log("step_callback_loaded", module=args.callback_module, every=callback_cadence)
        except Exception as e:  # noqa: BLE001 - a bad callback module must not kill the run
            log("step_callback_load_failed", module=args.callback_module, error=repr(e))

    model_dir = Path(args.model_dir)
    model_type, family = resolve_family(
        model_dir,
        allow_generic=args.allow_unverified_family,
        family_config=args.family_config,
    )
    if family.get("_source"):
        log("family_from_config", model_type=model_type, source=family["_source"])
    elif family.get("_unverified"):
        print("=" * 72, flush=True)
        print(f"WARNING: model_type={model_type!r} is UNREGISTERED. Using a best-effort "
              f"generic flat causal-LM family mapping. This run is UNVERIFIED: rely on the "
              f"strict-load + coverage gates below to catch a layout mismatch, and confirm "
              f"the adapter applies before trusting it. Pass --family-config for an exact "
              f"spec, or --no-allow-unverified-family to fail fast instead.", flush=True)
        print("=" * 72, flush=True)
        log("family_unverified", model_type=model_type, note=family.get("_note"))
    # Effective per-run family view. text -> the registry entry verbatim (identity,
    # zero behaviour change); vision -> the inverted skip/meta/freeze + vision scope
    # (SystemExit with a porting hint if this family declares no vision scope). Every
    # downstream consumer (loader translator, inventory, freeze, target scope) reads
    # this view, so the toggle is one conditional here, not scattered across bodies.
    family = family_view(family, args.train_target, include_projector=args.include_projector)
    if args.train_target == "vision":
        log("train_target_vision", model_type=model_type,
            vision_peft_scope=family["peft_scope"], include_projector=args.include_projector)
    elif args.train_target == "both":
        log("train_target_both", model_type=model_type,
            text_peft_scope=family.get("_text_peft_scope"),
            vision_peft_scope=family.get("_vision_peft_scope"),
            include_projector=args.include_projector)

    from nvfp4_lora.linear import set_train_dequant_cache_gb
    set_train_dequant_cache_gb(args.train_dequant_cache_gb)
    if args.train_dequant_cache_gb > 0:
        log("train_dequant_cache", gb=args.train_dequant_cache_gb)

    # Vision target suffixes: --vision-target-modules > --target-modules > the family's
    # vision_target_suffixes default. Text mode keeps --target-modules exactly as before.
    if args.train_target == "vision":
        tm = args.vision_target_modules or args.target_modules
        vsuffixes = list(tm.split(",")) if tm else list(family.get("vision_target_suffixes", ()))
        target_suffixes = [m.strip() for m in vsuffixes if m and m.strip()]
        if not target_suffixes:
            ap.error("no vision target suffixes: pass --vision-target-modules or use a "
                     "family with vision_target_suffixes")
    elif args.train_target == "both":
        # TWO suffix lists: TEXT (--target-modules) drives the mode decision + native
        # NVFP4/FP8 LoRA on the LLM; VISION (--vision-target-modules or the family default)
        # selects the bf16 tower Linears for pass B (the projector is path-scoped separately).
        target_suffixes = [m.strip() for m in args.target_modules.split(",") if m.strip()]
        vtm = args.vision_target_modules
        vsuffixes = list(vtm.split(",")) if vtm else list(family.get("vision_target_suffixes", ()))
        vision_suffixes = [m.strip() for m in vsuffixes if m and m.strip()]
        if not vision_suffixes and not family.get("_projector_scopes"):
            ap.error("--train-target both: no vision target suffixes and no projector scope to "
                     "train; pass --vision-target-modules or use a family with "
                     "vision_target_suffixes")
        # Stash on the view so load_model's pass B reads them (no signature change).
        family["_vision_target_suffixes"] = vision_suffixes
    else:
        target_suffixes = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_mode, coverage = detect_lora_mode(
        model_dir, target_suffixes,
        allow_partial_targets=args.allow_partial_targets,
        family=family,
    )
    if args.train_target in ("vision", "both") and lora_mode != "native":
        # bf16 targets (the whole tower; also e.g. nemotron's bf16 attention on the LLM)
        # classify as "peft"; the native BF16LoRALinear path is what wraps + co-trains them
        # (uniform save/merge keys, exercised by test_bf16_lora_linear). For `both`, PEFT mode
        # would be doubly wrong: replace_bf16_targets is gated on native (so NEITHER half wraps)
        # and attach_peft_lora's PeftModel re-pathing breaks the native save/grad-gate. Force it.
        log("force_native_vision" if args.train_target == "vision" else "force_native_both",
            prev_mode=lora_mode)
        lora_mode = "native"
    # Force native for EXPERT-ONLY runs on a fused-3D family: with no attention/MLP
    # targets, detect_lora_mode has nothing NVFP4 to classify and defaults to peft,
    # which would SILENTLY drop the expert LoRA. The fused-3D experts ARE native
    # NVFP4, so expert_lora_r on a moe_experts_class family must run native. (Needed
    # for e.g. Mistral-4/Mistral-3 MoE, whose MLA attention is BF16, so there is no
    # NVFP4 attention target to trip native the usual way.)
    if (args.expert_lora_r and lora_mode != "native"
            and family.get("moe_experts_class") and not target_suffixes):
        log("force_native_expert_only",
            reason="expert_lora_r set on a fused-3D family with no native targets",
            prev_mode=lora_mode)
        lora_mode = "native"
    log("strategy", model_type=model_type, auto_class=family["auto_class"],
        lora_mode=lora_mode, targets=target_suffixes)

    # Resume: if the adapter being resumed carries per-expert LoRA, the model MUST be
    # rebuilt with matching expert LoRA or the saved expert delta is silently dropped.
    # Read it from the resume adapter's config and override the CLI (warn on mismatch).
    if args.resume_from:
        _rc = Path(args.resume_from) / "adapter_config.json"
        if _rc.exists():
            _el = json.loads(_rc.read_text()).get("expert_lora")
            if _el:
                if args.expert_lora_r and args.expert_lora_r != _el["r"]:
                    log("resume_expert_lora_override_warn",
                        cli_r=args.expert_lora_r, adapter_r=_el["r"])
                args.expert_lora_r = _el["r"]
                args.expert_lora_alpha = _el.get("lora_alpha", args.expert_lora_alpha)
                log("resume_expert_lora_from_config", r=args.expert_lora_r, alpha=args.expert_lora_alpha)

    if args.expert_lora_r and lora_mode != "native":
        # Expert LoRA only exists on the native fused-3D NVFP4 path; in peft mode it is
        # silently dropped. Make the opt-in failure loud rather than silent.
        log("expert_lora_ignored_warn", reason=f"lora_mode={lora_mode!r} (expert LoRA needs native)",
            expert_lora_r=args.expert_lora_r)

    # Persist the exact target coverage next to the adapter so every run is
    # auditable: which modules were trained natively, which via PEFT, which
    # were FP8-demoted or skipped.
    coverage["model_type"] = model_type
    coverage["model_dir"] = str(model_dir)
    coverage["train_target"] = args.train_target
    if args.train_target == "both" and vision_suffixes:
        # The `coverage` above is the TEXT half (tower re-excluded, see build_target_inventory's
        # both branch). Record the VISION half separately, restricted to the tower, so the QC
        # artifact reports both halves without either polluting the other.
        from nvfp4_lora.loader import build_target_inventory as _bti
        _vis_view = dict(family)
        _vis_view["_train_target"] = "vision"
        coverage["vision_inventory"] = _bti(model_dir, vision_suffixes, family=_vis_view)
        coverage["vision_targets"] = list(vision_suffixes)
    (Path(args.output_dir) / "target_coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True)
    )
    run_meta = build_run_meta(args, coverage)
    meta_name = "resume_meta.json" if args.resume_from else "run_meta.json"
    (Path(args.output_dir) / meta_name).write_text(json.dumps(run_meta, indent=2, sort_keys=True))
    if args.resume_from:
        original_meta_path = Path(args.output_dir) / "run_meta.json"
        if original_meta_path.exists():
            try:
                original_args = json.loads(original_meta_path.read_text()).get("args", {})
                changed = sorted(
                    k for k, v in run_meta["args"].items()
                    if k != "resume_from" and original_args.get(k) != v
                )
            except Exception:
                changed = ["<unreadable run_meta.json>"]
            if changed:
                log("resume_args_differ", changed=changed)
    for suffix, info in coverage["inventory"].items():
        log("target_coverage", suffix=suffix, counts=info["counts"])

    device = torch.device("cuda")
    dtype = torch.bfloat16

    from nvfp4_lora.mistral_tok import MistralCommonTokenizer, has_tekken
    if has_tekken(args.model_dir):
        # Mistral repacks ship a broken HF tokenizer; use the native tekken one.
        tok = MistralCommonTokenizer(args.model_dir)
        log("tokenizer_loaded", backend="mistral_common",
            vocab_size=tok.vocab_size, pad_id=tok.pad_token_id)
    else:
        tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        log("tokenizer_loaded", backend="hf",
            vocab_size=tok.vocab_size, pad_id=tok.pad_token_id)

    # Vision/both run: build the model's AutoProcessor + the multimodal collate. Text mode
    # never constructs any of this (zero VLM plumbing on the text path).
    collate_fn = (lambda b: collate_batch(b, tok.pad_token_id))
    # Built even under --dry-run for vision/both: the OOM preflight synthesizes a real IMAGE
    # batch through this collator so it exercises the tower + image preproc (a text-only probe
    # would bypass the tower for `both` and crash a vision forward that mandates pixel_values).
    if args.train_target in ("vision", "both"):
        from transformers import AutoProcessor
        from nvfp4_lora.mm_data import (MultimodalCollator, MultimodalJsonlDataset,
                                        resolve_image_token_ids)
        processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
        proc_cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
        if args.max_image_tiles is not None:
            _ip = getattr(processor, "image_processor", None)
            if _ip is not None and hasattr(_ip, "max_num_tiles"):
                _old = _ip.max_num_tiles
                _ip.max_num_tiles = args.max_image_tiles
                log("image_tiles_capped", was=_old, now=args.max_image_tiles)
        image_token_ids = resolve_image_token_ids(processor, proc_cfg)
        proc_tok = getattr(processor, "tokenizer", None)
        pad_id = getattr(proc_tok, "pad_token_id", None)
        if pad_id is None:
            pad_id = tok.pad_token_id
        mm_collate = MultimodalCollator(
            processor, image_token_ids=image_token_ids,
            max_length=args.max_length, pad_token_id=pad_id,
            add_image_flags=bool(family.get("mm_needs_image_flags", False)),
            drop_keys=tuple(family.get("mm_drop_keys", ())))
        collate_fn = mm_collate
        log("processor_loaded", image_token_ids=sorted(image_token_ids), pad_id=pad_id,
            add_image_flags=bool(family.get("mm_needs_image_flags", False)))

    # In --dry-run we synthesize one batch later, so skip dataset/dataloader
    # construction entirely (no train-file needed, no tokenization cost).
    if not args.dry_run:
        if args.train_target in ("vision", "both"):
            train_ds = MultimodalJsonlDataset(args.train_file, max_examples=args.max_train_examples)
            val_ds = (MultimodalJsonlDataset(args.val_file, max_examples=args.max_val_examples)
                      if args.val_file else [])
            if args.train_target == "both":
                # The grad-gate fires on the first IMAGE batch, and the tower half only trains
                # on image rows. A `both` corpus with zero image rows would never exercise -- or
                # verify -- the tower; fail loudly at build, not after an epoch of text-only.
                n_img = sum(1 for paths in train_ds.image_paths if paths)
                if n_img == 0:
                    raise SystemExit(
                        "--train-target both: train set has ZERO image rows; the tower/projector "
                        "LoRA would never receive gradient. Provide image+text rows or use "
                        "--train-target text.")
                log("both_dataset_mix", image_rows=n_img, text_rows=len(train_ds) - n_img)
        else:
            train_ds = ChatJsonlDataset(args.train_file, tok, args.max_length, args.max_train_examples)
            val_ds = (ChatJsonlDataset(args.val_file, tok, args.max_length, args.max_val_examples)
                      if args.val_file else [])
        log("dataset_encoded", train=len(train_ds), val=len(val_ds))

        # Dedicated generator, re-seeded per epoch: order is a pure function of
        # (seed, epoch) so --resume-from can replay it exactly.
        data_gen = torch.Generator()
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  generator=data_gen,
                                  collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_fn)

    log("model_loading_start")
    t0 = time.time()
    # Default expert LoRA alpha to 2*r (so scale=2.0) when unset, matching the common
    # MoE-LoRA convention; only meaningful when --expert-lora-r > 0.
    expert_lora_alpha = args.expert_lora_alpha or (2 * args.expert_lora_r)
    model = load_model(
        model_dir, family, device, dtype,
        lora_mode=lora_mode, target_suffixes=target_suffixes,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        strict=not args.permissive_load,
        expert_lora_r=args.expert_lora_r,
        expert_lora_alpha=expert_lora_alpha,
        expert_lora_dropout=args.expert_lora_dropout,
    )
    log("model_loaded", seconds=round(time.time() - t0, 1))

    # Some trust-remote-code forwards call torch.distributed collectives/get_rank() (e.g.
    # NemotronH-Omni's InternVL-style forward has a rank-gated debug print) which raise if no
    # process group exists. Init a trivial single-process group so single-GPU runs work; a
    # torchrun launch already has one (is_initialized() short-circuits). No-op for other families.
    # A FileStore rendezvous avoids the TCP MASTER_PORT-collision risk of a fixed port.
    if family.get("needs_dist_init") and not torch.distributed.is_initialized():
        _store_path = str(Path(args.output_dir) / ".dist_filestore")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        _store = torch.distributed.FileStore(_store_path, 1)
        torch.distributed.init_process_group(
            backend="gloo", store=_store, world_size=1, rank=0)
        log("dist_init", backend="gloo", world_size=1, rendezvous="filestore")

    # Gated forward-compat hooks for exotic VLM forwards (NemotronH-Omni), so its OWN tested
    # forward runs unchanged -- no ~100-line forward re-implementation, robust to upstream edits.
    # Required for BOTH too: even with a trainable LLM, the embeddings stay frozen, so the
    # in-place scatter of grad-requiring tower features still hits the frozen-leaf-view trap.
    if family.get("_train_target") in ("vision", "both"):
        _compat = []
        # (a) Frozen-backbone scatter: the forward writes image features IN-PLACE into a VIEW of
        # the (frozen) input-embedding output; autograd forbids that on a view of a LEAF with
        # requires_grad=False. Make the embedding output a NON-leaf (o + a grad-requiring 0).
        if family.get("mm_embed_grad_hook"):
            _emb = model.language_model.get_input_embeddings() if hasattr(model, "language_model") \
                else model.get_input_embeddings()
            _emb.register_forward_hook(
                lambda _m, _i, o: o + torch.zeros((), device=o.device,
                                                  dtype=o.dtype).requires_grad_(True))
            _compat.append("embed_nonleaf")
        # (b) Mamba-hybrid output: the omni wrapper packs CausalLMOutputWithPast(
        # past_key_values=outputs.past_key_values), but the LM's NemotronHCausalLMOutput has no
        # such field -> add past_key_values=None to the LM output so the packing succeeds.
        if family.get("mm_lm_output_add_past_kv") and hasattr(model, "language_model"):
            def _add_past_kv(_m, _i, out):
                if not hasattr(out, "past_key_values"):
                    try:
                        out["past_key_values"] = None
                    except Exception:
                        try:
                            object.__setattr__(out, "past_key_values", None)
                        except Exception:
                            pass
                return out
            model.language_model.register_forward_hook(_add_past_kv)
            _compat.append("lm_output_past_kv")
        # (c) Text-only bypass (BOTH only): this VLM's training forward requires pixel_values and
        # runs the tower unconditionally, so a text-only row (no image) crashes. Wrap forward so
        # an IMAGE batch hits the model's OWN tested forward unchanged, while a text-only batch
        # (no pixel_values) runs the language_model directly on token embeddings -- the tower
        # stays out of the graph (no tower grad from text rows), no wasted tower compute.
        if family.get("mm_text_only_bypass") and family.get("_train_target") == "both" \
                and hasattr(model, "language_model"):
            _orig_forward = model.forward
            _lm = model.language_model
            _vocab = _lm.config.vocab_size

            def _both_text_bypass_forward(**kw):
                if kw.get("pixel_values", None) is not None:
                    return _orig_forward(**kw)                    # image row: untouched real forward
                input_ids = kw["input_ids"]
                inputs_embeds = _lm.get_input_embeddings()(input_ids)  # embed_nonleaf hook -> grad
                out = _lm(inputs_embeds=inputs_embeds,
                          attention_mask=kw.get("attention_mask"),
                          use_cache=False, return_dict=True)
                logits = out.logits
                loss = None
                labels = kw.get("labels")
                if labels is not None:
                    sl = logits[..., :-1, :].contiguous().view(-1, _vocab)
                    st = labels[..., 1:].contiguous().view(-1).to(sl.device)
                    loss = torch.nn.functional.cross_entropy(sl, st)  # ignore_index=-100 default
                from transformers.modeling_outputs import CausalLMOutputWithPast
                return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

            model.forward = _both_text_bypass_forward
            _compat.append("text_only_bypass")
        if _compat:
            log("forward_compat_hooks", installed=_compat)

    if lora_mode == "peft":
        model = attach_peft_lora(model, family, target_suffixes,
                                 args.lora_r, args.lora_alpha, args.lora_dropout)

    # Vision/both run: freeze EVERY param then re-enable exactly the LoRA A/B in scope. This
    # is the codex-corrected freeze -- it never wraps the frozen 4-bit LLM forward in
    # torch.no_grad (which would silently zero all gradients), and it catches embeddings /
    # norms / lm_head / router gates without enumerating them. LOAD-BEARING for `both`: the
    # load-time freeze loop (family["freeze"]) froze the freshly-wrapped tower LoRA A/B, and
    # this RE-ENABLES them (plus the LLM LoRA A/B).
    if args.train_target in ("vision", "both"):
        n_vis = freeze_all_then_enable_lora(model)
        log("freeze_enable", trainable_lora_tensors=n_vis, train_target=args.train_target,
            include_projector=args.include_projector)
        if n_vis == 0:
            raise SystemExit(
                f"--train-target {args.train_target} has no trainable LoRA params after "
                "freeze/enable; check the target suffixes and that the Linears fell in scope")
        if args.train_target == "both":
            # Per-HALF trainable assert: a nonzero TOTAL can still hide a zero-wrapped half
            # (e.g. good tower targets but wrong LLM suffixes). Split by the view's vision
            # scope regexes (NOT hand-coded prefixes -- families differ: mistral3 towers are
            # `model.vision_tower.*`, nemotron's `vision_model.*`).
            import re as _re
            _vscope = _re.compile(family.get("_vision_peft_scope") or r"(?!x)x")
            _proj_res = [_re.compile(p) for p in family.get("_projector_scopes", ())]
            def _is_vision_param(name: str) -> bool:
                return bool(_vscope.search(name) or any(pr.search(name) for pr in _proj_res))
            _trainable = [n for n, p in model.named_parameters() if p.requires_grad]
            n_text_half = sum(1 for n in _trainable if not _is_vision_param(n))
            n_vis_half = sum(1 for n in _trainable if _is_vision_param(n))
            log("both_freeze_enable", trainable_text=n_text_half, trainable_vision=n_vis_half)
            if n_text_half == 0:
                raise SystemExit("--train-target both: no trainable TEXT/LLM LoRA params after "
                                 "freeze/enable; check --target-modules")
            if n_vis_half == 0:
                raise SystemExit("--train-target both: no trainable TOWER/projector LoRA params "
                                 "after freeze/enable; check --vision-target-modules / projector")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_lora_modules = sum(1 for _, m in model.named_modules()
                         if isinstance(m, (NVFP4LoRALinear, FP8LoRALinear, BF16LoRALinear)) and m.r > 0)
    log("lora_attached", mode=lora_mode, targets=target_suffixes,
        native_modules=n_lora_modules, trainable=n_train)

    # Expert LoRA can dominate the trainable param count (256 experts x 2 projections);
    # surface its size + estimated AdamW optimizer-state footprint (param+grad+2 moments
    # ~ 14 bytes/param) so a large --expert-lora-r is an informed choice, not a silent OOM.
    expert_blocks = [m for _, m in model.named_modules()
                     if isinstance(m, NVFP4Experts3D) and getattr(m, "lora_r", 0) > 0]
    if expert_blocks:
        n_expert_params = sum(p.numel() for m in expert_blocks
                              for p in (m.lora_A_gate_up, m.lora_B_gate_up, m.lora_A_down, m.lora_B_down))
        log("expert_lora_active", blocks=len(expert_blocks), r=expert_blocks[0].lora_r,
            params=n_expert_params, est_optimizer_state_gb=round(n_expert_params * 14 / 1e9, 2),
            experimental=True, serving="GPU-gated, unproven on this arch (docs/plans/expert_lora_scope.md)")

    if args.fused_linear_ce:
        # Replace lm_head+CE with Liger's chunked fused linear cross-entropy. The
        # full (seq x vocab) logits + its fp32 upcast is the single largest train-time
        # activation spike at long seq (~10 GB at seq 8192, vocab 151552); FLCE never
        # materializes it. Bind ONLY lce_forward onto the causal-LM module: it calls
        # self.model(...) (the backbone, incl. MoE routing and the NVFP4 custom autograd)
        # unchanged and only chunks the head. Do NOT use apply_liger_kernel_to_glm4 — on
        # an instance it rewrites every decoder_layer.mlp into a dense SwiGLU MLP, which
        # corrupts the Glm4MoeNaiveMoe / NVFP4Experts3D blocks.
        from types import MethodType
        from liger_kernel.transformers.model.glm4 import lce_forward as _lce_forward
        ce_target = model.base_model.model if lora_mode == "peft" else model
        if not (hasattr(ce_target, "model") and hasattr(ce_target, "lm_head")):
            raise RuntimeError(
                f"--fused-linear-ce target {type(ce_target).__name__} lacks .model/.lm_head; "
                f"lce_forward binding only supports a causal-LM head."
            )
        ce_target.forward = MethodType(_lce_forward, ce_target)
        log("fused_linear_ce_enabled", target=type(ce_target).__name__)

    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except (ValueError, NotImplementedError, TypeError) as e:
            # Some composite multimodal wrappers (e.g. NemotronH-Omni) don't support GC at the
            # top level. Enable it on the sub-modules that do -- in a vision run the FROZEN LLM
            # backbone dominates the activation memory, and GC still recomputes its activations
            # for the backward that carries gradient into the tower -- and continue without it
            # where unavailable.
            print(f"[load] top-level gradient checkpointing unavailable ({e}); trying sub-modules",
                  flush=True)
            enabled = []
            for _name in ("language_model", "vision_model"):
                _sub = getattr(model, _name, None)
                if _sub is not None and hasattr(_sub, "gradient_checkpointing_enable"):
                    try:
                        _sub.gradient_checkpointing_enable(
                            gradient_checkpointing_kwargs={"use_reentrant": False})
                        enabled.append(_name)
                    except (ValueError, NotImplementedError, TypeError):
                        pass
            print(f"[load] gradient checkpointing enabled on sub-modules: {enabled or 'none'}",
                  flush=True)
    # enable_input_require_grads makes the (frozen) input-embedding OUTPUT require grad so
    # gradient-checkpointed blocks connect. Some VLM forwards then do an in-place scatter of the
    # image features into a reshaped VIEW of that leaf (`inputs_embeds[selected] = ...`, e.g.
    # NemotronH-Omni line ~192), which autograd forbids on a grad-requiring leaf view. In a
    # vision run the grad-requiring input already arrives via the tower's scatter, so the hook
    # is unnecessary there -- skip it when the family opts out.
    if hasattr(model, "enable_input_require_grads") and not family.get("skip_input_require_grads"):
        model.enable_input_require_grads()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.dry_run:
        # Preflight OOM probe. Everything memory-relevant (full load, LoRA attach, gradient
        # checkpointing, optimizer state) is already constructed above; the only thing left is a
        # real forward+backward at the configured shape. We synthesize batches rather than
        # touching the dataset. The peak the REAL run sees is the max over the batch KINDS it
        # will actually train on:
        #   text        -> a max-length text batch (LLM worst case).
        #   vision      -> an IMAGE batch (tower + projector + image preproc). A text-only probe
        #                  is invalid: no text-only rows, and the forward mandates pixel_values.
        #   both        -> BOTH: the text-only bypass path AND an image batch; report the max.
        def _probe(batch, kind):
            torch.cuda.reset_peak_memory_stats()
            batch = _move_batch_to_device(batch, device, compute_dtype=dtype)
            out = model(**batch)
            (out.loss / args.grad_accum).backward()
            optim.zero_grad(set_to_none=True)
            peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            log("dry_run_probe", kind=kind, batch_size=args.batch_size,
                max_length=args.max_length, loss=round(out.loss.item(), 4),
                cuda_max_allocated_gb=peak)
            return peak

        snap_post_load = memory_snapshot()
        peaks = {}
        # Text/LLM worst case: valid for text, and for `both` (text-only rows take the bypass).
        if args.train_target != "vision":
            synth = torch.randint(0, int(tok.vocab_size), (args.batch_size, args.max_length),
                                  dtype=torch.long, device=device)
            peaks["text"] = _probe({"input_ids": synth, "labels": synth.clone(),
                                    "attention_mask": torch.ones_like(synth)}, "text")
        # Tower/image path: exercises the vision tower + projector + on-device image preproc.
        if args.train_target in ("vision", "both"):
            from PIL import Image
            dummy_ex = {
                "messages": [
                    {"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": "describe the image"}]},
                    {"role": "assistant", "content": "ok"}],
                "images": [Image.new("RGB", (512, 512), (127, 127, 127))]}
            peaks["image"] = _probe(mm_collate([dummy_ex]), "image")
        snap_post_backward = memory_snapshot()
        log("dry_run_ok", train_target=args.train_target, peaks_gb=peaks,
            worst_gb=(max(peaks.values()) if peaks else None),
            post_load=snap_post_load, post_backward=snap_post_backward)
        return

    updates_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_updates = max(1, updates_per_epoch * args.epochs)
    if args.max_steps is not None:
        total_updates = min(total_updates, args.max_steps)
    warmup_steps = max(1, int(total_updates * args.warmup_ratio))
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_updates)
    log("optimizer_ready", total_updates=total_updates, warmup_steps=warmup_steps)

    resume_step = 0
    resuming = False
    if args.resume_from:
        resume_dir = Path(args.resume_from)
        _load_adapter_weights(model, resume_dir, lora_mode, log)
        resume_step = _load_train_state(resume_dir, optim, sched, log)
        resuming = resume_step > 0

    model.train()
    update_step = 0
    micro_step = 0
    # First-(image-)backward gradient gate (vision/both): the first backward on an image
    # batch must land non-zero grads on the tower LoRA lora_B, proving the autograd graph
    # survives the frozen 4-bit LLM (catches the no_grad footgun before a wasted run).
    grad_guard_pending = (args.train_target in ("vision", "both"))
    # For `both`, the gate fires on the first IMAGE batch and splits trainable lora_B into
    # text vs tower halves (asymmetric strictness). Build the splitter once, from the view's
    # vision scope regexes (not hand-coded prefixes -- families differ).
    _both_is_vision_param = None
    if args.train_target == "both":
        import re as _re_gg
        _gg_vscope = _re_gg.compile(family.get("_vision_peft_scope") or r"(?!x)x")
        _gg_proj = [_re_gg.compile(p) for p in family.get("_projector_scopes", ())]

        def _both_is_vision_param(name, _v=_gg_vscope, _p=_gg_proj):
            return bool(_v.search(name) or any(pr.search(name) for pr in _p))
    run_start = time.time()
    last_update_time = run_start
    window_supervised_tokens = 0
    window_loss_sum = 0.0
    window_loss_n = 0
    best_val_loss = float("inf")
    best_dir = Path(args.output_dir) / "best"

    both_meta = None
    if args.train_target == "both":
        both_meta = {
            "train_target": "both",
            "text_target_modules": list(target_suffixes),
            "vision_target_modules": list(vision_suffixes),
            "text_peft_scope": family.get("_text_peft_scope"),
            "vision_peft_scope": family.get("_vision_peft_scope"),
            "projector_scopes": list(family.get("_projector_scopes", ())),
            "include_projector": bool(args.include_projector),
            "note": ("both-adapter: contains LLM (text) + tower/projector LoRA keys; split by "
                     "scope at merge/serve time (tower -> merge into base; LLM -> merge or "
                     "runtime-LoRA)."),
        }

    def save_to(dest):
        _save_adapter_atomic(
            model, tok, dest, log,
            lora_mode=lora_mode, base_model_dir=args.model_dir,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, target_suffixes=target_suffixes,
            both_meta=both_meta,
        )

    # Subset-eval loader: when --eval-subset N>0, in-flight evals run over only
    # the first N val examples. Subset losses are NOT comparable to full-eval
    # losses, so they can never directly update best_val_loss (which always
    # tracks FULL values); a subset improvement only triggers a confirming full
    # eval. Built once over a fixed prefix so the cheap eval is reproducible.
    from torch.utils.data import Subset
    subset_n = args.eval_subset if args.eval_subset > 0 else 0
    if subset_n > 0 and len(val_ds) > 0:
        subset_val_loader = DataLoader(
            Subset(val_ds, range(min(subset_n, len(val_ds)))),
            batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn)
    else:
        subset_val_loader = None

    def run_inflight_eval(step):
        """Evaluate at `step`, honoring --eval-subset, and update best tracking.

        Subset evals are logged with subset=N. When a subset eval beats the
        current (full) best, an immediate FULL eval (logged subset="full")
        confirms it; only the full value may update best_val_loss / trigger
        new_best / save best/. Both events are emitted so metrics.jsonl is
        unambiguous about which loss is which.
        """
        nonlocal best_val_loss
        if subset_val_loader is not None:
            sub_loss = evaluate(model, subset_val_loader, device)
            log("eval", step=step, val_loss=round(sub_loss, 4), subset=subset_n)
            if sub_loss >= best_val_loss:
                return
            # Subset says we may have improved; confirm with a full eval. Its
            # value (not the subset's) is what is allowed to set the best.
            full_loss = evaluate(model, val_loader, device)
            log("eval", step=step, val_loss=round(full_loss, 4), subset="full")
        else:
            full_loss = evaluate(model, val_loader, device)
            log("eval", step=step, val_loss=round(full_loss, 4), subset="full")
        if full_loss < best_val_loss:
            prev = best_val_loss
            best_val_loss = full_loss
            log("new_best", step=step, val_loss=round(full_loss, 4),
                prev_best=(round(prev, 4) if prev != float("inf") else None),
                path=str(best_dir))
            save_to(best_dir)

    for epoch in range(args.epochs):
        data_gen.manual_seed(args.seed * 1000 + epoch)
        for batch in train_loader:
            micro_step += 1
            if resuming:
                if micro_step % args.grad_accum == 0:
                    update_step += 1
                    if update_step >= resume_step:
                        resuming = False
                        log("resume_fastforward_done", step=update_step, epoch=epoch)
                continue
            batch = _move_batch_to_device(batch, device, compute_dtype=dtype)
            out = model(**batch)
            if not torch.isfinite(out.loss):
                # A non-finite loss this micro-step means the whole accumulation
                # window is unusable: earlier micro-batches already wrote grads,
                # and stepping on a partial window would bias the update. Drop
                # every accumulated grad and abandon the window (update_step is
                # intentionally NOT incremented).
                log("nonfinite_loss_skipped", step=update_step, micro_step=micro_step,
                    loss=str(out.loss.detach().float().item()))
                optim.zero_grad(set_to_none=True)
                window_supervised_tokens = 0
                window_loss_sum = 0.0
                window_loss_n = 0
                continue
            window_supervised_tokens += int((batch["labels"] != -100).sum().item())
            window_loss_sum += float(out.loss.detach().float().item())
            window_loss_n += 1
            loss = out.loss / args.grad_accum
            loss.backward()
            if grad_guard_pending and (args.train_target != "both" or "pixel_values" in batch):
                # Run on the first (image-containing, for `both`) backward, before any
                # optimizer step / grad-accum divide affects the raw grads. A `both` text-only
                # batch has no tower grad by design, so defer the gate until an image batch.
                assert_vision_grads_flow(model, log, train_target=args.train_target,
                                         is_vision_param=_both_is_vision_param)
                grad_guard_pending = False
            if micro_step % args.grad_accum == 0:
                total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                if not torch.isfinite(total_norm):
                    # Grads went non-finite (overflow / a bad micro-batch that
                    # still produced a finite loss). Skip optim.step/sched.step,
                    # drop the grads, and do NOT increment update_step so the
                    # cosine schedule and step count stay consistent.
                    log("nonfinite_grad_skipped", step=update_step, micro_step=micro_step,
                        grad_norm=str(total_norm.detach().float().item()))
                    optim.zero_grad(set_to_none=True)
                    window_supervised_tokens = 0
                    window_loss_sum = 0.0
                    window_loss_n = 0
                    continue
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                update_step += 1
                now = time.time()
                metrics_extra = build_metrics_row(
                    update_step,
                    total_updates,
                    window_supervised_tokens,
                    now - run_start,
                    now - last_update_time,
                    window_loss_sum / window_loss_n if window_loss_n else None,
                )
                last_update_time = now
                log("train_step", step=update_step, epoch=epoch, loss=round(out.loss.item(), 4),
                    lr=round(sched.get_last_lr()[0], 7),
                    elapsed=round(now - run_start, 1),
                    **metrics_extra)
                window_supervised_tokens = 0
                window_loss_sum = 0.0
                window_loss_n = 0

                if args.eval_every > 0 and update_step % args.eval_every == 0 and len(val_ds) > 0:
                    run_inflight_eval(update_step)

                if args.checkpoint_every > 0 and update_step % args.checkpoint_every == 0:
                    ckpt_dir = Path(args.output_dir) / f"checkpoint_step_{update_step}"
                    log("checkpoint_start", step=update_step, path=str(ckpt_dir))
                    save_to(ckpt_dir)
                    _save_train_state(ckpt_dir, optim, sched, update_step, epoch)
                    _rotate_checkpoints(Path(args.output_dir), keep=2)
                    log("checkpoint_done", step=update_step)

                # Fire the out-of-tree monitor on its cadence, reading the current (just
                # possibly checkpointed) weight-state. Fully crash-isolated by run_step_callback.
                if (step_callback is not None and callback_cadence > 0
                        and update_step % callback_cadence == 0):
                    run_step_callback(step_callback, log, step=update_step, model=model,
                                      tokenizer=tok, output_dir=args.output_dir)

                if args.max_steps is not None and update_step >= args.max_steps:
                    break
        if args.max_steps is not None and update_step >= args.max_steps:
            break

    if len(val_ds) > 0:
        final_val = evaluate(model, val_loader, device)
        log("final_eval", val_loss=round(final_val, 4))
        if final_val < best_val_loss:
            prev = best_val_loss
            best_val_loss = final_val
            log("new_best", step=update_step, val_loss=round(final_val, 4),
                prev_best=(round(prev, 4) if prev != float("inf") else None),
                path=str(best_dir))
            save_to(best_dir)

    log("saving_adapter", path=args.output_dir)
    final_save_ok = _save_with_timeout(save_to, Path(args.output_dir), log,
                                       timeout_s=args.final_save_timeout, label="root")
    if not final_save_ok:
        _exit_after_final_save_timeout(Path(args.output_dir), log)
    log("done",
        total_seconds=round(time.time() - run_start, 1),
        total_updates=update_step,
        best_val_loss=(round(best_val_loss, 4) if best_val_loss != float("inf") else None))
    # Hard-exit after the final log. On GB10/UMA with a large resident MoE model,
    # interpreter + CUDA teardown can spin at ~100% CPU indefinitely (observed on
    # every GLM-4.5-Air expert-LoRA run: best/ saved, "done" never reached or the
    # process never returns). All artifacts are already flushed -- log() closes
    # metrics.jsonl per call and prints with flush=True -- so a hard exit loses
    # nothing and guarantees the process actually terminates.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
