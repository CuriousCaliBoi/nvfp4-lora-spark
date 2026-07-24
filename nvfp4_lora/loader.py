"""Loader: walk a Nemotron-3 NVFP4 checkpoint and replace target Linears with NVFP4LoRALinear.

1. Use `accelerate.init_empty_weights()` to construct the model architecture without
   allocating any weight memory.
2. Walk the model and replace each NVFP4-quantized Linear with our `NVFP4LoRALinear`:
   - target_modules (e.g., q_proj, v_proj) → NVFP4LoRALinear with r>0 (LoRA-trainable)
   - other NVFP4 modules → NVFP4LoRALinear with r=0 (frozen)
3. For non-NVFP4 modules (norms, embeddings, conv1d, Mamba state matrices, lm_head),
   load their on-disk bf16 weights normally into the meta-tensors via direct assignment.

This keeps Super-120B resident-memory near the 75 GB NVFP4 storage floor (no bf16 shadow
of any NVFP4-quantized weight).

Identifying NVFP4 vs non-NVFP4 modules:
- A module is NVFP4 iff it has `{module_name}.weight_scale` in the safetensors index.
- The exclude list in `hf_quant_config.json` is a sanity check, not authoritative.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import safetensors

from .linear import BF16LoRALinear, FP8LoRALinear, NVFP4LoRALinear
from .dequant import format_for_record
from . import families


# --------------------------------------------------------------------------------------
# Inventory: which modules are NVFP4-quantized vs plain bf16
# --------------------------------------------------------------------------------------

def _load_weight_map(model_dir: str | Path) -> dict[str, str]:
    idx_path = Path(model_dir) / "model.safetensors.index.json"
    with open(idx_path) as f:
        return json.load(f)["weight_map"]


def list_quantized_modules(model_dir: str | Path) -> set[str]:
    """Return the set of NVFP4-quantized module names by scanning the safetensors index.

    A module is NVFP4 iff it has ModelOpt `.weight` (uint8) + `.weight_scale` tensors,
    OR a compressed-tensors `.weight_packed` tensor.
    Keys are returned VERBATIM from the safetensors index; the caller is responsible
    for translating to model-attribute paths via `make_key_translator`.

    NOTE: FP8 per-tensor modules also carry `.weight` + `.weight_scale`, so they
    are included here (the load path probes the weight dtype and demotes them).
    Use `classify_module_storage` when the distinction matters before load time.
    """
    keys = set(_load_weight_map(model_dir).keys())
    out = set()
    for key in keys:
        if key.endswith(".weight_packed"):
            out.add(key[: -len(".weight_packed")])
        elif key.endswith(".weight"):
            prefix = key[: -len(".weight")]
            # Regular bf16 modules also have `.weight`; only quantized records have `.weight_scale`.
            if f"{prefix}.weight_scale" in keys:
                out.add(prefix)
    return out


def classify_module_storage(keys: set[str], prefix: str) -> str:
    """Classify one module prefix's storage format from index keys alone.

    Returns one of:
      "nvfp4_ct"        compressed-tensors NVFP4 (.weight_packed)
      "nvfp4_modelopt"  ModelOpt NVFP4 (.weight + .weight_scale + .weight_scale_2)
      "fp8"             FP8 per-tensor (.weight + .weight_scale, no .weight_scale_2)
      "bf16"            plain high-precision (.weight only)
      "absent"          no weight tensor under this prefix

    This is index-only and needs no shard reads: ModelOpt NVFP4 always carries
    the fp32 per-tensor `.weight_scale_2`, which FP8 per-tensor records lack.
    """
    if f"{prefix}.weight_packed" in keys:
        return "nvfp4_ct"
    if f"{prefix}.weight" in keys:
        if f"{prefix}.weight_scale" in keys:
            return "nvfp4_modelopt" if f"{prefix}.weight_scale_2" in keys else "fp8"
        return "bf16"
    return "absent"


# is_fp8_module has a single implementation in adapter_keys (torch-free, so the
# pre-flight surfaces -- nybbloris.plan / inspect / doctor -- classify FP8 without
# importing this module). Re-export it here so the loader and the pre-flight share
# ONE definition instead of two copies that can silently drift. classify_module_storage
# above keeps its own fp8 branch (it returns a richer taxonomy); test_fp8_module_agreement
# pins that branch to agree with the re-exported is_fp8_module across all storage classes.
from .adapter_keys import is_fp8_module  # noqa: E402  (re-export; see note above)


def list_weight_module_prefixes(keys: set[str]) -> set[str]:
    """All module prefixes that own a weight tensor (.weight or .weight_packed)."""
    out = set()
    for key in keys:
        if key.endswith(".weight_packed"):
            out.add(key[: -len(".weight_packed")])
        elif key.endswith(".weight"):
            out.add(key[: -len(".weight")])
    return out


_LAYER_RE_TEXT = r"\.layers\.(\d+)\."
# Vision-tower layer index for the `--train-target vision` coverage inventory. Both
# Pixtral (`vision_tower...transformer.layers.N`) and Llama-4 (`vision_model...layers.N`)
# nest their encoder blocks under a `layers.N`; anchoring to the tower prefix keeps the
# projector (which has no `layers.N`) in its own bucket. No leading `.`: the inventory
# runs over on-disk keys, which START with `vision_tower.` / `vision_model.`.
_LAYER_RE_VISION = r"(?:vision_tower|vision_model)\..*?\.layers\.(\d+)\."


def build_target_inventory(model_dir: str | Path, target_suffixes: Sequence[str],
                           family: dict | None = None) -> dict:
    """Full per-module inventory for each requested LoRA target suffix.

    For every suffix, enumerates every module in the safetensors index whose
    last name component equals the suffix and classifies its storage format
    individually. This is what makes partial quantization (the same suffix
    NVFP4 in some layers, BF16/FP8 in others) visible BEFORE load time.

    Modules the resolved family never loads into the training graph (its
    `skip_st_prefixes`: MTP speculation heads, vision towers) are excluded, so
    counts reflect what would actually train. A checkpoint whose family is not
    in the registry is inventoried in full (no skip-list known).

    `family` is the EFFECTIVE per-run view (families.family_view). When it is a
    vision view (`_train_target == "vision"`), the inventory is RESTRICTED to the
    tower + projector (`vision_st_prefixes`) and uses the vision layer regex, so
    the coverage gate reports the tower/projector Linears instead of silently
    counting zero (a text q_proj never matches, but a shared suffix like gate_proj
    would otherwise pick up the frozen NVFP4 text experts). `family=None` (the
    default, every existing caller) preserves the text behaviour byte-for-byte.

    Returns a JSON-able dict:
      {suffix: {"counts": {class: n, ...},
                "examples": {class: [up to 3 module names], ...},
                "layers": {class: sorted layer indices (capped)}}}
    """
    import re

    keys = set(_load_weight_map(model_dir).keys())
    prefixes = list_weight_module_prefixes(keys)
    train_target = family.get("_train_target") if family is not None else None
    vision_mode = train_target == "vision"
    both_mode = train_target == "both"
    if vision_mode:
        # Restrict to the tower + projector on-disk prefixes; everything else
        # (the frozen NVFP4 text backbone) is not a vision target.
        keep = tuple(family.get("vision_st_prefixes", ()))
        prefixes = {p for p in prefixes if keep and p.startswith(keep)}
        layer_re = _LAYER_RE_VISION
    elif both_mode:
        # `both` inventories the TEXT suffixes here. The both view un-skipped the tower for
        # LOADING, but the tower is not a text target: re-exclude vision_st_prefixes so a
        # tower Linear whose leaf name collides with a text suffix (a Pixtral tower's
        # `q_proj`/`down_proj`) cannot inflate the text coverage counts. (The vision half is
        # inventoried separately with the vision suffixes restricted to the tower.)
        skip = tuple(family.get("skip_st_prefixes", ())) + tuple(family.get("vision_st_prefixes", ()))
        if skip:
            prefixes = {p for p in prefixes if not p.startswith(skip)}
        layer_re = _LAYER_RE_TEXT
    else:
        # Drop modules outside the training graph (e.g. nemotron_h `mtp.*`,
        # multimodal towers) so they cannot inflate trainable-target counts.
        if family is not None:
            skip = tuple(family.get("skip_st_prefixes", ()))
        else:
            try:
                _, fam = families.resolve_family(model_dir)
                skip = tuple(fam.get("skip_st_prefixes", ()))
            except (SystemExit, OSError, ValueError):
                # Unknown family / missing or unreadable config.json -> no skip-list;
                # inventory the checkpoint in full (the inspector runs on these too).
                skip = ()
        if skip:
            prefixes = {p for p in prefixes if not p.startswith(skip)}
        layer_re = _LAYER_RE_TEXT
    inventory: dict[str, dict] = {}
    for suffix in target_suffixes:
        matches = sorted(p for p in prefixes if p.rsplit(".", 1)[-1] == suffix)
        counts: dict[str, int] = {}
        examples: dict[str, list[str]] = {}
        layers: dict[str, list[int]] = {}
        for p in matches:
            cls = classify_module_storage(keys, p)
            counts[cls] = counts.get(cls, 0) + 1
            examples.setdefault(cls, [])
            if len(examples[cls]) < 3:
                examples[cls].append(p)
            m = re.search(layer_re, p)
            if m:
                layers.setdefault(cls, [])
                idx = int(m.group(1))
                if idx not in layers[cls]:
                    layers[cls].append(idx)
        inventory[suffix] = {
            "counts": counts,
            "examples": examples,
            "layers": {cls: sorted(v) for cls, v in layers.items()},
        }
    return inventory


def decide_lora_mode(
    model_dir: str | Path,
    target_suffixes: Sequence[str],
    *,
    allow_partial_targets: bool = False,
    family: dict | None = None,
) -> tuple[str, dict]:
    """Decide native-NVFP4 vs PEFT LoRA from a full module inventory, fail-fast.

    A module's runtime form decides which mechanism trains it: NVFP4 -> native
    (NVFP4LoRALinear); FP8 per-tensor -> native (FP8LoRALinear: frozen FP8 base +
    bf16 LoRA) when in a native run, or PEFT (dequant->BF16->wrap) when unified
    with BF16; plain BF16 -> PEFT. So a suffix with NVFP4 is "native" (FP8 joins
    it natively); a suffix with no NVFP4 is "peft" if it also has BF16, else
    native (FP8-only).

    Hard error (SystemExit):
      * a suffix matches no module at all (typo / wrong family)

    NOT an error (changed since the early releases): a suffix that is NVFP4 in some
    layers and plain BF16 in others co-trains both -- quantized via NVFP4LoRALinear,
    in-scope BF16 via BF16LoRALinear -- so `allow_partial_targets` no longer gates
    anything. It is accepted for backward compatibility and recorded in `coverage`.

    Returns (mode, coverage) where mode is "native" or "peft" and coverage is
    a JSON-able report worth persisting next to the adapter.

    `family` is the effective per-run view (families.family_view); when it is a
    vision view the inventory is restricted to the tower + projector. Vision
    targets are unquantized bf16, so this returns mode="peft"; the trainer forces
    the native BF16LoRALinear path for a vision run (see train_nvfp4_lora). The
    default `family=None` is unchanged for every text caller.
    """
    inventory = build_target_inventory(model_dir, target_suffixes, family=family)
    problems: list[str] = []
    tot_nvfp4 = tot_fp8 = tot_bf16 = 0
    bf16_suffixes: list[str] = []
    for suffix in target_suffixes:
        counts = inventory[suffix]["counts"]
        n_nvfp4 = counts.get("nvfp4_ct", 0) + counts.get("nvfp4_modelopt", 0)
        n_fp8 = counts.get("fp8", 0)
        n_bf16 = counts.get("bf16", 0)
        if n_nvfp4 + n_fp8 + n_bf16 == 0:
            problems.append(
                f"target suffix '{suffix}' matches no module in the checkpoint "
                f"index (typo, or this family names its projections differently; "
                f"run scripts/inspect_nvfp4_checkpoint.py to list suffixes)"
            )
            continue
        tot_nvfp4 += n_nvfp4
        tot_fp8 += n_fp8
        tot_bf16 += n_bf16
        if n_bf16:
            bf16_suffixes.append(suffix)

    # NVFP4 / FP8 / BF16 targets ALL train natively now (NVFP4LoRALinear / FP8LoRALinear /
    # BF16LoRALinear), so ANY quantized target makes it a native run and BF16 targets
    # co-train via BF16LoRALinear -- the loader wraps the in-scope ones (peft_scope) and
    # leaves out-of-scope BF16 (an MTP/speculation head, a vision tower) frozen. Only an
    # all-BF16 target set takes the PEFT path. (`bf16_suffixes` retained for coverage.)
    mode = "native" if (tot_nvfp4 > 0 or tot_fp8 > 0) else "peft"
    _ = bf16_suffixes

    coverage = {
        "targets": list(target_suffixes),
        "inventory": inventory,
        "suffix_modes": {s: mode for s in target_suffixes if inventory[s]["counts"]},
        "allow_partial_targets": allow_partial_targets,
        "totals": {"nvfp4": tot_nvfp4, "fp8": tot_fp8, "bf16": tot_bf16},
    }
    if problems:
        raise SystemExit(
            "Target coverage check failed:\n  - " + "\n  - ".join(problems)
        )
    coverage["mode"] = mode
    return mode, coverage


def make_key_translator(model: nn.Module, model_dir: str | Path, family: dict | None = None):
    """Build a function that maps a safetensors key to the corresponding model attribute path.

    Per-family prefix-map architecture (Phase 0.2 redesign). The old single-level
    `named_children()` heuristic fails for both Qwen3.5 (`model.language_model.layers.*`)
    and Mistral3 (`model.language_model.layers.*` — `Mistral3ForConditionalGeneration`
    wraps a `Mistral3Model` which wraps a text-backbone). Instead we dispatch on
    `model.config.model_type` to a per-family explicit translator.

    Skipping logic — each family may want to skip safetensors keys that aren't part of
    the model we're loading (e.g. Nemotron MTP speculation layers, Qwen3.5 vision tower).

    Returns (translate, st_prefix_for_logging, model_prefix_for_logging).
    `translate(key)` returns the model attribute path, or None if the key should be skipped.
    """
    cfg = getattr(model, "config", None)
    model_type = getattr(cfg, "model_type", None)

    # ----- Effective per-run view (families.family_view) -----
    # When the caller threads an explicit family/view carrying a static
    # st_to_model (a `--train-target vision` view adds the tower/projector rewrite
    # rules and un-skips their prefixes), use it verbatim. For a registered text
    # family this view IS the registry entry, so the translator is identical to the
    # registry-lookup path below (zero behaviour change). `family=None` (every
    # pre-vision caller) falls straight through to the registry lookup.
    if family is not None and family.get("st_to_model") is not None:
        translate = families.make_family_translator(family)
        st_prefix, model_prefix = families.translator_log_prefixes(family)
        return translate, st_prefix, model_prefix

    # ----- Registry-driven families (Qwen3.5 MoE, Mistral3/4) -----
    # Skip prefixes and st->model rewrite rules live in nvfp4_lora/families.py
    # so the trainer, inspector and merge scripts all share one translation.
    # NB: `AutoModelForCausalLM.from_config(cfg)` instantiates the text-only
    # causal LM variant of Qwen3.5 whose `config.model_type` is
    # "qwen3_5_moe_text"; the registry carries both names.
    # A registry entry with st_to_model=None (nemotron_h) declares its layout
    # dynamic and falls through to the heuristic below.
    fam = families.FAMILIES.get(model_type)
    if fam is not None and fam["st_to_model"] is not None:
        translate = families.make_family_translator(fam)
        st_prefix, model_prefix = families.translator_log_prefixes(fam)
        return translate, st_prefix, model_prefix

    # ----- Nemotron-H (existing default heuristic — unchanged for backwards compat) -----
    #
    # Nemotron-3 family checkpoints use `backbone.X` in safetensors, but different model classes
    # use different in-memory submodule names:
    #   - Nano-30B-A3B-NVFP4: `self.backbone = NemotronHModel(...)` → in-memory path `backbone.X`
    #   - Super-120B-A12B-NVFP4: `self.model = NemotronHModel(...)` → in-memory path `model.X`
    # Also skips `mtp.X` (Multi-Token Prediction speculation layers vLLM uses for speculative
    # decoding; never trained).
    idx = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())
    safetensors_prefixes = {k.split(".", 1)[0] for k in idx["weight_map"].keys()}
    candidates = [p for p in safetensors_prefixes if p not in ("lm_head", "mtp")]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not determine safetensors submodel prefix; candidates={candidates}. "
            f"If this is a non-Nemotron model, add an explicit per-family branch above "
            f"(model_type={model_type!r})."
        )
    safetensors_prefix = candidates[0]

    model_prefix = None
    for n, c in model.named_children():
        if hasattr(c, "layers"):
            model_prefix = n
            break
    if model_prefix is None:
        raise RuntimeError(
            f"Could not find a child module with `.layers`. If this is a non-Nemotron "
            f"model, add an explicit per-family branch above (model_type={model_type!r})."
        )

    def translate(key: str) -> Optional[str]:
        if key.startswith("mtp."):
            return None
        if key.startswith(safetensors_prefix + "."):
            return model_prefix + "." + key[len(safetensors_prefix) + 1:]
        return key

    return translate, safetensors_prefix, model_prefix


def load_tensor(model_dir: str | Path, key: str, weight_map: dict) -> Optional[torch.Tensor]:
    """Load a single tensor from its shard. Returns None if key not present."""
    if key not in weight_map:
        return None
    shard_path = Path(model_dir) / weight_map[key]
    with safetensors.safe_open(str(shard_path), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


# --------------------------------------------------------------------------------------
# Module replacement
# --------------------------------------------------------------------------------------

def _get_parent(model: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    """Resolve 'a.b.c' → (model.a.b, 'c'). Used to swap a child module on its parent."""
    parts = dotted_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


@dataclass
class _QuantizedLinearRecord:
    name: str
    format: str  # "modelopt" or "compressed_tensors"
    st_name: str
    in_features: int
    out_features: int
    is_lora_target: bool
    weight_key: str
    weight_shape: tuple[int, ...]
    weight_dtype: torch.dtype
    scale_key: str
    scale_shape: tuple[int, ...]
    scale2_key: Optional[str]
    scale2_shape: Optional[tuple[int, ...]]
    bias_key: Optional[str]
    bias_shape: Optional[tuple[int, ...]]


_SAFE_DTYPE_TO_TORCH = {
    "U8": torch.uint8,
    "F8_E4M3": torch.float8_e4m3fn,
    "F32": torch.float32,
    "BF16": torch.bfloat16,
}


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


def _tensor_meta(model_dir: Path, key: str, weight_map: dict) -> tuple[tuple[int, ...], torch.dtype]:
    shard_path = model_dir / weight_map[key]
    with safetensors.safe_open(str(shard_path), framework="pt", device="cpu") as f:
        sl = f.get_slice(key)
        dtype_name = sl.get_dtype()
        if dtype_name not in _SAFE_DTYPE_TO_TORCH:
            raise RuntimeError(f"Unsupported safetensors dtype for {key}: {dtype_name}")
        return tuple(sl.get_shape()), _SAFE_DTYPE_TO_TORCH[dtype_name]


def _tensor_metas(model_dir: Path, keys: Sequence[str], weight_map: dict) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_shard[weight_map[key]].append(key)

    metas: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
    for shard_rel, shard_keys in by_shard.items():
        shard_path = model_dir / shard_rel
        with safetensors.safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in shard_keys:
                sl = f.get_slice(key)
                dtype_name = sl.get_dtype()
                if dtype_name not in _SAFE_DTYPE_TO_TORCH:
                    raise RuntimeError(f"Unsupported safetensors dtype for {key}: {dtype_name}")
                metas[key] = (tuple(sl.get_shape()), _SAFE_DTYPE_TO_TORCH[dtype_name])
    return metas


def _collect_quantized_linear_records(
    model: nn.Module,
    model_dir: Path,
    target_lora_suffixes: Sequence[str],
    family: dict | None = None,
) -> tuple[list[_QuantizedLinearRecord], dict[str, int]]:
    nvfp4_module_names_st = list_quantized_modules(model_dir)
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    keys = set(wm.keys())
    target_set = set(target_lora_suffixes)

    translate, st_prefix, model_prefix = make_key_translator(model, model_dir, family=family)
    model_to_st: dict[str, str] = {}
    for st_name in nvfp4_module_names_st:
        m_name = translate(st_name)
        if m_name is not None:
            model_to_st[m_name] = st_name
    print(f"  detected prefix: safetensors='{st_prefix}.', model='{model_prefix}.'")

    quantized_paths: list[tuple[str, nn.Linear, str, bool]] = []
    needed_keys: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or name not in model_to_st:
            continue
        st_name = model_to_st[name]
        suffix = name.split(".")[-1]
        is_lora_target = suffix in target_set
        fmt = format_for_record(keys, st_name)
        weight_key = f"{st_name}.weight_packed" if fmt == "compressed_tensors" else f"{st_name}.weight"
        scale_key = f"{st_name}.weight_scale"
        scale2_key = (
            f"{st_name}.weight_global_scale"
            if fmt == "compressed_tensors"
            else f"{st_name}.weight_scale_2"
        )
        if weight_key not in wm or scale_key not in wm:
            raise RuntimeError(f"Missing quantization tensors for safetensors module {st_name}")
        quantized_paths.append((name, module, st_name, is_lora_target))
        needed_keys.extend([weight_key, scale_key])
        if scale2_key in wm:
            needed_keys.append(scale2_key)
        if f"{st_name}.bias" in wm:
            needed_keys.append(f"{st_name}.bias")

    metas = _tensor_metas(model_dir, needed_keys, wm)

    records: list[_QuantizedLinearRecord] = []
    counts = {"lora": 0, "frozen_nvfp4": 0, "frozen_fp8": 0}
    for name, module, st_name, is_lora_target in quantized_paths:
        fmt = format_for_record(keys, st_name)
        weight_key = f"{st_name}.weight_packed" if fmt == "compressed_tensors" else f"{st_name}.weight"
        scale_key = f"{st_name}.weight_scale"
        weight_shape, weight_dtype = metas[weight_key]
        scale_shape, _ = metas[scale_key]
        scale2_name = (
            f"{st_name}.weight_global_scale"
            if fmt == "compressed_tensors"
            else f"{st_name}.weight_scale_2"
        )
        scale2_key = scale2_name if scale2_name in wm else None
        scale2_shape = metas[scale2_key][0] if scale2_key is not None else None
        bias_key = f"{st_name}.bias" if f"{st_name}.bias" in wm else None
        bias_shape = metas[bias_key][0] if bias_key is not None else None

        if weight_dtype == torch.uint8:
            if scale2_key is None:
                raise RuntimeError(f"NVFP4 module {st_name} missing {scale2_name.rsplit('.', 1)[-1]}")
            if weight_shape[-1] * 2 != module.in_features:
                raise RuntimeError(
                    f"NVFP4 shape mismatch for {name}: weight is {weight_shape}, "
                    f"expected in_features={module.in_features} -> packed dim {module.in_features // 2}"
                )
            counts["lora" if is_lora_target else "frozen_nvfp4"] += 1
        elif weight_dtype == torch.float8_e4m3fn:
            if is_lora_target:
                counts.setdefault("lora_demoted_fp8", 0)
                counts["lora_demoted_fp8"] += 1
            counts["frozen_fp8"] += 1
        else:
            raise RuntimeError(f"Unknown quantization format for {name}: weight dtype is {weight_dtype}")

        records.append(
            _QuantizedLinearRecord(
                name=name,
                format=fmt,
                st_name=st_name,
                in_features=module.in_features,
                out_features=module.out_features,
                is_lora_target=is_lora_target,
                weight_key=weight_key,
                weight_shape=weight_shape,
                weight_dtype=weight_dtype,
                scale_key=scale_key,
                scale_shape=scale_shape,
                scale2_key=scale2_key,
                scale2_shape=scale2_shape,
                bias_key=bias_key,
                bias_shape=bias_shape,
            )
        )
    return records, counts


def _view_from_pool(pool: torch.Tensor, offset: int, shape: Sequence[int]) -> torch.Tensor:
    return pool.narrow(0, offset, _numel(shape)).view(tuple(shape))


def _copy_safetensors_to_views(model_dir: Path, weight_map: dict, key_to_view: dict[str, torch.Tensor]) -> None:
    by_shard: dict[str, list[tuple[str, torch.Tensor]]] = defaultdict(list)
    for key, view in key_to_view.items():
        by_shard[weight_map[key]].append((key, view))

    for shard_rel, shard_items in by_shard.items():
        shard_path = model_dir / shard_rel
        with safetensors.safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key, view in shard_items:
                view.copy_(f.get_tensor(key))


def _assign_dequant_workspaces(
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[tuple[int, int, torch.dtype], torch.Tensor]:
    """Assign shared dequant workspaces to every NVFP4LoRALinear in the model.

    Pool keyed by `(out_features, in_features, dtype)` per Pre-M1b round-3
    audit: modules with identical shape AND dtype share one workspace buffer.
    Each buffer is allocated with `requires_grad=False`.
    """
    workspace_pool: dict[tuple[int, int, torch.dtype], torch.Tensor] = {}
    for module in model.modules():
        if not isinstance(module, NVFP4LoRALinear):
            continue
        key = (module.out_features, module.in_features, dtype)
        if key not in workspace_pool:
            workspace_pool[key] = torch.empty(
                module.out_features,
                module.in_features,
                dtype=dtype,
                device=device,
                requires_grad=False,
            )
        module.w_bf16_workspace = workspace_pool[key]

    _verify_dequant_workspaces(model)
    return workspace_pool


def _verify_dequant_workspaces(model: nn.Module) -> None:
    """Fail fast if an NVFP4LoRALinear workspace is missing or differentiable."""
    for name, module in model.named_modules():
        if not isinstance(module, NVFP4LoRALinear):
            continue
        if module.w_bf16_workspace is None:
            raise RuntimeError(f"{name} is missing its dequant workspace")
        assert module.w_bf16_workspace.requires_grad is False, f"{name} dequant workspace requires grad"


def replace_nvfp4_modules(
    model: nn.Module,
    model_dir: str | Path,
    target_lora_suffixes: Sequence[str],
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
    family: dict | None = None,
) -> dict[str, int]:
    """Replace every NVFP4 nn.Linear in `model` with an `NVFP4LoRALinear` instance.

    - Modules whose last name component is in `target_lora_suffixes` get LoRA at the
      requested rank (trainable).
    - Other NVFP4 modules get r=0 (frozen-only) but still as NVFP4LoRALinear so the
      dequant happens through our path (not via uninitialized memory).

    `family` is the effective per-run view threaded to `make_key_translator` (a
    `--train-target vision` view un-skips the tower keys). `family=None` is the
    unchanged registry-lookup path.

    Returns: dict with counts {"lora": N_lora, "frozen": N_frozen}.
    """
    model_dir = Path(model_dir)
    nvfp4_module_names_st = list_quantized_modules(model_dir)  # safetensors-side names
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    target_set = set(target_lora_suffixes)

    # Translate safetensors keys -> model attribute paths (handles Nano `backbone.` vs Super `model.`)
    translate, st_prefix, model_prefix = make_key_translator(model, model_dir, family=family)
    # Map model-side path -> safetensors-side name, so we can look up tensors via safetensors keys
    model_to_st: dict[str, str] = {}
    for st_name in nvfp4_module_names_st:
        m_name = translate(st_name)
        if m_name is not None:
            model_to_st[m_name] = st_name
    print(f"  detected prefix: safetensors='{st_prefix}.', model='{model_prefix}.'")

    counts = {"lora": 0, "frozen_nvfp4": 0, "frozen_fp8": 0}

    # Snapshot the list first; mutating the tree during iteration confuses named_modules
    quantized_paths: list[tuple[str, nn.Linear, str]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name not in model_to_st:
            continue
        quantized_paths.append((name, module, model_to_st[name]))

    for name, orig, st_name in quantized_paths:
        suffix = name.split(".")[-1]
        is_lora_target = suffix in target_set

        # Probe the weight dtype to decide quant format: NVFP4 (uint8 packed + fp8 group + fp32 per-tensor)
        # vs FP8 per-tensor (fp8 weight + fp32 scalar scale). Both share `.weight_scale` so we can't
        # tell from the safetensors index alone. Phase 0.1: also detect ModelOpt vs compressed-tensors
        # key naming convention.
        fmt = format_for_record(set(wm.keys()), st_name)
        weight_key = f"{st_name}.weight_packed" if fmt == "compressed_tensors" else f"{st_name}.weight"
        scale2_key = f"{st_name}.weight_global_scale" if fmt == "compressed_tensors" else f"{st_name}.weight_scale_2"
        w_tensor = load_tensor(model_dir, weight_key, wm)
        w_scale = load_tensor(model_dir, f"{st_name}.weight_scale", wm)
        bias = load_tensor(model_dir, f"{st_name}.bias", wm)
        if w_tensor is None or w_scale is None:
            raise RuntimeError(f"Missing quantization tensors for safetensors module {st_name}")

        in_features = orig.in_features
        out_features = orig.out_features
        parent, attr = _get_parent(model, name)

        if w_tensor.dtype == torch.uint8:
            # ---- NVFP4 path ----
            w_scale_2 = load_tensor(model_dir, scale2_key, wm)
            if w_scale_2 is None:
                raise RuntimeError(f"NVFP4 module {st_name} missing {scale2_key.rsplit('.', 1)[-1]}")
            if w_tensor.shape[-1] * 2 != in_features:
                raise RuntimeError(
                    f"NVFP4 shape mismatch for {name}: weight is {tuple(w_tensor.shape)}, "
                    f"expected in_features={in_features} → packed dim {in_features // 2}"
                )
            new_mod = NVFP4LoRALinear(
                in_features=in_features,
                out_features=out_features,
                weight_uint8=w_tensor,
                weight_scale_fp8=w_scale,
                weight_scale_2_fp32=w_scale_2,
                group_size=16,
                bias=bias,
                r=(r if is_lora_target else 0),
                lora_alpha=(lora_alpha if is_lora_target else 0),
                lora_dropout=(lora_dropout if is_lora_target else 0.0),
                device=device,
                dtype=dtype,
                format=fmt,
            )
            setattr(parent, attr, new_mod)
            counts["lora" if is_lora_target else "frozen_nvfp4"] += 1
        elif w_tensor.dtype == torch.float8_e4m3fn:
            # ---- FP8 per-tensor path ----
            # weight is fp8_e4m3fn (out, in); weight_scale is fp32 (scalar or per-channel).
            if is_lora_target and r > 0:
                # Trainable: frozen FP8 base + bf16 LoRA. This is what lets a native run
                # adapt FP8 targets (the 3.6's attention) instead of freezing them.
                new_mod = FP8LoRALinear(
                    in_features=in_features,
                    out_features=out_features,
                    weight_fp8=w_tensor,
                    weight_scale=w_scale,
                    bias=bias,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    device=device,
                    dtype=dtype,
                )
                setattr(parent, attr, new_mod)
                counts.setdefault("lora_fp8", 0)
                counts["lora_fp8"] += 1
                counts["lora"] += 1
            else:
                # Frozen: dequant to a plain bf16 Linear (no LoRA).
                scale = float(w_scale.to(torch.float32).item())
                W = w_tensor.to(torch.float32).mul_(scale).to(dtype=dtype, device=device).contiguous()
                new_mod = nn.Linear(in_features, out_features, bias=(bias is not None), device=device, dtype=dtype)
                with torch.no_grad():
                    new_mod.weight.copy_(W)
                    if bias is not None:
                        new_mod.bias.copy_(bias.to(device=device, dtype=dtype))
                new_mod.weight.requires_grad_(False)
                if bias is not None:
                    new_mod.bias.requires_grad_(False)
                setattr(parent, attr, new_mod)
                counts["frozen_fp8"] += 1
        else:
            raise RuntimeError(
                f"Unknown quantization format for {name}: weight dtype is {w_tensor.dtype}"
            )

    return counts


def replace_bf16_targets(
    model: nn.Module,
    target_lora_suffixes: Sequence[str],
    peft_scope: Optional[str],
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
    projector_scopes: Sequence[str] = (),
) -> int:
    """Wrap plain-BF16 TARGET Linears (under the family's peft_scope) in BF16LoRALinear.

    Call this AFTER the NVFP4/FP8 replacement AND after the BF16 weights are loaded: it lets
    genuinely-BF16 targets (e.g. attention a mixed-precision quantizer left in bf16) co-train
    natively alongside NVFP4/FP8 targets in one adapter, instead of being dropped (a native
    run can't host PEFT). `peft_scope` anchors to the text backbone, so out-of-scope BF16
    (MTP/speculation heads, vision towers) is left frozen. Returns the count wrapped.

    `projector_scopes` (vision runs) select bf16 Linears by PATH regardless of suffix: a
    vision projector may be an `nn.Sequential` whose Linears have non-distinctive leaf names
    (e.g. nemotron_omni's `mlp1.1`/`mlp1.3`) that can't be listed in `target_lora_suffixes`.
    Any bf16 `nn.Linear` under a projector scope is wrapped; matches are de-duplicated with
    the suffix path (a Linear is wrapped at most once).
    """
    import re

    target_set = set(target_lora_suffixes)
    scope_re = re.compile(peft_scope) if peft_scope else None
    proj_res = [re.compile(p) for p in projector_scopes]

    def _is_target(name: str) -> bool:
        # Projector: any Linear under a projector scope, regardless of leaf suffix.
        if any(pr.search(name) for pr in proj_res):
            return True
        # Backbone / tower: suffix must match AND be in the peft scope.
        return (name.rsplit(".", 1)[-1] in target_set
                and (scope_re is None or scope_re.search(name)))

    hits = [(name, mod) for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear) and _is_target(name)]
    count = 0
    for name, orig in hits:
        parent, attr = _get_parent(model, name)
        bias = orig.bias.detach() if orig.bias is not None else None
        setattr(parent, attr, BF16LoRALinear(
            in_features=orig.in_features, out_features=orig.out_features,
            weight=orig.weight.detach(), bias=bias,
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, device=device, dtype=dtype,
        ))
        count += 1
    return count


def replace_nvfp4_modules_pooled(
    model: nn.Module,
    model_dir: str | Path,
    target_lora_suffixes: Sequence[str],
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
    family: dict | None = None,
) -> dict[str, int]:
    """Pooled-storage variant of `replace_nvfp4_modules`.

    This preserves the module/parameter interface but backs the many immutable
    NVFP4 / FP8 buffers and LoRA tensors with a small number of flat CUDA allocations.

    Quantized targets train through the same native LoRALinears as the non-pooled
    path: NVFP4 -> NVFP4LoRALinear; FP8 LoRA targets -> FP8LoRALinear (frozen FP8 base
    kept in fp8 + bf16 LoRA). FP8 modules that are NOT LoRA targets are dequantized to
    a frozen bf16 nn.Linear. Pooling accounts for the different scale layouts: NVFP4
    carries a per-group `weight_scale` (fp8) + a per-tensor `weight_scale_2` (fp32),
    while FP8 carries only a per-tensor scalar or per-output-channel `weight_scale`
    (fp32) and no group structure.
    """
    model_dir = Path(model_dir)
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    records, counts = _collect_quantized_linear_records(
        model, model_dir, target_lora_suffixes, family=family)

    nvfp4_records = [rec for rec in records if rec.weight_dtype == torch.uint8]
    fp8_records = [rec for rec in records if rec.weight_dtype == torch.float8_e4m3fn]
    lora_records = [rec for rec in nvfp4_records if rec.is_lora_target and r > 0]

    # FP8 targets split by whether they train:
    #   * LoRA targets (is_lora_target and r>0) -> FP8LoRALinear: the fp8 base stays
    #     FROZEN AND FP8 (no bf16 shadow), pooled alongside its fp32 weight_scale; the
    #     only trainable params are the pooled bf16 LoRA A/B. This mirrors the non-pooled
    #     FP8 branch in replace_nvfp4_modules and is what makes FP8 LoRA train here too.
    #   * Non-targets (or r==0) -> dequant to a FROZEN bf16 nn.Linear (current behaviour),
    #     pooled into `fp8_weight_pool`.
    # NB: FP8 uses a per-tensor scalar or per-output-channel `weight_scale` and has NO
    # `weight_scale_2` / group structure, so its scale pooling is just flat-by-numel
    # (no group_size accounting); scale shape is preserved via the record's scale_shape.
    fp8_lora_records = [rec for rec in fp8_records if rec.is_lora_target and r > 0]
    fp8_frozen_records = [rec for rec in fp8_records if not (rec.is_lora_target and r > 0)]

    # `_collect_quantized_linear_records` counts every FP8 record as frozen_fp8 (and a
    # lora-target FP8 as lora_demoted_fp8) because it ran before knowing r / the pooled
    # path's FP8-LoRA support. Now FP8 LoRA targets DO train here, so re-derive coherent
    # counts: FP8 LoRA targets join `lora` (+ lora_fp8), only the rest stay frozen_fp8.
    counts["frozen_fp8"] = len(fp8_frozen_records)
    counts.pop("lora_demoted_fp8", None)
    if fp8_lora_records:
        counts["lora"] += len(fp8_lora_records)
        counts["lora_fp8"] = len(fp8_lora_records)

    total_weight_u8 = sum(_numel(rec.weight_shape) for rec in nvfp4_records)
    total_scale_fp8 = sum(_numel(rec.scale_shape) for rec in nvfp4_records)
    total_scale2 = sum(_numel(rec.scale2_shape or ()) for rec in nvfp4_records)
    total_lora_a = sum(r * rec.in_features for rec in lora_records)
    total_lora_b = sum(rec.out_features * r for rec in lora_records)
    total_fp8_weight = sum(_numel(rec.weight_shape) for rec in fp8_frozen_records)
    # FP8-LoRA pools: fp8 base weight kept native, fp32 weight_scale, bf16 LoRA A/B.
    total_fp8l_weight = sum(_numel(rec.weight_shape) for rec in fp8_lora_records)
    total_fp8l_scale = sum(_numel(rec.scale_shape) for rec in fp8_lora_records)
    total_fp8l_lora_a = sum(r * rec.in_features for rec in fp8_lora_records)
    total_fp8l_lora_b = sum(rec.out_features * r for rec in fp8_lora_records)

    print(
        "  pooled loader: allocating "
        f"u8={total_weight_u8/1e9:.2f}G elems, "
        f"fp8_scale={total_scale_fp8/1e9:.2f}G elems, "
        f"fp32_scale2={total_scale2/1e6:.2f}M elems, "
        f"lora={((total_lora_a + total_lora_b) * 2)/1e9:.2f}GB, "
        f"fp8_bf16={total_fp8_weight * 2 / 1e9:.2f}GB, "
        f"fp8_lora_base={total_fp8l_weight / 1e9:.2f}GB, "
        f"fp8_lora_adapter={((total_fp8l_lora_a + total_fp8l_lora_b) * 2)/1e9:.2f}GB"
    )
    weight_pool = torch.empty(total_weight_u8, device=device, dtype=torch.uint8)
    scale_pool = torch.empty(total_scale_fp8, device=device, dtype=torch.float8_e4m3fn)
    scale2_pool = torch.empty(total_scale2, device=device, dtype=torch.float32)
    lora_a_pool = torch.empty(total_lora_a, device=device, dtype=dtype) if total_lora_a else None
    lora_b_pool = torch.empty(total_lora_b, device=device, dtype=dtype) if total_lora_b else None
    fp8_weight_pool = torch.empty(total_fp8_weight, device=device, dtype=dtype) if total_fp8_weight else None
    # FP8-LoRA: keep the base in fp8 (1 byte/elem, no bf16 shadow) + its fp32 scale,
    # and pool the bf16 adapter the same way as the NVFP4 LoRA adapter.
    fp8l_weight_pool = (
        torch.empty(total_fp8l_weight, device=device, dtype=torch.float8_e4m3fn)
        if total_fp8l_weight else None
    )
    fp8l_scale_pool = (
        torch.empty(total_fp8l_scale, device=device, dtype=torch.float32)
        if total_fp8l_scale else None
    )
    fp8l_lora_a_pool = (
        torch.empty(total_fp8l_lora_a, device=device, dtype=dtype) if total_fp8l_lora_a else None
    )
    fp8l_lora_b_pool = (
        torch.empty(total_fp8l_lora_b, device=device, dtype=dtype) if total_fp8l_lora_b else None
    )
    # NVFP4LoRALinear.__init__ intentionally skips Kaiming/zero init when a tensor is
    # supplied via lora_A_tensor/lora_B_tensor (to preserve checkpoint values during
    # a future pooled-path resume). Our pools are torch.empty here, so we MUST apply
    # the standard PEFT init explicitly or the LoRA invariant B@A == 0 breaks at step 0
    # and the first forward computes a random adapter delta. lora_B can be zeroed at
    # the flat-pool level (shape-independent); lora_A's Kaiming is shape-dependent so
    # we apply it per-view inside the construction loop below. Same applies to the
    # FP8-LoRA adapter pools (FP8LoRALinear has the identical supplied-tensor contract).
    if lora_b_pool is not None:
        lora_b_pool.zero_()
    if fp8l_lora_b_pool is not None:
        fp8l_lora_b_pool.zero_()

    offsets = {
        "weight": 0, "scale": 0, "scale2": 0, "lora_a": 0, "lora_b": 0, "fp8": 0,
        "fp8l_weight": 0, "fp8l_scale": 0, "fp8l_lora_a": 0, "fp8l_lora_b": 0,
    }
    views_by_name: dict[str, dict[str, torch.Tensor | None]] = {}
    nvfp4_copy_targets: dict[str, torch.Tensor] = {}
    fp8_lora_set = {id(rec) for rec in fp8_lora_records}
    for rec in records:
        if rec.weight_dtype == torch.float8_e4m3fn and id(rec) in fp8_lora_set:
            # ---- FP8 LoRA target: pool fp8 base + fp32 scale + bf16 adapter ----
            weight_view = _view_from_pool(fp8l_weight_pool, offsets["fp8l_weight"], rec.weight_shape)
            offsets["fp8l_weight"] += _numel(rec.weight_shape)
            scale_view = _view_from_pool(fp8l_scale_pool, offsets["fp8l_scale"], rec.scale_shape)
            offsets["fp8l_scale"] += _numel(rec.scale_shape)
            # Copy fp8 base weight + fp32 scale straight from the shards (no dequant).
            nvfp4_copy_targets[rec.weight_key] = weight_view
            nvfp4_copy_targets[rec.scale_key] = scale_view

            lora_a_view = _view_from_pool(fp8l_lora_a_pool, offsets["fp8l_lora_a"], (r, rec.in_features))
            offsets["fp8l_lora_a"] += r * rec.in_features
            nn.init.kaiming_uniform_(lora_a_view, a=math.sqrt(5))
            lora_b_view = _view_from_pool(fp8l_lora_b_pool, offsets["fp8l_lora_b"], (rec.out_features, r))
            offsets["fp8l_lora_b"] += rec.out_features * r

            views_by_name[rec.name] = {
                "weight": weight_view,
                "scale": scale_view,
                "scale2": None,
                "lora_a": lora_a_view,
                "lora_b": lora_b_view,
            }
        elif rec.weight_dtype == torch.uint8:
            weight_view = _view_from_pool(weight_pool, offsets["weight"], rec.weight_shape)
            offsets["weight"] += _numel(rec.weight_shape)
            scale_view = _view_from_pool(scale_pool, offsets["scale"], rec.scale_shape)
            offsets["scale"] += _numel(rec.scale_shape)
            scale2_view = _view_from_pool(scale2_pool, offsets["scale2"], rec.scale2_shape or ())
            offsets["scale2"] += _numel(rec.scale2_shape or ())

            nvfp4_copy_targets[rec.weight_key] = weight_view
            nvfp4_copy_targets[rec.scale_key] = scale_view
            # Local invariant assert: the uint8 record path guards scale2_key non-None
            # ~240 lines above (in _collect_quantized_linear_records). Restate it here
            # so a future refactor that relaxes that guard fails fast instead of
            # silently inserting a None key into nvfp4_copy_targets.
            assert rec.scale2_key is not None, "NVFP4 uint8 record missing scale2_key"
            nvfp4_copy_targets[rec.scale2_key] = scale2_view

            lora_a_view = None
            lora_b_view = None
            if rec.is_lora_target and r > 0:
                lora_a_view = _view_from_pool(lora_a_pool, offsets["lora_a"], (r, rec.in_features))
                offsets["lora_a"] += r * rec.in_features
                # In-place Kaiming init on the view modifies the underlying pool storage
                # at this slice; standard PEFT init for LoRA A. lora_B was zeroed at pool
                # allocation time, so no per-view init needed for B.
                nn.init.kaiming_uniform_(lora_a_view, a=math.sqrt(5))
                lora_b_view = _view_from_pool(lora_b_pool, offsets["lora_b"], (rec.out_features, r))
                offsets["lora_b"] += rec.out_features * r

            views_by_name[rec.name] = {
                "weight": weight_view,
                "scale": scale_view,
                "scale2": scale2_view,
                "lora_a": lora_a_view,
                "lora_b": lora_b_view,
            }

    if nvfp4_copy_targets:
        print(f"  pooled loader: copying {len(nvfp4_copy_targets)} NVFP4 tensors by shard")
        _copy_safetensors_to_views(model_dir, wm, nvfp4_copy_targets)

    for rec in records:
        parent, attr = _get_parent(model, rec.name)
        bias = load_tensor(model_dir, rec.bias_key, wm) if rec.bias_key is not None else None

        if rec.weight_dtype == torch.uint8:
            views = views_by_name[rec.name]
            new_mod = NVFP4LoRALinear(
                in_features=rec.in_features,
                out_features=rec.out_features,
                weight_uint8=views["weight"],
                weight_scale_fp8=views["scale"],
                weight_scale_2_fp32=views["scale2"],
                group_size=16,
                bias=bias,
                r=(r if rec.is_lora_target else 0),
                lora_alpha=(lora_alpha if rec.is_lora_target else 0),
                lora_dropout=(lora_dropout if rec.is_lora_target else 0.0),
                device=device,
                dtype=dtype,
                copy_base_tensors=False,
                lora_A_tensor=views["lora_a"],
                lora_B_tensor=views["lora_b"],
            )
            setattr(parent, attr, new_mod)
        elif id(rec) in fp8_lora_set:
            # ---- FP8 LoRA target: frozen FP8 base (pooled, native fp8) + bf16 LoRA ----
            # Same trainable form as the non-pooled replace_nvfp4_modules FP8 branch:
            # FP8LoRALinear with copy_base_tensors=False over the pooled fp8 weight +
            # fp32 scale views, and the pooled bf16 adapter (already Kaiming-A / zero-B).
            views = views_by_name[rec.name]
            new_mod = FP8LoRALinear(
                in_features=rec.in_features,
                out_features=rec.out_features,
                weight_fp8=views["weight"],
                weight_scale=views["scale"],
                bias=bias,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                device=device,
                dtype=dtype,
                copy_base_tensors=False,
                lora_A_tensor=views["lora_a"],
                lora_B_tensor=views["lora_b"],
            )
            setattr(parent, attr, new_mod)
        else:
            # ---- Frozen FP8: dequant to a frozen bf16 nn.Linear (pooled bf16) ----
            weight_view = _view_from_pool(fp8_weight_pool, offsets["fp8"], rec.weight_shape)
            offsets["fp8"] += _numel(rec.weight_shape)
            w_tensor = load_tensor(model_dir, rec.weight_key, wm)
            w_scale = load_tensor(model_dir, rec.scale_key, wm)
            scale = float(w_scale.to(torch.float32).item())
            weight_view.copy_(w_tensor.to(torch.float32).mul_(scale).to(dtype=dtype))

            new_mod = nn.Linear(
                rec.in_features, rec.out_features, bias=(bias is not None), device="meta", dtype=dtype
            )
            new_mod.weight = nn.Parameter(weight_view, requires_grad=False)
            if bias is not None:
                new_mod.bias = nn.Parameter(bias.to(device=device, dtype=dtype), requires_grad=False)
            setattr(parent, attr, new_mod)

    return counts


# --------------------------------------------------------------------------------------
# Non-NVFP4 weight loading (norms, embeddings, conv1d, Mamba SSM state, lm_head)
# --------------------------------------------------------------------------------------

def load_non_nvfp4_weights(
    model: nn.Module,
    model_dir: str | Path,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
    strict: bool = True,
    family: dict | None = None,
) -> int:
    """Load all on-disk tensors that are NOT part of an NVFP4 module's storage.

    Skips:
    - `.weight`, `.weight_scale`, `.weight_scale_2`, `.input_scale`, `.bias` of any module
      already replaced with NVFP4LoRALinear (those buffers were already set during replacement)
    - tensors that are themselves NVFP4 metadata (scale/scale_2/input_scale)
    - keys the family translator marks as intentionally absent (vision tower, MTP)

    strict=True (default): any on-disk tensor that maps to a missing model path,
    or that fails to assign, is collected and raised as ONE RuntimeError after
    the walk (models built under init_empty_weights would otherwise defer the
    failure to first forward as an opaque meta-tensor error). strict=False
    restores the old warn-and-continue behavior for bring-up of new families.

    Returns: number of tensors loaded.
    """
    model_dir = Path(model_dir)
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]

    # Translate safetensors keys -> model attribute paths. `family` (the effective
    # per-run view) makes a vision run load + materialize the tower/projector bf16
    # weights that text mode skips; None keeps the registry-lookup path unchanged.
    translate, st_prefix, model_prefix = make_key_translator(model, model_dir, family=family)

    # Which modules in the model were swapped in by `replace_nvfp4_modules` (so their on-disk
    # NVFP4 / FP8 storage tensors must NOT be re-loaded here). Both NVFP4LoRALinear and the
    # frozen bf16 nn.Linear replacements set their weight at module-replacement time.
    replaced_module_names: set[str] = set()
    for name, m in model.named_modules():
        if isinstance(m, NVFP4LoRALinear):
            replaced_module_names.add(name)
    # Add FP8-replaced modules: detect via the safetensors index - anything whose key prefix
    # matches a quantized-module name AND has been swapped to a different class than expected.
    # Simpler: any module whose name appears in the quantized-module list has its weight
    # already handled (either by NVFP4LoRALinear above or by the FP8 dequant-to-bf16 path).
    nvfp4_module_names_st = list_quantized_modules(model_dir)
    for st_name in nvfp4_module_names_st:
        m_name = translate(st_name)
        if m_name is not None:
            replaced_module_names.add(m_name)
    nvfp4_replaced = replaced_module_names

    n_loaded = 0
    n_skipped_by_translator = 0
    missing_paths: list[str] = []
    failed_assignments: list[str] = []
    # Walk tensors in the safetensors index
    for key, shard_rel in wm.items():
        # Translate safetensors-side key to model-side path. None means the family
        # registry intentionally skips it (vision tower, projector, MTP).
        model_key = translate(key)
        if model_key is None:
            n_skipped_by_translator += 1
            continue

        # Identify the module name for this tensor (everything before the last dot, on model side)
        module_name, tensor_attr = model_key.rsplit(".", 1)

        # Skip NVFP4 storage tensors for replaced modules (already loaded into NVFP4LoRALinear buffers)
        if module_name in nvfp4_replaced and tensor_attr in (
            "weight", "weight_packed", "weight_scale", "weight_scale_2",
            "weight_global_scale", "input_scale", "input_global_scale", "bias"
        ):
            continue

        # Navigate to the parent module and find the attribute (using model-side path)
        try:
            parent, attr = _get_parent(model, model_key)
        except AttributeError:
            missing_paths.append(f"{model_key} (from safetensors key {key})")
            if not strict:
                print(f"  WARN: path not found in model: {model_key} (from safetensors key {key})")
            continue

        # For non-NVFP4 modules: load the tensor and assign to the model
        shard_path = model_dir / shard_rel
        with safetensors.safe_open(str(shard_path), framework="pt", device="cpu") as f:
            tensor = f.get_tensor(key)

        # Determine dtype to use:
        # - bf16 for general weights / norms / embeddings
        # - keep original dtype for special params (e.g., A_log, D, dt_bias for Mamba; they're often float32)
        target_dtype = dtype if tensor.dtype.is_floating_point and tensor.dtype != torch.float8_e4m3fn else tensor.dtype
        try:
            # Replace meta-tensor with real one
            t = tensor.to(device=device, dtype=target_dtype).contiguous()
            # nn.Parameter wrapping if the existing slot is a Parameter
            if isinstance(getattr(parent, attr, None), nn.Parameter):
                setattr(parent, attr, nn.Parameter(t, requires_grad=False))
            else:
                # Buffer or raw tensor
                # Use module method if available; otherwise direct setattr
                if hasattr(parent, "register_buffer") and attr in dict(parent.named_buffers(recurse=False)):
                    parent._buffers[attr] = t
                else:
                    setattr(parent, attr, t)
        except Exception as e:
            failed_assignments.append(f"{key}: {type(e).__name__}: {e}")
            if not strict:
                print(f"  WARN: failed to load {key}: {type(e).__name__}: {e}")
            continue

        n_loaded += 1
    if n_skipped_by_translator > 0:
        print(
            f"  skipped {n_skipped_by_translator} tensors via the family skip-list "
            f"(multimodal towers / MTP; intentionally not loaded)"
        )
    if (missing_paths or failed_assignments) and strict:
        sample = (missing_paths + failed_assignments)[:10]
        raise RuntimeError(
            f"strict load failed: {len(missing_paths)} on-disk tensor(s) map to "
            f"paths missing from the model and {len(failed_assignments)} failed "
            f"to assign. First {len(sample)}:\n  " + "\n  ".join(sample) + "\n"
            f"If these tensors are intentionally absent from the training graph "
            f"(a new multimodal tower or speculation head), add their prefix to "
            f"the family's skip_st_prefixes in nvfp4_lora/families.py. Use "
            f"strict=False / --permissive-load only for bring-up."
        )
    return n_loaded


def assert_no_meta_tensors(model: nn.Module, allowed_prefixes: Sequence[str] = ()) -> None:
    """Fail if any parameter or buffer is still on the meta device after loading.

    Models are built under init_empty_weights, so a tensor the loader never
    reached stays on meta and only explodes at first forward (or worse, at
    save time). Everything left on meta must be covered by an explicit
    `allowed_prefixes` entry (e.g. a frozen vision tower that text-only
    training never materializes).
    """
    allowed = tuple(allowed_prefixes)
    offenders: list[str] = []
    for name, p in model.named_parameters():
        if p.is_meta and not name.startswith(allowed):
            offenders.append(f"param {name}")
    for name, b in model.named_buffers():
        if b.is_meta and not name.startswith(allowed):
            offenders.append(f"buffer {name}")
    if offenders:
        raise RuntimeError(
            f"{len(offenders)} tensor(s) are still on the meta device after "
            f"loading (first {min(10, len(offenders))}):\n  "
            + "\n  ".join(offenders[:10])
            + "\nThese were never loaded from the checkpoint. If they are "
            f"intentionally unmaterialized (frozen multimodal tower), add the "
            f"prefix to the family's meta_allowed_prefixes in "
            f"nvfp4_lora/families.py; otherwise this is a load bug."
        )


# --------------------------------------------------------------------------------------
# Top-level entry
# --------------------------------------------------------------------------------------

def ensure_recursive_remote_code_imports() -> None:
    """Work around a transformers dynamic-module cache gap for multi-level custom code.

    `get_cached_module_file` stages a trust_remote_code modeling file plus the module
    names returned by `check_imports` -- which are the DIRECT relative imports only. But
    `get_class_in_module` then resolves the import graph RECURSIVELY. A 2-level custom-code
    import chain therefore leaves a transitively-imported sibling un-staged and raises
    `FileNotFoundError` at import time. Puzzle is the first onboarded checkpoint that trips
    this: `modeling_nemotron_h_puzzle` imports `modeling_nemotron_h` and
    `configuration_nemotron_h_puzzle`, both of which import `configuration_nemotron_h`, so
    the base config file is never copied into the modeling cache dir (verified on
    transformers 5.8.1; the base nemotron_h Nano/Super checkpoints only have a 1-level chain
    and are unaffected).

    Patch `check_imports` to return the FULL recursive closure of relative-import module
    basenames. This is a strict superset of the current return value, so it is a no-op for
    single-level custom code and for built-in models, and it preserves the original
    missing-pip-package check by delegating to it first. Assumes the flat single-directory
    layout the Hub enforces for remote code (all relative imports live beside the modeling
    file), which is exactly Puzzle's layout. Idempotent.
    """
    import os
    import transformers.dynamic_module_utils as _dmu

    if getattr(_dmu.check_imports, "_nvfp4_recursive", False):
        return
    _orig_check_imports = _dmu.check_imports

    def _recursive_check_imports(filename):
        direct = _orig_check_imports(filename)  # keeps the missing-package check + its raise
        try:
            files = _dmu.get_relative_import_files(filename)  # recursive absolute paths
        except Exception:
            return direct
        stems = [os.path.splitext(os.path.basename(f))[0] for f in files]
        return list(dict.fromkeys(list(direct) + stems))

    _recursive_check_imports._nvfp4_recursive = True
    _dmu.check_imports = _recursive_check_imports


def load_nemotron_with_nvfp4_lora(
    model_dir: str | Path,
    target_lora_suffixes: Sequence[str] = ("q_proj", "v_proj"),
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    pooled_loader_buffers: bool = False,
    strict: bool = True,
) -> nn.Module:
    """Load a Nemotron-3 NVFP4 checkpoint with NVFP4LoRALinear modules at target paths.

    Steps:
    1. Build model architecture via `init_empty_weights` (no allocations).
    2. Replace target NVFP4 Linears with `NVFP4LoRALinear` (LoRA-trainable on `target_lora_suffixes`,
       frozen elsewhere).
    3. Load all non-NVFP4 weights (embeddings, norms, Mamba state matrices, conv1d, lm_head).
    4. Verify no parameter or buffer is left on the meta device (strict=True).

    Returns the assembled model on `device` with the requested dtype.

    Caller is responsible for setting `model.train()` / `model.eval()` and for collecting
    `[p for p in model.parameters() if p.requires_grad]` for the optimizer.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    device = torch.device(device)
    model_dir = Path(model_dir)

    ensure_recursive_remote_code_imports()  # multi-level trust_remote_code chains (Puzzle)
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=dtype)

    print(f"=== loader: replacing NVFP4 Linears (LoRA targets: {list(target_lora_suffixes)}) ===")
    replace_fn = replace_nvfp4_modules_pooled if pooled_loader_buffers else replace_nvfp4_modules
    if pooled_loader_buffers:
        print("  pooled loader buffers: enabled")
    counts = replace_fn(
        model, model_dir, target_lora_suffixes,
        r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        device=device, dtype=dtype,
    )
    msg = (
        f"  replaced: {counts['lora']} LoRA-target NVFP4 + {counts.get('frozen_nvfp4', 0)} "
        f"frozen-NVFP4 + {counts.get('frozen_fp8', 0)} frozen-FP8 modules"
    )
    if counts.get("lora_demoted_fp8", 0):
        msg += f" ({counts['lora_demoted_fp8']} LoRA targets demoted to frozen on FP8 path)"
    print(msg)
    if r > 0 and counts["lora"] == 0:
        raise SystemExit(
            "No NVFP4 LoRA target modules were installed. "
            f"Tried suffixes: {list(target_lora_suffixes)}. "
            "Check target_lora_suffixes for typos or unsupported target modules."
        )

    workspace_pool = _assign_dequant_workspaces(model, device=device, dtype=dtype)
    print(f"  dequant workspace pool: {len(workspace_pool)} shared buffers")

    print("=== loader: loading non-NVFP4 weights (norms, embeddings, Mamba, conv1d, lm_head) ===")
    n = load_non_nvfp4_weights(model, model_dir, device=device, dtype=dtype, strict=strict)
    print(f"  loaded {n} non-NVFP4 tensors")

    # Tied embeddings never appear as a separate lm_head tensor on disk; re-tie
    # so lm_head does not stay on meta. No-op for untied checkpoints.
    if getattr(config, "tie_word_embeddings", False) and hasattr(model, "tie_weights"):
        model.tie_weights()

    if strict:
        # `mtp.` is allowlisted defensively: the translator already skips MTP
        # tensors on disk, and the in-memory causal-LM graph normally has no
        # mtp submodule at all.
        assert_no_meta_tensors(model, allowed_prefixes=("mtp.",))

    return model
