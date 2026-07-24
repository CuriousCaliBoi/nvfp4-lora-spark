"""Single source of truth for per-family model knowledge.

Every component that needs to know how a model family lays out its checkpoint
(the unified trainer, the loader's key translator, the fused-3D MoE replacer,
the checkpoint inspector, and the merge scripts) reads from this registry.
Adding support for a new NVFP4 family should mean adding ONE entry here plus
tests, not editing loader/trainer/merge bodies.

Registry fields, per `config.json` `model_type`:

  auto_class            which transformers Auto* builds the right text-trainable
                        graph: "causal_lm" or "image_text_to_text"
  expert_prefix         (in_memory_prefix, safetensors_prefix) for the text
                        backbone. Used for routed-expert key assembly AND for
                        translating PEFT adapter keys to on-disk base keys at
                        merge time (the two are the same translation).
  peft_scope            regex prefix anchoring PEFT target_modules to the text
                        backbone (so multimodal towers can never match a bare
                        suffix)
  freeze                submodules of model.model to freeze (multimodal towers)
  skip_st_prefixes      safetensors key prefixes that are intentionally never
                        loaded for text-only training (vision tower, projector,
                        MTP speculation layers). This is the explicit allowlist
                        for "tensor on disk, absent in the training graph".
  st_to_model           ordered (st_prefix, model_prefix) rewrite rules mapping
                        safetensors keys to in-memory attribute paths; first
                        match wins, unmatched keys pass through verbatim.
                        None means the layout cannot be expressed statically
                        and the loader's dynamic fallback translator applies
                        (Nemotron: Nano materializes the backbone as
                        `backbone.*` but Super as `model.*` under the SAME
                        model_type, so the prefix is probed from the live
                        module tree)
  meta_allowed_prefixes in-memory parameter-path prefixes allowed to remain on
                        the meta device after loading (text-only training never
                        materializes the multimodal towers). Everything else on
                        meta after load is a load bug and fails the no-meta
                        assertion.
  moe_experts_class     HF module class name of the fused-3D routed-experts
                        block this family uses (replaced by NVFP4Experts3D),
                        or None if the family has no supported fused-3D MoE

Optional VLM (vision-LoRA) fields — present ONLY on families that support
`--train-target vision`; every text-only family and the generic fallback omit
them, so text runs are byte-for-byte unaffected (`_REQUIRED_FAMILY_KEYS`
unchanged). The presence of `vision_peft_scope` is the capability flag: a family
without it refuses `--train-target vision`. `family_view(fam, target)` derives
the EFFECTIVE per-run family from these; the loader/trainer consume that view, so
the toggle is one conditional at the top, not N scattered across loader bodies.

  vision_peft_scope     regex anchoring the LoRA target scope to the vision
                        tower (so a bare q_proj in the tower matches while the
                        text backbone never does)
  vision_target_suffixes default projection suffixes for the tower + projector
  projector_modules     multimodal-connector submodule name(s) that straddle
                        text<->vision; a default vision-mode LoRA target,
                        NEVER a text-mode target, and never left on meta in
                        vision mode
  vision_st_prefixes    on-disk safetensors prefixes of the tower + projector.
                        In vision mode these are SUBTRACTED from skip_st_prefixes
                        (their bf16 weights must load) and drive the coverage
                        inventory's vision restriction.
  vision_st_to_model    extra (st_prefix, model_prefix) rewrite rules that map
                        the tower/projector on-disk keys to their in-memory
                        attribute paths (in text mode the tower is skipped, so
                        the base st_to_model omits them)
  vision_freeze         submodules of model.model that make up the TEXT backbone;
                        frozen in vision mode (the inverse of `freeze`)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

FAMILIES: dict[str, dict] = {
    "qwen3_5_moe": {
        "auto_class": "causal_lm",
        "expert_prefix": ("model.", "model.language_model."),
        "peft_scope": r"^model\.layers\.",
        "freeze": (),
        "skip_st_prefixes": ("model.visual.",),
        "st_to_model": (("model.language_model.", "model."),),
        "meta_allowed_prefixes": (),
        "moe_experts_class": "Qwen3_5MoeExperts",
    },
    "mistral3": {
        "auto_class": "image_text_to_text",
        "expert_prefix": ("model.language_model.", "language_model.model."),
        "peft_scope": r"^model\.language_model\.",
        "freeze": ("vision_tower", "multi_modal_projector"),
        "skip_st_prefixes": ("vision_tower.", "multi_modal_projector."),
        "st_to_model": (
            ("language_model.model.", "model.language_model."),
            ("language_model.lm_head.", "lm_head."),
        ),
        "meta_allowed_prefixes": ("model.vision_tower.", "model.multi_modal_projector."),
        "moe_experts_class": "Mistral4NaiveMoe",
        # --- VLM vision-LoRA (Pixtral tower + multimodal projector; both bf16) ---
        # Vision targets are unquantized bf16, so vision-LoRA rides the existing
        # BF16LoRALinear path (zero new kernels). See family_view() for how these
        # invert the tower's skip/meta/freeze treatment for a vision run.
        "vision_peft_scope": r"^model\.vision_tower\.",
        "vision_target_suffixes": ("q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj",
                                   "linear_1", "linear_2", "merging_layer"),
        "projector_modules": ("multi_modal_projector",),
        "vision_st_prefixes": ("vision_tower.", "multi_modal_projector."),
        "vision_st_to_model": (
            ("vision_tower.", "model.vision_tower."),
            ("multi_modal_projector.", "model.multi_modal_projector."),
        ),
        "vision_freeze": ("language_model",),
    },
    # Nemotron-3 Nano/Super (the original v1.0 family). Routed experts are
    # per-expert NVFP4 nn.Linear modules (no fused-3D container), so
    # expert_prefix and moe_experts_class are None and replace_nvfp4_modules
    # handles everything. Key translation is dynamic (st_to_model=None): the
    # checkpoint stores `backbone.*` but Nano materializes it as `backbone.*`
    # while Super materializes it as `model.*`, decided by the live module
    # tree, so the loader's fallback heuristic does the mapping. `mtp.*`
    # (Multi-Token Prediction speculation layers, serve-only) is skipped
    # there as well.
    "nemotron_h": {
        "auto_class": "causal_lm",
        "expert_prefix": None,
        "peft_scope": r"^(model|backbone)\.layers\.",
        "freeze": (),
        "skip_st_prefixes": ("mtp.",),
        "st_to_model": None,
        "meta_allowed_prefixes": (),
        "moe_experts_class": None,
    },
    # GLM-4.5-Air (106B-A12B). The checkpoint stores routed experts per-expert
    # (model.layers.N.mlp.experts.E.{gate,up,down}_proj), but transformers
    # materializes them as a FUSED-3D block (Glm4MoeNaiveMoe: gate_up_proj +
    # down_proj batched over experts) — structurally identical to
    # Mistral4NaiveMoe. So this is the fused-3D path: replace_moe_experts_with_
    # nvfp4_3d swaps Glm4MoeNaiveMoe -> NVFP4Experts3D and assemble_nvfp4_
    # experts3d_batched gathers the per-expert NVFP4 tensors into the fused
    # buffers (split_gate_up_scales is auto-probed from the shards). Text-only
    # causal LM, so in-memory and on-disk share the `model.` backbone prefix:
    # expert_prefix=("model.","model.") and st_to_model=None routes non-expert
    # weights through the loader's dynamic heuristic (identity translation,
    # safetensors_prefix==model_prefix=="model"). The router gate
    # (mlp.gate.weight, BF16), shared expert and q/k/v attention biases load as
    # frozen non-NVFP4 weights. No MTP/visual tensors to skip.
    "glm4_moe": {
        "auto_class": "causal_lm",
        "expert_prefix": ("model.", "model."),
        "peft_scope": r"^model\.layers\.",
        "freeze": (),
        "skip_st_prefixes": (),
        "st_to_model": None,
        "meta_allowed_prefixes": (),
        "moe_experts_class": "Glm4MoeNaiveMoe",
    },
    # Qwen3 dense (e.g. Qwen3-32B-NVFP4): a plain Qwen3ForCausalLM, no MoE, standard
    # attention. `model.layers.*` on disk and in memory, so st_to_model=None uses the
    # loader's dynamic identity translator; no fused experts (moe_experts_class=None).
    # NVFP4 attention + NVFP4 dense MLP, so q/k/v/o + gate/up/down all train natively.
    "qwen3": {
        "auto_class": "causal_lm",
        "expert_prefix": None,
        "peft_scope": r"^model\.layers\.",
        "freeze": (),
        "skip_st_prefixes": (),
        "st_to_model": None,
        "meta_allowed_prefixes": (),
        "moe_experts_class": None,
    },
    # Llama dense (e.g. Llama-3.1-8B-Instruct-NVFP4): a plain LlamaForCausalLM, no MoE,
    # standard attention. Same on-disk/in-memory `model.layers.*` layout as qwen3 dense,
    # so st_to_model=None uses the identity translator and moe_experts_class=None. NVFP4
    # attention + NVFP4 dense MLP, so q/k/v/o + gate/up/down all train natively.
    "llama": {
        "auto_class": "causal_lm",
        "expert_prefix": None,
        "peft_scope": r"^model\.layers\.",
        "freeze": (),
        "skip_st_prefixes": (),
        "st_to_model": None,
        "meta_allowed_prefixes": (),
        "moe_experts_class": None,
    },
    # Llama-4 Scout (109B-A17B, Llama4ForConditionalGeneration). Early-fusion VLM:
    # a `llama4_vision_model` tower + `multi_modal_projector.linear_1`, both bf16
    # (207 vision Linears in the quant `ignore`), over a per-expert-NVFP4 MoE text
    # backbone whose attention q/k/v/o + router are ALSO bf16. Registered here for
    # the `--train-target vision` capability + contract tests (Phase V1); the
    # single-box GPU proof is deferred (Scout re-rated TIGHT on one Spark). The
    # TEXT side (st_to_model, expert layout) mirrors the mistral3 wrapper and is
    # UNVERIFIED against a live Scout checkpoint — no on-disk fixture exists yet —
    # so confirm it before a text run; the VISION side is what this cycle exercises.
    "llama4": {
        "auto_class": "image_text_to_text",
        # Routed experts assemble from per-expert on-disk keys; the NVFP4Experts3D
        # module path == the on-disk key prefix (identity, no `model.` wrapper), so both
        # halves are "language_model." (the experts live under language_model.model.*).
        "expert_prefix": ("language_model.", "language_model."),
        # Llama4ForConditionalGeneration exposes its submodules at the TOP LEVEL
        # (language_model / vision_model / multi_modal_projector) with NO `model.`
        # wrapper, so the safetensors keys ARE the model param paths -> key mapping is
        # IDENTITY and the scopes carry no `model.` prefix. This differs from mistral3,
        # which wraps everything under `model.`. Verified by a meta-build of the real
        # Llama-4-Scout-109B-A17B-NVFP4 checkpoint (2026-07-05): its params are
        # language_model.model.layers.* / vision_model.model.layers.* /
        # multi_modal_projector.linear_1, matching the on-disk keys exactly.
        "peft_scope": r"^language_model\.model\.",
        "freeze": ("vision_model", "multi_modal_projector"),
        "skip_st_prefixes": ("vision_model.", "multi_modal_projector."),
        "st_to_model": None,
        "meta_allowed_prefixes": ("vision_model.", "multi_modal_projector."),
        # Scout stores routed experts PER-EXPERT on disk
        # (experts.N.{gate,up,down}_proj.*) but the transformers model fuses them into a
        # 3D Llama4TextExperts (gate_up_proj + down_proj), so the loader assembles the
        # per-expert NVFP4 tensors into the fused-3D layout (same path as mistral4/glm4).
        "moe_experts_class": "Llama4TextExperts",
        # --- VLM vision-LoRA (llama4_vision_model tower + projector; both bf16) ---
        "vision_peft_scope": r"^vision_model\.",
        "vision_target_suffixes": ("q_proj", "k_proj", "v_proj", "o_proj",
                                   "fc1", "fc2", "linear_1"),
        "projector_modules": ("multi_modal_projector",),
        # Projector is TOP-LEVEL (no `model.` wrapper) -> anchor the projector scope at
        # `^multi_modal_projector\.`. Without this the scope would be derived from the first
        # vision_st_to_model head (`vision_model.`) and mis-anchor to
        # `^vision_model\.multi_modal_projector\.`, which never matches (silent no-op).
        "vision_projector_mem_prefix": "",
        "vision_st_prefixes": ("vision_model.", "multi_modal_projector."),
        "vision_st_to_model": (
            ("vision_model.", "vision_model."),
            ("multi_modal_projector.", "multi_modal_projector."),
        ),
        "vision_freeze": ("language_model",),
    },
    "NemotronH_Nano_Omni_Reasoning_V3": {
        # Nemotron-3-Nano-Omni-30B-A3B: an OMNI wrapper with TOP-LEVEL submodules
        # (language_model / vision_model / mlp1 / sound_encoder / sound_projection, NO
        # `model.` wrapper). Backbone is a hybrid Mamba2 + MoE with MIXED quant: routed
        # experts NVFP4; Mamba2 in/out_proj + attn o_proj + shared-expert FP8; attn q/k/v +
        # conv1d/A_log/D/dt_bias + vision + mlp1 + sound bf16. Verified by a meta-build of
        # the real checkpoint (2026-07-05): AutoModelForCausalLM builds the full omni tree.
        "auto_class": "causal_lm",  # auto_map registers AutoModel/AutoModelForCausalLM at the
                                    # omni wrapper (has .vision_model/.mlp1); it is NOT stripped
                                    # to the text LM. `image_text_to_text` would fail to resolve.
        # The model declares no Flash-Attention-2 support and defaults to it -> force eager
        # (set recursively on llm/vision/sound sub-configs by the trainer's load path).
        "attn_implementation": "eager",
        # InternVL-style forward: needs an all-ones `image_flags` [num_tiles,1] the processor
        # doesn't emit, and calls torch.distributed.get_rank() (a debug print) which requires a
        # process group -> the trainer builds image_flags in the collator + inits a single-proc
        # group. Both are no-ops for other families.
        "mm_needs_image_flags": True,
        "mm_drop_keys": ("num_patches",),  # processor bookkeeping; not a forward() input (no **kwargs)
        "needs_dist_init": True,  # forward calls torch.distributed.get_rank() for a debug print
        # The forward scatters image features IN-PLACE into a view of the (frozen) input-embedding
        # output. Autograd forbids writing grad-requiring values (the tower-LoRA gradient) into a
        # view of a LEAF with requires_grad=False; upstream InternVL trains the LLM, so ITS
        # embedding output is a non-leaf and never trips this -- our frozen backbone is exactly the
        # config that does. Two gated fixes, WITHOUT re-implementing the model's ~100-line forward:
        #   - `skip_input_require_grads`: skip enable_input_require_grads (it makes the embedding
        #     output a grad-requiring LEAF -> the same forbidden in-place),
        #   - `mm_embed_grad_hook`: a forward hook making the embedding output a NON-leaf
        #     (o + a grad-requiring 0), so the model's own in-place scatter is legal and its tested
        #     forward runs unchanged. Robust to upstream forward changes (no source duplication).
        "skip_input_require_grads": True,
        "mm_embed_grad_hook": True,
        # The omni wrapper packs CausalLMOutputWithPast(past_key_values=outputs.past_key_values),
        # but the Mamba-hybrid LM output (NemotronHCausalLMOutput) has no past_key_values field ->
        # a gated LM-output hook adds past_key_values=None so the packing succeeds.
        "mm_lm_output_add_past_kv": True,
        # The training forward REQUIRES pixel_values + runs the vision tower unconditionally
        # (only generate() handles no-image). A `--train-target both` run interleaves TEXT-ONLY
        # rows, which have no pixel_values -> a gated wrapper (both only) routes text-only
        # batches straight through the language_model, skipping the tower/scatter.
        "mm_text_only_bypass": True,
        # Routed experts materialize as PER-EXPERT nn.Linear (experts.N.{up,down}_proj), not a
        # fused-3D block -> loaded by replace_nvfp4_modules; no 3D assembly, no expert_prefix.
        "expert_prefix": None,
        "moe_experts_class": None,
        "peft_scope": r"^language_model\.",
        "freeze": ("vision_model", "mlp1", "sound_encoder", "sound_projection"),
        # The RADIO patch_generator's `video_embedder` (2-frame temporal embed) is on disk but
        # NOT constructed for image-only use, so its weight has no home -> skip it. It is NOT in
        # vision_st_prefixes, so it stays skipped even in vision mode (where `vision_model.` is
        # un-skipped to load the rest of the tower). sound_* are the audio tower (unused here).
        "skip_st_prefixes": ("vision_model.", "mlp1.", "sound_encoder.", "sound_projection.",
                             "vision_model.radio_model.model.patch_generator.video_embedder."),
        # EXPLICIT identity rewrite (mandatory): the dynamic st_to_model=None heuristic
        # requires exactly ONE top-level candidate and RAISES on this 5-tower checkpoint.
        # Modules are top-level so on-disk keys == in-memory paths (identity).
        "st_to_model": (("language_model.", "language_model."),),
        "meta_allowed_prefixes": ("vision_model.", "mlp1.", "sound_encoder.", "sound_projection."),
        # --- VLM vision-LoRA (RADIO ViT tower + mlp1 projector; both bf16) ---
        "vision_peft_scope": r"^vision_model\.",
        # RADIO blocks use FUSED qkv + proj + mlp.fc1/fc2 (129 Linears under
        # vision_model.radio_model.model.blocks.N.*). `proj` is safe: scoped to vision_model.
        "vision_target_suffixes": ("qkv", "proj", "fc1", "fc2"),
        # Projector `mlp1` is an nn.Sequential(RMSNorm, Linear mlp1.1, SquaredReLU, Linear
        # mlp1.3); the Linears' leaf names ("1"/"3") aren't matchable suffixes, so they are
        # wrapped by PATH via _projector_scopes. Top-level -> projector mem prefix "".
        "projector_modules": ("mlp1",),
        "vision_projector_mem_prefix": "",
        "vision_st_prefixes": ("vision_model.", "mlp1."),
        "vision_st_to_model": (
            ("vision_model.", "vision_model."),
            ("mlp1.", "mlp1."),
        ),
        "vision_freeze": ("language_model",),
    },
}

# These share the exact checkpoint layout of their base entry (qwen3_5_moe_text is the
# text-only config split of qwen3_5_moe; mistral4 is the v4 point release of mistral3),
# so they alias to ONE object rather than a copied literal that could silently drift.
FAMILIES["qwen3_5_moe_text"] = FAMILIES["qwen3_5_moe"]
FAMILIES["mistral4"] = FAMILIES["mistral3"]

# Puzzle-75B-A9B (NVIDIA "Puzzle" NAS block-substitution search over a Nemotron-H hybrid
# Mamba/Attention/MoE backbone). Same on-disk layout as nemotron_h -- backbone.* prefix,
# per-expert (non-fused) NVFP4 Linears, mtp.* skipped, dynamic backbone/model prefix probe
# in the loader's fallback translator (st_to_model=None). Confirmed by direct inspection of
# modeling_nemotron_h_puzzle.py (a thin NemotronHBlock/NemotronHForCausalLM subclass that
# only swaps in a per-layer NemotronHConfig via config.block_configs) and by a meta-build of
# the real checkpoint under transformers 5.8.1 (44,539,891,200 params across 88 heterogeneous
# layers: 40 mamba / 40 moe / 8 attention; hidden_size constant 4096; lm_head untied). The NAS
# heterogeneity (per-MoE-layer moe_intermediate_size in {1280,1536,1792,2048,2688,...},
# num_experts_per_tok in 4..18, latent-space MoE with moe_latent_size=1024) is per-module shape
# variance the loader already handles (in_features/out_features come off the constructed
# nn.Linear, never a global config value) -- it costs nothing here. Alias rather than a copied
# literal so the two layouts cannot silently drift apart.
FAMILIES["nemotron_h_puzzle"] = FAMILIES["nemotron_h"]


def model_type_from_config(model_dir: str | Path) -> str | None:
    """Read `model_type` straight from config.json.

    Deliberately does NOT go through AutoConfig: this works even when the
    installed transformers version does not know the model type, and it keeps
    the inspector and the CPU test suite free of a transformers dependency for
    family resolution.
    """
    cfg_path = Path(model_dir) / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    return cfg.get("model_type")


# Best-effort defaults for an unregistered but structurally-standard NVFP4
# checkpoint: a flat causal-LM whose backbone is `model.layers.*` (or nemotron's
# `backbone.layers.*`) on disk and in memory. st_to_model=None routes non-expert
# weights through the loader's dynamic identity translator; moe_experts_class=None
# means any routed experts are handled per-expert by replace_nvfp4_modules (like
# nemotron_h / qwen3 / llama). This deliberately does NOT cover multimodal-wrapped
# bases (they need an explicit st_to_model rewrite) or unregistered FUSED-3D MoE
# blocks (they need an HF class name); both are caught downstream by the strict-load
# / coverage / no-meta gates, which fail fast rather than train a silent mismatch.
_GENERIC_FAMILY_DEFAULTS: dict = {
    "auto_class": "causal_lm",
    "expert_prefix": None,
    "peft_scope": r"^(model|backbone)\.layers\.",
    "freeze": (),
    "skip_st_prefixes": (),
    "st_to_model": None,
    "meta_allowed_prefixes": (),
    "moe_experts_class": None,
}

# The fields every family entry (registered or user-supplied) must define.
_REQUIRED_FAMILY_KEYS = (
    "auto_class", "expert_prefix", "peft_scope", "freeze", "skip_st_prefixes",
    "st_to_model", "meta_allowed_prefixes", "moe_experts_class",
)


def synthesize_generic_family(model_dir: str | Path) -> dict:
    """Best-effort family for an unregistered flat causal-LM NVFP4 checkpoint.

    Refuses (SystemExit, with a --family-config hint) for multimodal-wrapped
    architectures, whose decoder lives under a `language_model.` prefix and needs an
    explicit st_to_model rewrite the generic identity mapping cannot supply. The
    returned dict is tagged `_unverified` so callers can warn + stamp provenance.
    """
    cfg_path = Path(model_dir) / "config.json"
    cfg = {}
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError):
        pass
    arch = (cfg.get("architectures") or [None])[0] or ""
    if arch.endswith("ForConditionalGeneration"):
        raise SystemExit(
            f"model_type={cfg.get('model_type')!r} is an unregistered multimodal-wrapped "
            f"architecture ({arch}); the generic fallback only handles flat causal-LM "
            f"layouts. Supply an explicit --family-config family.json describing the "
            f"wrapped st_to_model rewrite (e.g. [[\"language_model.model.\", "
            f"\"model.language_model.\"]]) and peft_scope."
        )
    fam = dict(_GENERIC_FAMILY_DEFAULTS)
    fam["_generic"] = True
    fam["_unverified"] = True
    fam["_note"] = (f"synthesized generic family for model_type={cfg.get('model_type')!r} "
                    f"arch={arch!r}; verify with a short train + coverage check")
    return fam


def load_family_config(path: str | Path) -> dict:
    """Load a user-supplied family spec (the --family-config escape hatch).

    Lets a user onboard a model without editing library source. Validates that every
    required field is present and coerces list fields to the tuples consumers expect.
    """
    obj = json.loads(Path(path).read_text())
    missing = [k for k in _REQUIRED_FAMILY_KEYS if k not in obj]
    if missing:
        raise SystemExit(
            f"--family-config {path}: missing required keys {missing}. "
            f"Required: {list(_REQUIRED_FAMILY_KEYS)}."
        )
    for k in ("freeze", "skip_st_prefixes", "meta_allowed_prefixes"):
        obj[k] = tuple(obj[k]) if obj[k] is not None else ()
    if obj["st_to_model"] is not None:
        obj["st_to_model"] = tuple(tuple(r) for r in obj["st_to_model"])
    if obj["expert_prefix"] is not None:
        obj["expert_prefix"] = tuple(obj["expert_prefix"])
    # Optional VLM vision-LoRA fields: absent = a text-only family (back-compatible
    # with every family.json in the wild). Coerce the list fields to tuples when
    # present so family_view / the loader see the same types as a registry entry.
    for k in ("vision_target_suffixes", "projector_modules", "vision_st_prefixes",
              "vision_freeze"):
        if obj.get(k) is not None:
            obj[k] = tuple(obj[k])
    if obj.get("vision_st_to_model") is not None:
        obj["vision_st_to_model"] = tuple(tuple(r) for r in obj["vision_st_to_model"])
    obj["_source"] = str(path)
    return obj


def resolve_family(model_dir: str | Path, *, allow_generic: bool = False,
                   family_config: str | Path | None = None) -> tuple[str, dict]:
    """Map a checkpoint directory to (model_type, family registry entry).

    Resolution order: an explicit `family_config` wins; then the registry; then, only
    if `allow_generic`, a best-effort synthesized family for a flat causal-LM checkpoint
    (tagged `_unverified`). Otherwise raises SystemExit with the porting options. The
    default (allow_generic=False, no family_config) preserves the strict fail-fast.
    """
    model_type = model_type_from_config(model_dir)
    if family_config is not None:
        return (model_type or "custom", load_family_config(family_config))
    fam = FAMILIES.get(model_type)
    if fam is not None:
        return model_type, fam
    if allow_generic:
        return model_type, synthesize_generic_family(model_dir)
    raise SystemExit(
        f"Unsupported model_type={model_type!r}. Known: {sorted(FAMILIES)}. Options: "
        f"add a FAMILIES entry in nvfp4_lora/families.py; pass --family-config family.json; "
        f"or re-run with --allow-unverified-family for a best-effort flat causal-LM mapping "
        f"(guarded by the strict-load / coverage gates). Run "
        f"scripts/inspect_nvfp4_checkpoint.py on the checkpoint first to see its layout."
    )


def family_supports_vision(fam: dict) -> bool:
    """A family supports `--train-target vision` iff it declares a vision scope."""
    return bool(fam.get("vision_peft_scope"))


def _vision_projector_scopes(fam: dict) -> tuple[str, ...]:
    """Anchored regexes matching the projector modules at their in-memory path.

    The projector's in-memory prefix is the family's explicit
    `vision_projector_mem_prefix` when present (use ``""`` for a TOP-LEVEL projector
    module -- llama4's `multi_modal_projector`, nemotron_omni's `mlp1` -- so the
    scope anchors at `^<name>\\.`). Only when that field is absent do we fall back to
    the head of the first `vision_st_to_model` rule -- correct for wrapper-prefixed
    families whose towers live under `model.` (mistral3), WRONG for top-level towers
    (there the first rule head is the *tower* prefix, e.g. `vision_model.`, which
    would mis-anchor the projector to `^vision_model\\.<name>\\.` and never match).
    """
    projector_modules = fam.get("projector_modules", ())
    if not projector_modules:
        return ()
    mem_prefix = fam.get("vision_projector_mem_prefix")
    if mem_prefix is None:
        mem_prefix = "model."
        for _st, _mem in fam.get("vision_st_to_model", ()):  # e.g. ("vision_tower.", "model.vision_tower.")
            head = _mem.split(".", 1)[0]
            if head:
                mem_prefix = head + "."
                break
    return tuple(r"^" + re.escape(mem_prefix + name) + r"\." for name in projector_modules)


def family_view(fam: dict, train_target: str = "text", *, include_projector: bool = True) -> dict:
    """Return the EFFECTIVE per-run family for a `--train-target` value.

    `text` (default) returns `fam` UNCHANGED (identity — the same object), so a
    text run is byte-for-byte today's behaviour and every view-aware consumer takes
    its text branch. `vision` returns a NEW dict that inverts the tower's
    skip/meta/freeze treatment and re-scopes LoRA to the vision tower + projector:

      * `skip_st_prefixes`   := entry value MINUS `vision_st_prefixes` (the tower's
                                bf16 weights must now LOAD, not be skipped)
      * `st_to_model`        := entry rules PLUS `vision_st_to_model` (so the tower
                                on-disk keys translate to their in-memory paths)
      * `meta_allowed_prefixes` := entry value MINUS the tower's in-memory prefixes
                                (the tower MUST be materialized — you cannot LoRA a
                                meta tensor; `assert_no_meta_tensors` then enforces it)
      * `freeze`             := `vision_freeze` (the TEXT backbone submodules)
      * `peft_scope`         := the vision tower scope, ORed with the projector
                                scope when `include_projector` (so the same
                                `replace_bf16_targets` / target-suffix matching that
                                text mode uses now selects the tower + projector)

    `both` returns a NEW dict that LOADS the tower exactly like `vision` (same
    skip/meta/st_to_model inversions) but KEEPS the LLM trainable: `freeze` stays the
    registry value (the non-trained towers) instead of `vision_freeze`, and the text +
    vision LoRA scopes are carried SEPARATELY (`_text_peft_scope` / `_vision_peft_scope`)
    for two paired bf16 passes, with `peft_scope` pinned to the text scope.

    A `_train_target` tag ("vision"/"both") lets the loader's inventory/translator take
    the tower-loading branch. Refuses (SystemExit, porting hint) a family that declares no
    `vision_peft_scope`, mirroring the unsupported-family message style.
    """
    if train_target == "text":
        return fam
    if train_target not in ("vision", "both"):
        raise ValueError(
            f"train_target must be 'text', 'vision', or 'both', got {train_target!r}")
    if not family_supports_vision(fam):
        raise SystemExit(
            f"--train-target {train_target} is not supported for this family "
            f"(model_type has no `vision_peft_scope`). Known vision families: "
            f"{sorted(k for k, v in FAMILIES.items() if family_supports_vision(v))}. "
            f"To port a VLM, add `vision_peft_scope` / `vision_target_suffixes` / "
            f"`projector_modules` / `vision_st_prefixes` / `vision_st_to_model` / "
            f"`vision_freeze` to its FAMILIES entry (see mistral3), or supply them "
            f"via --family-config."
        )
    vision_st_prefixes = tuple(fam.get("vision_st_prefixes", ()))
    # In-memory prefixes of the tower/projector, from the vision_st_to_model targets
    # (e.g. "model.vision_tower.", "model.multi_modal_projector."). These are what
    # meta_allowed_prefixes uses, so subtracting them un-allows a meta tower.
    vision_mem_prefixes = tuple(mem for _st, mem in fam.get("vision_st_to_model", ()))

    base_scope = fam["vision_peft_scope"]
    scopes = [base_scope]
    if include_projector:
        scopes.extend(_vision_projector_scopes(fam))
    vision_scope = "|".join(scopes)

    # BOTH targets (vision, both) LOAD + materialize the tower and translate its on-disk
    # keys -- the divergence is only in what is FROZEN and how LoRA is scoped (below).
    view = dict(fam)
    view["_train_target"] = train_target
    view["_include_projector"] = include_projector
    # Projector scopes stashed for PATH-BASED target selection: a projector may be an
    # `nn.Sequential` whose Linears have non-distinctive leaf names (nemotron_omni's
    # `mlp1.1`/`mlp1.3`), so they can't be matched by `vision_target_suffixes`. The
    # loader wraps every bf16 Linear under these scopes regardless of suffix.
    view["_projector_scopes"] = tuple(_vision_projector_scopes(fam)) if include_projector else ()
    view["skip_st_prefixes"] = tuple(
        p for p in fam.get("skip_st_prefixes", ()) if p not in vision_st_prefixes
    )
    view["meta_allowed_prefixes"] = tuple(
        p for p in fam.get("meta_allowed_prefixes", ()) if p not in vision_mem_prefixes
    )
    base_rules = tuple(fam["st_to_model"]) if fam.get("st_to_model") is not None else ()
    view["st_to_model"] = base_rules + tuple(fam.get("vision_st_to_model", ()))
    view["vision_st_prefixes"] = vision_st_prefixes

    if train_target == "vision":
        # Freeze the TEXT backbone; LoRA only the bf16 tower + projector.
        view["freeze"] = tuple(fam.get("vision_freeze", ()))
        view["peft_scope"] = vision_scope
    else:  # both
        # KEEP the registry freeze (the NON-trained towers, e.g. nemotron's sound tower);
        # the LLM stays trainable. load_model's freeze loop then freezes the freshly-wrapped
        # tower LoRA A/B (they live under those submodules), which freeze_all_then_enable_lora
        # RE-ENABLES -- so that call is LOAD-BEARING for `both`, not a safety belt. Carry the
        # two scopes SEPARATELY for the two paired bf16 passes (text targets under the text
        # scope, tower/projector under the vision scope); pin `peft_scope` to the TEXT scope
        # so any stray consumer (attach_peft_lora, the vision log) behaves text-like.
        view["freeze"] = tuple(fam.get("freeze", ()))
        view["peft_scope"] = fam["peft_scope"]
        view["_text_peft_scope"] = fam["peft_scope"]
        view["_vision_peft_scope"] = vision_scope
    return view


def make_family_translator(fam: dict):
    """Build `translate(safetensors_key) -> model_path | None` from registry data.

    None means "intentionally skipped" (multimodal tower / MTP); unmatched keys
    pass through verbatim (lm_head etc.).
    """
    skip_prefixes = tuple(fam["skip_st_prefixes"])
    rules = tuple(fam["st_to_model"])

    def translate(key: str):
        for skip in skip_prefixes:
            if key.startswith(skip):
                return None
        for st_prefix, model_prefix in rules:
            if key.startswith(st_prefix):
                return model_prefix + key[len(st_prefix):]
        return key

    return translate


def translator_log_prefixes(fam: dict) -> tuple[str, str]:
    """(safetensors_prefix, model_prefix) of the primary rewrite rule, without
    trailing dots, for log lines."""
    st_prefix, model_prefix = fam["st_to_model"][0]
    return st_prefix.rstrip("."), model_prefix.rstrip(".")


def adapter_key_to_base_prefix(akey: str, mem_prefix: str, st_prefix: str,
                               adapter_prefix: str = "base_model.model.") -> tuple[str, str]:
    """Map a PEFT adapter tensor key to (on-disk base module prefix, 'A'|'B').

    The translation is the family's text-backbone prefix swap, i.e. exactly the
    inverse of what the trainer's loader does at load time:

      qwen3_5 (mem "model.", st "model.language_model."):
        base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight
            -> ("model.language_model.layers.3.self_attn.q_proj", "A")
      mistral3/4 (mem "model.language_model.", st "language_model.model."):
        base_model.model.model.language_model.layers.0.mlp.experts.0.gate_proj.lora_B.weight
            -> ("language_model.model.layers.0.mlp.experts.0.gate_proj", "B")

    Keys already carrying the on-disk prefix pass through unchanged.
    """
    import re

    if not akey.startswith(adapter_prefix):
        raise ValueError(f"adapter key {akey!r} does not start with {adapter_prefix!r}")
    tail = akey[len(adapter_prefix):]
    m = re.search(r"\.lora_(?P<side>[AB])\.weight$", tail)
    if m is None:
        raise ValueError(f"adapter key {akey!r} is not a lora_A/lora_B weight")
    prefix = tail[: m.start()]
    if prefix.startswith(st_prefix):
        pass  # already in on-disk form
    elif prefix.startswith(mem_prefix):
        prefix = st_prefix + prefix[len(mem_prefix):]
    else:
        raise ValueError(
            f"adapter key {akey!r} has unrecognized module path {prefix!r}; "
            f"expected it to start with {mem_prefix!r} (in-memory) or "
            f"{st_prefix!r} (on-disk)"
        )
    return prefix, m.group("side")
