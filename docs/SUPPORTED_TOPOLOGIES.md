# Supported NVFP4 checkpoint topologies (v1)

"Any NVFP4 model" is the goal; this document is the precise contract of what
the stack supports TODAY, so that a checkpoint outside the contract fails
early with a named assumption instead of producing a silently-partial run.
Run [`scripts/inspect_nvfp4_checkpoint.py`](../scripts/inspect_nvfp4_checkpoint.py)
on any checkpoint to evaluate it against this contract.

## Quantized linear storage

Two on-disk encodings of NVFP4 are supported, detected per module from the
safetensors index:

| | compressed-tensors (RedHatAI) | NVIDIA ModelOpt |
|---|---|---|
| packed weight | `<m>.weight_packed`, uint8 `(out, in/2)` | `<m>.weight`, uint8 `(out, in/2)` |
| group scale | `<m>.weight_scale`, fp8_e4m3fn `(out, in/16)` | `<m>.weight_scale`, fp8_e4m3fn `(out, in/16)` |
| per-tensor scale | `<m>.weight_global_scale`, fp32, stored as a divisor | `<m>.weight_scale_2`, fp32 |

Common requirements for both:

* group size 16, last-dimension grouping
* standard E2M1 nibble packing, low-nibble-first
* `in_features` even and divisible by 16

FP8 per-tensor modules (`.weight` in fp8_e4m3fn + scalar `.weight_scale`,
no `.weight_scale_2`) train natively via `FP8LoRALinear` (frozen FP8 base +
bf16 LoRA), so targeting them is a native run with no extra flag.

## Model families

Family knowledge (safetensors-to-in-memory key translation, PEFT scoping,
multimodal-tower skip lists, fused-MoE class names) lives in ONE registry:
[`nvfp4_lora/families.py`](../nvfp4_lora/families.py). Current entries:

| `model_type` | auto class | attention | routed experts | status |
|---|---|---|---|---|
| `qwen3_5_moe` / `qwen3_5_moe_text` | causal LM | NVFP4 q/k/v/o on full-attention layers (native LoRA); GatedDeltaNet linear-attention layers BF16 | per-expert CT keys, fused-3D in memory (`Qwen3_5MoeExperts`) | trained + runtime-LoRA served end-to-end (logprob-delta confirmed). NB: the 122B checkpoint is a vision-wrapped `...ForConditionalGeneration`, so the expert/attn adapter must be re-keyed to the `language_model.model.layers.N` path (`rekey_expert_lora_for_vllm.py --wrapped auto`); see `cross_arch_status.md` FINDING #3 |
| `mistral3` / `mistral4` | image-text-to-text (vision tower frozen + unmaterialized) | MLA attention BF16 (PEFT LoRA) | per-expert CT keys, fused-3D in memory (`Mistral4NaiveMoe`) | trained end-to-end |
| `nemotron_h` (Nemotron-3 Nano/Super) | causal LM | BF16/FP8 (not LoRA-targeted) | per-expert ModelOpt keys, per-expert in memory (no fused-3D container; `st_to_model`/`expert_prefix`/`moe_experts_class` are None and the loader's dynamic prefix heuristic applies, since Nano materializes `backbone.*` but Super `model.*`) | unified trainer (validated: Nano dry-run + 3-step smoke); `train/*.py` remain the frozen v1.0 measurement-run path |
| `nemotron_h_puzzle` (Puzzle-75B-A9B) | causal LM | FP8 per-channel on q/k/v/o_proj and `lm_head` (LoRA-targetable via `FP8LoRALinear`) | per-expert ModelOpt keys, per-expert in memory; NAS-heterogeneous per-layer `moe_intermediate_size` and `num_experts_per_tok`, latent-space MoE at `moe_latent_size` 1024 | registered as a straight alias of `nemotron_h` (byte-identical on-disk layout). Validated on one GB10: dry-run load (41,153 modules replaced, 882M trainable, loss 13.49, peak 62.8 GB) + 3-step smoke (losses 1.61 / 1.41 / 1.65, checkpoint and `train_state` written, peak ~47 GB). Loads cost ~86-91 min, dominated by the per-expert module replacements |

A `model_type` outside the registry is a hard error naming the registry file.

NemotronH-family entries (`nemotron_h`, `nemotron_h_puzzle`, the
NemotronH-Omni wrappers) import `mamba_ssm` for their Mamba2 mixer
layers, which needs a one-time source patch in a triton-only venv:
see [REPRODUCE.md](../REPRODUCE.md#mamba-ssm-for-nemotronh-family-checkpoints-needs-a-source-patch)
and `scripts/patch_mamba_ssm.py`.

### Serving routed-expert (MoE) LoRA: backend-gated, not merge-only

Routed-expert deltas serve LIVE at runtime on a LoRA-capable MoE backend
(`--moe-backend emulation`, or `marlin`); the fast NVFP4 kernels
(`cutlass`/`flashinfer`) report `supports_lora=False`, so a routed adapter loads
but no-ops there. `nybbloris inspect` reports this as `BLOCKED-ROUTED` with a
structured `routed` block (`live_on`/`blocked_on`), and `nybbloris serve`
auto-selects `--moe-backend emulation` for such adapters rather than aborting.
GLM-4.5-Air and Qwen3.5-122B expert-LoRA are proven end-to-end via emulation
(logprob-delta confirmed); see `cross_arch_status.md`.

The adapter key schema and every rekey transform (identity, the `language_model`
wrapped-model remap, per-expert vs fused-3D expert layout) live in ONE module,
[`nvfp4_lora/adapter_keys.py`](../nvfp4_lora/adapter_keys.py); the plan, the two
rekey scripts, and the serve patch all import from it, so the "silent no-op
adapter" class has a single source of truth.

## Fused-3D MoE contract

For families using `NVFP4Experts3D`, the checkpoint must satisfy:

* routed experts stored as per-expert 2D tensors, in either key naming:
  compressed-tensors
  `<moe>.<i>.{gate_proj,up_proj,down_proj}.{weight_packed,weight_scale,weight_global_scale}`
  or ModelOpt
  `<moe>.<i>.{gate_proj,up_proj,down_proj}.{weight,weight_scale,weight_scale_2}`
  (the trainer probes the index and picks the format; ModelOpt naming is
  CPU-parity-tested but no public fused-MoE checkpoint ships it yet)
* the in-memory HF module class matches the registry's `moe_experts_class`
  and exposes `num_experts` / `hidden_dim` / `intermediate_dim`

Per-expert gate/up per-tensor scales may be equal or differ. Equal scales
(every public CT checkpoint validated so far: 12288/12288 experts on both
Qwen3.5-122B releases) use the fused gate_up fast path, one dequant + one
bmm per K-batch. Differing scales select split gate/up storage at load
(probed from the shards in about a second), which is exact but costs two
dequant+bmm pairs for the gate_up projection.

## Trainability rules (enforced before load)

For each `--target-modules` suffix, every matching module in the index is
classified individually by its runtime form: NVFP4 -> native LoRA; FP8
per-tensor -> native LoRA over a frozen FP8 base (`FP8LoRALinear`); BF16 ->
PEFT-wrappable nn.Linear. A suffix with any NVFP4 or FP8 modules is
"native"; a suffix with only BF16 is "peft". The run proceeds only when:

* every target suffix matches at least one module
* native and PEFT suffixes are not mixed in one run
* a native suffix is not also partly BF16 (those BF16 instances would not
  train natively; override `--allow-partial-targets`)

FP8 targets train natively (`FP8LoRALinear`), so an FP8-landing suffix is a
native run that needs no extra flag.

The verdict plus the full inventory is persisted to
`<output_dir>/target_coverage.json`.

## Loading rules (enforced at load)

* every on-disk tensor must map to a model path, except prefixes on the
  family's `skip_st_prefixes` allowlist (vision towers, projectors, MTP
  speculation heads)
* after loading, no parameter or buffer may remain on the meta device, except
  prefixes on the family's `meta_allowed_prefixes` allowlist (frozen
  multimodal towers)
* `--permissive-load` downgrades both to warnings, for bring-up only

## Known out-of-contract layouts

Checkpoints that will be rejected today, by design:

* fused-3D expert storage on disk (single `(num_experts, ...)` tensors
  rather than per-expert keys)
* NVFP4 with group size other than 16
* MoE module classes not named in the registry (`moe_experts_class`)
* `model_type` values without a registry entry

Each rejection message names the assumption that broke and where to add the
support. If you hit one with a public checkpoint, please open an issue with
the inspector's `--json` output attached.
