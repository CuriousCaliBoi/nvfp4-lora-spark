# Reproducing nvfp4-lora-spark results

> **Looking for the quick public repro?** Use **[REPRODUCE_SPIDER.md](REPRODUCE_SPIDER.md)** (and
> `scripts/repro_spider.sh`): a public base + the public Spider dataset, driven end-to-end by the
> unified trainer + runtime-LoRA serve + `eval_retention.py`, with expected before/after numbers.
> That is the modern, no-private-data, train -> serve -> learned-behaviour loop.
>
> **This document is the LEGACY path** (v1.0 Nemotron/Super measurement runs): it edits constants in
> the frozen `train/*.py` scripts, uses private clinical/regulatory data for the Super example, and
> serves a *merged* checkpoint rather than a runtime adapter. It is kept for exact reproduction of the
> original headline stack, not as the recommended getting-started flow.

Exact stack used to produce the headline numbers in the README.

## Path conventions

Training and diagnostic scripts use placeholder paths of the form
`/path/to/Models/...`, `/path/to/adapters/...`, `/path/to/datasets/...`.
Edit the constants at the top of each `train/*.py` script to point at
your local layout. The serve scripts under `serve/` accept an env-var
override (`MODEL_DIR=...`), so they do not need editing as long as you
export the variable.

## Hardware

- **System**: NVIDIA DGX Spark
- **GPU**: NVIDIA GB10 (Blackwell consumer, sm_121, 128 GB unified LPDDR5x)
- **Compute capability**: 12.1
- **CUDA driver/runtime**: CUDA 13.0
- **OS**: Linux 6.17 aarch64 (Ubuntu kernel)

## Smoke tests (first-run correctness gates)

Run these against the Nano model before the full training/merge pipeline.

```bash
# CPU-only dequant round-trip; needs torchao
python smoke_tests/dequant_correctness.py --model-dir models/Nemotron-3-Nano-30B-A3B-NVFP4

# Forward-parity smoke; needs torch
python smoke_tests/linear_smoke.py --model-dir models/Nemotron-3-Nano-30B-A3B-NVFP4

# Loader smoke (loads a real Nano-sized model; needs ~25 GB free GPU)
python smoke_tests/loader_smoke.py --model-dir models/Nemotron-3-Nano-30B-A3B-NVFP4
```

## Software stack (versions verified 2026-05-24)

| Environment | Python | PyTorch | Key packages |
|-------------|--------|---------|--------------|
| Training venv | 3.12.3 | 2.12.0 | transformers 5.8.1, peft 0.19.1, safetensors 0.7.0, nvidia-modelopt 0.44.0, accelerate 1.13.0, huggingface-hub 1.14.0, causal-conv1d 1.6.2.post1 |
| Serving venv | 3.12.3 | 2.11.0+cu130 | vLLM 0.21.0, flashinfer-python 0.6.8.post1 |

Serving env install:

```bash
python -m venv .venv-serve
source .venv-serve/bin/activate
pip install vllm==0.21.0 flashinfer-python==0.6.8.post1 'torch==2.11.*'
```

## Build `causal-conv1d` from source (required for training)

The Mamba2 fast path needs `causal-conv1d` built against your CUDA
toolchain. Without it, training falls back to a Python scan that is
infeasible at any useful sequence length.

```bash
MAX_JOBS=1 pip install --no-build-isolation causal-conv1d==1.6.2.post1
```

`MAX_JOBS=1` is mandatory on Spark to prevent nvcc from being OOM-killed
during parallel compilation on the 128 GB unified pool.

## `mamba-ssm` for NemotronH-family checkpoints (needs a source patch)

Only needed if you are training or capturing on a NemotronH-family
checkpoint (`nemotron_h`, `nemotron_h_puzzle`, or a NemotronH-Omni
wrapper). Their custom modeling code imports `mamba_ssm` for the Mamba2
mixer layers. Other families do not import it at all.

```bash
pip install --no-build-isolation mamba-ssm==2.2.5
python scripts/patch_mamba_ssm.py
```

The install is triton-only: `selective_scan_cuda` is a Mamba-1 CUDA
extension that is not built here and is not needed, because these
checkpoints run the Mamba2 triton path. Two upstream assumptions in
mamba-ssm 2.2.5 then make `import mamba_ssm` fail outright even though
every submodule the model actually touches (`mamba_ssm.ops.triton.*`)
imports fine:

1. `ops/selective_scan_interface.py` does a bare top-level
   `import selective_scan_cuda`.
2. `__init__.py` eagerly imports the full Mamba-1/Mamba-2 model stack,
   which pulls in (1) and additionally reaches for transformers-4.x
   generation symbols that transformers 5.x removed.

`scripts/patch_mamba_ssm.py` wraps both in try/except with a `None`
fallback. It is idempotent and safe to run unconditionally. Nothing that
worked before stops working: on an install that does have
`selective_scan_cuda` built, both try-blocks succeed and module state is
identical to upstream.

**These are in-place edits to installed site-packages, so any
`pip install -U mamba-ssm` or venv rebuild silently reverts them.** Model
loads then fail at import time, which on a large checkpoint means burning
the load before you find out. Re-run the patcher after any upgrade, or
gate your run on the check mode, which writes nothing and exits 3 if a
patch is missing:

```bash
python scripts/patch_mamba_ssm.py --check
```

Use `--dry-run` to see the exact diff, and `--package-dir` to target a
different venv's `site-packages/mamba_ssm`. If the patcher does not
recognize the installed source it fails loudly (exit 2) rather than
reporting a false success, so a future mamba-ssm release cannot leave you
believing you are patched when you are not.

## Model artifacts

| Artifact | Source | Hash |
|----------|--------|------|
| Base: `Nemotron-3-Super-120B-A12B-NVFP4` | [HuggingFace](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) | (HF main at time of v1.0 release) |
| Trained Super-FT adapter (example) | Trained yourself via `train/train_super_nvfp4.py` - not shipped | |
| Merged Super-FT NVFP4 (example) | Produced by `scripts/merge_lora_into_nvfp4.py` | (hash recorded in `merge_manifest.json` after merge) |

The training data for the Super-FT example adapter is private clinical/
regulatory text. To reproduce a similar FT, train a LoRA against the base
on your own domain corpus following the recipe in `train/train_super_nvfp4.py`.

## Reproducing the headline numbers

### Operational note: clean-boot before large training

The NVRM driver on GB10 has a finite memory-descriptor pool in its GSP
firmware heap. Long-running boots that mix vLLM serves, model merges,
and repeated benchmarks accumulate descriptor-pool pressure. Loading a
large NVFP4 model (Super-120B in particular) under that accumulated
pressure can wedge the GPU and force a hard reboot.

Recommended pre-training procedure:

1. Reboot the host.
2. Confirm no stale Python or vLLM workers (`pgrep -af python`).
3. Run the training script directly (no co-tenant GPU jobs).
4. Watch `/var/log/kern.log` for `NVRM ... NV_ERR_NO_MEMORY` during the
   model-load phase. A short burst is benign; a sustained cascade is the
   signal to abort and reboot. See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
   for the exact signature and mitigation.

### Train

```bash
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --local-dir models/Nemotron-3-Super-120B-A12B-NVFP4
python train/train_super_nvfp4.py    # edit paths at top of file
```

Measured wall: 40.7 h on a single Spark, 1 epoch over 1081 chat-format
examples, max_len=1536, batch=1 with grad_accum=4 (effective batch 4),
AdamW lr=1e-4, gradient checkpointing on. Final train loss 0.81.

### Merge LoRA into NVFP4 base

Point `--lora-adapter-dir` at your own trained adapter directory.

```bash
python scripts/merge_lora_into_nvfp4.py \
    --base-model-dir models/Nemotron-3-Super-120B-A12B-NVFP4 \
    --lora-adapter-dir <your-adapter-dir> \
    --output-dir models/Nemotron-3-Super-120B-A12B-NVFP4-ich-v1.0
```

Measured wall: ~18 min on Spark for all 17 shards (~63 s/shard mean per the shipped merge_manifest.json).
Writes per-shard manifest (with source/output sha256 + per-tensor stats)
and `merge_stats.jsonl` for downstream validation.

### Validate the merge

```bash
python scripts/validate_merge.py \
    --base-model-dir models/Nemotron-3-Super-120B-A12B-NVFP4 \
    --merged-model-dir models/Nemotron-3-Super-120B-A12B-NVFP4-ich-v1.0 \
    --lora-adapter-dir <your-adapter-dir>
```

Reports: tokenizer/config integrity (must be byte-identical to base),
coverage (merged tensor count vs adapter target count), per-tensor
delta-to-quant-step audit, merge cosine similarity, no-op fraction, and
adapter consistency for `alpha_over_r` plus translated LoRA target count.

### Serve Super base (no FT) via CUTLASS

```bash
MODEL_DIR=models/Nemotron-3-Super-120B-A12B-NVFP4 \
    ./serve/run_super_base_inference_cutlass.sh
```

Measured throughput: ~11-14 tok/s, flat across prompt lengths
12-456 tokens, output lengths 32-256 tokens. See
`serve/diagnostics/bench_cutlass_eager_super_base_*.jsonl`.

### Serve Super-FT (merged) via CUTLASS

```bash
MODEL_DIR=models/Nemotron-3-Super-120B-A12B-NVFP4-ich-v1.0 \
    ./serve/run_super_ft_merged.sh
```

Same throughput as base CUTLASS (~11-14 tok/s). The FT behavior is
baked into the served weights.

### Distinguishing test (FT vs base)

```bash
# With base server running on port 8000:
python scripts/distinguish_ft.py collect \
    --url http://localhost:8000 \
    --model nemotron-3-super-a12b-nvfp4 \
    --output-jsonl /tmp/base_outputs.jsonl

# Kill base server, start FT server on port 8000:
python scripts/distinguish_ft.py collect \
    --url http://localhost:8000 \
    --model nemotron-3-super-a12b-nvfp4+ich_v1_0 \
    --output-jsonl /tmp/ft_outputs.jsonl

# Compare:
python scripts/distinguish_ft.py compare /tmp/base_outputs.jsonl /tmp/ft_outputs.jsonl
```

Visually inspect the differing prompts to confirm FT signal is present.

## Licensing and redistribution

This repository is Apache 2.0 (see [LICENSE](LICENSE)). The Nemotron-3
base models are under the [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/),
which is more restrictive than standard OSS licenses.

**What we publish**:
- The training pipeline, scripts, merge script, and serve recipes are
  Apache 2.0 (our own code/data).
- The example LoRA adapter is not shipped in this repository.

**What we do NOT publish**:
- The merged Super-FT NVFP4 checkpoint produced by `merge_lora_into_nvfp4.py`.
  Merged weights are a derivative work of the NVIDIA base model and fall
  under the NVIDIA Nemotron Open Model License's redistribution terms.
  To get the merged checkpoint, download the base from HuggingFace and
  apply our merge script locally.

**For commercial use**: read the NVIDIA Nemotron Open Model License
carefully; it has carve-outs that may or may not apply to your use case.

## Known divergences

Bit-for-bit reproducibility is NOT guaranteed across:

- Different versions of any package above.
- Different CUDA driver / hardware revisions.
- Different filesystem layouts (the safetensors mmap interacts with kernel
  page cache; CUDA memory shows different `cuda_free` after each load).

For reasonable numerical reproducibility (matching tok/s and FT behavior
within ~5%), the stack table above should be sufficient on any GB10
DGX Spark.
