# Troubleshooting recipes for NVFP4 + LoRA on DGX Spark

A playbook of common errors we hit during development, mapped to causes
and fixes. If you hit one of these, you can shortcut your debugging.

## vLLM serve errors

### `ValueError: NvFp4 MoE backend 'FLASHINFER_TRTLLM' does not support the deployment configuration since kernel does not support current device cuda`

**Cause**: vLLM 0.21's FlashInfer NVFP4 MoE kernels are not compiled for
sm_121 (Blackwell consumer / DGX Spark). Only `MARLIN`, `EMULATION`, and
`VLLM_CUTLASS` accept this device in 0.21.

**Fix**: pass `--moe-backend cutlass` (recommended for Super; gets
~11-14 tok/s with native NVFP4 path). If you hit memory issues, use
`--enforce-eager` to skip the CUDA graph capture phase.

### `ValueError: NvFp4 MoE backend 'VLLM_CUTLASS' does not support the deployment configuration since kernel does not support LoRA`

**Cause**: the CUTLASS MoE kernel (`CutlassExpertsFp4`) has
`supports_lora() = False`; so does `FlashInferExperts`. The kernel itself
has no LoRA-aware forward path, so a routed-expert LoRA delta loads but is a
silent no-op on these backends.

**This is BACKEND-GATED, not merge-only.** On vLLM 0.22.1 the `EMULATION`
backend applies routed-expert LoRA and is the blessed runtime-LoRA path for
routed-MoE NVFP4 (proven end-to-end for GLM-4.5-Air and Qwen3.5-122B;
`MARLIN` also claims LoRA support but repack-OOMs 120B-class on the 131 GB
box). This is exactly the `BLOCKED-ROUTED` inspect verdict: it means "needs
a LoRA-capable MoE backend", NOT "merge-only". `nybbloris serve` handles it
by auto-adding `--moe-backend emulation` when an adapter binds routed-expert
deltas, instead of aborting.

**Fixes, in order of preference:**

1. Serve at runtime on a LoRA-capable backend: `nybbloris serve ...`
   (auto-emulation), or vLLM directly with `--moe-backend emulation
   --enable-lora`. Slow single-stream but correct.
2. Merge-then-serve when you do not need request-time adapter swap and want
   the fast CUTLASS path: `scripts/merge_lora_into_nvfp4.py` bakes the LoRA
   into the NVFP4 base weights; serve the merged checkpoint via the CUTLASS
   recipe (`serve/run_super_ft_merged.sh`).

### `RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered ... _fused_moe_lora_expand`

**Cause**: a vLLM **0.21**-era bug. The Triton `_fused_moe_lora_expand`
kernel assumed a Marlin-format MoE weight layout; with the EMULATION
backend the weights are still in raw NVFP4 packed form, so the kernel
dereferenced off-end and crashed during dummy warmup.

**Fix**: **on vLLM 0.22.1 the EMULATION backend supports runtime LoRA and is the
blessed runtime-LoRA path for routed-MoE NVFP4** (`--moe-backend emulation
--enable-lora`), proven end-to-end for GLM-4.5-Air and Qwen3.5-122B expert-LoRA
(see `docs/cross_arch_status.md`). If you hit this crash you are on an older
vLLM; upgrade to 0.22.1. Merge-then-serve remains an alternative where you do
not need request-time adapter swap, or on backends that report
`supports_lora=False` (CUTLASS/flashinfer on sm_121).

### `ValueError: No available memory for the cache blocks. Try increasing gpu_memory_utilization`

**Cause**: `--gpu-memory-utilization` is too low. After loading the
~69 GiB Super weights, the remaining budget for KV cache went negative.

**Fix**: bump to at least 0.70 for Super with our tight knobs. The
working number is 0.70 (gives ~12 GiB KV budget) to 0.92 (~37 GiB,
default).

### `NVRM NV_ERR_NO_MEMORY` in dmesg, vLLM hangs

**Cause**: the only situation I've seen where vLLM exceeds the physical
memory ceiling on Spark is with `--moe-backend marlin` on a model whose
per-expert Marlin repack pushes transient memory above 130 GiB. This is
unrecoverable; the process becomes unkillable until a hard reboot.

**Fix**: do not use `--moe-backend marlin` for Super (or any model
where this happens). Use `cutlass` or `emulation` instead. See
`serve/diagnostics/` for the safety-thread harness that prevents this
from happening to a diagnostic process.

### `safety_emergency RAM avail X < Y GB` in diagnostic harness logs

**Not a vLLM bug.** Our diagnostic wrapper (`serve/diagnostics/diag_vllm_safe.py`)
has a safety thread that calls `killpg(SIGKILL)` if `psutil.virtual_memory().available`
drops below a configurable floor (default 4 GB). This prevents NVRM kernel
thrash. If you see this, vLLM was approaching real OOM; the diagnostic
wrapper saved you from a hard reboot.

### Memory metrics look stale or inconsistent

`torch.cuda.mem_get_info()` on Spark UMA reports MemFree-equivalent
which **excludes** reclaimable OS page cache. After loading a large
model and then exiting the process, `cuda_free` will look ~40 GB lower
than the boot baseline even though no process is using it. This is the
page cache from the safetensors mmap.

**The CUDA driver reclaims it on demand**, so it's not actually a
leak. Verify with `torch.empty(100_000_000_000, device='cuda')` - it'll
succeed even when `cuda_free` reports < 100 GB.

For correct UMA memory readings, prefer `psutil.virtual_memory().available`
(MemAvailable) over `cuda_free`.

## Training errors

### LoRA adapter loaded but model output is base-like (no FT signal)

**Cause**: in Super-120B, shared expert MLPs and Mamba projections are
FP8, not NVFP4. On the non-pooled loader these train natively via
`FP8LoRALinear`; but the POOLED loader (`pooled_loader_buffers=True`)
has no FP8 LoRA support, so a no-FT-signal result there usually means a
target landed on a module the pooled path can't adapt. (As of the
fail-loud guard, the pooled loader now raises instead of silently
freezing FP8 targets, so a silent demotion should no longer be possible.)

**Fix**: train FP8 targets on the non-pooled path
(`pooled_loader_buffers=False`), which installs `FP8LoRALinear` (frozen
FP8 base + trainable bf16 LoRA). Or, on the pooled path, target only
`up_proj` and `down_proj` on the routed (NVFP4) MoE experts.

### `causal-conv1d not found` or "naive Python scan" warning during training

**Cause**: `causal-conv1d` C++ extension didn't build against your
CUDA toolchain. Without it, Mamba2 falls back to a naive Python scan
and training is effectively infeasible at any useful sequence length.

**Fix**: rebuild with `MAX_JOBS=1 pip install --no-build-isolation causal-conv1d==1.6.2.post1`.
The `MAX_JOBS=1` cap prevents nvcc from being OOM-killed during
parallel compilation on the 128 GB unified pool.

### `ModuleNotFoundError: No module named 'selective_scan_cuda'` loading a NemotronH-family checkpoint

**Also seen as**: an `ImportError` out of
`mamba_ssm/__init__.py` naming a transformers symbol that no longer
exists (`GenerationMixin` internals, `load_config_hf`, and friends), on
any `nemotron_h`, `nemotron_h_puzzle` or NemotronH-Omni load.

**Cause**: mamba-ssm 2.2.5 assumes the Mamba-1 CUDA extension is built
and eagerly imports its full Mamba-1/Mamba-2 model stack from
`__init__.py`. On a triton-only install under transformers 5.x both
assumptions are false, so `import mamba_ssm` fails even though the
Mamba2 triton path these checkpoints actually run is fine. The failure
lands at model-import time, which on a large checkpoint means you find
out after paying for part of the load.

**Fix**: `python scripts/patch_mamba_ssm.py`. It guards both imports in
place, is idempotent, and is a no-op on an install that does have the
extension. See the setup section in
[REPRODUCE.md](../REPRODUCE.md#mamba-ssm-for-nemotronh-family-checkpoints-needs-a-source-patch).

**If it comes back**: the patcher edits installed site-packages, so a
`pip install -U mamba-ssm` or a venv rebuild reverts it. Add
`python scripts/patch_mamba_ssm.py --check` (writes nothing, exit 3 if a
patch is missing) to your preflight so the next revert costs a second
instead of a load.

### `FileNotFoundError` for a `configuration_*.py` under `~/.cache/huggingface/modules/transformers_modules/...`

**Cause**: a `trust_remote_code` checkpoint whose custom code has a
multi-level relative-import chain. transformers stages the modeling file
plus its DIRECT relative imports into the dynamic-module cache, but
resolves the import graph recursively at import time, so a
transitively-imported sibling is never copied in. Puzzle-75B is the
first onboarded checkpoint with a 2-level chain
(`modeling_nemotron_h_puzzle` to `modeling_nemotron_h` to
`configuration_nemotron_h`); single-level custom code is unaffected.

**Fix**: already handled. `ensure_recursive_remote_code_imports()` in
[`nvfp4_lora/loader.py`](../nvfp4_lora/loader.py) widens
`check_imports` to the full recursive closure and is called before the
loader's and the trainer's `Auto*` builds. If you build a model outside
those two entry points, call it yourself first.

### `NVRM: ... Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal` during model load

**Signature**: dozens to hundreds of `NV_ERR_NO_MEMORY` lines in
`/var/log/kern.log` while loading Super-120B (or any large NVFP4 model),
typically in a burst that ends when the load completes. Often coalesced
by the kernel into a `message repeated N times` line. As a concrete
example, the v1.0 release's measurement runs logged a 174-event burst
during the Super-120B training load and a 225-event burst across two
Super merged-FT inference loads; both bursts self-resolved within the
load window and the downstream training and inference completed cleanly.

**Cause**: NVRM/GSP allocation bookkeeping or backing-resource pressure
surfaced through the `_memdescAllocInternal` allocator path. The custom
NVFP4 loader registers ~40K module-level CUDA buffers/Parameters for
Super (packed weight, per-group scale, per-tensor scale, plus LoRA A/B
matrices per NVFP4 module). Loading them in rapid succession stresses
the underlying NVRM allocation paths; individual allocations fail and
retry, eventually succeed. The burst is benign if it self-resolves; if
it cascades on a long-running boot with accumulated NVRM/GSP state, it
can wedge the GPU and force a hard reboot (see the next entry).

**Mitigations**:

1. The training scripts now set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   automatically. PyTorch's expandable segments use larger contiguous
   virtual-address ranges, which reduces allocator-side fragmentation;
   it does not eliminate every NVRM backing allocation, but it is a
   low-risk pre-mitigation worth keeping.
2. Always run large training from a clean boot. Do not run vLLM serves,
   model merges, or repeated benchmarks in the same boot before training.
   Confirm no stale GPU workers with `pgrep -af 'python|vllm'`.
3. Watch the kernel ring during load and the first few optimizer steps
   with `journalctl -k -f -g 'NVRM|Xid'` or `tail -f /var/log/kern.log`.
   If a burst appears during load and stops once load completes, the
   burst was harmless. If a new NVRM burst or any `NVRM: Xid ...` event
   appears AFTER training has begun, abort and reboot before retrying.
4. The real fix is a loader-side allocation refactor (coalesce per-module
   buffers into a few pooled CUDA tensors and register module buffers as
   views). Queued for v1.1; see [PERFORMANCE_ROADMAP.md](PERFORMANCE_ROADMAP.md).

### Hard reboot during training with no Python traceback

**Signature**: training log stops cleanly after a normal step line; no
Python traceback, no `CUDA out of memory`, and PyTorch memory was well
below the 128 GB unified-memory ceiling. Kernel logs (`/var/log/kern.log`)
show NVIDIA RM errors such as:

- `NVRM: ... Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal ... mem_desc.c:1359`
- `NVRM: ... kgrctxAllocCtxBuffers ... kernel_graphics_object.c:215`
- optional earlier `NVRM: Xid ... 31 ... MMU Fault`

**Cause**: NVRM/GSP allocation bookkeeping or backing-resource exhaustion
or fragmentation after a long boot with heavy prior GPU workloads (vLLM
serves, large model loads, repeated benchmarks). The training process
itself was not OOM; `torch.cuda.max_memory_allocated()` only tracks the
PyTorch allocator, not driver-internal RM/GSP allocation paths (the
`_memdescAllocInternal` path and the underlying sysmem / FB-memory
backing resources).

**Fix / prevention**:

1. **Reboot before any long training run.** Do not run vLLM serving,
   model merges, or exploratory benchmarks in the same boot before a
   long training job.
2. **Start training from a clean shell** (no stale Python or vLLM
   workers) and avoid background GPU pollers.
3. **Smoke test 5-10 optimizer steps first** while watching
   `dmesg`/`/var/log/kern.log` for new `NV_ERR_NO_MEMORY`, `Xid`, or
   PCIe AER bursts. Abort and reboot if any appear.
4. **journald cannot be relied on** for crash recovery: on Spark, after
   a hard reboot the previous boot's journal often loses the actual
   crash window. `/var/log/kern.log` and `/var/log/syslog` (plain
   logrotate files) preserve the real timeline; check those first.

## Merge script (`scripts/merge_lora_into_nvfp4.py`)

### `shape '[X, Y]' is invalid for input of size Z` in `NVFP4QTensor.dequantize`

**Cause**: passed packed shape instead of logical shape when constructing
`NVFP4QTensor`. Packed shape is `(out, in//2)`; logical shape is
`(out, in)` since NVFP4 stores 2 elements per byte.

**Fix**: when constructing `NVFP4QTensor(input_shape, ...)`, pass
`(packed.shape[0], packed.shape[1] * 2)`. Our merge script does this
correctly via `get_nvfp4_dequant_then_merge`.

### Merge runs but `validate_merge.py` reports high no-op fraction

**Cause**: the LoRA delta magnitude is below the NVFP4 quant step for
many weights, so the delta gets rounded away during requantization. This
is a real risk for very small LoRA deltas.

**Fix**: re-train with higher `lora_alpha` (current is 16, try 32 or
64) to amplify the effective delta. OR accept the loss and document.
Verify FT signal still shows via `distinguish_ft.py`.

## GB10 unified-memory failure signatures

Four failure modes from a five-failure debugging session bringing up
Qwen3.5-122B on GB10 (2026-06-10). These are the signatures you hit when
porting a new NVFP4 family; the porting guide that produces them is
[PORTING.md](PORTING.md). All four are unified-memory consequences: CPU and
GPU share one ~131 GB DRAM pool, and they fail differently under pressure.

Note on observability before you start: NVML and `nvidia-smi` report `N/A` for
memory on GB10, so they are useless here. Use `torch.cuda.mem_get_info()` plus
`psutil.virtual_memory()`, which is exactly what `gb10_prep.memory_snapshot()`
returns and what the trainer's per-stage `[load-mem]` lines print.

### Trainer OOM-killed on the first training step, constant anon-RSS

**Signature**: the trainer is OOM-killed on (or just before) the first
training step. `dmesg` shows the python process killed with a near-constant
anon-rss (~48 GB in our case) **regardless of sequence length**, and the NVRM
lines show `NV_ERR_NO_MEMORY` from `_memdescAllocInternal`.

**Mechanism**: a weight-sized buffer was allocated on CPU. In our case the
fused MoE expert container defaulted to `device=None`, putting 65 GB of packed
experts into process RSS. On GB10 the CPU and GPU share one DRAM pool but fail
differently: the kernel reclaims page cache under anon pressure, while NVRM
allocations fail immediately. The constant anon-RSS across sequence lengths is
the fingerprint that distinguishes this from a true activation OOM (activation
OOM scales with sequence length; this does not).

**Fix**: pass `device="cuda"` to `replace_moe_experts_with_nvfp4_3d` (the
unified trainer already does this in `load_model`). Audit with the per-stage
`rss`/`cuda_free` load logs: if `process_rss_gb` jumps by a weight-sized amount
at `post-moe-replace`, or the `move-loop relocated NGB from CPU` WARNING fires,
a buffer landed on CPU.

### Post-load `cuda_free` is ~1-2 GB despite the model being only ~76 GB

**Signature**: after load, `cuda_free` is ~1-2 GB even though the model
accounts for only ~76 GB of the 131 GB pool. The first forward at any sequence
length dies.

**Mechanism**: ~50+ GB of safetensors shard pages linger in the OS page cache
after weight assembly, and NVRM cannot force-reclaim them. The bytes are not
leaked, but CUDA cannot allocate against page-cache-occupied DRAM the way the
kernel can for an anon CPU allocation.

**Fix**: call `gb10_prep.drop_shard_page_cache()` after assembly (the trainer
does this at the end of `load_model`). It evicts the clean shard pages via
`posix_fadvise(POSIX_FADV_DONTNEED)` and needs no privileges. The trainer logs
`dropped shard page cache: cuda_free X -> Y`; a healthy drop frees tens of GB.

### Backward crashes with "Triton Error [CUDA]: misaligned address" in fla

**Signature**: forward completes fine, but backward crashes with
`Triton Error [CUDA]: misaligned address` inside `fla`
`prepare_wy_repr_bwd_kernel` during autotune. The clean forward misleads:
nothing looks wrong until the first backward.

**Mechanism**: a flash-linear-attention 0.5.0 backward-kernel bug on GB10
(sm_121, aarch64, CUDA 13.0, triton 3.6.0). The forward kernels are fine, so
the failure is easy to misattribute to your own code or to the model.

**Fix**: pin `flash-linear-attention==0.4.2`. This affects hybrid
linear-attention models (Qwen3.5's GatedDeltaNet layers). Those layers also
require `causal-conv1d`; without both, transformers silently falls back to a
much slower torch path.

### First eval of an NVFP4-attention model spikes memory

**Signature**: training is stable, then the first eval of an NVFP4-attention
model spikes memory (and may OOM).

**Mechanism**: `NVFP4LoRALinear`'s eval-mode bf16 weight cache builds up to
30 GB by default. In eval the modules cache dequantized weights to avoid
recomputing them, and the process-wide cap is high enough to exhaust headroom
on an NVFP4-attention model where many attention projections are also cached.

**Fix**: set the `NVFP4_EVAL_CACHE_GB` env var to cap the cache. 8 is a good
value when post-load headroom is ~50 GB.
