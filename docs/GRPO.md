# Local GRPO against the quantized checkpoint

`scripts/train_grpo.py` trains ordinary attention LoRA on a frozen Nemotron NVFP4
checkpoint and samples from the same local checkpoint in vLLM. It uses the
checkpoint's reconstructed quantized values for differentiable computation;
it never loads the original BF16 base. This is adapter-only RL, not native FP4
pretraining or the 13-parameter TinyLoRA parametrization.

The first supported configuration is NVIDIA Nemotron 3.5 Lightning 30B-A3B on
one DGX Spark, with Transformers 5.15, vLLM 0.27.1, and PyTorch 2.13. These are
the integration targets; a test result must establish runtime compatibility.
The runner uses Marlin MoE, FlashInfer Mamba, aligned Mamba cache, FP8 KV cache,
eager execution, and vLLM's single-process external launcher. It requires
`datasets`, `huggingface_hub`, `safetensors`, and `psutil` in addition to the
model/runtime dependencies. All model and dataset files can be cached in
advance; `--offline` forbids Hub downloads.

## Run a bounded smoke experiment

Run only after freeing the GPU from the existing inference service, with a
supervisor that restores that service on success and failure. The runner does
not stop or start services. `MODEL_SNAPSHOT` must name the local HF snapshot
directory for the exact revision; learner and actor use this same directory.

```bash
python3 scripts/train_grpo.py \
  --model-dir "$MODEL_SNAPSHOT" \
  --model-revision bee7596271d1495f6992ae224aefde4410e816b8 \
  --dataset-revision 740312add88f781978c0658806c59bc2815b9866 \
  --output-dir /experiment/results \
  --steps 1 --batch-size 4 --num-generations 8 \
  --max-new-tokens 256 --max-model-len 1024 \
  --eval-size 16 --temperature 1.2 --seed 42 \
  --learning-rate 1e-4 --lora-rank 8 --lora-alpha 16 \
  --microbatch-size 1 --offline
```

The output directory must be empty. Append `--forward-backward-only` to load the
quantized learner and validate a full-model backward pass without starting
vLLM, loading the dataset, or updating any adapter. Every complete RL run also
performs this gate before constructing the actor. The probe uses a fixed
arithmetic sequence and is explicitly recorded as a gradient diagnostic, not
an RL update.

The model retains packed frozen experts and FP8 projections, with no persistent
dequantized cache. The runner checks all frozen tensor bytes, including scales,
before and after the gradient probe and each RL update. It checks storage dtype
accounting around scoring too. Only adapter parameters enter AdamW. Dropout is
zero, and non-reentrant gradient checkpointing bounds activation memory.

## Objective and data contract

Training uses four prompts with eight completions each, one GRPO optimizer
update, sample-mean reduction, and learning rate `1e-4`. The same temperature
(`1.2`) defines the vLLM behavior distribution and learner logits. vLLM returns
processed log probabilities for its exact native completion IDs. Sampling
uses `top_p=1`, `top_k=-1`, and `min_p=0`; no support-truncating filters are
allowed. Prompt tokens and padding are excluded from the loss, including when
padding shares the EOS token ID. EOS tokens actually sampled remain included.

Advantages are group-centered rewards divided by sample standard deviation.
PPO clipping compares current learner probabilities with detached pre-update
learner probabilities. A separate detached importance weight compares that
old learner with the behavior policy, clipped to `[0.1, 10]`. This bounds the
effect of backend and activation-precision differences; it does not prove
they are absent. Diagnostics report the measured log-probability difference,
ratio range, and clipped fraction. Microbatch gradients are weighted by their
actual sample counts, preserving the full-batch sample mean.

Training prompts come from `train[:-512]`. The runner predeclares disjoint
candidate batches by shuffling original row indices with seed 42. If all
group-relative advantages in a batch are zero, it records that batch and tries
the next predeclared batch, at most three total. Exhaustion produces
`inconclusive_zero_variance` and exit code 2. It never injects synthetic rewards,
uses held-out examples to obtain gradients, or relabels the gradient probe as
learning. Generations ending at the token limit and empty completions are
reported.

Sixteen held-out `test` rows are selected before training using seed 43 and
evaluated greedily before and after. Thinking is disabled in the chat template;
the prompt asks for a concise calculation followed by `#### <number>`. The
verifier compares the final `####` number, otherwise a boxed number, otherwise
the last numeric substring. This permissive fallback is part of the recorded
reward definition. The runner rejects prompts that exceed the model context
budget rather than silently truncating them.

## Evidence and interpretation

Artifacts include:

- `manifest.json`: pinned snapshot identities, checkpoint metadata hashes,
  canonical dataset parquet hashes, software versions, original dataset row
  indices, trainable parameter audit, actor configuration, and arguments.
  `NVFP4_SOURCE_REVISION` supplies the host-resolved commit for containers
  without Git. Direct runs discover Git when available; otherwise the source
  revision is explicitly `null` with source `unavailable`.
- `gradient-gate.json`: all 48 attention adapter tensors have finite gradients,
  all 24 B tensors have nonzero gradients (A may be zero initially), unchanged adapter
  and frozen tensor checks, persistent storage accounting, memory.
- `train-trajectories.jsonl`, `learner-logprobs.jsonl`, `batch-attempts.jsonl`:
  exact prompts, answers, native tokens, behavior/old-learner probabilities,
  rewards, advantages, sampling seeds, and unsuccessful candidate batches.
- `adapters/`: PEFT exports and native learner adapter tensors, with hashes in
  the synchronization proof. No frozen model checkpoint is copied.
- `reload-proof-*.json`: fixed-sequence log probabilities before updating,
  after loading the updated adapter, and after reloading that same adapter.
  A repeated baseline estimates numerical noise; update changes must exceed
  that noise, and updated/reloaded probabilities must agree within `1e-5`.
- `eval-before.jsonl`, `eval-after.jsonl`, `metrics.jsonl`, `result.json`:
  held-out samples, actual loss/gradient/update values, resource measurements,
  frozen-weight digests, status, and failure details.

The fixed-sequence synchronization check uses temperature 1 and the same
prompt tokens for every comparison. Training still uses temperature 1.2.
The initial zero-residual adapter establishes the baseline, and native LoRA
reload plus prefix-cache reset carries each update into the actor.

One update and sixteen questions establish execution and synchronization
only. Accuracy changes on this small sample do not establish a learning
improvement or a fair comparison with a different rank, algorithm, or backend.

CPU checks:

```bash
python3 -m pytest tests/test_grpo.py
```
