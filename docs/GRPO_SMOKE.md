# Nemotron NVFP4 local GRPO smoke

A single DGX Spark / NVIDIA GB10 completed one local GRPO update against the
frozen quantized Nemotron checkpoint. The exported attention adapter changed
vLLM's fixed-sequence log probabilities and reproduced the same probabilities
after reloading. This validates execution and adapter synchronization.
**Strict final-answer scoring stayed at 14/16 before and after.** The
recorded flexible verifier's apparent increase came from crediting a
truncated response through its last-number fallback, not a newly correct
final answer. No task-quality improvement is established.

The original inference container was restored and returned HTTP 200. Both
experiment and cleanup exit codes were zero; the training container stopped.

## Provenance and configuration

Run: `lightning-grpo-20260922-02`, September 22, 2026 (Pacific time).
Runtime source: `bf63b3e766364d3df4ce8a11a4e8b838c89e5760`.
The measurements below describe that source revision's flexible verifier.

| Item | Configuration |
| --- | --- |
| Model | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` |
| Model revision | `bee7596271d1495f6992ae224aefde4410e816b8` |
| Dataset | `openai/gsm8k`, `main` |
| Dataset revision | `740312add88f781978c0658806c59bc2815b9866` |
| Runtime | PyTorch 2.13.0+cu130, Transformers 5.15.0, vLLM 0.27.1 |
| Adapter | Rank 8, alpha 16; q/k/v/o in six attention blocks; 933,888 trainable parameters |
| Update | Four training prompts × eight completions; one GRPO epoch/update; learning rate `1e-4`; microbatch 1 |
| Sampling | Temperature 1.2, top-p 1, no top-k/min-p truncation; seed 42; thinking disabled |
| Limits | 256 completion tokens, 1,024 model context tokens |
| Objective | Sample-mean clipped GRPO, PPO epsilon 0.2, detached behavior correction clipped to `[0.1, 10]` |
| Evaluation | Same sixteen preselected test rows before/after; greedy decoding; selection seed 43 |

Training uses original GSM8K rows `[1653, 6866, 1368, 2021]` from the training
split excluding its final 512 rows. The verifier accepts the final `####`
number, otherwise a boxed number, otherwise the last numeric substring.

## Precision and memory

The learner loads the same NVFP4 checkpoint used by the rollout engine.
Its 5,935 NVFP4 projections and 46 FP8 Mamba projections remain packed and
frozen. Sensitive checkpoint tensors retain their original precision,
including FP32 state. Dequantized weights exist transiently for higher
precision forward/backward arithmetic; persistent dequantization caches
remain disabled. Only ordinary attention LoRA parameters are optimized.
This experiment does not implement NVIDIA's native FP4 pretraining recipe or
the separate 13-parameter TinyLoRA method.

The actor uses Marlin MoE, FlashInfer Mamba, FP8 KV cache, and eager execution.
The runner requests Mamba cache alignment; the installed vLLM runtime
resolves its cache mode to `none` with prefix caching disabled.

| Measurement | Result |
| --- | --- |
| Persistent frozen learner tensors | 18,888,937,436 bytes (17.59 GiB) |
| Persistent dequantization caches | 0 bytes before/after probe and update |
| Learner-only gradient probe CUDA peak allocated | 20,091,896,320 bytes (18.71 GiB) |
| Entire run CUDA peak allocated | 51,113,989,632 bytes (47.60 GiB) |

The entire-run peak includes the co-resident learner, actor and training
activations. CUDA allocated memory is not total system memory usage.

## Observed results

The full-model gradient probe produced finite gradients for all 48 adapter
tensors, with nonzero gradients for all 24 B tensors. A gradients are zero at
the initial zero-B adapter, as expected. Its diagnostic loss was 0.457949 and
gradient norm 0.585565; the probe did not perform an optimizer update.

| Measurement | Result |
| --- | --- |
| Training completions rewarded by the flexible verifier | 28/32 |
| Training completion tokens | 4,000 |
| Groups with zero reward variance | 3/4; one group supplied the learning signal |
| Length-limited training completions | 1/32 |
| RL loss | 0.00126918 |
| Gradient norm before clipping | 0.0293587 |
| Adapter update norm | 0.0589799 |
| Strict held-out scoring requiring a `####` final answer | 14/16 before; 14/16 after |
| Recorded flexible held-out reward | 14/16 before; 15/16 after; see verifier caveat below |
| Length-limited held-out completions | 0/16 before; 1/16 after |
| Maximum fixed-token actor log-probability change | 0.984044 nats |
| Maximum change after reloading the updated adapter | 0.0 |
| Repeated-baseline log-probability difference | 0.0 |
| Runner elapsed time | 703.14 seconds |

The only flexible-score change was test row 1042. Its updated completion
hit the token limit without a `####` final answer; the fallback rewarded the
trailing number in `3 races` because the dataset's gold answer is 3. The
stored flexible scores are preserved as measured, but this change is not
evidence of improved reasoning or answer accuracy. Strict rescoring requires
the explicit final-answer marker and yields 14/16 on both sides.

Independent replay confirmed that all 32 training rewards also match their
explicit `####` answers. The only nonzero-variance training group remains
four correct answers out of eight; its advantages and learning signal are
unchanged by strict regrading. Recomputing the loss from the saved tokens,
probabilities and advantages agrees with the recorded optimizer loss.

The old-learner/behavior ratio averaged 1.00595, ranging from 0.05930 to
20.9190; 0.1% of sampled-token correction weights required clipping.
Matching checkpoints therefore does not eliminate backend and activation
precision differences. The maximum update/reload changes are token-level
fixed-sequence diagnostics, not aggregate task scores.

All frozen weights and scales retained the same SHA-256 before/after both
the diagnostic probe and RL update:
`771893dc6b235fbb8e415089e0362ecf088aaf295bee62dcbfbe4fc4ff0cc9fb`.
The exported PEFT adapter SHA-256 is
`0f22785d3ef885eef9e5ee14e7b66a177240f395b5621d1727d3715cf18679b4`.

## Evidence and reproduction

The result is supported by `manifest.json`, `gradient-gate.json`,
`batch-attempts.jsonl`, `train-trajectories.jsonl`, `learner-logprobs.jsonl`,
`metrics.jsonl`, `reload-proof-0001.json`, `eval-before.jsonl`,
`eval-after.jsonl`, `result.json`, `service-verification.json`, and
`INDEPENDENT_AUDIT.md`. These preserve exact sampled token
IDs, behavior log probabilities, selected dataset rows, tensor audits,
checkpoint/dataset provenance, and adapter hashes.

Use [the GRPO learning contract](GRPO.md) and
[the Spark supervisor instructions](SPARK_GRPO.md) to reproduce the bounded
experiment. The current runner requires an explicit final `####` numeric
answer line; the recorded `bf63b3e` run used the legacy flexible verifier.
Its original artifacts remain unchanged, with strict rescoring reported above.
Supervisor restoration and exit markers are recorded separately
from the runner's result; restoration was verified at 00:41:58 UTC on
September 23, 2026.
