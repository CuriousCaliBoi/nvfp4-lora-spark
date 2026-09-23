# Local REEF serving and quantized learning

This integration connects the existing REEF installation to the frozen Nemotron NVFP4 checkpoint and attention LoRA. CPU REEF records inference receipts, accepts strict GSM8K feedback, schedules candidate training, evaluates it, commits accepted artifacts and publishes verified native vLLM adapters. Each GPU job restores the committed adapter, AdamW state and RNG state. A short campaign is a technical continuity check; it does not establish improved accuracy.

The reviewed environment is REEF `b637cbe42393e91e8ee75af2179f844da586c3dd`, reef-client 0.2.0, vLLM 0.27.1, and model revision `bee7596271d1495f6992ae224aefde4410e816b8`. The supervisor requires the separately built research image and its verified build report, resolves its immutable image ID, and requires committed tracked source.

The experiment source checkout on the Spark is `/home/nimitz/projects/reef-nvfp4-integration`; the separate `/home/nimitz/projects/REEF` checkout provides the CPU service and its virtual environment. Run the commands below from the experiment checkout. Direct client/data commands need that checkout on `PYTHONPATH`; the supervisor supplies it to its child processes.

| Endpoint | Role | Lifecycle |
| --- | --- | --- |
| 127.0.0.1:8900 | Existing Qwen/Tinker REEF | Preserved |
| 127.0.0.1:8901 | Ordinary REEF proxy to 30000 | Separate CPU service |
| 127.0.0.1:30000 | Existing production vLLM | Restored unchanged after learning |
| 127.0.0.1:8902 | Research REEF | Owned by experiment supervisor |
| 127.0.0.1:30001 | Research LoRA actor | Owned by experiment supervisor |

## Ordinary serving

The existing proxy on port 8901 has been verified with real buffered and SSE responses, durable inference receipts, and feedback referencing those receipts. The retained checks are `serving-proof.json` and `streaming-proof.json` under `/home/nimitz/projects/nvfp4-experiments/reef-proxy-20260922-01`. This establishes the normal serving path. The learning campaign requires its own completed validation evidence, as described below.

`examples/reef/proxy.yaml` preserves the provider's ordinary buffered and streaming protocol. To start a new proxy, provide the existing REEF interpreter, checkout and a fresh state directory. Do not run this command if port 8901 already has the established proxy.

```bash
REEF_TOKEN_FILE=/absolute/private/token-file \
  scripts/run_reef_proxy.sh /home/nimitz/projects/REEF/.venv/bin/python3 \
  /home/nimitz/projects/REEF /absolute/fresh/proxy-state
```

The command stays in the foreground and stops with its owning process. Credentials enter the service environment and never appear as command arguments. Keep token files and REEF's generated runtime configuration private; never publish the raw state directory. `/healthz` is public; other routes require bearer authentication.

For a concrete receipt and feedback smoke:

```bash
REEF_TOKEN_FILE=/absolute/private/token-file \
PYTHONPATH="$PWD" \
  /home/nimitz/projects/REEF/.venv/bin/python3 scripts/run_reef_gsm8k.py proxy-smoke \
  --url http://127.0.0.1:8901 --scenario local-proxy-check \
  --output /absolute/proxy-proof.json
```

Buffered inference returns `x-reef-agent-record-id`. A streaming receipt arrives in the final `reef.agent_record_id` SSE event. Report verified feedback to `/reef/report` with `references: [receipt]`; ordinary chat without a verifier is not automatically training data. Generic `inference_proxy` is a serving/recording runtime and does not train.

The normal proxy remains separate during research. Production's GPU allocation is temporarily stopped during the learning maintenance window, so its proxy upstream is unavailable until restoration. The research adapter is not automatically deployed to production.

## Prepare immutable experiment inputs

Use the cached `openai/gsm8k` main parquet files at revision `740312add88f781978c0658806c59bc2815b9866`. Preparation needs CPU PyArrow; it can run in the existing research image without `--gpus`. It never downloads data.

```bash
PYTHONPATH="$PWD" \
  python3 scripts/run_reef_gsm8k.py prepare \
  --train-parquet /absolute/hf/hub/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/train-00000-of-00001.parquet \
  --test-parquet /absolute/hf/hub/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet \
  --output /absolute/fresh/gsm8k-inputs
```

This predeclares six batches, four train questions each, and sixteen held-out test questions. Every batch requests eight distinct rollouts per question at temperature 1.2, full-support sampling and a 256-token completion limit. The same prompt template disables thinking. Strict final-line `#### <number>` scoring permits a terminal period, with no last-number fallback. Length-limited responses retain their actual sampled tokens and receive zero reward.

Prepare a JSON list of native token IDs for numerical verification, using this exact checkpoint's tokenizer or a retained native inference record. Use an identical fixed sequence of 2–1023 tokens for all probes. Do not retokenize training receipts. Supply that file with `--probe-token-ids`; preserve it with the input evidence.

## Build the isolated research image

The research image adds only the Python helper and call site from [upstream Marlin token-order PR 52532](https://github.com/vllm-project/vllm/pull/52532), pinned to commit `e8a07dcccf8e48dd4b9b42a355fc6d5b9db59073`. It orders routed tokens within complete contiguous expert regions, preserving expert assignment, padding and the single-token fast path. This pinned patch passed the unchanged native gates locally with the recorded actor profile; that result does not establish global batch invariance or qualify other models and execution settings.

The reusable image on nimitz is `reef-marlin-order:research-01`, immutable ID `sha256:2f0514d5fe0ab6413b4d9284e7d51462cd01c2ba5171e1c3ea8fd1f6c1215434`. Its build report is `/home/nimitz/projects/nvfp4-experiments/reef-marlin-image-20260922-01.json`. Reuse this pair when the reviewed build inputs match. To rebuild, choose a fresh image tag and report path from the committed experiment checkout:

```bash
python3 scripts/build_reef_marlin_image.py \
  --repo /home/nimitz/projects/reef-nvfp4-integration \
  --tag reef-marlin-order:research-02 \
  --output /absolute/fresh/marlin-build.json
```

This command uses no GPU, downloads no packages, and performs no vLLM or CUDA rebuild. The Dockerfile derives from exact base image ID `sha256:2b57e729b712509ed2eafbb49ca51d754a0f703e08b0b8b02bda8ab44f0a7925` through a fresh temporary local alias checked before and after building. Build steps use `--network=none` and `--pull=false`. Existing output tags are refused; the original base and production container remain unchanged. Only the temporary alias is removed afterward.

The installer rejects unexpected original bytes and duplicate application. The build report records the source revision, all copied input hashes, base/derived image IDs, upstream commit, original/patched source hashes, helper/patch hashes and installed manifest hash. The isolated context includes upstream attribution and its Apache license. A CPU-only verification container independently reads the installed source, manifest and installer and checks the vLLM package version. Keep the build JSON with the immutable image; do not reconstruct it from labels.

## Inspect and launch

Run on the GPU host. `--dry-run` prints the planned actor command, worker command and learning configuration without stopping or starting services. Replace the image and paths with their actual local values; the state output must be fresh. The existing proxy is optional and remains outside supervisor ownership.

```bash
cd /home/nimitz/projects/reef-nvfp4-integration
REEF_PROXY_TOKEN_FILE=/home/nimitz/projects/nvfp4-experiments/reef-proxy-20260922-01/.token \
  scripts/run_spark_reef_cycle.sh \
  --image reef-marlin-order:research-01 \
  --image-provenance /home/nimitz/projects/nvfp4-experiments/reef-marlin-image-20260922-01.json \
  --repo /home/nimitz/projects/reef-nvfp4-integration \
  --reef-repo /home/nimitz/projects/REEF \
  --reef-python /home/nimitz/projects/REEF/.venv/bin/python3 \
  --hf-cache /home/nimitz/.cache/huggingface \
  --model-dir /hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/snapshots/bee7596271d1495f6992ae224aefde4410e816b8 \
  --model-revision bee7596271d1495f6992ae224aefde4410e816b8 \
  --plan /home/nimitz/projects/nvfp4-experiments/reef-input-prep-20260922-01/inputs/campaign.json \
  --probe-token-ids /home/nimitz/projects/nvfp4-experiments/reef-proxy-20260922-01/probe-token-ids.json \
  --output /absolute/fresh/reef-campaign \
  --server nemotron35_lightning_vllm \
  --health-url http://127.0.0.1:30000/health \
  --proxy-url http://127.0.0.1:8901 \
  --max-runtime 10800 --dry-run
```

After reviewing the commands, omit `--dry-run` to execute. Native execution is appropriate only after the combined CPU tests and code review pass. The supervisor checks ports, exact REEF source, committed experiment source, production health and host-readable worker output before changing GPU ownership. It stops only the recorded production container, refuses other GPU owners, and restores the same container on success or failure.

Before changing GPU ownership, the supervisor requires the resolved image ID and current patch build inputs to match `--image-provenance`. After its owned actor starts, it independently reads and hashes the installed source, installer and manifest inside that container before starting CPU REEF. The resulting `actor-contract.json` binds the actual image ID, pinned base ID and checked patch evidence. An image label or successful API health response cannot satisfy this check. A mismatch stops admission and runs the same restoration path.

The owned actor uses 23% GPU allocation, eager execution, context 1024, no speculative draft, no prefix caching, rank8 LoRA and processed log probabilities. `--generation-config vllm` prevents hidden checkpoint sampling defaults. Runtime adapter updates are enabled only on the owned loopback actor. Its attestation is built from the inspected Docker command and actual vLLM version endpoint. Native numerical tests still have to prove that the loaded adapter changes probabilities and that an independent alias reload reproduces them.

The conservative diagnostic profile disables asynchronous scheduling, limits the actor to one sequence, uses BF16 KV cache and `TRITON_ATTN`, and sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Mamba cache mode is explicitly `none`; the Mamba backend remains FlashInfer. The actor contract records these settings from the inspected Docker command and environment. Frozen NVFP4 weights, Marlin, eager execution, context 1024, 23% memory allocation, disabled speculation/prefix caching and attention LoRA remain unchanged.

The conservative settings alone still failed repeatability checks. With the pinned Marlin patch, this profile passed the recorded bootstrap and trained-adapter checks. All repeatability, parity, adapter-effect, reload and held-out acceptance thresholds remain unchanged and run again for every campaign.

Workers run as the host UID/GID, with read-only code/model mounts, offline Hugging Face settings and writable cache/home directories. The absolute `learning/checkpoints` path is identical in the CPU service, worker and actor; the actor receives it read-only. Each worker has a unique name, ownership label, CID file and registry entry. Only the supervisor controls production lifecycle.

## Campaign, restart and rollback

The supervisor creates the scenario, runs a bounded first update, stops and restarts its CPU REEF service, verifies exact committed adapter/optimizer/RNG bytes and a new serving incarnation, then targets the second update. It finally calls the real REEF rollback API to return to the first accepted checkpoint and verifies both learner and native serving state.

The campaign counts `optimizer_step`, not REEF scenario steps. Each requested update gets at most three predeclared batches, including skipped zero-signal batches and candidates rejected by the held-out gate. Exhausting them produces exit status 2 and an explicit inconclusive result. Runtime/verification failures produce failure evidence. Neither case relaxes the policy or skips restoration.

The individual `create`, `cycle`, `snapshot`, `verify-resume` and `rollback` client subcommands support inspection and controlled continuation. Run `python3 scripts/run_reef_gsm8k.py --help` and each subcommand's help. They require `REEF_TOKEN_FILE` or `REEF_TOKEN`; the supervisor generates a private `.token` for its research service. Do not manually edit `incumbent.json` or adapter aliases to simulate acceptance or rollback.

`examples/reef/learning.yaml` documents the external plugin layout; its worker wrapper and probe IDs are placeholders. The supervisor generates the concrete `learning.json` with reviewed paths, command arrays and actor identity. Periodic learning can use the same plugin and receipt/report contract with newly declared tasks, while retaining checkpoint state and applying the same selection gate. The supplied campaign is deliberately bounded; it does not install a background scheduler.

## Evidence and scope

The run directory contains source/image provenance, immutable campaign binding, original container identity, actor attestation, worker registry, native receipts/reports, durable REEF commits, accepted checkpoint snapshots, restart/rollback proofs, result and restoration evidence. Each checkpoint includes standard PEFT weights, native LoRA tensors, AdamW, RNG, metrics and a hash manifest. `serving-state.json` binds the published head to native probability probes; API registry acknowledgement alone is insufficient.

Run07 (`/home/nimitz/projects/nvfp4-experiments/reef-cycle-20260922-07`) completed two real REEF feedback-to-training-to-commit updates, a CPU service restart, continued optimizer state and rollback to the first accepted checkpoint. It used source `fa20c094e9525fa5b8cf6c75752c6683378a6590` and the exact `research-01` image recorded above. The frozen tensor hash stayed unchanged. Each update consumed 32 actual captured rollouts and changed the attention adapters.

| Checkpoint | Optimizer step | Strict held-out correct |
| --- | --- | --- |
| Bootstrap `4a422de90ad6e899` | 0 | 14/16 |
| Accepted `22d4240e09fa51e2` | 1 | 15/16 |
| Accepted `c045364f4ebcfc40` | 2 | 15/16 |

These sixteen rows were used for candidate selection; the scores do not establish a general accuracy gain. Each evaluated checkpoint had one length-limited response scored incorrect. Native adapter effects reached 0.936278 and 0.490211 log-probability difference for the two updates; every measured null, repeat and reload difference was zero under the unchanged gates.

The real CPU REEF restart changed PID 760272 to 765703 and the serving runtime incarnation; the GPU actor stayed running. Comparing `campaign/accepted-1.json` with `resume-proof.json`'s `after` snapshot preserves the checkpoint and all six payload hashes, including optimizer and RNG state. Both snapshots inside `resume-proof.json` are post-restart; they alone are not a before/after restart comparison. The subsequent update advanced the optimizer from 1 to 2. `rollback-proof.json` then records restoration of the first accepted checkpoint and all six original payload hashes, returning optimizer step 2 to 1 with a new published runtime identity and the prior verified native binding. Rollback was not a third training update or a fresh nine-probe run.

Run07's `restoration.json` confirms healthy production restoration with an empty error list and the exact original container ID `67fc87bb3e3858e200bcaea09ee0741a26bcae60ccb2ade7335ad453c20f4db1`. Total supervised runtime including restoration was 1,923.49 seconds. The learned adapter was not deployed onto production.

Run06 had already verified one real frozen-NVFP4 update and native adapter gates, but stopped at the restart port probe. It is partial evidence; the complete continuation and rollback evidence belongs to run07.

Treat the bounded learning campaign as unvalidated until `result.json` records `status: complete`, two accepted optimizer increments, and verified restart/rollback, with matching underlying checkpoint, commit and native probe evidence. A successfully constructed bootstrap is still optimizer step zero. For this rank8/alpha16 configuration it contains 933,888 trainable attention-LoRA parameters; successful allocation, checkpoint export or a low memory peak does not demonstrate adapter compatibility or an accepted learning update.

Inspect `result.json` together with `restoration.json`. A complete learning result is not a successful handoff unless restoration is healthy. Early failures may have `failure.json` without `result.json`; missing completion evidence must never be interpreted as success. Sanitize credentials and generated runtime configuration before sharing evidence. Raw generated state is not a deliverable.

Learning currently supports one scenario, buffered text chat, one choice, this frozen checkpoint and attention adapters. Streaming/tools/schema transforms remain outside the learning capture contract. Tiny held-out scores are reported with their sample size and uncertainty; they are not evidence of broad retention or general performance gains. DSpark plus dynamic LoRA is not claimed by this experiment.

## Startup and native verification diagnostics

If the actor exits before becoming healthy, the supervisor stops waiting immediately and records `actor-startup-failure.json` with the owned container ID, image ID and Docker exit state. Inspect that file and the retained `reef-nvfp4-*-actor.log`. Argument-parser failures occur before model loading; the vLLM 0.27.1 command uses `--no-enable-log-requests`. Use the pinned image's actual help when investigating incompatible flags.

A stopped REEF listener can leave TCP connections in `TIME_WAIT`. The original restart probe used a plain bind, while REEF's `aiohttp.web.run_app` inherits asyncio's POSIX `SO_REUSEADDR` setting. A real socket regression reproduced that false rejection. The probe now matches the server's reuse behavior and tests listening; a short connection check still rejects active listeners, including wildcard listeners on BSD/macOS. An address-in-use error from an active service remains blocking.

If REEF initialization fails, inspect `reef-first.log`, `failure.json`, and `learning/jobs/bootstrap-*.log`. Completed bootstrap bytes live under `learning/checkpoints/<checkpoint_id>` and can establish that the frozen quantized learner and zero-delta adapter were constructed, even when subsequent actor verification fails. A zero-LoRA-versus-base probability mismatch is a native verification failure, not a training update. Preserve its probe evidence and diagnose the discrepancy without widening the tolerance or bypassing the gate.

Each native verification attempt records `learning/serving/native-probes/<checkpoint_id>/<attempt_id>/evidence.json`. Numbered sibling files `<NN>-<probe_key>.json` retain the exact native request and response before response validation. The journal binds actor identity, actor-contract hash, checkpoint/adapter hashes, aliases and fixed token IDs. Its `status` distinguishes `collecting`, `measured`, `failed` and `passed`; failed attempts retain the stage, error and completed probe paths. Inspect the reference, primary-adapter and independent reload probability vectors, their repeated measurements, and reference/adapter measurements after switching aliases. Collected vectors and a `measured` status are diagnostic evidence, not a passed verification gate; require `passed` and `native_verified: true` for success.

After every failed launch, inspect `restoration.json`. If production was originally running, require `status: healthy`, the exact `original_server_id`, and an empty `errors` list; an originally stopped server stays stopped. `restored-health.json` records the production health response. When an existing proxy was supplied, `existing-proxy-postcheck.json` checks it separately; the supervisor never owns that proxy. Inspect retained worker registry entries under `learning/jobs/worker-registry` if an interrupted GPU job needs attribution. Do not stop containers merely because their names resemble the experiment.
