# Local REEF serving and quantized learning

This integration connects the existing REEF installation to the frozen Nemotron NVFP4 checkpoint and attention LoRA. CPU REEF records inference receipts, accepts strict GSM8K feedback, schedules candidate training, evaluates it, commits accepted artifacts and publishes verified native vLLM adapters. Each GPU job restores the committed adapter, AdamW state and RNG state. A short campaign is a technical continuity check; it does not establish improved accuracy.

The reviewed environment is REEF `b637cbe42393e91e8ee75af2179f844da586c3dd`, reef-client 0.2.0, vLLM 0.27.1, and model revision `bee7596271d1495f6992ae224aefde4410e816b8`. The container image must already exist locally. The supervisor resolves its immutable image ID and requires committed tracked source.

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

## Inspect and launch

Run on the GPU host. `--dry-run` prints the planned actor command, worker command and learning configuration without stopping or starting services. Replace the image and paths with their actual local values; the state output must be fresh. The existing proxy is optional and remains outside supervisor ownership.

```bash
cd /home/nimitz/projects/reef-nvfp4-integration
REEF_PROXY_TOKEN_FILE=/home/nimitz/projects/nvfp4-experiments/reef-proxy-20260922-01/.token \
  scripts/run_spark_reef_cycle.sh \
  --image spark-vllm-tinylora:0.1 \
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

The owned actor uses 23% GPU allocation, eager execution, context 1024, no speculative draft, no prefix caching, rank8 LoRA and processed log probabilities. `--generation-config vllm` prevents hidden checkpoint sampling defaults. Runtime adapter updates are enabled only on the owned loopback actor. Its attestation is built from the inspected Docker command and actual vLLM version endpoint. Native numerical tests still have to prove that the loaded adapter changes probabilities and that an independent alias reload reproduces them.

The research profile explicitly sets `--no-async-scheduling` and records `async_scheduling: false` in the inspected actor contract. This tests the scheduling hypothesis behind the native probability discrepancy; it is not evidence that the discrepancy is fixed. Marlin, FP8 KV cache, FlashInfer Mamba, maximum 32 sequences and 23% memory allocation remain unchanged. All parity, effect and reload gates retain their existing thresholds.

Workers run as the host UID/GID, with read-only code/model mounts, offline Hugging Face settings and writable cache/home directories. The absolute `learning/checkpoints` path is identical in the CPU service, worker and actor; the actor receives it read-only. Each worker has a unique name, ownership label, CID file and registry entry. Only the supervisor controls production lifecycle.

## Campaign, restart and rollback

The supervisor creates the scenario, runs a bounded first update, stops and restarts its CPU REEF service, verifies exact committed adapter/optimizer/RNG bytes and a new serving incarnation, then targets the second update. It finally calls the real REEF rollback API to return to the first accepted checkpoint and verifies both learner and native serving state.

The campaign counts `optimizer_step`, not REEF scenario steps. Each requested update gets at most three predeclared batches, including skipped zero-signal batches and candidates rejected by the held-out gate. Exhausting them produces exit status 2 and an explicit inconclusive result. Runtime/verification failures produce failure evidence. Neither case relaxes the policy or skips restoration.

The individual `create`, `cycle`, `snapshot`, `verify-resume` and `rollback` client subcommands support inspection and controlled continuation. Run `python3 scripts/run_reef_gsm8k.py --help` and each subcommand's help. They require `REEF_TOKEN_FILE` or `REEF_TOKEN`; the supervisor generates a private `.token` for its research service. Do not manually edit `incumbent.json` or adapter aliases to simulate acceptance or rollback.

`examples/reef/learning.yaml` documents the external plugin layout; its worker wrapper and probe IDs are placeholders. The supervisor generates the concrete `learning.json` with reviewed paths, command arrays and actor identity. Periodic learning can use the same plugin and receipt/report contract with newly declared tasks, while retaining checkpoint state and applying the same selection gate. The supplied campaign is deliberately bounded; it does not install a background scheduler.

## Evidence and scope

The run directory contains source/image provenance, immutable campaign binding, original container identity, actor attestation, worker registry, native receipts/reports, durable REEF commits, accepted checkpoint snapshots, restart/rollback proofs, result and restoration evidence. Each checkpoint includes standard PEFT weights, native LoRA tensors, AdamW, RNG, metrics and a hash manifest. `serving-state.json` binds the published head to native probability probes; API registry acknowledgement alone is insufficient.

Treat the bounded learning campaign as unvalidated until `result.json` records `status: complete`, two accepted optimizer increments, and verified restart/rollback, with matching underlying checkpoint, commit and native probe evidence. A successfully constructed bootstrap is still optimizer step zero. For this rank8/alpha16 configuration it contains 933,888 trainable attention-LoRA parameters; successful allocation, checkpoint export or a low memory peak does not demonstrate adapter compatibility or an accepted learning update.

Inspect `result.json` together with `restoration.json`. A complete learning result is not a successful handoff unless restoration is healthy. Early failures may have `failure.json` without `result.json`; missing completion evidence must never be interpreted as success. Sanitize credentials and generated runtime configuration before sharing evidence. Raw generated state is not a deliverable.

Learning currently supports one scenario, buffered text chat, one choice, this frozen checkpoint and attention adapters. Streaming/tools/schema transforms remain outside the learning capture contract. Tiny held-out scores are reported with their sample size and uncertainty; they are not evidence of broad retention or general performance gains. DSpark plus dynamic LoRA is not claimed by this experiment.

## Startup and native verification diagnostics

If the actor exits before becoming healthy, the supervisor stops waiting immediately and records `actor-startup-failure.json` with the owned container ID, image ID and Docker exit state. Inspect that file and the retained `reef-nvfp4-*-actor.log`. Argument-parser failures occur before model loading; the vLLM 0.27.1 command uses `--no-enable-log-requests`. Use the pinned image's actual help when investigating incompatible flags.

If REEF initialization fails, inspect `reef-first.log`, `failure.json`, and `learning/jobs/bootstrap-*.log`. Completed bootstrap bytes live under `learning/checkpoints/<checkpoint_id>` and can establish that the frozen quantized learner and zero-delta adapter were constructed, even when subsequent actor verification fails. A zero-LoRA-versus-base probability mismatch is a native verification failure, not a training update. Preserve its probe evidence and diagnose the discrepancy without widening the tolerance or bypassing the gate.

Each native verification attempt records `learning/serving/native-probes/<checkpoint_id>/<attempt_id>/evidence.json`. Numbered sibling files `<NN>-<probe_key>.json` retain the exact native request and response before response validation. The journal binds actor identity, actor-contract hash, checkpoint/adapter hashes, aliases and fixed token IDs. Its `status` distinguishes `collecting`, `measured`, `failed` and `passed`; failed attempts retain the stage, error and completed probe paths. Inspect the reference, primary-adapter and independent reload probability vectors, their repeated measurements, and reference/adapter measurements after switching aliases. Collected vectors and a `measured` status are diagnostic evidence, not a passed verification gate; require `passed` and `native_verified: true` for success.

After every failed launch, inspect `restoration.json`. If production was originally running, require `status: healthy`, the exact `original_server_id`, and an empty `errors` list; an originally stopped server stays stopped. `restored-health.json` records the production health response. When an existing proxy was supplied, `existing-proxy-postcheck.json` checks it separately; the supervisor never owns that proxy. Inspect retained worker registry entries under `learning/jobs/worker-registry` if an interrupted GPU job needs attribution. Do not stop containers merely because their names resemble the experiment.
