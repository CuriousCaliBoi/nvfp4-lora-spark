# Run a local quantized GRPO smoke experiment

`scripts/run_spark_grpo_smoke.sh` supervises the experiment on the GPU host. It
temporarily stops the named inference container, runs the standalone GRPO
runner, and restores the original container with a health check, including
when training fails or exceeds its time budget. An originally stopped
inference container remains stopped. Other GPU users cause a refusal; the
supervisor never stops their processes or containers.

The checkpoint, dataset snapshot and Docker image must already be cached.
The supervisor resolves the image to its local immutable ID and records the
source commit, working tree status, runner arguments and original container
state. It mounts source and Hugging Face cache read-only, disables network
access, and writes results under a fresh output directory. The explicitly
chosen vLLM cache is writable. The runner stages dataset processing files in
the experiment output. Host Git provenance is passed through
`NVFP4_SOURCE_REVISION`, so the training image does not need Git installed;
an unavailable revision remains explicitly unknown.

For example, set these absolute host paths and use your existing image and
inference container:

```bash
scripts/run_spark_grpo_smoke.sh \
  --image "$TRAINING_IMAGE" \
  --repo "$REPO_DIR" \
  --hf-cache "$HF_CACHE_DIR" \
  --vllm-cache "$VLLM_CACHE_DIR" \
  --output "$NEW_RUN_DIR" \
  --server "$INFERENCE_CONTAINER" \
  --health-url http://127.0.0.1:30000/health \
  --max-runtime 2400 -- \
  --model-dir /hf/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/snapshots/bee7596271d1495f6992ae224aefde4410e816b8 \
  --model-revision bee7596271d1495f6992ae224aefde4410e816b8 \
  --dataset-revision 740312add88f781978c0658806c59bc2815b9866 \
  --steps 1 --batch-size 4 --num-generations 8 \
  --max-new-tokens 256 --max-model-len 1024 --eval-size 16 \
  --temperature 1.2 --seed 42 --learning-rate 1e-4 \
  --lora-rank 8 --lora-alpha 16 --microbatch-size 1
```

Arguments after `--` go directly to `scripts/train_grpo.py`. The supervisor
sets `--output-dir /experiment/results` and `--offline`. The runner performs
its full-model forward/backward gate before initializing vLLM. Add
`--forward-backward-only` to run that gate alone in a separate fresh output
directory; an additional probe is not needed before a normal run.

The default training deadline is 2,400 seconds; adjust `--max-runtime` for
the chosen experiment. Shutdown and restoration have separate bounds:
training stops receive 30 seconds, and inference health is retried for up
to `--health-timeout` seconds (default 600). A successful restoration is
recorded as `healthy` in `restoration-status`. Check that file along with
`experiment-exit-code`, `cleanup-exit-code` and `run.log`. Timeout exits 124;
failed restoration exits 1 even if training succeeded. The stopped training
container and its ID remain available for inspection.

The supervisor serializes runs for the configured inference container using
a temporary-directory lock. If interrupted by a host failure or SIGKILL,
inspect the recorded container ID and inference state before removing a
stale lock. Normal completion, training errors, SIGINT and SIGTERM invoke
owned-container cleanup and restoration.

This experiment uses standard attention LoRA with a frozen quantized base.
A successful smoke establishes gradients, an adapter update and actor
synchronization; it does not establish improved task performance.
