#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_spark_grpo_smoke.sh --image IMAGE --repo DIR --hf-cache DIR
       --vllm-cache DIR --output DIR --server CONTAINER --health-url URL
       [--container-name NAME] [--max-runtime SECONDS] [--health-timeout SECONDS]
       [--poll-interval SECONDS] -- TRAIN_GRPO_ARGUMENTS...

Run on the GPU host. Output must not exist. The existing image is pinned to its
local ID; no image or model is downloaded. The runner always receives
--output-dir /experiment/results and --offline. Pass checkpoint and experiment
arguments after --, including --forward-backward-only for a standalone probe.
EOF
}

fail() { printf '%s\n' "$*" >&2; exit 1; }
image= repo= hf_cache= vllm_cache= output= server= health_url=
container_name="nvfp4-grpo-$(date -u +%Y%m%dT%H%M%SZ)-$$"
max_runtime=2400 health_timeout=600 poll_interval=5
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --) shift; break ;;
        --image|--repo|--hf-cache|--vllm-cache|--output|--server|--health-url|--container-name|--max-runtime|--health-timeout|--poll-interval)
            (($# >= 2)) || fail "Missing value for $1"
            option=${1#--}; option=${option//-/_}
            case "$option" in hf_cache|vllm_cache|container_name|max_runtime|health_timeout|poll_interval|health_url|image|repo|output|server) printf -v "$option" '%s' "$2" ;; esac
            shift 2 ;;
        *) fail "Unknown supervisor option: $1 (runner arguments follow --)" ;;
    esac
done
for required in image repo hf_cache vllm_cache output server health_url; do
    [[ -n ${!required} ]] || fail "Missing --${required//_/-}"
done
for numeric in max_runtime health_timeout poll_interval; do
    [[ ${!numeric} =~ ^[1-9][0-9]*$ ]] || fail "${numeric//_/-} must be a positive integer"
done
[[ $server =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ && $container_name =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || fail "Invalid container name"
[[ $server != "$container_name" ]] || fail "Training and inference containers must have different names"
for path_arg in repo hf_cache vllm_cache output; do
    [[ ${!path_arg} = /* && ${!path_arg} != *:* && ${!path_arg} != *$'\n'* ]] || fail "$path_arg must be an absolute path without colons or newlines"
done
for arg in "$@"; do
    case "$arg" in --output-dir|--output-dir=*) fail "The supervisor owns --output-dir" ;; esac
done
[[ -f $repo/scripts/train_grpo.py ]] || fail "Missing $repo/scripts/train_grpo.py"
[[ -d $hf_cache ]] || fail "HF cache directory does not exist"
[[ ! -e $output ]] || fail "Output must be a fresh directory: $output"
for command in docker nvidia-smi curl; do command -v "$command" >/dev/null || fail "Missing command: $command"; done
image_id=$(docker image inspect --format '{{.Id}}' "$image")
server_id=$(docker inspect --format '{{.Id}}' "$server")
was_running=$(docker inspect --format '{{.State.Running}}' "$server_id")
[[ $was_running = true || $was_running = false ]] || fail "Cannot determine original server state"
if docker inspect "$container_name" >/dev/null 2>&1; then fail "Training container already exists: $container_name"; fi
mkdir -p "$vllm_cache"
mkdir "$output"
lock_dir="${TMPDIR:-/tmp}/nvfp4-grpo-${UID}-${server}.lock"
lock_owned=false restore_needed=false container_id=
owner="${container_name}-$$-${RANDOM}"

owned_container() {
    local candidate label
    candidate=$container_id
    if [[ -z $candidate && -s $output/container-id ]]; then candidate=$(cat "$output/container-id"); fi
    if [[ -z $candidate ]]; then candidate=$container_name; fi
    label=$(docker inspect --format '{{ index .Config.Labels "nvfp4-lora-spark.owner" }}' "$candidate" 2>/dev/null) || return 1
    [[ $label = "$owner" ]] || return 1
    container_id=$(docker inspect --format '{{.Id}}' "$candidate") || return 1
}

cleanup() {
    local result=$? cleanup_result=0 state deadline restoration
    trap - EXIT INT TERM
    set +e
    if owned_container; then
        state=$(docker inspect --format '{{.State.Running}}' "$container_id")
        if [[ $state = true ]]; then
            docker stop --timeout 30 "$container_id" >/dev/null
            state=$(docker inspect --format '{{.State.Running}}' "$container_id")
            if [[ $state = true ]]; then docker kill "$container_id" >/dev/null; fi
            [[ $(docker inspect --format '{{.State.Running}}' "$container_id") = false ]] || cleanup_result=1
        elif [[ $state != false ]]; then
            cleanup_result=1
        fi
        docker logs "$container_id" > "$output/run.log" 2>&1
        docker inspect --format '{{.State.ExitCode}}' "$container_id" > "$output/container-exit-code"
    fi
    printf '%s\n' "$result" > "$output/experiment-exit-code"
    restoration=not-needed-server-untouched
    if [[ $was_running = false ]]; then restoration=not-needed-originally-stopped; fi
    if [[ $restore_needed = true ]]; then
        restoration=failed-server-state
        state=$(docker inspect --format '{{.State.Running}}' "$server_id")
        if [[ $state = false ]]; then
            if docker start "$server_id" >/dev/null; then state=true; else restoration=failed-start; fi
        fi
        if [[ $state = true ]]; then
            restoration=failed-health
            deadline=$((SECONDS + health_timeout))
            while ((SECONDS < deadline)); do
                if curl --silent --show-error --fail --max-time 3 "$health_url" > "$output/server-health.txt" 2> "$output/server-health-error.txt"; then
                    restoration=healthy
                    break
                fi
                sleep "$poll_interval"
            done
        fi
        if [[ $restoration != healthy ]]; then
            printf 'Inference restoration failed: %s\n' "$restoration" >&2
            cleanup_result=1
        fi
    fi
    printf '%s\n' "$restoration" > "$output/restoration-status"
    if [[ $lock_owned = true ]]; then rmdir "$lock_dir" || cleanup_result=1; fi
    printf '%s\n' "$cleanup_result" > "$output/cleanup-exit-code"
    if ((cleanup_result)); then exit 1; fi
    exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir "$lock_dir" 2>/dev/null || fail "Another supervisor owns $lock_dir; inspect it before retrying"
lock_owned=true
was_running=$(docker inspect --format '{{.State.Running}}' "$server_id")
[[ $was_running = true || $was_running = false ]] || fail "Cannot determine locked server state"
docker inspect --format '{{json .State}}' "$server_id" > "$output/original-server-state.json"
source_revision=
if command -v git >/dev/null && git -C "$repo" rev-parse --verify HEAD > "$output/source-commit" 2>/dev/null; then
    source_revision=$(cat "$output/source-commit")
    [[ $source_revision =~ ^[[:xdigit:]]{40}$ || $source_revision =~ ^[[:xdigit:]]{64}$ ]] || fail "Host Git returned an invalid source revision"
    git -C "$repo" status --porcelain > "$output/source-status"
else
    printf 'unavailable\n' > "$output/source-commit"
fi
{
    printf 'image_id=%s\nserver_id=%s\noriginal_running=%s\ncontainer_name=%s\nmax_runtime=%s\n' "$image_id" "$server_id" "$was_running" "$container_name" "$max_runtime"
    printf 'runner_command='; printf '%q ' python3 -u /workspace/scripts/train_grpo.py "$@" --output-dir /experiment/results --offline; printf '\n'
} > "$output/supervisor.txt"
if [[ $was_running = true ]]; then
    # A failed Docker response can arrive after the stop took effect.
    restore_needed=true
    docker stop --timeout 60 "$server_id" >/dev/null
fi
sleep 3
remaining=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z ${remaining//[[:space:]]/} ]] || fail "Another compute process owns the GPU; leaving it untouched"

# Creating before starting provides an owned ID even if startup fails midway.
docker create --name "$container_name" --cidfile "$output/container-id" \
    --label "nvfp4-lora-spark.owner=$owner" --gpus all --ipc=host --network=none \
    --entrypoint python3 \
    -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
    -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONPATH=/workspace \
    -e "NVFP4_SOURCE_REVISION=$source_revision" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e NVFP4_EVAL_CACHE_GB=0 -e NVFP4_TRAIN_CACHE_GB=0 \
    -e VLLM_CACHE_ROOT=/vllm-cache -e TRITON_CACHE_DIR=/vllm-cache/triton \
    -e XDG_CACHE_HOME=/experiment/cache \
    -v "$repo:/workspace:ro" -v "$hf_cache:/hf:ro" \
    -v "$vllm_cache:/vllm-cache" -v "$output:/experiment" -w /workspace \
    "$image_id" -u /workspace/scripts/train_grpo.py "$@" --output-dir /experiment/results --offline > "$output/create.txt"
container_id=$(cat "$output/container-id")
owned_container || fail "Cannot verify training container ownership"
deadline=$((SECONDS + max_runtime))
docker start "$container_id" > "$output/start.txt"
while ((SECONDS < deadline)); do
    state=$(docker inspect --format '{{.State.Running}}' "$container_id")
    if [[ $state = false ]]; then
        result=$(docker inspect --format '{{.State.ExitCode}}' "$container_id")
        exit "$result"
    fi
    [[ $state = true ]] || fail "Cannot determine training container state"
    sleep "$poll_interval"
done
printf 'Training exceeded its %s-second runtime budget.\n' "$max_runtime" >&2
exit 124
