#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# != 3 ]]; then
    printf 'Usage: REEF_TOKEN_FILE=/secret/path run_reef_proxy.sh REEF_PYTHON REEF_REPO FRESH_STATE_DIR\n' >&2
    exit 2
fi
reef_python=$1
reef_repo=$2
proxy_state=$3
[[ $reef_python = /* && -x $reef_python && $reef_repo = /* && -d $reef_repo && $proxy_state = /* ]] || exit 2
[[ ! -e $proxy_state ]] || { printf 'Proxy state directory must be fresh.\n' >&2; exit 1; }
[[ -n ${REEF_TOKEN_FILE:-} && -f $REEF_TOKEN_FILE ]] || { printf 'REEF_TOKEN_FILE is required.\n' >&2; exit 2; }
IFS= read -r REEF_TOKEN < "$REEF_TOKEN_FILE" || [[ -n ${REEF_TOKEN:-} ]]
[[ -n $REEF_TOKEN ]] || exit 2
"$reef_python" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",8901)); s.close()'
mkdir -p "$proxy_state"
export REEF_TOKEN
export REEF_PROXY_STATE="$proxy_state"
config=$(cd "$(dirname "${BASH_SOURCE[0]}")/../examples/reef" && pwd)/proxy.yaml
cd "$reef_repo"
exec "$reef_python" -m reef.cli serve -c "$config"
