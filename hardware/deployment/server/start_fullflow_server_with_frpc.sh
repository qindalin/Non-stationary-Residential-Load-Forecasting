#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
default_frpc_bin="$repo_root/tools/frp_0.69.0_linux_amd64/frpc"
if [[ -x "$default_frpc_bin" ]]; then
    frpc_bin="${FRPC_BIN:-$default_frpc_bin}"
else
    frpc_bin="${FRPC_BIN:-frpc}"
fi
frpc_config="${FRPC_CONFIG:-$repo_root/deployment/server/frpc.toml}"

if [[ -x "$repo_root/.venv/bin/python" ]]; then
    python_bin="$repo_root/.venv/bin/python"
elif [[ -n "${PYTHON_BIN:-}" ]]; then
    python_bin="$PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
    python_bin="$(command -v python3)"
else
    python_bin="$(command -v python)"
fi

cleanup() {
    if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid"
        wait "$server_pid" || true
    fi
}

trap cleanup EXIT

if ! command -v "$frpc_bin" >/dev/null 2>&1; then
    echo "frpc binary not found: $frpc_bin" >&2
    echo "Set FRPC_BIN to the frpc executable path, for example /tmp/frp_0.69.0_linux_amd64/frpc" >&2
    exit 1
fi

if [[ ! -f "$frpc_config" ]]; then
    echo "FRPC config not found: $frpc_config" >&2
    exit 1
fi

cd "$repo_root"
"$python_bin" -m server_app.main --config deployment/server/config.server.local.fullflow.lstm.yaml &
server_pid=$!

"$frpc_bin" -c "$frpc_config"