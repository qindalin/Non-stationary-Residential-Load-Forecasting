#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"

if [[ -x "$repo_root/.venv/bin/python" ]]; then
	python_bin="$repo_root/.venv/bin/python"
elif [[ -n "${PYTHON_BIN:-}" ]]; then
	python_bin="$PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
	python_bin="$(command -v python3)"
else
	python_bin="$(command -v python)"
fi

cd "$repo_root"
"$python_bin" -m server_app.main --config deployment/server/config.server.local.fullflow.lstm.yaml