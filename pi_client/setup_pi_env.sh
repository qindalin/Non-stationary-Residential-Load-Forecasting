#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
torch_version="${TORCH_VERSION:-2.5.1}"
torch_cpu_index_url="${TORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
torch_extra_index_url="${TORCH_EXTRA_INDEX_URL:-https://pypi.org/simple}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_bin="$PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
    python_bin="$(command -v python3)"
else
    python_bin="$(command -v python)"
fi

cd "$repo_root"
"$python_bin" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip

python -m pip install --no-cache-dir -r requirements.pi.txt

if python - <<'PY'
import torch
print(torch.__version__)
PY
then
    echo "Using existing torch installation in the Pi environment"
else
    python -m pip install --no-cache-dir --index-url "$torch_cpu_index_url" --extra-index-url "$torch_extra_index_url" "torch==$torch_version"
fi

echo "Pi client environment ready at $repo_root/.venv"