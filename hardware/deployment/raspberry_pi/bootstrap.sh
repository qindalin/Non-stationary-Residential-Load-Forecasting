#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

echo "Bootstrap complete."
echo "Start server: python -m server_app.main --config deployment/server/config.server.yaml"
echo "Start client: python -m client_app.main --config deployment/raspberry_pi/config.client.yaml --max-steps 10"