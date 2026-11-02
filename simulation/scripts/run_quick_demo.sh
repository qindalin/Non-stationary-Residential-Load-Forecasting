#!/usr/bin/env bash
# Quick demo of the DCE-Forecast simulation on the bundled BDG2 clients.
# Trains the offline initialization if missing, then runs a short online
# co-adaptation episode (72 cycles) and prints the run summary.
set -euo pipefail
cd "$(dirname "$0")/.."

python main_co_adaptation.py \
    --model LSTM \
    --root-path data \
    --clients Lamb_office_Bertha Lamb_health_Ken Lamb_education_Mae \
    --train-if-missing \
    --train-epochs 3 \
    --fed-rounds 2 \
    --ensemble-epochs 2 \
    --warmup-ensemble \
    --max-steps 72 \
    --client-limit 3 \
    --gpu 0
