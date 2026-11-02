#!/usr/bin/env bash
# One-shot reproduction of the paper's main cross-backbone results.
#
# Prerequisites (see docs/REPRODUCTION.md):
#   - 18 Lamb_* client CSVs under simulation/data/ (BDG2)
#   - offline-initialization checkpoints under simulation/save/
#     (save/<client>/<model>/checkpoints/model.pkl and
#      save/offline federated/<model>/model.pkl)
#   - for PatchTST, checkpoints trained with the legacy 512-dim architecture
#
# Runs LSTM, CNN, MLP and PatchTST sequentially on the FULL online horizon
# (548 cycles). Expected GPU time: ~15 min (MLP/CNN/LSTM) + ~1.5 h (PatchTST).
# Results land in results/co_adaptation_repro/<run>/; afterwards run
#   python scripts/compare_with_paper.py
set -euo pipefail
cd "$(dirname "$0")/.."

CLIENTS="Lamb_office_Bertha Lamb_education_Gabrielle Lamb_industrial_Carla \
Lamb_health_Ken Lamb_industrial_Enrique Lamb_assembly_Librada \
Lamb_assembly_Elinor Lamb_assembly_Katelyn Lamb_assembly_Dorothy \
Lamb_assembly_Rosa Lamb_education_Lazaro Lamb_education_Eula \
Lamb_education_Sara Lamb_office_Caitlin Lamb_education_Mae \
Lamb_assembly_Cherie Lamb_education_Lucas Lamb_education_Armando"

COMMON="--root-path data --clients ${CLIENTS} --client-limit 18 \
--sync-threshold 0.05 --sync-budget-clients 6 \
--consider-error --warmup-ensemble --ensemble-epochs 2 \
--max-steps 548 --log-every 50 --output-root results/co_adaptation_repro"

echo "== LSTM =="
python main_co_adaptation.py --model LSTM ${COMMON} \
  --no-post-sync-broadcast --run-name lstm_18c_repro

echo "== CNN =="
python main_co_adaptation.py --model CNN ${COMMON} \
  --no-post-sync-broadcast --grad-clip-norm 1.0 --snapshot-blend 0.5 \
  --mu-max 0.03 --eta-max-scale 1.0 --performance-fallback-mix 0.3 \
  --run-name cnn_18c_repro

echo "== MLP =="
python main_co_adaptation.py --model MLP ${COMMON} \
  --post-sync-broadcast --grad-clip-norm 1.0 --snapshot-blend 0.5 \
  --server-lambda 0.0 --mu-max 0.03 --eta-min-scale 0.35 \
  --eta-max-scale 0.9 --eta-bias 0.7 --run-name mlp_18c_repro

echo "== PatchTST (legacy 512-dim config) =="
python main_co_adaptation.py --model PatchTST ${COMMON} \
  --seq-len 96 --online-lr 1e-4 --d-model 512 --d-ff 2048 \
  --n-heads 8 --e-layers 2 --no-post-sync-broadcast --grad-clip-norm 1.0 \
  --snapshot-blend 0.5 --mu-max 0.03 --eta-max-scale 1.0 \
  --run-name patchtst_18c_repro

echo "All runs finished. Compare with: python scripts/compare_with_paper.py"
