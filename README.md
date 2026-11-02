# DCE-Forecast

**Dynamic Cloud-Edge Collaboration for Non-Stationary Residential Load Forecasting**

This repository contains the open-source release of **DCE-Forecast**, a dynamic
cloud-edge collaboration framework for 24-hour-ahead residential load
forecasting in smart grids. The framework is described in:

> *Non-Stationary Residential Load Forecasting: A Dynamic Cloud-Edge Collaboration Approach*, IEEE Transactions on Smart Grid.

## What is DCE-Forecast?

Residential loads are non-stationary: households change their behavior
asynchronously, so a cloud model that is refreshed periodically quickly becomes
stale, while a purely local edge model overfits short-lived fluctuations.
DCE-Forecast treats this as a closed-loop coordination problem between one
cloud server and many edge clients (smart meters) under a limited communication
budget. It combines three coupled mechanisms:

1. **Drift-aware communication.** Each client maintains exponentially weighted
   loss statistics and combines four drift symptoms — standardized loss
   degradation, cloud-edge prediction disagreement, residual volatility, and
   synchronization age — into a single drift score. Clients whose urgency
   exceeds a threshold request synchronization, and the server activates only a
   budget-limited subset. Instead of always downloading the newest global
   model, a selected client searches the server's **memory bank** of recent
   global snapshots for the anchor that best matches its recent window
   (*selective downlink retrieval*). A downlink is accepted only when the
   retrieved anchor is empirically more useful than the client's own recent
   edge baseline.

2. **Robust cloud-edge co-evolution.** The edge learning rate increases
   smoothly with drift intensity (plasticity), while the cloud-edge tethering
   strength increases with cloud utility and decreases with drift (selective
   regularization). A separate client-side *cloud branch* transforms the
   retrieved anchor into an innovation `Δ = θ^g − θ̄^c`; the server aggregates
   reliability-calibrated, staleness-discounted innovations instead of plain
   parameter averages.

3. **Hierarchical prediction-space fusion.** Each stream (edge / cloud) is
   first stabilized by an exponentially weighted buffer of historical
   predictors whose memory scale `λ` is learned online; the stabilized streams
   are then fused by a dynamic gate `α` that combines three views: a
   long-term reliability weight (inertia), a short-term context MLP
   (responsiveness), and a recent-performance score (interpretable
   calibration).

On real-world residential data, DCE-Forecast reduces upload volume by
**66.7%–77.8%** while preserving or improving forecasting accuracy across
multiple forecasting backbones (MLP, LSTM, CNN, DLinear, PatchTST).

## Repository layout

| Path            | Description |
|-----------------|-------------|
| [`simulation/`](simulation/README.md) | The core algorithm and the in-process simulation used for the paper experiments (single machine, one server + N clients). |
| [`hardware/`](hardware/README.md)     | The deployment-oriented sibling: a FastAPI cloud server and a Raspberry-Pi-ready client agent with the same co-adaptation logic, configs and systemd templates. |
| [`pi_client/`](pi_client/README.md)   | Standalone Raspberry Pi reproduction package: five-client full-flow launcher, CPU-only setup script and bundled LSTM checkpoints. |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Equation-by-equation mapping from the paper to the code. |
| [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) | Reproducing the paper's main cross-backbone results: exact commands, verified comparison table and gotchas. |
| [`docs/HARDWARE_DEPLOYMENT.md`](docs/HARDWARE_DEPLOYMENT.md) | Parameter mapping, communication flow and step-by-step hardware deployment guide. |

This release focuses on the **core algorithm and the deployment runtime**.
Paper-figure-generation scripts, exhaustive ablation harnesses and large
notebook artifacts from the original experimental workspace are intentionally
not included.

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CPU-only is fine for the quick demo)
- numpy, pandas, scikit-learn, tqdm, matplotlib

```bash
pip install -r requirements.txt
```

## Quick start (simulation)

The simulation ships with five residential clients from the BDG2 dataset
(`Lamb_*` buildings, hourly, 2016–2017) under `simulation/data/`.

```bash
cd simulation
# Warm-start by training the offline initialization on the bundled clients
# (local models + one round of federated training), then run a short online
# co-adaptation episode (72 online steps):
python main_co_adaptation.py \
    --model LSTM \
    --root-path data \
    --clients Lamb_office_Bertha Lamb_health_Ken Lamb_education_Mae \
    --train-if-missing \
    --train-epochs 3 --fed-rounds 2 --ensemble-epochs 2 \
    --max-steps 72 \
    --gpu 0

# or use the convenience script:
bash scripts/run_quick_demo.sh
```

Results are written to `results/co_adaptation/run_XXX/`:

- `run_summary.json` — overall MAE/MSE/EPE and communication totals
- `co_adaptation_metrics.csv` — per-client ensemble/local/global errors
- `communication_summary.csv` — per-client sync counts and upload/download MB
- `sync_events.csv` — per-event drift score, utility, staleness, bytes

The full 112-building dataset used in the paper is the **BDG2 dataset**
(Miller et al., *Scientific Data*, 2020, CC-BY 4.0) and is **not**
redistributed here. See [`simulation/scripts/prepare_bdg2_data.py`](simulation/scripts/prepare_bdg2_data.py)
for a script that assembles the per-building CSVs from the official BDG2
release ([github.com/buds-lab/the-building-data-genome-project-2](https://github.com/buds-lab/the-building-data-genome-project-2)).

## Reproducing the paper's main results

The paper's main cross-backbone table (MLP/CNN/LSTM/PatchTST, 18 Lamb clients,
full 548-cycle online horizon) has been verified against this release:

- **LSTM** (flagship row) and **CNN**: reproduced **exactly**
  (LSTM MAE 0.36341 vs 0.3634, MSE 0.70520 vs 0.7052, EPE 0.02911 vs 0.0291;
  sync 182.67, upload 268.93 MB, download 44.90 MB).
- **MLP**: within +1.6% MAE / +3.1% MSE (the paper row came from an older
  code revision; a control run of the uncleaned codebase gives identical
  numbers, and the result still beats all baselines).
- **PatchTST**: reproduction with today's improved implementation is
  *better* than the paper row (MAE 0.3406 vs 0.3929), with identical model
  size and upload payload.
- Communication structure (sync counts, payload bytes) matches **exactly**
  for all four backbones.

Full details, per-backbone commands and the verified comparison table:
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md). One-shot script:
`simulation/scripts/run_paper_main_results.sh`; regenerate the table with
`simulation/scripts/compare_with_paper.py`.

> ⚠️ Two gotchas: pass `--max-steps 548` for full-horizon runs (the CLI
> default of 72 is a demo cap), and load paper-era PatchTST checkpoints with
> the legacy `--d-model 512 --d-ff 2048 --n-heads 8 --e-layers 2` config.

## Quick start (hardware server + Pi client)

```bash
# Workstation (server)
cd hardware
pip install -e .
python -m server_app.main --config deployment/server/config.server.reproduce.lstm.yaml

# Raspberry Pi (client, CPU-only PyTorch)
cd pi_client
./setup_pi_env.sh
python main.py --config configs/config.client.reproduce.lstm.yaml
```

See [`docs/HARDWARE_DEPLOYMENT.md`](docs/HARDWARE_DEPLOYMENT.md) for the full
five-client flow (FRP tunneling, systemd services, multi-Pi deployment).

## Citation

```bibtex
@article{dceforecast,
  title   = {Non-Stationary Residential Load Forecasting: A Dynamic Cloud-Edge Collaboration Approach},
  journal = {IEEE Transactions on Smart Grid},
  year    = {2026}
}
```

## License

[MIT](LICENSE). The bundled `Lamb_*` CSV streams are derived from the BDG2
dataset (CC-BY 4.0, see above); please cite the BDG2 paper when using them.
