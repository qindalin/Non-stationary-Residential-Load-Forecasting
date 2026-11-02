# Simulation — Active Cloud-Edge Co-Adaptation

In-process reproduction of the paper's core algorithm: one server and N
clients, each holding a personalized **edge** model, a client-side **cloud
branch** (global model copy), and the hierarchical fusion module. The online
loop follows the paper exactly:

```
for each online cycle t (every pred_len steps of the test stream):
  1. predict: edge / cloud / stabilized / fused forecasts
  2. observe y_t, update drift statistics -> drift score, urgency
  3. clients with urgency > threshold request sync; server selects budgeted set
  4. selected clients retrieve best-aligned snapshot from the server memory bank
  5. downlink if drift > tau_d and utility > tau_g
  6. update edge model (adaptive lr, selective cloud proximal term)
  7. update cloud branch; upload staleness-discounted innovation
  8. server aggregates reliability-weighted innovations (lambda_c stabilization)
  9. append new snapshot to memory bank; optional broadcast
 10. update fusion variables (long-term gate, context MLP, decay rates)
```

## Files

| File | Role |
|------|------|
| `main_co_adaptation.py` | CLI entry point, model defaults, checkpoint loading |
| `co_adaptation_strategy.py` | `ActiveCoAdaptationRunner`: the full DCE control loop (drift scoring, client selection, snapshot retrieval, co-evolution, aggregation, logging) |
| `online_adaptation.py` | `EdgeAdapter`: bounded-memory online edge update (online / AGEM / MIR / DER replay variants) |
| `Client.py` / `Server.py` | Client (edge + cloud branch + fusion) and server (FedAvg initialization, aggregation) |
| `models/` | Forecasting backbones: `MLP`, `LSTM`, `CNN`, `DLinear`, `PatchTST`; `Ensemble_MLP` (short-term gating network) |
| `layers/` | Minimal layers used by `DLinear` (series decomposition) and `PatchTST` (patch embedding, attention encoder) |
| `data_provider/` | Hourly CSV dataset with the paper's 20% / 5% / 75% train-val-test split |
| `utils/` | Training / ensemble trainers, initialization utilities, early stopping, time features |

## Data format

Each client CSV has a `date` column, one or more feature columns, and a target
column (default `value`):

```csv
date,airTemp,value
2016-01-01 00:00:00,5.0,30.3
2016-01-01 01:00:00,5.0,59.2
```

Five residential clients from the BDG2 "Lamb" subset are bundled under
`data/`. The full 112-building dataset can be assembled with
[`scripts/prepare_bdg2_data.py`](scripts/prepare_bdg2_data.py).

## Running

From this directory (`simulation/`):

```bash
python main_co_adaptation.py --model LSTM --root-path data --train-if-missing --max-steps 72
```

By default all bundled clients are used (limited by `--client-limit 6`).

### The three phases

1. **Offline initialization** — local models and the federated global model
   are loaded from `save/<client>/<model>/checkpoints/model.pkl` and
   `save/offline federated/<model>/model.pkl`. With `--train-if-missing`,
   missing checkpoints are trained first (local training, then
   `Server.offline_fed()` FedAvg). Use `--warmup-ensemble` to pretrain the
   fusion gate on the training stream.
2. **Online co-adaptation** — `ActiveCoAdaptationRunner.run()` executes the
   DCE loop over the test stream (`--max-steps` caps the number of cycles).
3. **Reporting** — CSVs + JSON under `results/co_adaptation/<run>/`.

### Key flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `LSTM` | `MLP` / `LSTM` / `CNN` / `DLinear` / `PatchTST` |
| `--fed-type` | `none` | initialization: `none` (FedAvg) / `fedavgm` / `fedprox` / `fedsgd` |
| `--seq-len` / `--pred-len` | 168 / 24 | lookback window / forecast horizon |
| `--online-lr` | 1e-3 | online learning rate (edge LR is adapted around it) |
| `--sync-threshold` | 0.80 | uplink urgency trigger $\tau_{up}$ |
| `--sync-budget-clients` | 4 | per-cycle server budget $B_t$ |
| `--cloud-memory-size` | 5 | server snapshot memory bank size $M$ |
| `--edge-only` | off | disable cloud collaboration (edge-only baseline) |
| `--edge-adapter` | `online` | `online` / `agem` / `mir` / `der` |
| `--downlink-loss-rate` | 0.0 | simulated downlink failure probability |
| `--max-steps` | 72 | cap on online cycles (**demo default**; pass `--max-steps 548` for the full paper experiment) |
| `--gpu` | 0 | device index |

All `StrategyConfig` fields are exposed as flags; see
[`../docs/METHODOLOGY.md`](../docs/METHODOLOGY.md) for the equation mapping and
defaults.

## Notes

- Determinism: `--seed` fixes numpy/torch seeds per process.
- `--consider-error` adds the recent-performance fusion view (error-mean +
  error-std scores) to the gate.
- `--retrieval-policy latest|random` reproduces the ablation baselines of the
  memory-bank retrieval study.
