# Pi Client — Edge Runtime for Raspberry Pi

Standalone Raspberry Pi package implementing the edge side of DCE-Forecast:
the client-side drift scoring, snapshot blending, adaptive edge/cloud-branch
updates and hierarchical fusion of the paper, over HTTP against the
[`hardware/`](../hardware/) FastAPI server.

## What is included

- `client_runtime/` — the edge client implementation (`EdgeAgent`,
  `OnlineForecastTrainer`, `OnlineEnsembleTrainer`, DLinear/LSTM models,
  state serialization, HTTP sync client).
- `data/` — the five `Lamb_*` client CSV streams used in the five-client
  hardware experiment (BDG2 subset).
- `sample_data/` — tiny synthetic CSV for quick smoke tests.
- `artifacts/` — initial local/global LSTM checkpoints for the full-flow run.
- `run_fullflow_local.py` — one-command launcher that starts all five clients
  against an external workstation server.
- `run_fullflow_hardware.sh` — shell wrapper (set `ONLINE_FED_SERVER_URL` or
  edit the default).
- `setup_pi_env.sh` — one-command Pi environment setup (CPU-only PyTorch,
  no pip cache, avoids the large CUDA wheels).
- `configs/` — client YAML configs (demo / DLinear reproduce / LSTM reproduce).

## Reproduce the five-client hardware run

1. Create the Pi environment: `./setup_pi_env.sh`
2. On the workstation, start the hardware server (see `hardware/README.md`):
   `python -m server_app.main --config deployment/server/config.server.local.fullflow.lstm.yaml`
3. Make the server reachable from the Pi (LAN IP or FRP tunnel).
4. On the Pi:
   `python run_fullflow_local.py --server-url http://192.168.1.100:8010`

The launcher probes the external server, generates per-client configs under
`runtime/local_fullflow/configs/` and runs the five clients against the
bundled CSV streams and checkpoints. Metrics are recorded on the workstation
server; the Pi-side launcher prints a comparison against
`fullflow_expected_metrics.json`.

## Useful commands

```bash
# quick smoke test
python run_fullflow_local.py --server-url http://192.168.1.100:8010 --max-steps 64

# replay a subset
python run_fullflow_local.py --server-url http://192.168.1.100:8010 \
    --clients Lamb_office_Bertha Lamb_health_Ken

# reuse interrupted run state
python run_fullflow_local.py --server-url http://192.168.1.100:8010 --keep-state

# single-client entry point (demo config, synthetic sample CSV)
python main.py --config configs/config.client.yaml

# single-client LSTM reproduce
python main.py --config configs/config.client.reproduce.lstm.yaml
```

`setup_pi_env.sh` installs the small runtime dependencies first and then
installs CPU-only `torch` from the PyTorch CPU wheel index with
`--no-cache-dir`.

## Expected output

After a completed run, inspect on the Pi:

- `runtime/local_fullflow/logs/*.log`

On the workstation server:

- `runtime/server/fullflow/metrics/metrics_summary.json`
- `runtime/server/fullflow/metrics/metrics_reports.jsonl`

## Notes

- This folder is intended to run the five clients on the Raspberry Pi; the
  cloud server remains on the workstation.
- The bundled runtime state under `state/` in the original workspace is a
  captured reference run and is **not** shipped; the default launcher uses
  `state/fullflow_reproduce/` and recreates it as needed.
- Server and client model settings for the full flow are fixed to the LSTM
  experiment: `enc_in=2`, `seq_len=168`, `pred_len=24`, `hidden_dims=64`.
