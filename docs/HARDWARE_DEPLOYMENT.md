# Hardware Deployment Guide

This guide describes how to move from the in-process simulation to a real
two-tier deployment: a FastAPI **cloud server** on a workstation and
**edge clients** running on Raspberry Pis.

## 1. Parameter mapping from the simulation

The hardware runtime implements the same DCE control loop. Choose the config
that matches the experiment family you want.

Recommended **LSTM** mapping (`config.server.reproduce.lstm.yaml` +
`config.client.reproduce.lstm.yaml`):

| Parameter | Value |
|-----------|-------|
| `model_name` | `LSTM` |
| `seq_len` / `pred_len` | 168 / 24 |
| `hidden_dims` | 64 |
| `batch_size` / `lr` / `online_lr` | 32 / 1e-4 / 1e-3 |
| `model_buffer_size` | 7 |
| `sync_threshold` | 0.80 |
| `downlink_drift_threshold` / `downlink_utility_threshold` | 0.60 / 0.01 |
| `sync_budget_clients` | 4 |
| `server_lambda` / `stale_gamma` | 0.5 / 1.0 |
| `eta_min/max_scale`, `eta_slope`, `eta_bias` | 0.5 / 2.0 / 2.0 / 0.5 |
| `mu_max`, `mu_cloud_gain`, `mu_drift_penalty` | 0.05 / 4.0 / 2.5 |
| `recent_window` / `resource_window` / `age_horizon` | 6 / 12 / 12 |

Recommended **DLinear** mapping: same strategy values, with `seq_len: 96`,
`pred_len: 24`, `label_len: 0` (see `config.server.reproduce.yaml` /
`config.client.reproduce.yaml`).

The Pi client's `split_mode: original_online_federated` mirrors the original
dataset partitioning: pretrain on the first 20% of days, online evaluation on
the last 75%, test stream starting at `len(data) - num_test - seq_len`.

## 2. Server-client communication flow

1. Client starts and sends `POST /register`.
2. Client calls `GET /status` and optionally `GET /snapshot/{snapshot_id}` to
   bootstrap the latest cloud state.
3. At each online step the client sends `POST /heartbeat`.
4. If local urgency exceeds the threshold, the client sends
   `POST /sync/request`.
5. If accepted, the client may pull the current cloud snapshot and blend it
   into the local global branch.
6. After local/global online adaptation, the client uploads the global-branch
   delta via `POST /sync/upload`.
7. The server aggregates uploaded deltas (reliability-weighted, staleness
   discounted, $\lambda_c$-stabilized) into a new global snapshot.
8. The client fetches the staged response via `GET /sync/download/{client_id}`.
9. The client reports metrics through `POST /metrics/report`.

## 3. Quick start (single client + server, same LAN)

Assume server IP `192.168.1.100`, port `8010`.

**Server (workstation):**

```bash
cd hardware
pip install -e .                       # or: ./deployment/server/setup_server_env.sh
python -m server_app.main --config deployment/server/config.server.reproduce.lstm.yaml
```

The LSTM config loads the bundled initial global LSTM state from
`hardware/artifacts/lstm_fullflow/global_lstm_state.pt`. Set the config host to
`0.0.0.0` and open TCP port 8010 on the firewall.

**Client (Raspberry Pi):**

```bash
cd pi_client
./setup_pi_env.sh                       # CPU-only PyTorch, no pip cache
# edit configs/config.client.reproduce.lstm.yaml:
#   server_url: http://192.168.1.100:8010
python main.py --config configs/config.client.reproduce.lstm.yaml
```

Verify connectivity first:

```bash
curl http://192.168.1.100:8010/status
```

## 4. Five-client full-flow run (hardware experiment)

The paper's hardware experiment runs five `Lamb_*` clients on one Raspberry Pi
against an external workstation server.

```bash
# Workstation
cd hardware
./deployment/server/start_fullflow_server.sh
#   (starts the server with deployment/server/config.server.local.fullflow.lstm.yaml)

# Raspberry Pi — replay all five bundled clients
cd pi_client
python run_fullflow_local.py --server-url http://192.168.1.100:8010
# smoke test first:
python run_fullflow_local.py --server-url http://192.168.1.100:8010 --max-steps 64
# replay a subset:
python run_fullflow_local.py --server-url http://192.168.1.100:8010 \
    --clients Lamb_office_Bertha Lamb_health_Ken
```

The Pi-side launcher probes the external server, generates per-client configs
under `runtime/local_fullflow/configs/` and runs the five clients against the
bundled CSV streams (`pi_client/data/`) and checkpoints
(`pi_client/artifacts/lstm_fullflow/`). Metrics are recorded on the
workstation:

- Pi: `runtime/local_fullflow/logs/*.log`
- Server: `runtime/server/fullflow/metrics/metrics_summary.json` and
  `metrics_reports.jsonl`

The launcher prints a comparison against `fullflow_expected_metrics.json`.

## 5. FRP tunneling (optional, for cross-network access)

If the Pi cannot reach the workstation directly, expose the server with
[FRP](https://github.com/fatedier/frp). A **template** is provided at
`hardware/deployment/server/frpc.example.toml` — copy it to `frpc.toml` and
fill in your own frps address, port, token and subdomain. Never commit the
real `frpc.toml`.

```bash
cd hardware
./deployment/server/start_fullflow_server_with_frpc.sh
# (server + frpc; set FRPC_BIN / FRPC_CONFIG as needed)
```

On the Pi, point `--server-url` at your tunnel endpoint (e.g. the raw TCP
tunnel `http://your-proxy.example.com:42023`; use the HTTPS subdomain only if
it routes `/status` correctly).

## 6. Real multi-Pi deployment checklist

Before running, ensure:

- server and client use the same `seq_len`, `pred_len`, `moving_avg`;
- `feature_columns` order matches the training CSV columns used on the Pi
  (for the bundled LSTM client: `date,airTemp,value` with
  `target_col: value`, `aux_feature_col: airTemp`);
- server `enc_in` equals the number of client input features;
- the client CSV has enough rows and exceeds `seq_len + pred_len`;
- each client has a unique `client_id` and its own `state_dir`;
- server and client clocks are roughly synchronized.

Per client Pi:

1. copy the `pi_client/` directory to the Pi;
2. change `client_id` and `state_dir` in the client config;
3. replace `csv_path`, `initial_local_state_path`, `initial_global_state_path`
   if the Pi does not share the same paths;
4. set `server_url` to the server LAN IP;
5. start with `python main.py --config configs/config.client.reproduce.lstm.yaml`
   (or `run_fullflow_local.py` for the bundled full-flow).

## 7. systemd services

Templates are provided:

- `hardware/deployment/server/systemd/online-federated-server.service`
- `hardware/deployment/raspberry_pi/systemd/online-federated-client.service`

Edit the paths/`ExecStart` lines to match your installation, copy to
`/etc/systemd/system/`, then `systemctl enable --now <unit>`.

## 8. Recommended first real run

1. Run the server alone and verify `GET /status`.
2. Run one Pi client and watch the server log for `/register`, `/heartbeat`,
   `/sync/request`, `/sync/upload`, `/sync/download`.
3. Bring up the second and third clients.
4. Only then switch from bundled CSVs to your real building CSVs or a live
   meter feed (replace `data_source.py` with your ingestion logic).
