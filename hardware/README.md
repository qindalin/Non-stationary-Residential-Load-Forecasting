# Hardware Runtime (server + client)

Deployment-oriented sibling of the simulation: the same DCE control loop,
split across a FastAPI cloud server and an edge client agent that runs on
resource-constrained devices (tested on Raspberry Pi, CPU-only PyTorch).

## Structure

- `shared/` — protocol (Pydantic schemas, state-dict serialization), common
  config/logging utilities, lightweight model implementations (DLinear, LSTM)
  and the drift/priority/aggregation/snapshot strategy modules.
- `server_app/` — FastAPI endpoints (`api.py`), `Coordinator` (drift-aware
  sync decision, snapshot memory bank, aggregation), registry, snapshot store
  and global model manager.
- `client_app/` — the client agent, local state store, CSV data adapter,
  inference loop and HTTP sync client.
- `deployment/` — server and Raspberry Pi config templates, setup/start
  scripts, systemd units, Dockerfile and an FRP example template.
- `scripts/` — `summarize_metrics.py` for post-run metric summaries.
- `artifacts/` — initial LSTM state dicts for the five-client full-flow
  experiment.
- `sample_data/` — tiny synthetic CSV for smoke tests.

## Quick start

```bash
# 1. create a virtual environment
python3 -m venv .venv && . .venv/bin/activate

# 2. install (CPU-only torch option available, see deployment/server/setup_server_env.sh)
pip install -e .

# 3. server
python -m server_app.main --config deployment/server/config.server.yaml

# 4. client (in another shell)
python -m client_app.main --config deployment/raspberry_pi/config.client.yaml
```

## Server API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/status` | GET | server health, snapshot id, budget |
| `/register` | POST | register a client |
| `/heartbeat` | POST | per-cycle telemetry (drift score, urgency) |
| `/sync/request` | POST | request participation (urgency-gated) |
| `/sync/upload` | POST | upload global-branch innovation Δ |
| `/sync/download/{client_id}` | GET | fetch staged aggregated snapshot |
| `/snapshot/{snapshot_id}` | GET | fetch a memory-bank snapshot |
| `/metrics/report` | POST | client metric reports |

## Full-flow (five-client hardware experiment)

```bash
./deployment/server/start_fullflow_server.sh
# server listens on 0.0.0.0:8010 with deployment/server/config.server.local.fullflow.lstm.yaml
```

The five Pi-side clients are launched from the sibling [`pi_client/`](../pi_client/)
package. See [`../docs/HARDWARE_DEPLOYMENT.md`](../docs/HARDWARE_DEPLOYMENT.md)
for the complete deployment guide (FRP tunneling, systemd, multi-Pi).

## Notes

- Model implementations are intentionally lightweight (LSTM / DLinear in
  `shared/models/`). The full backbone zoo lives in
  [`../simulation/models/`](../simulation/models/).
- The client loop uses a CSV replay adapter as the deployment MVP; replace
  `client_app/data_source.py` with your live meter feed for production use.
