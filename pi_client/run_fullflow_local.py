from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "lstm_fullflow"
RUNTIME_ROOT = REPO_ROOT / "runtime" / "local_fullflow"
DEFAULT_STATE_ROOT = REPO_ROOT / "state" / "fullflow_reproduce"
REFERENCE_METRICS_PATH = REPO_ROOT / "fullflow_expected_metrics.json"
CLIENTS = [
    "Lamb_assembly_Librada",
    "Lamb_education_Mae",
    "Lamb_health_Ken",
    "Lamb_industrial_Carla",
    "Lamb_office_Bertha",
]


def build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the five-client Pi-side full-flow LSTM reproduction against an external server")
    parser.add_argument(
        "--server-url",
        default=os.environ.get("ONLINE_FED_SERVER_URL"),
        help="Reachable workstation server URL, for example http://192.168.1.10:8010 or an FRP/HTTPS tunnel URL",
    )
    parser.add_argument("--clients", nargs="+", default=None, help="Optional subset of client IDs to run")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap for a smoke test or partial replay")
    parser.add_argument("--stagger-seconds", type=float, default=0.05)
    parser.add_argument("--server-timeout", type=float, default=30.0, help="Seconds to wait for the external server status probe")
    parser.add_argument("--sync-threshold", type=float, default=0.05)
    parser.add_argument("--short-w-lr", type=float, default=1.0e-4)
    parser.add_argument("--long-w-lr", type=float, default=0.5)
    parser.add_argument("--performance-fallback-mix", type=float, default=0.0)
    parser.add_argument("--ensemble-epochs", type=int, default=2)
    parser.add_argument("--min-mix", type=float, default=0.15)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--keep-state", action="store_true", help="Reuse the chosen state root instead of clearing it")
    return parser.parse_args()


def validate_server_url(server_url: str | None) -> str:
    if not server_url:
        raise ValueError("Missing server URL. Pass --server-url or set ONLINE_FED_SERVER_URL.")
    normalized = server_url.strip().rstrip("/")
    lowered = normalized.lower()
    if "workstation_ip" in lowered:
        raise ValueError(
            "The server URL still contains the WORKSTATION_IP placeholder. Replace it with a real LAN IP or an FRP/HTTPS tunnel domain."
        )
    if not (lowered.startswith("http://") or lowered.startswith("https://")):
        raise ValueError("Server URL must start with http:// or https://")
    return normalized


def select_clients(explicit_clients: list[str] | None) -> list[str]:
    if explicit_clients is None:
        return list(CLIENTS)
    unknown = [client for client in explicit_clients if client not in CLIENTS]
    if unknown:
        raise ValueError(f"Unknown clients requested: {unknown}")
    return explicit_clients


def validate_assets(clients: list[str]) -> None:
    missing = []
    for client_name in clients:
        csv_path = DATA_DIR / f"{client_name}.csv"
        local_state = ARTIFACT_DIR / f"{client_name}_local_lstm_state.pt"
        if not csv_path.exists():
            missing.append(str(csv_path))
        if not local_state.exists():
            missing.append(str(local_state))
    global_state = ARTIFACT_DIR / "global_lstm_state.pt"
    if not global_state.exists():
        missing.append(str(global_state))
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Missing reproduction assets:\n{joined}")


def prepare_run_dirs(state_root: Path, keep_state: bool) -> dict[str, Path]:
    config_dir = RUNTIME_ROOT / "configs"
    logs_dir = RUNTIME_ROOT / "logs"
    if not keep_state:
        shutil.rmtree(RUNTIME_ROOT, ignore_errors=True)
        shutil.rmtree(state_root, ignore_errors=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    return {"config_dir": config_dir, "logs_dir": logs_dir, "state_root": state_root}


def write_client_configs(clients: list[str], config_dir: Path, state_root: Path, args: argparse.Namespace) -> list[Path]:
    config_paths = []
    for client_name in clients:
        payload = {
            "client": {
                "client_id": client_name,
                "model_name": "LSTM",
                "server_url": args.server_url.rstrip("/"),
                "csv_path": str(DATA_DIR / f"{client_name}.csv"),
                "state_dir": str(state_root / client_name),
                "initial_local_state_path": str(ARTIFACT_DIR / f"{client_name}_local_lstm_state.pt"),
                "initial_global_state_path": str(ARTIFACT_DIR / "global_lstm_state.pt"),
                "target_col": "value",
                "aux_feature_col": "airTemp",
                "feature_columns": ["airTemp", "value"],
                "seq_len": 168,
                "pred_len": 24,
                "moving_avg": 25,
                "split_mode": "original_online_federated",
                "train_ratio": None,
                "batch_size": 32,
                "train_epochs": 10,
                "lr": 0.0001,
                "online_lr": 0.001,
                "hidden_dims": 64,
                "unsynced_global_lr_scale": 1.0,
                "model_buffer_size": 7,
                "decay_rate": 0.0,
                "update_stride": 24,
                "snapshot_blend": 1.0,
                "timeout_seconds": 30.0,
                "device": "cpu",
            },
            "strategy": {
                "sync_threshold": args.sync_threshold,
                "consider_error": True,
                "adaptive_fusion": True,
                "warmup_ensemble": True,
                "ensemble_epochs": args.ensemble_epochs,
                "short_w_lr": args.short_w_lr,
                "long_w_lr": args.long_w_lr,
                "performance_fallback_mix": args.performance_fallback_mix,
                "min_mix": args.min_mix,
                "global_decay_lr": 0.1,
                "local_decay_lr": 0.1,
                "downlink_drift_threshold": 0.60,
                "downlink_utility_threshold": 0.01,
                "beta_resource": 1.0,
                "age_horizon": 12,
                "recent_window": 6,
                "resource_window": 12,
                "eta_min_scale": 0.5,
                "eta_max_scale": 2.0,
                "eta_slope": 2.0,
                "eta_bias": 0.5,
                "mu_max": 0.05,
                "mu_cloud_gain": 4.0,
                "mu_drift_penalty": 2.5,
                "drift_weight_loss": 0.40,
                "drift_weight_disagreement": 0.30,
                "drift_weight_volatility": 0.20,
                "drift_weight_age": 0.10,
                "epsilon": 1.0e-6,
            },
        }
        config_path = config_dir / f"{client_name}.yaml"
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        config_paths.append(config_path)
    return config_paths


def wait_for_server(server_url: str, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    status_url = f"{server_url.rstrip('/')}/status"
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(status_url, timeout=2.0)
            response.raise_for_status()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.5)
    hint = (
        "Check that the workstation server is running, the address is reachable from the Raspberry Pi, and FRP is active if you are using the public proxy. "
        "If you see a 404 on the HTTPS subdomain for /status, the HTTPS proxy is not routing to the FastAPI service; use the raw TCP tunnel URL instead."
    )
    raise RuntimeError(f"Server did not become ready: {last_error}. {hint}")


def start_clients(config_paths: list[Path], logs_dir: Path, max_steps: int | None, stagger_seconds: float) -> list[tuple[str, subprocess.Popen[str]]]:
    processes = []
    env = build_subprocess_env()
    for config_path in config_paths:
        client_name = config_path.stem
        log_handle = (logs_dir / f"{client_name}.log").open("w", encoding="utf-8")
        command = [sys.executable, "main.py", "--config", str(config_path)]
        if max_steps is not None:
            command.extend(["--max-steps", str(max_steps)])
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        processes.append((client_name, process))
        time.sleep(stagger_seconds)
    return processes


def wait_for_clients(processes: list[tuple[str, subprocess.Popen[str]]]) -> None:
    failures = []
    for client_name, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append((client_name, return_code))
    if failures:
        failure_lines = []
        for client_name, return_code in failures:
            log_path = RUNTIME_ROOT / "logs" / f"{client_name}.log"
            log_tail = ""
            if log_path.exists():
                tail_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-10:]
                log_tail = "\n".join(tail_lines)
            detail = f"{client_name} exited with code {return_code}"
            if log_tail:
                detail = f"{detail}\n{log_tail}"
            failure_lines.append(detail)
        raise RuntimeError("Client failures:\n" + "\n\n".join(failure_lines))


def fetch_remote_summary(server_url: str) -> dict[str, object] | None:
    try:
        response = requests.get(f"{server_url.rstrip('/')}/metrics/summary", timeout=10.0)
        response.raise_for_status()
    except requests.RequestException:
        return None
    return response.json()


def compare_with_reference(summary: dict[str, object] | None) -> dict[str, object]:
    if summary is None:
        return {"current": None, "reference": None if not REFERENCE_METRICS_PATH.exists() else json.loads(REFERENCE_METRICS_PATH.read_text(encoding="utf-8")), "delta": None}
    if not REFERENCE_METRICS_PATH.exists():
        return {"current": summary.get("global", {}), "reference": None, "delta": None}
    reference = json.loads(REFERENCE_METRICS_PATH.read_text(encoding="utf-8"))
    current_global = summary.get("global", {})
    delta = {
        "avg_mae": None if current_global.get("avg_exact_mae") is None else current_global["avg_exact_mae"] - reference["avg_ensemble_mae"],
        "avg_mse": None if current_global.get("avg_exact_mse") is None else current_global["avg_exact_mse"] - reference["avg_ensemble_mse"],
    }
    return {
        "current": current_global,
        "reference": reference,
        "delta": delta,
    }


def main() -> None:
    args = parse_args()
    args.server_url = validate_server_url(args.server_url)
    clients = select_clients(args.clients)
    validate_assets(clients)
    run_dirs = prepare_run_dirs(args.state_root, args.keep_state)
    wait_for_server(args.server_url, args.server_timeout)
    config_paths = write_client_configs(clients, run_dirs["config_dir"], run_dirs["state_root"], args)
    client_processes = start_clients(config_paths, run_dirs["logs_dir"], args.max_steps, args.stagger_seconds)
    wait_for_clients(client_processes)
    summary = fetch_remote_summary(args.server_url)
    comparison = compare_with_reference(summary)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()