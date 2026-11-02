from __future__ import annotations

import argparse

import uvicorn

from server_app.api import create_app
from shared.common.config import load_yaml_config
from shared.models.factory import ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the online federated hardware server")
    parser.add_argument("--config", required=True, help="Path to the server YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    server_config = config.get("server", {})
    model_config = config.get("model", {})
    app = create_app(
        snapshot_dir=server_config.get("snapshot_dir", "runtime/server/snapshots"),
        sync_threshold=float(server_config.get("sync_threshold", 0.80)),
        sync_budget_clients=int(server_config.get("sync_budget_clients", 4)),
        model_config=ModelConfig(
            name=str(model_config.get("name", "DLinear")),
            seq_len=int(model_config.get("seq_len", 96)),
            pred_len=int(model_config.get("pred_len", 24)),
            enc_in=int(model_config.get("enc_in", 1)),
            moving_avg=int(model_config.get("moving_avg", 25)),
            individual=bool(model_config.get("individual", False)),
            hidden_dims=int(model_config.get("hidden_dims", 64)),
        ),
        server_lambda=float(server_config.get("server_lambda", 0.5)),
        stale_gamma=float(server_config.get("stale_gamma", 1.0)),
        post_sync_broadcast=bool(server_config.get("post_sync_broadcast", True)),
        initial_state_path=server_config.get("initial_state_path"),
    )
    uvicorn.run(
        app,
        host=server_config.get("host", "0.0.0.0"),
        port=int(server_config.get("port", 8000)),
    )


if __name__ == "__main__":
    main()