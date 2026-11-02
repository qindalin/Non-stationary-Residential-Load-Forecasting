from __future__ import annotations

from collections import deque
from typing import Deque

from client_app.data_source import CsvReplayDataSource
from client_app.inference_loop import run_mvp_inference_step
from client_app.local_store import LocalStore
from client_app.runtime_state import RuntimeState
from client_app.sync_client import SyncClient
from shared.protocol.serialization import encode_state_dict
from shared.protocol.schemas import (
    DeviceInfo,
    HeartbeatRequest,
    MetricsReport,
    ModelUpdatePayload,
    RegisterRequest,
    ResourceStats,
    SyncRequest,
)
from shared.strategy.drift import DriftWeights, LossMoments, compute_drift_score, compute_resource_pressure, compute_urgency, standardized_loss_shock


class EdgeAgent:
    def __init__(self, config: dict) -> None:
        client_cfg = config.get("client", {})
        strategy_cfg = config.get("strategy", {})
        self.client_id = client_cfg["client_id"]
        self.pred_len = int(client_cfg.get("pred_len", 24))
        self.sync_client = SyncClient(client_cfg["server_url"], timeout_seconds=float(client_cfg.get("timeout_seconds", 10.0)))
        self.local_store = LocalStore(client_cfg.get("state_dir", "runtime/client_state"))
        self.data_source = CsvReplayDataSource(client_cfg["csv_path"])
        self.state = self.local_store.load_state(self.client_id)
        self.loss_moments = LossMoments(
            mean=float(self.state.local_loss_mean),
            variance=float(self.state.local_loss_variance),
        )
        self.sync_threshold = float(strategy_cfg.get("sync_threshold", 0.80))
        self.beta_resource = float(strategy_cfg.get("beta_resource", 1.0))
        self.age_horizon = int(strategy_cfg.get("age_horizon", 12))
        self.recent_window = int(strategy_cfg.get("recent_window", 6))
        self.weights = DriftWeights(
            loss=float(strategy_cfg.get("drift_weight_loss", 0.40)),
            disagreement=float(strategy_cfg.get("drift_weight_disagreement", 0.30)),
            volatility=float(strategy_cfg.get("drift_weight_volatility", 0.20)),
            age=float(strategy_cfg.get("drift_weight_age", 0.10)),
        )
        self.residual_window: Deque[float] = deque(self.state.residuals[-self.recent_window :], maxlen=self.recent_window)

    def register(self) -> None:
        request = RegisterRequest(
            client_id=self.client_id,
            device=DeviceInfo(
                hostname=self.client_id,
                model_name="deployment-mvp",
                tags=["csv-replay", "raspberry-pi-mvp"],
            ),
        )
        self.sync_client.register(request)

    def run(self, max_steps: int | None = None) -> None:
        self.register()
        for timestep, row in enumerate(self.data_source.iter_rows()):
            if max_steps is not None and timestep >= max_steps:
                break
            self._handle_row(timestep, row)
        self.local_store.save_state(self.state)

    def _handle_row(self, timestep: int, row: dict) -> None:
        output = run_mvp_inference_step(timestep, row)
        self.loss_moments.update(output.local_loss)
        loss_shock = standardized_loss_shock(output.local_loss, self.loss_moments)
        self.residual_window.append(output.residual)
        residual_mean = sum(self.residual_window) / len(self.residual_window)
        residual_volatility = 0.0
        if residual_mean > 0 and len(self.residual_window) > 1:
            variance = sum((item - residual_mean) ** 2 for item in self.residual_window) / len(self.residual_window)
            residual_volatility = (variance ** 0.5) / residual_mean
        sync_age = min((timestep - self.state.last_sync_step) / max(self.age_horizon, 1), 1.0)
        drift_score = compute_drift_score(
            loss_shock=loss_shock,
            disagreement=output.disagreement,
            volatility=residual_volatility,
            sync_age=sync_age,
            weights=self.weights,
        )
        resource_pressure = compute_resource_pressure(int(bool(self.state.last_sync_step)), max(self.recent_window, 1))
        urgency = compute_urgency(drift_score, resource_pressure, self.beta_resource)

        self.sync_client.heartbeat(
            HeartbeatRequest(
                client_id=self.client_id,
                timestep=timestep,
                resources=ResourceStats(network_type="ethernet"),
                latest_mae=output.local_loss,
                latest_mse=output.local_loss ** 2,
            )
        )

        decision = self.sync_client.request_sync(
            SyncRequest(
                client_id=self.client_id,
                timestep=timestep,
                drift_score=drift_score,
                urgency=urgency,
                local_loss=output.local_loss,
                global_loss=output.global_loss,
                resource_pressure=resource_pressure,
            )
        )

        if decision.accepted and decision.upload_required:
            encoded_state = encode_state_dict(
                {
                    "client_id": self.client_id,
                    "timestep": timestep,
                    "local_loss_mean": self.loss_moments.mean,
                }
            )
            self.sync_client.upload_update(
                ModelUpdatePayload(
                    client_id=self.client_id,
                    timestep=timestep,
                    drift_score=drift_score,
                    state_dict_b64=encoded_state,
                )
            )
            self.state.upload_bytes += len(encoded_state.encode("utf-8"))
            download = self.sync_client.download_snapshot(self.client_id)
            if download.available:
                self.state.pending_snapshot_id = download.snapshot_id
                self.state.download_bytes += len((download.state_dict_b64 or "").encode("utf-8"))
            self.state.last_sync_step = timestep

        self.state.timestep = timestep
        self.state.latest_drift_score = drift_score
        self.state.latest_urgency = urgency
        self.state.local_loss_mean = self.loss_moments.mean
        self.state.local_loss_variance = self.loss_moments.variance
        self.state.residuals = list(self.residual_window)

        self.sync_client.report_metrics(
            MetricsReport(
                client_id=self.client_id,
                timestep=timestep,
                mae=output.local_loss,
                mse=output.local_loss ** 2,
                upload_bytes=self.state.upload_bytes,
                download_bytes=self.state.download_bytes,
            )
        )
        self.local_store.save_state(self.state)