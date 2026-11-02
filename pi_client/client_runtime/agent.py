from __future__ import annotations

from collections import deque
from math import exp, sqrt
import os
from pathlib import Path
from time import perf_counter
from typing import Deque, Dict

import numpy as np
import torch

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

from client_runtime.data import CsvWindowStream, WindowSample
from client_runtime.local_store import LocalStore
from client_runtime.model_factory import ModelConfig, build_model
from client_runtime.serialization import decode_state_dict, encode_state_dict
from client_runtime.state import RuntimeState
from client_runtime.sync_client import SyncClient
from client_runtime.trainer import OnlineEnsembleTrainer, OnlineForecastTrainer, blend_state_dicts, clone_state_dict_cpu, state_dict_difference


class EdgeAgent:
    def __init__(self, config: dict) -> None:
        client_cfg = config.get("client", {})
        strategy_cfg = config.get("strategy", {})
        self.client_id = str(client_cfg["client_id"])
        self.model_name = str(client_cfg.get("model_name", "DLinear"))
        self.seq_len = int(client_cfg.get("seq_len", 96))
        self.pred_len = int(client_cfg.get("pred_len", 24))
        self.update_stride = int(client_cfg.get("update_stride", self.pred_len))
        raw_train_ratio = client_cfg.get("train_ratio", None)
        self.train_ratio = None if raw_train_ratio is None else float(raw_train_ratio)
        self.split_mode = str(client_cfg.get("split_mode", "original_online_federated"))
        self.train_epochs = int(client_cfg.get("train_epochs", 5))
        self.batch_size = int(client_cfg.get("batch_size", 32))
        self.lr = float(client_cfg.get("lr", 5e-4))
        self.online_lr = float(client_cfg.get("online_lr", 1e-3))
        self.unsynced_global_lr_scale = float(client_cfg.get("unsynced_global_lr_scale", 1.0))
        self.model_buffer_size = int(client_cfg.get("model_buffer_size", 7))
        self.decay_rate = float(client_cfg.get("decay_rate", 0.1))
        self.local_decay_lr = float(client_cfg.get("local_decay_lr", 0.1))
        self.global_decay_lr = float(client_cfg.get("global_decay_lr", 0.1))
        self.snapshot_blend = float(client_cfg.get("snapshot_blend", 1.0))
        self.device = torch.device(str(client_cfg.get("device", "cpu")))
        self.initial_local_state_path = client_cfg.get("initial_local_state_path")
        self.initial_global_state_path = client_cfg.get("initial_global_state_path")
        self.sync_client = SyncClient(str(client_cfg["server_url"]), timeout_seconds=float(client_cfg.get("timeout_seconds", 10.0)))
        self.local_store = LocalStore(client_cfg.get("state_dir", "state/default-client"))

        feature_columns = client_cfg.get("feature_columns")
        if not feature_columns:
            raise ValueError("feature_columns must be provided in config.client.yaml")
        target_col = str(client_cfg.get("target_col", feature_columns[-1]))
        aux_feature_col = client_cfg.get("aux_feature_col")
        self.stream = CsvWindowStream(
            csv_path=client_cfg["csv_path"],
            feature_columns=feature_columns,
            target_col=target_col,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            train_ratio=self.train_ratio,
            split_mode=self.split_mode,
        )
        self.aux_idx = None if aux_feature_col is None else list(feature_columns).index(str(aux_feature_col))
        if self.model_name == "LSTM" and self.aux_idx is None:
            raise ValueError("aux_feature_col must be set for LSTM reproduction")
        self.model_config = ModelConfig(
            name=self.model_name,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            enc_in=len(feature_columns),
            moving_avg=int(client_cfg.get("moving_avg", 25)),
            individual=bool(client_cfg.get("individual", False)),
            hidden_dims=int(client_cfg.get("hidden_dims", 64)),
        )

        self.state = self.local_store.load_state(self.client_id)
        self.sync_threshold = float(strategy_cfg.get("sync_threshold", 0.80))
        self.consider_error = bool(strategy_cfg.get("consider_error", True))
        self.adaptive_fusion = bool(strategy_cfg.get("adaptive_fusion", True))
        self.warmup_ensemble = bool(strategy_cfg.get("warmup_ensemble", False))
        self.ensemble_epochs = int(strategy_cfg.get("ensemble_epochs", 2))
        self.short_w_lr = float(strategy_cfg.get("short_w_lr", 1.0e-4))
        self.long_w_lr = float(strategy_cfg.get("long_w_lr", 0.5))
        self.short_only = bool(strategy_cfg.get("short_only", False))
        self.long_only = bool(strategy_cfg.get("long_only", False))
        self.error_only = bool(strategy_cfg.get("error_only", False))
        self.performance_fallback_mix = float(strategy_cfg.get("performance_fallback_mix", 0.0))
        self.min_mix = float(strategy_cfg.get("min_mix", 0.15))
        self.downlink_drift_threshold = float(strategy_cfg.get("downlink_drift_threshold", 0.60))
        self.downlink_utility_threshold = float(strategy_cfg.get("downlink_utility_threshold", 0.01))
        self.beta_resource = float(strategy_cfg.get("beta_resource", 1.0))
        self.age_horizon = int(strategy_cfg.get("age_horizon", 12))
        self.recent_window = int(strategy_cfg.get("recent_window", 6))
        self.resource_window = int(strategy_cfg.get("resource_window", 12))
        self.eta_min_scale = float(strategy_cfg.get("eta_min_scale", 0.5))
        self.eta_max_scale = float(strategy_cfg.get("eta_max_scale", 2.0))
        self.eta_slope = float(strategy_cfg.get("eta_slope", 2.0))
        self.eta_bias = float(strategy_cfg.get("eta_bias", 0.5))
        self.mu_max = float(strategy_cfg.get("mu_max", 0.05))
        self.mu_cloud_gain = float(strategy_cfg.get("mu_cloud_gain", 4.0))
        self.mu_drift_penalty = float(strategy_cfg.get("mu_drift_penalty", 2.5))
        self.drift_weight_loss = float(strategy_cfg.get("drift_weight_loss", 0.40))
        self.drift_weight_disagreement = float(strategy_cfg.get("drift_weight_disagreement", 0.30))
        self.drift_weight_volatility = float(strategy_cfg.get("drift_weight_volatility", 0.20))
        self.drift_weight_age = float(strategy_cfg.get("drift_weight_age", 0.10))
        self.epsilon = float(strategy_cfg.get("epsilon", 1.0e-6))

        self.local_model = build_model(self.model_config).to(self.device)
        self.global_model = build_model(self.model_config).to(self.device)
        self.local_trainer = OnlineForecastTrainer(self.local_model, self.model_config, self.stream.target_idx, self.aux_idx, self.device)
        self.global_trainer = OnlineForecastTrainer(self.global_model, self.model_config, self.stream.target_idx, self.aux_idx, self.device)
        self.ensemble_trainer = OnlineEnsembleTrainer(
            pred_len=self.pred_len,
            short_w_lr=self.short_w_lr,
            long_w_lr=self.long_w_lr,
            device=self.device,
            adaptive_fusion=self.adaptive_fusion,
            consider_error=self.consider_error,
            short_only=self.short_only,
            long_only=self.long_only,
            error_only=self.error_only,
            min_mix=self.min_mix,
        )
        self.local_decay_rate = torch.tensor(self.decay_rate, device=self.device, requires_grad=True)
        self.global_decay_rate = torch.tensor(self.decay_rate, device=self.device, requires_grad=True)
        self.local_buffer, self.global_buffer = self.local_store.load_buffers(self.model_buffer_size)
        self.anchor_state = clone_state_dict_cpu(self.global_model.state_dict())
        self._process = psutil.Process() if psutil is not None else None
        self._peak_rss_mb = 0.0
        self._prime_resource_samplers()
        self._restore_or_initialize_models()

    def _prime_resource_samplers(self) -> None:
        if self._process is None:
            return
        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

    def _collect_resource_stats(self) -> Dict[str, float | int | None]:
        stats: Dict[str, float | int | None] = {
            "system_cpu_percent": None,
            "system_memory_percent": None,
            "process_cpu_percent": None,
            "process_rss_mb": None,
            "process_memory_percent": None,
            "process_threads": None,
            "peak_rss_mb": self._peak_rss_mb,
            "loadavg_1m": None,
            "loadavg_5m": None,
            "loadavg_15m": None,
        }
        if self._process is None:
            return stats

        system_memory = psutil.virtual_memory()
        process_memory = self._process.memory_info().rss / (1024 ** 2)
        self._peak_rss_mb = max(self._peak_rss_mb, float(process_memory))
        stats.update(
            {
                "system_cpu_percent": float(psutil.cpu_percent(interval=None)),
                "system_memory_percent": float(system_memory.percent),
                "process_cpu_percent": float(self._process.cpu_percent(interval=None)),
                "process_rss_mb": float(process_memory),
                "process_memory_percent": float(self._process.memory_percent()),
                "process_threads": int(self._process.num_threads()),
                "peak_rss_mb": float(self._peak_rss_mb),
            }
        )
        if hasattr(os, "getloadavg"):
            load_1, load_5, load_15 = os.getloadavg()
            stats.update(
                {
                    "loadavg_1m": float(load_1),
                    "loadavg_5m": float(load_5),
                    "loadavg_15m": float(load_15),
                }
            )
        return stats

    def _restore_or_initialize_models(self) -> None:
        local_state = self.local_store.load_model_state("local")
        global_state = self.local_store.load_model_state("global")
        anchor_state = self.local_store.load_model_state("anchor")
        ensemble_state = self.local_store.load_ensemble_state()
        if local_state is not None and global_state is not None and anchor_state is not None:
            self.local_model.load_state_dict(local_state)
            self.global_model.load_state_dict(global_state)
            self.anchor_state = clone_state_dict_cpu(anchor_state)
            if ensemble_state is not None:
                self.ensemble_trainer.load_state_dict(ensemble_state)
                self.local_decay_rate = torch.as_tensor(
                    ensemble_state.get("local_decay_rate", self.decay_rate),
                    device=self.device,
                    dtype=torch.float32,
                ).reshape(()).detach().clone().requires_grad_(True)
                self.global_decay_rate = torch.as_tensor(
                    ensemble_state.get("global_decay_rate", self.decay_rate),
                    device=self.device,
                    dtype=torch.float32,
                ).reshape(()).detach().clone().requires_grad_(True)
            return

        if self.initial_local_state_path and self.initial_global_state_path:
            local_path = Path(self.initial_local_state_path)
            global_path = Path(self.initial_global_state_path)
            if local_path.exists() and global_path.exists():
                local_state = torch.load(local_path, map_location="cpu")
                global_state = torch.load(global_path, map_location="cpu")
                self.local_model.load_state_dict(local_state)
                self.global_model.load_state_dict(global_state)
                self.anchor_state = clone_state_dict_cpu(global_state)
                self.local_buffer.append(clone_state_dict_cpu(local_state))
                self.global_buffer.append(clone_state_dict_cpu(global_state))
                if ensemble_state is not None:
                    self.ensemble_trainer.load_state_dict(ensemble_state)
                    self.local_decay_rate = torch.as_tensor(
                        ensemble_state.get("local_decay_rate", self.decay_rate),
                        device=self.device,
                        dtype=torch.float32,
                    ).reshape(()).detach().clone().requires_grad_(True)
                    self.global_decay_rate = torch.as_tensor(
                        ensemble_state.get("global_decay_rate", self.decay_rate),
                        device=self.device,
                        dtype=torch.float32,
                    ).reshape(()).detach().clone().requires_grad_(True)
                elif self.warmup_ensemble:
                    self.ensemble_trainer.warmup(
                        self.stream.train_samples(),
                        self.local_trainer,
                        self.global_trainer,
                        self.ensemble_epochs,
                    )
                self._persist_runtime()
                return

        train_samples = self.stream.train_samples()
        self.local_trainer.train_initial(train_samples, lr=self.lr, epochs=self.train_epochs, batch_size=self.batch_size)
        initial_state = clone_state_dict_cpu(self.local_model.state_dict())
        self.global_model.load_state_dict(initial_state)
        self.anchor_state = clone_state_dict_cpu(initial_state)
        self.local_buffer.append(clone_state_dict_cpu(initial_state))
        self.global_buffer.append(clone_state_dict_cpu(initial_state))
        if self.warmup_ensemble:
            self.ensemble_trainer.warmup(
                train_samples,
                self.local_trainer,
                self.global_trainer,
                self.ensemble_epochs,
            )
        self._persist_runtime()

    def _buffered_prediction(
        self,
        state_buffer: Deque[Dict[str, torch.Tensor]],
        fallback_state: Dict[str, torch.Tensor],
        sample: WindowSample,
        decay_rate: torch.Tensor,
        decay_lr: float,
        update_decay: bool,
    ) -> torch.Tensor:
        states = list(state_buffer) if state_buffer else [fallback_state]
        weights = torch.zeros(len(states), device=self.device)
        for index in range(len(states)):
            weights[index] = torch.exp(-decay_rate * (len(states) - index - 1))
        weights = weights / torch.clamp(weights.sum(), min=1.0e-12)

        prediction = torch.zeros(self.pred_len, device=self.device)
        target = torch.from_numpy(sample.y[:, self.stream.target_idx]).float().to(self.device)
        x_tensor = torch.from_numpy(sample.x).unsqueeze(0).float().to(self.device)
        y_tensor = torch.from_numpy(sample.y).unsqueeze(0).float().to(self.device)
        for weight, state_dict in zip(weights, states):
            model = build_model(self.model_config).to(self.device)
            model.load_state_dict(state_dict)
            model.eval()
            with torch.no_grad():
                if self.model_name == "LSTM":
                    branch_pred = model(x_tensor[:, :, self.stream.target_idx], y_tensor[:, :, self.aux_idx]).squeeze(0)
                else:
                    branch_pred = model(x_tensor)[:, :, self.stream.target_idx].squeeze(0)
            prediction = prediction + weight * branch_pred

        if update_decay:
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()
            with torch.no_grad():
                decay_rate -= decay_lr * decay_rate.grad
                decay_rate.grad.zero_()
        return prediction.detach().cpu()

    def register(self) -> None:
        self.sync_client.register(
            {
                "client_id": self.client_id,
                "device": {
                    "hostname": self.client_id,
                    "hardware": "raspberry-pi",
                    "model_name": self.model_name,
                    "tags": [self.model_name.lower(), "active-coadaptation", "standalone-client"],
                },
            }
        )

    def _bootstrap_from_server(self) -> None:
        status = self.sync_client.status()
        snapshot_id = status.get("latest_snapshot_id")
        if not snapshot_id or snapshot_id == self.state.latest_snapshot_id:
            return
        payload = self.sync_client.fetch_snapshot(snapshot_id)
        if payload.get("available"):
            downloaded = decode_state_dict(payload["state_dict_b64"])
            self.global_model.load_state_dict(downloaded)
            self.anchor_state = clone_state_dict_cpu(downloaded)
            self.state.latest_snapshot_id = payload.get("snapshot_id")
            self.state.anchor_step = int(payload.get("metadata", {}).get("timestep", 0))

    def run(self, max_steps: int | None = None) -> None:
        self.register()
        self._bootstrap_from_server()
        step_counter = 0
        for sample in self.stream.online_samples():
            if sample.target_start_index % self.update_stride != 0:
                continue
            self._handle_sample(sample)
            step_counter += 1
            if max_steps is not None and step_counter >= max_steps:
                break
        self._persist_runtime()

    def _sigmoid(self, value: float) -> float:
        return 1.0 / (1.0 + exp(-value))

    def _update_loss_shock(self, local_loss: float) -> float:
        mean = float(self.state.local_loss_mean)
        var = float(self.state.local_loss_variance)
        if mean == 0.0 and var <= self.epsilon:
            self.state.local_loss_mean = local_loss
            self.state.local_loss_variance = max(local_loss * local_loss, self.epsilon)
            return 0.0
        updated_mean = 0.9 * mean + 0.1 * local_loss
        updated_var = 0.9 * var + 0.1 * ((local_loss - updated_mean) ** 2)
        self.state.local_loss_mean = updated_mean
        self.state.local_loss_variance = max(updated_var, self.epsilon)
        return max((local_loss - updated_mean) / (sqrt(self.state.local_loss_variance) + self.epsilon), 0.0)

    def _residual_volatility(self) -> float:
        if len(self.state.residuals) < 2:
            return 0.0
        residuals = np.asarray(self.state.residuals[-self.recent_window :], dtype=np.float64)
        return float(residuals.std() / (residuals.mean() + self.epsilon))

    def _resource_pressure(self) -> float:
        history = self.state.sync_history[-self.resource_window :]
        if not history:
            return 0.0
        return float(np.mean(np.asarray(history, dtype=np.float64)))

    def _adaptive_local_lr(self, drift_score: float) -> float:
        local_lr_floor = self.online_lr * self.eta_min_scale
        local_lr_ceiling = self.online_lr * self.eta_max_scale
        scaled = self._sigmoid(self.eta_slope * (drift_score - self.eta_bias))
        return local_lr_floor + (local_lr_ceiling - local_lr_floor) * scaled

    def _adaptive_mu(self, drift_score: float, cloud_utility: float) -> float:
        return self.mu_max * self._sigmoid(self.mu_cloud_gain * cloud_utility - self.mu_drift_penalty * drift_score)

    def _fused_prediction(self, global_pred: torch.Tensor, local_pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, float]:
        error_ratio = self.state.online_error_ratio if self.consider_error else None
        fused, _, _, combine_w = self.ensemble_trainer.online_step(
            global_pred=global_pred,
            local_pred=local_pred,
            target=target,
            error_ratio=error_ratio,
        )
        if error_ratio is not None and self.performance_fallback_mix > 0.0:
            perf_weight = torch.as_tensor(error_ratio, device=fused.device, dtype=fused.dtype).reshape(1)
            perf_pred = perf_weight * global_pred + (1.0 - perf_weight) * local_pred
            mix = float(np.clip(self.performance_fallback_mix, 0.0, 1.0))
            fused = (1.0 - mix) * fused + mix * perf_pred

        realized_ratio = None
        if self.consider_error:
            global_error = torch.abs(global_pred - target)
            local_error = torch.abs(local_pred - target)
            global_score = 1.0 / (float(global_error.mean().item()) + float(global_error.std().item()) + self.epsilon)
            local_score = 1.0 / (float(local_error.mean().item()) + float(local_error.std().item()) + self.epsilon)
            realized_ratio = global_score / (local_score + global_score + self.epsilon)
        self.state.online_error_ratio = 0.5 if realized_ratio is None else realized_ratio
        return fused, float(combine_w.reshape(-1)[0].item())

    def _evaluate_snapshot_utility(self, sample: WindowSample, snapshot_state: Dict[str, torch.Tensor]) -> float:
        local_loss = self.local_trainer.loss_on_sample(sample.x, sample.y)
        snapshot_loss = self.global_trainer.loss_on_sample(sample.x, sample.y, state_dict=snapshot_state)
        return max(local_loss - snapshot_loss, 0.0) / (local_loss + self.epsilon)

    def _apply_downloaded_snapshot(self, payload: Dict[str, object]) -> None:
        if not payload.get("available"):
            return
        snapshot_state = decode_state_dict(str(payload["state_dict_b64"]))
        self.global_model.load_state_dict(snapshot_state)
        self.anchor_state = clone_state_dict_cpu(snapshot_state)
        self.state.latest_snapshot_id = str(payload.get("snapshot_id"))
        self.state.anchor_step = int(payload.get("metadata", {}).get("timestep", self.state.timestep))
        self.state.download_bytes += len(str(payload.get("state_dict_b64", "")).encode("utf-8"))

    def _persist_runtime(self) -> None:
        self.local_store.save_state(self.state)
        self.local_store.save_models(
            local_state=clone_state_dict_cpu(self.local_model.state_dict()),
            global_state=clone_state_dict_cpu(self.global_model.state_dict()),
            anchor_state=clone_state_dict_cpu(self.anchor_state),
            local_buffer=self.local_buffer,
            global_buffer=self.global_buffer,
        )
        ensemble_state = self.ensemble_trainer.state_dict()
        ensemble_state["local_decay_rate"] = self.local_decay_rate.detach().cpu()
        ensemble_state["global_decay_rate"] = self.global_decay_rate.detach().cpu()
        self.local_store.save_ensemble_state(ensemble_state)

    def _handle_sample(self, sample: WindowSample) -> None:
        step_start = perf_counter()
        predict_start = perf_counter()
        target = torch.from_numpy(sample.y[:, self.stream.target_idx]).float()
        local_pred = self.local_trainer.predict(sample.x, sample.y)
        global_pred = self.global_trainer.predict(sample.x, sample.y)
        inc_local_pred = self._buffered_prediction(
            self.local_buffer,
            clone_state_dict_cpu(self.local_model.state_dict()),
            sample,
            self.local_decay_rate,
            self.local_decay_lr,
            update_decay=True,
        )
        inc_global_pred = self._buffered_prediction(
            self.global_buffer,
            clone_state_dict_cpu(self.global_model.state_dict()),
            sample,
            self.global_decay_rate,
            self.global_decay_lr,
            update_decay=True,
        )
        predict_ms = (perf_counter() - predict_start) * 1000.0

        drift_start = perf_counter()
        local_loss = float(torch.mean((local_pred - target) ** 2).item())
        global_loss = float(torch.mean((global_pred - target) ** 2).item())
        loss_shock = self._update_loss_shock(local_loss)
        disagreement = float(torch.mean(torch.abs(inc_local_pred - inc_global_pred)).item())
        drift_ms = (perf_counter() - drift_start) * 1000.0

        fusion_start = perf_counter()
        fused_pred, _ = self._fused_prediction(inc_global_pred, inc_local_pred, target)
        fusion_ms = (perf_counter() - fusion_start) * 1000.0

        drift_post_fusion_start = perf_counter()
        residual = float(torch.mean(torch.abs(fused_pred - target)).item())
        self.state.residuals.append(residual)
        if len(self.state.residuals) > self.recent_window:
            self.state.residuals = self.state.residuals[-self.recent_window :]
        residual_volatility = self._residual_volatility()
        sync_age = min((sample.target_start_index - self.state.last_sync_step) / max(self.age_horizon, 1), 1.0)
        drift_score = (
            self.drift_weight_loss * loss_shock
            + self.drift_weight_disagreement * disagreement
            + self.drift_weight_volatility * residual_volatility
            + self.drift_weight_age * sync_age
        )
        resource_pressure = self._resource_pressure()
        urgency = drift_score / (1.0 + self.beta_resource * resource_pressure)
        drift_ms += (perf_counter() - drift_post_fusion_start) * 1000.0

        heartbeat_start = perf_counter()
        heartbeat_resources = self._collect_resource_stats()
        self.sync_client.heartbeat(
            {
                "client_id": self.client_id,
                "timestep": sample.target_start_index,
                "resources": {
                    "network_type": "ethernet",
                    "cpu_percent": heartbeat_resources["system_cpu_percent"],
                    "memory_percent": heartbeat_resources["system_memory_percent"],
                },
                "latest_mae": float(torch.mean(torch.abs(fused_pred - target)).item()),
                "latest_mse": float(torch.mean((fused_pred - target) ** 2).item()),
            }
        )
        heartbeat_ms = (perf_counter() - heartbeat_start) * 1000.0

        selected_for_sync = False
        cloud_utility = 0.0
        downlink = False
        decision: Dict[str, object] | None = None
        sync_request_ms = 0.0
        snapshot_fetch_ms = 0.0
        upload_ms = 0.0
        staged_download_ms = 0.0
        if urgency > self.sync_threshold:
            sync_request_start = perf_counter()
            decision = self.sync_client.request_sync(
                {
                    "client_id": self.client_id,
                    "timestep": sample.target_start_index,
                    "drift_score": drift_score,
                    "urgency": urgency,
                    "local_loss": local_loss,
                    "global_loss": global_loss,
                    "resource_pressure": resource_pressure,
                }
            )
            sync_request_ms = (perf_counter() - sync_request_start) * 1000.0
            selected_for_sync = bool(decision.get("accepted", False))
            snapshot_id = decision.get("snapshot_id")
            if selected_for_sync and snapshot_id:
                snapshot_fetch_start = perf_counter()
                snapshot_payload = self.sync_client.fetch_snapshot(str(snapshot_id))
                snapshot_fetch_ms = (perf_counter() - snapshot_fetch_start) * 1000.0
                if snapshot_payload.get("available"):
                    snapshot_state = decode_state_dict(str(snapshot_payload["state_dict_b64"]))
                    cloud_utility = self._evaluate_snapshot_utility(sample, snapshot_state)
                    if drift_score > self.downlink_drift_threshold and cloud_utility > self.downlink_utility_threshold:
                        blended = blend_state_dicts(
                            clone_state_dict_cpu(self.global_model.state_dict()),
                            snapshot_state,
                            self.snapshot_blend,
                        )
                        self.global_model.load_state_dict(blended)
                        self.anchor_state = clone_state_dict_cpu(blended)
                        self.state.anchor_step = int(snapshot_payload.get("metadata", {}).get("timestep", self.state.anchor_step))
                        self.state.download_bytes += len(str(snapshot_payload.get("state_dict_b64", "")).encode("utf-8"))
                        downlink = True

        local_lr = self._adaptive_local_lr(drift_score)
        mu_t = self._adaptive_mu(drift_score, cloud_utility)
        local_update_start = perf_counter()
        self.local_trainer.online_update(
            sample.x,
            sample.y,
            lr=local_lr,
            proximal_state=self.anchor_state if downlink else None,
            mu=mu_t if downlink else 0.0,
        )
        local_update_ms = (perf_counter() - local_update_start) * 1000.0
        global_lr = self.online_lr if selected_for_sync else self.online_lr * self.unsynced_global_lr_scale
        global_update_ms = 0.0
        if global_lr > 0.0:
            global_update_start = perf_counter()
            self.global_trainer.online_update(sample.x, sample.y, lr=global_lr)
            global_update_ms = (perf_counter() - global_update_start) * 1000.0

        self.local_buffer.append(clone_state_dict_cpu(self.local_model.state_dict()))
        self.global_buffer.append(clone_state_dict_cpu(self.global_model.state_dict()))

        if selected_for_sync:
            delta_state = state_dict_difference(clone_state_dict_cpu(self.global_model.state_dict()), self.anchor_state)
            payload_b64 = encode_state_dict(delta_state)
            staleness = max(sample.target_start_index - self.state.anchor_step, 0) / max(self.pred_len, 1)
            upload_start = perf_counter()
            self.sync_client.upload_update(
                {
                    "client_id": self.client_id,
                    "timestep": sample.target_start_index,
                    "drift_score": drift_score,
                    "cloud_utility": cloud_utility,
                    "volatility": residual_volatility,
                    "staleness": staleness,
                    "state_dict_b64": payload_b64,
                    "metadata": {"timestamp": sample.timestamp},
                }
            )
            upload_ms = (perf_counter() - upload_start) * 1000.0
            self.state.upload_bytes += len(payload_b64.encode("utf-8"))
            staged_download_start = perf_counter()
            staged_download = self.sync_client.download_staged(self.client_id)
            staged_download_ms = (perf_counter() - staged_download_start) * 1000.0
            if staged_download.get("available"):
                self._apply_downloaded_snapshot(staged_download)
            self.state.last_sync_step = sample.target_start_index
            self.state.sync_history.append(1)
        else:
            self.state.sync_history.append(0)
        if len(self.state.sync_history) > self.resource_window:
            self.state.sync_history = self.state.sync_history[-self.resource_window :]

        self.state.timestep = sample.target_start_index
        self.state.latest_drift_score = drift_score
        self.state.latest_urgency = urgency
        self.state.latest_cloud_utility = cloud_utility

        edge_abs_error_sum = float(torch.sum(torch.abs(inc_local_pred - target)).item())
        edge_sq_error_sum = float(torch.sum((inc_local_pred - target) ** 2).item())
        edge_mae = float(torch.mean(torch.abs(inc_local_pred - target)).item())
        edge_mse = float(torch.mean((inc_local_pred - target) ** 2).item())
        cloud_abs_error_sum = float(torch.sum(torch.abs(inc_global_pred - target)).item())
        cloud_sq_error_sum = float(torch.sum((inc_global_pred - target) ** 2).item())
        cloud_mae = float(torch.mean(torch.abs(inc_global_pred - target)).item())
        cloud_mse = float(torch.mean((inc_global_pred - target) ** 2).item())
        abs_error_sum = float(torch.sum(torch.abs(fused_pred - target)).item())
        sq_error_sum = float(torch.sum((fused_pred - target) ** 2).item())
        sample_count = int(target.numel())

        total_step_ms = (perf_counter() - step_start) * 1000.0
        resource_stats = self._collect_resource_stats()
        metrics_metadata = {
            "timestamp": sample.timestamp,
            "selected_for_sync": selected_for_sync,
            "predict_ms": predict_ms,
            "drift_ms": drift_ms,
            "fusion_ms": fusion_ms,
            "heartbeat_ms": heartbeat_ms,
            "sync_request_ms": sync_request_ms,
            "snapshot_fetch_ms": snapshot_fetch_ms,
            "local_update_ms": local_update_ms,
            "global_update_ms": global_update_ms,
            "upload_ms": upload_ms,
            "staged_download_ms": staged_download_ms,
            "total_step_ms": total_step_ms,
            "edge_mae": edge_mae,
            "edge_mse": edge_mse,
            "edge_abs_error_sum": edge_abs_error_sum,
            "edge_sq_error_sum": edge_sq_error_sum,
            "cloud_mae": cloud_mae,
            "cloud_mse": cloud_mse,
            "cloud_abs_error_sum": cloud_abs_error_sum,
            "cloud_sq_error_sum": cloud_sq_error_sum,
            "system_cpu_percent": resource_stats["system_cpu_percent"],
            "system_memory_percent": resource_stats["system_memory_percent"],
            "process_cpu_percent": resource_stats["process_cpu_percent"],
            "process_rss_mb": resource_stats["process_rss_mb"],
            "process_memory_percent": resource_stats["process_memory_percent"],
            "process_threads": resource_stats["process_threads"],
            "peak_rss_mb": resource_stats["peak_rss_mb"],
            "loadavg_1m": resource_stats["loadavg_1m"],
            "loadavg_5m": resource_stats["loadavg_5m"],
            "loadavg_15m": resource_stats["loadavg_15m"],
        }

        self.sync_client.report_metrics(
            {
                "client_id": self.client_id,
                "timestep": sample.target_start_index,
                "mae": float(torch.mean(torch.abs(fused_pred - target)).item()),
                "mse": float(torch.mean((fused_pred - target) ** 2).item()),
                "abs_error_sum": abs_error_sum,
                "sq_error_sum": sq_error_sum,
                "sample_count": sample_count,
                "upload_bytes": self.state.upload_bytes,
                "download_bytes": self.state.download_bytes,
                "inference_latency_ms": total_step_ms,
                "metadata": metrics_metadata,
            }
        )
        self._persist_runtime()