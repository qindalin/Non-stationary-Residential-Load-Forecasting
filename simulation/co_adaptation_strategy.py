import copy
import json
import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn, optim

from online_adaptation import EdgeAdapter
from utils.utils import get_tensor_batch
from utils.utils import inverse_with_scaler, safe_epe


def _clone_state_dict_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def _clone_batch_cpu(batch: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    return tuple(item.detach().cpu().clone() for item in batch)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


@dataclass
class StrategyConfig:
    edge_adapter: str = "online"
    edge_memory_size: int = 7
    edge_replay_size: int = 2
    edge_replay_weight: float = 0.5
    edge_distill_weight: float = 0.5
    edge_adapter_seed: int = 0
    enable_cloud_collaboration: bool = True
    recent_window: int = 6
    resource_window: int = 12
    cloud_memory_size: int = 5
    sync_budget_clients: int = 4
    max_sync_budget_clients: Optional[int] = None
    adaptive_budget_gain: float = 0.0
    residual_priority_weight: float = 0.0
    forced_refresh_rounds: int = 0
    sync_threshold: float = 0.80
    downlink_drift_threshold: float = 0.60
    downlink_utility_threshold: float = 0.01
    stale_penalty: float = 0.01
    stale_gamma: float = 1.0
    server_lambda: float = 0.5
    eta_min_scale: float = 0.5
    eta_max_scale: float = 2.0
    eta_slope: float = 2.0
    eta_bias: float = 0.5
    mu_max: float = 0.05
    mu_cloud_gain: float = 4.0
    mu_drift_penalty: float = 2.5
    rho_m: float = 0.9
    rho_v: float = 0.9
    age_horizon: int = 12
    beta_resource: float = 1.0
    drift_weight_loss: float = 0.40
    drift_weight_disagreement: float = 0.30
    drift_weight_volatility: float = 0.20
    drift_weight_age: float = 0.10
    agg_beta_drift: float = 1.0
    agg_beta_utility: float = 0.75
    agg_beta_volatility: float = 0.50
    agg_beta_staleness: float = 0.25
    grad_clip_norm: Optional[float] = None
    snapshot_blend: float = 1.0
    performance_fallback_mix: float = 0.0
    unsynced_global_lr_scale: float = 1.0
    post_sync_broadcast: bool = True
    use_latest_cloud_snapshot: bool = False
    retrieval_policy: str = "historical_match"
    retrieval_seed: int = 0
    downlink_loss_rate: float = 0.0
    network_seed: int = 0
    edge_only_on_downlink_failure: bool = False
    max_steps: Optional[int] = None
    log_every: int = 10
    decay_update: bool = False
    adaptive_fusion: bool = True
    epsilon: float = 1e-6


@dataclass
class ClientRuntimeState:
    client_name: str
    loss_mean: Optional[float] = None
    loss_var: float = 1e-6
    residuals: Deque[float] = field(default_factory=deque)
    recent_batches: Deque[Tuple[torch.Tensor, ...]] = field(default_factory=deque)
    sync_history: Deque[int] = field(default_factory=deque)
    last_sync_step: int = 0
    anchor_step: int = 0
    cloud_anchor_state: Optional[Dict[str, torch.Tensor]] = None
    latest_drift_score: float = 0.0
    latest_urgency: float = 0.0
    latest_cloud_utility: float = 0.0
    latest_disagreement: float = 0.0
    latest_volatility: float = 0.0
    sync_count: int = 0
    upload_bytes: int = 0
    download_bytes: int = 0
    downlink_attempts: int = 0
    successful_downlinks: int = 0
    failed_downlinks: int = 0
    attempted_download_bytes: int = 0
    last_successful_downlink_step: int = 0
    cloud_available: bool = True
    selected_candidate_loss: float = 0.0
    latest_candidate_loss: float = 0.0
    random_candidate_loss: float = 0.0
    selected_steps: List[int] = field(default_factory=list)
    drift_scores: List[float] = field(default_factory=list)
    urgencies: List[float] = field(default_factory=list)
    cloud_utilities: List[float] = field(default_factory=list)
    local_losses: List[float] = field(default_factory=list)
    global_losses: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.residuals:
            self.residuals = deque(maxlen=1)
        if not self.recent_batches:
            self.recent_batches = deque(maxlen=1)
        if not self.sync_history:
            self.sync_history = deque(maxlen=1)


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


class ActiveCoAdaptationRunner:
    def __init__(
        self,
        clients: Sequence,
        server,
        args,
        strategy_config: StrategyConfig,
        output_dir: Path,
        writer=None,
        logger: Optional[logging.Logger] = None,
        sync_schedule: Optional[Dict[int, Sequence[str]]] = None,
        downlink_schedule: Optional[Set[Tuple[int, str]]] = None,
    ) -> None:
        self.clients = list(clients)
        self.server = server
        self.args = args
        self.config = strategy_config
        self.output_dir = Path(output_dir)
        self.writer = writer if writer is not None else NullSummaryWriter()
        self.logger = logger or logging.getLogger(__name__)
        self.device = self.server.device
        self.criterion = nn.MSELoss()
        self.edge_adapter = EdgeAdapter(
            name=self.config.edge_adapter,
            memory_size=self.config.edge_memory_size,
            replay_size=self.config.edge_replay_size,
            replay_weight=self.config.edge_replay_weight,
            distill_weight=self.config.edge_distill_weight,
            seed=self.config.edge_adapter_seed,
        )
        self.global_lr = self.args.online_lr
        self.local_lr_floor = self.args.online_lr * self.config.eta_min_scale
        self.local_lr_ceiling = self.args.online_lr * self.config.eta_max_scale
        self.cloud_memory: Deque[Tuple[int, Dict[str, torch.Tensor]]] = deque(
            maxlen=self.config.cloud_memory_size
        )
        self.runtime_states: Dict[str, ClientRuntimeState] = {}
        self.client_lookup = {client.data_name: client for client in self.clients}
        self.parameter_bytes = self._estimate_model_bytes(self.server.global_model.state_dict())
        if not 0.0 <= self.config.downlink_loss_rate <= 1.0:
            raise ValueError("downlink_loss_rate must be in [0, 1].")
        if self.config.retrieval_policy not in {"historical_match", "latest", "random"}:
            raise ValueError("retrieval_policy must be one of: historical_match, latest, random.")
        self.network_rng = np.random.default_rng(self.config.network_seed)
        self.retrieval_rng = np.random.default_rng(self.config.retrieval_seed)
        self.diagnostic_rng = np.random.default_rng(self.config.retrieval_seed + 1000003)
        self.sync_schedule = {
            int(timestep): set(client_names)
            for timestep, client_names in (sync_schedule or {}).items()
        }
        self.downlink_schedule = downlink_schedule
        self.global_metrics: List[Dict[str, float]] = []
        self.client_step_metrics: List[Dict[str, float]] = []
        self.sync_records: List[Dict[str, float]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_runtime_states()

    def _initialize_runtime_states(self) -> None:
        initial_global_state = _clone_state_dict_cpu(self.server.global_model.state_dict())
        self.cloud_memory.append((0, initial_global_state))
        for client in self.clients:
            client.init_weight()
            client.set_model(initial_global_state, type="global")
            state = ClientRuntimeState(
                client_name=client.data_name,
                residuals=deque(maxlen=self.config.recent_window),
                recent_batches=deque(maxlen=self.config.recent_window),
                sync_history=deque(maxlen=self.config.resource_window),
                cloud_anchor_state=initial_global_state,
                cloud_available=not (
                    self.config.edge_only_on_downlink_failure
                    and self.config.downlink_loss_rate >= 1.0
                ),
            )
            self.runtime_states[client.data_name] = state

    def _blend_state_dicts(
        self,
        base_state: Dict[str, torch.Tensor],
        incoming_state: Dict[str, torch.Tensor],
        blend: float,
    ) -> Dict[str, torch.Tensor]:
        if blend >= 1.0:
            return _clone_state_dict_cpu(incoming_state)
        if blend <= 0.0:
            return _clone_state_dict_cpu(base_state)
        return {
            name: (1.0 - blend) * base_state[name].detach().cpu() + blend * incoming_state[name].detach().cpu()
            for name in base_state.keys()
        }

    def run(self) -> Dict[str, pd.DataFrame]:
        test_timesteps = len(self.clients[0].test_loader.dataset)
        timesteps = list(range(0, test_timesteps, self.args.pred_len))
        if self.config.max_steps is not None:
            timesteps = timesteps[: self.config.max_steps]

        for step_idx, timestep in enumerate(timesteps):
            step_contexts = []
            for client in self.clients:
                context = self._collect_step_context(client, timestep)
                step_contexts.append(context)

            selected_names = self._select_clients(step_contexts)
            uploaded_updates = []

            for context in step_contexts:
                update = self._adapt_client(context, selected_names)
                if update is not None:
                    uploaded_updates.append(update)

            if uploaded_updates:
                self._aggregate_cloud_updates(timestep, uploaded_updates)
            else:
                self.cloud_memory.append(
                    (timestep, _clone_state_dict_cpu(self.server.global_model.state_dict()))
                )

            self._log_step(step_idx, timestep, selected_names)

        result_tables = self._finalize_outputs()
        self.writer.close()
        return result_tables

    def _collect_step_context(self, client, timestep: int) -> Dict[str, object]:
        state = self.runtime_states[client.data_name]
        batch = get_tensor_batch(client.test_loader.dataset, timestep)
        state.recent_batches.append(_clone_batch_cpu(batch))

        local_pred_tensor, target_tensor = self._predict_current(client.local_trainer, batch)
        if self.config.enable_cloud_collaboration:
            global_pred_tensor, _ = self._predict_current(client.global_trainer, batch)
            inc_global_pred = client.incremental_model_ensemble(
                batch, decay_update=self.config.decay_update, type="global"
            )
            inc_local_pred = client.incremental_model_ensemble(
                batch, decay_update=self.config.decay_update, type="local"
            )
        else:
            global_pred_tensor = local_pred_tensor
            inc_global_pred = local_pred_tensor
            inc_local_pred = local_pred_tensor

        local_loss = float(self.criterion(local_pred_tensor, target_tensor).item())
        global_loss = float(self.criterion(global_pred_tensor, target_tensor).item())
        local_loss_shock = self._update_loss_shock(state, local_loss)
        disagreement = self._compute_disagreement(inc_local_pred, inc_global_pred)
        true_np = target_tensor.detach().cpu().numpy()
        if self.config.enable_cloud_collaboration and state.cloud_available:
            ensemble_pred, long_w, short_w, combine_w = self._run_fusion_step(
                client,
                batch,
                inc_global_pred,
                inc_local_pred,
                true_np,
            )
        else:
            ensemble_pred = local_pred_tensor.detach()
            long_w = torch.zeros(1, device=self.device)
            short_w = torch.zeros(1, device=self.device)
            combine_w = torch.zeros(1, device=self.device)

        residual = float(
            np.mean(
                np.abs(ensemble_pred.detach().cpu().numpy() - true_np)
            )
        )
        state.residuals.append(residual)
        residual_volatility = self._compute_residual_volatility(state)
        sync_age = min(
            (timestep - state.last_sync_step) / max(self.config.age_horizon, 1),
            1.0,
        )
        drift_score = (
            self.config.drift_weight_loss * local_loss_shock
            + self.config.drift_weight_disagreement * disagreement
            + self.config.drift_weight_volatility * residual_volatility
            + self.config.drift_weight_age * sync_age
        )
        resource_pressure = self._resource_pressure(state)
        urgency = drift_score / (1.0 + self.config.beta_resource * resource_pressure)
        request_sync = self.config.enable_cloud_collaboration and urgency > self.config.sync_threshold

        state.latest_drift_score = drift_score
        state.latest_urgency = urgency
        state.latest_disagreement = disagreement
        state.latest_volatility = residual_volatility
        state.drift_scores.append(drift_score)
        state.urgencies.append(urgency)
        state.local_losses.append(local_loss)
        state.global_losses.append(global_loss)

        self._record_predictions(
            client=client,
            local_pred=local_pred_tensor.detach().cpu().numpy(),
            global_pred=global_pred_tensor.detach().cpu().numpy(),
            inc_local_pred=inc_local_pred.detach().cpu().numpy(),
            inc_global_pred=inc_global_pred.detach().cpu().numpy(),
            ensemble_pred=ensemble_pred.detach().cpu().numpy(),
            true_np=true_np,
            long_w=long_w,
            short_w=short_w,
            combine_w=combine_w,
        )

        return {
            "client": client,
            "state": state,
            "batch": batch,
            "timestep": timestep,
            "local_loss": local_loss,
            "global_loss": global_loss,
            "drift_score": drift_score,
            "urgency": urgency,
            "request_sync": request_sync,
            "local_pred": local_pred_tensor,
            "global_pred": global_pred_tensor,
            "target": target_tensor,
        }

    def _adapt_client(
        self,
        context: Dict[str, object],
        selected_names: Sequence[str],
    ) -> Optional[Dict[str, object]]:
        client = context["client"]
        state = context["state"]
        batch = context["batch"]
        timestep = context["timestep"]
        drift_score = float(context["drift_score"])
        selected_for_sync = client.data_name in selected_names
        selected_snapshot = None
        selected_snapshot_step = state.anchor_step
        cloud_utility = 0.0
        downlink_requested = False
        downlink = False
        download_bytes_before = state.download_bytes

        if selected_for_sync:
            selected_snapshot_step, selected_snapshot, cloud_utility = self._select_cloud_snapshot(
                client,
                state,
                timestep,
            )
            state.latest_cloud_utility = cloud_utility
            state.cloud_utilities.append(cloud_utility)
            if self.downlink_schedule is None:
                downlink_requested = (
                    drift_score > self.config.downlink_drift_threshold
                    and cloud_utility > self.config.downlink_utility_threshold
                )
            else:
                downlink_requested = (timestep, client.data_name) in self.downlink_schedule
            if downlink_requested:
                downlink = self._deliver_cloud_snapshot(
                    client=client,
                    state=state,
                    snapshot=selected_snapshot,
                    snapshot_step=selected_snapshot_step,
                    timestep=timestep,
                )
        download_bytes = state.download_bytes - download_bytes_before

        local_lr = self._adaptive_local_lr(drift_score)
        mu_t = self._adaptive_mu(drift_score, cloud_utility)
        if self.config.edge_only_on_downlink_failure and not state.cloud_available:
            self._edge_update(client, batch, self.global_lr)
            local_lr = self.global_lr
            mu_t = 0.0
        else:
            if self.config.edge_adapter == "online":
                self._online_update(
                    trainer=client.local_trainer,
                    batch=batch,
                    lr=local_lr,
                    proximal_state=state.cloud_anchor_state if downlink else None,
                    mu=mu_t if downlink else 0.0,
                )
            else:
                self._edge_update(client, batch, local_lr)
        global_lr = self.global_lr
        if not selected_for_sync:
            global_lr = self.global_lr * self.config.unsynced_global_lr_scale
        if self.config.enable_cloud_collaboration and global_lr > 0.0:
            self._online_update(
                trainer=client.global_trainer,
                batch=batch,
                lr=global_lr,
                proximal_state=None,
                mu=0.0,
            )

        client.local_model_buffer.append(_clone_state_dict_cpu(client.local_model.state_dict()))
        if len(client.local_model_buffer) > self.args.model_buffer_size:
            client.local_model_buffer.pop(0)
        if self.config.enable_cloud_collaboration:
            client.global_model_buffer.append(_clone_state_dict_cpu(client.global_model.state_dict()))
            if len(client.global_model_buffer) > self.args.model_buffer_size:
                client.global_model_buffer.pop(0)

        if not selected_for_sync:
            state.sync_history.append(0)
            return None

        anchor_state = state.cloud_anchor_state
        delta_state = self._state_dict_difference(
            client.global_model.state_dict(), anchor_state
        )
        staleness = max(timestep - state.anchor_step, 0) / max(self.args.pred_len, 1)
        payload_bytes = self.parameter_bytes
        state.sync_history.append(1)
        state.sync_count += 1
        state.last_sync_step = timestep
        state.upload_bytes += payload_bytes
        state.selected_steps.append(timestep)

        self.sync_records.append(
            {
                "client": client.data_name,
                "timestep": timestep,
                "drift_score": drift_score,
                "urgency": float(context["urgency"]),
                "cloud_utility": cloud_utility,
                "downlink_requested": int(downlink_requested),
                "downlink": int(downlink),
                "downlink_failed": int(downlink_requested and not downlink),
                "upload_bytes": payload_bytes,
                "download_bytes": download_bytes,
                "staleness": staleness,
                "selected_snapshot_step": selected_snapshot_step,
                "selected_snapshot_age_rounds": max(timestep - selected_snapshot_step, 0)
                / max(self.args.pred_len, 1),
                "selected_candidate_loss": state.selected_candidate_loss,
                "latest_candidate_loss": state.latest_candidate_loss,
                "random_candidate_loss": state.random_candidate_loss,
                "local_lr": local_lr,
                "global_lr": global_lr,
                "mu_t": mu_t,
            }
        )
        return {
            "client_name": client.data_name,
            "delta_state": delta_state,
            "drift_score": drift_score,
            "cloud_utility": cloud_utility,
            "volatility": state.latest_volatility,
            "staleness": staleness,
        }

    def _edge_update(self, client, batch: Sequence[torch.Tensor], lr: float) -> None:
        self.edge_adapter.update(
            client_name=client.data_name,
            trainer=client.local_trainer,
            batch=batch,
            lr=lr,
            grad_clip_norm=self.config.grad_clip_norm,
        )

    def _deliver_cloud_snapshot(
        self,
        client,
        state: ClientRuntimeState,
        snapshot: Dict[str, torch.Tensor],
        snapshot_step: int,
        timestep: int,
    ) -> bool:
        """Atomically install a cloud snapshot only after a successful downlink."""
        state.downlink_attempts += 1
        state.attempted_download_bytes += self.parameter_bytes
        if self.network_rng.random() < self.config.downlink_loss_rate:
            state.failed_downlinks += 1
            if self.config.edge_only_on_downlink_failure:
                state.cloud_available = False
            return False

        blended_snapshot = self._blend_state_dicts(
            client.global_model.state_dict(),
            snapshot,
            self.config.snapshot_blend,
        )
        client.set_model(blended_snapshot, type="global")
        state.cloud_anchor_state = blended_snapshot
        state.anchor_step = snapshot_step
        state.last_successful_downlink_step = timestep
        state.cloud_available = True
        state.successful_downlinks += 1
        state.download_bytes += self.parameter_bytes
        return True

    def _aggregate_cloud_updates(
        self,
        timestep: int,
        updates: Sequence[Dict[str, object]],
    ) -> None:
        weights = self._compute_aggregation_weights(updates)
        global_state = _clone_state_dict_cpu(self.server.global_model.state_dict())
        coeff = 2.0 / (self.config.server_lambda + 2.0)
        new_state = {}
        for name, tensor in global_state.items():
            aggregated_delta = torch.zeros_like(tensor)
            for weight, update in zip(weights, updates):
                delta_state = update["delta_state"]
                stale_delta = delta_state[name] * (
                    (1.0 + float(update["staleness"])) ** (-self.config.stale_gamma)
                )
                aggregated_delta += weight * stale_delta
            new_state[name] = tensor + coeff * aggregated_delta

        self.server.set_model(new_state)
        self.cloud_memory.append((timestep, _clone_state_dict_cpu(new_state)))
        if self.config.post_sync_broadcast:
            for update in updates:
                client_name = str(update["client_name"])
                client = self.client_lookup[client_name]
                state = self.runtime_states[client_name]
                self._deliver_cloud_snapshot(
                    client=client,
                    state=state,
                    snapshot=new_state,
                    snapshot_step=timestep,
                    timestep=timestep,
                )

    def _compute_aggregation_weights(
        self, updates: Sequence[Dict[str, object]]
    ) -> List[float]:
        logits = []
        for update in updates:
            logits.append(
                self.config.agg_beta_drift * float(update["drift_score"])
                + self.config.agg_beta_utility * float(update["cloud_utility"])
                - self.config.agg_beta_volatility * float(update["volatility"])
                - self.config.agg_beta_staleness * float(update["staleness"])
            )
        logits = np.array(logits, dtype=np.float64)
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return probs.astype(np.float32).tolist()

    def _predict_current(self, trainer, batch: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        with torch.no_grad():
            outputs = trainer.backbone_forward(batch_x, batch_y, batch_x_mark, batch_y_mark)
        target = batch_y[:, -self.args.pred_len :, -1].float().to(self.device)
        return outputs, target

    def _run_fusion_step(
        self,
        client,
        batch: Sequence[torch.Tensor],
        inc_global_pred: torch.Tensor,
        inc_local_pred: torch.Tensor,
        true_np: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        error_ratio = None
        if self.config.adaptive_fusion and self.args.consider_error:
            global_error_mean = np.mean(np.abs(inc_global_pred.detach().cpu().numpy() - true_np))
            local_error_mean = np.mean(np.abs(inc_local_pred.detach().cpu().numpy() - true_np))
            global_error_std = np.std(inc_global_pred.detach().cpu().numpy() - true_np)
            local_error_std = np.std(inc_local_pred.detach().cpu().numpy() - true_np)
            local_score = 1.0 / (local_error_mean + local_error_std + self.config.epsilon)
            global_score = 1.0 / (global_error_mean + global_error_std + self.config.epsilon)
            client.online_global_local_error_ratio = global_score / (
                local_score + global_score + self.config.epsilon
            )
            error_ratio = client.online_global_local_error_ratio

        outputs, long_w, short_w, combine_w = client.ensemble_trainer.online_step(
            batch,
            inc_global_pred,
            inc_local_pred,
            error_ratio=error_ratio,
        )
        if not self.config.adaptive_fusion:
            outputs = 0.5 * inc_global_pred.detach() + 0.5 * inc_local_pred.detach()
        if error_ratio is not None and self.config.performance_fallback_mix > 0.0:
            perf_weight = torch.as_tensor(
                error_ratio,
                device=outputs.device,
                dtype=outputs.dtype,
            )
            perf_pred = perf_weight * inc_global_pred.detach() + (1.0 - perf_weight) * inc_local_pred.detach()
            mix = float(np.clip(self.config.performance_fallback_mix, 0.0, 1.0))
            outputs = (1.0 - mix) * outputs + mix * perf_pred
        return outputs, long_w, short_w, combine_w

    def _record_predictions(
        self,
        client,
        local_pred: np.ndarray,
        global_pred: np.ndarray,
        inc_local_pred: np.ndarray,
        inc_global_pred: np.ndarray,
        ensemble_pred: np.ndarray,
        true_np: np.ndarray,
        long_w: torch.Tensor,
        short_w: torch.Tensor,
        combine_w: torch.Tensor,
    ) -> None:
        client.online_local_preds.append(local_pred)
        client.online_global_preds.append(global_pred)
        client.online_inc_local_preds.append(inc_local_pred)
        client.online_inc_global_preds.append(inc_global_pred)
        client.online_ensemble_preds.append(ensemble_pred)
        client.online_trues.append(true_np)
        client.online_long_weight.append(long_w.detach().cpu().numpy())
        client.online_short_weight.append(short_w.detach().cpu().numpy())
        client.online_weight.append(combine_w.detach().cpu().numpy().reshape(-1))

    def _select_clients(self, step_contexts: Sequence[Dict[str, object]]) -> List[str]:
        if self.sync_schedule:
            timestep = int(step_contexts[0]["timestep"])
            return [
                client.data_name
                for client in self.clients
                if client.data_name in self.sync_schedule.get(timestep, set())
            ]
        requested = [ctx for ctx in step_contexts if ctx["request_sync"]]
        forced = [ctx for ctx in step_contexts if self._needs_forced_refresh(ctx)]

        merged = {}
        for ctx in requested + forced:
            merged[ctx["client"].data_name] = ctx
        candidates = list(merged.values())
        if not candidates:
            return []

        residuals = np.array(
            [
                float(np.mean(ctx["state"].residuals)) if len(ctx["state"].residuals) > 0 else 0.0
                for ctx in candidates
            ],
            dtype=np.float64,
        )
        residual_scale = residuals.mean() + self.config.epsilon

        def candidate_score(item: Dict[str, object]) -> float:
            state = item["state"]
            residual_mean = float(np.mean(state.residuals)) if len(state.residuals) > 0 else 0.0
            residual_term = self.config.residual_priority_weight * (residual_mean / residual_scale)
            forced_bonus = 1.0 if self._needs_forced_refresh(item) else 0.0
            return float(item["urgency"]) + residual_term + forced_bonus

        candidates.sort(key=candidate_score, reverse=True)

        budget = self.config.sync_budget_clients
        if self.config.adaptive_budget_gain > 0.0 and len(requested) > self.config.sync_budget_clients:
            extra = int(np.ceil(self.config.adaptive_budget_gain * (len(requested) - self.config.sync_budget_clients)))
            budget += extra
        max_budget = self.config.max_sync_budget_clients or len(self.clients)
        budget = min(max_budget, budget)
        budget = max(budget, min(len(forced), len(candidates)))
        selected = candidates[:budget]
        return [ctx["client"].data_name for ctx in selected]

    def _needs_forced_refresh(self, context: Dict[str, object]) -> bool:
        if self.config.forced_refresh_rounds <= 0:
            return False
        state = context["state"]
        timestep = int(context["timestep"])
        rounds_since_sync = max(timestep - state.last_sync_step, 0) / max(self.args.pred_len, 1)
        return rounds_since_sync >= self.config.forced_refresh_rounds

    def _select_cloud_snapshot(
        self,
        client,
        state: ClientRuntimeState,
        timestep: int,
    ) -> Tuple[int, Dict[str, torch.Tensor], float]:
        if not self.cloud_memory:
            return state.anchor_step, state.cloud_anchor_state, 0.0

        local_reference_loss = self._evaluate_model_on_batches(
            client.local_model,
            client.local_trainer,
            list(state.recent_batches),
        )
        policy = "latest" if self.config.use_latest_cloud_snapshot else self.config.retrieval_policy

        latest_step, latest_state = self.cloud_memory[-1]
        latest_model = copy.deepcopy(client.global_model).to(self.device)
        latest_model.load_state_dict(latest_state)
        state.latest_candidate_loss = self._evaluate_model_on_batches(
            latest_model,
            client.global_trainer,
            list(state.recent_batches),
        )
        diagnostic_index = int(self.diagnostic_rng.integers(len(self.cloud_memory)))
        random_step, random_state = self.cloud_memory[diagnostic_index]
        random_model = copy.deepcopy(client.global_model).to(self.device)
        random_model.load_state_dict(random_state)
        state.random_candidate_loss = self._evaluate_model_on_batches(
            random_model,
            client.global_trainer,
            list(state.recent_batches),
        )

        if policy in {"latest", "random"}:
            if policy == "latest":
                snapshot_step, snapshot_state = self.cloud_memory[-1]
            else:
                snapshot_idx = int(self.retrieval_rng.integers(len(self.cloud_memory)))
                snapshot_step, snapshot_state = self.cloud_memory[snapshot_idx]
            snapshot_model = copy.deepcopy(client.global_model).to(self.device)
            snapshot_model.load_state_dict(snapshot_state)
            snapshot_loss = self._evaluate_model_on_batches(
                snapshot_model,
                client.global_trainer,
                list(state.recent_batches),
            )
            state.selected_candidate_loss = snapshot_loss
            cloud_utility = max(local_reference_loss - snapshot_loss, 0.0) / (
                local_reference_loss + self.config.epsilon
            )
            return snapshot_step, snapshot_state, cloud_utility

        best_score = None
        best_step = state.anchor_step
        best_state = state.cloud_anchor_state
        best_loss = local_reference_loss

        for snapshot_step, snapshot_state in self.cloud_memory:
            snapshot_model = copy.deepcopy(client.global_model).to(self.device)
            snapshot_model.load_state_dict(snapshot_state)
            snapshot_loss = self._evaluate_model_on_batches(
                snapshot_model,
                client.global_trainer,
                list(state.recent_batches),
            )
            snapshot_age = max(timestep - snapshot_step, 0) / max(self.args.pred_len, 1)
            score = snapshot_loss + self.config.stale_penalty * snapshot_age
            if best_score is None or score < best_score:
                best_score = score
                best_step = snapshot_step
                best_state = snapshot_state
                best_loss = snapshot_loss

        cloud_utility = max(local_reference_loss - best_loss, 0.0) / (
            local_reference_loss + self.config.epsilon
        )
        state.selected_candidate_loss = best_loss
        return best_step, best_state, cloud_utility

    def _evaluate_model_on_batches(
        self,
        model,
        trainer,
        batches: Sequence[Tuple[torch.Tensor, ...]],
    ) -> float:
        if not batches:
            return 0.0
        losses = []
        model.eval()
        with torch.no_grad():
            for batch in batches:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                outputs = trainer.backbone_forward(
                    batch_x.to(self.device),
                    batch_y.to(self.device),
                    batch_x_mark.to(self.device),
                    batch_y_mark.to(self.device),
                    model=model,
                )
                target = batch_y[:, -self.args.pred_len :, -1].float().to(self.device)
                losses.append(float(self.criterion(outputs, target).item()))
        model.train()
        return float(np.mean(losses))

    def _online_update(
        self,
        trainer,
        batch: Sequence[torch.Tensor],
        lr: float,
        proximal_state: Optional[Dict[str, torch.Tensor]],
        mu: float,
    ) -> Tuple[float, float]:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        target = batch_y[:, -self.args.pred_len :, -1]
        optimizer = optim.AdamW(trainer.model.parameters(), lr=lr)
        optimizer.zero_grad()
        outputs = trainer.backbone_forward(batch_x, batch_y, batch_x_mark, batch_y_mark)
        pred_loss = trainer.criterion(outputs, target)
        energy_weight = float(getattr(self.args, "energy_loss_weight", 0.0) or 0.0)
        energy_loss = torch.zeros(1, device=self.device)
        if energy_weight > 0.0:
            pred_energy = outputs.sum(dim=-1)
            true_energy = target.sum(dim=-1)
            if getattr(self.args, "energy_loss_relative", False):
                denom = torch.clamp(true_energy.abs(), min=1e-6)
                energy_loss = energy_weight * torch.mean(((pred_energy - true_energy) / denom) ** 2)
            else:
                energy_loss = energy_weight * torch.mean((pred_energy - true_energy) ** 2)
        total_loss = pred_loss + energy_loss
        if proximal_state is not None and mu > 0.0:
            prox_term = torch.zeros(1, device=self.device)
            for name, param in trainer.model.named_parameters():
                prox_term = prox_term + torch.sum(
                    (param - proximal_state[name].to(self.device)) ** 2
                )
            total_loss = total_loss + 0.5 * mu * prox_term
        total_loss.backward()
        if self.config.grad_clip_norm is not None and self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), self.config.grad_clip_norm)
        optimizer.step()
        return float(pred_loss.item()), float(total_loss.item())

    def _update_loss_shock(self, state: ClientRuntimeState, local_loss: float) -> float:
        if state.loss_mean is None:
            state.loss_mean = local_loss
            state.loss_var = max(local_loss * local_loss, self.config.epsilon)
            return 0.0
        state.loss_mean = self.config.rho_m * state.loss_mean + (1.0 - self.config.rho_m) * local_loss
        state.loss_var = self.config.rho_v * state.loss_var + (1.0 - self.config.rho_v) * (
            local_loss - state.loss_mean
        ) ** 2
        z_score = (local_loss - state.loss_mean) / (np.sqrt(state.loss_var) + self.config.epsilon)
        return float(max(z_score, 0.0))

    def _compute_disagreement(self, local_pred: torch.Tensor, global_pred: torch.Tensor) -> float:
        return float(
            torch.mean(torch.abs(local_pred.detach() - global_pred.detach())).item()
        )

    def _compute_residual_volatility(self, state: ClientRuntimeState) -> float:
        if len(state.residuals) < 2:
            return 0.0
        residuals = np.array(state.residuals, dtype=np.float64)
        return float(residuals.std() / (residuals.mean() + self.config.epsilon))

    def _resource_pressure(self, state: ClientRuntimeState) -> float:
        if not state.sync_history:
            return 0.0
        return float(np.mean(np.array(state.sync_history, dtype=np.float64)))

    def _adaptive_local_lr(self, drift_score: float) -> float:
        scaled = _sigmoid(self.config.eta_slope * (drift_score - self.config.eta_bias))
        return self.local_lr_floor + (self.local_lr_ceiling - self.local_lr_floor) * scaled

    def _adaptive_mu(self, drift_score: float, cloud_utility: float) -> float:
        return self.config.mu_max * _sigmoid(
            self.config.mu_cloud_gain * cloud_utility
            - self.config.mu_drift_penalty * drift_score
        )

    def _state_dict_difference(
        self,
        current_state: Dict[str, torch.Tensor],
        anchor_state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return {
            name: tensor.detach().cpu() - anchor_state[name]
            for name, tensor in current_state.items()
        }

    def _estimate_model_bytes(self, state_dict: Dict[str, torch.Tensor]) -> int:
        return int(
            sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
        )

    def _log_step(self, step_idx: int, timestep: int, selected_names: Sequence[str]) -> None:
        ensemble_mae = []
        ensemble_mse = []
        ensemble_epe = []
        for client in self.clients:
            if not client.online_ensemble_preds:
                continue
            state = self.runtime_states[client.data_name]
            pred = np.array(client.online_ensemble_preds[-1]).reshape(-1)
            true = np.array(client.online_trues[-1]).reshape(-1)
            client_mae = mean_absolute_error(true, pred)
            client_mse = mean_squared_error(true, pred)
            ensemble_mae.append(client_mae)
            ensemble_mse.append(client_mse)
            pred_inv = inverse_with_scaler(pred, client.scaler).reshape(-1)
            true_inv = inverse_with_scaler(true, client.scaler).reshape(-1)
            client_epe = safe_epe(true_inv, pred_inv)
            ensemble_epe.append(client_epe)
            self.client_step_metrics.append(
                {
                    "step": step_idx,
                    "timestep": timestep,
                    "client": client.data_name,
                    "ensemble_mae": client_mae,
                    "ensemble_mse": client_mse,
                    "ensemble_epe": client_epe,
                    "drift_score": state.latest_drift_score,
                    "selected_for_sync": int(client.data_name in selected_names),
                }
            )

        avg_mae = float(np.mean(ensemble_mae)) if ensemble_mae else 0.0
        avg_mse = float(np.mean(ensemble_mse)) if ensemble_mse else 0.0
        avg_epe = float(np.mean(ensemble_epe)) if ensemble_epe else 0.0
        avg_drift = float(
            np.mean([state.latest_drift_score for state in self.runtime_states.values()])
        )
        gpu_mem_mb = 0.0
        if torch.cuda.is_available():
            gpu_mem_mb = float(torch.cuda.memory_allocated(self.device) / (1024 ** 2))

        self.global_metrics.append(
            {
                "step": step_idx,
                "timestep": timestep,
                "avg_mae": avg_mae,
                "avg_mse": avg_mse,
                "avg_epe": avg_epe,
                "avg_drift": avg_drift,
                "selected_clients": len(selected_names),
                "gpu_mem_mb": gpu_mem_mb,
            }
        )
        self.writer.add_scalar("coadapt/avg_mae", avg_mae, step_idx)
        self.writer.add_scalar("coadapt/avg_mse", avg_mse, step_idx)
        self.writer.add_scalar("coadapt/avg_epe", avg_epe, step_idx)
        self.writer.add_scalar("coadapt/avg_drift", avg_drift, step_idx)
        self.writer.add_scalar("coadapt/selected_clients", len(selected_names), step_idx)
        self.writer.add_scalar("system/gpu_mem_mb", gpu_mem_mb, step_idx)

        if step_idx % max(self.config.log_every, 1) == 0:
            self.logger.info(
                "Step %d | timestep=%d | selected=%s | avg_mae=%.4f | avg_mse=%.4f | avg_drift=%.4f | gpu_mem_mb=%.1f",
                step_idx,
                timestep,
                ",".join(selected_names) if selected_names else "none",
                avg_mae,
                avg_mse,
                avg_drift,
                gpu_mem_mb,
            )

    def _finalize_outputs(self) -> Dict[str, pd.DataFrame]:
        per_client_rows = []
        communication_rows = []
        for client in self.clients:
            true = np.concatenate(client.online_trues, axis=0).reshape(-1)
            ensemble_pred = np.concatenate(client.online_ensemble_preds, axis=0).reshape(-1)
            local_pred = np.concatenate(client.online_local_preds, axis=0).reshape(-1)
            global_pred = np.concatenate(client.online_global_preds, axis=0).reshape(-1)
            state = self.runtime_states[client.data_name]

            per_client_rows.append(
                {
                    "Client": client.data_name,
                    "Ensemble_MSE": mean_squared_error(true, ensemble_pred),
                    "Ensemble_MAE": mean_absolute_error(true, ensemble_pred),
                    "Ensemble_EPE": safe_epe(
                        inverse_with_scaler(true, client.scaler).reshape(-1),
                        inverse_with_scaler(ensemble_pred, client.scaler).reshape(-1),
                    ),
                    "Local_MSE": mean_squared_error(true, local_pred),
                    "Local_MAE": mean_absolute_error(true, local_pred),
                    "Local_EPE": safe_epe(
                        inverse_with_scaler(true, client.scaler).reshape(-1),
                        inverse_with_scaler(local_pred, client.scaler).reshape(-1),
                    ),
                    "Global_MSE": mean_squared_error(true, global_pred),
                    "Global_MAE": mean_absolute_error(true, global_pred),
                    "Global_EPE": safe_epe(
                        inverse_with_scaler(true, client.scaler).reshape(-1),
                        inverse_with_scaler(global_pred, client.scaler).reshape(-1),
                    ),
                }
            )
            communication_rows.append(
                {
                    "Client": client.data_name,
                    "Sync_Count": state.sync_count,
                    "Upload_MB": state.upload_bytes / (1024 ** 2),
                    "Download_MB": state.download_bytes / (1024 ** 2),
                    "Downlink_Attempts": state.downlink_attempts,
                    "Successful_Downlinks": state.successful_downlinks,
                    "Failed_Downlinks": state.failed_downlinks,
                    "Attempted_Download_MB": state.attempted_download_bytes / (1024 ** 2),
                    "Mean_Drift": float(np.mean(state.drift_scores)) if state.drift_scores else 0.0,
                    "Mean_Urgency": float(np.mean(state.urgencies)) if state.urgencies else 0.0,
                    "Mean_Cloud_Utility": float(np.mean(state.cloud_utilities))
                    if state.cloud_utilities
                    else 0.0,
                }
            )

        metrics_df = pd.DataFrame(per_client_rows)
        communication_df = pd.DataFrame(communication_rows)
        step_df = pd.DataFrame(self.global_metrics)
        client_step_df = pd.DataFrame(self.client_step_metrics)
        sync_df = pd.DataFrame(self.sync_records)

        metrics_df.to_csv(self.output_dir / "co_adaptation_metrics.csv", index=False)
        communication_df.to_csv(self.output_dir / "communication_summary.csv", index=False)
        step_df.to_csv(self.output_dir / "step_metrics.csv", index=False)
        client_step_df.to_csv(self.output_dir / "client_step_metrics.csv", index=False)
        sync_df.to_csv(self.output_dir / "sync_events.csv", index=False)

        summary = {
            "avg_ensemble_mae": float(metrics_df["Ensemble_MAE"].mean()),
            "avg_ensemble_mse": float(metrics_df["Ensemble_MSE"].mean()),
            "avg_ensemble_epe": float(metrics_df["Ensemble_EPE"].mean()),
            "avg_sync_count": float(communication_df["Sync_Count"].mean()),
            "total_upload_mb": float(communication_df["Upload_MB"].sum()),
            "total_download_mb": float(communication_df["Download_MB"].sum()),
            "total_downlink_attempts": int(communication_df["Downlink_Attempts"].sum()),
            "total_successful_downlinks": int(communication_df["Successful_Downlinks"].sum()),
            "total_failed_downlinks": int(communication_df["Failed_Downlinks"].sum()),
            "observed_downlink_loss_rate": float(
                communication_df["Failed_Downlinks"].sum()
                / max(communication_df["Downlink_Attempts"].sum(), 1)
            ),
        }
        with open(self.output_dir / "run_summary.json", "w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)

        return {
            "metrics": metrics_df,
            "communication": communication_df,
            "step": step_df,
            "client_step": client_step_df,
            "sync": sync_df,
        }
