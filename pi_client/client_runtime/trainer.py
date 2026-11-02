from __future__ import annotations

import copy
from collections import deque
from typing import Deque, Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch import nn, optim

from client_runtime.model_factory import ModelConfig, build_model


def clone_state_dict_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def state_dict_difference(current_state: Dict[str, torch.Tensor], anchor_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: current_state[name].detach().cpu() - anchor_state[name].detach().cpu() for name in current_state.keys()}


def blend_state_dicts(base_state: Dict[str, torch.Tensor], incoming_state: Dict[str, torch.Tensor], blend: float) -> Dict[str, torch.Tensor]:
    if blend >= 1.0:
        return clone_state_dict_cpu(incoming_state)
    if blend <= 0.0:
        return clone_state_dict_cpu(base_state)
    return {
        name: (1.0 - blend) * base_state[name].detach().cpu() + blend * incoming_state[name].detach().cpu()
        for name in base_state.keys()
    }


class OnlineForecastTrainer:
    def __init__(self, model, model_config: ModelConfig, target_idx: int, aux_idx: int | None, device: torch.device) -> None:
        self.model = model.to(device)
        self.model_config = model_config
        self.target_idx = target_idx
        self.aux_idx = aux_idx
        self.device = device
        self.criterion = nn.MSELoss()

    def _forward(self, model, x_np: np.ndarray, y_np: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(x_np).unsqueeze(0).float().to(self.device)
        y = torch.from_numpy(y_np).unsqueeze(0).float().to(self.device)
        if self.model_config.name == "LSTM":
            if self.aux_idx is None:
                raise ValueError("aux_idx is required for LSTM")
            load_hist = x[:, :, self.target_idx]
            future_aux = y[:, :, self.aux_idx]
            return model(load_hist, future_aux)
        return model(x)[:, :, self.target_idx]

    def predict(self, x_np: np.ndarray, y_np: np.ndarray) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            outputs = self._forward(self.model, x_np, y_np)
        self.model.train()
        return outputs.squeeze(0).detach().cpu()

    def loss_on_sample(self, x_np: np.ndarray, y_np: np.ndarray, state_dict: Dict[str, torch.Tensor] | None = None) -> float:
        model = self.model
        if state_dict is not None:
            model = copy.deepcopy(self.model).to(self.device)
            model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            y = torch.from_numpy(y_np).unsqueeze(0).float().to(self.device)
            pred = self._forward(model, x_np, y_np)
            target = y[:, :, self.target_idx]
            loss = self.criterion(pred, target)
        model.train()
        return float(loss.item())

    def train_initial(self, samples: Sequence, lr: float, epochs: int, batch_size: int) -> None:
        if not samples:
            return
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for _ in range(max(epochs, 0)):
            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                y = torch.from_numpy(np.stack([sample.y for sample in batch], axis=0)).float().to(self.device)
                x_np = np.stack([sample.x for sample in batch], axis=0).astype(np.float32)
                y_np = np.stack([sample.y for sample in batch], axis=0).astype(np.float32)
                if self.model_config.name == "LSTM":
                    if self.aux_idx is None:
                        raise ValueError("aux_idx is required for LSTM")
                    x_tensor = torch.from_numpy(x_np).float().to(self.device)
                    load_hist = x_tensor[:, :, self.target_idx]
                    future_aux = y[:, :, self.aux_idx]
                    pred = self.model(load_hist, future_aux)
                else:
                    x_tensor = torch.from_numpy(x_np).float().to(self.device)
                    pred = self.model(x_tensor)[:, :, self.target_idx]
                target = y[:, :, self.target_idx]
                loss = self.criterion(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def online_update(
        self,
        x_np: np.ndarray,
        y_np: np.ndarray,
        lr: float,
        proximal_state: Dict[str, torch.Tensor] | None = None,
        mu: float = 0.0,
    ) -> float:
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        y = torch.from_numpy(y_np).unsqueeze(0).float().to(self.device)
        pred = self._forward(self.model, x_np, y_np)
        target = y[:, :, self.target_idx]
        loss = self.criterion(pred, target)
        if proximal_state is not None and mu > 0.0:
            prox_term = torch.zeros(1, device=self.device)
            for name, param in self.model.named_parameters():
                prox_term = prox_term + torch.sum((param - proximal_state[name].to(self.device)) ** 2)
            loss = loss + 0.5 * mu * prox_term
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.item())


class EnsembleMLP(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, mlp_width: int, mlp_depth: int, mlp_dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(n_inputs, mlp_width)
        self.dropout = nn.Dropout(mlp_dropout)
        self.hiddens = nn.ModuleList([nn.Linear(mlp_width, mlp_width) for _ in range(max(mlp_depth - 2, 0))])
        self.output = nn.Linear(mlp_width, n_outputs)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        x = self.input(x)
        if train:
            x = self.dropout(x)
        x = self.act(x)
        for hidden in self.hiddens:
            x = hidden(x)
            if train:
                x = self.dropout(x)
            x = self.act(x)
        return self.output(x)


class OnlineEnsembleTrainer:
    def __init__(
        self,
        pred_len: int,
        short_w_lr: float,
        long_w_lr: float,
        device: torch.device,
        adaptive_fusion: bool = True,
        consider_error: bool = True,
        short_only: bool = False,
        long_only: bool = False,
        error_only: bool = False,
        min_mix: float = 0.15,
    ) -> None:
        self.pred_len = pred_len
        self.short_w_lr = short_w_lr
        self.long_w_lr = long_w_lr
        self.device = device
        self.adaptive_fusion = adaptive_fusion
        self.consider_error = consider_error
        self.short_only = short_only
        self.long_only = long_only
        self.error_only = error_only
        self.criterion = nn.MSELoss()
        self.ensembler = EnsembleMLP(
            n_inputs=pred_len * 3,
            n_outputs=1,
            mlp_width=32,
            mlp_depth=3,
            mlp_dropout=0.1,
        ).to(device)
        self.long_weight = torch.zeros(1, device=device, requires_grad=True)
        self.short_weight = torch.zeros(1, device=device)
        self.min_mix = min_mix

    def _smooth_gate(self, raw_weight: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(raw_weight)
        return self.min_mix + (1.0 - 2.0 * self.min_mix) * gate

    def state_dict(self) -> Dict[str, object]:
        return {
            "ensembler": self.ensembler.state_dict(),
            "long_weight": self.long_weight.detach().cpu(),
            "short_weight": self.short_weight.detach().cpu(),
        }

    def load_state_dict(self, payload: Dict[str, object]) -> None:
        self.ensembler.load_state_dict(payload["ensembler"])
        long_weight = torch.as_tensor(payload.get("long_weight", torch.zeros(1)), device=self.device, dtype=torch.float32)
        short_weight = torch.as_tensor(payload.get("short_weight", torch.zeros(1)), device=self.device, dtype=torch.float32)
        self.long_weight = long_weight.reshape(1).detach().clone().requires_grad_(True)
        self.short_weight = short_weight.reshape(1).detach().clone()

    def warmup(self, samples: Sequence, local_trainer: OnlineForecastTrainer, global_trainer: OnlineForecastTrainer, epochs: int) -> None:
        if epochs <= 0 or not samples:
            return
        self.ensembler.train()
        for _ in range(epochs):
            for sample in samples:
                target = torch.from_numpy(sample.y[:, local_trainer.target_idx]).float().to(self.device)
                local_pred = local_trainer.predict(sample.x, sample.y).to(self.device)
                global_pred = global_trainer.predict(sample.x, sample.y).to(self.device)
                self._train_step(global_pred, local_pred, target, error_ratio=None)

    def online_step(
        self,
        global_pred: torch.Tensor,
        local_pred: torch.Tensor,
        target: torch.Tensor,
        error_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs, combine_w = self.predict(global_pred, local_pred, error_ratio)
        if self.adaptive_fusion:
            self.update(global_pred, local_pred, target, error_ratio if self.consider_error else None)
        return outputs.detach().cpu(), self.long_weight.detach().cpu(), self.short_weight.detach().cpu(), combine_w.detach().cpu()

    def predict(
        self,
        global_pred: torch.Tensor,
        local_pred: torch.Tensor,
        error_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y1 = global_pred.detach().to(self.device)
        y2 = local_pred.detach().to(self.device)
        if not self.adaptive_fusion:
            combine_w = torch.full((1,), 0.5, device=self.device, dtype=y1.dtype)
            outputs = 0.5 * y1 + 0.5 * y2
            return outputs, combine_w

        error_bias = None
        if self.consider_error and error_ratio is not None:
            error_bias = torch.as_tensor(error_ratio, device=self.device, dtype=y1.dtype)

        if self.long_only:
            combine_w = self._smooth_gate(self.long_weight)
        elif self.short_only:
            combine_w = self._smooth_gate(self.short_weight)
        elif self.error_only:
            combine_w = torch.as_tensor(error_ratio if error_ratio is not None else 0.5, device=self.device, dtype=y1.dtype).reshape(1)
            combine_w = self.min_mix + (1.0 - 2.0 * self.min_mix) * combine_w
        else:
            combine_w = self._smooth_gate(self.long_weight + self.short_weight + (error_bias if error_bias is not None else 0.0))
        return combine_w * y1 + (1.0 - combine_w) * y2, combine_w

    def update(
        self,
        global_pred: torch.Tensor,
        local_pred: torch.Tensor,
        target: torch.Tensor,
        error_ratio: float | None = None,
    ) -> None:
        y1 = global_pred.detach().to(self.device)
        y2 = local_pred.detach().to(self.device)
        true = target.detach().to(self.device)
        self._train_step(y1, y2, true, error_ratio)

    def _train_step(self, y1: torch.Tensor, y2: torch.Tensor, true: torch.Tensor, error_ratio: float | None) -> tuple[torch.Tensor, torch.Tensor]:
        opt_w = optim.Adam([self.long_weight], lr=self.long_w_lr)
        opt_en = optim.Adam(self.ensembler.parameters(), lr=self.short_w_lr)
        error_bias = None if error_ratio is None else torch.as_tensor(error_ratio, device=self.device, dtype=y1.dtype)

        if self.long_only:
            combine_w = self._smooth_gate(self.long_weight)
            outputs = combine_w * y1 + (1.0 - combine_w) * y2
            loss_long_w = self.criterion(outputs, true)
            loss_long_w.backward()
            opt_w.step()
            opt_w.zero_grad()
            combine_w = self._smooth_gate(self.long_weight)
            return combine_w * y1 + (1.0 - combine_w) * y2, combine_w

        if self.short_only:
            inputs_ensemble = torch.cat([y1, y2, true], dim=0).unsqueeze(0)
            self.short_weight = self.ensembler(inputs_ensemble).reshape(1)
            combine_w = self._smooth_gate(self.short_weight)
            loss_short_w = self.criterion(combine_w * y1 + (1.0 - combine_w) * y2, true)
            loss_short_w.backward(retain_graph=True)
            opt_en.step()
            opt_en.zero_grad()
            self.short_weight = self.ensembler(inputs_ensemble, train=False).reshape(1)
            combine_w = self._smooth_gate(self.short_weight)
            return combine_w * y1 + (1.0 - combine_w) * y2, combine_w

        if self.error_only:
            combine_w = torch.as_tensor(error_ratio if error_ratio is not None else 0.5, device=self.device, dtype=y1.dtype).reshape(1)
            combine_w = self.min_mix + (1.0 - 2.0 * self.min_mix) * combine_w
            return combine_w * y1 + (1.0 - combine_w) * y2, combine_w

        long_w = self._smooth_gate(self.long_weight)
        inputs_ensemble = torch.cat([long_w * y1, (1.0 - long_w) * y2, true], dim=0).unsqueeze(0)
        self.short_weight = self.ensembler(inputs_ensemble).reshape(1)
        combine_w = self._smooth_gate(self.long_weight + self.short_weight + (error_bias if error_bias is not None else 0.0))
        loss_short_w = self.criterion(combine_w * y1 + (1.0 - combine_w) * y2, true)
        loss_short_w.backward(retain_graph=True)
        opt_en.step()
        opt_en.zero_grad()

        long_w = self._smooth_gate(self.long_weight)
        loss_long_w = self.criterion(long_w * y1 + (1.0 - long_w) * y2, true)
        loss_long_w.backward()
        opt_w.step()
        opt_w.zero_grad()

        long_w = self._smooth_gate(self.long_weight)
        inputs_ensemble = torch.cat([long_w * y1, (1.0 - long_w) * y2, true], dim=0).unsqueeze(0)
        self.short_weight = self.ensembler(inputs_ensemble, train=False).reshape(1)
        combine_w = self._smooth_gate(self.long_weight + self.short_weight + (error_bias if error_bias is not None else 0.0))
        return combine_w * y1 + (1.0 - combine_w) * y2, combine_w


def buffered_prediction(
    model_config: ModelConfig,
    state_buffer: Deque[Dict[str, torch.Tensor]],
    fallback_state: Dict[str, torch.Tensor],
    x_np: np.ndarray,
    y_np: np.ndarray,
    target_idx: int,
    aux_idx: int | None,
    device: torch.device,
    decay_rate: float,
) -> torch.Tensor:
    states = list(state_buffer) if state_buffer else [fallback_state]
    weights = np.array([np.exp(-decay_rate * (len(states) - idx - 1)) for idx in range(len(states))], dtype=np.float64)
    weights = weights / max(weights.sum(), 1.0e-12)
    prediction = torch.zeros(model_config.pred_len, dtype=torch.float32)
    x_tensor = torch.from_numpy(x_np).unsqueeze(0).float().to(device)
    y_tensor = torch.from_numpy(y_np).unsqueeze(0).float().to(device)
    for weight, state_dict in zip(weights, states):
        model = build_model(model_config).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            if model_config.name == "LSTM":
                if aux_idx is None:
                    raise ValueError("aux_idx is required for LSTM")
                branch_pred = model(x_tensor[:, :, target_idx], y_tensor[:, :, aux_idx]).squeeze(0).detach().cpu()
            else:
                branch_pred = model(x_tensor)[:, :, target_idx].squeeze(0).detach().cpu()
        prediction = prediction + float(weight) * branch_pred
    return prediction