from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import optim


Batch = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class Experience:
    batch: Batch
    cached_prediction: torch.Tensor


@dataclass
class AdapterState:
    memory: List[Experience] = field(default_factory=list)
    seen: int = 0


class EdgeAdapter:
    """Client-private, bounded-memory online adaptation for regression."""

    def __init__(
        self,
        name: str,
        memory_size: int = 7,
        replay_size: int = 2,
        replay_weight: float = 0.5,
        distill_weight: float = 0.5,
        seed: int = 0,
    ) -> None:
        if name not in {"online", "agem", "mir", "der"}:
            raise ValueError("edge adapter must be one of: online, agem, mir, der.")
        self.name = name
        self.memory_size = memory_size
        self.replay_size = replay_size
        self.replay_weight = replay_weight
        self.distill_weight = distill_weight
        self.seed = seed
        self.states: Dict[str, AdapterState] = {}

    def update(self, client_name: str, trainer, batch: Sequence[torch.Tensor], lr: float, grad_clip_norm=None) -> None:
        state = self.states.setdefault(client_name, AdapterState())
        device_batch = self._to_device(batch, trainer.device)
        model = trainer.model
        model.train()

        current_loss, current_prediction = self._loss_and_prediction(trainer, device_batch)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        optimizer.zero_grad()

        if self.name == "online" or not state.memory:
            loss = current_loss
            loss.backward()
        elif self.name == "agem":
            self._agem_backward(trainer, current_loss, state.memory)
        elif self.name == "mir":
            replay = self._select_mir(trainer, device_batch, current_loss, state.memory, lr)
            current_loss, _ = self._loss_and_prediction(trainer, device_batch)
            loss = current_loss + self.replay_weight * self._mean_supervised_loss(trainer, replay)
            loss.backward()
        else:
            replay = self._random_replay(state.memory, client_name)
            replay_loss = self._mean_supervised_loss(trainer, replay)
            distill_loss = self._mean_distill_loss(trainer, replay)
            loss = current_loss + self.replay_weight * replay_loss + self.distill_weight * distill_loss
            loss.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        self._add_experience(state, device_batch, current_prediction.detach().cpu(), client_name)

    @staticmethod
    def _to_device(batch: Sequence[torch.Tensor], device) -> Batch:
        return tuple(item.float().to(device) for item in batch)  # type: ignore[return-value]

    @staticmethod
    def _clone_batch_cpu(batch: Batch) -> Batch:
        return tuple(item.detach().cpu().clone() for item in batch)  # type: ignore[return-value]

    @staticmethod
    def _loss_and_prediction(trainer, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        prediction = trainer.backbone_forward(batch_x, batch_y, batch_x_mark, batch_y_mark)
        target = batch_y[:, -trainer.args.pred_len :, -1]
        return trainer.criterion(prediction, target) + trainer._energy_loss(prediction, target), prediction

    def _agem_backward(self, trainer, current_loss: torch.Tensor, memory: List[Experience]) -> None:
        parameters = [parameter for parameter in trainer.model.parameters() if parameter.requires_grad]
        current_grads = torch.autograd.grad(current_loss, parameters)
        memory_loss = self._mean_supervised_loss(trainer, memory)
        memory_grads = torch.autograd.grad(memory_loss, parameters)
        dot_product = sum(torch.sum(current * reference) for current, reference in zip(current_grads, memory_grads))
        reference_norm = sum(torch.sum(reference * reference) for reference in memory_grads)
        if dot_product < 0:
            scale = dot_product / (reference_norm + 1e-12)
            current_grads = [current - scale * reference for current, reference in zip(current_grads, memory_grads)]
        for parameter, gradient in zip(parameters, current_grads):
            parameter.grad = gradient.detach().clone()

    def _select_mir(self, trainer, current_batch: Batch, current_loss: torch.Tensor, memory: List[Experience], lr: float) -> List[Experience]:
        parameters = [parameter for parameter in trainer.model.parameters() if parameter.requires_grad]
        gradients = torch.autograd.grad(current_loss, parameters)
        original = [parameter.detach().clone() for parameter in parameters]
        with torch.no_grad():
            for parameter, gradient in zip(parameters, gradients):
                parameter.add_(gradient, alpha=-lr)
        after_losses = [float(self._supervised_loss(trainer, item).detach().item()) for item in memory]
        with torch.no_grad():
            for parameter, value in zip(parameters, original):
                parameter.copy_(value)
        before_losses = [float(self._supervised_loss(trainer, item).detach().item()) for item in memory]
        ranked = np.argsort(np.asarray(after_losses) - np.asarray(before_losses))[::-1]
        return [memory[index] for index in ranked[: min(self.replay_size, len(memory))]]

    def _random_replay(self, memory: List[Experience], client_name: str) -> List[Experience]:
        count = min(self.replay_size, len(memory))
        rng = np.random.default_rng(self.seed + len(memory) + sum(map(ord, client_name)))
        indices = rng.choice(len(memory), size=count, replace=False)
        return [memory[int(index)] for index in indices]

    def _supervised_loss(self, trainer, experience: Experience) -> torch.Tensor:
        batch = self._to_device(experience.batch, trainer.device)
        loss, _ = self._loss_and_prediction(trainer, batch)
        return loss

    def _mean_supervised_loss(self, trainer, experiences: List[Experience]) -> torch.Tensor:
        return torch.stack([self._supervised_loss(trainer, item) for item in experiences]).mean()

    def _mean_distill_loss(self, trainer, experiences: List[Experience]) -> torch.Tensor:
        losses = []
        for experience in experiences:
            batch = self._to_device(experience.batch, trainer.device)
            _, prediction = self._loss_and_prediction(trainer, batch)
            losses.append(torch.mean((prediction - experience.cached_prediction.to(trainer.device)) ** 2))
        return torch.stack(losses).mean()

    def _add_experience(self, state: AdapterState, batch: Batch, cached_prediction: torch.Tensor, client_name: str) -> None:
        experience = Experience(batch=self._clone_batch_cpu(batch), cached_prediction=cached_prediction.clone())
        state.seen += 1
        if len(state.memory) < self.memory_size:
            state.memory.append(experience)
            return
        rng = np.random.default_rng(self.seed + state.seen + sum(map(ord, client_name)))
        index = int(rng.integers(0, state.seen))
        if index < self.memory_size:
            state.memory[index] = experience