from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

from shared.models.factory import ModelConfig, build_model
from shared.runtime.state_ops import apply_weighted_deltas, clone_state_dict_cpu


@dataclass
class GlobalModelManager:
    model_config: ModelConfig
    server_lambda: float = 0.5
    stale_gamma: float = 1.0
    initial_state_path: str | None = None

    def __post_init__(self) -> None:
        self.model = build_model(self.model_config)
        self.current_state = clone_state_dict_cpu(self.model.state_dict())
        if self.initial_state_path:
            state_path = Path(self.initial_state_path)
            if state_path.exists():
                state_dict = torch.load(state_path, map_location="cpu")
                self.set_state(state_dict)

    def get_state(self) -> Dict[str, torch.Tensor]:
        return clone_state_dict_cpu(self.current_state)

    def set_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.current_state = clone_state_dict_cpu(state_dict)
        self.model.load_state_dict(self.current_state)

    def aggregate_updates(self, updates: List[Dict[str, object]], weights: List[float]) -> Dict[str, torch.Tensor]:
        base_state = self.get_state()
        deltas = [update["delta_state"] for update in updates]
        stale_factors = [((1.0 + float(update.get("staleness", 0.0))) ** (-self.stale_gamma)) for update in updates]
        new_state = apply_weighted_deltas(base_state, deltas, weights, stale_factors, self.server_lambda)
        self.set_state(new_state)
        return self.get_state()