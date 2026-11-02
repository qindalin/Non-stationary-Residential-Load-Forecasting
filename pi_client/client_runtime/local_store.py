from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

import torch

from client_runtime.state import RuntimeState


class LocalStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root_dir / "runtime_state.json"
        self.local_model_path = self.root_dir / "local_model.pt"
        self.global_model_path = self.root_dir / "global_model.pt"
        self.anchor_model_path = self.root_dir / "anchor_model.pt"
        self.buffers_path = self.root_dir / "buffers.pt"
        self.ensemble_state_path = self.root_dir / "ensemble_state.pt"

    def load_state(self, client_id: str) -> RuntimeState:
        if not self.state_path.exists():
            return RuntimeState(client_id=client_id)
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        return RuntimeState.from_dict(payload)

    def save_state(self, state: RuntimeState) -> None:
        self.state_path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")

    def save_models(
        self,
        local_state: Dict[str, torch.Tensor],
        global_state: Dict[str, torch.Tensor],
        anchor_state: Dict[str, torch.Tensor],
        local_buffer: Deque[Dict[str, torch.Tensor]],
        global_buffer: Deque[Dict[str, torch.Tensor]],
    ) -> None:
        torch.save(local_state, self.local_model_path)
        torch.save(global_state, self.global_model_path)
        torch.save(anchor_state, self.anchor_model_path)
        torch.save(
            {
                "local_buffer": list(local_buffer),
                "global_buffer": list(global_buffer),
            },
            self.buffers_path,
        )

    def load_model_state(self, model_name: str) -> Optional[Dict[str, torch.Tensor]]:
        path_map = {
            "local": self.local_model_path,
            "global": self.global_model_path,
            "anchor": self.anchor_model_path,
        }
        path = path_map[model_name]
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu")

    def load_buffers(self, maxlen: int) -> tuple[Deque[Dict[str, torch.Tensor]], Deque[Dict[str, torch.Tensor]]]:
        if not self.buffers_path.exists():
            return deque(maxlen=maxlen), deque(maxlen=maxlen)
        payload = torch.load(self.buffers_path, map_location="cpu")
        return (
            deque(payload.get("local_buffer", []), maxlen=maxlen),
            deque(payload.get("global_buffer", []), maxlen=maxlen),
        )

    def save_ensemble_state(self, payload: Dict[str, object]) -> None:
        torch.save(payload, self.ensemble_state_path)

    def load_ensemble_state(self) -> Optional[Dict[str, object]]:
        if not self.ensemble_state_path.exists():
            return None
        return torch.load(self.ensemble_state_path, map_location="cpu")