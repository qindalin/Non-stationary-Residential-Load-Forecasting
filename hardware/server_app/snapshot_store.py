from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from shared.common.config import ensure_dir
from shared.protocol.serialization import decode_state_dict, encode_state_dict
from shared.strategy.snapshot import SnapshotRecord


@dataclass
class SnapshotStore:
    root_dir: Path
    metadata: List[SnapshotRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root_dir = ensure_dir(self.root_dir)

    def save(self, snapshot_id: str, timestep: int, state_dict_b64: str, metadata: Dict[str, float]) -> None:
        state_dict = decode_state_dict(state_dict_b64)
        snapshot_path = self.root_dir / f"{snapshot_id}.pt"
        import torch

        torch.save(state_dict, snapshot_path)
        self.metadata.append(SnapshotRecord(snapshot_id=snapshot_id, timestep=timestep, metadata=metadata))

    def load_encoded(self, snapshot_id: str) -> str:
        import torch

        snapshot_path = self.root_dir / f"{snapshot_id}.pt"
        state_dict = torch.load(snapshot_path, map_location="cpu")
        return encode_state_dict(state_dict)

    def load_state_dict(self, snapshot_id: str) -> Dict[str, object]:
        import torch

        snapshot_path = self.root_dir / f"{snapshot_id}.pt"
        return torch.load(snapshot_path, map_location="cpu")

    def metadata_for(self, snapshot_id: str) -> Optional[SnapshotRecord]:
        for item in self.metadata:
            if item.snapshot_id == snapshot_id:
                return item
        return None

    def latest(self) -> Optional[SnapshotRecord]:
        if not self.metadata:
            return None
        return sorted(self.metadata, key=lambda item: item.timestep, reverse=True)[0]

    def list(self) -> List[SnapshotRecord]:
        return list(self.metadata)