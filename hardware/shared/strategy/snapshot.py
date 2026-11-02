from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional


@dataclass
class SnapshotRecord:
    snapshot_id: str
    timestep: int
    metadata: Dict[str, float]


def select_snapshot(
    snapshots: Iterable[SnapshotRecord],
    min_utility: float,
    preferred_snapshot_id: Optional[str] = None,
) -> Optional[SnapshotRecord]:
    ordered = list(snapshots)
    if not ordered:
        return None
    if preferred_snapshot_id is not None:
        for snapshot in ordered:
            if snapshot.snapshot_id == preferred_snapshot_id:
                return snapshot
    eligible = [item for item in ordered if item.metadata.get("utility", 0.0) >= min_utility]
    if eligible:
        return sorted(eligible, key=lambda item: item.timestep, reverse=True)[0]
    return sorted(ordered, key=lambda item: item.timestep, reverse=True)[0]