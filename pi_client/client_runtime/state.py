from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RuntimeState:
    client_id: str
    timestep: int = 0
    last_sync_step: int = 0
    anchor_step: int = 0
    latest_drift_score: float = 0.0
    latest_urgency: float = 0.0
    latest_cloud_utility: float = 0.0
    local_loss_mean: float = 0.0
    local_loss_variance: float = 1.0e-6
    residuals: List[float] = field(default_factory=list)
    sync_history: List[int] = field(default_factory=list)
    online_error_ratio: float = 0.5
    upload_bytes: int = 0
    download_bytes: int = 0
    latest_snapshot_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RuntimeState":
        return cls(**payload)