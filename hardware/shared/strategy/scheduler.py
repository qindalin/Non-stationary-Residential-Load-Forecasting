from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class ClientPriority:
    client_id: str
    urgency: float
    drift_score: float
    residual_priority: float = 0.0

    @property
    def score(self) -> float:
        return self.urgency + self.residual_priority


def select_clients(candidates: Iterable[ClientPriority], budget: int, threshold: float) -> List[str]:
    eligible = [item for item in candidates if item.urgency >= threshold]
    ordered = sorted(eligible, key=lambda item: (item.score, item.drift_score), reverse=True)
    return [item.client_id for item in ordered[: max(budget, 0)]]