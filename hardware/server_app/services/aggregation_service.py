from __future__ import annotations

from typing import Dict, List

from shared.strategy.aggregator import compute_aggregation_weights


class AggregationService:
    def compute_weights(self, updates: List[Dict[str, float]]) -> List[float]:
        return compute_aggregation_weights(updates)