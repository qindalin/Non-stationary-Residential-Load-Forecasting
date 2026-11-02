from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def compute_aggregation_weights(
    updates: Iterable[Dict[str, float]],
    beta_drift: float = 1.0,
    beta_utility: float = 0.75,
    beta_volatility: float = 0.50,
    beta_staleness: float = 0.25,
) -> List[float]:
    payload = list(updates)
    if not payload:
        return []

    logits = []
    for item in payload:
        logits.append(
            beta_drift * float(item.get("drift_score", 0.0))
            + beta_utility * float(item.get("cloud_utility", 0.0))
            - beta_volatility * float(item.get("volatility", 0.0))
            - beta_staleness * float(item.get("staleness", 0.0))
        )

    array = np.asarray(logits, dtype=np.float64)
    array = array - array.max()
    probs = np.exp(array)
    probs /= probs.sum()
    return probs.astype(np.float32).tolist()