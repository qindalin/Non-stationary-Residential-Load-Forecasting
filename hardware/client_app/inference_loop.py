from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, List

from client_app.data_source import select_numeric_features


@dataclass
class InferenceOutput:
    timestep: int
    local_loss: float
    global_loss: float
    disagreement: float
    residual: float


def run_mvp_inference_step(timestep: int, row: Dict[str, float | str]) -> InferenceOutput:
    features = select_numeric_features(row, excluded={"date"})
    if not features:
        baseline = 0.0
    else:
        baseline = mean(features)

    variability = pstdev(features) if len(features) > 1 else 0.0
    local_loss = abs(baseline) * 0.01
    global_loss = abs(baseline) * 0.012
    disagreement = abs(local_loss - global_loss)
    residual = variability * 0.01
    return InferenceOutput(
        timestep=timestep,
        local_loss=local_loss,
        global_loss=global_loss,
        disagreement=disagreement,
        residual=residual,
    )