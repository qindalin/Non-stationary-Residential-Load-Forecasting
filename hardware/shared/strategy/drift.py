from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass
class DriftWeights:
    loss: float = 0.40
    disagreement: float = 0.30
    volatility: float = 0.20
    age: float = 0.10


@dataclass
class LossMoments:
    mean: float = 0.0
    variance: float = 1e-6
    rho_mean: float = 0.9
    rho_variance: float = 0.9

    def update(self, value: float) -> None:
        previous_mean = self.mean
        self.mean = self.rho_mean * self.mean + (1.0 - self.rho_mean) * value
        delta = value - previous_mean
        self.variance = self.rho_variance * self.variance + (1.0 - self.rho_variance) * (delta * delta)


def standardized_loss_shock(loss_value: float, moments: LossMoments, epsilon: float = 1e-6) -> float:
    denom = sqrt(max(moments.variance, epsilon)) + epsilon
    return max((loss_value - moments.mean) / denom, 0.0)


def compute_drift_score(
    loss_shock: float,
    disagreement: float,
    volatility: float,
    sync_age: float,
    weights: DriftWeights,
) -> float:
    return (
        weights.loss * loss_shock
        + weights.disagreement * disagreement
        + weights.volatility * volatility
        + weights.age * sync_age
    )


def compute_resource_pressure(sync_count: int, window_size: int) -> float:
    if window_size <= 0:
        return 0.0
    return max(min(sync_count / window_size, 1.0), 0.0)


def compute_urgency(drift_score: float, resource_pressure: float, beta_resource: float = 1.0) -> float:
    return drift_score / (1.0 + beta_resource * max(resource_pressure, 0.0))