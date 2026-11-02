from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from shared.models.dlinear import DLinear, DLinearConfig
from shared.models.lstm import LSTM, LSTMConfig


@dataclass
class ModelConfig:
    name: str = "DLinear"
    seq_len: int = 96
    pred_len: int = 24
    enc_in: int = 1
    moving_avg: int = 25
    individual: bool = False
    hidden_dims: int = 64


def build_model(config: ModelConfig) -> nn.Module:
    if config.name == "DLinear":
        return DLinear(
            DLinearConfig(
                seq_len=config.seq_len,
                pred_len=config.pred_len,
                enc_in=config.enc_in,
                moving_avg=config.moving_avg,
                individual=config.individual,
            )
        )
    if config.name == "LSTM":
        return LSTM(
            LSTMConfig(
                pred_len=config.pred_len,
                hidden_dims=config.hidden_dims,
            )
        )
    raise ValueError(f"Unsupported model name: {config.name}")