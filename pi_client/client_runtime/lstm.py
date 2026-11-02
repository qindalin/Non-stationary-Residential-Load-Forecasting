from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LSTMConfig:
    pred_len: int
    hidden_dims: int = 64


class LSTM(nn.Module):
    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_dims
        self.pred_len = config.pred_len
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size + 1, self.hidden_size),
            nn.ReLU(),
        )
        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, load_hist: torch.Tensor, future_aux: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.lstm(load_hist.unsqueeze(-1))
        future_aux = future_aux.unsqueeze(-1)
        fused = torch.cat([hidden[:, -future_aux.shape[1] :, :], future_aux], dim=-1)
        projected = self.fc(fused)
        output = self.linear(projected)
        return output[:, :, -1]