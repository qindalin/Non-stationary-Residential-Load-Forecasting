from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        padded = torch.cat([front, x, end], dim=1)
        pooled = self.avg(padded.permute(0, 2, 1))
        return pooled.permute(0, 2, 1)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        return x - moving_mean, moving_mean


@dataclass
class DLinearConfig:
    seq_len: int = 96
    pred_len: int = 24
    enc_in: int = 1
    moving_avg: int = 25
    individual: bool = False


class DLinear(nn.Module):
    def __init__(self, config: DLinearConfig) -> None:
        super().__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.channels = config.enc_in
        self.individual = config.individual
        self.decomposition = SeriesDecomposition(config.moving_avg)

        if self.individual:
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()
            for _ in range(self.channels):
                seasonal = nn.Linear(self.seq_len, self.pred_len)
                trend = nn.Linear(self.seq_len, self.pred_len)
                seasonal.weight = nn.Parameter((1.0 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                trend.weight = nn.Parameter((1.0 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.linear_seasonal.append(seasonal)
                self.linear_trend.append(trend)
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
            self.linear_seasonal.weight = nn.Parameter((1.0 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.linear_trend.weight = nn.Parameter((1.0 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len], dtype=seasonal_init.dtype, device=seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len], dtype=trend_init.dtype, device=trend_init.device)
            for idx in range(self.channels):
                seasonal_output[:, idx, :] = self.linear_seasonal[idx](seasonal_init[:, idx, :])
                trend_output[:, idx, :] = self.linear_trend[idx](trend_init[:, idx, :])
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)

        combined = seasonal_output + trend_output
        return combined.permute(0, 2, 1)