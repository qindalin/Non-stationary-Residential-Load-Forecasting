from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np


@dataclass
class WindowSample:
    start_index: int
    target_start_index: int
    timestamp: str
    x: np.ndarray
    y: np.ndarray


class StandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> None:
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ < 1e-6] = 1.0

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fit before transform")
        return (values - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fit before inverse_transform")
        return values * self.scale_ + self.mean_


class CsvWindowStream:
    def __init__(
        self,
        csv_path: str | Path,
        feature_columns: Sequence[str],
        target_col: str,
        seq_len: int,
        pred_len: int,
        train_ratio: float | None,
        split_mode: str = "original_online_federated",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.feature_columns = list(feature_columns)
        self.target_col = target_col
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.train_ratio = train_ratio
        self.split_mode = split_mode
        self.timestamps: List[str] = []
        self.values = self._load_numeric_values()
        self.target_idx = self.feature_columns.index(self.target_col)
        self.scaler = StandardScaler()
        self.train_split_index, self.online_start_index = self._resolve_split_indices()
        self.scaler.fit(self.values[: self.train_split_index])
        self.scaled_values = self.scaler.transform(self.values)

    def _resolve_split_indices(self) -> tuple[int, int]:
        if self.split_mode == "original_online_federated":
            day_num = len(self.values) / 24
            num_train = int(day_num * 0.2) * 24
            num_test = int(day_num * 0.75) * 24
            train_split_index = max(num_train, self.seq_len + self.pred_len)
            online_start_index = max(len(self.values) - num_test - self.seq_len, 0)
            return train_split_index, online_start_index

        if self.train_ratio is None:
            raise ValueError("train_ratio must be provided when split_mode is not 'original_online_federated'")
        train_split_index = max(self.seq_len + self.pred_len, int(len(self.values) * self.train_ratio))
        return train_split_index, max(train_split_index - self.seq_len, 0)

    def _load_numeric_values(self) -> np.ndarray:
        rows: List[List[float]] = []
        with self.csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = [name for name in self.feature_columns if name not in reader.fieldnames]
            if missing:
                raise ValueError(f"Missing required columns in CSV: {missing}")
            for row in reader:
                self.timestamps.append(row.get("date", ""))
                rows.append([float(row[name]) for name in self.feature_columns])
        if len(rows) < self.seq_len + self.pred_len + 1:
            raise ValueError("CSV does not contain enough rows for the configured seq_len and pred_len")
        return np.asarray(rows, dtype=np.float32)

    def train_samples(self) -> List[WindowSample]:
        samples: List[WindowSample] = []
        last_start = self.train_split_index - self.seq_len - self.pred_len
        for start in range(0, max(last_start + 1, 0)):
            target_start = start + self.seq_len
            samples.append(
                WindowSample(
                    start_index=start,
                    target_start_index=target_start,
                    timestamp=self.timestamps[target_start + self.pred_len - 1],
                    x=self.scaled_values[start : start + self.seq_len],
                    y=self.scaled_values[target_start : target_start + self.pred_len],
                )
            )
        return samples

    def online_samples(self) -> Iterator[WindowSample]:
        start_index = self.online_start_index
        last_start = len(self.scaled_values) - self.seq_len - self.pred_len
        for start in range(start_index, last_start + 1):
            target_start = start + self.seq_len
            yield WindowSample(
                start_index=start,
                target_start_index=target_start,
                timestamp=self.timestamps[target_start + self.pred_len - 1],
                x=self.scaled_values[start : start + self.seq_len],
                y=self.scaled_values[target_start : target_start + self.pred_len],
            )