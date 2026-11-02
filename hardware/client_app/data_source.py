from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, List


class CsvReplayDataSource:
    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)

    def iter_rows(self) -> Iterator[Dict[str, float | str]]:
        with self.csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield row


def select_numeric_features(row: Dict[str, float | str], excluded: Iterable[str] | None = None) -> List[float]:
    blocked = set(excluded or [])
    numeric_values: List[float] = []
    for key, value in row.items():
        if key in blocked:
            continue
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue
    return numeric_values