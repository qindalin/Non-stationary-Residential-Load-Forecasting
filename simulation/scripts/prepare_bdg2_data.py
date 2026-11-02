#!/usr/bin/env python3
"""Assemble per-building CSVs for the simulation from the official BDG2 release.

The paper uses hourly electricity consumption of residential buildings from
the Building Data Genome Project 2 (BDG2):

    Miller et al., "The building data genome project 2, energy meter data from
    the ASHRAE great energy predictor III competition", Scientific Data 7, 2020.

The dataset is publicly available (CC-BY 4.0) at:
    https://github.com/buds-lab/the-building-data-genome-project-2

This script is NOT required for the quick demo (five `Lamb_*` clients are
already bundled under `simulation/data/`). It regenerates the full per-client
CSVs that `main_co_adaptation.py` expects.

Steps performed
---------------
1. Downloads `metadata.csv` and `weather.csv` from the official repository
   (raw GitHub URLs) unless they already exist locally.
2. Downloads the meter readings of the requested buildings
   (`<building>.csv` under `data/meters/cleaned/`).
3. Aligns hourly weather temperature and resamples readings to hourly
   frequency, producing `<site>_<primary_use>_<building>.csv` with columns
   `date,airTemp,value` (same format as the bundled files).

Usage
-----
    python scripts/prepare_bdg2_data.py \
        --out data \
        --buildings Lamb_office_Bertha Lamb_health_Ken \
        [--source-dir /path/to/existing/bdg2/checkout]

If you already have a local checkout of the BDG2 repository, pass
`--source-dir` to use it instead of downloading.
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

RAW_BASE = "https://raw.githubusercontent.com/buds-lab/the-building-data-genome-project-2/main"
METER_BASE = f"{RAW_BASE}/data/meters/cleaned"


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (fixed public URLs)


def assemble(source_dir: Path, buildings: list[str], out: Path) -> None:
    meta = pd.read_csv(source_dir / "metadata.csv")
    weather = pd.read_csv(source_dir / "weather.csv")

    for building in buildings:
        rows = meta[meta["building_id"] == building]
        if rows.empty:
            print(f"[skip] unknown building {building}")
            continue
        site = rows.iloc[0]["site_id"]
        use = rows.iloc[0]["primary_use"]

        meter = pd.read_csv(source_dir / building, parse_dates=["timestamp"])
        meter = meter.set_index("timestamp").resample("1h").mean().reset_index()

        site_weather = weather[weather["site_id"] == site]
        if site_weather.empty:
            print(f"[warn] no weather for site {site}; airTemp filled with NaN")
            temp = float("nan")
        else:
            temp = site_weather["air_temperature"].mean()
            temp = float(round(temp, 1))

        frame = pd.DataFrame(
            {
                "date": meter["timestamp"],
                "airTemp": temp,
                "value": meter["meter_reading"],
            }
        )
        frame = frame.dropna(subset=["value"])
        name = f"{site}_{use}_{building}.csv"
        dest = out / name
        frame.to_csv(dest, index=False)
        print(f"wrote {dest} ({len(frame)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buildings", nargs="+", required=True, help="BDG2 building ids")
    parser.add_argument("--out", default="data", help="output directory")
    parser.add_argument("--source-dir", default=None, help="local BDG2 checkout (optional)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.source_dir:
        source_dir = Path(args.source_dir)
    else:
        source_dir = Path("bdg2_cache")
        download(f"{RAW_BASE}/metadata.csv", source_dir / "metadata.csv")
        download(f"{RAW_BASE}/weather.csv", source_dir / "weather.csv")
        for building in args.buildings:
            download(f"{METER_BASE}/{building}.csv", source_dir / f"{building}.csv")

    assemble(source_dir, args.buildings, out)


if __name__ == "__main__":
    main()
