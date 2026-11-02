#!/usr/bin/env python3
"""Assemble per-building CSVs for the simulation from the official BDG2 release.

The paper uses hourly electricity consumption of residential buildings from
the Building Data Genome Project 2 (BDG2):

    Miller et al., "The building data genome project 2, energy meter data from
    the ASHRAE great energy predictor III competition", Scientific Data 7, 2020.

Official sources:

    - GitHub repository (files are stored with Git LFS):
      https://github.com/buds-lab/building-data-genome-project-2
    - Archived v1.0 dataset (~600 MB zip, no LFS needed):
      https://doi.org/10.5281/zenodo.3887306

This script is NOT required for the quick demo (five `Lamb_*` clients are
already bundled under `simulation/data/`). It regenerates per-building CSVs in
the same `date,airTemp,value` format that `main_co_adaptation.py` expects.

Data inputs (all inside the BDG2 archive):

    data/metadata/metadata.csv
    data/weather/weather.csv
    data/meters/cleaned/electricity_cleaned.csv

Usage
-----
    # Let the script download the Zenodo archive and cache it:
    python scripts/prepare_bdg2_data.py \
        --buildings Lamb_office_Bertha Lamb_health_Ken --out data

    # Or reuse a local checkout (clone the repo and run `git lfs pull`):
    python scripts/prepare_bdg2_data.py \
        --source-dir /path/to/building-data-genome-project-2 \
        --buildings Lamb_office_Bertha Lamb_health_Ken --out data
"""
from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ZENODO_URL = (
    "https://zenodo.org/api/records/3887306/files/"
    "buds-lab/building-data-genome-project-2-v1.0.zip/content"
)

METADATA_REL = "data/metadata/metadata.csv"
WEATHER_REL = "data/weather/weather.csv"
ELECTRICITY_REL = "data/meters/cleaned/electricity_cleaned.csv"


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}\n  -> {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (fixed public URL)


def extract_from_zip(zip_path: Path, cache_dir: Path) -> None:
    """Extract the three CSVs we need, whatever the top-level folder name is."""
    wanted = {
        METADATA_REL: "metadata.csv",
        WEATHER_REL: "weather.csv",
        ELECTRICITY_REL: "electricity_cleaned.csv",
    }
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for rel, out_name in wanted.items():
            dest = cache_dir / out_name
            if dest.exists():
                continue
            match = next((n for n in names if n.endswith(rel)), None)
            if match is None:
                raise FileNotFoundError(f"{rel} not found inside {zip_path}")
            with zf.open(match) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            print(f"extracted {out_name}")


def pick_temperature_column(weather: pd.DataFrame) -> str:
    for col in weather.columns:
        if col.lower() == "airtemperature":
            return col
    for col in weather.columns:
        low = col.lower()
        if "temp" in low and "dew" not in low:
            return col
    raise KeyError("no air-temperature column found in weather.csv")


def assemble(source: Path, buildings: list[str], out: Path) -> None:
    metadata = pd.read_csv(source / "metadata.csv")
    weather = pd.read_csv(source / "weather.csv")
    elec = pd.read_csv(source / "electricity_cleaned.csv")

    if "building_id" not in metadata.columns or "site_id" not in metadata.columns:
        raise KeyError("metadata.csv must contain building_id and site_id")
    meta_by_building = metadata.set_index("building_id")

    temp_col = pick_temperature_column(weather)
    first_col = elec.columns[0]
    if str(first_col).lower() in {"timestamp", "unnamed: 0", "date", "datetime"}:
        elec = elec.set_index(first_col)
        elec.index = pd.to_datetime(elec.index)
    else:
        elec["timestamp"] = pd.to_datetime(elec["timestamp"])
        elec = elec.set_index("timestamp")

    for building in buildings:
        if building not in meta_by_building.index:
            print(f"[skip] {building}: not in metadata.csv")
            continue
        if building not in elec.columns:
            print(f"[skip] {building}: not in electricity_cleaned.csv")
            continue

        site = meta_by_building.loc[building, "site_id"]
        use = meta_by_building.loc[building].get("primary_use", "unknown")
        if isinstance(use, pd.Series):
            use = use.iloc[0]

        site_weather = weather[weather["site_id"] == site].copy()
        if site_weather.empty:
            raise KeyError(f"no weather rows for site {site}")
        site_weather[temp_col] = pd.to_numeric(site_weather[temp_col], errors="coerce")
        site_weather["timestamp"] = pd.to_datetime(site_weather["timestamp"])
        site_weather = site_weather[["timestamp", temp_col]].drop_duplicates("timestamp")
        site_weather = site_weather.set_index("timestamp").sort_index()

        series = pd.to_numeric(elec[building], errors="coerce").resample("1h").mean()
        frame = pd.DataFrame(
            {
                "date": series.index,
                "airTemp": site_weather[temp_col].reindex(series.index).values.round(1),
                "value": series.values,
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
    parser.add_argument("--cache-dir", default="bdg2_cache", help="download/extract cache")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.source_dir:
        root = Path(args.source_dir)
        src_metadata = root / "data" / "metadata" / "metadata.csv"
        if not src_metadata.exists():
            raise FileNotFoundError(
                f"{src_metadata} not found; run `git lfs pull` inside the BDG2 checkout"
            )
        staged = Path(args.cache_dir)
        staged.mkdir(parents=True, exist_ok=True)
        for rel, name in (
            (root / METADATA_REL, "metadata.csv"),
            (root / WEATHER_REL, "weather.csv"),
            (root / ELECTRICITY_REL, "electricity_cleaned.csv"),
        ):
            dest = staged / name
            if not dest.exists():
                dest.write_bytes(rel.read_bytes())
        source = staged
    else:
        cache = Path(args.cache_dir)
        zip_path = cache / "bdg2-v1.0.zip"
        download(ZENODO_URL, zip_path)
        extract_from_zip(zip_path, cache)
        source = cache

    assemble(source, args.buildings, out)


if __name__ == "__main__":
    main()
