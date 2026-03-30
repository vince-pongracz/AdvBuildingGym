"""Extract household energy consumption from WPuQ SFH CSV files.

Reads the 1-min resolution SFH CSVs (already extracted from HDF5 by
extract_sfh_csv.py), extracts the ``hh_P_TOT`` column (active power in W),
resamples to 5-min resolution, converts to kW, and writes:

NOTE: This script relies on the upstream ``extract_sfh_csv.py`` step to
select the correct HDF5 group. By default that step uses the ``NO_PV``
group, so the resulting SFH CSVs contain only household consumption
without photovoltaic generation. If the upstream extraction is re-run
with a different group (e.g. ``--sfh-group WITH_PV``), the data
processed here will reflect that choice.

 - One CSV per building  (e.g. ``2018_SFH10.csv``)
 - One aggregated CSV    (e.g. ``2018_aggregated_consumption.csv``)

Output column: ``hh_consumption_kW`` (raw kW, not normalised).

Usage:
    python -m preproc.hh_consumption.extract_hh_consumption
    python -m preproc.hh_consumption.extract_hh_consumption --input-dir data/weather/zenodo/csvs_2018_data_1min
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

try:
    from ..utils import resolve_path, parse_year_from_filename
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils import resolve_path, parse_year_from_filename

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Source column in SFH CSVs (active power in Watts)
SOURCE_COLUMN: str = "hh_P_TOT"

# Output column name (kW)
OUTPUT_COLUMN: str = "hh_consumption_kW"

# Resample from 1-min to 5-min using mean aggregation
RESAMPLE_RULE: str = "5min"

# W -> kW conversion factor
W_TO_KW: float = 1e-3


def resample_to_5min(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Resample a 1-min Series to 5-min resolution using the mean.

    Args:
        df: DataFrame with DatetimeIndex and the specified column.
        column: Column name to resample.

    Returns:
        DataFrame with 5-min resolution and a single output column.
    """
    resampled = df[[column]].resample(RESAMPLE_RULE).mean()
    return resampled


def process_single_building(
    csv_path: Path,
) -> pd.DataFrame | None:
    """Read one SFH CSV, extract hh_P_TOT, resample to 5-min, convert to kW.

    Args:
        csv_path: Path to the SFH CSV file (1-min resolution).

    Returns:
        DataFrame with DatetimeIndex and ``hh_consumption_kW`` column,
        or None if the source column is missing.
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")

    if SOURCE_COLUMN not in df.columns:
        logger.warning("Column '%s' not found in %s — skipping", SOURCE_COLUMN, csv_path.name)
        return None

    resampled = resample_to_5min(df, SOURCE_COLUMN)

    # Convert W -> kW
    resampled[OUTPUT_COLUMN] = resampled[SOURCE_COLUMN] * W_TO_KW
    resampled = resampled.drop(columns=[SOURCE_COLUMN])

    # Drop rows with NaN (edges of resampling window)
    resampled = resampled.dropna()

    return resampled


def extract_hh_consumption(
    input_dir: str | Path,
    output_dir: str | Path,
    year: int | None = None,
) -> dict[str, int | list[str]]:
    """Extract household consumption from all SFH CSVs in a directory.

    Produces one CSV per building and one aggregated CSV (mean across all
    buildings).

    Args:
        input_dir: Directory containing SFH CSV files (e.g. SFH10.csv).
        output_dir: Directory to write output CSVs.
        year: Year label for output filenames. If None, inferred from
            the input directory name.

    Returns:
        Dict with extraction statistics.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if year is None:
        # Infer year from directory name like "csvs_2018_data_1min"
        try:
            year = parse_year_from_filename(input_dir)
        except ValueError:
            logger.warning(
                "Could not infer year from '%s', using 0000 as placeholder",
                input_dir.name,
            )
            year = 0

    stats: dict[str, int | list[str]] = {
        "buildings_found": 0,
        "buildings_processed": 0,
        "files_created": [],
    }

    sfh_files = sorted(input_dir.glob("SFH*.csv"))
    stats["buildings_found"] = len(sfh_files)

    if not sfh_files:
        logger.warning("No SFH*.csv files found in %s", input_dir)
        return stats

    logger.info("Found %d SFH files in %s", len(sfh_files), input_dir)

    building_frames: list[pd.DataFrame] = []

    for csv_path in sfh_files:
        building_name = csv_path.stem  # e.g. "SFH10"
        logger.info("Processing %s ...", building_name)

        result = process_single_building(csv_path)
        if result is None:
            continue

        # Save individual building CSV
        out_path = output_dir / f"{year}_{building_name}.csv"
        result.to_csv(out_path, index=True)
        stats["buildings_processed"] += 1
        stats["files_created"].append(str(out_path))

        logger.info(
            "  %s: %d rows, max %.2f kW, mean %.2f kW",
            building_name,
            len(result),
            result[OUTPUT_COLUMN].max(),
            result[OUTPUT_COLUMN].mean(),
        )

        building_frames.append(result)

    # Aggregate: mean consumption across all buildings
    if building_frames:
        # Rename columns to unique names before concat, then compute mean
        renamed = [
            df.rename(columns={OUTPUT_COLUMN: f"{OUTPUT_COLUMN}_{i}"})
            for i, df in enumerate(building_frames)
        ]
        aggregated = pd.concat(renamed, axis=1)
        value_cols = [c for c in aggregated.columns if c.startswith(OUTPUT_COLUMN)]
        aggregated[OUTPUT_COLUMN] = aggregated[value_cols].mean(axis=1)
        aggregated = aggregated[[OUTPUT_COLUMN]]
        aggregated = aggregated.dropna()

        agg_path = output_dir / f"{year}_aggregated_consumption.csv"
        aggregated.to_csv(agg_path, index=True)
        stats["files_created"].append(str(agg_path))

        logger.info(
            "Aggregated: %d rows, max %.2f kW, mean %.2f kW",
            len(aggregated),
            aggregated[OUTPUT_COLUMN].max(),
            aggregated[OUTPUT_COLUMN].mean(),
        )

    return stats


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract household consumption from WPuQ SFH CSVs"
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default="data/weather/zenodo/csvs_2018_data_1min",
        help="Directory containing SFH CSV files (default: data/weather/zenodo/csvs_2018_data_1min)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/hh_consumption/wpuq",
        help="Output directory for consumption CSVs (default: data/hh_consumption/wpuq)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year label for output filenames (default: inferred from input dir name)",
    )
    args = parser.parse_args()

    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)

    stats = extract_hh_consumption(input_dir, output_dir, year=args.year)

    print(f"\n{'=' * 60}")
    print("HOUSEHOLD CONSUMPTION EXTRACTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Buildings found:     {stats['buildings_found']}")
    print(f"Buildings processed: {stats['buildings_processed']}")
    print(f"Files created:       {len(stats['files_created'])}")
    print(f"Output directory:    {output_dir}")


if __name__ == "__main__":
    main()
