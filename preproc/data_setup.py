"""Unified data setup script for AdvBuildingGym.

By default, this script runs both pipelines:
1) electricity price fetch + preprocessing (+ optional augmentation)
2) weather/Zenodo fetch + preprocessing

Examples:
    python preproc/data_setup.py
    python preproc/data_setup.py --skip-weather
    python preproc/data_setup.py --skip-prices --steps zenodo-extract weather-csv
    python preproc/data_setup.py --skip-weather --years 2025 --skip-price-fetch --raw-price-files data/e_price/2025_prices.csv
    python preproc/data_setup.py --skip-weather --years 2023 --skip-price-fetch --raw-price-files data/e_price/2023_prices.csv --augment
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is importable when script is run directly
_PROJECT_ROOT_STR = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_STR)

from preproc.pipelines import run_price_pipeline, run_weather_pipeline

logger = logging.getLogger(__name__)

DEFAULT_YEARS: list[int] = list(range(2017, 2027))

ALL_PRICE_STEPS: set[str] = {"price-fetch", "price-preproc"}
ALL_WEATHER_STEPS: set[str] = {"zenodo-download", "zenodo-extract", "weather-csv", "sfh-csv"}
ALL_STEPS: set[str] = ALL_PRICE_STEPS | ALL_WEATHER_STEPS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified data setup for AdvBuildingGym")

    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=DEFAULT_YEARS,
        help=f"Target years for price data fetch/preprocessing (default: {DEFAULT_YEARS})",
    )
    parser.add_argument(
        "--price-source",
        choices=["awattar", "energy-charts"],
        default="energy-charts",
        help="Price data source (default: awattar)",
    )
    parser.add_argument(
        "--energy-charts-bzn",
        default="DE-LU",
        help="Bidding zone for Energy Charts fetch (default: DE-LU)",
    )
    parser.add_argument(
        "--price-output-dir",
        default="data/e_price",
        help="Directory for raw and preprocessed price CSV files",
    )
    parser.add_argument(
        "--skip-prices",
        action="store_true",
        help="Skip the entire price pipeline",
    )
    parser.add_argument(
        "--skip-price-fetch",
        action="store_true",
        help="Skip fetching price data and use existing raw files",
    )
    parser.add_argument(
        "--raw-price-files",
        nargs="+",
        default=None,
        help="Explicit raw price CSV file paths (used when --skip-price-fetch is set)",
    )

    parser.add_argument(
        "--skip-weather",
        action="store_true",
        help="Skip the entire weather/Zenodo pipeline",
    )
    parser.add_argument(
        "--zenodo-links",
        default="data/weather/zenodo/ds_links.txt",
        help="Path to text file containing Zenodo download links",
    )
    parser.add_argument(
        "--zenodo-dir",
        default="data/weather/zenodo",
        help="Directory to store Zenodo files",
    )
    parser.add_argument(
        "--overwrite-downloads",
        action="store_true",
        help="Overwrite already downloaded files and re-extract archives",
    )

    parser.add_argument(
        "--weather-hdf5",
        nargs="*",
        default=None,
        help="HDF5 file(s) for weather CSV extraction (default: auto-discover *_weather.hdf5 in zenodo dir)",
    )
    parser.add_argument(
        "--weather-csv-dir",
        default="data/weather/zenodo/csvs_weather",
        help="Output directory for merged weather CSVs",
    )
    parser.add_argument(
        "--weather-timestamp-unit",
        choices=["s", "ns"],
        default="s",
        help="Timestamp unit used in weather HDF5 index (default: s)",
    )
    parser.add_argument(
        "--sfh-hdf5",
        nargs="*",
        default=None,
        help="HDF5 file(s) for SFH CSV extraction (default: auto-discover *_data_1min.hdf5 in zenodo dir)",
    )
    parser.add_argument(
        "--sfh-csv-dir",
        default=None,
        help="Output directory for SFH CSV extraction (default: data/weather/zenodo/csvs_<hdf5_stem> per file)",
    )
    parser.add_argument(
        "--sfh-group",
        default="NO_PV",
        help="HDF5 group for SFH extraction (default: NO_PV)",
    )

    parser.add_argument(
        "--steps",
        nargs="+",
        default=sorted(ALL_STEPS),
        choices=sorted(ALL_STEPS),
        metavar="STEP",
        help=(
            "Select which steps to run within each pipeline "
            f"(default: all; choices: {', '.join(sorted(ALL_STEPS))})"
        ),
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help="Run price augmentation after preprocessing",
    )
    parser.add_argument(
        "--augment-noise-std",
        type=float,
        default=0.3,
        help="Gaussian noise std for price augmentation in ct/kWh (default: 0.3)",
    )
    parser.add_argument(
        "--augment-seed",
        type=int,
        default=None,
        help="Random seed for price augmentation (default: non-deterministic)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.skip_prices and args.skip_weather:
        raise ValueError("Nothing to do: both --skip-prices and --skip-weather were set")

    raw_price_files, preprocessed_price_files = run_price_pipeline(args)
    run_weather_pipeline(args)

    logger.info("Data setup complete.")
    logger.info("Raw price files: %d", len(raw_price_files))
    logger.info("Preprocessed price files: %d", len(preprocessed_price_files))


if __name__ == "__main__":
    main()
