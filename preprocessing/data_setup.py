"""Unified data setup script for AdvBuildingGym.

By default, this script runs all three pipelines:
1) electricity price fetch + preprocessing (+ optional synthesis)
2) weather/Zenodo fetch + preprocessing
3) DWD CDC weather download + preprocessing

Examples:
    python preprocessing/data_setup.py
    python preprocessing/data_setup.py --skip-weather
    python preprocessing/data_setup.py --skip-prices --steps zenodo-extract weather-csv
    python preprocessing/data_setup.py --skip-weather --years 2025 --skip-price-fetch --raw-price-files data/e_price/2025_prices.csv
    python preprocessing/data_setup.py --skip-prices --skip-wpuq --steps dwd-fetch dwd-preprocess
    python preprocessing/data_setup.py --skip-prices --skip-wpuq --dwd-station-id 04177 --dwd-upsample-method duplicate
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

from preprocessing.pipelines import (
    run_price_pipeline,
    run_weather_pipeline,
    run_dwd_pipeline,
    run_hh_consumption_pipeline,
    run_synthesize,
    run_quality_report,
)

logger = logging.getLogger(__name__)

DEFAULT_YEARS: list[int] = list(range(2016, 2027))

ALL_PRICE_STEPS: set[str] = {"price-fetch", "price-preproc"}
ALL_WEATHER_STEPS: set[str] = {"zenodo-download", "zenodo-extract", "weather-csv", "sfh-csv", "hh-consumption"}
ALL_DWD_STEPS: set[str] = {"dwd-fetch", "dwd-preprocess"}
ALL_REPORT_STEPS: set[str] = {"data-quality-report"}
ALL_STEPS: set[str] = ALL_PRICE_STEPS | ALL_WEATHER_STEPS | ALL_DWD_STEPS | ALL_REPORT_STEPS


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
        nargs="+",
        choices=["awattar", "energy-charts"],
        default=["awattar", "energy-charts"],
        help="Price data source(s) to fetch and preprocess (default: both)",
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
        "--skip-wpuq",
        action="store_true",
        help="Skip the entire WPuQ/Zenodo weather pipeline",
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
        "--skip-weather",
        action="store_true",
        help="Skip all weather pipelines (both WPuQ/Zenodo and DWD)",
    )
    parser.add_argument(
        "--skip-dwd",
        action="store_true",
        help="Skip the entire DWD weather pipeline",
    )
    parser.add_argument(
        "--dwd-station-id",
        default="04177",
        help="DWD station ID (default: 04177 = Rheinstetten)",
    )
    parser.add_argument(
        "--dwd-output-dir",
        default="data/weather/dwd",
        help="Directory for DWD downloaded and preprocessed data",
    )
    parser.add_argument(
        "--dwd-upsample-method",
        choices=["average", "duplicate"],
        default="average",
        help="Upsample method for DWD 10-min to 5-min (default: average)",
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
        "--hh-consumption-output-dir",
        default="data/hh_consumption/wpuq",
        help="Output directory for household consumption CSVs (default: data/hh_consumption/wpuq)",
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
        "--quality-report-dir",
        nargs="?",
        const="data/quality_reports",
        default="data/quality_reports",
        help="Output directory for the data quality report CSV and chart "
             "(default: data/quality_reports; bare flag uses the same default)",
    )

    parser.add_argument(
        "--synthesize",
        action="store_true",
        help="Run synthetic dataset generation on yearly price and weather CSVs after all pipelines. "
            "Active configs are listed in preprocessing/synthesize_config.yaml.",
    )
    parser.add_argument(
        "--synthesize-config",
        default="preprocessing/synthesize_config.yaml",
        help="Top-level synthesise config (default: preprocessing/synthesize_config.yaml)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def _log_settings(args: argparse.Namespace) -> None:
    """Log the submitted command and parsed CLI settings before pipeline execution."""
    settings = ", ".join(f"{k}={v!r}" for k, v in sorted(vars(args).items()))
    cmd = " ".join(sys.argv)
    HR = "=" * 80
    logger.info("\n%s\nCMD: %s\nData setup settings: %s\n%s", HR, cmd, settings, HR)


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # --skip-weather is a convenience shorthand for --skip-wpuq --skip-dwd
    if args.skip_weather:
        args.skip_wpuq = True
        args.skip_dwd = True

    _log_settings(args)

    has_quality_step = "data-quality-report" in set(args.steps)
    if args.skip_prices and args.skip_wpuq and args.skip_dwd and not has_quality_step:
        raise ValueError("Nothing to do: --skip-prices, --skip-wpuq, and --skip-dwd were all set")

    def _step_banner(title: str) -> None:
        HR = "=" * 80
        logger.info("\n%s\nSTEP: %s\n%s", HR, title, HR)

    _step_banner("Price pipeline")
    price_stats = run_price_pipeline(args)

    _step_banner("Zenodo weather pipeline")
    weather_stats = run_weather_pipeline(args)

    _step_banner("DWD pipeline")
    dwd_stats = run_dwd_pipeline(args)

    _step_banner("Household consumption pipeline")
    hh_stats = run_hh_consumption_pipeline(args)

    _step_banner("Synthesis")
    # Synthesis discovers its inputs by globbing the preprocessed output dirs.
    syn_stats = run_synthesize(args)

    _step_banner("Data quality report")
    # Runs last so it covers all produced files
    quality_stats = run_quality_report(args)

    logger.info("====================")
    logger.info("Data setup complete.")
    logger.info("--- Price pipeline ---")
    logger.info("  Raw price files: %d", price_stats["raw"])
    logger.info("  Preprocessed price files: %d", price_stats["preprocessed"])
    logger.info("--- Zenodo weather pipeline ---")
    logger.info("  Files downloaded: %d", weather_stats["downloaded"])
    logger.info("  Archives extracted: %d", weather_stats["extracted"])
    logger.info("  Weather CSVs produced: %d", weather_stats["weather_csvs"])
    logger.info("  SFH CSVs produced: %d", weather_stats["sfh_csvs"])
    logger.info("--- DWD pipeline ---")
    logger.info("  Data types fetched: %d", dwd_stats["data_types"])
    logger.info("  Merged rows (10-min): %d", dwd_stats["merged_rows"])
    logger.info("  Years processed: %d", dwd_stats["years"])
    logger.info("--- Household consumption pipeline ---")
    logger.info("  Files produced: %d", hh_stats["files"])
    logger.info("--- Synthesis ---")
    logger.info("  Price files synthesised: %d", syn_stats["price"])
    logger.info("  Weather files synthesised: %d", syn_stats["weather"])
    logger.info("--- Data quality report ---")
    logger.info("  Datasets analysed: %d", quality_stats["datasets"])
    logger.info("====================")
    logger.info("Data setup finished!")


if __name__ == "__main__":
    main()
