"""Price and weather data pipelines for the unified data setup."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from preproc.download import download_links, extract_zip_files
from preproc.e_price.awattar_fetch import fetch_market_data as fetch_awattar_market_data
from preproc.e_price.awattar_price_preproc import preprocess_prices
from preproc.e_price.energy_charts_fetch import (
    fetch_market_data as fetch_energy_charts_market_data,
)
from preproc.e_price.price_augment import augment_prices
from preproc.weather.extract_sfh_csv import extract_sfh_data
from preproc.weather.extract_weather_csv import extract_weather_data

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

SOURCE_SUBDIRS: dict[str, str] = {
    "awattar": "awattar",
    "energy-charts": "e_charts",
}


def resolve_setup_path(path: str | Path) -> Path:
    """Resolve path relative to repository root if needed."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def parse_year_from_filename(file_path: Path) -> int:
    """Extract a year token (e.g. 2025) from a filename."""
    match = re.search(r"(20\d{2})", file_path.name)
    if match is None:
        raise ValueError(f"Could not infer year from filename: {file_path}")
    return int(match.group(1))


def _source_dir(base_dir: Path, source: str) -> Path:
    """Return source-specific subdirectory under base_dir (e.g. data/e_price/awattar)."""
    subdir = base_dir / SOURCE_SUBDIRS[source]
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def _fetch_price_data_for_year(
    year: int,
    source: str,
    output_dir: Path,
    energy_charts_bzn: str,
) -> Path | None:
    """Fetch one year of price data and store it in the source subdirectory.

    Returns:
        Path to the saved CSV file, or None if the fetch failed or returned no data.
    """
    dest_dir = _source_dir(output_dir, source)

    try:
        if source == "awattar":
            start = f"{year}-01-01T00:00:00Z"
            end = f"{year}-12-31T23:59:59Z"
            data = fetch_awattar_market_data(start, end)
            output_path = dest_dir / f"{year}_prices.csv"
        else:
            start = f"{year}-01-01"
            end = f"{year}-12-31"
            data = fetch_energy_charts_market_data(start, end, bzn=energy_charts_bzn)
            output_path = dest_dir / f"{year}_15m_prices.csv"
    except Exception as exc:
        logger.warning(
            "Fetch failed for year %d (source=%s): %s — skipping year", year, source, exc,
        )
        return None

    if data.empty:
        logger.warning(
            "No price data returned for year %d (source=%s) — skipping year", year, source,
        )
        return None

    data.to_csv(output_path, index=False)
    logger.info("Saved fetched price data to %s (%d records)", output_path, len(data))
    return output_path


def _preprocess_price_file(raw_price_file: Path) -> Path:
    """Run awattar_price_preproc.py logic on one raw input file.

    Output is written to the same directory as the raw file.
    """
    if not raw_price_file.exists():
        raise FileNotFoundError(f"Raw price file not found: {raw_price_file}")

    year = parse_year_from_filename(raw_price_file)
    output_path = raw_price_file.parent / f"price_data_{year}_norm.csv"
    preprocess_prices(str(raw_price_file), str(output_path))
    logger.info("Saved preprocessed price data to %s", output_path)
    return output_path


def _augment_price_file(
    preprocessed_file: Path,
    noise_std: float,
    seed: int | None,
) -> Path:
    """Run price augmentation on one preprocessed file.

    Output is written to the same directory as the preprocessed file.
    """
    year = parse_year_from_filename(preprocessed_file)
    output_path = preprocessed_file.parent / f"price_data_{year}_aug.csv"
    augment_prices(str(preprocessed_file), str(output_path), noise_std=noise_std, seed=seed)
    logger.info("Saved augmented price data to %s", output_path)
    return output_path


def run_price_pipeline(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    """Run fetch + preprocess pipeline for electricity prices."""
    if args.skip_prices:
        logger.info("Skipping price pipeline (--skip-prices).")
        return [], []

    active_steps = set(args.steps)
    price_output_dir = resolve_setup_path(args.price_output_dir)
    raw_price_files: list[Path] = []

    source_dir = _source_dir(price_output_dir, args.price_source)

    if "price-fetch" not in active_steps or args.skip_price_fetch:
        if args.raw_price_files:
            raw_price_files = [resolve_setup_path(path) for path in args.raw_price_files]
        else:
            suffix = "_prices.csv" if args.price_source == "awattar" else "_15m_prices.csv"
            raw_price_files = [source_dir / f"{year}{suffix}" for year in args.years]

        missing_files = [path for path in raw_price_files if not path.exists()]
        if missing_files:
            missing_str = "\n".join(str(path) for path in missing_files)
            raise FileNotFoundError(f"Missing raw price files:\n{missing_str}")
    else:
        for year in args.years:
            raw_file = _fetch_price_data_for_year(
                year=year,
                source=args.price_source,
                output_dir=price_output_dir,
                energy_charts_bzn=args.energy_charts_bzn,
            )
            if raw_file is not None:
                raw_price_files.append(raw_file)

        if not raw_price_files:
            logger.warning("No price data fetched for any year — skipping preprocessing.")
            return [], []

    preprocessed_files: list[Path] = []
    if "price-preproc" not in active_steps:
        logger.info("Skipping price preprocessing (price-preproc not in --steps).")
    else:
        for raw_file in raw_price_files:
            preprocessed_file = _preprocess_price_file(raw_file)
            preprocessed_files.append(preprocessed_file)

        if args.augment and preprocessed_files:
            logger.info("Running price augmentation (noise_std=%.3f).", args.augment_noise_std)
            for preprocessed_file in preprocessed_files:
                _augment_price_file(
                    preprocessed_file,
                    noise_std=args.augment_noise_std,
                    seed=args.augment_seed,
                )

    return raw_price_files, preprocessed_files


def _discover_hdf5(zenodo_dir: Path, pattern: str, explicit: list[str] | None) -> list[Path]:
    """Return HDF5 files: explicit list if given, otherwise glob from zenodo_dir."""
    if explicit:
        return sorted(resolve_setup_path(p) for p in explicit)
    found = sorted(zenodo_dir.glob(pattern))
    if not found:
        logger.warning("No files matching %s found in %s", pattern, zenodo_dir)
    return found


def run_weather_pipeline(args: argparse.Namespace) -> None:
    """Run Zenodo download and weather preprocessing pipeline."""
    if args.skip_weather:
        logger.info("Skipping weather pipeline (--skip-weather).")
        return

    active_steps = set(args.steps)
    zenodo_dir = resolve_setup_path(args.zenodo_dir)
    links_file = resolve_setup_path(args.zenodo_links)

    download_count = 0
    extract_count = 0

    if "zenodo-download" not in active_steps:
        logger.info("Skipping Zenodo downloads (zenodo-download not in --steps).")
    else:
        downloaded = download_links(links_file, zenodo_dir, overwrite=args.overwrite_downloads)
        download_count = len(downloaded)

    if "zenodo-extract" not in active_steps:
        logger.info("Skipping archive extraction (zenodo-extract not in --steps).")
    else:
        extracted = extract_zip_files(zenodo_dir, overwrite=args.overwrite_downloads)
        extract_count = len(extracted)

    if "weather-csv" not in active_steps:
        logger.info("Skipping weather CSV extraction (weather-csv not in --steps).")
    else:
        weather_csv_dir = resolve_setup_path(args.weather_csv_dir)
        weather_files = _discover_hdf5(zenodo_dir, "*_weather.hdf5", args.weather_hdf5)
        logger.info("Weather HDF5 files to process: %d", len(weather_files))
        for hdf5_path in weather_files:
            logger.info("Extracting weather CSV from %s", hdf5_path.name)
            try:
                extract_weather_data(
                    hdf5_path=hdf5_path,
                    output_dir=weather_csv_dir,
                    timestamp_unit=args.weather_timestamp_unit,
                )
            except Exception as exc:
                logger.warning("Weather extraction failed for %s: %s — skipping", hdf5_path.name, exc)

    if "sfh-csv" not in active_steps:
        logger.info("Skipping SFH CSV extraction (sfh-csv not in --steps).")
    else:
        sfh_files = _discover_hdf5(zenodo_dir, "*_data_1min.hdf5", args.sfh_hdf5)
        logger.info("SFH HDF5 files to process: %d", len(sfh_files))
        for hdf5_path in sfh_files:
            if args.sfh_csv_dir is not None:
                sfh_csv_dir = resolve_setup_path(args.sfh_csv_dir)
            else:
                sfh_csv_dir = zenodo_dir / f"csvs_{hdf5_path.stem}"

            logger.info("Extracting SFH CSV from %s → %s", hdf5_path.name, sfh_csv_dir)
            try:
                extract_sfh_data(
                    hdf5_path=hdf5_path,
                    output_dir=sfh_csv_dir,
                    group=args.sfh_group,
                )
            except Exception as exc:
                logger.warning("SFH extraction failed for %s: %s — skipping", hdf5_path.name, exc)

    logger.info(
        "Weather pipeline summary: %d downloaded, %d extracted",
        download_count,
        extract_count,
    )
