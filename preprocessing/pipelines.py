"""Price and weather data pipelines for the unified data setup."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from preprocessing.download import download_links, extract_zip_files
from preprocessing.e_price.awattar_fetch import fetch_market_data as fetch_awattar_market_data
from preprocessing.e_price.awattar_price_preproc import preprocess_prices
from preprocessing.e_price.energy_charts_fetch import (
    fetch_market_data as fetch_energy_charts_market_data,
)
from preprocessing.synthesize import run_synthesis
from preprocessing.data_quality_report import run_data_quality_report
from preprocessing.utils import filter_paths_by_years, parse_year_from_filename, resolve_path
from preprocessing.weather.dwd.dwd_fetch import fetch_all as dwd_fetch_all
from preprocessing.weather.dwd.dwd_preprocess import preprocess as dwd_preprocess
from preprocessing.hh_consumption.extract_hh_consumption import extract_hh_consumption
from preprocessing.weather.extract_sfh_csv import extract_sfh_data
from preprocessing.weather.extract_weather_csv import extract_weather_data

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

SOURCE_SUBDIRS: dict[str, str] = {
    "awattar": "awattar",
    "energy-charts": "e_charts",
}


def resolve_setup_path(path: str | Path) -> Path:
    """Resolve path relative to repository root if needed."""
    return resolve_path(path, PROJECT_ROOT)



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
    output_path = raw_price_file.parent / f"price_data_{year}.csv"
    preprocess_prices(str(raw_price_file), str(output_path), year=year)
    logger.info("Saved preprocessed price data to %s", output_path)
    return output_path


def _run_price_pipeline_for_source(
    source: str,
    args: argparse.Namespace,
    active_steps: set[str],
) -> tuple[list[Path], list[Path]]:
    """Run fetch + preprocess for a single price source."""
    price_output_dir = resolve_setup_path(args.price_output_dir)
    raw_price_files: list[Path] = []

    source_dir = _source_dir(price_output_dir, source)

    if "price-fetch" not in active_steps or args.skip_price_fetch:
        if args.raw_price_files:
            raw_price_files = [resolve_setup_path(path) for path in args.raw_price_files]
        else:
            suffix = "_prices.csv" if source == "awattar" else "_15m_prices.csv"
            raw_price_files = [source_dir / f"{year}{suffix}" for year in args.years]

        missing_files = [path for path in raw_price_files if not path.exists()]
        if missing_files:
            missing_str = "\n".join(str(path) for path in missing_files)
            raise FileNotFoundError(f"Missing raw price files ({source}):\n{missing_str}")
    else:
        for year in args.years:
            raw_file = _fetch_price_data_for_year(
                year=year,
                source=source,
                output_dir=price_output_dir,
                energy_charts_bzn=args.energy_charts_bzn,
            )
            if raw_file is not None:
                raw_price_files.append(raw_file)

        if not raw_price_files:
            logger.warning("No price data fetched for any year (source=%s) — skipping preprocessing.", source)
            return [], []

    preprocessed_files: list[Path] = []
    if "price-preproc" not in active_steps:
        logger.info("Skipping price preprocessing for %s (price-preproc not in --steps).", source)
    else:
        for raw_file in raw_price_files:
            preprocessed_file = _preprocess_price_file(raw_file)
            preprocessed_files.append(preprocessed_file)

    return raw_price_files, preprocessed_files


def run_price_pipeline(args: argparse.Namespace) -> dict[str, int]:
    """Run fetch + preprocess pipeline for electricity prices.

    Supports multiple price sources (e.g. awattar and energy-charts).
    args.price_source can be a single string or a list of strings.

    Returns:
        Counts dict: {"raw": <n>, "preprocessed": <n>}.
    """
    if args.skip_prices:
        logger.info("Skipping price pipeline (--skip-prices).")
        return {"raw": 0, "preprocessed": 0}

    active_steps = set(args.steps)

    # Normalise price_source to a list for uniform handling
    sources = args.price_source if isinstance(args.price_source, list) else [args.price_source]

    raw_count = 0
    preprocessed_count = 0

    for source in sources:
        logger.info("Running price pipeline for source: %s", source)
        raw, preprocessed = _run_price_pipeline_for_source(source, args, active_steps)
        raw_count += len(raw)
        preprocessed_count += len(preprocessed)

    return {"raw": raw_count, "preprocessed": preprocessed_count}


def _discover_hdf5(
    zenodo_dir: Path,
    pattern: str,
    explicit: list[str] | None,
    years: list[int] | None,
) -> list[Path]:
    """Return HDF5 files: explicit list if given (bypasses the year filter),
    otherwise glob from zenodo_dir restricted to the selected years."""
    if explicit:
        return sorted(resolve_setup_path(p) for p in explicit)
    found = sorted(zenodo_dir.glob(pattern))
    if not found:
        logger.warning("No files matching %s found in %s", pattern, zenodo_dir)
        return found
    selected = filter_paths_by_years(found, years)
    if len(selected) < len(found):
        logger.info(
            "Year filter %s: keeping %d of %d file(s) matching %s",
            sorted(set(years)), len(selected), len(found), pattern,
        )
    return selected


def run_weather_pipeline(args: argparse.Namespace) -> dict[str, int]:
    """Run Zenodo download and weather preprocessing pipeline.

    Returns:
        Dict with counts: downloaded, extracted, weather_csvs, sfh_csvs.
    """
    stats = {"downloaded": 0, "extracted": 0, "weather_csvs": 0, "sfh_csvs": 0}
    if args.skip_wpuq:
        logger.info("Skipping WPuQ/Zenodo weather pipeline (--skip-wpuq).")
        return stats

    active_steps = set(args.steps)
    zenodo_dir = resolve_setup_path(args.zenodo_dir)
    links_file = resolve_setup_path(args.zenodo_links)

    download_count = 0
    extract_count = 0

    if "zenodo-download" not in active_steps:
        logger.info("Skipping Zenodo downloads (zenodo-download not in --steps).")
    else:
        downloaded = download_links(
            links_file, zenodo_dir, overwrite=args.overwrite_downloads, years=args.years,
        )
        download_count = len(downloaded)

    if "zenodo-extract" not in active_steps:
        logger.info("Skipping archive extraction (zenodo-extract not in --steps).")
    else:
        extracted = extract_zip_files(zenodo_dir, overwrite=args.overwrite_downloads, years=args.years)
        extract_count = len(extracted)

    if "weather-csv" not in active_steps:
        logger.info("Skipping weather CSV extraction (weather-csv not in --steps).")
    else:
        weather_csv_dir = resolve_setup_path(args.weather_csv_dir)
        weather_files = _discover_hdf5(zenodo_dir, "*_weather.hdf5", args.weather_hdf5, args.years)
        logger.info("Weather HDF5 files to process: %d", len(weather_files))
        for hdf5_path in weather_files:
            logger.info("Extracting weather CSV from %s", hdf5_path.name)
            try:
                extract_weather_data(
                    hdf5_path=hdf5_path,
                    output_dir=weather_csv_dir,
                    timestamp_unit=args.weather_timestamp_unit,
                )
                stats["weather_csvs"] += 1
            except Exception as exc:
                logger.warning("Weather extraction failed for %s: %s — skipping", hdf5_path.name, exc)

    if "sfh-csv" not in active_steps:
        logger.info("Skipping SFH CSV extraction (sfh-csv not in --steps).")
    else:
        sfh_files = _discover_hdf5(zenodo_dir, "*_data_1min.hdf5", args.sfh_hdf5, args.years)
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
                stats["sfh_csvs"] += 1
            except Exception as exc:
                logger.warning("SFH extraction failed for %s: %s — skipping", hdf5_path.name, exc)

    stats["downloaded"] = download_count
    stats["extracted"] = extract_count

    logger.info(
        "Weather pipeline summary: %d downloaded, %d extracted",
        download_count,
        extract_count,
    )

    return stats


def run_dwd_pipeline(args: argparse.Namespace) -> dict[str, int]:
    """Run DWD weather data download and preprocessing pipeline.

    Steps:
      1. dwd-fetch: Download raw 10-min data from DWD CDC open-data server
      2. dwd-preprocess: Merge, upsample to 5-min, split by year

    Returns:
        Dict with counts: data_types, merged_rows, years.
    """
    stats = {"data_types": 0, "merged_rows": 0, "years": 0}
    if args.skip_dwd:
        logger.info("Skipping DWD pipeline (--skip-dwd).")
        return stats

    active_steps = set(args.steps)
    station_id = args.dwd_station_id
    dwd_output_dir = resolve_setup_path(args.dwd_output_dir)

    dataframes: dict | None = None

    if "dwd-fetch" not in active_steps:
        logger.info("Skipping DWD fetch (dwd-fetch not in --steps).")
    else:
        download_dir = dwd_output_dir / "downloaded"
        download_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Fetching DWD data for station %s → %s", station_id, download_dir)
        try:
            dataframes = dwd_fetch_all(station_id=station_id, output_dir=download_dir, years=args.years)
        except Exception as exc:
            logger.error("DWD fetch failed: %s", exc)
            return stats

        if not dataframes:
            logger.warning("No DWD data fetched — skipping preprocessing.")
            return stats

        stats["data_types"] = len(dataframes)
        logger.info("DWD fetch complete: %d data type(s)", len(dataframes))

    if "dwd-preprocess" not in active_steps:
        logger.info("Skipping DWD preprocessing (dwd-preprocess not in --steps).")
        return stats

    if dataframes is None:
        # Fetch was skipped — run it now to get dataframes for preprocessing
        download_dir = dwd_output_dir / "downloaded"
        logger.info("Running DWD fetch (needed for preprocessing) for station %s", station_id)
        try:
            dataframes = dwd_fetch_all(station_id=station_id, output_dir=download_dir, years=args.years)
        except Exception as exc:
            logger.error("DWD fetch failed: %s", exc)
            return stats

        if not dataframes:
            logger.warning("No DWD data fetched — cannot preprocess.")
            return stats

    preprocess_dir = dwd_output_dir / "preprocessed"
    logger.info("Preprocessing DWD data → %s", preprocess_dir)
    merged = dwd_preprocess(
        dataframes,
        output_dir=preprocess_dir,
        station_id=station_id,
        upsample_method=args.dwd_upsample_method,
        years=args.years,
    )
    if merged is not None:
        stats["merged_rows"] = len(merged)
        stats["years"] = merged["timestamp"].dt.year.nunique()
        logger.info("DWD preprocessing complete: %d rows", len(merged))
    else:
        logger.error("DWD preprocessing produced no data.")

    return stats


def _discover_synthesis_inputs(args: argparse.Namespace) -> tuple[list[Path], list[Path], list[Path]]:
    """Discover yearly preprocessed CSVs eligible for synthesis.

    Globs the price (awattar/e_charts), weather (DWD/Zenodo), and
    hh_consumption output directories for per-year CSVs, excluding any
    pre-existing synthesised siblings (``*_syn_cfg_*.csv``), aggregated
    consumption files, and files outside the selected years.
    """
    price_dir = resolve_setup_path(args.price_output_dir)
    dwd_dir = resolve_setup_path(args.dwd_output_dir) / "preprocessed"
    zenodo_csv_dir = resolve_setup_path(args.weather_csv_dir)
    hh_dir_path = resolve_setup_path(args.hh_consumption_output_dir)
    years = getattr(args, "years", None)

    def _eligible(paths) -> list[Path]:
        return filter_paths_by_years(sorted(p for p in paths if "_syn_cfg_" not in p.name), years)

    price_files: list[Path] = []
    if price_dir.is_dir():
        for subdir in SOURCE_SUBDIRS.values():
            sub = price_dir / subdir
            if sub.is_dir():
                price_files.extend(_eligible(sub.glob("price_data_*.csv")))

    weather_files: list[Path] = []
    if dwd_dir.is_dir():
        weather_files.extend(_eligible(dwd_dir.glob("*_merged_*.csv")))
    if zenodo_csv_dir.is_dir():
        weather_files.extend(_eligible(zenodo_csv_dir.glob("*_weather.csv")))

    hh_files: list[Path] = []
    if hh_dir_path.is_dir():
        # Per-building only; aggregated CSV is a sum and doesn't need its own noise stream.
        hh_files.extend(_eligible(p for p in hh_dir_path.glob("*_SFH*.csv")))

    return price_files, weather_files, hh_files


def run_synthesize(args: argparse.Namespace) -> dict[str, int]:
    """Run synthetic dataset generation as the final pipeline step.

    Discovers price and weather CSVs by globbing the configured output
    directories (excluding any ``*_syn_cfg_*.csv`` siblings), then for each
    input CSV and each active syn_cfg listed in
    ``preprocessing/synthesize_config.yaml`` writes a sibling
    ``<stem>_<syn_cfg_name>.csv`` next to the source file.

    Returns:
        Dict with counts of synthesised files per domain.
    """
    if not getattr(args, "synthesize", False):
        logger.info("No synthesis")
        return {"price": 0, "weather": 0, "hh_consumption": 0}

    price_files, weather_files, hh_files = _discover_synthesis_inputs(args)
    logger.info(
        "Discovered %d price + %d weather + %d hh_consumption CSVs for synthesis",
        len(price_files), len(weather_files), len(hh_files),
    )

    cfg_path = resolve_setup_path(getattr(args, "synthesize_config", "preprocessing/synthesize_config.yaml"))

    return run_synthesis(
        price_files=price_files,
        weather_files=weather_files,
        hh_consumption_files=hh_files,
        top_cfg_path=cfg_path,
    )


def run_hh_consumption_pipeline(args: argparse.Namespace) -> dict[str, int]:
    """Extract household consumption CSVs from SFH data.

    Reads 1-min SFH CSVs (produced by sfh-csv step), resamples to 5-min,
    converts W to kW, and writes per-building + aggregated CSVs.

    Returns:
        Dict with count of files produced.
    """
    stats = {"files": 0}
    if getattr(args, "skip_wpuq", False):
        logger.info("Skipping hh-consumption pipeline (--skip-wpuq).")
        return stats

    active_steps = set(args.steps)
    if "hh-consumption" not in active_steps:
        logger.info("Skipping hh-consumption (hh-consumption not in --steps).")
        return stats

    zenodo_dir = resolve_setup_path(args.zenodo_dir)
    output_dir = resolve_setup_path(args.hh_consumption_output_dir)

    # Discover all csvs_<year>_data_1min directories for the selected years
    all_sfh_dirs = sorted(zenodo_dir.glob("csvs_*_data_1min"))
    if not all_sfh_dirs:
        logger.warning("No csvs_*_data_1min directories found in %s", zenodo_dir)
        return stats

    sfh_dirs = filter_paths_by_years(all_sfh_dirs, getattr(args, "years", None))
    if len(sfh_dirs) < len(all_sfh_dirs):
        logger.info(
            "Year filter %s: keeping %d of %d SFH data directories",
            sorted(set(args.years)), len(sfh_dirs), len(all_sfh_dirs),
        )
    if not sfh_dirs:
        logger.warning("No SFH data directories left for selected years %s", sorted(set(args.years)))
        return stats

    logger.info("Found %d SFH data directories to process", len(sfh_dirs))

    for sfh_dir in sfh_dirs:
        logger.info("Extracting hh consumption from %s", sfh_dir.name)
        try:
            result = extract_hh_consumption(
                input_dir=sfh_dir,
                output_dir=output_dir,
            )
            stats["files"] += len(result.get("files_created", []))
        except Exception as exc:
            logger.warning(
                "hh-consumption extraction failed for %s: %s — skipping",
                sfh_dir.name, exc,
            )

    return stats


def run_quality_report(args: argparse.Namespace) -> dict[str, int]:
    """Run the data-quality-report step if requested.

    Returns:
        Dict with count of datasets analysed.
    """
    active_steps = set(args.steps)
    if "data-quality-report" not in active_steps:
        return {"datasets": 0}

    dwd_dir = resolve_setup_path(args.dwd_output_dir) / "preprocessed"
    zenodo_dir = resolve_setup_path(args.weather_csv_dir)
    price_dir = resolve_setup_path(args.price_output_dir)
    output_dir = resolve_setup_path(
        getattr(args, "quality_report_dir", "data/quality_reports")
    )

    # Deliberately NOT year-filtered: the report covers the full data inventory.
    reports = run_data_quality_report(
        dwd_dir=dwd_dir,
        zenodo_weather_dir=zenodo_dir,
        price_dir=price_dir,
        output_dir=output_dir,
    )
    return {"datasets": len(reports)}
