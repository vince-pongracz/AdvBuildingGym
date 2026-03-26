"""Download and extract historic + recent 10-minute DWD CDC weather data.

Fetches wind, solar, and air_temperature zip archives for a given station
from the DWD open-data server, extracts them, and converts the
semicolon-delimited .txt files to .csv.

Link: https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/
"""

import io
import logging
import re
import sys
import zipfile
from pathlib import Path

_PROJECT_ROOT_STR = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_STR)

import pandas as pd

from preproc.utils import fetch_with_retry

logger = logging.getLogger("main")

STATION_ID: str = "04177"  # Rheinstetten
DATA_TYPES: list[str] = ["wind", "solar", "air_temperature"]
URL_BASE_TEMPLATE: str = (
    "https://opendata.dwd.de/climate_environment/CDC/"
    "observations_germany/climate/10_minutes/{data_type}/{period}/"
)
PERIODS: list[str] = ["historical", "recent"]
# Resolve project root relative to this file: dwd_fetch.py -> preproc/weather/dwd/
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
DWD_DIR: Path = PROJECT_ROOT / "data" / "weather" / "dwd"
DOWNLOAD_DIR: Path = DWD_DIR / "downloaded"


def get_zip_urls_for_station(base_url: str, station_id: str) -> list[str]:
    """Parse the DWD directory listing and return zip URLs matching station_id."""
    response = fetch_with_retry(base_url, timeout=60)

    # DWD listing is plain HTML with <a href="filename.zip"> links
    # Historical: 10minutenwerte_TU_04177_20081101_20091231_hist.zip
    # Recent:     10minutenwerte_TU_04177_akt.zip
    pattern = re.compile(
        rf'href="([^"]*_{station_id}_[^"]*\.zip)"', re.IGNORECASE
    )
    filenames = pattern.findall(response.text)
    if not filenames:
        logger.warning("No zip files found for station %s at %s", station_id, base_url)
        return []

    urls = [base_url + fname for fname in filenames]
    logger.info("Found %d zip file(s) for station %s at %s", len(urls), station_id, base_url)
    return urls


def download_and_extract(url: str, output_dir: Path) -> list[Path]:
    """Download a zip from url and extract to output_dir. Return extracted paths."""
    logger.info("Downloading %s", url)
    response = fetch_with_retry(url, timeout=120)

    extracted_paths: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        for member in zf.namelist():
            target = output_dir / member
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as f:
                f.write(zf.read(member))
            extracted_paths.append(target)
            logger.info("  Extracted: %s", target.name)

    return extracted_paths


def txt_to_csv(txt_path: Path) -> Path:
    """Convert a DWD semicolon-delimited .txt to .csv and return the csv path."""
    csv_path = txt_path.with_suffix(".csv")
    df = pd.read_csv(txt_path, sep=";", skipinitialspace=True)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    df.to_csv(csv_path, index=False)
    logger.info("  Converted %s -> %s", txt_path.name, csv_path.name)
    return csv_path


def load_and_concat_csvs(csv_paths: list[Path]) -> pd.DataFrame:
    """Load multiple CSVs (same schema) and concatenate into one DataFrame."""
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    df.sort_values("MESS_DATUM", inplace=True)
    df.drop_duplicates(subset="MESS_DATUM", keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_data_type(data_type: str, station_id: str, output_dir: Path) -> pd.DataFrame | None:
    """Download, extract, convert all zips for one data type (historical + recent)."""
    zip_urls: list[str] = []
    for period in PERIODS:
        base_url = URL_BASE_TEMPLATE.format(data_type=data_type, period=period)
        zip_urls.extend(get_zip_urls_for_station(base_url, station_id))

    if not zip_urls:
        return None

    all_data_txt_paths: list[Path] = []
    for url in zip_urls:
        extracted = download_and_extract(url, output_dir)
        # Keep only the produkt_*.txt files (actual measurement data)
        data_files = [p for p in extracted if p.name.lower().startswith("produkt") and p.suffix.lower() == ".txt"]
        all_data_txt_paths.extend(data_files)

    if not all_data_txt_paths:
        logger.warning("No produkt*.txt files found for %s", data_type)
        return None

    csv_paths = [txt_to_csv(p) for p in all_data_txt_paths]
    df = load_and_concat_csvs(csv_paths)
    logger.info("Loaded %s: %d rows, columns: %s", data_type, len(df), list(df.columns))
    return df


def fetch_all(station_id: str = STATION_ID, output_dir: Path = DOWNLOAD_DIR) -> dict[str, pd.DataFrame]:
    """Download all data types for a station. Return dict of DataFrames keyed by type."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframes: dict[str, pd.DataFrame] = {}
    for data_type in DATA_TYPES:
        logger.info("=== Fetching %s data for station %s ===", data_type, station_id)
        df = fetch_data_type(data_type, station_id, output_dir)
        if df is not None:
            dataframes[data_type] = df
        else:
            logger.error("Failed to fetch %s data — skipping", data_type)

    return dataframes


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    dataframes = fetch_all()
    if not dataframes:
        logger.error("No data fetched for any type. Exiting.")
        return
    logger.info("Fetched %d data type(s): %s", len(dataframes), list(dataframes.keys()))


if __name__ == "__main__":
    main()
