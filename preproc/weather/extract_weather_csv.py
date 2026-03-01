"""
Extract weather data from HDF5 to a single merged CSV file.

Reads weather data from the WEATHER_SERVICE/IN group in the HDF5 file,
merges all weather variables into a single DataFrame, drops NaN rows,
and saves to CSV.

Usage:
    python -m preproc.weather.extract_weather_csv
    # or
    python preproc/weather/extract_weather_csv.py
    # or with custom input file
    python preproc/weather/extract_weather_csv.py --input data/weather/zenodo/2018_weather.hdf5
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

try:
    from .preproc_types import WeatherExtractionStats
    from ..utils import ensure_datetime_index, load_config, resolve_path
except ImportError:
    from preproc_types import WeatherExtractionStats
    from utils import ensure_datetime_index, load_config, resolve_path

# Suppress NaturalNameWarning from PyTables when reading HDF5 files
# The warning is about column names with colons (e.g., 'TEMPERATURE:TOTAL')
# which aren't valid Python identifiers - this is harmless for our use case
warnings.filterwarnings("ignore", category=Warning, module="tables.path")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

f_name: str = "config.yaml"


def clean_column_name(key: str) -> str:
    """
    Clean HDF5 key path to create a readable column name.

    Example: '/WEATHER_SERVICE/IN/WEATHER_TEMPERATURE_TOTAL' -> 'temperature'

    Args:
        key: HDF5 key path string.

    Returns:
        Cleaned lowercase column name.
    """
    # Get the last part of the path
    name = key.split("/")[-1]
    # Remove WEATHER_ prefix
    if name.startswith("WEATHER_"):
        name = name[8:]
    # Remove _TOTAL suffix
    if name.endswith("_TOTAL"):
        name = name[:-6]
    # Remove _GLOBAL suffix
    elif name.endswith("_GLOBAL"):
        name = name[:-7]
    # Convert to lowercase
    return name.lower()


def extract_weather_data(
    hdf5_path: str | Path,
    output_dir: str | Path,
    timestamp_unit: str = "ns",
) -> WeatherExtractionStats:
    """
    Extract weather data from HDF5 file and save as a merged CSV file.

    Args:
        hdf5_path: Path to the HDF5 weather file.
        output_dir: Directory to save CSV file.
        timestamp_unit: Time unit for timestamp conversion ('s' or 'ns').

    Returns:
        Dictionary with extraction statistics.
    """
    hdf5_path_obj = Path(hdf5_path)
    if not hdf5_path_obj.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path_obj}")

    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    stats: WeatherExtractionStats = {
        "variables_found": 0,
        "variables_loaded": 0,
        "rows_before_dropna": 0,
        "rows_after_dropna": 0,
        "columns": [],
        "output_file": None,
        "errors": [],
    }

    with pd.HDFStore(str(hdf5_path_obj), "r") as store:
        # Get all keys
        all_keys = store.keys()

        # Filter for WEATHER_SERVICE keys
        weather_keys = [k for k in all_keys if "WEATHER_SERVICE" in k]
        stats["variables_found"] = len(weather_keys)

        logger.info("Found %d weather variables in HDF5 file", len(weather_keys))

        if not weather_keys:
            raise ValueError("No weather data found in HDF5 file")

        # Load each weather variable as a Series/DataFrame
        dataframes: list[pd.DataFrame] = []

        for key in weather_keys:
            try:
                # Load the data
                data = store[key]

                # Convert Series to DataFrame if needed
                if isinstance(data, pd.Series):
                    df = data.to_frame()
                else:
                    df = data

                # Convert index to datetime if needed
                df = ensure_datetime_index(df, unit=timestamp_unit)

                # Clean up column names
                col_name = clean_column_name(key)

                # If single column, rename it
                if len(df.columns) == 1:
                    df.columns = [col_name]
                else:
                    # Prefix all columns with the variable name
                    df.columns = [f"{col_name}_{c}" for c in df.columns]

                # Remove duplicate timestamps (keep first occurrence)
                if df.index.duplicated().any():
                    n_dups = df.index.duplicated().sum()
                    logger.warning(
                        "%s has %d duplicate timestamps, keeping first occurrence",
                        key.split("/")[-1],
                        n_dups,
                    )
                    df = df[~df.index.duplicated(keep="first")]

                dataframes.append(df)
                stats["variables_loaded"] += 1

                logger.info(
                    "Loaded %s: %d rows, columns: %s",
                    key.split("/")[-1],
                    len(df),
                    list(df.columns),
                )

            except Exception as e:
                stats["errors"].append({"key": key, "error": str(e)})
                logger.error("Failed to load %s: %s", key, e)

        if not dataframes:
            raise ValueError("No weather data could be loaded")

        # Merge all dataframes efficiently using concat (O(n) vs O(n²) for repeated merge)
        logger.info("Merging %d weather variables...", len(dataframes))
        df_merged = pd.concat(dataframes, axis=1, join="inner")

        stats["rows_before_dropna"] = len(df_merged)
        stats["columns"] = list(df_merged.columns)

        # Drop rows with any NaN
        df_merged = df_merged.dropna()
        stats["rows_after_dropna"] = len(df_merged)

        rows_dropped = stats["rows_before_dropna"] - stats["rows_after_dropna"]
        logger.info(
            "Merged data: %d rows (%d dropped due to NaN), %d columns",
            stats["rows_after_dropna"],
            rows_dropped,
            len(df_merged.columns),
        )

        # Set index name
        df_merged.index.name = "timestamp"

        # Generate output filename from input filename
        input_stem = hdf5_path_obj.stem  # e.g., "2018_weather"
        output_file = output_dir_obj / f"{input_stem}.csv"

        # Save to CSV
        df_merged.to_csv(output_file)
        stats["output_file"] = str(output_file)

        logger.info("Saved to: %s", output_file)

    return stats


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract weather data from HDF5 to CSV"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help=f"Path to HDF5 input file (default: from {f_name})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/weather/zenodo/csvs_weather",
        help="Output directory for CSV file (default: data/weather/zenodo/csvs_weather)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help=f"Path to {f_name} (optional)",
    )
    parser.add_argument(
        "--timestamp-unit",
        "-t",
        type=str,
        default="ns",
        choices=["s", "ns"],
        help="Timestamp unit in HDF5 file (default: ns)",
    )
    args = parser.parse_args()

    # Determine input file
    input_file = args.input
    if input_file is None:
        # Try to load from config
        try:
            config = load_config(args.config, f_name)
            input_file = config.get("input_file")
        except FileNotFoundError as e:
            logger.debug("Config file not found, using defaults: %s", e)

        # Default to 2018_weather.hdf5 if not specified or if it's not a weather file
        if input_file is None or "weather" not in input_file.lower():
            input_file = "data/weather/zenodo/2018_weather.hdf5"

    # Resolve relative paths
    input_file_path = resolve_path(input_file)

    logger.info("Input file: %s", input_file_path)
    logger.info("Output directory: %s", args.output)

    try:
        stats = extract_weather_data(
            hdf5_path=input_file_path,
            output_dir=args.output,
            timestamp_unit=args.timestamp_unit,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Variables found: {stats['variables_found']}")
        print(f"Variables loaded: {stats['variables_loaded']}")
        print(f"Rows (before dropna): {stats['rows_before_dropna']}")
        print(f"Rows (after dropna): {stats['rows_after_dropna']}")
        print(f"Columns: {stats['columns']}")
        print(f"\nOutput file: {stats['output_file']}")

        if stats["errors"]:
            print("\nErrors:")
            for err in stats["errors"]:
                print(f"  - {err['key']}: {err['error']}")

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error("Extraction failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
