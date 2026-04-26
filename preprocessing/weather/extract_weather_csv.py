"""
Extract weather data from HDF5 to a single merged CSV file.

Reads weather data from the WEATHER_SERVICE/IN group in the HDF5 file,
merges all weather variables into a single DataFrame, drops NaN rows,
and saves to CSV.

Usage:
    python -m preproc.weather.extract_weather_csv
    # or
    python preprocessing/weather/extract_weather_csv.py
    # or with custom input file
    python preprocessing/weather/extract_weather_csv.py --input data/weather/zenodo/2018_weather.hdf5
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

try:
    from .preproc_types import WeatherExtractionStats
    from ..utils import ensure_datetime_index, resolve_path
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from preproc_types import WeatherExtractionStats
    from utils import ensure_datetime_index, resolve_path

# Suppress NaturalNameWarning from PyTables when reading HDF5 files
# The warning is about column names with colons (e.g., 'TEMPERATURE:TOTAL')
# which aren't valid Python identifiers - this is harmless for our use case
warnings.filterwarnings("ignore", category=Warning, module="tables.path")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Rename WPuQ/Zenodo column names to match environment statesource conventions
COLUMN_RENAMES: dict[str, str] = {
    "temperature": "temp_amb",
    "relative_humidity": "rel_humidity",
    "solar_irradiance": "direct_sun_shine",
    "wind_direction": "wind_dir",
    "wind_speed": "avg_wind_speed",
}

# Columns to drop (not used by the environment)
COLUMNS_TO_DROP: list[str] = [
    "atmospheric_pressure",
    "precipitation_rate",
    "probability_of_precipitation",
    "apparent_temperature",
    "wind_gust_speed",
]


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
    timestamp_unit: str = "s",
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
                        "%s/%s has %d duplicate timestamps, keeping first occurrence (source: %s)",
                        hdf5_path_obj.name,
                        key.split("/")[-1],
                        n_dups,
                        hdf5_path_obj,
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

        # Per-column NaN counts: how many rows each column would have dropped on its own
        nan_per_col = df_merged.isna().sum()
        nan_breakdown = {col: int(n) for col, n in nan_per_col.items() if n > 0}

        # Drop rows with any NaN
        df_merged = df_merged.dropna()
        stats["rows_after_dropna"] = len(df_merged)

        rows_dropped = stats["rows_before_dropna"] - stats["rows_after_dropna"]
        if nan_breakdown:
            breakdown_str = ", ".join(f"{c}={n}" for c, n in sorted(nan_breakdown.items(), key=lambda kv: -kv[1]))
            logger.info(
                "Merged data: %d rows (%d dropped due to NaN; per-column NaN counts: %s), %d columns",
                stats["rows_after_dropna"],
                rows_dropped,
                breakdown_str,
                len(df_merged.columns),
            )
        else:
            logger.info(
                "Merged data: %d rows (0 dropped due to NaN), %d columns",
                stats["rows_after_dropna"],
                len(df_merged.columns),
            )

        # Rename columns to match environment statesource conventions
        df_merged.rename(columns=COLUMN_RENAMES, inplace=True)

        # Zenodo CSVs have only direct_sun_shine (no diffuse component).
        # Create sun_shine alias so downstream code can use a single column name.
        if "direct_sun_shine" in df_merged.columns and "sun_shine" not in df_merged.columns:
            df_merged["sun_shine"] = df_merged["direct_sun_shine"]

        # Replace negative sentinel values in irradiance columns with 0
        # (real irradiance is never negative; some sources use e.g. -999 for missing data)
        for col in ("sun_shine", "direct_sun_shine"):
            if col in df_merged.columns:
                neg_mask = df_merged[col] < 0
                n_neg = neg_mask.sum()
                if n_neg > 0:
                    logger.warning(
                        "%d negative sentinel values in '%s' replaced with 0",
                        n_neg, col,
                    )
                    df_merged.loc[neg_mask, col] = 0.0

        # Drop columns not needed by the environment
        cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df_merged.columns]
        if cols_to_drop:
            df_merged.drop(columns=cols_to_drop, inplace=True)
            logger.info("Dropped columns: %s", cols_to_drop)

        stats["columns"] = list(df_merged.columns)

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
        default="data/weather/zenodo/2018_weather.hdf5",
        help="Path to HDF5 input file (default: data/weather/zenodo/2018_weather.hdf5)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/weather/zenodo/csvs_weather",
        help="Output directory for CSV file (default: data/weather/zenodo/csvs_weather)",
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

    # Resolve relative paths
    input_file_path = resolve_path(args.input)

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
