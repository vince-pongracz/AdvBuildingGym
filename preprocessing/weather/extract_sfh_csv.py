"""
Extract SFH (Single Family Home) data from HDF5 to CSV files.

Extracts data from NO_PV group in the HDF5 file, keeping only _TOT columns
from both HEATPUMP and HOUSEHOLD datasets, merged into a single CSV per SFH.

Usage:
    python -m preproc.weather.extract_sfh_csv
    # or
    python preprocessing/weather/extract_sfh_csv.py
    # or with custom input file
    python preprocessing/weather/extract_sfh_csv.py --input data/weather/zenodo/2018_data_1min.hdf5
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

try:
    from .preproc_types import SFHExtractionStats
    from ..utils import ensure_datetime_index, resolve_path
except ImportError:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from preproc_types import SFHExtractionStats
    from utils import ensure_datetime_index, resolve_path

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_tot_columns(df: pd.DataFrame) -> list[str]:
    """Get columns ending with _TOT suffix."""
    return [col for col in df.columns if col.endswith("_TOT")]


def extract_sfh_data(
    hdf5_path: str | Path,
    output_dir: str | Path,
    group: str = "NO_PV",
) -> SFHExtractionStats:
    """
    Extract SFH (single family home) data from HDF5 file and save as CSV files.

    Args:
        hdf5_path: Path to the HDF5 file.
        output_dir: Directory to save CSV files.
        group: HDF5 group to extract from (default: "NO_PV").

    Returns:
        Dictionary with extraction statistics.
    """
    hdf5_path_obj = Path(hdf5_path)
    if not hdf5_path_obj.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path_obj}")

    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    stats: SFHExtractionStats = {
        "total_buildings": 0,
        "successful": 0,
        "failed": 0,
        "files_created": [],
        "errors": [],
    }

    with pd.HDFStore(str(hdf5_path_obj), "r") as store:
        # Get all keys in the specified group
        all_keys = store.keys()
        group_keys = [k for k in all_keys if f"/{group}/" in k]

        # Extract unique building names (e.g., SFH10, SFH11, ...)
        buildings = sorted(set(k.split("/")[2] for k in group_keys))
        stats["total_buildings"] = len(buildings)

        logger.info(
            "Found %d buildings in %s group: %s",
            len(buildings),
            group,
            buildings[:5] if len(buildings) > 5 else buildings,
        )
        if len(buildings) > 5:
            logger.info("... and %d more", len(buildings) - 5)

        for building in buildings:
            try:
                hp_key = f"/{group}/{building}/HEATPUMP"
                hh_key = f"/{group}/{building}/HOUSEHOLD"

                # Load dataframes
                df_hp = store[hp_key]
                df_hh = store[hh_key]

                # Convert index to datetime if needed
                df_hp = ensure_datetime_index(df_hp, unit="s")
                df_hh = ensure_datetime_index(df_hh, unit="s")

                # Get only _TOT columns
                hp_tot_cols = get_tot_columns(df_hp)
                hh_tot_cols = get_tot_columns(df_hh)

                df_hp_tot = df_hp[hp_tot_cols]
                df_hh_tot = df_hh[hh_tot_cols]

                # Rename columns with prefix
                df_hp_tot = df_hp_tot.rename(columns={col: f"hp_{col}" for col in df_hp_tot.columns})
                df_hh_tot = df_hh_tot.rename(columns={col: f"hh_{col}" for col in df_hh_tot.columns})

                # Merge on index (inner join to keep only matching timestamps)
                df_merged = pd.merge(
                    df_hp_tot,
                    df_hh_tot,
                    left_index=True,
                    right_index=True,
                    how="inner",
                )

                # Per-column NaN counts: how many rows each column would have dropped on its own
                nan_per_col = df_merged.isna().sum()
                nan_breakdown = {col: int(n) for col, n in nan_per_col.items() if n > 0}

                # Drop rows with any NaN in _TOT columns
                rows_before = len(df_merged)
                df_merged = df_merged.dropna()
                rows_after = len(df_merged)
                rows_dropped = rows_before - rows_after

                # Set index name for CSV
                df_merged.index.name = "timestamp"

                # Save to CSV
                output_file = output_dir_obj / f"{building}.csv"
                df_merged.to_csv(output_file)

                stats["successful"] += 1
                stats["files_created"].append(str(output_file))

                if nan_breakdown:
                    breakdown_str = ", ".join(f"{c}={n}" for c, n in sorted(nan_breakdown.items(), key=lambda kv: -kv[1]))
                    logger.info(
                        "Extracted %s: %d rows (%d dropped due to NaN; per-column NaN counts: %s), columns: %s",
                        building,
                        rows_after,
                        rows_dropped,
                        breakdown_str,
                        list(df_merged.columns),
                    )
                else:
                    logger.info(
                        "Extracted %s: %d rows (0 dropped due to NaN), columns: %s",
                        building,
                        rows_after,
                        list(df_merged.columns),
                    )

            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append({"building": building, "error": str(e)})
                logger.error("Failed to extract %s: %s", building, e)

    return stats


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract SFH data from HDF5 to CSV files"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/weather/zenodo/2018_data_1min.hdf5",
        help="Path to HDF5 input file (default: data/weather/zenodo/2018_data_1min.hdf5)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/weather/zenodo/csvs",
        help="Output directory for CSV files (default: data/weather/zenodo/csvs)",
    )
    parser.add_argument(
        "--group",
        "-g",
        type=str,
        default="NO_PV",
        help="HDF5 group to extract (default: NO_PV)",
    )
    args = parser.parse_args()

    # Resolve relative paths
    input_file_path = resolve_path(args.input)

    # Generate output directory name from input filename
    raw_input_fname = input_file_path.stem
    output_dir = f"{args.output}_{raw_input_fname}"

    logger.info("Input file: %s", input_file_path)
    logger.info("Output directory: %s", output_dir)
    logger.info("Group: %s", args.group)

    try:
        stats = extract_sfh_data(
            hdf5_path=input_file_path,
            output_dir=output_dir,
            group=args.group,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Total buildings: {stats['total_buildings']}")
        print(f"Successfully extracted: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"\nFiles created in: {output_dir}")

        if stats["errors"]:
            print("\nErrors:")
            for err in stats["errors"]:
                print(f"  - {err['building']}: {err['error']}")

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error("Extraction failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
