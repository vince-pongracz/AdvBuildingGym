"""Preprocess and merge downloaded DWD 10-minute weather data.

Takes per-type DataFrames (from dwd_fetch), selects relevant columns,
merges on MESS_DATUM, renames columns, and drops all-missing rows.
Can be run standalone or imported.

Units / columns (irradiance): DWD ``GS_10`` / ``DS_10`` are documented as
10-minute sums of *global* / *diffuse* shortwave radiation in J/cm² (see DWD
BESCHREIBUNG_obsgermany_climate_10min_solar_*.pdf). They are converted here
to mean W/m² over the 10-min interval (factor 10000/600 ≈ 16.6667) before
the 10→5-min upsampling. Column mapping:

    GS_10 → ``sun_shine``       — global horizontal irradiance, W/m²
    DS_10 → ``diff_sun_shine``  — diffuse-only diagnostic, W/m²

``sun_shine`` is the canonical global irradiance column used downstream
(``WeatherDataSource``, plotting, training); it matches the Zenodo/WPuQ
``solar_irradiance`` column. ``diff_sun_shine`` is kept for diagnostics only
— *do not* sum it with ``sun_shine``: that double-counts the diffuse
component (a bug that lived in earlier revisions of this script).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.utils import DWD_MISSING_VALUE, get_measurement_columns, select_columns
from preprocessing.weather.dwd.dwd_fetch import STATION_ID, DWD_DIR, fetch_all

PREPROCESS_DIR: Path = DWD_DIR / "preprocessed"

logger = logging.getLogger("main")

KEEP_COLUMNS: list[str] = ["MESS_DATUM", "GS_10", "DS_10", "FF_10", "DD_10", "TT_10", "RF_10"]
RENAME_COLUMNS: dict[str, str] = {
    "MESS_DATUM": "timestamp",
    "FF_10": "avg_wind_speed",
    "DD_10": "wind_dir",
    "GS_10": "sun_shine",        # global horizontal irradiance (not direct!)
    "DS_10": "diff_sun_shine",    # diffuse-only diagnostic
    "TT_10": "temp_amb",
    "RF_10": "rel_humidity",
}


def merge_dataframes(dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """Merge multiple DataFrames on MESS_DATUM with outer join for full coverage."""
    merged: pd.DataFrame | None = None

    for data_type, df in dataframes.items():
        df = select_columns(df, KEEP_COLUMNS, source=data_type)

        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="MESS_DATUM", how="outer")

    if merged is None:
        return None

    merged.sort_values("MESS_DATUM", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    merged.rename(columns=RENAME_COLUMNS, inplace=True)

    # Convert timestamp from YYYYMMDDHHmm integer to UTC datetime
    merged["timestamp"] = pd.to_datetime(
        merged["timestamp"].astype(str), format="%Y%m%d%H%M", utc=True,
    )

    # Replace -999 sentinel values with NaN before computing derived columns,
    # so that sums like sun_shine = direct + diffuse don't produce bogus
    # negative values (e.g. -1998) from sentinel arithmetic.
    measurement_cols = get_measurement_columns(merged)
    sentinel_mask = merged[measurement_cols] == DWD_MISSING_VALUE
    n_sentinels = int(sentinel_mask.sum().sum())
    if n_sentinels > 0:
        merged[measurement_cols] = merged[measurement_cols].where(~sentinel_mask)
        logger.info("Replaced %d sentinel values (-999) with NaN", n_sentinels)

    # Drop rows where all measurement columns are NaN
    all_missing = merged[measurement_cols].isna().all(axis=1)
    n_dropped = int(all_missing.sum())
    if n_dropped > 0:
        merged = merged[~all_missing].reset_index(drop=True)
        logger.info("Dropped %d rows where all measurements were NaN", n_dropped)

    # Convert irradiance from J/cm² per 10 min (DWD archival unit) to mean W/m²
    # over the 10-min interval: (x J/cm²) * (10000 cm²/m²) / (600 s) = x * 50/3 W/m².
    # Done before upsampling so all downstream values share a single unit (W/m²)
    # and linear interpolation between samples is meaningful (mean irradiance at
    # the 5-min midpoint).
    J_PER_CM2_PER_10MIN_TO_W_PER_M2 = 10000.0 / 600.0
    for col in ("sun_shine", "diff_sun_shine"):
        if col in merged.columns:
            merged[col] = merged[col] * J_PER_CM2_PER_10MIN_TO_W_PER_M2

    return merged


def upsample_to_5min(df: pd.DataFrame, method: str = "average") -> pd.DataFrame:
    """Upsample from 10-minute to 5-minute resolution.

    Args:
        df: DataFrame with 'timestamp' as datetime column.
        method: 'average' interpolates between consecutive rows,
                'duplicate' repeats each row for the intermediate 5-min slot.
    """
    if method not in ("average", "duplicate"):
        raise ValueError(f"Unknown upsample method: {method!r}. Use 'average' or 'duplicate'.")

    df = df.set_index("timestamp").sort_index()
    measurement_cols = list(df.columns)

    # Resample to 5-min grid, original rows land on their timestamps
    upsampled = df.resample("5min").asfreq()

    if method == "average":
        # Linearly interpolate NaN slots (the new 5-min midpoints).
        # Sentinels are already replaced with NaN in merge_dataframes(),
        # so limit=1 avoids interpolating across multi-row gaps.
        upsampled = upsampled.interpolate(method="linear", limit=1)
    elif method == "duplicate":
        # Forward-fill: each new 5-min row gets the previous 10-min value
        upsampled = upsampled.ffill(limit=1)

    upsampled = upsampled.reset_index()
    logger.info(
        "Upsampled to 5-min (%s): %d -> %d rows",
        method, len(df), len(upsampled),
    )
    return upsampled


def split_by_year(
    merged: pd.DataFrame,
    output_dir: Path,
    station_id: str,
    upsample_method: str = "average",
) -> None:
    """Split by year, write missing reports, then upsample per year.

    Pipeline per year:
      1. Write missing-entry report (on original 10-min data)
      2. Upsample to 5-min resolution
      3. Write upsampled CSV
    """
    measurement_cols = get_measurement_columns(merged)
    years = merged["timestamp"].dt.year
    merged = merged.copy()
    merged["year"] = years

    for year, year_df in merged.groupby("year"):
        year_df = year_df.drop(columns=["year"])

        # 1. Missing-entry report (on original 10-min data, before upsampling)
        # Sentinels are already replaced with NaN in merge_dataframes()
        has_missing = year_df[measurement_cols].isna().any(axis=1)
        year_df_with_date = year_df.loc[has_missing].copy()
        year_df_with_date["date"] = year_df_with_date["timestamp"].dt.date
        days_with_missing = year_df_with_date["date"].unique()

        report_path = output_dir / f"{year}_missing_entries.txt"
        with open(report_path, "w") as f:
            f.write(f"Year {year}: {len(days_with_missing)} day(s) with at least one missing measurement\n\n")
            for day in sorted(days_with_missing):
                day_rows = year_df_with_date[year_df_with_date["date"] == day]
                missing_cols_per_row = day_rows[measurement_cols].isna().sum(axis=0)
                cols_affected = [c for c in measurement_cols if missing_cols_per_row[c] > 0]
                f.write(f"  {day}: {len(day_rows)} missing row(s), columns: {', '.join(cols_affected)}\n")
        logger.info("Missing report: %s (%d days)", report_path.name, len(days_with_missing))

        # 2. Upsample to 5-min resolution
        year_df = upsample_to_5min(year_df, method=upsample_method)

        # 3. Write upsampled CSV
        year_csv = output_dir / f"{year}_merged_{station_id}.csv"
        year_df.to_csv(year_csv, index=False)
        logger.info("Written %s (%d rows)", year_csv.name, len(year_df))


def preprocess(
    dataframes: dict[str, pd.DataFrame],
    output_dir: Path,
    station_id: str,
    upsample_method: str = "average",
) -> pd.DataFrame | None:
    """Run the full preprocessing pipeline.

    Steps:
      1. Merge data types on MESS_DATUM, rename columns, drop all-missing rows
      2. Write full merged CSV (10-min resolution)
      3. Per year: missing report -> upsample to 5-min -> write CSV

    Args:
        upsample_method: 'average' (linear interpolation) or 'duplicate' (forward-fill).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = merge_dataframes(dataframes)
    if merged is None:
        logger.error("Merge produced no data.")
        return None

    # Write the full merged CSV (10-min resolution)
    full_path = output_dir / f"merged_{station_id}.csv"
    merged.to_csv(full_path, index=False)
    logger.info(
        "Merged CSV written to %s (%d rows, %d columns)",
        full_path, len(merged), len(merged.columns),
    )

    split_by_year(merged, output_dir, station_id, upsample_method)

    return merged


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    dataframes = fetch_all()
    if not dataframes:
        logger.error("No data fetched for any type. Exiting.")
        return

    preprocess(dataframes, PREPROCESS_DIR, STATION_ID)


if __name__ == "__main__":
    main()
