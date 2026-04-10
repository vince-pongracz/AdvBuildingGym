"""Data loading utilities for day-based CSV plotting.

Two loading strategies:

1. **Year-partitioned files** (weather, price, household consumption):
   ``load_day_csv`` / ``load_days`` locate the correct ``{year}`` file and
   extract a single calendar day.

2. **Profile files** (desired temperature, EV schedule):
   ``load_profile_csv`` / ``load_profiles`` load standalone single-day
   template CSVs that are date-independent.

All loaders add a ``minutes`` column (minutes since midnight, float32)
used as the common x-axis across all day-data plots.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_day_csv(
    directory: Path,
    file_pattern: str,
    timestamp_col: str,
    date: datetime,
) -> pd.DataFrame:
    """Load CSV data for a single calendar day.

    Returns an empty DataFrame if the file does not exist or contains no
    data for the requested day.
    """
    year = date.year
    filename = file_pattern.format(year=year)
    filepath = directory / filename
    if not filepath.exists():
        logger.warning("File not found: %s", filepath)
        return pd.DataFrame()

    df = pd.read_csv(filepath, parse_dates=[timestamp_col])
    if df[timestamp_col].dt.tz is not None:
        df[timestamp_col] = df[timestamp_col].dt.tz_localize(None)

    day_start = pd.Timestamp(date)
    day_end = day_start + pd.Timedelta(days=1)
    mask = (df[timestamp_col] >= day_start) & (df[timestamp_col] < day_end)
    day_df = df.loc[mask].copy()

    if day_df.empty:
        logger.warning("No data for %s in %s", date.date(), filepath)
        return pd.DataFrame()

    day_df["minutes"] = (
        (day_df[timestamp_col] - day_start).dt.total_seconds() / 60.0
    ).astype(np.float32)

    return day_df


def load_days(
    directory: Path,
    file_pattern: str,
    timestamp_col: str,
    dates: list[datetime],
) -> dict[str, pd.DataFrame]:
    """Load CSV data for multiple days. Returns ``{date_label: DataFrame}``."""
    result: dict[str, pd.DataFrame] = {}
    for date in dates:
        df = load_day_csv(directory, file_pattern, timestamp_col, date)
        if not df.empty:
            result[str(date.date())] = df
    return result


def load_profile_csv(
    filepath: Path,
    timestamp_col: str,
) -> pd.DataFrame:
    """Load a single-day profile CSV and add a ``minutes`` column.

    Returns an empty DataFrame when the file is missing or empty.
    """
    if not filepath.exists():
        logger.warning("Profile file not found: %s", filepath)
        return pd.DataFrame()

    df = pd.read_csv(filepath, parse_dates=[timestamp_col])
    if df.empty:
        return pd.DataFrame()

    if df[timestamp_col].dt.tz is not None:
        df[timestamp_col] = df[timestamp_col].dt.tz_localize(None)

    day_start = df[timestamp_col].iloc[0].normalize()
    df["minutes"] = (
        (df[timestamp_col] - day_start).dt.total_seconds() / 60.0
    ).astype(np.float32)
    return df


def load_profiles(
    directory: Path,
    filenames: list[str],
    timestamp_col: str,
) -> dict[str, pd.DataFrame]:
    """Load all profile CSVs. Returns ``{filename_stem: DataFrame}``."""
    result: dict[str, pd.DataFrame] = {}
    for name in filenames:
        df = load_profile_csv(directory / name, timestamp_col)
        if not df.empty:
            result[Path(name).stem] = df
    return result
