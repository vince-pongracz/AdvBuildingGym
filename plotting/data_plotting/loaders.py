"""Data loading utilities for day-based CSV plotting.

Two loading strategies:

1. **Year-partitioned files** (weather, price, household consumption):
   ``load_days`` reads each ``{year}`` file exactly once and groups rows
   by calendar date, returning a per-day slice for every requested date.

2. **Profile files** (desired temperature, EV schedule):
   ``load_profile_csv`` / ``load_profiles`` load standalone single-day
   template CSVs that are date-independent.

All loaders add a ``minutes`` column (minutes since midnight, float32)
used as the common x-axis across all day-data plots.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_WARN_FUTURE_DATA: bool = False


def set_warn_future_data(enabled: bool) -> None:
    """Toggle "No data for <date>" warnings for dates in the future.

    Default behaviour (disabled) suppresses the warning when the requested
    day has not yet occurred, since missing data is expected in that case.
    """
    global _WARN_FUTURE_DATA
    _WARN_FUTURE_DATA = bool(enabled)


def _is_future(d: datetime) -> bool:
    return d.date() > date.today()


def _read_year_csv(filepath: Path, timestamp_col: str) -> pd.DataFrame:
    """Read a year CSV once and normalise tz-aware timestamps to naive."""
    df = pd.read_csv(filepath, parse_dates=[timestamp_col])
    if df[timestamp_col].dt.tz is not None:
        df[timestamp_col] = df[timestamp_col].dt.tz_localize(None)
    return df


def _slice_days(
    df: pd.DataFrame,
    timestamp_col: str,
    dates: list[datetime],
    filepath: Path,
) -> dict[str, pd.DataFrame]:
    """Slice a year DataFrame into per-day frames keyed by ISO date string.

    Groups rows by calendar date once, so emitting N day-slices costs a
    single linear pass over the year instead of N full scans.
    """
    date_keys = df[timestamp_col].dt.normalize()
    # {Timestamp(YYYY-MM-DD): ndarray of positional indices}
    indices = date_keys.groupby(date_keys).indices

    out: dict[str, pd.DataFrame] = {}
    for d in dates:
        day_start = pd.Timestamp(d.year, d.month, d.day)
        idx = indices.get(day_start)
        if idx is None or len(idx) == 0:
            if _WARN_FUTURE_DATA or not _is_future(d):
                logger.warning("No data for %s in %s", d.date(), filepath)
            continue
        day_df = df.iloc[idx].copy()
        day_df["minutes"] = (
            (day_df[timestamp_col] - day_start).dt.total_seconds() / 60.0
        ).astype(np.float32)
        out[str(d.date())] = day_df
    return out


def _available_years(
    directory: Path,
    file_pattern: str,
    years: set[int],
) -> set[int]:
    """Return the subset of *years* for which a data file exists on disk."""
    available: set[int] = set()
    for year in years:
        filepath = directory / file_pattern.format(year=year)
        if filepath.exists():
            available.add(year)
    return available


def load_days(
    directory: Path,
    file_pattern: str,
    timestamp_col: str,
    dates: list[datetime],
) -> dict[str, pd.DataFrame]:
    """Load CSV data for multiple days. Returns ``{date_label: DataFrame}``.

    Reads each ``{year}`` file exactly once and groups its rows by date,
    so total parse cost is O(years) instead of O(days).
    """
    by_year: dict[int, list[datetime]] = {}
    for d in dates:
        by_year.setdefault(d.year, []).append(d)

    available = _available_years(directory, file_pattern, set(by_year))
    missing = sorted(set(by_year) - available)
    if missing:
        logger.warning(
            "Skipping years with no data file in %s (pattern %s): %s",
            directory, file_pattern, ", ".join(str(y) for y in missing),
        )

    result: dict[str, pd.DataFrame] = {}
    for year in sorted(by_year):
        if year not in available:
            continue
        filepath = directory / file_pattern.format(year=year)
        df = _read_year_csv(filepath, timestamp_col)
        result.update(_slice_days(df, timestamp_col, by_year[year], filepath))
    return result


def _syn_cfg_stem(file_pattern: str, year: int) -> str:
    """Stem (no .csv) of the canonical per-year file for *year*."""
    name = file_pattern.format(year=year)
    return name[:-4] if name.endswith(".csv") else name


def discover_syn_cfg_files(
    directory: Path,
    file_pattern: str,
    years: set[int],
) -> dict[str, dict[int, Path]]:
    """Find sibling synthesised CSVs for every year that has them.

    Returns ``{cfg_name: {year: filepath}}``. ``cfg_name`` is everything
    after the canonical stem, e.g. ``syn_cfg_1_pos`` for a file like
    ``2016_merged_04177_syn_cfg_1_pos.csv``.
    """
    result: dict[str, dict[int, Path]] = {}
    if not directory.exists():
        return result
    for year in years:
        stem = _syn_cfg_stem(file_pattern, year)
        for match in directory.glob(f"{stem}_syn_cfg_*.csv"):
            cfg_name = match.stem[len(stem) + 1:]  # strip "<stem>_"
            result.setdefault(cfg_name, {})[year] = match
    return result


def load_syn_cfg_days(
    directory: Path,
    file_pattern: str,
    timestamp_col: str,
    dates: list[datetime],
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load every discoverable ``*_syn_cfg_*.csv`` for the given dates.

    Returns ``{cfg_name: {date_label: DataFrame}}``. Empty when no
    synthesised siblings exist in *directory*. Each per-year file is
    parsed once per cfg variant.
    """
    by_year: dict[int, list[datetime]] = {}
    for d in dates:
        by_year.setdefault(d.year, []).append(d)

    cfg_files = discover_syn_cfg_files(directory, file_pattern, set(by_year))
    if not cfg_files:
        return {}

    result: dict[str, dict[str, pd.DataFrame]] = {}
    for cfg_name, year_paths in sorted(cfg_files.items()):
        frames: dict[str, pd.DataFrame] = {}
        for year in sorted(year_paths):
            year_dates = by_year.get(year)
            if not year_dates:
                continue
            filepath = year_paths[year]
            df = _read_year_csv(filepath, timestamp_col)
            frames.update(_slice_days(df, timestamp_col, year_dates, filepath))
        if frames:
            result[cfg_name] = frames
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
