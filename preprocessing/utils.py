"""Shared utilities for preprocessing module."""

import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# DWD CDC uses -999 as a sentinel for missing measurements
DWD_MISSING_VALUE: float = -999.0

# Columns that carry metadata rather than measurement values
TIMESTAMP_COLUMNS: set[str] = {"timestamp", "start", "start_timestamp", "end_timestamp"}
METADATA_COLUMNS: set[str] = TIMESTAMP_COLUMNS | {"unit", "hour"}


def resolve_path(path: str | Path, project_root: Path | None = None) -> Path:
    """Resolve relative path against project root.

    Args:
        path: Path to resolve (absolute or relative).
        project_root: Project root directory. Defaults to parent of preproc directory.

    Returns:
        Resolved absolute path.
    """
    path = Path(path)
    if path.is_absolute():
        return path

    if project_root is None:
        project_root = Path(__file__).parent.parent

    return project_root / path


def parse_year_from_filename(file_path: Path) -> int:
    """Extract a 4-digit year (e.g. 2025) from a filename."""
    match = re.search(r"(20\d{2})", file_path.name)
    if match is None:
        raise ValueError(f"Could not infer year from filename: {file_path}")
    return int(match.group(1))


def select_columns(
    df: pd.DataFrame,
    keep_columns: list[str],
    source: str | None = None,
) -> pd.DataFrame:
    """Keep only the requested columns, warn about any that are absent.

    Args:
        df: Input dataframe.
        keep_columns: Columns to keep.
        source: Optional label identifying the source dataframe (e.g. DWD data
            type or CSV path) — included in the warning so it's clear which
            input was missing the columns.
    """
    available_columns = [col for col in keep_columns if col in df.columns]
    missing_columns = set(keep_columns) - set(available_columns)
    if missing_columns:
        if source:
            logger.warning("Columns not found in DataFrame [%s]: %s", source, missing_columns)
        else:
            logger.warning("Columns not found in DataFrame: %s", missing_columns)
    return df[available_columns]


def get_measurement_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that carry measurement data (exclude metadata)."""
    return [c for c in df.columns if c not in METADATA_COLUMNS]


def parse_timestamp_column(df: pd.DataFrame) -> pd.Series:
    """Find and parse the timestamp column into a datetime Series."""
    for col in ("timestamp", "start"):
        if col in df.columns:
            return pd.to_datetime(df[col], utc=False)
    raise ValueError(f"No timestamp column found. Columns: {list(df.columns)}")


def is_missing(series: pd.Series, check_sentinel: bool = False) -> pd.Series:
    """Return boolean mask: True where value is NaN or sentinel (-999)."""
    mask = series.isna()
    if check_sentinel:
        mask = mask | (series == DWD_MISSING_VALUE)
    return mask


def ensure_datetime_index(df: pd.DataFrame, unit: str = "s") -> pd.DataFrame:
    """Convert DataFrame index to DatetimeIndex if needed.

    Args:
        df: DataFrame with numeric or datetime index.
        unit: Time unit for conversion ('s' for seconds, 'ns' for nanoseconds).

    Returns:
        DataFrame with DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, unit=unit)
    return df


def fetch_with_retry(
    url: str,
    *,
    params: dict | None = None,
    timeout: int = 60,
    max_retries: int = 3,
    backoff_base: float = 4.0,
    stream: bool = False,
) -> requests.Response:
    """HTTP GET with exponential backoff on transient failures.

    Retries on connection errors, timeouts, and 5xx / 429 status codes.
    Raises on 4xx client errors (except 429) immediately.

    Args:
        url: Request URL.
        params: Query parameters.
        timeout: Per-request timeout in seconds.
        max_retries: Maximum number of retry attempts.
        backoff_base: Base for exponential backoff (sleep = base ** attempt).
        stream: Whether to stream the response body.

    Returns:
        The successful Response object.

    Raises:
        requests.exceptions.RequestException: After all retries are exhausted.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(
                url, params=params, timeout=timeout, stream=stream,
            )
            # Retry on server errors and rate limiting
            if response.status_code == 429 or response.status_code >= 500:
                logger.warning(
                    "HTTP %d from %s (attempt %d/%d)",
                    response.status_code, url, attempt + 1, max_retries + 1,
                )
                last_exc = requests.exceptions.HTTPError(response=response)
            else:
                response.raise_for_status()
                return response
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as exc:
            logger.warning(
                "Request to %s failed (attempt %d/%d): %s",
                url, attempt + 1, max_retries + 1, exc,
            )
            last_exc = exc

        if attempt < max_retries:
            delay = backoff_base ** attempt
            logger.info("Retrying in %.1fs …", delay)
            time.sleep(delay)

    raise last_exc  # type: ignore[misc]
