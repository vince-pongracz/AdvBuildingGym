"""Shared utilities for preprocessing module."""

from pathlib import Path

import pandas as pd


def resolve_path(path: str | Path, project_root: Path | None = None) -> Path:
    """Resolve relative path against project root.

    Args:
        path: Path to resolve (absolute or relative).
        project_root: Project root directory. Defaults to parent of preproc directory.

    Returns:
        Resolved absolute path.
    """
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path

    if project_root is None:
        project_root = Path(__file__).parent.parent

    return project_root / path


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
