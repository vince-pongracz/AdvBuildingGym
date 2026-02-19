"""Shared utilities for preprocessing module."""

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(config_path: str | Path | None, default_config_name: str) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file, or None to use default.
        default_config_name: Default config filename to use if config_path is None.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    if config_path is None:
        config_path = Path(__file__).parent / default_config_name

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
