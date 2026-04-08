"""Reusable normalisation utilities for pandas Series."""

import logging
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


class Normalisation(Enum):
    """Normalisation types"""
    MAX_ABS_SCALING = "max_abs"
    ABS_MIN_MAX_SCALING = "abs_min_max"
    MIN_MAX_SCALING = "min_max"
    STANDARDISATION = "std"

    @classmethod
    def init(cls, value: "Normalisation | str | None") -> "Normalisation | None":
        """Convert a string to a Normalisation enum member, pass through enum/None unchanged."""
        if value is None or isinstance(value, cls):
            return value
        return cls(value)


def _safe_divide(series: pd.Series, numerator: pd.Series, divisor: float) -> pd.Series:
    """Divide numerator by divisor, returning zeros and logging if divisor is zero."""
    if divisor != 0:
        return numerator / divisor
    else:
        logger.warning("Divisor is zero at normalising %s!", series.name)

    if series.notna().any():
        logger.warning("Series '%s' is constant zero — normalisation returns zeros", series.name)
    return series * 0.0


def get_scale_factor(series: pd.Series, method: Normalisation | None) -> float:
    """Return the denominator that ``normalise_series`` would divide by.

    Useful when downstream code needs to convert between raw and normalised
    values (e.g. ``raw = norm * scale_factor``) using the same scale that
    was applied during normalisation.

    For ``MIN_MAX_SCALING`` the relationship is
    ``raw = norm * scale_factor + series.min()``; for ``STANDARDISATION``
    it is ``raw = norm * scale_factor + series.mean()``.

    Returns 1.0 when *method* is ``None`` (no normalisation).
    """
    match method:
        case Normalisation.ABS_MIN_MAX_SCALING:
            return float(max(abs(series.min()), abs(series.max()))) or 1.0
        case Normalisation.MAX_ABS_SCALING:
            return float(series.abs().max()) or 1.0
        case Normalisation.MIN_MAX_SCALING:
            return float(series.max() - series.min()) or 1.0
        case Normalisation.STANDARDISATION:
            return float(series.std()) or 1.0
        case None:
            return 1.0
        case _:
            raise ValueError(f"Unknown normalisation method: {method}")


def normalise_series(series: pd.Series, method: Normalisation | None) -> pd.Series:
    """Normalise a pandas Series using the given method.

    Args:
        series: Input data.
        method: Normalisation strategy. None returns the series unchanged.

    Returns:
        Normalised pandas Series.
    """
    match method:
        case Normalisation.ABS_MIN_MAX_SCALING:
            # norm = val / max(|min|, |max|), maps to [-1, 1]
            abs_max = max(abs(series.min()), abs(series.max()))
            return _safe_divide(series, series, abs_max)
        case Normalisation.MAX_ABS_SCALING:
            max_abs = series.abs().max()
            return _safe_divide(series, series, max_abs)
        case Normalisation.MIN_MAX_SCALING:
            range_ = series.max() - series.min()
            return _safe_divide(series, series - series.min(), range_)
        case Normalisation.STANDARDISATION:
            std = series.std()
            return _safe_divide(series, series - series.mean(), std)
        case None:
            return series
        case _:
            raise ValueError(f"Unknown normalisation method: {method}")
