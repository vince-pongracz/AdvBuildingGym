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


def _safe_divide(numerator: pd.Series, divisor: float) -> pd.Series:
    """Divide numerator by divisor, returning zeros and logging if divisor is zero."""
    if divisor != 0:
        return numerator / divisor
    else:
        logger.warning("Divisor is zero at normalising %s!", numerator.name)

    if numerator.notna().any():
        logger.warning("Series '%s' is constant zero — normalisation returns zeros", numerator.name)
    return numerator * 0.0


def normalise_with_scale_factor(
    series: pd.Series,
    method: Normalisation | None,
) -> tuple[pd.Series, float]:
    """Normalise a pandas Series and return the scale factor used.

    Returns:
        (normalised_series, scale_factor) — the denominator that was
        divided out.  ``raw ≈ norm * scale_factor`` for symmetric methods
        (ABS_MIN_MAX, MAX_ABS).  For MIN_MAX_SCALING the offset is
        ``series.min()``; for STANDARDISATION it is ``series.mean()``.
        scale_factor is 1.0 when *method* is ``None``.
    """
    match method:
        case Normalisation.ABS_MIN_MAX_SCALING:
            scale = float(max(abs(series.min()), abs(series.max()))) or 1.0
            return _safe_divide(series, scale), scale
        case Normalisation.MAX_ABS_SCALING:
            scale = float(series.abs().max()) or 1.0
            return _safe_divide(series, scale), scale
        case Normalisation.MIN_MAX_SCALING:
            scale = float(series.max() - series.min()) or 1.0
            return _safe_divide(series - series.min(), scale), scale
        case Normalisation.STANDARDISATION:
            scale = float(series.std()) or 1.0
            return _safe_divide(series - series.mean(), scale), scale
        case None:
            return series, 1.0
        case _:
            raise ValueError(f"Unknown normalisation method: {method}")


def get_scale_factor(series: pd.Series, method: Normalisation | None) -> float:
    """Return the denominator that normalisation would divide by.

    Standalone implementation — computes only the scalar, without
    normalising the full series.
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
    """Normalise a pandas Series using the given method."""
    normalised, _ = normalise_with_scale_factor(series, method)
    return normalised
