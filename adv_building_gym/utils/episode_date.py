"""Utility for deriving episode date strings from statesource data."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from adv_building_gym.devices.statesources import StateSource


def resolve_episode_date(
    statesources: Sequence[StateSource],
    row_offset: int,
    control_step: int,
) -> str:
    """Derive a date string from *row_offset* using the first statesource with a ``start`` column.

    Args:
        statesources: Sequence of statesource instances to inspect.
        row_offset: Row index into the underlying timeseries.
        control_step: Control timestep in seconds (used for the day-index fallback).

    Returns:
        A date string such as ``"2025-07-15"`` or ``"day-3"`` if no date column exists.
    """
    for src in statesources:
        if src.ts is not None and "start" in src.ts.columns and row_offset < len(src.ts):
            return str(pd.to_datetime(src.ts.iloc[row_offset]["start"]).date())
    # Fallback: day-of-year index
    steps_per_day = int(86400 / control_step)
    return f"day-{row_offset // steps_per_day}"
