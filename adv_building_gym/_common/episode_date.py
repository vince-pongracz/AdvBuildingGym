"""Utility for deriving episode date strings from statesource data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import pandas as pd

from adv_building_gym._common.constants import SECONDS_PER_DAY

if TYPE_CHECKING:
    from adv_building_gym.components.statesources import StateSource


def resolve_episode_date(
    statesources: Sequence[StateSource],
    row_offset: int,
    control_step: int,
) -> str:
    """Date string for *row_offset* from the first statesource with a ``start`` column,
    else a ``"day-N"`` fallback (uses control_step for the day index)."""
    for src in statesources:
        if src.ts is not None and "start" in src.ts.columns and row_offset < len(src.ts):
            return str(pd.to_datetime(src.ts.iloc[row_offset]["start"]).date())

    # Fallback: day-of-year index
    steps_per_day = int(SECONDS_PER_DAY / control_step)
    return f"day-{row_offset // steps_per_day}"
