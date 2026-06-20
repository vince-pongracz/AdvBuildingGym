"""Utility for deriving episode date strings from a reference statesource."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import pandas as pd

from adv_building_gym._common.constants import SECONDS_PER_DAY

if TYPE_CHECKING:
    from adv_building_gym.components.statesources import StateSource

# Per-row timestamp columns recognised in reference CSVs, in priority order.
# Weather CSVs label it "timestamp"; price CSVs label it "start".
_DATE_COLUMNS = ("start", "timestamp", "start_timestamp", "date", "datetime")


def date_column(df: pd.DataFrame) -> Optional[str]:
    """First recognised per-row timestamp column present in *df*, else None."""
    for col in _DATE_COLUMNS:
        if col in df.columns:
            return col
    return None


def resolve_episode_date(
    date_ref_source: Optional["StateSource"],
    row_offset: int,
    control_step: int,
) -> str:
    """Date string for *row_offset* read from *date_ref_source*'s timestamp column,
    else a ``"day-N"`` fallback (uses control_step for the day index)."""
    if date_ref_source is not None and date_ref_source.ts is not None:
        ts = date_ref_source.ts
        col = date_column(ts)
        if col is not None and row_offset < len(ts):
            return str(pd.to_datetime(ts.iloc[row_offset][col]).date())

    # Fallback: day-of-year index when no dated reference source is available
    steps_per_day = int(SECONDS_PER_DAY / control_step)
    return f"day-{row_offset // steps_per_day}"
