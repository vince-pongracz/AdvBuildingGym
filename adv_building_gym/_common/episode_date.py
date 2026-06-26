"""Utility for deriving episode date strings from a reference statesource."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from adv_building_gym._common.constants import SECONDS_PER_DAY
from adv_building_gym._common.date_provider import DateProvider

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
    date_source: Optional[DateProvider],
    row_offset: int,
    control_step: int,
) -> str:
    """Date string for *row_offset* from the *date_source*, else a ``"day-N"`` fallback."""
    if date_source is not None:
        date = date_source.date_at(row_offset)
        if date is not None:
            return date

    # Fallback: day-of-year index when no dated source is available
    steps_per_day = int(SECONDS_PER_DAY / control_step)
    return f"day-{row_offset // steps_per_day}"
