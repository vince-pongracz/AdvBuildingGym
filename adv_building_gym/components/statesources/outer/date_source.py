"""DateSource — single source of truth for the calendar date, driven by a dated CSV."""

import logging
from collections import OrderedDict
from typing import ClassVar, Set

import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from ..base import StateSource
from ..csv_loader import CsvLoader
from ..reloadable import CsvReloadable
from adv_building_gym.components.registry import ComponentRegistry
from adv_building_gym._common.episode_date import date_column

logger = logging.getLogger(__name__)


class DateSource(StateSource, CsvReloadable):
    """Owns ``s_date`` (normalised day-of-year) and answers the day-offset queries the
    ``DataVariantManager`` needs (``available_rows`` / ``start_year`` / ``date_at``) without
    exposing ``ts`` to the core layer.
    """

    _exclude_params: ClassVar[Set[str]] = {"s_date_norm"}

    def __init__(self, name: str = "date", ds_path: str | None = None) -> None:
        super().__init__(name=name)
        self._date_col: str | None = None
        self.s_date_norm: float = 0.0
        self.loader = CsvLoader(ds_path, on_reload=self._run_post_load)

    def _post_load_data_processing(self) -> None:
        """Locate the timestamp column and precompute normalised day-of-year (leap → 366)."""
        assert self.ts is not None, "ts must be set by CsvLoader before _post_load_data_processing."
        self._date_col = date_column(self.ts)
        if self._date_col is not None:
            parsed = pd.to_datetime(self.ts[self._date_col], utc=True, errors="coerce")
            year_length = np.where(parsed.dt.is_leap_year, 366.0, 365.0)
            self.ts["s_date"] = ((parsed.dt.dayofyear - 1) / year_length).astype(np.float32)
        else:
            if self.is_new_data_source:
                logger.warning("DateSource '%s': no date column — s_date set to 0.", self.name)
            self.ts["s_date"] = np.float32(0.0)

        # Only the normalised day-of-year (s_date) and the raw timestamp column
        # (start_year / date_at) are read afterwards; drop the rest of the CSV.
        self._keep_ts_columns({"s_date", *( {self._date_col} if self._date_col else set() )})

    def setup_spaces(self, state_spaces: OrderedDict, action_spaces: OrderedDict) -> tuple:
        if "s_date" not in state_spaces:
            state_spaces["s_date"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        return state_spaces, action_spaces

    def update_state(self, states, info: dict) -> None:
        if self.ts is None:
            raise RuntimeError(
                f"DateSource '{self.name}': no CSV loaded. The DataCombinator must push a "
                "date variant before update_state is called."
            )
        idx = min(self.effective_index, len(self.ts) - 1)
        self.s_date_norm = float(self.ts["s_date"].iloc[idx])
        states["s_date"][0] = np.float32(self.s_date_norm)

    # ----- day-offset facts for DataVariantManager (no ts leak) -----
    def available_rows(self) -> int:
        return len(self.ts) if self.ts is not None else 0

    def start_year(self) -> int | None:
        if self.ts is None or self._date_col is None or len(self.ts) == 0:
            return None
        return int(pd.Timestamp(self.ts[self._date_col].iloc[0]).year)

    def date_at(self, row_offset: int) -> str | None:
        """ISO date string at ``row_offset``, or ``None`` when undated / out of range."""
        if self.ts is None or self._date_col is None or row_offset >= len(self.ts):
            return None
        return str(pd.to_datetime(self.ts[self._date_col].iloc[row_offset]).date())


# register with ComponentRegistry
ComponentRegistry.register('statesource', DateSource)
