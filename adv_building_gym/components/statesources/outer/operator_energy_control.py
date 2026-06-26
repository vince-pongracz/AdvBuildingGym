import logging
from collections import OrderedDict
from typing import ClassVar, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from adv_building_gym._common.constants import SECONDS_PER_HOUR

from ..base import StateSource
from ..csv_loader import CsvLoader
from ..forecastable import Forecastable
from ..reloadable import CsvReloadable
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)

_SECONDS_PER_DAY: int = 24 * SECONDS_PER_HOUR


class OperatorEnergyControl(StateSource, Forecastable, CsvReloadable):
    """Grid-operator power limit, optionally driven by a sparse step-change CSV.

    Publishes ``ctxt_operator_max_power_kW`` (raw kW) every step (single source of truth).
    Two modes
    ---------
    - Constant (no ``ds_path``): ``max_power_kW`` every step.
    - CSV-driven (event-based, like ``EVState``): the sparse CSV is parsed into a
      sorted list of step-change events keyed by a per-day iteration index
      (``seconds_from_midnight // control_step``). ``update_state`` holds the last
      limit and only changes it when an event fires; the profile repeats daily, so the
      held value is reset to ``max_power_kW`` (the pre-first-event fallback) at the start
      of each simulated day. ``forecast`` resolves future held values from the same events.

    CSV format
    ----------
    Columns: ``start, max_power_kW``. Example::

        start,max_power_kW
        2025-01-02 00:00:00,20.0
        2025-01-02 07:00:00,12.0
    """

    _context_params: ClassVar[Set[str]] = {"control_step"}
    _exclude_params: ClassVar[Set[str]] = {"_events", "_event_lookup", "_steps_per_day", "_held_limit_kW"}

    def __init__(
        self,
        name: str,
        max_power_kW: float,
        ds_path: str | None = None,
        control_step: float = 300.0,
    ) -> None:
        """
        Args:
            name: Datasource identifier.
            max_power_kW: Fallback limit in kW. Used directly when no CSV
                is supplied, and as the initial value held before the first
                event when a CSV is supplied.
            ds_path: Optional path to a sparse step-change CSV.
            control_step: Environment control step in seconds.
        """
        if max_power_kW <= 0:
            raise ValueError(
                f"OperatorEnergyControl '{name}': max_power_kW must be > 0, got {max_power_kW}."
            )
        super().__init__(name=name, control_step=control_step)
        self.max_power_kW: float = float(max_power_kW)

        # control steps per simulated day — wraps iteration into the per-day profile
        self._steps_per_day: int = int(_SECONDS_PER_DAY // int(self.control_step))
        if self._steps_per_day <= 0:
            raise ValueError(
                f"OperatorEnergyControl '{name}': control_step={control_step}s is too large "
                "to fit at least one step per day."
            )

        # Sparse step-change events (CSV-driven), keyed by per-day iteration index.
        # _events: sorted [(day_index, kW)]; 
        # _event_lookup: {day_index: kW} for O(1) hits.
        # Empty in constant mode → the held limit stays at the fallback.
        self._events: List[Tuple[int, float]] = []
        self._event_lookup: Dict[int, float] = {}
        # Persistent held limit (kW); updated only when an event fires (EVState-style).
        self._held_limit_kW: float = self.max_power_kW

        if ds_path is not None:
            # two-phase init: CsvLoader(ds_path=...) would call _post_load before
            # self.loader is assigned (self.ts None during that window)
            self.loader = CsvLoader(None, on_reload=self._run_post_load)
            self.loader.reload(ds_path)
            logger.info(
                "OperatorEnergyControl '%s': CSV-driven from %s (fallback %.3f kW)",
                name, ds_path, self.max_power_kW,
            )
        else:
            logger.info(
                "OperatorEnergyControl '%s': constant limit = %.3f kW",
                name, self.max_power_kW,
            )

    def _post_load_data_processing(self) -> None:
        """Parse the sparse CSV into step-change events keyed by per-day iteration index."""
        df = self.ts
        if df is None or len(df) == 0:
            raise ValueError(f"OperatorEnergyControl '{self.name}': CSV {self.ds_path} is empty.")
        if "start" not in df.columns or "max_power_kW" not in df.columns:
            raise ValueError(f"OperatorEnergyControl '{self.name}': CSV {self.ds_path} must have columns 'start' and 'max_power_kW'.")

        df = df.copy()
        df["start"] = pd.to_datetime(df["start"])

        # Per-day iteration index from time-of-day; sort so later same-index rows win.
        df["iter_idx"] = (
            df["start"].dt.hour * SECONDS_PER_HOUR
            + df["start"].dt.minute * 60
            + df["start"].dt.second
        ) // int(self.control_step)
        df = df.sort_values("iter_idx").reset_index(drop=True)

        # reject NaN / non-positive — the reward divides by this
        if df["max_power_kW"].isna().any() or (df["max_power_kW"] <= 0).any():
            raise ValueError(
                f"OperatorEnergyControl '{self.name}': max_power_kW must be > 0 in every "
                f"row of {self.ds_path}."
            )

        # Build the sparse event map (last same-index row wins). Out-of-day events are dropped.
        lookup: Dict[int, float] = {}
        for iter_idx, value in zip(df["iter_idx"].to_numpy(), df["max_power_kW"].to_numpy()):
            iter_idx = int(iter_idx)
            if 0 <= iter_idx < self._steps_per_day:
                lookup[iter_idx] = float(value)
        self._event_lookup = lookup
        self._events = sorted(lookup.items())
        # Reset the held limit to the fallback (re-applied per day in update_state).
        self._held_limit_kW = self.max_power_kW

        if self.is_new_data_source:
            values = [v for _, v in self._events] or [self.max_power_kW]
            logger.info(
                "OperatorEnergyControl '%s': loaded %d step-change(s) from %s (range %.3f-%.3f kW)",
                self.name, len(self._events), self.ds_path, min(values), max(values),
            )

        # The sparse CSV is fully digested into the event map; nothing reads self.ts, so drop columns.
        self._keep_ts_columns(set())

    def reload(self, ds_path: str) -> None:
        """Reload from a new CSV (lazily creating the loader), like other scheduled sources;
        a constant ``max_power_kW`` degrades to the pre-CSV fallback until the first push."""
        if self.loader is None:
            self.loader = CsvLoader(None, on_reload=self._run_post_load)
        super().reload(ds_path)

    def _resolve_limit(self, day_index: int) -> float:
        """Hold-last limit at a per-day index: the latest step-change at or before it,
        else the fallback. Used for look-ahead (forecast can't be resolved incrementally)."""
        limit = self.max_power_kW
        for idx, value in self._events:  # sorted ascending
            if idx <= day_index:
                limit = value
            else:
                break
        return limit

    def setup_spaces(self,
                    state_spaces: OrderedDict,
                    action_spaces: OrderedDict) -> tuple[OrderedDict, OrderedDict]:
        if "ctxt_operator_max_power_kW" not in state_spaces:
            state_spaces["ctxt_operator_max_power_kW"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        return state_spaces, action_spaces

    def update_state(self, states, info=None) -> None:
        """Hold the last step-change limit; reset to the fallback at the start of each
        simulated day so the sparse profile repeats daily (EVState-style event hold)."""
        day_index = self.iteration % self._steps_per_day
        if day_index == 0:
            self._held_limit_kW = self.max_power_kW
        event = self._event_lookup.get(day_index)
        if event is not None:
            self._held_limit_kW = event
        states["ctxt_operator_max_power_kW"][0] = np.float32(self._held_limit_kW)

    def forecast_keys(self) -> tuple[str, ...]:
        return ("ctxt_fc_operator_max_power_kW",)

    def forecast(self, selected_future_steps: list[int]) -> dict[str, list[float]]:
        # Future values follow the same per-day cyclic profile (hold-last over events).
        max_power_forecast = [self._resolve_limit((self.iteration + step) % self._steps_per_day)
                                for step in selected_future_steps]
        return {
            "ctxt_fc_operator_max_power_kW": max_power_forecast,
        }


# register with ComponentRegistry
ComponentRegistry.register('statesource', OperatorEnergyControl)
