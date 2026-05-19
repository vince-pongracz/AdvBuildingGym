import logging
from collections import OrderedDict
from typing import ClassVar, Optional, Set

import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from adv_building_gym.utils.constants import SECONDS_PER_HOUR

from ..base import StateSource
from ..csv_loader import CsvLoader
from ..forecastable import Forecastable
from adv_building_gym.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)

_SECONDS_PER_DAY: int = 24 * SECONDS_PER_HOUR


class OperatorEnergyControl(StateSource, Forecastable):
    """Grid-operator power limit, optionally driven by a sparse step-change CSV.

    Publishes ``ctxt_operator_max_power_kW`` (raw kW) every step. The reward
    function reads that ctxt key, so the limit has a single source of truth.

    Two modes
    ---------
    * **Constant** (no ``ds_path``): publishes ``max_power_kW`` every step.
    * **CSV-driven** (``ds_path`` given): loads a sparse profile where each
      row sets ``max_power_kW`` at its timestamp; the value is held until
      the next row (forward-fill). Timestamps map to per-day iteration
      indices via ``seconds_from_midnight // control_step`` (same scheme
      as ``EVState``), so the profile repeats every simulated day.
      ``max_power_kW`` is the fallback held before the first event of the
      day.

    CSV format
    ----------
    Columns: ``start, max_power_kW``. Example::

        start,max_power_kW
        2025-01-02 00:00:00,20.0
        2025-01-02 07:00:00,12.0
        2025-01-02 17:00:00,25.0
    """

    _context_params: ClassVar[Set[str]] = {"control_step"}
    _exclude_params: ClassVar[Set[str]] = {"_per_iter_kW", "_steps_per_day"}

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

        # Number of control steps that fit in one simulated day -- used to
        # wrap iteration indices into the per-day profile array.
        self._steps_per_day: int = int(_SECONDS_PER_DAY // int(self.control_step))
        if self._steps_per_day <= 0:
            raise ValueError(
                f"OperatorEnergyControl '{name}': control_step={control_step}s is too large "
                "to fit at least one step per day."
            )

        # Dense per-iteration kW array (length steps_per_day) when CSV-driven;
        # None in constant-limit mode.
        self._per_iter_kW: Optional[np.ndarray] = None

        if ds_path is not None:
            # Two-phase init avoids the chicken-and-egg where
            # ``CsvLoader.__init__(ds_path=...)`` would call back into
            # ``_post_load_data_processing`` before ``self.loader`` finishes
            # being assigned (``self.ts`` returns None during that window).
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
        """Forward-fill the sparse CSV into a per-iteration kW array."""
        df = self.ts
        if df is None or len(df) == 0:
            raise ValueError(f"OperatorEnergyControl '{self.name}': CSV {self.ds_path} is empty.")
        if "start" not in df.columns or "max_power_kW" not in df.columns:
            raise ValueError(
                f"OperatorEnergyControl '{self.name}': CSV {self.ds_path} must have columns 'start' and 'max_power_kW'."
            )

        df = df.copy()
        df["start"] = pd.to_datetime(df["start"])

        # Sort by time-of-day so the forward-fill respects intra-day ordering
        # even if the CSV mixes dates.
        df["iter_idx"] = (
            df["start"].dt.hour * SECONDS_PER_HOUR
            + df["start"].dt.minute * 60
            + df["start"].dt.second
        ) // int(self.control_step)
        df = df.sort_values("iter_idx").reset_index(drop=True)

        # NaN / non-positive values would silently propagate into the reward
        # (which divides by this), so reject them up-front.
        if df["max_power_kW"].isna().any() or (df["max_power_kW"] <= 0).any():
            raise ValueError(
                f"OperatorEnergyControl '{self.name}': max_power_kW must be > 0 in every "
                f"row of {self.ds_path}."
            )

        # Forward-fill into a dense array, seeded with the constructor
        # fallback so iterations before the first event keep the contract.
        per_iter = np.full(self._steps_per_day, self.max_power_kW, dtype=np.float64)
        for iter_idx, value in zip(df["iter_idx"].to_numpy(), df["max_power_kW"].to_numpy()):
            iter_idx = int(iter_idx)
            if 0 <= iter_idx < self._steps_per_day:
                per_iter[iter_idx:] = float(value)
        self._per_iter_kW = per_iter

        if self.is_new_data_source:
            logger.info(
                "OperatorEnergyControl '%s': loaded %d step-change(s) from %s "
                "(range %.3f-%.3f kW)",
                self.name, len(df), self.ds_path, per_iter.min(), per_iter.max(),
            )

    def reload(self, ds_path: str) -> None:
        """Reload from a new CSV, lazily creating the loader on first call.

        Mirrors every other scheduled source (weather, E_price, EVState):
        when the data scheduler pushes an ``operator_energy_control`` variant
        the source picks it up regardless of whether it was constructed with
        a static ``max_power_kW`` or an explicit ``ds_path``. The constant
        ``max_power_kW`` thus degrades to a pre-CSV fallback held until the
        first scheduler push.
        """
        if self.loader is None:
            self.loader = CsvLoader(None, on_reload=self._run_post_load)
        super().reload(ds_path)

    def _current_limit_kW(self, iteration: int) -> float:
        """Resolve the kW limit at a given env iteration (cyclic per day)."""
        if self._per_iter_kW is None:
            return self.max_power_kW
        return float(self._per_iter_kW[iteration % self._steps_per_day])

    def setup_spaces(self,
                    state_spaces: OrderedDict,
                    action_spaces: OrderedDict) -> tuple[OrderedDict, OrderedDict]:
        if "ctxt_operator_max_power_kW" not in state_spaces:
            state_spaces["ctxt_operator_max_power_kW"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        return state_spaces, action_spaces

    def update_state(self, states, info=None) -> None:
        states["ctxt_operator_max_power_kW"][0] = np.float32(self._current_limit_kW(self.iteration))

    def forecast_keys(self) -> tuple[str, ...]:
        return ("s_fc_operator_max_power_kW",)

    def forecast(self, selected_future_steps: list[int]) -> dict[str, list[float]]:
        # Future values follow the same per-day cyclic profile.
        return {
            "s_fc_operator_max_power_kW": [
                self._current_limit_kW(self.iteration + step) for step in selected_future_steps
            ],
        }


# Register OperatorEnergyControl with the component registry
ComponentRegistry.register('statesource', OperatorEnergyControl)
