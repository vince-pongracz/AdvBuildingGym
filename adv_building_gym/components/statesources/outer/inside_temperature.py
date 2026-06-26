import logging
from collections import OrderedDict
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from ..base import StateSource
from ..csv_loader import CsvLoader
from ..forecastable import Forecastable
from ..csv_lookahead import CsvLookahead
from ..reloadable import CsvReloadable
from adv_building_gym.components.registry import ComponentRegistry
from adv_building_gym._common.constants import TEMP_ABS_MAX_CELSIUS

logger = logging.getLogger(__name__)

class InsideTemperature(StateSource, Forecastable, CsvLookahead, CsvReloadable):
    """Data source for desired inside temperature setpoint."""

    def __init__(self, name: str, ds_path: str | None = None) -> None:
        super().__init__(name=name)
        self.desired_temp_in_raw: float = 0.0  # Raw desired temperature (°C)
        # cached temp_abs_max from last update_state — forecast() has no info channel access
        self._last_temp_abs_max: float = TEMP_ABS_MAX_CELSIUS

        self.loader = CsvLoader(ds_path, on_reload=self._run_post_load)
        if ds_path is not None:
            logger.info("Use data file: %s", ds_path)

    def _post_load_data_processing(self) -> None:
        """Detect the raw temperature column; normalisation is deferred to update_state
        (to reuse temp_abs_max from WeatherDataSource)."""
        # expected column: "desired_temp_in" or "desired_temp_in"
        if "desired_temp_in" in self.ts.columns:
            self._raw_column = "desired_temp_in"
        else:
            raise ValueError(f"InsideTemperature '{self.name}': CSV '{self.ds_path}' has no 'desired_temp_in' column.")

        # Only the raw setpoint column is read.
        # Normalisation is deferred to update_state / forecast)
        # Drop other CSV columns.
        self._keep_ts_columns({self._raw_column})

    # NOTE VP 2026.03.24. : Choosing the inside_temperature profile should depend on the date -- or on user interaction, but this part comes later, keep it in the TODO comment
    def setup_spaces(self,
                    state_spaces: OrderedDict,
                    action_spaces: OrderedDict
                    ) -> tuple[OrderedDict, OrderedDict]:
        """Setup observation space: the comfort error (indoor temp − setpoint).

        Collapses the former {s_temp_in_norm, s_desired_temp_in_norm} pair into a single
        normalised error in [-1, 1] (the control-relevant signal). The absolute indoor
        temperature is the integration variable on the info channel; the absolute setpoint
        is not exposed (the policy sees only its deviation from it).
        """
        if "s_temp_error_norm" not in state_spaces.keys():
            state_spaces["s_temp_error_norm"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        return state_spaces, action_spaces

    def _desired_temp_in_norm(self, states, info=None) -> float:
        """Setpoint for the current iteration, normalised by temp_abs_max (same scale as
        the indoor/outdoor temperatures). Caches temp_abs_max for forecast()."""
        if self.ts is None:
            raise RuntimeError(
                f"InsideTemperature '{self.name}': no CSV loaded. The DataCombinator "
                "must push a desired_temp_in variant before update_state is called."
            )
        # Fixed temperature normalisation scale from the info channel (WeatherDataSource).
        temp_abs_max: float = float(info["temp_abs_max"]) if info is not None and "temp_abs_max" in info else TEMP_ABS_MAX_CELSIUS
        self._last_temp_abs_max = temp_abs_max if temp_abs_max != 0 else TEMP_ABS_MAX_CELSIUS

        # single-day profile: index by time-of-day so it repeats daily (ignores row_offset)
        arr = self._forecast_array_cache.get(self._raw_column)
        if arr is None:
            arr = self.ts[self._raw_column].to_numpy()
            self._forecast_array_cache[self._raw_column] = arr
        idx = self.iteration % arr.shape[0]
        raw_temp = float(arr[idx])
        self.desired_temp_in_raw = raw_temp
        desired_temp_in_norm = raw_temp / temp_abs_max if temp_abs_max != 0 else 0.0
        return float(np.clip(desired_temp_in_norm, -1.0, 1.0))

    def update_state(self, states, info=None) -> None:
        """Publish the comfort error (indoor temp − setpoint), normalised in [-1, 1].

        Indoor temperature is read from the shared info channel (info["temp_in_norm"],
        owned by HP + BuildingHeatLoss). As an exogenous source this runs last in the
        step, so it sees the resulting indoor temperature (s')."""
        desired_temp_in_norm = self._desired_temp_in_norm(states, info)
        temp_in_norm = float(info.get("temp_in_norm", 0.0)) if info is not None else 0.0
        error_norm = float(np.clip(temp_in_norm - desired_temp_in_norm, -1.0, 1.0))
        states["s_temp_error_norm"][0] = np.float32(error_norm)

    def reset(self, states, info=None) -> None:
        """Seed the indoor temperature near the setpoint (±2 °C) on the info channel and
        publish the initial comfort error."""
        desired_temp_in_norm = self._desired_temp_in_norm(states, info)
        # ±2 °C in normalised space (2/70 ≈ 0.029 at the fixed 70 °C scale)
        temp_abs_max = self._last_temp_abs_max
        max_offset_norm = 2.0 / temp_abs_max if temp_abs_max != 0 else 0.0
        # offset from env rng (info["_rng"], deterministic per-worker; standalone fallback)
        rng = (info.get("_rng") if info else None) or np.random.default_rng()
        variance = rng.uniform(-max_offset_norm, max_offset_norm)
        temp_in_seed = float(np.clip(desired_temp_in_norm + variance, -1.0, 1.0))
        if info is not None:
            info["temp_in_norm"] = temp_in_seed

        error_norm = float(np.clip(temp_in_seed - desired_temp_in_norm, -1.0, 1.0))
        states["s_temp_error_norm"][0] = np.float32(error_norm)

    def forecast_keys(self) -> tuple[str, ...]:
        return ("s_fc_desired_temp_in_norm",)

    def forecast(self, selected_future_steps: list[int]) -> dict[str, list[float]]:
        # single-day profile: wrap-around index (not zero-fill), so _csv_forecast isn't reused
        if self.ts is None:
            return {"s_fc_desired_temp_in_norm": [0.0] * len(selected_future_steps)}
        arr = self._forecast_array_cache.get(self._raw_column)
        if arr is None:
            arr = self.ts[self._raw_column].to_numpy()
            self._forecast_array_cache[self._raw_column] = arr
        scale = self._last_temp_abs_max if self._last_temp_abs_max != 0 else TEMP_ABS_MAX_CELSIUS
        idxs = (np.asarray(selected_future_steps, dtype=np.int64) + self.iteration) % arr.shape[0]
        vals = np.clip(arr[idxs] / scale, -1.0, 1.0)
        return {"s_fc_desired_temp_in_norm": vals.tolist()}

    def get_raw_values(self) -> dict[str, float]:
        return {"raw_desired_temp_in": self.desired_temp_in_raw}


# register with ComponentRegistry
ComponentRegistry.register('statesource', InsideTemperature)
