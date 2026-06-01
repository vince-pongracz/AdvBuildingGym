import logging
from collections import OrderedDict
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from ..base import StateSource
from ..csv_loader import CsvLoader
from ..forecastable import Forecastable
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)

class InsideTemperature(StateSource, Forecastable):
    """Data source for desired inside temperature setpoint."""

    def __init__(self, name: str, ds_path: str | None = None) -> None:
        super().__init__(name=name)
        self.desired_temp_in_raw: float = 0.0  # Raw desired temperature (°C)
        # Cached temp_abs_max from the last update_state call — used by forecast()
        # since the forecast API has no access to the shared state dict.
        self._last_temp_abs_max: float = 60.0

        self.loader = CsvLoader(ds_path, on_reload=self._run_post_load)
        if ds_path is not None:
            logger.info("Use data file: %s", ds_path)

    def _post_load_data_processing(self) -> None:
        """Detect the raw temperature column after CSV load / reload.

        Normalisation is deferred to update_state() so it can use the
        same scale (temp_abs_max) as WeatherDataSource / temp_in_norm.
        """
        # Expected column: "desired_temp_in [°C]" or similar
        if "desired_temp_in [°C]" in self.ts.columns:
            self._raw_column = "desired_temp_in [°C]"
        elif "desired_temp_in" in self.ts.columns:
            self._raw_column = "desired_temp_in"
        else:
            raise ValueError(
                f"InsideTemperature '{self.name}': CSV '{self.ds_path}' has no "
                "'desired_temp_in [°C]' or 'desired_temp_in' column."
            )

    # NOTE VP 2026.03.24. : Choosing the inside_temperature profile should depend on the date -- or on user interaction, but this part comes later, keep it in the TODO comment
    def setup_spaces(self,
                    state_spaces: OrderedDict,
                    action_spaces: OrderedDict
                    ) -> tuple[OrderedDict, OrderedDict]:
        """Setup observation spaces for desired user temperature."""
        
        if "s_desired_temp_in_norm" not in state_spaces.keys():
            state_spaces["s_desired_temp_in_norm"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        return state_spaces, action_spaces

    def update_state(self, states, info=None) -> None:
        """Update desired temperature state based on current iteration.

        When CSV data is available the raw °C value is normalised at
        runtime using the same scale as temp_in_norm / temp_out_norm
        (ABS_MIN_MAX_SCALING with temp_abs_max from WeatherDataSource).
        This ensures the reward function sees comparable values.
        """
        if self.ts is None:
            raise RuntimeError(
                f"InsideTemperature '{self.name}': no CSV loaded. The DataCombinator "
                "must push a desired_temp_in variant before update_state is called."
            )

        # Shared temperature scale published by WeatherDataSource into the state dict.
        # Fallback 60 °C is a safe default when no weather data is loaded.
        temp_abs_max: float = float(states["ctxt_temp_abs_max"][0]) if "ctxt_temp_abs_max" in states else 60.0
        self._last_temp_abs_max = temp_abs_max if temp_abs_max != 0 else 60.0

        # Profile CSVs cover a single day (e.g. 288 rows at 5-min steps).
        # Index by time-of-day so the profile repeats daily regardless of
        # the actual simulation date or row_offset.
        arr = self._forecast_array_cache.get(self._raw_column)
        if arr is None:
            arr = self.ts[self._raw_column].to_numpy()
            self._forecast_array_cache[self._raw_column] = arr
        idx = self.iteration % arr.shape[0]
        raw_temp = float(arr[idx])
        self.desired_temp_in_raw = raw_temp
        # Normalise on the same scale as temp_out_norm / temp_in_norm
        desired_temp_in_norm = raw_temp / temp_abs_max if temp_abs_max != 0 else 0.0

        # Ensure float32 dtype and clip to bounds
        desired_temp_in_norm = np.float32(np.clip(desired_temp_in_norm, -1.0, 1.0))
        states["s_desired_temp_in_norm"][0] = desired_temp_in_norm

    def reset(self, states, info=None) -> None:
        """Populate initial desired temperature and seed temp_in_norm.

        At episode start the indoor temperature starts near the desired
        setpoint with a small random offset so the agent does not always
        begin in a perfectly comfortable state.
        """
        self.update_state(states, info)
        if "s_temp_in_norm" in states and "s_desired_temp_in_norm" in states:
            # ±2 °C variance in normalised space (temp_abs_max default 60 °C
            # ⇒ 2/60 ≈ 0.033 normalised units)
            temp_abs_max = float(states["ctxt_temp_abs_max"][0]) if "ctxt_temp_abs_max" in states else 60.0
            max_offset_norm = 2.0 / temp_abs_max if temp_abs_max != 0 else 0.0
            # Draw the offset from the env rng (published on the shared info
            # channel as "_rng"), so it shares the deterministic, per-worker
            # stream seeded by reset(seed=...). Fallback only for standalone use.
            rng = (info.get("_rng") if info else None) or np.random.default_rng()
            variance = rng.uniform(-max_offset_norm, max_offset_norm)
            
            states["s_temp_in_norm"][0] = np.float32(np.clip(
                states["s_desired_temp_in_norm"][0] + variance, -1.0, 1.0
            ))

    def forecast_keys(self) -> tuple[str, ...]:
        return ("s_fc_desired_temp_in_norm",)

    def forecast(self, selected_future_steps: list[int]) -> dict[str, list[float]]:
        # Profile is single-day and repeats — wrap-around index, not zero-fill,
        # so Forecastable._csv_forecast (which zero-fills) is not reused here.
        if self.ts is None:
            return {"s_fc_desired_temp_in_norm": [0.0] * len(selected_future_steps)}
        arr = self._forecast_array_cache.get(self._raw_column)
        if arr is None:
            arr = self.ts[self._raw_column].to_numpy()
            self._forecast_array_cache[self._raw_column] = arr
        scale = self._last_temp_abs_max if self._last_temp_abs_max != 0 else 60.0
        idxs = (np.asarray(selected_future_steps, dtype=np.int64) + self.iteration) % arr.shape[0]
        vals = np.clip(arr[idxs] / scale, -1.0, 1.0)
        return {"s_fc_desired_temp_in_norm": vals.tolist()}

    def get_raw_values(self) -> dict[str, float]:
        return {"raw_desired_temp_in": self.desired_temp_in_raw}


# Register InsideTemperature with the component registry
ComponentRegistry.register('statesource', InsideTemperature)
