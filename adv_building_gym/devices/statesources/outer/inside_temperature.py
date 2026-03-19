import logging
from collections import OrderedDict
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from ..base import StateSource
from adv_building_gym.config.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)

# TODO VP 2026.01.07. : Looking for options, how can be a datasource dynamic during evaluation -- 
# user can set a new setpoint or a whole curve as a profile during runtime...
# 1st create profiles in .csv-s about random user set_targets -- use user set_targets programatically
# Implement similar user intervention logic as it is for the EV charger -- it is easier here as it's a single Celsius temperature value.

class InsideTemperature(StateSource):
    """Data source for desired inside temperature setpoint."""

    def __init__(self, name: str, ds_path: str | None = None) -> None:
        super().__init__(name, ds_path)
        self.desired_temp_in_raw: float = 0.0  # Raw desired temperature (°C)

        if self.ts is not None:
            logger.info("Use data file: %s", ds_path)
            self._post_load_data_processing()
        else:
            logger.debug("No initial data file for '%s', using synthetic temperature profile", name)

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
            logger.warning("No 'desired_temp_in' column found in CSV, will use synthetic data")
            self.ts = None
            return

    # TODO VP 2026.03.10. : Crete a time series for this -- for the 4 seasons
    def setup_spaces(self,
                    state_spaces: OrderedDict,
                    action_spaces: OrderedDict
                    ) -> tuple[OrderedDict, OrderedDict]:
        """Setup observation spaces for desired user temperature."""
        
        if "desired_temp_in_norm" not in state_spaces.keys():
            state_spaces["desired_temp_in_norm"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        if "sim_hour" not in state_spaces.keys():
            state_spaces["sim_hour"] = Box(low=np.full((1,), 0, dtype=np.float32),
                                            high=np.full((1,), np.inf, dtype=np.float32),
                                            shape=(1,),
                                            dtype=np.float32)

        return state_spaces, action_spaces

    def update_state(self, states, info=None) -> None:
        """Update desired temperature state based on current iteration.

        When CSV data is available the raw °C value is normalised at
        runtime using the same scale as temp_in_norm / temp_out_norm
        (MAX_ABS_SCALING with temp_abs_max from WeatherDataSource).
        This ensures the reward function sees comparable values.
        """
        # Shared temperature scale written by WeatherDataSource to info.
        # Fallback 40 °C covers typical European outdoor range.
        temp_abs_max: float = float((info or {}).get("_temp_abs_max", 40.0))

        if self.ts is not None:
            idx = min(self.effective_index, len(self.ts) - 1)
            row = self.ts.iloc[idx]
            raw_temp = float(row[self._raw_column])
            self.desired_temp_in_raw = raw_temp
            # Normalise on the same scale as temp_out_norm / temp_in_norm
            desired_temp_in_norm = raw_temp / temp_abs_max if temp_abs_max != 0 else 0.0
        else:
            # sim_hour is actual hour of day (0–24)
            sim_hour = float(states.get("sim_hour", np.zeros(shape=(1,), dtype=np.float32))[0])
            sim_hour = sim_hour % 24
            # Synthetic setpoint profile — values on the same normalised
            # scale as the synthetic weather temp_out_norm (0.0–0.5).
            # temp_in_norm starts at 0 and drifts toward temp_out_norm via
            # BuildingHeatLoss, so desired values should be in that range.
            if sim_hour < 6:          # Night
                desired_temp_in_norm = 0.15
            elif sim_hour < 8:        # Morning
                desired_temp_in_norm = 0.25
            elif sim_hour < 12:       # Mid-morning
                desired_temp_in_norm = 0.30
            elif sim_hour < 17:       # Afternoon
                desired_temp_in_norm = 0.35
            elif sim_hour < 22:       # Evening
                desired_temp_in_norm = 0.30
            elif sim_hour < 24:       # Late evening
                desired_temp_in_norm = 0.20
            else:
                desired_temp_in_norm = 0.25

        # Ensure float32 dtype and clip to bounds
        desired_temp_in_norm = np.float32(np.clip(desired_temp_in_norm, -1.0, 1.0))
        states["desired_temp_in_norm"][0] = desired_temp_in_norm


# Register InsideTemperature with the component registry
ComponentRegistry.register('statesource', InsideTemperature)
