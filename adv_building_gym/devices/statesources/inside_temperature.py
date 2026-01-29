import logging
from collections import OrderedDict
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from .base import StateSource
from adv_building_gym.config.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)

# TODO VP 2026.01.08. : Refactor temp reward based on this, consider this as well
# TODO VP 2026.01.07. : Research options, how can be a datasource dynamic during execution -- 
# user can set a new setpoint or a whole curve as a profile during runtime...
# 1st create profiles in .csv-s about random user set_targets -- use user set_targets programatically

class InsideTemperature(StateSource):
    """Data source for desired inside temperature setpoint."""

    def __init__(self, name: str, ds_path: str | None = None) -> None:
        super().__init__(name, ds_path)

        if self.ts is not None:
            logger.info("Use data file: %s", ds_path)
            # Normalize temperature setpoint if CSV data is provided
            # Expected column: "desired_temp_in [°C]" or similar
            if "desired_temp_in [°C]" in self.ts.columns:
                column_name = "desired_temp_in [°C]"
            elif "desired_temp_in" in self.ts.columns:
                column_name = "desired_temp_in"
            else:
                logger.warning("No 'desired_temp_in' column found in CSV, will use synthetic data")
                self.ts = None
                return

            # Normalize to [-1, 1] range (assuming typical range: 15-30°C)
            temp_min = self.ts[column_name].min()
            temp_max = self.ts[column_name].max()
            self.ts["desired_temp_in_norm"] = (self.ts[column_name] - temp_min) / (temp_max - temp_min) # scale onto [0, 1]
            self.ts["desired_temp_in_norm"] = 2 * self.ts["desired_temp_in_norm"] # scale to [0, 2]
            self.ts["desired_temp_in_norm"] = self.ts["desired_temp_in_norm"] - 1 # push to [-1, 1]
        else:
            logger.warning("No data file provided, will use synthetic data")

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

    def update_state(self, states) -> None:
        """Update desired temperature state based on current iteration."""
        if self.ts is not None:
            if self.iteration < len(self.ts):
                desired_temp_in_norm = float(self.ts.iloc[int(self.iteration)]["desired_temp_in_norm"])
            else:
                desired_temp_in_norm = float(self.ts.iloc[-1]["desired_temp_in_norm"])
        else:
            current_sim_hour = states.get("sim_hour", np.zeros(shape=(1,), dtype=np.float32))[0]
            # Apply a simple time-based temperature setpoint profile
            # Night: cooler (20°C ~ -0.33), Day: moderate (22°C ~ 0), Evening: warmer (23°C ~ 0.33)
            if current_sim_hour < 6:
                desired_temp_in_norm = -0.4  # Night: cooler setpoint
            elif current_sim_hour < 8:
                desired_temp_in_norm = -0.2  # Morning: warming up
            elif current_sim_hour < 12:
                desired_temp_in_norm = 0.0   # Mid-morning: comfortable
            elif current_sim_hour < 17:
                desired_temp_in_norm = 0.1   # Afternoon: slightly warmer
            elif current_sim_hour < 22:
                desired_temp_in_norm = 0.2   # Evening: warmer preference
            elif current_sim_hour < 24:
                desired_temp_in_norm = -0.2  # Late evening: cooling down
            else:
                desired_temp_in_norm = 0.0   # Default

        # Ensure float32 dtype and clip to bounds
        desired_temp_in_norm = np.float32(np.clip(desired_temp_in_norm, -1.0, 1.0))
        states["desired_temp_in_norm"][0] = desired_temp_in_norm


# Register InsideTemperature with the component registry
ComponentRegistry.register('statesource', InsideTemperature)
