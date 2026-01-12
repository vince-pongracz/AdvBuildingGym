import logging
from collections import OrderedDict

import numpy as np
from gymnasium.spaces import Box

# TODO VP 2026.01.07. : change scaling implementation, use sklearn classes and functions for that -- not really important, low priority task
from sklearn import preprocessing

from .base import DataSource

logger = logging.getLogger(__name__)

from enum import Enum


class Normalisation(Enum):
    """Normalisation types"""
    MAX_ABS_SCALING = "max_abs"
    MIN_MAX_SCALING = "min_max"
    STANDARDISATION = "std"
    

class WeatherDataSource(DataSource):
    """WeatherDataSource
    """

    def __init__(self, name: str, ds_path: str | None = None, 
                 normalise: Normalisation | None = Normalisation.MAX_ABS_SCALING) -> None:
        super().__init__(name, ds_path)
        
        if self.ts is not None:
            logger.info("Use data file: %s", ds_path)
            column_name: str = "temp_amb [°C]"
            match normalise:
                case Normalisation.MAX_ABS_SCALING:
                    self.ts["temp_norm_out"] = self.ts[column_name] / self.ts[column_name].abs().max()
                case Normalisation.MIN_MAX_SCALING:
                    self.ts["temp_norm_out"] = (self.ts[column_name] - self.ts[column_name].min()) / (self.ts[column_name].max() - self.ts[column_name].min())
                case Normalisation.STANDARDISATION:
                    self.ts["temp_norm_out"] = (self.ts[column_name] - self.ts[column_name].mean()) / self.ts[column_name].std()
                case None:
                    self.ts["temp_norm_out"] = self.ts[column_name]
                    


    def setup_spaces(self,
                     state_spaces: OrderedDict,
                     action_spaces: OrderedDict
                     ) -> tuple[OrderedDict, OrderedDict]:
        state_spaces["temp_norm_out"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        if "sim_hour" not in state_spaces.keys():
            state_spaces["sim_hour"] = Box(low=np.full((1,), 0, dtype=np.float32),
                                            high=np.full((1,), np.inf, dtype=np.float32),
                                            shape=(1,),
                                            dtype=np.float32)
        if "temp_diff" not in state_spaces.keys():
            state_spaces["temp_diff"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        return state_spaces, action_spaces

    def update_state(self, states) -> None:
        if self.ts is not None:
            if self.iteration < len(self.ts):
                temp_norm_out = float(self.ts.iloc[int(self.iteration)]["temp_norm_out"])
            else:
                temp_norm_out = float(self.ts.iloc[-1]["temp_norm_out"])
        else:
            current_sim_hour = states.get("sim_hour", np.zeros(shape=(1,), dtype=np.float32))[0]
            # Apply a simple time-based temperature profile if no CSV data is provided
            if current_sim_hour < 5:
                temp_norm_out = 0.0
            elif current_sim_hour < 6:
                temp_norm_out = 0.3
            elif current_sim_hour < 8:
                temp_norm_out = 0.4
            elif current_sim_hour < 12:
                temp_norm_out = 0.45
            elif current_sim_hour < 16:
                temp_norm_out = 0.5
            elif current_sim_hour < 18:
                temp_norm_out = 0.35
            elif current_sim_hour < 21.5:
                temp_norm_out = 0.2
            elif current_sim_hour < 24:
                temp_norm_out = 0.1
            else:
                temp_norm_out = 0.3

        # Ensure float32 dtype for all updates
        temp_norm_out = np.float32(temp_norm_out)
        states["temp_norm_out"][0] = temp_norm_out

        # Track simple difference between indoor and outdoor normalised temps
        temp_norm_value = states.get("temp_norm_in", np.zeros(1, dtype=np.float32))[0]
        temp_diff = np.float32(np.clip(temp_norm_value - temp_norm_out, -1.0, 1.0))
        states["temp_diff"][0] = temp_diff
