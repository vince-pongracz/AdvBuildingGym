import logging
from collections import OrderedDict
from enum import Enum
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

# TODO VP 2026.01.07. : change scaling implementation, use sklearn classes and functions for that -- not really important, low priority task
from sklearn import preprocessing

from .base import StateSource
from adv_building_gym.config.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)


class Normalisation(Enum):
    """Normalisation types"""
    MAX_ABS_SCALING = "max_abs"
    MIN_MAX_SCALING = "min_max"
    STANDARDISATION = "std"


class WeatherDataSource(StateSource):
    """WeatherDataSource"""

    # normalise is an enum, need special handling for serialization
    _exclude_params: ClassVar[Set[str]] = {'iteration', 'ts'}

    def __init__(self, name: str, ds_path: str | None = None,
                 normalise: Normalisation | str | None = Normalisation.MAX_ABS_SCALING) -> None:
        super().__init__(name, ds_path)

        # Convert string to enum if needed (for deserialization)
        if isinstance(normalise, str):
            normalise = Normalisation(normalise)
        self.normalise = normalise  # Store for serialization

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
        else:
            logger.info("No data file provided, will use synthetic data")


    def setup_spaces(self,
                     state_spaces: OrderedDict,
                     action_spaces: OrderedDict
                     ) -> tuple[OrderedDict, OrderedDict]:
        if "temp_norm_out" not in state_spaces.keys():
            state_spaces["temp_norm_out"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        if "sim_hour" not in state_spaces.keys():
            state_spaces["sim_hour"] = Box(low=np.full((1,), 0, dtype=np.float32),
                                            high=np.full((1,), np.inf, dtype=np.float32),
                                            shape=(1,),
                                            dtype=np.float32)

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

    def _get_serialize_value(self, param_name: str, value):
        """Handle enum serialization for normalise parameter."""
        if param_name == 'normalise' and isinstance(value, Normalisation):
            return value.value  # Serialize as string
        return super()._get_serialize_value(param_name, value)


# Register WeatherDataSource with the component registry
ComponentRegistry.register('statesource', WeatherDataSource)