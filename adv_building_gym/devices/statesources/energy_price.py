import logging
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from .base import StateSource
from adv_building_gym.config.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)


class EnergyPriceDataSource(StateSource):
    """Data source for energy pricing information."""

    # price_max is derived from data, don't serialize
    _exclude_params: ClassVar[Set[str]] = {'iteration', 'ts', 'price_max'}

    def __init__(self, name: str, ds_path: str | None = None) -> None:
        super().__init__(name, ds_path)
        
        if self.ts is not None:
            logger.info("Use data file: %s", ds_path)
            self.price_max = float(self.ts["price_normalized"].max())
        else:
            self.price_max = 1.0
        
    def setup_spaces(self,
                     state_spaces,
                     action_spaces
                     ) -> tuple:

        if "E_price" not in state_spaces.keys():
            state_spaces["E_price"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        if "E_price_max" not in state_spaces.keys():
            state_spaces["E_price_max"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        if "sim_hour" not in state_spaces.keys():
            state_spaces["sim_hour"] = Box(low=np.full((1,), 0, dtype=np.float32),
                                           high=np.full((1,), np.inf, dtype=np.float32),
                                           shape=(1,),
                                           dtype=np.float32)

        return state_spaces, action_spaces

    def update_state(self, states) -> None:
        if self.ts is not None:
            if self.iteration < len(self.ts):
                energy_price = float(self.ts.iloc[int(self.iteration)]["price_normalized"])
            else:
                energy_price = float(self.ts.iloc[-1]["price_normalized"])
        else:
            current_sim_hour = states.get("sim_hour", np.zeros(shape=(1,), dtype=np.float32))[0]
            # Apply a simple time-of-use tariff if no CSV data is provided
            if current_sim_hour < 4:
                energy_price = 0.25
            elif current_sim_hour < 8:
                energy_price = 0.50
            else:
                energy_price = 0.75

        states["E_price"][0] = np.float32(energy_price)
        states["E_price_max"][0] = np.float32(self.price_max)


# Register EnergyPriceDataSource with the component registry
ComponentRegistry.register('statesource', EnergyPriceDataSource)
