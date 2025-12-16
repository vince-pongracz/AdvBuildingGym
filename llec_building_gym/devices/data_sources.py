
import logging
from collections import OrderedDict
from abc import ABC

import numpy as np
import pandas as pd

from gymnasium.spaces import Box, Dict as SDict

# or device --> general stuff for HP, battery, charger

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataSource(ABC):
    def __init__(self,
                 name: str,
                 # TODO VP 2025.12.03. : update to python newest (3.14 maybe?) But at least 3.11...
                 ds_path: str | None = None,
                 ) -> None:
        self.name = name
        self.price_max = None
        if ds_path is not None:
            self.ts = pd.read_csv(ds_path)
        else:
            self.ts = None

    def setup_spaces(self,
                     state_spaces,
                     action_spaces
                     ):
        # NOTE VP 2025.12.01. : implement in derived classes
        return state_spaces, action_spaces

    def update_state(self, states) -> None:
        pass

class EnergyPriceDataSource(DataSource):
    def setup_spaces(self,
                     state_spaces,
                     action_spaces
                     ) -> tuple:
        state_spaces["E_price"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        state_spaces["E_price_max"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        if "iteration" not in state_spaces.keys():
            state_spaces["iteration"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.int32)
        if "sim_hour" not in state_spaces.keys():
            logger.debug("sim_hour present")
            state_spaces["sim_hour"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.int32)

        return state_spaces, action_spaces

    def update_state(self, states) -> None:
        iteration = states.get("iteration", np.zeros(shape=(1,)))[0]

        if self.ts is not None:
            if iteration < len(self.ts):
                energy_price = self.ts.iloc[int(iteration)]["price_normalized"]
            else:
                energy_price = self.ts.iloc[-1]["price_normalized"]
            if self.price_max is None:
                self.price_max = float(self.ts["price_normalized"].max())
        else:
            current_sim_hour = states.get("sim_hour", np.zeros(shape=(1,)))[0]
            # Apply a simple time-of-use tariff if no CSV data is provided
            if current_sim_hour < 4:
                energy_price = 0.25
            elif current_sim_hour < 8:
                energy_price = 0.50
            else:
                energy_price = 0.75
            if self.price_max is None:
                self.price_max = 1.0

        states["E_price"][0] = energy_price
        states["E_price_max"][0] = self.price_max


class WeatherDataSource(DataSource):
    """WeatherDataSource

    Normalised weather data.
    temp_out_norm = 1 --> 50 C
    temp_out_norm = -1 --> -50 C
    Args:
        DataSource (_type_): Parent class
    """
    def __init__(self, name: str, ds_path: str | None = None) -> None:
        super().__init__(name, ds_path)
        self.temp_out_max = 50
        self.temp_out_min = -50
    
    def setup_spaces(self,
                     state_spaces: OrderedDict,
                     action_spaces: OrderedDict
                     ) -> tuple[OrderedDict, OrderedDict]:
        state_spaces["temp_out_norm"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        if "iteration" not in state_spaces.keys():
            state_spaces["iteration"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.int32)
        if "sim_hour" not in state_spaces.keys():
            state_spaces["sim_hour"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.int32)
        if "temp_diff" not in state_spaces.keys():
            state_spaces["temp_diff"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        return state_spaces, action_spaces

    def update_state(self, states) -> None:
        iteration = states.get("iteration", np.zeros(shape=(1,)))[0]

        if self.ts is not None:
            if iteration < len(self.ts):
                temp_out_norm = self.ts.iloc[int(iteration)]["temp_out_norm"]
            else:
                temp_out_norm = self.ts.iloc[-1]["temp_out_norm"]
        else:
            current_sim_hour = states.get("sim_hour", np.zeros(shape=(1,)))[0]
            # Apply a simple time-of-use tariff if no CSV data is provided
            if current_sim_hour < 5:
                temp_out_norm = 0
            elif current_sim_hour < 6:
                temp_out_norm = 0.3
            elif current_sim_hour < 8:
                temp_out_norm = 0.4
            elif current_sim_hour < 12:
                temp_out_norm = 0.45
            elif current_sim_hour < 16:
                temp_out_norm = 0.5
            elif current_sim_hour < 18:
                temp_out_norm = 0.35
            elif current_sim_hour < 21.5:
                temp_out_norm = 0.2
            elif current_sim_hour < 24:
                temp_out_norm = 0.1
            else:
                temp_out_norm = 0.3
        states["temp_out_norm"][0] = temp_out_norm
        states["temp_norm_out"][0] = temp_out_norm
        # Track simple difference between indoor and outdoor normalised temps
        states["temp_diff"][0] = states.get("temp_norm", np.zeros(1))[0] - temp_out_norm
