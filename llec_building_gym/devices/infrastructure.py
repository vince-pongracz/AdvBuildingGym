
from typing import Dict

import numpy as np

from gymnasium.spaces import Box, Dict as SDict

from llec_building_gym.utils import EnvSyncInterface

# or device --> general stuff for HP, battery, charger


class Infrastructure(EnvSyncInterface):
    def __init__(self,
                 name: str,
                 Q_electric_max: float
                 ) -> None:
        super().__init__()

        self.name = name
        self.Q_electric_max = Q_electric_max  # ~ power consumption max

    def setup_spaces(self,
                     state_spaces,
                     action_spaces
                     ):
        # NOTE VP 2025.12.01. : implement in derived classes
        return state_spaces, action_spaces

    def set_target(self, target: float) -> None:
        pass

    def exec_action(self, actions, states) -> None:
        """
        Action execution of the infrastructure.

        Args:
            actions (Dict): contains actions
            states (Dict): should be treated as immutable, holds information for the action execution)
        """
        pass

    def update_state(self, states:Dict) -> None:
        pass


class HP(Infrastructure):
    def __init__(self,
                 name: str,
                 Q_electric_max: float,
                 Q_hp_max: float,
                 K: float,
                 mC: float,
                 cop_heat: float = 1.0,
                 cop_cool: float = 1.0,
                 ) -> None:
        super().__init__(name, Q_electric_max)

        self.cop_heat = cop_heat  # [-] heating COP
        self.cop_cool = cop_cool  # [-] cooling COP
        self.Q_hp_max = Q_hp_max
        self.timestep = 0
        self.K = K
        self.mC = mC
        
        self.temp_norm = 0
        self.temp_norm_change = 0
        self.temp_norm_out = 0
        
        if self.cop_heat <= 0 or self.cop_cool <= 0:
            raise ValueError("cop_heat and cop_cool must be positive.")

    def setup_spaces(self,
                     state_spaces,
                     action_spaces
                     ):
        action_spaces["HP_action"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        if "temp_norm" not in state_spaces:
            state_spaces["temp_norm"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        if "temp_norm_out" not in state_spaces:
            state_spaces["temp_norm_out"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        return state_spaces, action_spaces

    def exec_action(self, actions, states) -> None:
        action = actions["HP_action"]
        q_hp = action * self.Q_hp_max

        dT = (0.001
              * (self.timestep / self.mC)
              * (self.K * (states["temp_norm"][0] - states["temp_norm_out"][0]) + q_hp))
        self.temp_norm -= dT
        self.temp_norm_change = dT

    def update_state(self, states) -> None:
        states["temp_norm"][0] += self.temp_norm_change


class HHConsumers(Infrastructure):
    def __init__(self, name: str, Q_electric_max: float) -> None:
        super().__init__(name, Q_electric_max)
    
    def setup_spaces(self,
                     state_spaces,
                     action_spaces
                     ):
        pass
    

class Battery(Infrastructure):
    def __init__(self, name: str,
                 Q_electric_max: float,
                 max_cap_kWh: float,
                 max_charging_amps: float,
                 start_percentage: float = 0.3,  # as a starting/default value
                 target_pct: float = 1.0,
                 history_length: int = 4
                 ) -> None:
        super().__init__(name, Q_electric_max)

        self.max_cap_kWh = max_cap_kWh
        self.max_charging_amps = max_charging_amps
        self.percentage = start_percentage
        self.target_pct = target_pct
        self.history_length = history_length

    # TODO VP 2025.12.02. : How to realise forecasting, based on the prev states and prev actions
    # ~history for n timesteps
    def setup_spaces(self,
                     state_spaces,
                     action_spaces
                     ):

        # TODO VP 2025.12.02. : With shape, action forecasting is also included? --> guess so
        action_spaces["battery_action"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        state_spaces["battery_pct"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        state_spaces["battery_target_pct"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        state_spaces["battery_pct_hist"] = Box(low=0, high=1, shape=(self.history_length,), dtype=np.float32)

        return state_spaces, action_spaces

    def set_target(self, target: float | None = None) -> None:
        self.target_pct = target

    def exec_action(self, actions: Dict, states: Dict) -> None:
        action = actions["battery_action"][0]
        # TODO VP 2025.12.02. : Improve battery model
        self.percentage += action

        # action guard
        if self.percentage > 1.0:
            self.percentage = 1
        if self.percentage < 0:
            self.percentage = 0


    def update_state(self, states: Dict) -> None:
        states["battery_pct"][0] = self.percentage

        history = states["battery_pct_hist"]
        # Shift all rows up (drop oldest)
        history[:-1] = history[1:]
        # Insert new state at the end
        history[-1] = self.percentage

