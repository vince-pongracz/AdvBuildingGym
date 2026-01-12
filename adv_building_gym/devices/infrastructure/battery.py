import logging
from typing import Dict, Optional

import numpy as np
from gymnasium.spaces import Box

from .base import Infrastructure

logger = logging.getLogger(__name__)


class Battery(Infrastructure):
    """Battery infrastructure component."""

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

    def setup_spaces(self,
                     state_spaces,
                     action_spaces
                     ):
        action_spaces["battery_action"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        state_spaces["battery_pct"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        state_spaces["battery_target_pct"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        state_spaces["battery_pct_hist"] = Box(low=0, high=1, shape=(self.history_length,), dtype=np.float32)

        return state_spaces, action_spaces

    def set_target(self, target: Optional[float] = None) -> None:
        self.target_pct = target

    def exec_action(self, actions: Dict, states: Dict) -> None:
        action = float(np.atleast_1d(actions["battery_action"])[0])
        # TODO VP 2025.12.02. : Improve battery model
        self.percentage += action

        # Clip percentage to valid range [0, 1]
        self.percentage = float(np.clip(self.percentage, 0.0, 1.0))

    def update_state(self, states: Dict) -> None:
        # Ensure float32 dtype for all updates
        states["battery_pct"][0] = np.float32(self.percentage)
        states["battery_target_pct"][0] = np.float32(self.target_pct)

        history = states["battery_pct_hist"]
        # Shift all rows up (drop oldest)
        history[:-1] = history[1:]
        # Insert new state at the end
        history[-1] = np.float32(self.percentage)
