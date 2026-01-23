import logging
from typing import ClassVar, Dict, Optional, Set

import numpy as np
from gymnasium.spaces import Box

from ..base import Infrastructure
from adv_building_gym.config.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)


class BatteryLinear(Infrastructure):
    """Battery infrastructure using a simple linear model.

    This model provides a straightforward battery simulation where action
    directly translates to charge/discharge power. No voltage variations
    or efficiency losses are modeled - it's an ideal battery.

    Power-based action:
        - action in [-1, 1] maps to [-Q_electric_max, Q_electric_max] kW
        - Positive action: charge battery (consume power from grid)
        - Negative action: discharge battery (provide power to grid)

    Energy change per timestep:
        delta_E (kWh) = action * Q_electric_max (kW) * control_step (s) / 3600
        delta_SoC = delta_E / max_cap_kWh
    """

    _context_params: ClassVar[Set[str]] = {'control_step'}

    _exclude_params: ClassVar[Set[str]] = {
        'iteration', 'soc', 'actual_power_kW'
    }

    def __init__(self, name: str,
                 Q_electric_max: float,
                 max_cap_kWh: float,
                 start_soc_percentage: float = 0.3,
                 target_soc: float = 0.95,
                 control_step: int = 300,
                 history_length: int = 4,
                 soc_min: float = 0.1,
                 soc_max: float = 0.95,
                 ) -> None:
        """Initialize linear battery model.

        Args:
            name: Component identifier
            Q_electric_max: Maximum charge/discharge power in kW
            max_cap_kWh: Battery capacity in kWh
            start_soc_percentage: Initial state of charge [0, 1]
            target_soc: Target state of charge [0, 1]
            control_step: Timestep duration in seconds
            history_length: Number of past SoC values to track
            soc_min: Minimum allowed SoC to prevent damage
            soc_max: Maximum allowed SoC to prevent damage
        """
        super().__init__(name, Q_electric_max)

        self.max_cap_kWh = max_cap_kWh
        self.soc = start_soc_percentage
        self.start_percentage = start_soc_percentage
        self.target_soc = target_soc
        self.control_step = control_step
        self.history_length = history_length
        self.soc_min = soc_min
        self.soc_max = soc_max

        self.actual_power_kW = 0.0

    def setup_spaces(self,
                     state_spaces,
                     action_spaces
                     ):
        if "battery_action" not in action_spaces.keys():
            action_spaces["battery_action"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        if "battery_pct" not in state_spaces.keys():
            state_spaces["battery_pct"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        if "battery_target_pct" not in state_spaces.keys():
            state_spaces["battery_target_pct"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        if "battery_pct_hist" not in state_spaces.keys():
            state_spaces["battery_pct_hist"] = Box(low=0, high=1, shape=(self.history_length,), dtype=np.float32)

        return state_spaces, action_spaces

    def set_target(self, target: Optional[float] = None) -> None:
        self.target_soc = target

    def exec_action(self, actions: Dict, states: Dict) -> None:
        """Execute battery charge/discharge action using linear model.

        Action in [-1, 1]:
            - Positive: charge battery (consume power from grid)
            - Negative: discharge battery (provide power to grid)

        The action represents fraction of max power (Q_electric_max).
        """
        action = float(np.atleast_1d(actions["battery_action"])[0])

        # Calculate requested power in kW
        requested_power_kW = action * self.Q_electric_max

        # Calculate energy change in this timestep
        # E (kWh) = P (kW) * t (h)
        time_hours = self.control_step / 3600.0
        delta_energy_kWh = requested_power_kW * time_hours

        # Convert energy to SoC change
        delta_soc = delta_energy_kWh / self.max_cap_kWh if self.max_cap_kWh > 0 else 0.0

        # Apply SoC change and clip to valid range
        old_soc = self.soc
        new_soc = self.soc + delta_soc
        self.soc = float(np.clip(new_soc, self.soc_min, self.soc_max))

        # Calculate actual energy transferred (may be limited by SoC bounds)
        actual_delta_soc = self.soc - old_soc
        actual_energy_kWh = actual_delta_soc * self.max_cap_kWh

        # Calculate actual power for consumption reporting
        self.actual_power_kW = actual_energy_kWh / time_hours if time_hours > 0 else 0.0

        # Update the action dict to reflect actual (clipped) action
        actual_action = self.actual_power_kW / self.Q_electric_max if self.Q_electric_max > 0 else 0.0
        actions["battery_action"] = np.array([np.float32(actual_action)], dtype=np.float32)

    def update_state(self, states: Dict) -> None:
        states["battery_pct"][0] = np.float32(self.soc)
        states["battery_target_pct"][0] = np.float32(self.target_soc)

        history = states["battery_pct_hist"]
        history[:-1] = history[1:]
        history[-1] = np.float32(self.soc)

    def get_electric_consumption(self, actions: Dict) -> float:
        """Get current electric energy consumption from battery in kW.

        Returns:
            Positive value when charging (consuming from grid),
            negative value when discharging (providing to grid).
        """
        return self.actual_power_kW


ComponentRegistry.register('infrastructure', BatteryLinear)
