import logging
from typing import ClassVar, Dict, Set

import numpy as np
from gymnasium.spaces import Box

from adv_building_gym.utils.constants import SECONDS_PER_HOUR
from ..base import Infrastructure
from adv_building_gym.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)


class BatteryLinear(Infrastructure):
    """Battery infrastructure using a simple linear model.

    This model provides a straightforward battery simulation where action
    directly translates to charge/discharge power. No voltage variations
    or efficiency losses are modeled - it's an ideal battery.

    Power-based action:
        - action in [-1, 1] maps to [-max_power_kW, max_power_kW] kW
        - Positive action: charge battery (consume power from grid)
        - Negative action: discharge battery (provide power to grid)

    Energy change per timestep:
        delta_E (kWh) = action * max_power_kW (kW) * control_step (s) / 3600
        delta_SoC = delta_E / max_cap_kWh

    NOTE on healthy-band semantics:
        Reward-related concepts (the (min_pct, max_pct) operating band, any
        target SoC setpoint) live on the reward side, not on the battery.
        BatteryTargetReward owns ``min_pct`` / ``max_pct`` as constructor
        arguments and reads ``s_battery_pct`` from the obs.  An alternative
        we considered (Option B) was to drive the band from a CSV schedule —
        each row gives a (min, max) pair, allowing the band to vary over
        time (e.g. wider during the day, narrower overnight).  We chose the
        static form for simplicity; if a time-varying band is ever wanted,
        plumb a small data source that publishes the two values and have
        the reward (or a connector) pull them per step.
    """

    POWER_FLOW = "bidirectional"

    _context_params: ClassVar[Set[str]] = {'control_step'}

    _exclude_params: ClassVar[Set[str]] = {
        'iteration', 'soc', 'actual_power_kW'
    }

    def __init__(self, name: str,
                max_power_kW: float,
                max_cap_kWh: float,
                control_step: int,
                start_soc_percentage: float,
                history_length: int,
                soc_min: float,
                soc_max: float,
                ) -> None:
        """Initialize linear battery model.

        Args:
            name: Component identifier
            max_power_kW: Maximum charge/discharge power in kW
            max_cap_kWh: Battery capacity in kWh
            control_step: Timestep duration in seconds
            start_soc_percentage: Initial state of charge [0, 1]
            history_length: Number of past SoC values to track
            soc_min: Hardware minimum SoC (clipping floor)
            soc_max: Hardware maximum SoC (clipping ceiling)
        """
        super().__init__(name, max_power_kW)

        self.max_cap_kWh = max_cap_kWh
        self.start_soc_percentage = start_soc_percentage
        self.soc = start_soc_percentage
        self.control_step = control_step
        self.history_length = history_length
        self.soc_min = soc_min
        self.soc_max = soc_max

        self.actual_power_kW = 0.0

    def setup_spaces(self,
                    state_spaces,
                    action_spaces):
        if "a_battery" not in action_spaces.keys():
            action_spaces["a_battery"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        if "s_battery_pct" not in state_spaces.keys():
            state_spaces["s_battery_pct"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        # Policy-side history of s_battery_pct is assembled by
        # StridedHistoryConnector on the rollout/learner side; the env no
        # longer stores it in the observation dict.

        # Raw battery capacity (kWh) — constant hardware parameter.
        if "ctxt_battery_capacity_kWh" not in state_spaces.keys():
            state_spaces["ctxt_battery_capacity_kWh"] = Box(
                low=0, high=np.inf, shape=(1,), dtype=np.float32
            )

        return state_spaces, action_spaces

    def exec_action(self, actions: Dict, states: Dict, info=None) -> None:
        """Execute battery charge/discharge action using linear model.

        Action in [-1, 1]:
            - Positive: charge battery (consume power from grid)
            - Negative: discharge battery (provide power to grid)

        The action represents fraction of max power (max_power_kW).
        """
        action = float(np.atleast_1d(actions["a_battery"])[0])

        # Calculate requested power in kW
        requested_power_kW = action * self.max_power_kW

        # Calculate energy change in this timestep
        # E (kWh) = P (kW) * t (h)
        time_hours = self.control_step / SECONDS_PER_HOUR
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
        actual_action = self.actual_power_kW / self.max_power_kW if self.max_power_kW > 0 else 0.0
        actions["a_battery"] = np.array([np.float32(actual_action)], dtype=np.float32)

    def update_state(self, states: Dict, info=None) -> None:
        super().update_state(states, info)
        states["s_battery_pct"][0] = np.float32(self.soc)
        states["ctxt_battery_capacity_kWh"][0] = np.float32(self.max_cap_kWh)

    def reset(self, states: Dict, info=None) -> None:
        """Re-initialise transient state at the start of every episode.

        The base implementation only re-emits update_state(), which would
        leave self.soc carrying over from the previous episode.
        """
        self.soc = self.start_soc_percentage
        self.actual_power_kW = 0.0
        super().reset(states, info)

    def get_electric_consumption(self, actions: Dict) -> float:
        """Get current electric energy consumption from battery in kW.

        Returns:
            Positive value when charging (consuming from grid),
            negative value when discharging (providing to grid).
        """
        return self.actual_power_kW


ComponentRegistry.register('infrastructure', BatteryLinear)
