import logging
from typing import ClassVar, Dict, Set

import numpy as np
from gymnasium.spaces import Box

from .base import Infrastructure
from adv_building_gym.utils.serializable import ComponentRegistry
from adv_building_gym.utils.seed_provider import RngService

logger = logging.getLogger(__name__)


class HouseholdEnergyConsumers(Infrastructure):
    """Passive household energy consumers infrastructure.

    Reads the normalized consumption signal from the DesiredUserEnergyNeed
    statesource (``desired_energy_need`` in states) and converts it to a
    physical kW consumption value — the same pattern SolarPanel uses with
    irradiance.

    There is no policy-controlled action. The actual consumption is written
    into ``actions['hh_consumption_action']`` as a read-only output so that
    reward functions can account for it.

    Action convention: positive = consumption from grid.
    hh_consumption_action value: 0 = no consumption, 1 = peak consumption.
    """

    # control_step comes from config context
    _context_params: ClassVar[Set[str]] = {'control_step'}

    # Internal state variables — don't serialize
    _exclude_params: ClassVar[Set[str]] = {
        'iteration', 'consumption_norm', 'current_consumption_kW'
    }

    def __init__(self,
                name: str,
                max_power_kW: float,
                peak_consumption_kW: float = 8.0,
                control_step: int = 300
                ) -> None:
        """Initialize household energy consumers infrastructure.

        Args:
            name: Component identifier
            max_power_kW: Maximum power consumption in kW (typically = peak_consumption_kW)
            peak_consumption_kW: Peak household consumption in kW
            control_step: Control timestep in seconds
        """
        super().__init__(name, max_power_kW)

        self.peak_consumption_kW = peak_consumption_kW
        self.control_step = control_step

        # State variables
        self.consumption_norm = 0.0  # Normalized consumption [0, 1]
        self.current_consumption_kW = 0.0  # Actual consumption in kW

    def setup_spaces(self,
                    state_spaces,
                    action_spaces):
        """Setup observation space for household consumption.

        Household consumers have no policy-controlled action space —
        consumption is determined by the DesiredUserEnergyNeed statesource.
        Only state space is registered.
        """
        if "hh_consumption_norm" not in state_spaces:
            state_spaces["hh_consumption_norm"] = Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            )

        return state_spaces, action_spaces

    def exec_action(self, actions: Dict, states: Dict, info=None) -> None:
        """Compute household consumption and write it into actions.

        Reads ``desired_energy_need`` from states (set by
        DesiredUserEnergyNeed statesource). Falls back to a synthetic
        time-of-day profile when no statesource signal is available.
        """
        # Read normalized consumption signal from statesource
        if "desired_energy_need" in states:
            self.consumption_norm = float(states["desired_energy_need"][0])
        else:
            # Fallback: synthetic time-based profile
            self.consumption_norm = self._synthetic_consumption(states)

        # Scale normalized signal to physical kW
        self.current_consumption_kW = self.consumption_norm * self.peak_consumption_kW

        # Write normalized consumption as read-only output (positive = consumption)
        if "hh_consumption_action" not in actions:
            actions["hh_consumption_action"] = np.array(
                [self.consumption_norm], dtype=np.float32
            )
        else:
            actions["hh_consumption_action"][0] = self.consumption_norm

    def update_state(self, states: Dict, info=None) -> None:
        """Write current normalized consumption into states for observation."""
        super().update_state(states, info)
        states["hh_consumption_norm"][0] = np.float32(self.consumption_norm)

    def _synthetic_consumption(self, states: Dict) -> float:
        """Generate synthetic consumption based on time of day.

        Simple stepped profile matching DesiredUserEnergyNeed's synthetic
        pattern, with added Gaussian noise for realism.
        """
        sim_hour = float(states.get("sim_hour", np.array([12.0]))[0]) % 24

        if sim_hour < 6:
            base = 0.2   # Low demand during night
        elif sim_hour < 9:
            base = 0.6   # Morning peak
        elif sim_hour < 17:
            base = 0.4   # Daytime moderate
        elif sim_hour < 21:
            base = 0.8   # Evening peak
        else:
            base = 0.3   # Late evening

        seed = RngService.get().get_random(self.name)
        noise = np.random.default_rng(seed).normal(loc=0.0, scale=0.05)
        return float(np.clip(base + noise, 0.0, 1.0))

    def get_electric_consumption(self, actions: Dict) -> float:
        """Get current electric energy consumption from household consumers.

        Sign convention: positive = consumption from grid.

        Returns:
            Positive value representing energy consumed from the grid (kW).
        """
        return self.current_consumption_kW


# Register HouseholdEnergyConsumers with the component registry
ComponentRegistry.register('infrastructure', HouseholdEnergyConsumers)
