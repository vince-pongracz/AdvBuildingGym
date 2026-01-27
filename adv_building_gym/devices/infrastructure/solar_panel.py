import logging
from typing import ClassVar, Dict, Optional, Set

import numpy as np
from gymnasium.spaces import Box

from .base import Infrastructure
from adv_building_gym.config.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)

# TODO VP 2026.01.20. : Get solar irradiation data -- at climate/weather data

class SolarPanel(Infrastructure):
    """Solar Panel (PV) infrastructure component.

    Action convention: positive = consumption (from grid), negative = production (to grid).
    Solar panels only produce energy, so action is in [-1, 0] where -1 = max production.

    Models a photovoltaic system that produces power based on solar irradiance.
    Supports optional curtailment control to limit power output.

    Irradiance can be provided via:
    - External state update (from a DataSource providing irradiance)
    - Synthetic time-based profile (default)
    """

    # control_step and seed come from config context
    _context_params: ClassVar[Set[str]] = {'control_step', 'seed'}

    # Internal state variables - don't serialize
    _exclude_params: ClassVar[Set[str]] = {
        'iteration', 'irradiance_norm', 'current_production_kW', 'curtailment_factor'
    }

    def __init__(self,
                 name: str,
                 Q_electric_max: float,
                 peak_power_kW: float,
                 seed: int,
                 control_step: int = 300,
                 ) -> None:
        """Initialize Solar Panel infrastructure.

        Args:
            name: Component identifier
            Q_electric_max: Maximum power production in kW (typically = peak_power_kW)
            peak_power_kW: Peak power output under standard test conditions (STC)
            seed: Random seed for reproducible noise generation
            control_step: Control timestep in seconds (stored for future use)
        """
        super().__init__(name, Q_electric_max)
        
        # TODO VP 2026.01.24. : add at the weather data the irradiance angles 
        # as well and the cloudiness factor -- 
        # if it's cloudy, the sun is not that strong to 
        # produce as much energy as without clouds

        # NOTE VP 2026.01.24. : Inverter efficiency is not considered, 
        # peak power means peak output power, produced by the solar panel
        self.peak_power_kW = peak_power_kW # -1.0 at actions means the peak power
        self.rng = np.random.default_rng(seed=seed)
        self.control_step = control_step

        # State variables
        self.irradiance_norm = 0.0  # Normalized irradiance [0, 1]
        self.current_production_kW = 0.0  # Actual power production after curtailment


    def setup_spaces(self,
                     state_spaces,
                     action_spaces
                     ):
        """Setup observation and action spaces for solar panel.

        Action convention: negative = production (energy to grid), positive = consumption (energy from grid).
        Solar panels only produce, so action is in [-1, 0].
        """

        # Action: production level [-1, 0]
        # Sign convention: negative = production (providing energy to grid)
        # 0 = no production (fully curtailed, no influence on grid)
        # -1 = full production (maximum energy provided to grid)
        if "solar_action" not in action_spaces.keys():
            action_spaces["solar_action"] = Box(
                low=-1, high=0, shape=(1,), dtype=np.float32
            )

        # States
        if "solar_irradiance" not in state_spaces.keys():
            # Normalized irradiance [0, 1]
            state_spaces["solar_irradiance"] = Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            )

        return state_spaces, action_spaces


    def exec_action(self, actions: Dict, states: Dict) -> None:
        """
        Get expected action and based on state, calculate production.
        
        It can easilly happen that the expected E amount can not be delivered from the solar panel.
        So solar action is rather just an expectation or need about E amounts.
        """

        # solar_action is rather about what is the expected E amount from the controller's
        # perspective --> update it every time with the real E amount, which is created by
        # the solar panel, which can be delivered
        expexted_action = float(np.atleast_1d(actions["solar_action"])[0])

        # Update irradiance from state if available (set by DataSource)
        if "solar_irradiance" in states:
            self.irradiance_norm = float(states["solar_irradiance"][0])

        # If no external irradiance, use synthetic time-based profile
        if self.irradiance_norm == 0.0 and "sim_hour" in states:
            self.irradiance_norm = self._synthetic_irradiance(states)

        # Calculate production
        # Production = irradiance * peak_power
        potential_production = self.irradiance_norm * self.peak_power_kW
        
        real_action: float = -1.0 * potential_production / self.peak_power_kW
        
        if real_action > expexted_action:
            # We could create more E --> give it to the system?
            # --> no, we do not need that much, but somehow signalise that we can deliver more power from the panel
            # --> 2d solar action -- 0 if it is the max, 1 if more could be delivered
            diff = real_action - expexted_action
            diff *= 0.001
            real_action += diff

        actions["solar_action"][0] = real_action

    def _synthetic_irradiance(self, states: Dict) -> float:
        """Generate synthetic irradiance based on time of day.

        Simple bell curve approximation of solar irradiance with Gaussian noise.
        Peak at solar noon (12:00), zero at night.
        """
        sim_hour = float(states.get("sim_hour", np.array([12.0]))[0])

        # Sunrise ~6:00, sunset ~18:00, peak at 12:00
        if sim_hour < 6 or sim_hour > 18:
            return 0.0

        # Cosine-based profile centered at noon
        # Maps 6-18 hours to 0-pi, with peak at pi/2 (noon)
        hour_fraction = (sim_hour - 6) / 12.0  # [0, 1] over daylight hours
        base_irradiance = np.sin(hour_fraction * np.pi)

        # Add Gaussian noise for realistic cloud cover variations
        noise = self.rng.normal(loc=0.0, scale=0.05)
        irradiance = base_irradiance + noise

        return float(np.clip(irradiance, 0.0, 1.0))

    def get_electric_consumption(self, actions: Dict) -> float:
        """Get current electric energy consumption (production) from solar panel.

        Sign convention: positive = consumption from grid, negative = production to grid.
        Solar panels produce energy, so this returns a negative value.

        Returns:
            Negative value representing energy provided to the building/grid (kW).
        """
        # Negative consumption = production to grid
        return -self.current_production_kW


# Register SolarPanel with the component registry
ComponentRegistry.register('infrastructure', SolarPanel)
