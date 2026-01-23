import logging
from typing import ClassVar, Dict, Optional, Set

import numpy as np
from gymnasium.spaces import Box

from .base import Infrastructure
from adv_building_gym.config.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)

# TODO VP 2026.01.20. : Get solar irradiation data!
# TODO VP 2026.01.23. : Review the full class, how is the action exec

class SolarPanel(Infrastructure):
    """Solar Panel (PV) infrastructure component.

    Models a photovoltaic system that produces power based on solar irradiance.
    Supports optional curtailment control to limit power output.

    Irradiance can be provided via:
    - External state update (from a DataSource providing irradiance)
    - Synthetic time-based profile (default)
    """

    # control_step comes from config context
    _context_params: ClassVar[Set[str]] = {'control_step'}

    # Internal state variables - don't serialize
    _exclude_params: ClassVar[Set[str]] = {
        'iteration', 'irradiance_norm', 'current_production_kW', 'curtailment_factor'
    }

    def __init__(self,
                 name: str,
                 Q_electric_max: float,
                 peak_power_kW: float,
                 panel_efficiency: float = 0.20,
                 inverter_efficiency: float = 0.96,
                 curtailment_enabled: bool = False,
                 control_step: int = 300,
                 ) -> None:
        """Initialize Solar Panel infrastructure.

        Args:
            name: Component identifier
            Q_electric_max: Maximum power production in kW (typically = peak_power_kW)
            peak_power_kW: Peak power output under standard test conditions (STC)
            panel_efficiency: Panel conversion efficiency [0, 1]
            inverter_efficiency: Inverter efficiency [0, 1]
            curtailment_enabled: Whether curtailment action is available
            control_step: Control timestep in seconds (stored for future use)
        """
        super().__init__(name, Q_electric_max)

        self.peak_power_kW = peak_power_kW
        self.panel_efficiency = panel_efficiency
        self.inverter_efficiency = inverter_efficiency
        self.curtailment_enabled = curtailment_enabled
        self.control_step = control_step

        # State variables
        self.irradiance_norm = 0.0  # Normalized irradiance [0, 1]
        self.current_production_kW = 0.0  # Actual power production after curtailment
        self.curtailment_factor = 1.0  # Applied curtailment [0, 1]

        if panel_efficiency <= 0 or panel_efficiency > 1:
            raise ValueError("panel_efficiency must be in (0, 1].")
        if inverter_efficiency <= 0 or inverter_efficiency > 1:
            raise ValueError("inverter_efficiency must be in (0, 1].")

    def setup_spaces(self,
                     state_spaces,
                     action_spaces
                     ):
        """Setup observation and action spaces for solar panel."""

        # Action: curtailment factor [0, 1]
        # 0 = fully curtailed (no production, no influence on the grid), 
        # -1 = full production (grid gets E)
        # Always create action space for consistency, even if curtailment is disabled
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

        if "solar_production_norm" not in state_spaces.keys():
            # Normalized production relative to peak [0, 1]
            state_spaces["solar_production_norm"] = Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            )

        return state_spaces, action_spaces

    def set_irradiance(self, irradiance_norm: float) -> None:
        """Set normalized irradiance externally (e.g., from DataSource)."""
        self.irradiance_norm = float(np.clip(irradiance_norm, 0.0, 1.0))

    def exec_action(self, actions: Dict, states: Dict) -> None:
        """Execute curtailment action and calculate production."""
        # Get curtailment factor from action or default to 1.0 (no curtailment)
        if self.curtailment_enabled and "solar_action" in actions:
            self.curtailment_factor = float(
                np.atleast_1d(actions["solar_action"])[0]
            )
            self.curtailment_factor = self.curtailment_factor * (-1.0)
            self.curtailment_factor = float(np.clip(self.curtailment_factor, 0.0, 1.0))
        else:
            self.curtailment_factor = 1.0

        # Update irradiance from state if available (set by DataSource)
        if "solar_irradiance" in states:
            self.irradiance_norm = float(states["solar_irradiance"][0])

        # If no external irradiance, use synthetic time-based profile
        if self.irradiance_norm == 0.0 and "sim_hour" in states:
            self.irradiance_norm = self._synthetic_irradiance(states)

        # Calculate production
        # Production = irradiance * peak_power * inverter_efficiency * curtailment
        potential_production = (
            self.irradiance_norm *
            self.peak_power_kW *
            self.inverter_efficiency
        )
        self.current_production_kW = potential_production * self.curtailment_factor

    def _synthetic_irradiance(self, states: Dict) -> float:
        """Generate synthetic irradiance based on time of day.

        Simple bell curve approximation of solar irradiance.
        Peak at solar noon (12:00), zero at night.
        """
        sim_hour = float(states.get("sim_hour", np.array([12.0]))[0])

        # Sunrise ~6:00, sunset ~18:00, peak at 12:00
        if sim_hour < 6 or sim_hour > 18:
            return 0.0

        # Cosine-based profile centered at noon
        # Maps 6-18 hours to 0-pi, with peak at pi/2 (noon)
        hour_fraction = (sim_hour - 6) / 12.0  # [0, 1] over daylight hours
        irradiance = np.sin(hour_fraction * np.pi)

        return float(np.clip(irradiance, 0.0, 1.0))

    def update_state(self, states: Dict) -> None:
        """Update observable state."""
        states["solar_irradiance"][0] = np.float32(self.irradiance_norm)

        # Normalized production relative to peak
        production_norm = self.current_production_kW / self.peak_power_kW if self.peak_power_kW > 0 else 0.0
        states["solar_production_norm"][0] = np.float32(np.clip(production_norm, 0.0, 1.0))

    def get_electric_consumption(self, actions: Dict) -> float:
        """Get current electric energy consumption (production) from solar panel.

        Returns negative value since solar produces energy rather than consuming.
        This represents energy provided to the building/grid.
        """
        # Negative consumption = production
        return -self.current_production_kW


# Register SolarPanel with the component registry
ComponentRegistry.register('infrastructure', SolarPanel)
