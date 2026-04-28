import logging
from typing import ClassVar, Dict, Set

import numpy as np
from gymnasium.spaces import Box

from .base import Infrastructure
from adv_building_gym.utils.serializable import ComponentRegistry
from adv_building_gym.utils.rng_service import RngService

logger = logging.getLogger(__name__)

# TODO VP 2026.02.17. : Add parameters for solar panel modeling.
# E.g. temperature effects, panel orientation, inverter efficiency, etc.
# For now it's kept simple with a direct mapping from irradiance to production.

class SolarPanel(Infrastructure):
    """Solar Panel (PV) infrastructure component.

    Solar panels always produce the full energy amount determined by solar irradiance.
    There is no policy-controlled action — production is purely a function of irradiance
    and peak power capacity. The actual production is written into actions['solar_action']
    as a read-only output for other components to observe.

    Action convention: negative = production (energy to grid).
    solar_action value: -1 = full peak production, 0 = no production.

    Irradiance can be provided via:
    - External state update (from a DataSource providing irradiance)
    - Synthetic time-based profile (default)
    """

    POWER_FLOW = "generator"

    # Internal state variables - don't serialize
    _exclude_params: ClassVar[Set[str]] = {
        'iteration', 'irradiance_norm', 'current_production_kW'
    }

    def __init__(self,
                name: str,
                max_power_kW: float,
                ) -> None:
        """Initialize Solar Panel infrastructure.

        Args:
            name: Component identifier
            max_power_kW: Peak power output under standard test conditions (STC).
                ``-1.0`` in ``a_solar`` corresponds to production at this value.
        """
        # NOTE VP 2026.01.24. : Inverter efficiency is not considered,
        # max power means peak output power, produced by the solar panel.
        super().__init__(name, max_power_kW)

        # State variables
        self.irradiance_norm = 0.0  # Normalized irradiance [0, 1]
        self.current_production_kW = 0.0  # Actual power production in kW

    def setup_spaces(self,
                    state_spaces,
                    action_spaces):
        """Setup observation and action spaces for solar panel.

        Solar panel has no policy-controlled action space — production is
        fully determined by solar irradiance. Only state space is registered.
        """

        if "s_solar_irradiance_norm" not in state_spaces.keys():
            # Normalized irradiance [0, 1]
            state_spaces["s_solar_irradiance_norm"] = Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            )

        # Raw peak power capacity (kW) — static context variable, only
        # changes between episodes if the config is swapped.
        if "ctxt_solar_max_power_kW" not in state_spaces.keys():
            state_spaces["ctxt_solar_max_power_kW"] = Box(
                low=0, high=np.inf, shape=(1,), dtype=np.float32
            )

        return state_spaces, action_spaces


    def exec_action(self, actions: Dict, states: Dict, info=None) -> None:
        """Compute solar production from irradiance and write it into actions.

        Production is fully determined by solar irradiance — there is no
        policy-controlled input. The result is written into actions['solar_action']
        as a read-only output for other components.
        """

        # Update irradiance from state if available (set by DataSource)
        if "s_solar_irradiance_norm" in states:
            self.irradiance_norm = float(states["s_solar_irradiance_norm"][0])

        # Use synthetic irradiance ONLY when no weather data source is active.
        # When a weather source exists, irradiance=0.0 means "no sunshine"
        # (e.g. nighttime, overcast), not "data unavailable".
        # The weather source publishes ctxt_temp_abs_max into the state dict when active.
        weather_active = "ctxt_temp_abs_max" in states
        if not weather_active and self.irradiance_norm == 0.0 and "raw_sim_hour" in states:
            self.irradiance_norm = self._synthetic_irradiance(states)

        # Production = irradiance * peak_power
        self.current_production_kW = self.irradiance_norm * self.max_power_kW

        # Write normalized production as read-only output (negative = production)
        solar_action = -self.irradiance_norm
        if "a_solar" not in actions:
            actions["a_solar"] = np.array([solar_action], dtype=np.float32)
        else:
            actions["a_solar"][0] = solar_action

    def update_state(self, states: Dict, info=None) -> None:
        """Publish static peak power into the observable state."""
        super().update_state(states, info)
        states["ctxt_solar_max_power_kW"][0] = np.float32(self.max_power_kW)

    def reset(self, states: Dict, info=None) -> None:
        """Clear per-episode irradiance/production readouts."""
        self.irradiance_norm = 0.0
        self.current_production_kW = 0.0
        super().reset(states, info)

    def _synthetic_irradiance(self, states: Dict) -> float:
        """Generate synthetic irradiance based on time of day.

        Simple bell curve approximation of solar irradiance with Gaussian noise.
        Peak at solar noon (12:00), zero at night.
        """
        sim_hour = float(states.get("raw_sim_hour", np.array([12.0]))[0])

        # Sunrise ~6:00, sunset ~18:00, peak at 12:00
        if sim_hour < 6 or sim_hour > 18:
            return 0.0

        # Cosine-based profile centered at noon
        # Maps 6-18 hours to 0-pi, with peak at pi/2 (noon)
        hour_fraction = (sim_hour - 6) / 12.0  # [0, 1] over daylight hours
        base_irradiance = np.sin(hour_fraction * np.pi)

        # Add Gaussian noise for realistic cloud cover variations
        seed = RngService.get().get_random(self.name)
        noise = np.random.default_rng(seed).normal(loc=0.0, scale=0.05)
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
