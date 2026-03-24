import logging
from typing import ClassVar, Dict, Set

import numpy as np
from gymnasium.spaces import Box

from .base import Infrastructure
from adv_building_gym.config.utils.serializable import ComponentRegistry

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

    # control_step and seed come from config context
    _context_params: ClassVar[Set[str]] = {'control_step', 'seed'}

    # Internal state variables - don't serialize
    _exclude_params: ClassVar[Set[str]] = {
        'iteration', 'irradiance_norm', 'current_production_kW', '_base_seed'
    }

    def __init__(self,
                name: str,
                Q_electric_max: float,
                peak_power_kW: float,
                # TODO VP 2026.03.17. : Solar panel seed -- channel global seed in.
                seed: int = 42,
                control_step: int = 300
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

        # NOTE VP 2026.01.24. : Inverter efficiency is not considered, 
        # peak power means peak output power, produced by the solar panel
        self.peak_power_kW = peak_power_kW # -1.0 at actions means the peak power
        self._base_seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.control_step = control_step

        # State variables
        self.irradiance_norm = 0.0  # Normalized irradiance [0, 1]
        self.current_production_kW = 0.0  # Actual power production in kW


    def synchronise(self, iteration: int, row_offset: int | None = None) -> None:
        super().synchronise(iteration, row_offset)
        # Reseed RNG at episode reset (row_offset is only passed on reset, not
        # per-step) so that solar noise is reproducible per episode.
        if row_offset is not None:
            self.rng = np.random.default_rng(seed=self._base_seed + row_offset)

    def setup_spaces(self,
                    state_spaces,
                    action_spaces):
        """Setup observation and action spaces for solar panel.

        Solar panel has no policy-controlled action space — production is
        fully determined by solar irradiance. Only state space is registered.
        """

        if "solar_irradiance_norm" not in state_spaces.keys():
            # Normalized irradiance [0, 1]
            state_spaces["solar_irradiance_norm"] = Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            )

        return state_spaces, action_spaces


    def exec_action(self, actions: Dict, states: Dict, info=None) -> None:
        """Compute solar production from irradiance and write it into actions.

        Production is fully determined by solar irradiance — there is no
        policy-controlled input. The result is written into actions['solar_action']
        as a read-only output for other components.
        """

        # Update irradiance from state if available (set by DataSource)
        if "solar_irradiance_norm" in states:
            self.irradiance_norm = float(states["solar_irradiance_norm"][0])

        # If no external irradiance, use synthetic time-based profile
        if self.irradiance_norm == 0.0 and "sim_hour" in states:
            self.irradiance_norm = self._synthetic_irradiance(states)

        # Production = irradiance * peak_power
        self.current_production_kW = self.irradiance_norm * self.peak_power_kW

        # Write normalized production as read-only output (negative = production)
        solar_action = -self.irradiance_norm
        if "solar_action" not in actions:
            actions["solar_action"] = np.array([solar_action], dtype=np.float32)
        else:
            actions["solar_action"][0] = solar_action

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
