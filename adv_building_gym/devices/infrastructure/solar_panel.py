import logging
from typing import ClassVar, Dict, Set

import numpy as np
from gymnasium.spaces import Box

from .base import Infrastructure
from adv_building_gym.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)

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
                control_step: int,
                # Link: https://www.ise.fraunhofer.de/content/dam/ise/de/documents/publications/studies/Photovoltaics-Report.pdf
                pv_efficiency: float,
                panel_area_m2: float
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
        self.irradiance_Jcm2 = 0.0  # Irradiance
        self.current_production_kW = 0.0  # Actual power production in kW
        self.pv_efficiency = pv_efficiency
        self.panel_area_m2 = panel_area_m2
        self.control_step = control_step

    def setup_spaces(self,
                    state_spaces,
                    action_spaces):
        """Setup observation and action spaces for solar panel.

        Solar panel has no policy-controlled action space — production is
        fully determined by solar irradiance. Only state space is registered.
        """

        if "s_solar_irradiance_norm" not in state_spaces.keys():
            # Normalized irradiance [0, 1]
            state_spaces["s_solar_irradiance_norm"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Raw peak power capacity (kW) — static context variable, only
        # changes between episodes if the config is swapped.
        if "ctxt_solar_max_power_kW" not in state_spaces.keys():
            state_spaces["ctxt_solar_max_power_kW"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
            
        if "ctxt_pv_efficiency" not in state_spaces.keys():
            state_spaces["ctxt_pv_efficiency"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        return state_spaces, action_spaces


    def exec_action(self, actions: Dict, states: Dict, info=None) -> None:
        """Compute solar production from irradiance and write it into actions.

        Production is fully determined by solar irradiance — there is no
        policy-controlled input. The result is written into actions['solar_action']
        as a read-only output for other components.
        """

        # Denormalise irradiance: s_solar_irradiance_norm in [0,1], scale by
        # ctxt_solar_irradiance_max (J/cm² per control step, published by WeatherDataSource).
        irradiance_norm = float(states["s_solar_irradiance_norm"][0])
        irradiance_max_Jcm2 = float(states["ctxt_solar_irradiance_max"][0])
        self.irradiance_Jcm2 = irradiance_norm * irradiance_max_Jcm2

        # Convert energy [J/cm² per step] over panel area [m²] to mean power [kW] over the step.
        # 1 m² = 10_000 cm² → J per m² = Jcm2 * 10_000; J → kJ → /1000; kJ / s = kW → /control_step.
        # Combined factor: 10_000 / 1000 / control_step = 10 / control_step.
        # Link: https://www.alternative-energy-tutorials.com/solar-power/solar-panel-efficiency.html
        self.current_production_kW = (
            self.irradiance_Jcm2 * self.panel_area_m2 * self.pv_efficiency
            * 10.0 / self.control_step
        )
        self.current_production_kW = np.clip(self.current_production_kW, 0.0, self.max_power_kW)

        # Write normalized production as read-only output (negative = production)
        # TODO VP 2026.05.06.: no meaning behind positive and negative action signs,
        # as power generation and consumption do not depend on the action sign, because it's per Infra computed
        solar_action = -irradiance_norm
        if "a_solar" not in actions:
            actions["a_solar"] = np.array([solar_action], dtype=np.float32)
        else:
            actions["a_solar"][0] = solar_action

    def update_state(self, states: Dict, info=None) -> None:
        """Publish static peak power into the observable state."""
        super().update_state(states, info)
        states["ctxt_solar_max_power_kW"][0] = np.float32(self.max_power_kW)
        states["ctxt_pv_efficiency"][0] = np.float32(self.pv_efficiency)

    def reset(self, states: Dict, info=None) -> None:
        """Clear per-episode irradiance/production readouts."""
        self.irradiance_Jcm2 = 0.0
        self.current_production_kW = 0.0
        super().reset(states, info)
    
    def get_raw_values(self) -> Dict[str, float]:
        return {
            "raw_pv_prod": self.current_production_kW,
            "raw_pv_max": self.max_power_kW
        }

    def get_E(self, actions: Dict) -> tuple[float, float]:
        """Get current electric energy consumption (production) from solar panel.

        Solar panels produce energy, so this returns a negative value.

        Returns:
            float1 -- production
            float2 -- consumption
        """
        # Negative consumption = production to grid
        return self.current_production_kW, 0.0


# Register SolarPanel with the component registry
ComponentRegistry.register('infrastructure', SolarPanel)
