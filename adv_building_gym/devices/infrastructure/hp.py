import logging
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from .base import Infrastructure
from adv_building_gym.config.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)


# NOTE VP 2026.01.14. : There is 2 types of states:
# 1. affected by actions -- handled in exec_action and update_state
# 2. static states -- set in setup_spaces, updated externally via datasources
# It is important to keep this distinction in mind and not mix them up.

class HP(Infrastructure):
    """Heat Pump infrastructure component."""

    # K and mC come from building_props context
    _context_params: ClassVar[Set[str]] = {'K', 'mC'}

    # Internal state variables - don't serialize
    _exclude_params: ClassVar[Set[str]] = {'iteration', 'temp_norm_in', 'temp_norm_in_change', 'control_step'}

    def __init__(self,
                 name: str,
                 Q_electric_max: float,
                 K: float,
                 mC: float,
                 cop_heat: float = 1.0,
                 cop_cool: float = 1.0,
                 control_step: int = 300
                 ) -> None:
        super().__init__(name, Q_electric_max)

        # NOTE VP 2026.01.20. : COP, link: https://en.wikipedia.org/wiki/Coefficient_of_performance
        # COP = Q_thermal / P_electric => Q_thermal = P_electric * COP
        self.cop_heat = cop_heat  # [-] heating COP
        self.cop_cool = cop_cool  # [-] cooling COP
        self.control_step = control_step
        self.K = K
        self.mC = mC

        self.temp_norm_in = 0
        self.temp_norm_in_change = 0

        if self.cop_heat <= 0 or self.cop_cool <= 0:
            raise ValueError("cop_heat and cop_cool must be positive.")

    def setup_spaces(self,
                     state_spaces,
                     action_spaces
                     ):
        # HP action is 2D: [energy, mode]
        # - energy: [-1, 0] - HP always consumes energy (no positive E possible)
        # - mode: [0, 1] - <0.4: cooling, >0.6: heating, [0.4, 0.6]: no action
        action_spaces["HP_action"] = Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([0.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        if "temp_norm_in" not in state_spaces:
            state_spaces["temp_norm_in"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        if "temp_norm_out" not in state_spaces:
            state_spaces["temp_norm_out"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        return state_spaces, action_spaces

    def exec_action(self, actions, states) -> None:
        action = actions["HP_action"]
        # Action is 2D: [energy, mode]
        energy = float(np.atleast_1d(action)[0])
        mode = float(np.atleast_1d(action)[1])

        # NOTE VP 2026.01.20. : Thermal model is 1R1C, same as links below
        # Determine mode: cooling (<0.4), heating (>0.6), or no action ([0.4, 0.6])
        # energy is in [-1, 0], thermal power Q_thermal = abs(energy) * Q_electric_max * COP
        if mode < 0.4:
            # Cooling mode: remove heat from building (negative q_hp)
            cop = self.cop_cool
            q_hp = -abs(energy) * self.Q_electric_max * cop  # heat removed from building
        elif mode > 0.6:
            # Heating mode: add heat to building (positive q_hp)
            cop = self.cop_heat
            q_hp = abs(energy) * self.Q_electric_max * cop  # heat added to building
        else:
            # No action zone [0.4, 0.6]
            q_hp = 0.0
            # Set also the energy part to 0 in this case -- at rewards it is useful to have the real actions
            actions["HP_action"][0] = 0.0
            self.temp_norm_in_change = 0.0
            return

        # TODO VP 2026.01.20. : Add forecasting window (and thus MPC) for the states and the
        # actions as well in the config, generally window size is 0.
        # Allow it only for the forecasted desired states -- not for the actual system states
        # Handle if no more forecasting is available (csv ended and similar scenarios)

        # NOTE VP 2026.01.20. : Heat loss Q_transfer is now handled by BuildingHeatLoss datasource.
        # It is a continous efferct
        # HP only applies its heating/cooling effect
        # NOTE VP 2026.01.20. : Thermal model (1R1C) -- lumped-parameter models
        # paper1: Particle Swarm Optimization and Kalman Filtering for Demand Prediction of Commercial Buildings
        # Link: https://www.researchgate.net/publication/301310479_Particle_Swarm_Optimization_and_Kalman_Filtering_for_Demand_Prediction_of_Commercial_Buildings
        # paper2: EKF based self-adaptive thermal model for a passive house
        # Link: https://www.sciencedirect.com/science/article/pii/S0378778812003039?via%3Dihub
        # NOTE VP 2026.01.20. : According to paper2, 1R1C mean RMS error to the reality is ~0.47 C --> influences precision

        dTemp = 0.001 * self.control_step * q_hp / self.mC

        # Check if temperature would be clipped after the change
        current_temp = states["temp_norm_in"][0]
        new_temp = current_temp + dTemp

        if new_temp > 1.0 or new_temp < -1.0:
            # Calculate actual temperature change needed to reach the limit
            if new_temp > 1.0:
                actual_dTemp = 1.0 - current_temp
            else:  # new_temp < -1.0
                actual_dTemp = -1.0 - current_temp

            # Back-calculate actual q_hp from actual dTemp
            # dTemp = 0.001 * control_step * q_hp / mC
            # => q_hp = dTemp * mC / (0.001 * control_step)
            actual_q_hp = actual_dTemp * self.mC / (0.001 * self.control_step)

            # Back-calculate actual energy from actual q_hp
            if mode < 0.4:
                # Cooling: q_hp = -abs(energy) * Q_electric_max * cop
                # => abs(energy) = -q_hp / (Q_electric_max * cop)
                # energy in [-1, 0], so energy = -abs(energy) = q_hp / (Q_electric_max * cop)
                actual_energy = actual_q_hp / (self.Q_electric_max * cop) if (self.Q_electric_max * cop) > 0 else 0.0
            else:  # mode > 0.6 (heating)
                # Heating: q_hp = abs(energy) * Q_electric_max * cop
                # => abs(energy) = q_hp / (Q_electric_max * cop)
                # energy in [-1, 0], so energy = -abs(energy) = -q_hp / (Q_electric_max * cop)
                actual_energy = -actual_q_hp / (self.Q_electric_max * cop) if (self.Q_electric_max * cop) > 0 else 0.0

            # Update action with the reduced energy (preserve mode)
            actions["HP_action"][0] = np.float32(actual_energy)

            # Store the actual temperature change
            self.temp_norm_in_change = actual_dTemp
        else:
            # No clipping needed, use the original dTemp
            self.temp_norm_in_change = dTemp

    def update_state(self, states) -> None:
        new_temp = states["temp_norm_in"][0] + self.temp_norm_in_change
        # Clipping ensured in exec_action -- maybe reintroduction needed later
        states["temp_norm_in"][0] = np.float32(new_temp)

    def get_electric_consumption(self, actions) -> float:
        """Get current electric energy consumption from heat pump.

        HP action is [energy, mode] where energy is in [-1, 0].
        Actual consumption is abs(energy) * Q_electric_max.
        """
        if "HP_action" not in actions:
            return 0.0

        action = actions["HP_action"]
        energy = float(np.atleast_1d(action)[0])
        # Energy is negative ([-1, 0]), so take absolute value
        return abs(energy) * self.Q_electric_max


# Register HP with the component registry
ComponentRegistry.register('infrastructure', HP)
