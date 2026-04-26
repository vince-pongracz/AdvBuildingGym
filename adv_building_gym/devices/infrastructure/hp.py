from collections import OrderedDict
import logging
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from .base import Infrastructure
from adv_building_gym.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)


# NOTE VP 2026.01.14. : There is 2 types of states:
# 1. affected by actions -- handled in exec_action and update_state
# 2. static states -- set in setup_spaces, updated externally via datasources
# It is important to keep this distinction in mind and not mix them up.

class HP(Infrastructure):
    """Heat Pump infrastructure component.

    Action convention: single value in [-1, 1].
    Negative = cooling, positive = heating, magnitude = energy level.
    Heat pumps only consume energy (|action| * max_power_kW).
    """

    POWER_FLOW = "consumer"

    # K and mC come from building_props context
    _context_params: ClassVar[Set[str]] = {'K', 'mC'}

    # Internal state variables - don't serialize
    _exclude_params: ClassVar[Set[str]] = {'iteration', 'temp_in_norm', 'temp_in_norm_change', 'control_step', 'actual_power_kW'}

    def __init__(self,
                name: str,
                max_power_kW: float,
                K: float,
                mC: float,
                control_step: int,
                cop_heat: float = 1.0,
                cop_cool: float = 1.0,
                ) -> None:
        super().__init__(name, max_power_kW)

        # NOTE VP 2026.01.20. : COP, link: https://en.wikipedia.org/wiki/Coefficient_of_performance
        # COP = Q_thermal / P_electric => Q_thermal = P_electric * COP
        self.cop_heat = cop_heat  # [-] heating COP
        self.cop_cool = cop_cool  # [-] cooling COP
        self.control_step = control_step
        self.K = K
        self.mC = mC

        self.temp_in_norm = 0
        self.temp_in_norm_change = 0
        self.actual_power_kW = 0.0  # Track actual electric consumption for reporting

        if self.cop_heat <= 0 or self.cop_cool <= 0:
            raise ValueError("cop_heat and cop_cool must be positive.")

    def setup_spaces(self,
                    state_spaces: OrderedDict,
                    action_spaces: OrderedDict) -> tuple[OrderedDict, OrderedDict]:
        # HP action is 1D: [-1, 1]
        # Negative = cooling, positive = heating, magnitude = energy level
        action_spaces["a_hp"] = Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32
        )

        if "s_temp_in_norm" not in state_spaces:
            state_spaces["s_temp_in_norm"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        if "s_temp_out_norm" not in state_spaces:
            state_spaces["s_temp_out_norm"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Raw electric capacity (kW) — static context variable, only
        # changes between episodes if the config is swapped.
        if "ctxt_hp_max_power_kW" not in state_spaces:
            state_spaces["ctxt_hp_max_power_kW"] = Box(
                low=0, high=np.inf, shape=(1,), dtype=np.float32
            )

        return state_spaces, action_spaces

    def exec_action(self, actions, states, info=None) -> None:
        # Action is 1D: [-1, 1]. Negative = cooling, positive = heating.
        hp_action = float(np.atleast_1d(actions["a_hp"])[0])
        energy = abs(hp_action)

        # NOTE VP 2026.01.20. : Thermal model is 1R1C, same as links below
        # Thermal power Q_thermal = energy * max_power_kW * COP
        # Sign of q_hp follows the action: positive = heating, negative = cooling
        if hp_action < 0:
            cop = self.cop_cool
            q_hp = -energy * self.max_power_kW * cop  # heat removed from building
        elif hp_action > 0:
            cop = self.cop_heat
            q_hp = energy * self.max_power_kW * cop  # heat added to building
        else:
            self.temp_in_norm_change = 0.0
            self.actual_power_kW = 0.0
            return

        # TODO VP 2026.03.16. : Refinement idea for slow cooling/ slow heating. Add venting system / window open controller (as infrastructure),
        # which can cool the house faster if the temperature diff is too big and cooling is not fast enough.
        # Possible to schedule it, if once fired, then it can't be fire again in an hour -- physics of venting/ventillating a house?
        # action, but with minimal energy (as window open and close is there).
        # Refinement idea: If the wind is too strong or wind is higher than a threshold and it's raining, do not allow this action

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
        current_temp = states["s_temp_in_norm"][0]
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
            # |q_hp| = energy * max_power_kW * cop
            # => energy = |q_hp| / (max_power_kW * cop)
            actual_energy = abs(actual_q_hp) / (self.max_power_kW * cop) if (self.max_power_kW * cop) > 0 else 0.0
            actual_energy = np.clip(actual_energy, 0.0, 1.0)

            # Update action preserving sign (cooling/heating direction)
            sign = -1.0 if hp_action < 0 else 1.0
            actions["a_hp"][0] = np.float32(sign * actual_energy)

            # Store the actual temperature change and power consumption
            self.temp_in_norm_change = actual_dTemp
            self.actual_power_kW = actual_energy * self.max_power_kW
        else:
            # No clipping needed, use the original dTemp
            self.temp_in_norm_change = dTemp
            self.actual_power_kW = energy * self.max_power_kW

    def update_state(self, states, info=None) -> None:
        super().update_state(states, info)
        new_temp = states["s_temp_in_norm"][0] + self.temp_in_norm_change
        # Clipping ensured in exec_action -- maybe reintroduction needed later
        states["s_temp_in_norm"][0] = np.float32(new_temp)
        states["ctxt_hp_max_power_kW"][0] = np.float32(self.max_power_kW)

    def reset(self, states, info=None) -> None:
        """Clear per-episode transient state before publishing initial obs."""
        self.temp_in_norm = 0
        self.temp_in_norm_change = 0
        self.actual_power_kW = 0.0
        super().reset(states, info)

    def get_electric_consumption(self, actions) -> float:
        """Get current electric energy consumption from heat pump in kW.

        Always positive — HP only consumes energy regardless of heating/cooling mode.
        Uses the actual power computed during exec_action (accounts for clipping).
        """
        return self.actual_power_kW


# Register HP with the component registry
ComponentRegistry.register('infrastructure', HP)
