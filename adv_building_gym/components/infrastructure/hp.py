from collections import OrderedDict
import logging
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from .base import Infrastructure
from adv_building_gym.components.registry import ComponentRegistry
from adv_building_gym._common.constants import SLOWDOWN_TERM, W_PER_KW, TEMP_ABS_MAX_CELSIUS

logger = logging.getLogger(__name__)


# NOTE VP 2026.01.14. : There is 2 types of states:
# 1. affected by actions -- handled in exec_action and update_state
# 2. static states -- set in setup_spaces, updated externally via datasources
# It is important to keep this distinction in mind and not mix them up.

class HP(Infrastructure):
    """Heat pump. Action a_hp in [-1, 1]: negative=cool, positive=heat; consumes |a_hp|*max_power_kW."""

    POWER_FLOW = "consumer"

    # control_step from env context; mC read from ctxt_building_mC obs at runtime.
    _context_params: ClassVar[Set[str]] = {'control_step'}

    # Internal state - not serialised
    _exclude_params: ClassVar[Set[str]] = {'iteration', 'temp_in_norm', 'temp_in_norm_change', 'control_step', 'actual_power_kW', 'temp_in_raw'}

    def __init__(self,
                name: str,
                max_power_kW: float,
                control_step: int,
                cop_heat: float = 1.0,
                cop_cool: float = 1.0,
                emit_ctxt: bool = False,
                ) -> None:
        super().__init__(name, max_power_kW)
        self.emit_ctxt = emit_ctxt

        # NOTE VP 2026.01.20. : COP, link: https://en.wikipedia.org/wiki/Coefficient_of_performance
        # COP = Q_thermal / P_electric => Q_thermal = P_electric * COP
        self.cop_heat = cop_heat  # [-] heating COP
        self.cop_cool = cop_cool  # [-] cooling COP
        self.control_step = control_step

        self.temp_in_norm = 0
        self.temp_in_norm_change = 0
        self.actual_power_kW = 0.0  # actual electric draw (kW), for reporting
        self.temp_in_raw = 0.0  # Denormalised indoor temp after this component's update (°C)

        if self.cop_heat <= 0 or self.cop_cool <= 0:
            raise ValueError("cop_heat and cop_cool must be positive.")

    def setup_spaces(self,
                    state_spaces: OrderedDict,
                    action_spaces: OrderedDict) -> tuple[OrderedDict, OrderedDict]:
        # a_hp in [-1, 1]: sign = cool/heat, magnitude = level
        action_spaces["a_hp"] = Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,), dtype=np.float32
        )

        # Indoor/outdoor temperature are not observations — indoor temp is the shared
        # integration variable on info["temp_in_norm"], outdoor temp lives on
        # info["temp_out_norm"]; the policy sees the comfort error (s_temp_error_norm).

        # Raw electric capacity (kW) — policy-only conditioning, gated by emit_ctxt.
        self._publish_ctxt(state_spaces, "ctxt_hp_max_power_kW", Box(low=0, high=np.inf, shape=(1,), dtype=np.float32))

        return state_spaces, action_spaces

    def exec_action(self, actions, states, info=None) -> None:
        # a_hp in [-1, 1]: negative=cool, positive=heat.
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

        # TODO noprio VP 2026.03.16. : Refinement idea for slow cooling/ slow heating. Add venting system / window open controller (as infrastructure),
        # which can cool the house faster if the temperature diff is too big and cooling is not fast enough.
        # Possible to schedule it, if once fired, then it can't be fire again in an hour -- physics of venting/ventillating a house?
        # action, but with minimal energy (as window open and close is there).
        # Refinement idea: If the wind is too strong or wind is higher than a threshold and it's raining, do not allow this action

        # NOTE VP 2026.01.20. : Heat loss Q_transfer is now handled by BuildingHeatLoss datasource.
        # It is a continous efferct
        # HP only applies its heating/cooling effect
        # NOTE VP 2026.01.20. : Thermal model (1R1C) -- lumped-parameter models
        # paper1: Particle Swarm Optimization and Kalman Filtering for Demand Prediction of Commercial Buildings
        # Link: https://www.researchgate.net/publication/301310479_Particle_Swarm_Optimization_and_Kalman_Filtering_for_Demand_Prediction_of_Commercial_Buildings
        # paper2: EKF based self-adaptive thermal model for a passive house
        # Link: https://www.sciencedirect.com/science/article/pii/S0378778812003039?via%3Dihub
        # NOTE VP 2026.01.20. : According to paper2, 1R1C mean RMS error to the reality is ~0.47 C --> influences precision

        # Thermal mass owned/published by BuildingHeatLoss; read as obs to keep
        # envelope params on a single owner.
        mC = float(states["ctxt_building_mC"][0])
        # Fixed temperature normalisation scale from the info channel (WeatherDataSource).
        temp_abs_max = float(info["temp_abs_max"]) if info is not None and "temp_abs_max" in info else TEMP_ABS_MAX_CELSIUS

        # 1R1C update (SI): dT_raw [K] = SLOWDOWN_TERM * dt * q_hp_W / mC.
        # q_hp converted kW->W; SLOWDOWN_TERM is the dynamical slowdown (see constants.py).
        # Indoor temperature is normalised, so divide the raw °C change by temp_abs_max.
        q_hp_W = q_hp * W_PER_KW
        dT_raw = SLOWDOWN_TERM * self.control_step * q_hp_W / mC
        dTemp_norm = dT_raw / temp_abs_max if temp_abs_max > 0 else 0.0

        # Indoor temperature: shared integration variable on the info channel
        # (info["temp_in_norm"]), not an observation — the policy sees the comfort error
        # (s_temp_error_norm) instead. Check if the change would clip at the ±1 bounds.
        current_temp_norm = float(info.get("temp_in_norm", 0.0)) if info is not None else 0.0
        new_temp_norm = current_temp_norm + dTemp_norm

        if new_temp_norm > 1.0 or new_temp_norm < -1.0:
            # temp change needed to reach the limit
            if new_temp_norm > 1.0:
                actual_dTemp = 1.0 - current_temp_norm
            else:  # new_temp < -1.0
                actual_dTemp = -1.0 - current_temp_norm

            # Invert the forward path: dTemp(norm) -> dT_raw -> q_hp_W -> q_hp(kW) -> energy
            actual_dT_raw = actual_dTemp * temp_abs_max
            actual_q_hp_W = actual_dT_raw * mC / (SLOWDOWN_TERM * self.control_step)
            actual_q_hp_kW = actual_q_hp_W / W_PER_KW
            actual_energy = abs(actual_q_hp_kW) / (self.max_power_kW * cop) if (self.max_power_kW * cop) > 0 else 0.0
            actual_energy = np.clip(actual_energy, 0.0, 1.0)

            # preserve sign (cool/heat)
            sign = -1.0 if hp_action < 0 else 1.0
            actions["a_hp"][0] = np.float32(sign * actual_energy)

            # store actual temp change + power
            self.temp_in_norm_change = actual_dTemp
            self.actual_power_kW = actual_energy * self.max_power_kW
        else:
            # no clip needed
            self.temp_in_norm_change = dTemp_norm
            self.actual_power_kW = energy * self.max_power_kW

    def update_state(self, states, info=None) -> None:
        super().update_state(states, info)

        # Apply the HP's thermal effect to the shared indoor temperature on info.
        current_temp_norm = float(info.get("temp_in_norm", 0.0)) if info is not None else 0.0
        new_temp: float = current_temp_norm + self.temp_in_norm_change  # clip ensured in exec_action
        if info is not None:
            info["temp_in_norm"] = new_temp
        self._write_ctxt(states, "ctxt_hp_max_power_kW", np.float32(self.max_power_kW))

        # Cache raw indoor temp after this component's heat. BuildingHeatLoss
        # re-derives it later; collected after infras, so its value wins.
        temp_abs_max = float(info["temp_abs_max"]) if info is not None and "temp_abs_max" in info else TEMP_ABS_MAX_CELSIUS
        self.temp_in_raw = new_temp * temp_abs_max

    def reset(self, states, info=None) -> None:
        """Clear per-episode transient state before publishing initial obs."""
        self.temp_in_norm = 0
        self.temp_in_norm_change = 0
        self.actual_power_kW = 0.0
        self.temp_in_raw = 0.0
        super().reset(states, info)
        
    def get_raw_values(self) -> dict[str, float]:
        return {
            "raw_hp_kW": self.actual_power_kW,
            "raw_temp_in": self.temp_in_raw,
        }

    def get_E(self, actions) -> tuple[float, float]:
        """Electric consumption (kW), always positive. Uses post-clip power from exec_action."""
        return 0.0, self.actual_power_kW


# register with ComponentRegistry
ComponentRegistry.register('infrastructure', HP)
