import logging
from collections import OrderedDict
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from ..base import StateSource
from adv_building_gym.components.registry import ComponentRegistry
from adv_building_gym._common.constants import SLOWDOWN_TERM

logger = logging.getLogger(__name__)

# Inner statesource: deterministic physics update of s_temp_in_norm (no action).
# Sole owner of envelope params (K, mC), published as ctxt_building_K / ctxt_building_mC.

class BuildingHeatLoss(StateSource):
    """Building heat loss/gain from indoor-outdoor temperature difference.

    1R1C envelope transfer ``Q_transfer = K * (T_out - T_in)`` applied to s_temp_in_norm.
    No time-series data needed — it only updates the indoor temperature state.
    """

    # timestep comes from env_meta control_step; K and mC are explicit YAML params.
    _context_params: ClassVar[Set[str]] = {'timestep'}

    # Endogenous physics: runs before the reward under the observed exogenous row
    # (see StateSource.UPDATE_PHASE / env._update_endogenous).
    UPDATE_PHASE: ClassVar[str] = "endogenous"

    def __init__(self,
                name: str,
                K: float,
                mC: float,
                timestep: float = 300) -> None:
        """K: heat transfer coefficient [W/K]; mC: thermal mass [J/K]; timestep: seconds."""
        super().__init__(name=name)
        self.K = K
        self.mC = mC
        self.timestep = timestep
        self.temp_in_raw = 0.0  # Denormalised indoor temp after this component's update (°C)

    def setup_spaces(self,
                    state_spaces: OrderedDict,
                    action_spaces: OrderedDict) -> tuple[OrderedDict, OrderedDict]:
        """Register temp_in_norm/temp_out_norm and the building K/mC context."""
        # ensure temperature states exist (may be created by other components)
        if "s_temp_in_norm" not in state_spaces:
            state_spaces["s_temp_in_norm"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        if "s_temp_out_norm" not in state_spaces:
            state_spaces["s_temp_out_norm"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Static building physics — context, changes only on a building swap.
        if "ctxt_building_K" not in state_spaces:
            state_spaces["ctxt_building_K"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        if "ctxt_building_mC" not in state_spaces:
            state_spaces["ctxt_building_mC"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        return state_spaces, action_spaces

    def update_state(self, states: OrderedDict, info=None) -> None:
        """Update indoor temperature with envelope heat loss/gain (after infra exec_action)."""

        # NOTE VP 2026.01.14. : Reference to the 1R1C thermal model
        # Paper: EKF based self-adaptive thermal model for a passive house
        # Link: https://www.sciencedirect.com/science/article/pii/S0378778812003039?via%3Dihub
        # 1R1C (SI): Q_transfer[W] = K*(Tout_raw - Tin_raw); dT_raw[K] = SLOWDOWN_TERM*dt*Q/mC.
        # s_temp_in_norm is normalised by temp_abs_max (WeatherDataSource), so denormalise,
        # apply physics, then renormalise the increment.
        Tin_norm = states["s_temp_in_norm"][0]
        Tout_norm = states["s_temp_out_norm"][0]
        temp_abs_max = float(states["ctxt_temp_abs_max"][0]) if "ctxt_temp_abs_max" in states else 60.0

        Tin_raw = Tin_norm * temp_abs_max
        Tout_raw = Tout_norm * temp_abs_max

        # Heat transfer -- drawn from inside to the outside
        Q_transfer = self.K * (Tout_raw - Tin_raw)

        # Temperature change due to heat loss (raw °C), then renormalise.
        dT_raw = SLOWDOWN_TERM * self.timestep * Q_transfer / self.mC
        dTemp_norm = dT_raw / temp_abs_max if temp_abs_max > 0 else 0.0

        # Apply heat loss to indoor temperature
        new_temp = Tin_norm + dTemp_norm

        # Clip to observation space bounds and ensure float32
        states["s_temp_in_norm"][0] = np.float32(np.clip(new_temp, -1.0, 1.0))

        # Cache raw indoor temp after heat loss. Runs after HP, so this final value
        # wins in RawStateTracker (infras collected before statesources).
        self.temp_in_raw = float(states["s_temp_in_norm"][0]) * temp_abs_max

        # publish static building physics
        states["ctxt_building_K"][0] = np.float32(self.K)
        states["ctxt_building_mC"][0] = np.float32(self.mC)

    def get_raw_values(self) -> dict[str, float]:
        return {"raw_temp_in": self.temp_in_raw}


# register with ComponentRegistry
ComponentRegistry.register('statesource', BuildingHeatLoss)
