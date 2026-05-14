import logging
from collections import OrderedDict
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from ..base import StateSource
from adv_building_gym.utils.serializable import ComponentRegistry
from adv_building_gym.utils.constants import SLOWDOWN_TERM

logger = logging.getLogger(__name__)

# Inner statesource: deterministic physics update of s_temp_in_norm (no action input).
# Sole owner of the building envelope params (K, mC); other components that need them
# read from the published `ctxt_building_K` / `ctxt_building_mC` observations.

class BuildingHeatLoss(StateSource):
    """
    Data source for building heat loss/gain due to temperature difference
    between inside and outside.

    This datasource models heat transfer through building envelope using:
    Q_transfer = K * (T_in - T_out)

    where K is the building's heat transfer coefficient and temperatures
    are the normalized indoor and outdoor temperatures.

    **Note**: No additional time series data is required for this statesource,
    it is just updates the indoor temperature state.
    """

    # timestep comes from env_meta control_step; K and mC are explicit YAML params.
    _context_params: ClassVar[Set[str]] = {'timestep'}

    def __init__(self,
                name: str,
                K: float,
                mC: float,
                timestep: float = 300,
                ds_path: str | None = None) -> None:
        """
        Initialize BuildingHeatLoss datasource.

        Args:
            name: Datasource identifier
            K: Heat transfer coefficient [W/K]
            mC: Building thermal mass [J/K]
            timestep: Time step duration in seconds (default 300s = 5min)
            ds_path: Optional data file path (not used for this datasource)
        """
        super().__init__(name, ds_path)
        self.K = K
        self.mC = mC
        self.timestep = timestep

    def setup_spaces(self,
                    state_spaces: OrderedDict,
                    action_spaces: OrderedDict) -> tuple[OrderedDict, OrderedDict]:
        """Setup observation spaces - requires temp_in_norm and temp_out_norm."""
        # Ensure temperature states exist (may be created by other components)
        if "s_temp_in_norm" not in state_spaces:
            state_spaces["s_temp_in_norm"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        if "s_temp_out_norm" not in state_spaces:
            state_spaces["s_temp_out_norm"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Static building physics parameters — context variables that only
        # change between episodes if building_props is swapped.
        if "ctxt_building_K" not in state_spaces:
            state_spaces["ctxt_building_K"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        if "ctxt_building_mC" not in state_spaces:
            state_spaces["ctxt_building_mC"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        return state_spaces, action_spaces

    def update_state(self, states: OrderedDict, info=None) -> None:
        """
        Update indoor temperature based on heat loss/gain to outdoor environment.

        This should be called after infrastructure exec_action to apply
        heat loss on top of any heating/cooling provided by HP or other devices.
        """

        # NOTE VP 2026.01.14. : Reference to the 1R1C thermal model
        # Paper: EKF based self-adaptive thermal model for a passive house
        # Link: https://www.sciencedirect.com/science/article/pii/S0378778812003039?via%3Dihub
        # 1R1C update in strict SI (LLEC convention):
        #   Q_transfer [W] = K [W/K] * (Tout_raw - Tin_raw) [K]
        #   dT_raw [K]     = SLOWDOWN_TERM * dt * Q_transfer / mC
        # The state buffer s_temp_in_norm is normalised by temp_abs_max
        # (published by WeatherDataSource), so the formula denormalises
        # to raw °C, applies physics, then renormalises the increment.
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

        # Publish static building physics parameters.
        states["ctxt_building_K"][0] = np.float32(self.K)
        states["ctxt_building_mC"][0] = np.float32(self.mC)


# Register BuildingHeatLoss with the component registry
ComponentRegistry.register('statesource', BuildingHeatLoss)
