import logging
from collections import OrderedDict
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from ..base import StateSource
from adv_building_gym.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)


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

    # K and mC come from building_props, timestep from control_step
    _context_params: ClassVar[Set[str]] = {'K', 'mC', 'timestep'}

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
        if "temp_in_norm" not in state_spaces:
            state_spaces["temp_in_norm"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        if "temp_out_norm" not in state_spaces:
            state_spaces["temp_out_norm"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Static building physics parameters — context variables that only
        # change between episodes if building_props is swapped.
        if "building_K" not in state_spaces:
            state_spaces["building_K"] = Box(
                low=0, high=np.inf, shape=(1,), dtype=np.float32
            )
        if "building_mC" not in state_spaces:
            state_spaces["building_mC"] = Box(
                low=0, high=np.inf, shape=(1,), dtype=np.float32
            )

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
        Tin = states["temp_in_norm"][0]
        Tout = states["temp_out_norm"][0]

        # Heat transfer -- drawn from inside to the outside
        Q_transfer = self.K * (Tout - Tin)

        # Temperature change due to heat loss
        dTemp = 0.001 * self.timestep * Q_transfer / self.mC

        # Apply heat loss to indoor temperature
        new_temp = Tin + dTemp

        # Clip to observation space bounds and ensure float32
        states["temp_in_norm"][0] = np.float32(np.clip(new_temp, -1.0, 1.0))

        # Publish static building physics parameters.
        states["building_K"][0] = np.float32(self.K)
        states["building_mC"][0] = np.float32(self.mC)


# Register BuildingHeatLoss with the component registry
ComponentRegistry.register('statesource', BuildingHeatLoss)
