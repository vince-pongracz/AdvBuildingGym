import logging
from collections import OrderedDict
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from ..base import StateSource
from adv_building_gym.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)


class OperatorEnergyControl(StateSource):
    """
    Data source for grid operator energy consumption limits.

    This datasource provides the maximum allowed power draw from the grid
    at each timestep, representing constraints imposed by the grid operator
    (e.g., peak demand management, grid stability requirements).
    """

    def __init__(self,
                name: str,
                max_power_kW: float,
                ds_path: str) -> None:
        """
        Initialize OperatorEnergyControl datasource.

        Args:
            name: Datasource identifier
            max_power_kW: Maximum power limit in kW for normalization
            ds_path: CSV file path with operator energy limit time series (required)
        """
        super().__init__(name, ds_path)
        self.max_power_kW = max_power_kW

        if self.ts is None:
            raise ValueError(
                f"OperatorEnergyControl '{name}' requires a CSV at ds_path; got '{ds_path}'."
            )

        logger.info("Use data file: %s", ds_path)
        if "operator_energy_max [kW]" in self.ts.columns:
            column_name = "operator_energy_max [kW]"
        elif "operator_energy_max" in self.ts.columns:
            column_name = "operator_energy_max"
        else:
            raise ValueError(
                f"OperatorEnergyControl '{name}': CSV '{ds_path}' has no "
                "'operator_energy_max [kW]' or 'operator_energy_max' column."
            )

        self.max_power_kW = float(self.ts[column_name].max())
        # Normalize to [0, 1] based on max_power_kW; clip in case CSV exceeds it.
        self.ts["operator_energy_max_norm"] = (self.ts[column_name] / self.max_power_kW).clip(0.0, 1.0)

    def setup_spaces(self,
                    state_spaces: OrderedDict,
                    action_spaces: OrderedDict) -> tuple[OrderedDict, OrderedDict]:
        """Setup observation spaces for operator energy control limit."""
        # Normalized to [0, 1] range (non-negative power limit)
        if "s_operator_energy_max" not in state_spaces.keys():
            state_spaces["s_operator_energy_max"] = Box(
                low=0, high=1, shape=(1,), dtype=np.float32)

        # Raw maximum operator power limit (kW) — static context variable,
        # only changes when a new data variant is loaded.
        if "ctxt_operator_max_power_kW" not in state_spaces.keys():
            state_spaces["ctxt_operator_max_power_kW"] = Box(
                low=0, high=np.inf, shape=(1,), dtype=np.float32)

        # Instantaneous grid power consumption in kW
        # This will be calculated by the environment using infrastructure.get_electric_consumption()
        # grid_power_kW removed from observation space — it was never updated
        # and the actual value is available via info["step_power_kW"].

        if "raw_sim_hour" not in state_spaces.keys():
            state_spaces["raw_sim_hour"] = Box(low=0,
                high=np.inf,
                shape=(1,),
                dtype=np.float32
            )

        return state_spaces, action_spaces

    def update_state(self, states, info=None) -> None:
        """Update operator energy limit state based on current iteration."""
        assert self.ts is not None
        if self.effective_index < len(self.ts):
            operator_energy_max_norm = float(self.ts.iloc[self.effective_index]["operator_energy_max_norm"])
        else:
            operator_energy_max_norm = float(self.ts.iloc[-1]["operator_energy_max_norm"])

        # Ensure float32 dtype and clip to bounds [0, 1]
        operator_energy_max_norm = np.float32(np.clip(operator_energy_max_norm, 0.0, 1.0))
        states["s_operator_energy_max"][0] = operator_energy_max_norm
        # Raw maximum power limit (kW) — constant within an episode.
        states["ctxt_operator_max_power_kW"][0] = np.float32(self.max_power_kW)


# Register OperatorEnergyControl with the component registry
ComponentRegistry.register('statesource', OperatorEnergyControl)
