import logging
from collections import OrderedDict

import numpy as np
from gymnasium.spaces import Box

from .base import DataSource

logger = logging.getLogger(__name__)

# TODO VP 2026.01.13. : it is a good starting point -- inspect & refine

class OperatorEnergyControl(DataSource):
    """
    Data source for grid operator energy consumption limits.

    This datasource provides the maximum allowed power draw from the grid
    at each timestep, representing constraints imposed by the grid operator
    (e.g., peak demand management, grid stability requirements).
    """

    def __init__(self,
                 name: str,
                 max_power_kW: float = 10.0,
                 ds_path: str | None = None) -> None:
        """
        Initialize OperatorEnergyControl datasource.

        Args:
            name: Datasource identifier
            max_power_kW: Maximum power limit in kW for normalization (default: 10.0 kW residential)
            ds_path: Optional CSV file path with time series data
        """
        super().__init__(name, ds_path)
        self.max_power_kW = max_power_kW

        if self.ts is not None:
            logger.info("Use data file: %s", ds_path)
            # Normalize operator energy limit if CSV data is provided
            # Expected column: "operator_energy_max [kW]" or "operator_energy_max"
            if "operator_energy_max [kW]" in self.ts.columns:
                column_name = "operator_energy_max [kW]"
            elif "operator_energy_max" in self.ts.columns:
                column_name = "operator_energy_max"
            else:
                logger.warning("No 'operator_energy_max' column found in CSV, will use synthetic data")
                self.ts = None
                return

            # Normalize to [0, 1] range based on max_power_kW
            # Operator limit should be non-negative (0 to max_power_kW)
            self.ts["operator_energy_max_norm"] = self.ts[column_name] / self.max_power_kW
            # Clip to [0, 1] in case CSV has values exceeding max_power_kW
            self.ts["operator_energy_max_norm"] = self.ts["operator_energy_max_norm"].clip(0.0, 1.0)
        else:
            logger.warning("No data file provided, will use synthetic data")

    def setup_spaces(self,
                     state_spaces: OrderedDict,
                     action_spaces: OrderedDict
                     ) -> tuple[OrderedDict, OrderedDict]:
        """Setup observation spaces for operator energy control limit."""
        # Normalized to [0, 1] range (non-negative power limit)
        state_spaces["operator_energy_max"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Instantaneous grid power consumption in kW
        # This will be calculated by the environment using infrastructure.get_electric_consumption()
        state_spaces["grid_power_kW"] = Box(
            low=np.zeros((1,), dtype=np.float32),
            high=np.full((1,), np.inf, dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        if "sim_hour" not in state_spaces.keys():
            state_spaces["sim_hour"] = Box(low=np.full((1,), 0, dtype=np.float32),
                                            high=np.full((1,), np.inf, dtype=np.float32),
                                            shape=(1,),
                                            dtype=np.float32)

        return state_spaces, action_spaces

    def update_state(self, states) -> None:
        """Update operator energy limit state based on current iteration."""
        if self.ts is not None:
            if self.iteration < len(self.ts):
                operator_energy_max_norm = float(self.ts.iloc[int(self.iteration)]["operator_energy_max_norm"])
            else:
                operator_energy_max_norm = float(self.ts.iloc[-1]["operator_energy_max_norm"])
        else:
            current_sim_hour = states.get("sim_hour", np.zeros(shape=(1,), dtype=np.float32))[0]
            # Apply a simple time-based power limit profile
            # Peak hours (morning/evening): lower limit to encourage load shifting
            # Off-peak hours (night/midday): higher limit
            if current_sim_hour < 6:
                # Night: high limit (off-peak)
                operator_energy_max_norm = 0.9
            elif current_sim_hour < 9:
                # Morning peak: reduced limit
                operator_energy_max_norm = 0.5
            elif current_sim_hour < 12:
                # Mid-morning: moderate limit
                operator_energy_max_norm = 0.7
            elif current_sim_hour < 17:
                # Afternoon: moderate limit
                operator_energy_max_norm = 0.7
            elif current_sim_hour < 21:
                # Evening peak: reduced limit
                operator_energy_max_norm = 0.4
            elif current_sim_hour < 24:
                # Late evening: higher limit
                operator_energy_max_norm = 0.8
            else:
                # Default
                operator_energy_max_norm = 0.7

        # Ensure float32 dtype and clip to bounds [0, 1]
        operator_energy_max_norm = np.float32(np.clip(operator_energy_max_norm, 0.0, 1.0))
        states["operator_energy_max"][0] = operator_energy_max_norm
