import logging
from collections import OrderedDict
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from .base import StateSource
from adv_building_gym.config.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)

# TODO VP 2026.01.08. : Search for energy need profile data during a day. Pay attention to weekdays, weekends, etc...
# TODO VP 2026.01.08. : add reward function to this

class DesiredUserEnergyNeed(StateSource):
    """
    Data source for desired user energy need information.
    
    state key: desired_energy_need
    Definition desired_energy_need:
    - positive: user consumes E.
    - negative: no meaning
    - zero: user does not need E.
    """

    def __init__(self, name: str, ds_path: str | None = None) -> None:
        super().__init__(name, ds_path)

        if self.ts is not None:
            logger.info("Use data file: %s", ds_path)
            # TODO VP: Add column name and normalization logic if needed
            # Example: self.energy_need_max = float(self.ts["energy_need"].max())
        else:
            logger.info("No data file provided, will use synthetic data")

    def setup_spaces(self,
                     state_spaces: OrderedDict,
                     action_spaces: OrderedDict
                     ) -> tuple[OrderedDict, OrderedDict]:
        """Setup observation spaces for desired user energy need."""

        if "desired_energy_need" not in state_spaces.keys():
            state_spaces["desired_energy_need"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        if "sim_hour" not in state_spaces.keys():
            state_spaces["sim_hour"] = Box(low=np.full((1,), 0, dtype=np.float32),
                                            high=np.full((1,), np.inf, dtype=np.float32),
                                            shape=(1,),
                                            dtype=np.float32)

        return state_spaces, action_spaces

    def update_state(self, states) -> None:
        """Update desired energy need state based on current iteration."""
        if self.ts is not None:
            if self.iteration < len(self.ts):
                # TODO VP 2026.01.14. : Update with actual column name from CSV
                # desired_energy = float(self.ts.iloc[int(self.iteration)]["energy_need_normalized"])
                desired_energy = 0.0  # Placeholder
            else:
                # desired_energy = float(self.ts.iloc[-1]["energy_need_normalized"])
                desired_energy = 0.0  # Placeholder
        else:
            current_sim_hour = states.get("sim_hour", np.zeros(shape=(1,), dtype=np.float32))[0]
            # Apply a simple time-based energy need profile if no CSV data is provided
            if current_sim_hour < 6:
                desired_energy = 0.2  # Low demand during night
            elif current_sim_hour < 9:
                desired_energy = 0.6  # Morning peak
            elif current_sim_hour < 17:
                desired_energy = 0.4  # Daytime moderate
            elif current_sim_hour < 21:
                desired_energy = 0.8  # Evening peak
            elif current_sim_hour < 24:
                desired_energy = 0.3  # Evening low
            else:
                desired_energy = 0.2  # Default

        # Ensure float32 dtype for all updates
        desired_energy = np.float32(desired_energy)
        states["desired_energy_need"][0] = desired_energy


# Register DesiredUserEnergyNeed with the component registry
ComponentRegistry.register('statesource', DesiredUserEnergyNeed)
