import logging

import numpy as np
from gymnasium.spaces import Box

from .base import Infrastructure

logger = logging.getLogger(__name__)


class HP(Infrastructure):
    """Heat Pump infrastructure component."""

    def __init__(self,
                 name: str,
                 Q_electric_max: float,
                 Q_hp_max: float,
                 K: float,
                 mC: float,
                 cop_heat: float = 1.0,
                 cop_cool: float = 1.0,
                 ) -> None:
        super().__init__(name, Q_electric_max)

        self.cop_heat = cop_heat  # [-] heating COP
        self.cop_cool = cop_cool  # [-] cooling COP
        self.Q_hp_max = Q_hp_max
        self.timestep = 300 # a timestep is 300 secs -- # TODO VP 2026.01.07. : extract this to config, it should be configurable
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

        # Determine mode: cooling (<0.4), heating (>0.6), or no action ([0.4, 0.6])
        if mode < 0.4:
            # Cooling mode
            cop = self.cop_cool
            q_hp = energy * self.Q_hp_max / cop  # negative energy -> negative q_hp (cooling)
        elif mode > 0.6:
            # Heating mode
            cop = self.cop_heat
            q_hp = energy * self.Q_hp_max / cop  # negative energy -> negative q_hp, but sign convention may need review
        else:
            # No action zone [0.4, 0.6]
            q_hp = 0.0
            # Set also the energy part to 0 in this case -- at rewards it is useful to have the real actions
            actions["HP_action"][0] = 0.0

        # Note: Heat loss Q_transfer is now handled by BuildingHeatLoss datasource
        # HP only applies its heating/cooling effect
        # TODO VP 2026.01.08. : add reference to this model and computation
        dTemp = 0.001 * self.timestep * q_hp / self.mC

        self.temp_norm_in -= dTemp
        self.temp_norm_in_change = dTemp

    def update_state(self, states) -> None:
        new_temp = states["temp_norm_in"][0] + self.temp_norm_in_change
        # Clip to observation space bounds and ensure float32
        states["temp_norm_in"][0] = np.float32(np.clip(new_temp, -1.0, 1.0))
