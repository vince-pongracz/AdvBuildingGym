import logging
from collections import OrderedDict
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from ..base import StateSource
from adv_building_gym.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)


class OperatorEnergyControl(StateSource):
    """Constant grid-operator power limit, configured at construction time.

    Publishes a fixed ``max_power_kW`` every step as
    ``ctxt_operator_max_power_kW`` (raw kW). The reward function reads
    that ctxt key, so the limit has a single source of truth in the env
    config.

    No CSV is loaded — the limit is a static contract from the grid
    operator and changes only via env config edits. If a time-varying
    profile is ever required, that's a separate concern (e.g. a
    ``OperatorEnergyControlFromCSV`` sibling source).
    """

    _exclude_params: ClassVar[Set[str]] = {'iteration'}

    def __init__(self, name: str, max_power_kW: float) -> None:
        """
        Args:
            name: Datasource identifier.
            max_power_kW: Maximum allowed grid power draw in kW.
        """
        if max_power_kW <= 0:
            raise ValueError(f"OperatorEnergyControl '{name}': max_power_kW must be > 0, got {max_power_kW}.")
        super().__init__(name, ds_path=None)
        self.max_power_kW: float = float(max_power_kW)
        logger.info("OperatorEnergyControl '%s': constant limit = %.3f kW", name, self.max_power_kW)

    def setup_spaces(self,
                    state_spaces: OrderedDict,
                    action_spaces: OrderedDict) -> tuple[OrderedDict, OrderedDict]:
        if "ctxt_operator_max_power_kW" not in state_spaces:
            state_spaces["ctxt_operator_max_power_kW"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        if "raw_sim_hour" not in state_spaces:
            state_spaces["raw_sim_hour"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        return state_spaces, action_spaces

    def update_state(self, states, info=None) -> None:
        states["ctxt_operator_max_power_kW"][0] = np.float32(self.max_power_kW)


# Register OperatorEnergyControl with the component registry
ComponentRegistry.register('statesource', OperatorEnergyControl)
