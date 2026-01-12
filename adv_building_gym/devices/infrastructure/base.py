import logging
from typing import Dict

from adv_building_gym.utils import EnvSyncInterface

logger = logging.getLogger(__name__)


class Infrastructure(EnvSyncInterface):
    """Base class for infrastructure components in the building environment."""

    def __init__(self,
                 name: str,
                 Q_electric_max: float
                 ) -> None:
        super().__init__()

        self.name = name
        self.Q_electric_max = Q_electric_max  # ~ power consumption max

    def setup_spaces(self,
                     state_spaces,
                     action_spaces
                     ):
        """Setup observation and action spaces. Implement in derived classes."""
        return state_spaces, action_spaces

    def set_target(self, target: float) -> None:
        """Set target for infrastructure component."""
        pass

    def exec_action(self, actions, states) -> None:
        """Execute action of the infrastructure.

        Args:
            actions (Dict): contains actions
            states (Dict): should be treated as immutable, holds information for the action execution
        """
        pass

    def update_state(self, states: Dict) -> None:
        """Update state based on current iteration. Implement in derived classes."""
        pass
