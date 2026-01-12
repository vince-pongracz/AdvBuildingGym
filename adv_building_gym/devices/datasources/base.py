import logging

import pandas as pd

from adv_building_gym.utils import EnvSyncInterface

logger = logging.getLogger(__name__)


class DataSource(EnvSyncInterface):
    """Base class for data sources in the building environment."""

    def __init__(self,
                 name: str,
                 ds_path: str | None = None,
                 ) -> None:
        super().__init__()

        self.name = name
        if ds_path is not None:
            self.ts = pd.read_csv(ds_path) 
            """Time series"""
        else:
            self.ts = None

    def setup_spaces(self,
                     state_spaces,
                     action_spaces
                     ):
        """Setup observation and action spaces. Implement in derived classes."""
        return state_spaces, action_spaces

    def update_state(self, states) -> None:
        """Update state based on current iteration. Implement in derived classes."""
        pass
