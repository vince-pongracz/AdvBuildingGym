import logging

from .base import Infrastructure

logger = logging.getLogger(__name__)

# TODO VP 2026.01.07. : Decide what to do with this class...

class HHConsumers(Infrastructure):
    """Household consumers infrastructure component."""

    def __init__(self, name: str, Q_electric_max: float) -> None:
        super().__init__(name, Q_electric_max)

    def setup_spaces(self,
                     state_spaces,
                     action_spaces
                     ):
        pass
