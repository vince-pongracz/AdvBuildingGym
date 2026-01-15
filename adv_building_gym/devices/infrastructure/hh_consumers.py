import logging

from .base import Infrastructure

logger = logging.getLogger(__name__)

# TODO VP 2026.01.07. : Decide what to do with this class...

class HouseholdEnergyConsumers(Infrastructure):
    """Household consumers infrastructure component."""

    def __init__(self, name: str, Q_electric_max: float) -> None:
        super().__init__(name, Q_electric_max)

    def setup_spaces(self,
                     state_spaces,
                     action_spaces
                     ):
        pass

    def get_electric_consumption(self, actions) -> float:
        """Get current electric energy consumption from household consumers.

        HouseholdEnergyConsumers represent baseline electricity consumption (non-controllable load).

        Returns:
            Constant baseline consumption Q_electric_max
        """
        return self.Q_electric_max
