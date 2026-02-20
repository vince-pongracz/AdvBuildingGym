import logging

from .base import Infrastructure
from adv_building_gym.config.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)

# TODO VP 2026.02.20. : Use this infrastructure element as a passive infrastructure, depending on a statesource, producing actions. 
# Use it similarly to the SolarPanel, but with a fixed consumption instead of production. 
# This way we can have a non-controllable load in the environment, which the agent has to learn to work with. 
# We can also use it to test if the agent learns to minimize energy consumption when we give it a reward for that.

class HouseholdEnergyConsumers(Infrastructure):
    """Household consumers infrastructure component."""

    def __init__(self, name: str, Q_electric_max: float) -> None:
        super().__init__(name, Q_electric_max)

    def setup_spaces(self, state_spaces, action_spaces):
        pass

    def get_electric_consumption(self, actions) -> float:
        """Get current electric energy consumption from household consumers.

        HouseholdEnergyConsumers represent baseline electricity consumption (non-controllable load).

        Returns:
            Constant baseline consumption Q_electric_max
        """
        return self.Q_electric_max


# Register HouseholdEnergyConsumers with the component registry
ComponentRegistry.register('infrastructure', HouseholdEnergyConsumers)
