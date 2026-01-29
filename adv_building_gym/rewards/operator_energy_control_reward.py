import logging
import numpy as np
from typing import ClassVar, List, Set

from .base import RewardFunction
from adv_building_gym.config.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)


class OperatorEnergyControlReward(RewardFunction):
    """
    Reward function for respecting grid operator energy consumption limits.

    This reward encourages the agent to keep total grid power consumption below
    the operator-specified limit (operator_energy_max). The reward is scaled
    based on how far the actual consumption is from the limit:
    - Full reward (1.0) when consumption is at 0
    - Zero reward (0.0) when consumption equals the limit
    - Negative penalty when consumption exceeds the limit

    Linear reward calculation:
        reward = (operator_energy_max - grid_power_kW) / operator_energy_max
    """

    # infrastructures comes from context (the Config's infras list)
    _context_params: ClassVar[Set[str]] = {'infrastructures'}

    def __init__(self,
                 infrastructures: List,
                 weight: float,
                 max_power_kW: float = 10.0,
                 name: str = "operator_energy_control_reward",
                 harsh_penalty: float = -10.0
                 ) -> None:
        """
        Initialize OperatorEnergyControlReward.

        Args:
            infrastructures: List of Infrastructure objects to query for power consumption
            weight: Reward weight (scaling factor)
            max_power_kW: Maximum power limit in kW for denormalization (default: 10.0 kW)
            name: Reward function name
        """
        super().__init__(weight, name)
        self.infrastructures = infrastructures
        self.max_power_kW = max_power_kW
        self.harsh_penalty = harsh_penalty

    def get_reward(self, actions, states) -> float:
        """
        Calculate reward based on grid power consumption vs operator limit.

        This method:
        1. Calculates total grid power by calling get_electric_consumption() on all infrastructures
        2. Stores grid_power_kW in states for observability
        3. Compares against operator_energy_max limit
        4. Returns scaled reward

        Args:
            actions: Dictionary of actions taken by infrastructures
            states: Dictionary containing "operator_energy_max" (normalized limit [0, 1])

        Returns:
            Scaled reward value
        """
        # Calculate total grid E consumption by summing all infrastructure consumption
        grid_power_kW = 0.0
        for infra in self.infrastructures:
            grid_power_kW += infra.get_electric_consumption(actions)

        # Store in state for observability (e.g., logging, other reward functions)
        # TODO VP 2026.01.14. : Store it in info instead?
        # states["grid_power_kW"] = np.array([grid_power_kW], dtype=np.float32)

        # Get normalized operator limit from state [0, 1]
        operator_limit_norm = float(states.get("operator_energy_max", np.array([1.0]))[0])

        # Denormalize to actual kW
        operator_limit_kW = operator_limit_norm * self.max_power_kW

        # Calculate scaled distance-based reward
        # reward = 1.0 when grid_power = 0
        # reward = 0.0 when grid_power = operator_limit
        # reward < 0.0 when grid_power > operator_limit (penalty)
        if operator_limit_kW > 0:
            reward = (operator_limit_kW - grid_power_kW) / operator_limit_kW
        else:
            # If limit is 0, any consumption is a violation
            reward = -1.0 if grid_power_kW > 0 else 0.0

        # Clip to reasonable range to avoid extreme penalties
        clipped_reward = np.clip(reward, self.harsh_penalty, 1.0)

        return float(self.weight * clipped_reward)


# Register OperatorEnergyControlReward with the component registry
ComponentRegistry.register('reward', OperatorEnergyControlReward)