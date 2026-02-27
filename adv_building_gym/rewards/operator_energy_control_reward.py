import logging
import numpy as np
from typing import ClassVar, List, Set

from .base import RewardFunction
from adv_building_gym.config.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)


class OperatorEnergyControlReward(RewardFunction):
    """Reward function for respecting grid operator energy consumption limits.

    Three-zone reward based on consumption ratio (grid_power / operator_limit):

    - Below 90% of limit: full reward (1.0)
    - 90%-100% of limit: exponential decay from 1.0 towards 0
      using exp(-5 * (ratio - 0.9) / 0.1), where the scale factor 5
      gives exp(-5) ~ 0.007 at the limit boundary
    - Above limit: harsh_penalty (default -10.0)
    """

    # infrastructures comes from context (the Config's infras list)
    _context_params: ClassVar[Set[str]] = {'infrastructures'}

    # Scale factor for the exponential decay in the transition zone.
    # exp(-5) ~ 0.007, so reward nearly reaches 0 right at the limit.
    _DECAY_SCALE: float = 5.0

    def __init__(self,
                 infrastructures: List,
                 weight: float,
                 max_power_kW: float = 10.0,
                 name: str = "operator_energy_control_reward",
                 harsh_penalty: float = -10.0,
                 soft_threshold_pct: float = 0.9,
                 ) -> None:
        """Initialize OperatorEnergyControlReward.

        Args:
            infrastructures: List of Infrastructure objects to query for power consumption.
            weight: Reward weight (scaling factor).
            max_power_kW: Maximum power in kW for denormalization (default: 10.0 kW).
            name: Reward function name.
            harsh_penalty: Flat penalty when consumption exceeds the operator limit.
            soft_threshold_pct: Fraction of operator limit below which reward is 1.0
                (default 0.9 = 90%). Must be in (0, 1).
        """
        super().__init__(weight, name)
        self.infrastructures = infrastructures
        self.max_power_kW = max_power_kW
        self.harsh_penalty = harsh_penalty
        if not 0.0 < soft_threshold_pct < 1.0:
            raise ValueError("soft_threshold_pct must be in (0, 1)")
        self.soft_threshold_pct = soft_threshold_pct

    def get_reward(self, actions, states) -> float:
        """Calculate reward based on grid power consumption vs operator limit.

        Args:
            actions: Dictionary of actions taken by infrastructures.
            states: Dictionary containing "operator_energy_max" (normalized limit [0, 1]).

        Returns:
            Weighted reward value.
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

        if operator_limit_kW <= 0:
            # If limit is 0, any consumption is a violation
            reward = self.harsh_penalty if grid_power_kW > 0 else 1.0
            return float(self.weight * reward)

        ratio = grid_power_kW / operator_limit_kW

        if ratio <= self.soft_threshold_pct:
            # Below soft threshold: full reward
            reward = 1.0
        elif ratio <= 1.0:
            # Transition zone: exponential decay from 1.0 towards 0
            # At soft_threshold_pct: t=0 -> exp(0) = 1.0
            # At 1.0:               t=1 -> exp(-5) ~ 0.007
            t = (ratio - self.soft_threshold_pct) / (1.0 - self.soft_threshold_pct)
            reward = float(np.exp(-self._DECAY_SCALE * t))
        else:
            # Above operator limit: harsh penalty
            reward = self.harsh_penalty

        return float(self.weight * reward)


# Register OperatorEnergyControlReward with the component registry
ComponentRegistry.register('reward', OperatorEnergyControlReward)