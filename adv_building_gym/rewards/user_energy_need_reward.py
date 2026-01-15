import numpy as np

from .base import RewardFunction


class UserEnergyNeedReward(RewardFunction):
    """User energy need matching reward function.

    Rewards the agent for matching actual energy production/consumption
    to the desired user energy need. Positive actions produce energy,
    negative actions consume energy.
    """

    def __init__(self, weight: float, name: str = "user_energy_need_reward") -> None:
        super().__init__(weight, name)

    def get_reward(self, actions, states) -> float:
        """Calculate reward based on meeting desired energy need.

        Penalizes underproduction but does not penalize overproduction.

        Args:
            actions: Dictionary of actions (positive = produce, negative = consume)
            states: Dictionary containing 'desired_energy_need' state

        Returns:
            Reward value (higher when actual energy meets or exceeds desired need)
        """
        # Get desired energy need from states
        desired_energy = float(states.get("desired_energy_need", [0.0])[0])

        # Sum all energy-related actions (positive = produce, negative = consume)
        actual_energy = sum(
            float(np.atleast_1d(actions[k])[0])
            for k in actions.keys()
            if "_action" in k
        )

        # Only penalize when underproducing, not when overproducing
        if actual_energy >= desired_energy:
            # No penalty for meeting or exceeding energy need
            reward = 1.0
        else:
            # Penalize based on shortfall (desired - actual)
            shortfall = desired_energy - actual_energy
            reward = np.exp(-shortfall)

        return float(self.weight * reward)
