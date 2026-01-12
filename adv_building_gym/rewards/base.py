from abc import ABC


class RewardFunction(ABC):
    """Base class for reward functions."""

    def __init__(self, weight: float, name: str = "default") -> None:
        self.weight = weight
        self.name = name

    def get_reward(self, actions, states) -> float:
        """Calculate reward based on actions and states.

        Args:
            actions: Dictionary of actions taken by the agent
            states: Dictionary of current environment states

        Returns:
            Calculated reward value
        """
        raise NotImplementedError()
