from typing import Any, ClassVar, Dict, Set, Type, TypeVar

from adv_building_gym.config.utils.serializable import Serializable, ComponentRegistry

T = TypeVar('T', bound='RewardFunction')


class RewardFunction(Serializable):
    """Base class for reward functions."""

    # Parameters derived from context
    _context_params: ClassVar[Set[str]] = set()

    # Internal state - never serialize
    _exclude_params: ClassVar[Set[str]] = set()

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

    @classmethod
    def from_dict(
        cls: Type[T],
        data: Dict[str, Any],
        context: Dict[str, Any] | None = None
    ) -> T:
        """
        Reconstruct a RewardFunction from a dictionary.

        Uses the ComponentRegistry to find the correct class by name,
        then constructs it with serialized data merged with context.

        Args:
            data: Dictionary containing 'class' key and constructor parameters
            context: Optional context with derived parameters (e.g., infrastructures)

        Returns:
            Reconstructed RewardFunction instance
        """
        class_name = data.get('class')
        if class_name is None:
            raise ValueError("Missing 'class' key in reward data")

        # Get the actual class from registry
        reward_class = ComponentRegistry.get('reward', class_name)

        # Build kwargs from data and context
        kwargs = reward_class._get_init_args(data, context)

        return reward_class(**kwargs)
