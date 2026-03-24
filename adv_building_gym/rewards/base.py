from typing import Any, ClassVar, Dict, Set, Type, TypeVar

from adv_building_gym.config.utils.serializable import Serializable, ComponentRegistry

T = TypeVar('T', bound='RewardFunction')


class RewardFunction(Serializable):
    """Base class for reward functions."""

    # Inherited from Serializable; redeclared here as empty-set defaults.
    # Subclasses override as needed (e.g. EconomicReward sets
    # _context_params = {'infrastructures'}).
    _context_params: ClassVar[Set[str]] = set()
    _exclude_params: ClassVar[Set[str]] = set()

    # Maximum raw (unweighted) reward this function can return per step.
    # Subclasses whose raw output exceeds 1.0 must override this.
    max_reward: float = 1.0

    def __init__(self, weight: float, name: str = "default") -> None:
        self.weight = weight
        self.name = name

    def get_reward(self, actions, states) -> tuple[float, float]:
        """Calculate reward and step-wise maximum for this reward function.

        Args:
            actions: Dictionary of actions taken by the agent.
            states: Dictionary of current environment states.

        Returns:
            Tuple of (reward, max_reward_for_this_step).
            The max reward may be state-dependent (e.g. 0 when the EV
            is disconnected).
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
