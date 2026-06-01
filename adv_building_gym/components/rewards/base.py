from typing import ClassVar, Set

from adv_building_gym.components.registry import Serializable


class RewardFunction(Serializable):
    """Base class for reward functions."""

    # Inherited from Serializable; redeclared here as empty-set defaults.
    _context_params: ClassVar[Set[str]] = set()
    _exclude_params: ClassVar[Set[str]] = set()

    # Maximum raw (unweighted) reward this function can return per step.
    # Subclasses whose raw output exceeds 1.0 must override this.
    max_reward_in_step: float = 1.0

    def __init__(self, weight: float, name: str = "default") -> None:
        self.weight = weight
        self.name = name

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        """Calculate reward and step-wise maximum for this reward function.

        Args:
            actions: Dictionary of actions taken by the agent.
            states: Dictionary of current environment states.
            info: Shared inter-component dict with raw physical values
                (e.g. net_power_kW, EV charger params). Passed from the
                environment's ``_component_info``.

        Returns:
            Tuple of (reward, max_reward_for_this_step).
            The max reward may be state-dependent (e.g. 0 when the EV
            is disconnected).
        """
        raise NotImplementedError()
    
    def get_01_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        """Transform the reward to a 0-1 scale.

        Args:
            actions: Dictionary of actions taken by the agent.
            states: Dictionary of current environment states.
            info: Shared inter-component dict with raw physical values
                (e.g. net_power_kW, EV charger params). Passed from the
                environment's ``_component_info``.

        Returns:
            Tuple of (reward, max_reward_for_this_step).
            The max reward may be state-dependent (e.g. 0 when the EV
            is disconnected).
        """
        raise NotImplementedError()

    def on_reset(self, states, info: dict | None = None) -> None:
        """Called once per episode after the env has populated initial
        observations. Use to capture per-episode baselines (e.g. starting
        SoC) that ``get_reward`` will later compare against. Default: no-op.
        """
        return None

    def should_terminate(self, actions, states, info: dict | None = None) -> bool:
        """Return True if this reward judges the episode should end this step.

        Called by the env in a dedicated termination pre-pass *before*
        ``get_reward``, so the resulting ``info["terminated"]`` flag is
        already correct when rewards run. Replaces the earlier pattern
        where rewards mutated ``info["request_terminate"]`` mid-aggregation
        — that coupled rewards by ordering and hid side effects inside
        ``get_reward``. Must be pure (no info mutation, no internal state
        change); ``get_reward`` is responsible for the actual penalty.

        Default: False. Override in rewards that represent hard bands
        (operator over-limit, comfort breach, EV session failure, ...).
        """
        return False
