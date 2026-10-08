from typing import ClassVar, Set

from adv_building_gym.components.registry import Serializable


class RewardFunction(Serializable):
    """Base class for reward functions."""

    # Inherited from Serializable; redeclared here as empty-set defaults.
    _context_params: ClassVar[Set[str]] = set()
    _exclude_params: ClassVar[Set[str]] = set()

    def __init__(self, weight: float, name: str = "default") -> None:
        self.weight = weight
        self.name = name

    def get_reward(self, actions, state, next_state, info: dict) -> float:
        """Reward over the (s, a, s') transition.

        ``state`` (s): inputs the agent acted under (price, setpoints, EV schedule, time).
        ``next_state`` (s'): action outcomes (indoor temp, battery/EV SoC). ``info``:
        shared raw values (net_power_kW, EV params). Returns the weighted reward.
        """
        raise NotImplementedError()

    def get_01_reward(self, actions, state, next_state, info: dict) -> float:
        """Reward on a 0-1 scale; args as ``get_reward``. Returns the weighted reward."""
        raise NotImplementedError()

    def on_reset(self, states, info: dict) -> None:
        """Once per episode after initial obs are populated; capture per-episode
        baselines (e.g. starting SoC) for ``get_reward``. Default: no-op.
        """
        return None

    def should_terminate(self, actions, state, next_state, info: dict) -> bool:
        """True if this reward judges the episode should end this step.

        Judges the (s, a, s') transition (usually ``next_state``: comfort breach,
        EV session failure). Run in a termination pre-pass *before* ``get_reward``,
        so ``info["terminated"]`` is correct when rewards run. Must be pure (no info
        mutation, no state change); ``get_reward`` applies the penalty.
        Default: False. Override for hard bands (operator over-limit, comfort, EV failure).
        """
        return False
