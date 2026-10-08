"""V0 alias of :class:`ActionSmoothnessReward` — thin subclass reusing the V1 logic;
only the class name and default ``name`` differ, so V0 configs can reference it.
"""

from ..v1.action_smoothness_reward import ActionSmoothnessReward
from adv_building_gym.components.registry import ComponentRegistry


class ActionSmoothnessRewardV0(ActionSmoothnessReward):
    """Alias of :class:`ActionSmoothnessReward` for the V0 reward set."""

    def __init__(self, weight: float, name: str = "action_smoothness_v0", **kwargs) -> None:
        super().__init__(weight=weight, name=name, **kwargs)


ComponentRegistry.register('reward', ActionSmoothnessRewardV0)
