"""V0 wrapper around :class:`ActionSmoothnessReward`.

Per user instruction "the ActionSmoothness logic is good as is", this V0
class is a thin subclass that reuses the V1 implementation verbatim. Only
the class name and default ``name`` parameter differ, so the registry
exposes a ``ActionSmoothnessRewardV0`` entry that can be referenced from
V0 reward configs without duplicating the spectral-analysis logic.
"""

from ..v1.action_smoothness_reward import ActionSmoothnessReward
from adv_building_gym.components.registry import ComponentRegistry


class ActionSmoothnessRewardV0(ActionSmoothnessReward):
    """Alias of :class:`ActionSmoothnessReward` for the V0 reward set."""

    def __init__(self, weight: float, name: str = "action_smoothness_v0", **kwargs) -> None:
        super().__init__(weight=weight, name=name, **kwargs)


ComponentRegistry.register('reward', ActionSmoothnessRewardV0)
