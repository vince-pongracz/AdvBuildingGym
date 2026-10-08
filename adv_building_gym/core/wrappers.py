"""Gymnasium action-space wrappers for AdvBuildingGym.

FlattenAction (Dict → flat Box) + RescaleAction give the policy the [-1, 1] flat
interface RL libs expect while the env keeps named Dict actions internally.

Wrapper chain (outermost first)::

    RescaleAction(min=-1, max=1)   # policy sees Box(-1, 1)
      FlattenAction                # env sees Dict with real bounds
        AdvBuildingGym             # native Dict action space
"""

import gymnasium
import numpy as np
from gymnasium.spaces.utils import flatten_space, unflatten
from gymnasium.wrappers import RescaleAction


class FlattenAction(gymnasium.ActionWrapper):
    """Flatten a Dict action space into a single Box (via flatten_space); 
    incoming flat actions are unflattened back to a Dict for the wrapped env.
    """

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        # Store original Dict space for unflattening
        self._dict_action_space = env.action_space
        # Expose flat Box with real per-component bounds
        self.action_space = flatten_space(env.action_space)

    def action(self, action: np.ndarray) -> dict:
        """Convert flat array to Dict action."""
        return unflatten(
            self._dict_action_space,
            np.asarray(action, dtype=np.float32),
        )


def wrap_action_space(env: gymnasium.Env) -> gymnasium.Env:
    """FlattenAction + RescaleAction: policy sees Box(-1, 1), env sees real Dict bounds.

    Pure-Gymnasium so both Ray and SB3 drivers can reuse it.
    """
    env = FlattenAction(env)
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    return env
