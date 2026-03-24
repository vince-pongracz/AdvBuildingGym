"""Gymnasium action-space wrappers for AdvBuildingGym.

FlattenAction converts a Dict action space into a flat Box, complementing
Gymnasium's built-in FlattenObservation wrapper.  Composed with
RescaleAction it provides the standard [-1, 1] flat interface that
RL libraries expect while letting the environment work with named
Dict actions internally.

Wrapper chain (outermost first)::

    RescaleAction(min=-1, max=1)   # policy sees Box(-1, 1)
      FlattenAction                # env sees Dict with real bounds
        AdvBuildingGym             # native Dict action space
"""

import gymnasium
import numpy as np
from gymnasium.spaces.utils import flatten_space, unflatten


class FlattenAction(gymnasium.ActionWrapper):
    """Flatten a Dict action space into a single Box.

    Uses gymnasium.spaces.utils.flatten_space to produce a flat Box whose
    bounds match the concatenated per-key bounds of the original Dict.
    Incoming flat actions are unflattened back to a Dict before being
    forwarded to the wrapped environment.
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
