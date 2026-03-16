import numpy as np

from .base import RewardFunction
from adv_building_gym.config.utils.serializable import ComponentRegistry


class ActionSmoothnessReward(RewardFunction):
    """Penalise large action changes between consecutive timesteps.

    For each action key, computes a per-key penalty in [-1, 0]:

        penalty_k = -||a_t - a_{t-1}||² / (4 * n_dims_k)

    where 4 * n_dims_k is the maximum possible squared L2 norm (each
    dimension changes by at most 2 in [-1, 1]).  The penalties are summed
    across all action keys, giving a raw reward in [-n_action_keys, 0].
    The final reward is ``weight * sum(penalties)``.

    ``max_reward = 0.0`` so the smoothness term contributes nothing to the
    achievable reward ceiling in reward_rate, but pulls down achieved_reward
    when actions oscillate.

    The reward reads the most recent entry from the per-key action history
    window maintained by the environment in ``prev_{key}_hist``
    (shape ``(window, *action_shape)``).

    Reference: Higher-Order Action Regularisation for RL in Building Energy
    Management (NeurIPS 2025 UrbanAI Workshop)
    Link: https://arxiv.org/abs/2601.02061
    """
    # TODO VP 2026.03.14. : Read the paper
    # TODO VP 2026.03.14. : Fix reward ranges and adjust reward rate computations

    # max_reward = 0.0 means the best this reward can return is 0 (no penalty).
    # It contributes nothing to max_achievable in reward_rate, but pulls down
    # achieved_reward when actions oscillate.
    max_reward: float = 0.0

    def __init__(self, weight: float, name: str = "action_smoothness") -> None:
        super().__init__(weight, name)
        # Number of action keys observed so far; used to compute min_reward
        self._n_action_keys: int = 0

    @property
    def min_reward(self) -> float:
        """Minimum raw (unweighted) reward: -1 per action key."""
        return -self._n_action_keys

    def get_reward(self, actions: dict, states: dict) -> float:
        penalties: list[float] = []

        for key, current_action in actions.items():
            hist_key = f"prev_{key}_hist"
            if hist_key not in states:
                continue

            a_current = np.atleast_1d(current_action).astype(np.float32)
            a_history = states[hist_key]
            # Most recent previous action is the last row in the history window
            prev_a = a_history[-1]

            a_diff = a_current - prev_a
            n_dims = a_current.size
            # dot(a_diff, a_diff) = ||a_t - a_{t-1}||² (squared L2 norm).
            # Each dimension is in [-1, 1], so max change per dim is 2,
            # max squared change per dim is 4, max across n_dims is 4*n_dims.
            # Dividing normalises to [0, 1] ("fraction of max possible change"),
            # the leading minus flips it to [-1, 0] as a penalty.
            penalties.append(-float(np.dot(a_diff, a_diff)) / (4.0 * n_dims))

        if not penalties:
            return 0.0

        # Update action key count on first call (stable across episode)
        if self._n_action_keys == 0:
            self._n_action_keys = len(penalties)

        # Sum of per-key penalties; range is [-n_action_keys, 0]
        action_diff_penalty = sum(penalties)

        return float(self.weight * action_diff_penalty)


# Register with the component registry
ComponentRegistry.register('reward', ActionSmoothnessReward)
