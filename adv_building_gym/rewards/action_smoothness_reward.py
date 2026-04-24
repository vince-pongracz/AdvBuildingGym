import numpy as np

from .base import RewardFunction
from adv_building_gym.utils.serializable import ComponentRegistry


class ActionSmoothnessReward(RewardFunction):
    """Penalise oscillatory action patterns over a rolling history window.

    Instead of penalising raw step-to-step changes (which also punishes
    justified ramps), this reward targets *direction reversals* weighted
    by acceleration magnitude.  A reversal occurs when consecutive first-
    differences flip sign — the hallmark of oscillation.

    For each action key and each dimension independently:

    1. Compute first-differences from the history window:
       ``d_i = a_{i+1} - a_i`` for consecutive pairs.
    2. For each pair of consecutive diffs ``(d_i, d_{i+1})``, check for
       a sign reversal: ``sign(d_i) * sign(d_{i+1}) < 0``.
    3. Where a reversal is detected, compute the second-difference
       (acceleration): ``accel = d_{i+1} - d_i``, and accumulate
       ``accel²``.
    4. Normalise by the maximum possible ``accel²`` (16 per dimension
       per pair, since each diff ∈ [-2, 2] and worst-case swing is 4).

    The per-key penalty is averaged across all checked pairs and lives
    in [-1, 0].  Summing across action keys and shifting by
    ``max_reward`` gives a raw reward in
    ``[max_reward - n_keys, max_reward]``.

    This design ensures:
    - Smooth ramps (all diffs same sign): zero penalty.
    - Single justified step-changes from steady state: zero penalty
      (``sign(0) * sign(x) = 0``, not negative).
    - Persistent oscillation: heavy penalty, proportional to amplitude.

    Reference: Higher-Order Action Regularisation for RL in Building
    Energy Management (NeurIPS 2025 UrbanAI Workshop)
    Link: https://arxiv.org/abs/2601.02061
    """
    
    # TODO VP 2026.04.22. : Rewrite docsstring

    max_reward: float = 0.3

    def __init__(self, weight: float, name: str = "action_smoothness") -> None:
        super().__init__(weight, name)
        self._n_action_keys: int = 0

    @property
    def min_reward(self) -> float:
        """Minimum raw (unweighted) reward: -1 per action key."""
        # TODO VP 2026.04.22. : Refactor this, add weigthing and -1 is not always the min.
        return -self._n_action_keys

    def get_reward(self, actions: dict, states: dict, info: dict | None = None) -> tuple[float, float]:
        max_step = self.weight * self.max_reward
        penalties: list[float] = []

        # Action history for the reward lives in the env's rolling deque,
        # published via info["action_history"]. The policy sees a separate
        # "<action_key>_prev" obs channel each step (optionally stacked by
        # StridedHistoryConnector); this reward uses the env deque directly.
        # Link: docs/hst_mgmt.md
        action_history = (info or {}).get("action_history")
        if not action_history:
            return 0.0, max_step

        for key, current_action in actions.items():
            a_current = np.atleast_1d(current_action).astype(np.float32)
            history_rows = [
                np.atleast_1d(snapshot[key]).astype(np.float32).reshape(-1)
                for snapshot in action_history
                if key in snapshot
            ]
            if not history_rows:
                continue

            # Build the full action sequence: deque entries (oldest-first)
            # followed by the just-taken action.  Shape: (len + 1, n_dims).
            sequence = np.vstack(history_rows + [a_current.reshape(-1)])

            # First differences: d[i] = sequence[i+1] - sequence[i]
            diffs = np.diff(sequence, axis=0)  # (window, n_dims)
            n_dims = diffs.shape[1]

            if diffs.shape[0] < 2:
                # Need at least 2 diffs to detect a reversal
                penalties.append(0.0)
                continue

            # Check consecutive diff pairs for per-dimension sign reversals.
            # sign(d_i) * sign(d_{i+1}) < 0  iff  genuine +/- flip.
            # Link: https://numpy.org/doc/stable/reference/generated/numpy.sign.html
            signs = np.sign(diffs)
            sign_products = signs[:-1] * signs[1:]  # (n_pairs, n_dims)

            # Acceleration (second difference) at each pair
            accels = np.diff(diffs, axis=0)  # (n_pairs, n_dims)

            # Mask: only penalise where a reversal occurred
            reversal_mask = sign_products < 0  # (n_pairs, n_dims)

            # Squared acceleration, zeroed where no reversal
            accel_sq = np.where(reversal_mask, accels ** 2, 0.0)

            # Max accel² per dimension per pair = 16 (swing from -2 to +2)
            n_pairs = accel_sq.shape[0]
            max_possible = 16.0 * n_dims * n_pairs

            # Penalty in [-1, 0]: fraction of worst-case oscillation
            penalty = -float(accel_sq.sum()) / max_possible
            penalties.append(penalty)

        if not penalties:
            return 0.0, max_step

        if self._n_action_keys == 0:
            self._n_action_keys = len(penalties)

        raw_reward = self.max_reward + sum(penalties)
        return float(self.weight * raw_reward), max_step


# Register with the component registry
ComponentRegistry.register('reward', ActionSmoothnessReward)
