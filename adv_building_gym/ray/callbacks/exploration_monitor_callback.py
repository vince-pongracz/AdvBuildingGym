"""Warn when the SAC entropy temperature (alpha) collapses toward zero.

SAC auto-tunes the entropy coefficient ``alpha`` to hold policy entropy near
``target_entropy``. If that temperature loop runs away — e.g. under large /
high-variance returns — ``alpha`` decays to ~0, the entropy bonus vanishes,
exploration dies, and the policy freezes into a deterministic corner it cannot
escape. This surfaces that failure in the SLURM logs the moment it happens
instead of only being visible post-hoc in result.json.

PPO and DreamerV3 have no such key, so the monitor is a no-op when ``alpha`` is
absent from the result dict.
"""

import logging

logger = logging.getLogger(__name__)

# alpha below this is treated as collapsed. Healthy SAC alpha is ~0.1–1.0; runs
# that lost exploration hit ~1e-4 and below (log_alpha -> -inf).
ALPHA_COLLAPSE_THRESHOLD = 1e-3


def create_exploration_monitor_on_train_result_cb(
    alpha_path: tuple[str, ...] = ("learners", "default_policy", "alpha_value"),
    warn_threshold: float = ALPHA_COLLAPSE_THRESHOLD,
):
    """Factory returning an ``on_train_result`` callable that warns once when the
    SAC temperature collapses below ``warn_threshold`` (and once if it recovers).

    Args:
        alpha_path: Nested path to the temperature value inside the result dict.
        warn_threshold: alpha below this is considered collapsed.
    """
    state = {"collapsed": False}

    def on_train_result(*, algorithm, result: dict, **kwargs) -> None:
        node = result
        for key in alpha_path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        alpha = node if isinstance(node, (int, float)) and not isinstance(node, bool) else None
        if alpha is None:
            return  # not a SAC run — no entropy temperature to monitor

        iteration = int(result.get("training_iteration", 0) or 0)

        if alpha < warn_threshold and not state["collapsed"]:
            state["collapsed"] = True
            logger.warning(
                "[exploration] SAC entropy temperature COLLAPSED at iter %d: alpha=%.3e < %.0e. "
                "Exploration is effectively off — the policy will freeze and likely cannot "
                "recover. Check reward scale/variance, target_entropy, and alpha_lr.",
                iteration, alpha, warn_threshold,
            )
        elif alpha >= warn_threshold and state["collapsed"]:
            state["collapsed"] = False
            logger.warning(
                "[exploration] SAC alpha recovered above %.0e at iter %d: alpha=%.3e.",
                warn_threshold, iteration, alpha,
            )

    return on_train_result
