"""Warn when the SAC entropy temperature (alpha) collapses toward zero.

If SAC's auto-tuned ``alpha`` runs away to ~0, exploration dies and the policy freezes.
This surfaces it in the SLURM logs immediately. No-op when ``alpha`` is absent (PPO/DreamerV3).
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
    """Factory → ``on_train_result`` that warns once when SAC alpha collapses below
    ``warn_threshold`` (and once on recovery)."""
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
