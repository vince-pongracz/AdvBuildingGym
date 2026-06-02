"""Promote the latest evaluation return to a top-level result key for checkpoint scoring.

Why this exists
---------------
Ray Tune ranks retained checkpoints by ``CheckpointConfig.checkpoint_score_attribute``,
but its insertion gate tests ``score_attr in checkpoint_result.metrics`` against the
*un-flattened* (nested) result dict, while the scorer flattens first
(ray/train/_internal/checkpoint_manager.py — ``register_checkpoint`` vs
``_get_checkpoint_score``). A slashed key such as
``evaluation/env_runners/episode_return_mean`` is therefore never found at the top
level of the nested dict, so checkpoints silently fall back to time-ordered retention
and ``num_to_keep`` keeps the most *recent* N instead of the *best* N.

Fix: expose the eval return as a flat, top-level scalar (no ``/``) that the membership
test can find. The eval metric is absent from the raw on_train_result on non-eval
iterations (``evaluation_interval > 1``), so we carry the last completed eval forward —
each checkpoint is then scored by the most recent evaluation. 
If ``checkpoint_frequency == evaluation_interval``, checkpoints land on fresh-eval
iterations and the carried value is never stale.
"""

import logging
import math

logger = logging.getLogger(__name__)

# Top-level, non-slashed result key consumed by CheckpointConfig.checkpoint_score_attribute.
EVAL_SCORE_KEY = "best_eval_return"

# Single source of truth for CheckpointConfig(num_to_keep=...) across the Ray drivers,
# so the retention-mirror logging below matches the real CheckpointManager decision.
CHECKPOINT_NUM_TO_KEEP = 3


def _log_checkpoint_decision(state: dict, score: float, iteration: int,
                            num_to_keep: int, score_key: str) -> None:
    """Mirror Ray's _CheckpointManager retention to log the save/keep decision.

    Ray keeps the ``num_to_keep`` highest-scoring checkpoints and always force-keeps
    the latest (even when it scores below all kept ones). We replicate that bookkeeping
    here purely for logging — the real decision is still made by the CheckpointManager
    from the published ``score_key``. Runs once per fresh-eval iteration which, because
    checkpoint_frequency == evaluation_interval, coincides with each checkpoint save.
    """
    kept = state["kept"]
    prev_best = max((s for s, _ in kept), default=float("-inf"))

    new_entry = (score, iteration)
    ranked = sorted(kept + [new_entry], key=lambda item: item[0])  # ascending, worst first
    keep_on_merit = ranked[-num_to_keep:]                          # the N highest scores
    is_kept_on_merit = new_entry in keep_on_merit
    worst_kept = min(s for s, _ in keep_on_merit)

    # Real manager retains best-N plus the force-protected latest (≤ N+1 entries).
    retained = list(keep_on_merit)
    if new_entry not in retained:
        retained.append(new_entry)
    state["kept"] = retained

    if score > prev_best:
        logger.info(
            "[ckpt] iter %d: NEW BEST checkpoint — %s=%.4f beats previous best %.4f; "
            "saved and retained on merit.",
            iteration, score_key, score, prev_best,
        )
    elif is_kept_on_merit:
        logger.info(
            "[ckpt] iter %d: new top-%d checkpoint — %s=%.4f enters the best set "
            "(best=%.4f, worst kept=%.4f); saved and retained on merit.",
            iteration, num_to_keep, score_key, score, prev_best, worst_kept,
        )
    else:
        logger.info(
            "[ckpt] iter %d: %s=%.4f is BELOW the retained best-%d (worst kept=%.4f) — "
            "not retained on merit; kept only as the latest checkpoint and evicted at the "
            "next checkpoint unless it improves.",
            iteration, score_key, score, num_to_keep, worst_kept,
        )

    # The latest checkpoint is always auto-saved by Ray regardless of its score.
    logger.info("[ckpt] iter %d: latest checkpoint auto-saved (%s=%.4f).",
                iteration, score_key, score)


def create_eval_score_promote_on_train_result_cb(
    checkpoint_interval: int,
    source: tuple[str, ...] = ("evaluation", "env_runners", "episode_return_mean"),
    score_key: str = EVAL_SCORE_KEY,
    num_to_keep: int = CHECKPOINT_NUM_TO_KEEP,
):
    """Factory returning an ``on_train_result`` callable that mirrors the eval return
    into a flat top-level result key (carried forward on non-eval iterations) and logs
    the resulting checkpoint save/keep/evict decision.

    The retention log fires on the iterations where Tune actually saves a checkpoint —
    ``training_iteration % checkpoint_interval == 0`` — NOT merely when an eval metric
    is present. RLlib carries the last eval forward in the result dict on non-eval
    iterations, so an eval-presence trigger would fire every iteration and drift out of
    sync with the real saves. ``checkpoint_interval`` must equal ``checkpoint_frequency``
    (which is tied to ``evaluation_interval``).

    Args:
        checkpoint_interval: Iterations between checkpoint saves (== checkpoint_frequency).
        source: Nested path to the eval metric inside the result dict.
        score_key: Flat top-level key written to the result dict each iteration.
        num_to_keep: Mirrors CheckpointConfig.num_to_keep for the retention logging.
    """
    interval = max(1, int(checkpoint_interval))
    # kept: list[(score, iteration)] mirroring the CheckpointManager retained set.
    state = {"last": float("-inf"), "kept": []}

    def on_train_result(*, algorithm, result: dict, **kwargs) -> None:
        node = result
        for key in source:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        value = node if isinstance(node, (int, float)) and not isinstance(node, bool) else None
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            state["last"] = float(value)  # latest eval (fresh on eval iters, carried otherwise)

        # Always publish a flat scalar so whatever iteration Tune checkpoints on
        # carries a comparable score.
        result[score_key] = state["last"]

        # Log only on the iterations Tune actually saves a checkpoint, so the log
        # is in lock-step with the real CheckpointManager registrations.
        iteration = int(result.get("training_iteration", 0) or 0)
        if iteration > 0 and iteration % interval == 0:
            _log_checkpoint_decision(state, state["last"], iteration, num_to_keep, score_key)

    return on_train_result
