"""Per-iteration timer logging via on_train_result.

Surfaces RLlib's built-in iteration timers (``result["timers"]``) in the SLURM logs.
Keys absent on iterations where the phase didn't run, so read defensively.

EMA is disabled: timer ``Stats`` are pre-registered with ``reduce="mean", window=10000,
clear_on_reduce=True``, so each iter's ``compile()`` returns the within-iter per-call mean,
then clears. 
Set ``RAW_TIMERS = False`` to restore RLlib's default EMA.
"""

import logging
import time

from ray.rllib.utils.metrics import TIMERS
from ray.rllib.utils.metrics.stats import Stats

logger = logging.getLogger(__name__)

# Toggle EMA smoothing for the tracked timers.
# True  -> window=1 Stats, peek() == last raw per-call value.
# False -> default RLlib EMA (ema_coeff=0.01) — values smoothed across iters.
RAW_TIMERS = True

# RLlib timer keys (seconds; not all present every iter).
# - learner_update_timer / replay_buffer_sampling_timer absent during warmup.
# - synch_env_connectors only fires on env-runner state broadcast.
# - replay_buffer_sampling_timer is one sample() call inside the UTD loop; with RAW_TIMERS
#   the value is the within-iter MEAN (× sample_and_train_weight for the per-iter total).
_TRACKED_TIMER_KEYS = (
    "training_iteration",              # total wall time of one Algorithm.training() call (n training_step() calls)
    "training_step",                   # duration of a single Algorithm.training_step() call
    "env_runner_sampling_timer",       # env_runner_group.sample() — rollout collection duration on env runners
    "learner_update_timer",            # learner_group.update() — forward/backward/optimizer step duration on the learner(s)
    "replay_buffer_sampling_timer",    # one local_replay_buffer.sample() call (per-call value inside the UTD loop)
    "replay_buffer_add_data_timer",    # local_replay_buffer.add() — inserting freshly sampled episodes into the buffer
    "replay_buffer_update_prios_timer",# local_replay_buffer.update_priorities() — per priority writeback (no-op for uniform)
    "synch_env_connectors",            # broadcasting env-runner connector states (e.g. running mean stats) across runners
    "synch_weights",                   # env_runner_group.sync_weights() — pushing learner weights to env runners
    "restore_env_runners",             # bringing failed training env runners back before the training step
    "evaluation_iteration",            # full Algorithm.evaluate() call duration
    "synch_eval_env_connectors",       # broadcasting connector states across eval env runners
    "restore_eval_env_runners",        # bringing failed eval env runners back before evaluation
)


# big enough for every per-call push within one iter (~1500 calls); cleared each iter
# via clear_on_reduce=True, so a per-iter cap not a lifetime cap
_TIMER_WINDOW = 10000


def _force_raw_timer_stats(algorithm) -> None:
    """Replace EMA-mode timer Stats with within-iter-mean ones (idempotent).

    Pre-installs ``reduce="mean", window=_TIMER_WINDOW, clear_on_reduce=True`` under each
    timer key, so subsequent ``log_time`` pushes reduce to the within-iter mean at ``compile()``.
    Called every tick to survive state-restore/re-init; no-op when already in place.
    """
    metrics = getattr(algorithm, "metrics", None)
    if metrics is None:
        return
    for key in _TRACKED_TIMER_KEYS:
        flat_key = (TIMERS, key)
        try:
            cur = metrics._get_key(flat_key, key_error=False)
        except Exception:
            cur = None
        if (
            isinstance(cur, Stats)
            and getattr(cur, "_window", None) == _TIMER_WINDOW
            and getattr(cur, "_clear_on_reduce", False)
        ):
            continue
        try:
            metrics._set_key(
                flat_key,
                Stats(reduce="mean", window=_TIMER_WINDOW, clear_on_reduce=True),
            )
        except Exception as exc:
            logger.debug("Could not pre-register raw Stats for %s: %s", flat_key, exc)


def create_iter_timing_on_train_result_cb():
    """Factory returning an ``on_train_result`` callable that logs RLlib timers."""

    state = {"last_call": None}

    def on_train_result(*, algorithm, result: dict, **kwargs) -> None:
        if RAW_TIMERS:
            _force_raw_timer_stats(algorithm)

        iteration: int = result.get("training_iteration", 0)
        timers: dict = result.get("timers", {}) or {}

        now = time.perf_counter()
        last = state["last_call"]
        wall_iter_ms = (now - last) * 1000.0 if last is not None else None
        state["last_call"] = now

        parts = []
        if wall_iter_ms is None:
            parts.append("wall_iter=NA")
        else:
            parts.append(f"wall_iter={wall_iter_ms:.1f} ms")

        for timer_key in _TRACKED_TIMER_KEYS:
            duration_s = timers.get(timer_key)
            if isinstance(duration_s, (int, float)):
                parts.append(f"{timer_key}={duration_s * 1000.0:.1f} ms")
            else:
                parts.append(f"{timer_key}=NA")

        log_str: str = f"Iteration {iteration} timers:\n {"\n ".join(parts)}"
        logger.info(log_str)

    return on_train_result
