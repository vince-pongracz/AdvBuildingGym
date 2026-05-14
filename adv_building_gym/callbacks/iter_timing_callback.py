"""Per-iteration timer logging via on_train_result.

Surfaces RLlib's built-in iteration timers so the relative cost of sampling,
learner updates, and weight sync is visible in the SLURM logs without
opening TensorBoard.

The timer keys live under ``result["timers"]`` and are populated by RLlib's
algorithm step (see ray/rllib/algorithms/algorithm.py — ``_TIMERS``). They
are absent on iterations where the corresponding phase did not run, so we
read them defensively.

NOTE: every value under ``result["timers"]`` is an EMA with default
``ema_coeff=0.01`` (see ray/rllib/utils/metrics/metrics_logger.py
log_value:394-396 — the fallback when ``reduce="mean"`` with no window).
A step change in real wall time (e.g. crossing
``num_steps_sampled_before_learning_starts``) takes ~hundreds of iterations
to converge in the EMA. ``wall_iter_ms`` below records the true wall-clock
gap between consecutive ``on_train_result`` calls so the actual cost is
visible immediately.
"""

import logging
import time

logger = logging.getLogger(__name__)

# RLlib new API stack timer keys (values are seconds; not all are present every iter).
# - learner_update_timer / replay_buffer_sampling_timer are absent during the warmup
#   phase (num_steps_sampled_before_learning_starts).
# - synch_env_connectors only fires when env-runner state is broadcast.
# - replay_buffer_sampling_timer wraps one local_replay_buffer.sample() call inside
#   the UTD inner loop in DQN.training_step (SAC inherits) — so it is the
#   per-call EMA, not the cumulative per-iter time.
_TRACKED_TIMER_KEYS = (
    "training_iteration",              # total wall time of one Algorithm.training() call (n training_step() calls)
    "training_step",                   # duration of a single Algorithm.training_step() call
    "env_runner_sampling_timer",       # env_runner_group.sample() — rollout collection duration on env runners
    "learner_update_timer",            # learner_group.update() — forward/backward/optimizer step duration on the learner(s)
    "replay_buffer_sampling_timer",    # one local_replay_buffer.sample() call (per-call EMA inside the UTD loop)
    "replay_buffer_add_data_timer",    # local_replay_buffer.add() — inserting freshly sampled episodes into the buffer
    "replay_buffer_update_prios_timer",# local_replay_buffer.update_priorities() — PER priority writeback (no-op for uniform)
    "synch_env_connectors",            # broadcasting env-runner connector states (e.g. running mean stats) across runners
    "synch_weights",                   # env_runner_group.sync_weights() — pushing learner weights to env runners
    "restore_env_runners",             # bringing failed training env runners back before the training step
    "evaluation_iteration",            # full Algorithm.evaluate() call duration
    "synch_eval_env_connectors",       # broadcasting connector states across eval env runners
    "restore_eval_env_runners",        # bringing failed eval env runners back before evaluation
)


def create_iter_timing_on_train_result_cb():
    """Factory returning an ``on_train_result`` callable that logs RLlib timers."""

    state = {"last_call": None}

    def on_train_result(*, algorithm, result: dict, **kwargs) -> None:
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

        log_str:str = f"Iteration {iteration} timers:\n {"\n ".join(parts)}"
        logger.info(log_str)

    return on_train_result
