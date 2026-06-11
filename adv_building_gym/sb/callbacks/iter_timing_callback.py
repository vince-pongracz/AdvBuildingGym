"""SB3 iter-timing callback (equivalent of the Ray iter_timing_callback).

SB3 has no per-iteration ``on_train_result`` hook the way RLlib does
— learning is one long stream of ``model.learn`` with periodic
``_dump_logs`` writes. To surface the same kind of timing breakdown
(sampling vs. learner update vs. eval) we *wrap* the model's own
``train()`` and ``collect_rollouts()`` methods on
``_on_training_start`` and time each call.

What it surfaces (under the SB3 Logger ``timers/`` prefix — visible
both in the SB3 console table and in TensorBoard graphs):

- ``timers/sampling_s``       — wall time per ``collect_rollouts`` call
                                (on-policy: per PPO/A2C rollout;
                                off-policy: per ``train_freq`` window).
- ``timers/learner_update_s`` — wall time per ``train()`` call
                                (PPO/A2C: per epoch-batch update;
                                SAC/TD3: per gradient_steps batch).
- ``timers/eval_s``           — wall time per EvalCallback evaluation.
                                Recorded by ``wrap_eval_callback_with_timer``
                                below, which monkey-patches the supplied
                                ``EvalCallback._on_step`` to bracket only
                                the ticks that actually run an evaluation.
- ``timers/wall_step_s``      — wall time between two consecutive
                                ``_on_step`` invocations (env step rate
                                from the callback's perspective).
- ``timers/total_train_calls`` — running count of ``train()`` invocations.
- ``timers/total_sample_calls`` — running count of ``collect_rollouts``.

How to view them
----------------

1. In the SB3 console table: SB3's Logger prints ``timers/*`` keys to
   stdout every ``log_interval`` (PPO: every rollout; SAC: every 4
   env steps by default). Look for the ``timers/`` section.

2. In TensorBoard: pointed at ``models/<trial>/sb3/<algo>/<run>/tb/``,
   scalar tags appear under ``timers/`` alongside ``rollout/`` and
   ``train/``.

3. Programmatically: ``model.logger.name_to_value`` carries the most
   recently recorded value (used by ``SBBestCheckpointCallback`` for
   metric reads — same pattern).

Notes
-----

* Method wrapping happens ONCE in ``_on_training_start``. We rebind
  the method on the instance (not the class) so siblings using the
  same model class aren't affected.
* SB3's existing ``time/fps`` is "lifetime fps from the start of
  learn()", not per-iteration. ``timers/wall_step_s`` complements
  it with a tight per-step delta.
"""

from __future__ import annotations

import logging
import time
from functools import wraps

from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class SBIterTimingCallback(BaseCallback):
    """Records sampling / training / eval / per-step wall times."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._train_calls = 0
        self._sample_calls = 0
        self._last_step_perf: float | None = None
        # stored to restore the unwrapped methods on training_end (tidy across multiple learn() calls)
        self._unwrapped_train = None
        self._unwrapped_collect = None
        self._unwrapped_excluded = None

    # ------------------------------------------------------------------
    # Lifecycle: wrap model methods once we have the model in hand.
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        model = self.model
        self._wrap_train(model)
        self._wrap_collect_rollouts(model)
        # the wrapped methods close over `self`, which reaches ABC instances; SB3's
        # save() cloudpickles model.__dict__ and crashes on `_abc_data`. Exclude the
        # patched attrs so save skips them (restore falls back to the class methods).
        self._extend_excluded_save_params(model)
        self._last_step_perf = time.perf_counter()

    def _extend_excluded_save_params(self, model) -> None:
        """Extend ``model._excluded_save_params`` to also exclude our patches.

        Captures the resolved list (no model reference) so the replacement stays picklable;
        also excludes ``_excluded_save_params`` itself.
        """
        original = model._excluded_save_params
        base_exclude = list(original())
        # add our two patched methods + this attribute itself; dedupe defensively
        extra = ["train", "collect_rollouts", "_excluded_save_params"]

        def extended():
            return list(base_exclude) + [k for k in extra if k not in base_exclude]

        model._excluded_save_params = extended  # type: ignore[method-assign]
        self._unwrapped_excluded = original  # restored in _on_training_end

    def _wrap_train(self, model) -> None:
        original = model.train
        self._unwrapped_train = original
        logger_record = self.logger.record

        @wraps(original)
        def timed_train(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                self._train_calls += 1
                logger_record("timers/learner_update_s", float(dt))
                logger_record("timers/total_train_calls", int(self._train_calls))

        # Bind on the instance so subclasses / other models stay clean.
        model.train = timed_train  # type: ignore[method-assign]

    def _wrap_collect_rollouts(self, model) -> None:
        original = getattr(model, "collect_rollouts", None)
        if original is None:
            return
        self._unwrapped_collect = original
        logger_record = self.logger.record

        @wraps(original)
        def timed_collect(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                self._sample_calls += 1
                logger_record("timers/sampling_s", float(dt))
                logger_record("timers/total_sample_calls", int(self._sample_calls))

        model.collect_rollouts = timed_collect  # type: ignore[method-assign]

    # ------------------------------------------------------------------
    # Per-step delta.
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        now = time.perf_counter()
        if self._last_step_perf is not None:
            self.logger.record("timers/wall_step_s", float(now - self._last_step_perf))
        self._last_step_perf = now
        return True

    # ------------------------------------------------------------------
    # Cleanup.
    # ------------------------------------------------------------------

    def _on_training_end(self) -> None:
        """Delete the instance-attribute patches so the class methods come back.

        Re-assigning would still leave a closure-capturing attr (re-triggering the cloudpickle
        ABC error); ``del`` falls through to the clean class method and clears ``__dict__``.
        """
        model = self.model
        for attr in ("train", "collect_rollouts", "_excluded_save_params"):
            if attr in model.__dict__:
                try:
                    delattr(model, attr)
                except AttributeError:
                    pass


def wrap_eval_callback_with_timer(eval_callback) -> None:
    """Wrap ``EvalCallback._on_step`` to record ``timers/eval_s`` only on ticks that actually
    evaluate (``n_calls % eval_freq == 0``). Rebinds on the instance; idempotent."""
    if getattr(eval_callback, "_iter_timing_wrapped", False):
        return

    original = eval_callback._on_step

    @wraps(original)
    def timed_on_step(*args, **kwargs):
        # read eval_freq / n_calls before invoking; n_calls already reflects this call
        eval_freq = getattr(eval_callback, "eval_freq", 0)
        n_calls = getattr(eval_callback, "n_calls", 0)
        eval_will_run = eval_freq > 0 and n_calls % eval_freq == 0
        if not eval_will_run:
            return original(*args, **kwargs)

        t0 = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            eval_callback.logger.record("timers/eval_s", float(dt))

    eval_callback._on_step = timed_on_step  # type: ignore[method-assign]
    eval_callback._iter_timing_wrapped = True  # type: ignore[attr-defined]
