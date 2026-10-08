"""SB3 early-stopping callback — parity with the Ray Tune ``EvalPlateauStopper``.

Stops training when the held-out eval metric plateaus (no fresh eval beats the running
EMA average for ``patience_episodes``), measured in episode units. Both drivers share the decision core
(:class:`adv_building_gym._common.early_stopping.PlateauEarlyStopper`), so the behaviour
is identical; only the integ differs.

Wired as ``SBEvalStateActionCallback``'s after-eval hook (mirroring SB3's own
``EvalCallback`` + ``StopTrainingOnNoModelImprovement`` pattern): it fires once per eval
round, reads the fresh eval mean reward from ``self.parent.last_mean_reward``, and returns
``False`` to abort ``model.learn`` when the plateau rule triggers. The hard episode cap is
already enforced by ``run_train_sb.py``'s ``total_timesteps`` (= max_episodes × EPISODE_LENGTH),
so — unlike the Ray side — no separate max-episodes stopper is needed here.
"""

from __future__ import annotations

import logging

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from adv_building_gym._common.early_stopping import PlateauEarlyStopper

logger = logging.getLogger(__name__)


class SBEarlyStoppingCallback(BaseCallback):
    """Abort SB3 training on a rolling-mean eval plateau, in episode units.

    Args:
        episode_length: Steps per episode; lifetime episodes = ``num_timesteps // episode_length``
            (aggregated across the training VecEnv, matching Ray's ``num_episodes_lifetime``).
        patience_episodes: Episodes without a fresh eval beating the running average
            before stopping.
        mode: ``"max"`` (higher is better) or ``"min"``.
        min_delta: Margin a new score must clear the running average by to count as improvement.
        grace_episodes: Do not stop before this many lifetime episodes (warmup guard).
        smoothing_window: EMA span in eval rounds (``alpha = 2/(smoothing_window+1)``).
    """

    def __init__(self, *, episode_length: int, patience_episodes: int, mode: str = "max",
                min_delta: float = 0.0, grace_episodes: int = 0,
                smoothing_window: int = 5, verbose: int = 1):
        super().__init__(verbose)
        self._episode_length = max(1, int(episode_length))
        self._core = PlateauEarlyStopper(
            patience_episodes=patience_episodes, mode=mode, min_delta=min_delta,
            grace_episodes=grace_episodes, smoothing_window=smoothing_window,
        )

    def _on_step(self) -> bool:
        # Invoked only after a fresh eval round; the parent eval callback exposes its
        # latest eval mean reward. `num_timesteps` is refreshed by BaseCallback.on_step.
        value = getattr(self.parent, "last_mean_reward", None)
        if value is None or not np.isfinite(value):
            return True  # no usable eval yet — keep training
        episodes = int(self.num_timesteps // self._episode_length)
        if self._core.update(episodes, float(value)):
            logger.info(
                "[early-stop] %s — stopping training at %d episodes (%d timesteps).",
                self._core.reason, episodes, self.num_timesteps,
            )
            return False  # False from any callback aborts model.learn()
        return True
