"""Episode-unit early stopping for a single Ray Tune trial.

Ray's built-in stoppers don't cover "stop when the held-out eval metric has not
improved for N episodes". 
``TrialPlateauStopper`` measures rolling-stdev *flatness*
with its window counted in training *iterations*, and reads the metric via a flat
``result.get`` that misses nested RLlib keys. This module adds an episode-unit rule on the
eval metric, OR-combined with a hard episode cap:

* mean-plateau — keep a running (EMA/Polyak) average of the eval scores; stop when no fresh
  eval has beaten that average (by > ``min_delta``) for ``patience_episodes`` episodes.

Implementation notes
--------------------
* The stopper receives the RLlib result dict *un-flattened*
  (ray/tune/execution/tune_controller.py: ``self._stopper(trial.trial_id, result)`` is
  handed ``result``, not ``flat_result``), so nested keys are navigated by splitting on '/'.
* The eval metric is absent on non-eval iterations (``evaluation_interval > 1``); only
  *fresh* eval samples are recorded, so carried-forward duplicates never depress the
  stdev. Consequently the effective resolution is one eval round — set
  ``patience_episodes`` to span several rounds (≈ evaluation_interval ×
  episodes_per_iteration episodes each).
"""

import logging
import math
from typing import Any

from ray.tune.stopper import CombinedStopper, Stopper

from adv_building_gym._common.early_stopping import PlateauEarlyStopper

logger = logging.getLogger(__name__)

# Nested result key holding the lifetime episode count (same key the legacy dict-stop used).
_EPISODES_KEY = "env_runners/num_episodes_lifetime"


def _dig(result: dict, path: list[str]) -> Any:
    """Return the value at a slash-split path in a nested result dict, or None if absent."""
    node: Any = result
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _as_number(value: Any) -> float | None:
    """Coerce a finite int/float (not bool) to float, else None."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


class MaxEpisodesStopper(Stopper):
    """Hard cap: stop once lifetime episodes reach ``max_episodes`` (nested-key aware).

    Replaces the legacy flattened dict-stop when early stopping is enabled, so the cap
    and the plateau rules can share one ``CombinedStopper``.
    """

    def __init__(self, max_episodes: int, episodes_key: str = _EPISODES_KEY):
        self._max = int(max_episodes)
        self._path = episodes_key.split("/")

    def __call__(self, trial_id: str, result: dict) -> bool:
        episodes = _as_number(_dig(result, self._path))
        return episodes is not None and episodes >= self._max

    def stop_all(self) -> bool:
        return False


class EvalPlateauStopper(Stopper):
    """Rolling-mean plateau early stopping on the eval metric, in episode units.

    Thin Tune-Stopper adapter over the shared ``PlateauEarlyStopper`` core: it digs the
    lifetime episode count and the (possibly absent) fresh eval value out of the nested
    result and hands them to the core, which owns the decision logic shared with the SB3
    driver's early-stopping callback.

    Args:
        metric: Slash-nested result key to monitor (e.g.
            ``evaluation/env_runners/episode_return_mean``).
        patience_episodes: Episodes without a fresh eval beating the running average
            before stopping.
        mode: ``"max"`` (higher is better) or ``"min"``.
        min_delta: Margin a new score must clear the running average by to count as improvement.
        grace_episodes: Do not stop before this many lifetime episodes (warmup guard).
        smoothing_window: EMA span in eval rounds (``alpha = 2/(smoothing_window+1)``).
        episodes_key: Slash-nested key holding the lifetime episode count.
    """

    def __init__(self, metric: str, patience_episodes: int, mode: str = "max",
                min_delta: float = 0.0, grace_episodes: int = 0, smoothing_window: int = 5,
                episodes_key: str = _EPISODES_KEY):
        self._metric_path = metric.split("/")
        self._episodes_path = episodes_key.split("/")
        self._core = PlateauEarlyStopper(
            patience_episodes=patience_episodes, mode=mode, min_delta=min_delta,
            grace_episodes=grace_episodes, smoothing_window=smoothing_window,
        )

    def __call__(self, trial_id: str, result: dict) -> bool:
        episodes = _as_number(_dig(result, self._episodes_path))
        if episodes is None:
            return False
        # Fresh eval value, or None on non-eval iterations (metric absent).
        value = _as_number(_dig(result, self._metric_path))
        if self._core.update(int(episodes), value):
            logger.info("[early-stop] %s — stopping %s.", self._core.reason, trial_id)
            return True
        return False

    def stop_all(self) -> bool:
        return False


def build_stop_criteria(training_param_config, metric: str, mode: str = "max"):
    """Return the Tune ``stop`` value for a trial.

    When ``early_stopping.patience_episodes <= 0`` returns the legacy single-key dict cap
    (unchanged behaviour). Otherwise returns a ``CombinedStopper`` (OR semantics) of the
    hard episode cap and the rolling-mean plateau rule.

    ``metric`` is the already-resolved eval metric string (e.g.
    ``evaluation/env_runners/episode_return_mean``); ``mode`` matches the tuner's mode.
    """
    max_episodes = training_param_config.max_episodes_to_run
    early_stopping = training_param_config.early_stopping
    patience = int(early_stopping.patience_episodes or 0)
    if patience <= 0:
        return {_EPISODES_KEY: max_episodes}

    logger.info(
        "Early stopping enabled: patience=%d episodes, smoothing_window=%d evals, "
        "min_delta=%.4g, grace=%d episodes, metric=%s (mode=%s), hard cap=%d episodes.",
        patience, early_stopping.smoothing_window, early_stopping.min_delta,
        early_stopping.grace_episodes, metric, mode, max_episodes,
    )
    return CombinedStopper(
        MaxEpisodesStopper(max_episodes),
        EvalPlateauStopper(
            metric=metric,
            patience_episodes=patience,
            mode=mode,
            min_delta=early_stopping.min_delta,
            grace_episodes=early_stopping.grace_episodes,
            smoothing_window=early_stopping.smoothing_window,
        ),
    )
