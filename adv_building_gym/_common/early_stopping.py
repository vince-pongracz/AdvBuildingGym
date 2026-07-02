"""Framework-agnostic early-stop decision on a scalar eval metric, in episode units.

Keeps a running EMA (Polyak average) of the eval scores; a fresh eval "improves" only if it
beats the current EMA by more than ``min_delta``. Training stops once no eval has improved
for ``patience_episodes`` episodes — averaging rather than tracking a single best value
smooths out eval noise. The EMA uses ``alpha = 2 / (smoothing_window + 1)`` (pandas
``ewm(span=...)`` convention) and updates in O(1) with no stored window.

:meth:`update` is called once per check with ``(episodes, value)``; ``value`` is ``None``
when there is no fresh eval (e.g. an RLlib non-eval iteration) — the episode clock advances
but the EMA is untouched. No stop fires until ``smoothing_window`` evals have been seen
(warmup) and ``episodes >= grace_episodes``.

Wrapped by ``ray/utils/early_stopping.py`` (Tune Stopper) and
``sb/callbacks/early_stopping_callback.py`` (SB3 callback), so both drivers behave identically.
"""

import math

# Recent eval rounds spanned by the EMA when unset by the caller.
DEFAULT_SMOOTHING_WINDOW = 30


class PlateauEarlyStopper:
    """Running-mean (EMA/Polyak) plateau stop decision on a scalar metric, in episode units.

    Args:
        patience_episodes: Episodes without a fresh eval beating the EMA before stopping.
        mode: ``"max"`` (higher is better) or ``"min"``.
        min_delta: Margin a new score must clear the EMA by to count as an improvement.
        grace_episodes: Do not stop before this many episodes (warmup guard).
        smoothing_window: EMA span in eval rounds (``alpha = 2/(smoothing_window+1)``); also the
            warmup sample count.
    """

    def __init__(self, patience_episodes: int, mode: str = "max", min_delta: float = 0.0,
                grace_episodes: int = 0, smoothing_window: int = DEFAULT_SMOOTHING_WINDOW):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        if smoothing_window < 1:
            raise ValueError(f"smoothing_window must be >= 1, got {smoothing_window}")
        self.patience_episodes = int(patience_episodes)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.grace_episodes = int(grace_episodes)
        self.smoothing_window = int(smoothing_window)
        # EMA decay (pandas ewm span convention): span N ⇒ alpha = 2 / (N + 1).
        self._alpha = 2.0 / (self.smoothing_window + 1)
        # Running EMA of eval scores (None until the first sample seeds it) + count seen.
        self._ema: float | None = None
        self._samples_seen = 0
        # Episode at which a fresh eval last beat the running average.
        self.last_improve_episode: int | None = None
        # Human-readable explanation of the most recent stop decision (empty until it fires).
        self.reason = ""

    def _beats_average(self, value: float, avg: float) -> bool:
        if self.mode == "max":
            return value > avg + self.min_delta
        return value < avg - self.min_delta

    def update(self, episodes: int, value: float | None = None) -> bool:
        """Record a check at ``episodes`` (optionally with a fresh ``value``); return True to stop."""
        episodes = int(episodes)

        # Fresh, finite eval: compare against the running average of recent history (before this
        # score is folded in), then advance the EMA in O(1).
        if value is not None and math.isfinite(value):
            value = float(value)
            if self._ema is None:
                self._ema = value                     # first sample seeds the average
                self.last_improve_episode = episodes  # ...and the improvement baseline
            else:
                if self._beats_average(value, self._ema):
                    self.last_improve_episode = episodes
                # ema_new = (1 - α)·ema_old + α·value
                # = ema_old − α·ema_old + α·value
                # = ema_old + α·(value − ema_old)
                self._ema += self._alpha * (value - self._ema)  # Polyak / EMA update
            self._samples_seen += 1

        # Decide only once enough evals have been seen (warmup) and past the grace period.
        if (self.last_improve_episode is None
                or self._samples_seen < self.smoothing_window
                or episodes < self.grace_episodes):
            return False

        # Mean-plateau: no fresh eval has beaten the running average for `patience_episodes`.
        stalled = episodes - self.last_improve_episode
        if stalled >= self.patience_episodes:
            self.reason = (
                f"mean-plateau: no eval beat the running mean (EMA span {self.smoothing_window}, "
                f"{self._ema:.4f}) by > {self.min_delta:.4g} for {stalled} episodes "
                f"(last improvement at ep {self.last_improve_episode}, now ep {episodes})"
            )
            return True
        return False
