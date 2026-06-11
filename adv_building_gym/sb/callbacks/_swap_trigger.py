"""Episode-counting swap gate for SB3 schedule callbacks.

SB3 has no ``on_train_result``; so callbacks get ``_on_step`` with per-env ``dones``. 
This counts lifetime episodes and fires after ``max(swap_every_n_episodes, num_env_runners)`` new
ones (matching the Ray swap-gate semantics).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SBSwapDecision:
    should_fire: bool
    is_first_fire: bool
    delta_episodes: int
    episodes_lifetime: int


class EpisodeCountingSwapGate:
    """Counts completed episodes across a VecEnv and gates swap events.

    ``update(dones)`` per ``_on_step`` accumulates the count; ``check()`` decides whether to fire.
    First call fires with ``is_first_fire=True`` (initial push); schedulers skip ``advance()`` then.
    """

    def __init__(self, name: str, configured_n: int, num_env_runners: int) -> None:
        effective_n = max(int(configured_n), int(num_env_runners))
        if effective_n != configured_n:
            logger.info(
                "%s: clamping swap_every_n_episodes %d -> %d (num_env_runners=%d).",
                name, configured_n, effective_n, num_env_runners,
            )
        self.name = name
        self.effective_n = effective_n
        self._episodes_lifetime = 0
        self._last_swap_episodes = 0
        self._first_call = True

    def update(self, dones) -> None:
        """Accumulate lifetime episode count from a ``dones`` array."""
        if dones is None:
            return
        try:
            self._episodes_lifetime += int(sum(bool(d) for d in dones))
        except TypeError:
            # Single env case (scalar)
            self._episodes_lifetime += int(bool(dones))

    def check(self) -> SBSwapDecision:
        episodes = self._episodes_lifetime
        if self._first_call:
            self._first_call = False
            self._last_swap_episodes = episodes
            logger.info(
                "%s: first-fire push (episodes_lifetime=%d, effective_n=%d).",
                self.name, episodes, self.effective_n,
            )
            return SBSwapDecision(True, True, 0, episodes)

        delta = episodes - self._last_swap_episodes
        if delta < self.effective_n:
            return SBSwapDecision(False, False, delta, episodes)

        self._last_swap_episodes = episodes
        logger.info(
            "%s swap fired (Δepisodes=%d, episodes_lifetime=%d, effective_n=%d).",
            self.name, delta, episodes, self.effective_n,
        )
        return SBSwapDecision(True, False, delta, episodes)


def make_swap_gate(name: str, configured_n: int, num_env_runners: int) -> Callable:
    """Build an EpisodeCountingSwapGate (kept as a factory for symmetry)."""
    return EpisodeCountingSwapGate(name, configured_n, num_env_runners)
