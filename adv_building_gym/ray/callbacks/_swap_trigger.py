"""Shared swap-trigger primitive for the four scheduler callbacks.

All four (data/reward/infra/statesource) gate ``on_train_result`` on one predicate:
"have the env_runners produced ≥ N more episodes since the last swap?" — sourced from
``result["env_runners"]["num_episodes_lifetime"]``. This packages that + the first-fire-always rule.

Invariant (iter-aligned, regime-pure metrics): the gate fires only from ``on_train_result``,
so swaps land on iter boundaries. With (1) ``rollout_fragment_length == EPISODE_LENGTH`` and
(2) ``window=_WITHIN_ITER_WINDOW, clear_on_reduce=True`` on episode metrics, each TB scalar
reflects exactly one regime. These three form one contract — audit together.

RLlib's built-in ``EPISODE_RETURN_*`` keys keep standard windowed smoothing (small
``metrics_num_episodes_for_smoothing``); checkpoint selection targets
``evaluation/env_runners/episode_return_mean``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SwapDecision:
    """Outcome of a single swap-gate check."""
    should_fire: bool
    is_first_fire: bool
    delta_episodes: int
    delta_iters: int
    episodes_lifetime: int


def make_swap_gate(name: str, configured_n: int, num_env_runners: int) -> Callable[[int, dict], "SwapDecision"]:
    """Build a swap-trigger closure ``check(iteration, result) -> SwapDecision``.

    Args:
        name: Human-readable scheduler name (for log lines).
        configured_n: YAML-supplied ``swap_every_n_episodes`` value.
        num_env_runners: Floor on the effective threshold so at least
            one episode per runner is sampled between swaps.

    Returns:
        ``check(iteration, result) -> SwapDecision``.

        The first call always returns ``should_fire=True`` and
        ``is_first_fire=True`` (initial push) even if no episodes have
        been sampled yet. Subsequent calls fire only when
        ``episodes_lifetime - last_swap_episodes >= effective_n``;
        ``is_first_fire`` is ``False`` for those. Schedulers with an
        explicit ``advance()`` step should skip it when ``is_first_fire``
        so the initial config is pushed unmodified.
    """
    effective_n = max(int(configured_n), int(num_env_runners))
    if effective_n != configured_n:
        logger.info(
            "%s: clamping swap_every_n_episodes %d -> %d (num_env_runners=%d).",
            name, configured_n, effective_n, num_env_runners,
        )

    state = {
        "first_call": True,
        "last_swap_episodes": 0,
        "last_swap_iteration": 0,
    }

    def check(iteration: int, result: dict) -> SwapDecision:
        episodes = int(result.get("env_runners", {}).get("num_episodes_lifetime", 0))

        if state["first_call"]:
            state["first_call"] = False
            state["last_swap_episodes"] = episodes
            state["last_swap_iteration"] = iteration
            logger.info(
                "%s: first-fire push at iter=%d (episodes_lifetime=%d, effective_n=%d).",
                name, iteration, episodes, effective_n,
            )
            return SwapDecision(
                should_fire=True,
                is_first_fire=True,
                delta_episodes=0,
                delta_iters=0,
                episodes_lifetime=episodes,
            )

        delta_eps = episodes - state["last_swap_episodes"]
        delta_iters = iteration - state["last_swap_iteration"]
        if delta_eps < effective_n:
            return SwapDecision(
                should_fire=False,
                is_first_fire=False,
                delta_episodes=delta_eps,
                delta_iters=delta_iters,
                episodes_lifetime=episodes,
            )

        state["last_swap_episodes"] = episodes
        state["last_swap_iteration"] = iteration
        logger.info(
            "%s swap fired at iter=%d (Δiters=%d, Δepisodes=%d, "
            "episodes_lifetime=%d, effective_n=%d).",
            name, iteration, delta_iters, delta_eps, episodes, effective_n,
        )
        return SwapDecision(
            should_fire=True,
            is_first_fire=False,
            delta_episodes=delta_eps,
            delta_iters=delta_iters,
            episodes_lifetime=episodes,
        )

    check.effective_n = effective_n  # exposed for startup log
    return check
