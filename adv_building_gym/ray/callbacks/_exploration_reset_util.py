"""Shared helper for the event-driven exploration reset (reward/infra/statesource swap callbacks).

Fires on every swap matching the configured ``trigger`` (the owning callback gates via
``ExplorationResetConfig.fires_on``). The exploration parameter is *raised* per algorithm:

- **SAC** — the live temperature ``learner.curr_log_alpha[mid]`` (an ``nn.Parameter`` shared
  with the ``"alpha"`` optimiser and the loss, ``sac_torch_learner.py``) is raised *only if it
  has fallen below* ``log(sac_alpha)``. SAC's own ``alpha_lr``/``target_entropy`` tuner relaxes
  it again between swaps, so there is no manual decay for SAC.
- **PPO** — ``entropy_coeff`` is not auto-tuned; it is read from a per-module ``Scheduler``
  (``ppo_learner.py``). On a swap it is raised to ``ppo_entropy_coeff`` and then linearly
  decayed back to its *original configured value* (captured once) over ``decay_iterations``.
  Writing ``Scheduler._curr_value`` is the runtime hook: ``Scheduler.update()`` is a no-op for a
  fixed value and ``get_current_value()`` returns ``_curr_value`` (``schedules/scheduler.py``).
"""

from __future__ import annotations

import logging
import math

from adv_building_gym.config.training.exploration_reset import ExplorationResetConfig

logger = logging.getLogger(__name__)


def make_decay_loop(exploration_reset: ExplorationResetConfig, event: str):
    """Return ``(state, maybe_decay, fire_bump)`` for one swap event.

    ``fire_bump(algorithm, iteration)`` raises the exploration parameter on each swap.
    ``maybe_decay(algorithm, iteration)`` runs every iteration and is PPO-only: it linearly
    decays a previous entropy bump back to the captured baseline. For SAC it is a no-op
    (``bump_iter`` is never armed); the temperature is relaxed by SAC's own alpha tuner.
    """
    state: dict = {"bump_iter": None, "ppo_floor": None}
    fires_here = exploration_reset.fires_on(event)
    target_log = math.log(exploration_reset.sac_alpha)

    # --- in-learner mutators (run inside each learner via foreach_learner; mutate in place) ---
    def _raise_sac(learner) -> None:
        """SAC: raise the live temperature to log(sac_alpha), only where it sits below it."""
        curr = getattr(learner, "curr_log_alpha", None)
        if curr is None:
            return
        for mid in learner.module.keys():
            log_alpha = curr[mid]  # nn.Parameter, shape [1]; same tensor the optimiser/loss use
            if float(log_alpha.item()) < target_log:
                log_alpha.data.fill_(target_log)

    def _set_ppo(learner, value: float) -> None:
        """PPO: set the per-module entropy coefficient (read by the loss via get_current_value)."""
        sched = getattr(learner, "entropy_coeff_schedulers_per_module", None)
        if sched is None:
            return
        for mid in learner.module.keys():
            sched[mid]._curr_value = float(value)

    def _read_ppo_floor(learner) -> float | None:
        """Return the pristine configured entropy_coeff, or None when the learner is not PPO."""
        sched = getattr(learner, "entropy_coeff_schedulers_per_module", None)
        if sched is None:
            return None
        mids = list(learner.module.keys())
        return float(sched[mids[0]].get_current_value()) if mids else None

    def maybe_decay(algorithm, iteration: int) -> None:
        # PPO-only: decay the in-flight entropy bump back to the captured baseline.
        if not fires_here or state["bump_iter"] is None:
            return
        elapsed = iteration - state["bump_iter"]
        floor, peak = state["ppo_floor"], exploration_reset.ppo_entropy_coeff
        if elapsed >= exploration_reset.decay_iterations:
            value = floor
            state["bump_iter"] = None
            logger.info("PPO entropy bump decayed back to baseline %.4g (iter=%d).", floor, iteration)
        else:
            value = floor + (1.0 - elapsed / exploration_reset.decay_iterations) * (peak - floor)
        algorithm.learner_group.foreach_learner(lambda l: _set_ppo(l, value))

    def fire_bump(algorithm, iteration: int) -> None:
        if not fires_here:
            return
        # Capture the pristine entropy_coeff once, before the first raise. None ⇒ pure-SAC run.
        if state["ppo_floor"] is None:
            results = [
                r.get()
                for r in algorithm.learner_group.foreach_learner(_read_ppo_floor).ignore_errors()
            ]
            floors = [f for f in results if f is not None]
            state["ppo_floor"] = floors[0] if floors else None

        # SAC: raise-only set of the live temperature (no-op for PPO learners).
        algorithm.learner_group.foreach_learner(_raise_sac)

        # PPO: raise entropy to the peak and arm the decay clock (no-op / unarmed for SAC).
        if state["ppo_floor"] is not None:
            algorithm.learner_group.foreach_learner(
                lambda l: _set_ppo(l, exploration_reset.ppo_entropy_coeff)
            )
            state["bump_iter"] = iteration
            logger.info(
                "Exploration kick (event=%s, iter=%d): PPO entropy_coeff %.4g -> %.4g, "
                "decay over %d iters.",
                event, iteration, state["ppo_floor"], exploration_reset.ppo_entropy_coeff,
                exploration_reset.decay_iterations,
            )
        else:
            logger.info(
                "Exploration kick (event=%s, iter=%d): SAC temperature raised to sac_alpha=%.4g "
                "where below target (auto-relaxed by the alpha tuner).",
                event, iteration, exploration_reset.sac_alpha,
            )

    return state, maybe_decay, fire_bump
