"""Shared helper for the SB3 event-driven exploration reset.

SB3 port of :mod:`adv_building_gym.ray.callbacks._exploration_reset_util`. Used by the
reward / infra / statesource swap callbacks; the owning callback gates via
``ExplorationResetConfig.fires_on`` and drives ``maybe_decay`` / ``fire_bump``.

SB3 has a single in-process ``model`` (no learner group). The exploration parameter is
raised per algorithm:

- **SAC** (``ent_coef='auto'``) — the learned temperature ``model.log_ent_coef`` (an
  ``nn.Parameter`` optimised by ``model.ent_coef_optimizer`` toward ``target_entropy``,
  ``sb3 sac.py``) is raised *only if it has fallen below* ``log(sac_alpha)``; the optimiser
  relaxes it again, so there is no manual decay. A fixed ``ent_coef`` (no ``log_ent_coef``)
  is skipped — it is not a learned, relaxable quantity.
- **PPO** — ``model.ent_coef`` is a fixed float in the loss (``sb3 ppo.py``) with no tuner, so
  it is raised to ``ppo_entropy_coeff`` then linearly decayed back to its original configured
  value (captured once on the model) over ``decay_iterations`` callback ticks.
"""

from __future__ import annotations

import logging
import math

import torch as th

from adv_building_gym.config.training.exploration_reset import ExplorationResetConfig

logger = logging.getLogger(__name__)

# Pristine PPO ``ent_coef`` captured before the first bump, stashed on the model so all
# swap-event callbacks share one baseline.
_PPO_FLOOR_ATTR = "_sb_exploration_ppo_floor"


def _is_sac(model) -> bool:
    """SB3 SAC always defines ``ent_coef_optimizer`` (None for a fixed coef); PPO never does."""
    return hasattr(model, "ent_coef_optimizer")


def make_decay_loop(exploration_reset: ExplorationResetConfig, *, event: str):
    """Return ``(maybe_decay, fire_bump)`` for one swap event.

    ``fire_bump(model)`` raises the exploration parameter on each swap. ``maybe_decay(model)``
    runs every ``_on_step`` and is PPO-only: it linearly decays a previous entropy bump back to
    the captured baseline over ``decay_iterations`` ticks. For SAC it is a no-op (the
    ``ent_coef_optimizer`` relaxes the temperature).
    """
    fires_here = exploration_reset.fires_on(event)
    target_log = math.log(exploration_reset.sac_alpha)
    state: dict = {"bump_step": None}

    def maybe_decay(model) -> None:
        if not fires_here or state["bump_step"] is None:
            return
        floor = getattr(model, _PPO_FLOOR_ATTR, None)
        if floor is None:  # SAC (no PPO floor captured) -> nothing to decay
            return
        elapsed = int(getattr(model, "num_timesteps", 0)) - int(state["bump_step"])
        peak = exploration_reset.ppo_entropy_coeff
        if elapsed >= exploration_reset.decay_iterations:
            model.ent_coef = float(floor)
            state["bump_step"] = None
            logger.info("SB PPO entropy bump decayed back to baseline %.4g (%s).", floor, event)
        else:
            frac = 1.0 - elapsed / exploration_reset.decay_iterations
            model.ent_coef = float(floor + frac * (peak - floor))

    def fire_bump(model) -> None:
        if not fires_here:
            return
        if _is_sac(model):
            log_ent_coef = getattr(model, "log_ent_coef", None)
            if not isinstance(log_ent_coef, th.Tensor):
                logger.debug(
                    "SB exploration kick (%s): SAC has a fixed ent_coef (no log_ent_coef); "
                    "nothing to raise.", event,
                )
                return
            if float(log_ent_coef.detach().cpu().item()) < target_log:
                with th.no_grad():
                    log_ent_coef.data.fill_(target_log)
            logger.info(
                "SB exploration kick (%s) at step=%d: SAC temperature raised to sac_alpha=%.4g "
                "where below target (auto-relaxed by ent_coef_optimizer).",
                event, int(getattr(model, "num_timesteps", 0)), exploration_reset.sac_alpha,
            )
            return

        # PPO: capture the pristine ent_coef once (shared across events), raise, arm decay.
        if getattr(model, _PPO_FLOOR_ATTR, None) is None:
            setattr(model, _PPO_FLOOR_ATTR, float(model.ent_coef))
        floor = getattr(model, _PPO_FLOOR_ATTR)
        model.ent_coef = float(exploration_reset.ppo_entropy_coeff)
        state["bump_step"] = int(getattr(model, "num_timesteps", 0))
        logger.info(
            "SB exploration kick (%s) at step=%d: PPO ent_coef %.4g -> %.4g, decay over %d ticks.",
            event, state["bump_step"], floor, exploration_reset.ppo_entropy_coeff,
            exploration_reset.decay_iterations,
        )

    return maybe_decay, fire_bump
