"""Standalone exploration-reset configuration.

Event-driven exploration kick fired on reward / infra / statesource swap
events.  See ``configs/schedules/reward/EXPLORATION_RESET_README.md``.

The config is owned by the trial (``trial.exploration_reset``) and shared
by the reward / infra / statesource swap callbacks; each callback only
fires the bump when the configured ``trigger`` includes its event.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ExplorationResetTrigger(str, Enum):
    OFF = "off"
    ON_REWARD_SWAP = "on_reward_swap"
    ON_INFRA_SWAP = "on_infra_swap"
    ON_STATESOURCE_SWAP = "on_statesource_swap"
    ON_REWARD_AND_INFRA = "on_reward_and_infra"  # reward + infra
    ALL = "all"  # reward + infra + statesource

    def fires_on(self, event: str) -> bool:
        if self is ExplorationResetTrigger.OFF:
            return False
        if self is ExplorationResetTrigger.ALL:
            return True
        if self is ExplorationResetTrigger.ON_REWARD_AND_INFRA:
            return event in ("on_reward_swap", "on_infra_swap")
        return self.value == event


@dataclass
class ExplorationResetConfig:
    """Exploration kick on swap events (event-driven dynamic callback).
    
    By default: switched OFF -- no exploration reset
    """

    enabled: bool = False
    trigger: ExplorationResetTrigger = ExplorationResetTrigger.ON_REWARD_SWAP
    ppo_entropy_coeff: float = 0.05
    ppo_entropy_baseline: float = 0.0
    sac_alpha: float = 0.5
    decay_iterations: int = 25
    lr_multiplier: float = 1.0

    @staticmethod
    def from_dict(cfg_raw: dict | None) -> "ExplorationResetConfig":
        if not cfg_raw:
            return ExplorationResetConfig()

        trigger_raw = cfg_raw.get("trigger", "on_reward_swap")
        try:
            trigger = ExplorationResetTrigger(trigger_raw)
        except ValueError as exc:
            valid = [m.value for m in ExplorationResetTrigger]
            raise ValueError(
                f"exploration_reset.trigger must be one of {valid}, got {trigger_raw!r}"
            ) from exc

        cfg = ExplorationResetConfig(
            enabled=bool(cfg_raw.get("enabled", False)),
            trigger=trigger,
            ppo_entropy_coeff=float(cfg_raw.get("ppo_entropy_coeff", 0.05)),
            ppo_entropy_baseline=float(cfg_raw.get("ppo_entropy_baseline", 0.0)),
            sac_alpha=float(cfg_raw.get("sac_alpha", 0.5)),
            decay_iterations=int(cfg_raw.get("decay_iterations", 25)),
            lr_multiplier=float(cfg_raw.get("lr_multiplier", 1.0)),
        )

        if cfg.decay_iterations < 1:
            raise ValueError(f"exploration_reset.decay_iterations must be >= 1, got {cfg.decay_iterations}")
        if cfg.sac_alpha <= 0:
            raise ValueError(f"exploration_reset.sac_alpha must be > 0, got {cfg.sac_alpha}")
        if cfg.lr_multiplier <= 0:
            raise ValueError(f"exploration_reset.lr_multiplier must be > 0, got {cfg.lr_multiplier}")
        return cfg

    def fires_on(self, event: str) -> bool:
        """Return True if this config should fire the bump on *event*."""
        return self.enabled and self.trigger.fires_on(event)


# Backwards-compatible alias for legacy imports.
ExplorationBumpConfig = ExplorationResetConfig
