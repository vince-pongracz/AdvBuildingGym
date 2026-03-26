"""Stateful reward schedule manager.

Loads a reward schedule YAML config, tracks the current position in the
schedule, and creates reward subsets based on the active mode.

Modes:
    off          — All rewards active from the start, no swapping.
    gradual_add  — Start with the first reward, add the next one every
                   swap cycle.  Once all are added they stay active.
    iterate      — Only one reward is active at a time; rotate to the
                   next every swap cycle (round-robin).
    random       — Each swap randomly selects a subset of rewards
                   (at least 1, up to all).

Usage::

    manager = RewardConfigManager.from_yaml("configs/reward_schedule_train.yaml")
    rewards = manager.create_active_rewards()   # initial set
    manager.advance()                           # next swap
    rewards = manager.create_active_rewards()   # updated set
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from adv_building_gym.config.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)


class RewardScheduleMode(Enum):
    """Available reward schedule modes."""

    OFF = "off"
    GRADUAL_ADD = "gradual_add"
    ITERATE = "iterate"
    RANDOM = "random"


class RewardConfigManager:
    """Stateful reward schedule service.

    Loads a reward schedule YAML, tracks swap position, and creates
    reward subsets based on mode.

    The ``advance()`` / ``create_active_rewards()`` methods are called
    from the driver process (``on_train_result`` callback) so there is
    no concurrency concern.

    **SAC replay buffer caveat**: After a reward swap, old transitions in
    the replay buffer carry rewards computed by the *previous* reward set.
    This inconsistency is accepted — old rewards wash out as new
    transitions fill the buffer.  Keep ``swap_every_n_iterations`` large
    relative to the replay buffer turnover to minimise impact.
    """

    def __init__(
        self,
        mode: RewardScheduleMode,
        swap_every_n_iterations: int,
        seed: int,
        reward_specs: list[dict[str, Any]],
    ) -> None:
        if not reward_specs:
            raise ValueError("reward_specs must contain at least one entry")

        self.mode = mode
        self.swap_every_n_iterations = max(1, swap_every_n_iterations)
        self._reward_specs = reward_specs
        self._rng = np.random.default_rng(seed)
        self._swap_index: int = 0
        # Cached active specs — ensures RNG-dependent modes (RANDOM) produce
        # a consistent selection between get_active_reward_names() and
        # create_active_rewards() within the same swap step.
        self._active_specs_cache: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def from_yaml(path: str | Path) -> RewardConfigManager:
        """Build a RewardConfigManager from a YAML config file.

        Args:
            path: Path to the reward schedule YAML
                  (e.g. ``configs/reward_schedule_train.yaml``).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Reward schedule YAML not found: {path}"
            )

        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        reward_specs: list[dict[str, Any]] = []
        for entry in cfg["rewards"]:
            reward_specs.append({
                "class_name": entry["class_name"],
                "weight": entry["weight"],
                "params": entry.get("params", {}),
            })

        manager = RewardConfigManager(
            mode=RewardScheduleMode(cfg["mode"]),
            swap_every_n_iterations=cfg.get("swap_every_n_iterations", 50),
            seed=cfg.get("seed", 42),
            reward_specs=reward_specs,
        )
        logger.info(
            "Loaded reward schedule from %s: mode=%s, %d rewards, "
            "swap every %d iterations",
            path.name, manager.mode, len(reward_specs),
            manager.swap_every_n_iterations,
        )
        return manager

    # ------------------------------------------------------------------
    # Reward creation
    # ------------------------------------------------------------------

    def _create_reward_from_spec(self, spec: dict[str, Any]):
        """Instantiate a single RewardFunction from a spec dict."""
        # Trigger reward module imports so the ComponentRegistry is populated
        import adv_building_gym.rewards  # noqa: F401

        cls = ComponentRegistry.get("reward", spec["class_name"])
        kwargs = {"weight": spec["weight"], **spec["params"]}
        return cls(**kwargs)

    def create_all_rewards(self) -> list:
        """Create fresh instances of ALL configured rewards."""
        return [self._create_reward_from_spec(s) for s in self._reward_specs]

    def create_active_rewards(self) -> list:
        """Create fresh instances of the currently active reward subset.

        The active subset depends on ``self.mode`` and ``self._swap_index``.
        """
        specs = self._get_active_specs()
        return [self._create_reward_from_spec(s) for s in specs]

    def _get_active_specs(self) -> list[dict[str, Any]]:
        """Return the reward specs that are currently active.

        Results are cached per swap step so that RNG-dependent modes
        (RANDOM) return the same selection across multiple calls within
        the same step (e.g. ``get_active_reward_names()`` followed by
        ``create_active_rewards()``).  The cache is invalidated by
        ``advance()``.
        """
        if self._active_specs_cache is not None:
            return self._active_specs_cache

        self._active_specs_cache = self._compute_active_specs()
        return self._active_specs_cache

    def _compute_active_specs(self) -> list[dict[str, Any]]:
        """Compute the active specs for the current swap index."""
        n = len(self._reward_specs)

        if self.mode is RewardScheduleMode.OFF:
            return list(self._reward_specs)

        if self.mode is RewardScheduleMode.GRADUAL_ADD:
            # Start with 1 reward, add one more per swap (cap at total)
            count = min(self._swap_index + 1, n)
            return list(self._reward_specs[:count])

        if self.mode is RewardScheduleMode.ITERATE:
            idx = self._swap_index % n
            return [self._reward_specs[idx]]

        if self.mode is RewardScheduleMode.RANDOM:
            # Random subset: at least 1, up to all.
            # RNG is consumed exactly once per swap step (cache ensures this).
            k = int(self._rng.integers(1, n + 1))
            indices = self._rng.choice(n, size=k, replace=False)
            return [self._reward_specs[i] for i in sorted(indices)]

        # Should not reach here due to __init__ validation
        return list(self._reward_specs)

    # ------------------------------------------------------------------
    # Schedule state
    # ------------------------------------------------------------------

    def advance(self) -> bool:
        """Advance the schedule by one swap step.

        Invalidates the active-specs cache so the next call to
        ``create_active_rewards()`` / ``get_active_reward_names()``
        recomputes (and, for RANDOM mode, draws fresh RNG values).

        Returns:
            True if the active reward set changed (caller should push to
            env_runners), False otherwise.
        """
        old_names = self.get_active_reward_names()
        self._swap_index += 1
        self._active_specs_cache = None  # invalidate before recompute
        new_names = self.get_active_reward_names()
        changed = old_names != new_names
        if changed:
            logger.info(
                "Reward schedule advanced (swap_index=%d): %s",
                self._swap_index, new_names,
            )
        return changed

    def get_active_reward_names(self) -> list[str]:
        """Return class names of currently active rewards (for logging)."""
        return [s["class_name"] for s in self._get_active_specs()]
