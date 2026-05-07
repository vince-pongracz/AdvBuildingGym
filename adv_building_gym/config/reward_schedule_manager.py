"""Stateful reward schedule manager.

Loads a reward schedule YAML config, tracks the current position in the
schedule, and creates reward subsets based on the active mode.

Modes:
    off          — All rewards active from the start, no swapping.
    gradual_add  — Start with the first reward, add the next one every
                   swap cycle.  Once all are added they stay active.
    random       — Maintain a stable active set of ``random_active_count``
                   rewards.  Each swap, ``random_swap_count`` currently
                   active rewards are swapped out for the same number of
                   currently inactive rewards.  The active-set size stays
                   constant (so at least N rewards are always live).

Usage::

    manager = RewardScheduleManager.load(
        "configs/reward_cfgs/rewards.yaml",
        "configs/schedules/reward/train.yaml",
    )
    rewards = manager.create_active_rewards()   # initial set
    manager.advance()                           # next swap
    rewards = manager.create_active_rewards()   # updated set
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from adv_building_gym.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)


class RewardScheduleMode(Enum):
    """Available reward schedule modes."""

    OFF = "off"
    GRADUAL_ADD = "gradual_add"
    RANDOM = "random"


@dataclass
class ExplorationBumpConfig:
    """Exploration kick on reward swap (event-driven dynamic callback).

    On every reward set change, push exploration up so the policy re-tests
    the action space under the new objective, then linearly decay back to
    baseline over ``decay_iterations`` iterations.

    PPO   → bump ``entropy_coeff`` (read fresh each loss step).
    SAC   → force ``log_alpha`` to ``log(sac_alpha)``; ``alpha_lr`` will
            pull it back down toward target entropy on its own, but the
            decay still applies as an upper envelope.
    Both  → multiply optimiser LRs by ``lr_multiplier`` so the critic /
            value head can recalibrate to the shifted reward landscape.
    """

    enabled: bool = False
    ppo_entropy_coeff: float = 0.05
    ppo_entropy_baseline: float = 0.0
    sac_alpha: float = 0.5
    decay_iterations: int = 25
    lr_multiplier: float = 1.0

    @staticmethod
    def from_dict(cfg_raw: dict | None) -> "ExplorationBumpConfig":
        if not cfg_raw:
            # Return the default config
            return ExplorationBumpConfig()
        cfg = ExplorationBumpConfig(
            enabled=bool(cfg_raw.get("enabled", False)),
            ppo_entropy_coeff=float(cfg_raw.get("ppo_entropy_coeff", 0.05)),
            ppo_entropy_baseline=float(cfg_raw.get("ppo_entropy_baseline", 0.0)),
            sac_alpha=float(cfg_raw.get("sac_alpha", 0.5)),
            decay_iterations=int(cfg_raw.get("decay_iterations", 25)),
            lr_multiplier=float(cfg_raw.get("lr_multiplier", 1.0)),
        )

        if cfg.decay_iterations < 1:
            raise ValueError(f"exploration_bump.decay_iterations must be >= 1, got {cfg.decay_iterations}")
        if cfg.sac_alpha <= 0:
            raise ValueError(f"exploration_bump.sac_alpha must be > 0, got {cfg.sac_alpha}")
        if cfg.lr_multiplier <= 0:
            raise ValueError(f"exploration_bump.lr_multiplier must be > 0, got {cfg.lr_multiplier}")
        return cfg


class RewardScheduleManager:
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
        random_active_count: int | None = None,
        random_swap_count: int = 1,
        exploration_bump: ExplorationBumpConfig | None = None,
    ) -> None:
        if not reward_specs:
            raise ValueError("reward_specs must contain at least one entry")

        n = len(reward_specs)
        self.mode = mode
        self.swap_every_n_iterations = max(1, swap_every_n_iterations)
        self._reward_specs = reward_specs
        self._rng = np.random.default_rng(seed)
        self._swap_index: int = 0

        # RANDOM-mode parameters: keep a stable active set and swap a fraction
        # of it on each cycle.
        if random_active_count is None:
            random_active_count = max(1, (n + 1) // 2)
        if not 1 <= random_active_count <= n:
            raise ValueError(
                f"random_active_count must be in [1, {n}], got {random_active_count}"
            )
        max_swap = min(random_active_count, n - random_active_count)
        if random_swap_count < 0:
            raise ValueError(
                f"random_swap_count must be >= 0, got {random_swap_count}"
            )
        if random_swap_count > max_swap and mode is RewardScheduleMode.RANDOM:
            logger.warning(
                "random_swap_count=%d clamped to %d (active=%d, total=%d)",
                random_swap_count, max_swap, random_active_count, n,
            )
        self.random_active_count = random_active_count
        self.random_swap_count = max(0, min(random_swap_count, max_swap))
        self._random_active_indices: list[int] | None = None

        # Cached active specs — ensures RNG-dependent modes (RANDOM) produce
        # a consistent selection between get_active_reward_names() and
        # create_active_rewards() within the same swap step.
        self._active_specs_cache: list[dict[str, Any]] | None = None

        self.exploration_bump = exploration_bump or ExplorationBumpConfig()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def load(
        rewards_path: str | Path,
        schedule_path: str | Path | None = None,
        default_seed: int | None = None,
    ) -> RewardScheduleManager:
        """Build a RewardScheduleManager from a rewards YAML and an optional schedule YAML.

        The two concerns are split:

        * ``rewards_path`` (required) — *which* rewards exist
          (``configs/reward_cfgs/<name>.yaml``).
        * ``schedule_path`` (optional) — *how* those rewards are scheduled
          across training (``configs/schedules/reward/<name>.yaml``).
          When omitted, mode is ``OFF`` (all rewards live from the start).

        Seed resolution: if the schedule YAML defines ``seed`` it wins; else
        ``default_seed`` is used.  Without a schedule, ``default_seed`` is
        used directly (RANDOM mode is not reachable in this case so this
        seed is essentially unused).

        Args:
            rewards_path: Repo-relative path to the rewards definition YAML.
            schedule_path: Repo-relative path to the scheduling YAML.
            default_seed: Fallback seed when the schedule omits ``seed``.
        """
        rewards_path = Path(rewards_path)
        if not rewards_path.exists():
            raise FileNotFoundError(f"Rewards YAML not found: {rewards_path}")

        with open(rewards_path, "r") as reward_cfg_file:
            rewards_cfg = yaml.safe_load(reward_cfg_file) or {}
        if "rewards" not in rewards_cfg:
            raise ValueError(
                f"Rewards YAML {rewards_path.name} must contain a top-level "
                f"'rewards:' list of {{class_name, weight, params}} entries"
            )
        reward_specs: list[dict[str, Any]] = [
            {
                "class_name": entry["class_name"],
                "weight": entry["weight"],
                "params": entry.get("params", {}),
            }
            for entry in rewards_cfg["rewards"]
        ]

        if schedule_path is not None:
            schedule_path = Path(schedule_path)
            if not schedule_path.exists():
                raise FileNotFoundError(
                    f"Reward schedule YAML not found: {schedule_path}"
                )
            with open(schedule_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            if "mode" not in cfg:
                raise ValueError(
                    f"Reward schedule YAML {schedule_path.name} must declare 'mode'"
                )
        else:
            cfg = {"mode": "off"}

        if "seed" in cfg:
            seed = cfg["seed"]
        elif default_seed is not None:
            seed = default_seed
        else:
            raise ValueError(f"Reward schedule omits 'seed' and no default_seed was supplied")

        manager = RewardScheduleManager(
            mode=RewardScheduleMode(cfg["mode"]),
            swap_every_n_iterations=cfg.get("swap_every_n_iterations", 50),
            seed=seed,
            reward_specs=reward_specs,
            random_active_count=cfg.get("random_active_count"),
            random_swap_count=cfg.get("random_swap_count", 1),
            exploration_bump=ExplorationBumpConfig.from_dict(cfg.get("exploration_bump")),
        )
        reward_lines = [
            f"  {s['class_name']}: weight={s['weight']}"
            + (f", params={s['params']}" if s["params"] else "")
            for s in reward_specs
        ]
        schedule_label = schedule_path.name if schedule_path is not None else "(none — mode=off)"
        logger.info(
            "Loaded rewards from %s, schedule %s: mode=%s, "
            "swap every %d iterations\n"
            "Rewards (%d):\n%s",
            rewards_path.name, schedule_label, manager.mode,
            manager.swap_every_n_iterations,
            len(reward_specs), "\n".join(reward_lines),
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

        if self.mode is RewardScheduleMode.RANDOM:
            # Lazy-init the stable active set on first compute. The set is
            # mutated by ``advance()`` (swap_count out, swap_count in).
            if self._random_active_indices is None:
                indices = self._rng.choice(
                    n, size=self.random_active_count, replace=False,
                )
                self._random_active_indices = sorted(int(i) for i in indices)
            return [self._reward_specs[i] for i in self._random_active_indices]

        # Should not reach here due to __init__ validation
        return list(self._reward_specs)

    # ------------------------------------------------------------------
    # Schedule state
    # ------------------------------------------------------------------

    def advance(self) -> bool:
        """Advance the schedule by one swap step.

        For RANDOM mode this swaps ``random_swap_count`` currently active
        rewards out for the same number of currently inactive ones, keeping
        the active-set size constant.

        Invalidates the active-specs cache so the next call to
        ``create_active_rewards()`` / ``get_active_reward_names()``
        recomputes.

        Returns:
            True if the active reward set changed (caller should push to
            env_runners), False otherwise.
        """
        old_names = self.get_active_reward_names()
        self._swap_index += 1

        if (
            self.mode is RewardScheduleMode.RANDOM
            and self._random_active_indices is not None
            and self.random_swap_count > 0
        ):
            self._apply_random_swap()

        self._active_specs_cache = None  # invalidate before recompute
        new_names = self.get_active_reward_names()
        changed = old_names != new_names
        if changed:
            active_reward_specs = self._get_active_specs()
            reward_details = [
                f"{reward_spec['class_name']} (weight={reward_spec.get('weight', 1.0)})"
                for reward_spec in active_reward_specs
            ]
            logger.info(
                "Reward schedule advanced (swap_index=%d): %s",
                self._swap_index, reward_details,
            )
        return changed

    def get_active_reward_names(self) -> list[str]:
        """Return class names of currently active rewards (for logging)."""
        return [s["class_name"] for s in self._get_active_specs()]

    def _apply_random_swap(self) -> None:
        """Swap ``random_swap_count`` active indices for inactive ones.

        Mutates ``self._random_active_indices`` in place. Called from
        ``advance()``. Caller is responsible for invalidating the spec cache.
        """
        n = len(self._reward_specs)
        active = set(self._random_active_indices or [])
        inactive = [i for i in range(n) if i not in active]
        k = min(self.random_swap_count, len(active), len(inactive))
        if k == 0:
            return
        out = self._rng.choice(sorted(active), size=k, replace=False)
        in_ = self._rng.choice(inactive, size=k, replace=False)
        active.difference_update(int(i) for i in out)
        active.update(int(i) for i in in_)
        self._random_active_indices = sorted(active)
