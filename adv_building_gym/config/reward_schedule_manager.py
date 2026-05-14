"""Stateful reward schedule manager.

A trial declares the **reward pool** (``trial.rewards``) and an optional
inlined **reward schedule** (``trial.reward_schedule``).  The schedule
selects/orders/weights members of the pool over training episodes.

Modes (see ``configs/schedules/reward/README.md``):

* ``off``         — entire pool active, no swapping.
* ``fix``         — only ``on_rewards`` active for the whole run.
* ``gradual_add`` — start with the first ``reward_order`` entry, add the
                    next every swap.  Composite entries add all their
                    members at once.
* ``random``      — keep ``random_active_count`` active; each swap exchange
                    ``random_swap_count`` active for inactive.
* ``dirichlet``   — fixed active set (``on_rewards``); resample weights
                    every swap from a Dirichlet distribution with optional
                    rejection sampling against per-reward ``w_low`` floors.
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Any

import numpy as np

from adv_building_gym.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)


class RewardScheduleMode(Enum):
    OFF = "off"
    FIX = "fix"
    GRAD_ADD = "gradual_add"
    RANDOM = "random"
    DIRICHLET = "dirichlet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry_name(entry: dict[str, Any]) -> str:
    """Return the reward class_name an entry refers to."""
    if "name" in entry:
        return entry["name"]
    if "class_name" in entry:
        return entry["class_name"]
    raise ValueError(f"Schedule entry missing 'name'/'class_name': {entry}")


def _entry_weight_override(entry: dict[str, Any]) -> float | None:
    """Return ``w_override`` (or legacy ``w``) when present, else None."""
    if "w_override" in entry and entry["w_override"] is not None:
        return float(entry["w_override"])
    if "w" in entry and entry["w"] is not None:
        return float(entry["w"])
    return None


def _spec_by_name(pool: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {s["class_name"]: s for s in pool}


def _filter_pool_by_entries(
    rewards_pool: list[dict[str, Any]],
    reward_selector_entries: list[dict[str, Any]],
    *,
    field_label: str,
) -> list[dict[str, Any]]:
    """Project the pool down to *entries*, applying ``w_override`` per entry.

    Keeps the order in which ``entries`` are listed.  Raises if an entry
    references a name absent from the pool.
    """
    by_name = _spec_by_name(rewards_pool)
    filtered: list[dict[str, Any]] = []
    for entry in reward_selector_entries:
        reward_name = _entry_name(entry)
        if reward_name not in by_name:
            raise ValueError(
                f"reward_schedule.{field_label} references unknown reward "
                f"'{reward_name}' (not in trial 'rewards' pool)"
            )
        spec = by_name[reward_name]
        w = _entry_weight_override(entry)
        filtered.append({**spec, "weight": w if w is not None else spec["weight"]})
    return filtered


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class RewardScheduleManager:
    """Stateful reward schedule service.

    Driver-process only — ``advance`` / ``create_active_rewards`` are
    invoked from the ``on_train_result`` callback so there is no
    concurrency concern.

    SAC replay-buffer caveat: after a reward swap, old transitions in the
    buffer carry rewards from the previous reward set.  Keep
    ``swap_every_n_episodes`` large relative to buffer turnover.
    """
    # TODO VP 2026.05.11.: Flush half of the replay buffer in case of new objectives?
    # Or just gather some samples without training after a swap, to "prime" the buffer with the new rewards?
    # So the buffer is half old-reward, half new-reward for a while, then fully new-reward after that.

    def __init__(
        self,
        mode: RewardScheduleMode,
        swap_every_n_episodes: int,
        seed: int,
        reward_specs: list[dict[str, Any]],
        *,
        # GRAD_ADD
        grad_add_groups: list[list[dict[str, Any]]] | None = None,
        # RANDOM
        random_active_count: int | None = None,
        random_swap_count: int = 1,
        # DIRICHLET
        dirichlet_w_low: dict[str, float] | None = None,
        dirichlet_rejection_sampling: bool = True,
        dirichlet_uniform_start: bool = True,
        first_swap_after_n_episodes: int | None = None,
    ) -> None:
        if not reward_specs and mode is not RewardScheduleMode.GRAD_ADD:
            raise ValueError("reward_specs must contain at least one entry")
        if mode is RewardScheduleMode.GRAD_ADD and not grad_add_groups:
            raise ValueError("GRAD_ADD requires non-empty 'reward_order'")

        self.mode = mode
        self.swap_every_n_episodes = max(1, int(swap_every_n_episodes))
        self._reward_specs = reward_specs
        self._rng = np.random.default_rng(seed)
        self._swap_index: int = 0
        self.first_swap_after_n_episodes = first_swap_after_n_episodes

        # GRAD_ADD bookkeeping
        self._grad_add_groups = grad_add_groups or []

        # RANDOM bookkeeping
        n = len(reward_specs)
        if mode is RewardScheduleMode.RANDOM:
            if random_active_count is None:
                random_active_count = max(1, (n + 1) // 2)
            if not 1 <= random_active_count <= n:
                raise ValueError(f"random_active_count must be in [1, {n}], got {random_active_count}")
            max_swap = min(random_active_count, n - random_active_count)
            if random_swap_count < 0:
                raise ValueError(f"random_swap_count must be >= 0, got {random_swap_count}")
            if random_swap_count > max_swap:
                logger.warning(
                    "random_swap_count=%d clamped to %d (active=%d, total=%d)",
                    random_swap_count, max_swap, random_active_count, n,
                )
            self.random_active_count = random_active_count
            self.random_swap_count = max(0, min(random_swap_count, max_swap))
        else:
            self.random_active_count = random_active_count or 0
            self.random_swap_count = random_swap_count
        self._random_active_indices: list[int] | None = None

        # DIRICHLET bookkeeping
        self._dir_w_low = dirichlet_w_low or {}
        self._dir_rejection = bool(dirichlet_rejection_sampling)
        self._dir_uniform_start = bool(dirichlet_uniform_start)
        # Per-swap weights override; populated lazily.
        self._dir_weights: list[float] | None = None

        # Cached active specs invalidated on advance()
        self._active_specs_cache: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def from_dict(
        rewards: list[dict[str, Any]],
        reward_sch_cfg: dict[str, Any] | None,
        default_seed: int | None,
    ) -> "RewardScheduleManager":
        """Build a manager from the trial's reward pool + inlined schedule.

        Args:
            rewards: trial.rewards — list of {class_name, weight, params}.
            reward_sch_cfg: trial.reward_schedule (None ⇒ mode=off).
            default_seed: fallback seed when the schedule omits one.
        """
        if not rewards:
            raise ValueError("Trial 'rewards' pool is empty")

        cfg = reward_sch_cfg or {"mode": "off"}
        mode = RewardScheduleMode(cfg.get("mode", "off"))

        if "seed" in cfg and cfg["seed"] is not None:
            seed = int(cfg["seed"])
        elif default_seed is not None:
            seed = int(default_seed)
        else:
            raise ValueError("Reward schedule omits 'seed' and no default_seed was supplied")

        # Number of episodes (summed across all env_runners) between
        # swaps. The scheduler callback clamps this to
        # max(N, num_env_runners) at registration time. Not meaningful
        # for OFF / FIX (no swapping happens), so it's optional there.
        static_modes = (RewardScheduleMode.OFF, RewardScheduleMode.FIX)
        if "swap_every_n_episodes" not in cfg:
            if mode not in static_modes:
                raise ValueError(
                    "Reward schedule YAML must declare 'swap_every_n_episodes' "
                    "(episodes across all env_runners between reward swaps) "
                    f"for mode={mode.value}."
                )
            swap_n = 1
        else:
            swap_n = int(cfg["swap_every_n_episodes"])

        if mode is RewardScheduleMode.OFF:
            mgr = RewardScheduleManager(mode, swap_n, seed, reward_specs=list(rewards))

        elif mode is RewardScheduleMode.FIX:
            on_rewards = cfg.get("on_rewards") or []
            specs = _filter_pool_by_entries(rewards, on_rewards, field_label="on_rewards")
            mgr = RewardScheduleManager(mode, swap_n, seed, reward_specs=specs)

        elif mode is RewardScheduleMode.RANDOM:
            on_rewards = cfg.get("on_rewards") or []
            specs = _filter_pool_by_entries(rewards, on_rewards, field_label="on_rewards") \
                if on_rewards else list(rewards)
            mgr = RewardScheduleManager(
                mode, swap_n, seed, reward_specs=specs,
                random_active_count=cfg.get("random_active_count"),
                random_swap_count=int(cfg.get("random_swap_count", 1)),
            )

        elif mode is RewardScheduleMode.GRAD_ADD:
            order = cfg.get("reward_order") or []
            if not order:
                raise ValueError("gradual_add requires 'reward_order'")

            # Each top-level entry becomes a group of 1+ specs (composite).
            groups: list[list[dict[str, Any]]] = []
            seen: list[str] = []
            by_name = _spec_by_name(rewards)
            for entry in order:
                if "composite" in entry:
                    group_entries = entry["composite"]
                else:
                    group_entries = [entry]
                group_specs: list[dict[str, Any]] = []
                for sub in group_entries:
                    name = _entry_name(sub)
                    if name not in by_name:
                        raise ValueError(f"reward_order references unknown reward '{name}'")
                    base = by_name[name]
                    final_weight = _entry_weight_override(sub)
                    group_specs.append({**base, "weight": final_weight if final_weight is not None else base["weight"]})
                    seen.append(name)
                groups.append(group_specs)

            # Manager spec list is the *flat union of all groups* (in order).
            flat = [s for g in groups for s in g]
            mgr = RewardScheduleManager(
                mode, swap_n, seed, reward_specs=flat,
                grad_add_groups=groups,
            )

        elif mode is RewardScheduleMode.DIRICHLET:
            on_rewards = cfg.get("on_rewards") or []
            if not on_rewards:
                raise ValueError("dirichlet requires 'on_rewards'")
            specs = _filter_pool_by_entries(rewards, on_rewards, field_label="on_rewards")
            w_low = {
                _entry_name(e): float(e["w_low"])
                for e in on_rewards
                if e.get("w_low") is not None
            }
            mgr = RewardScheduleManager(
                mode, swap_n, seed, reward_specs=specs,
                dirichlet_w_low=w_low,
                dirichlet_rejection_sampling=bool(cfg.get("rejection_sampling", True)),
                dirichlet_uniform_start=(cfg.get("start_weights") == "uniform"),
                first_swap_after_n_episodes=cfg.get("first_swap_after_n_episodes"),
            )
        else:
            raise ValueError(f"Unsupported reward schedule mode: {mode}")

        logger.info(
            "RewardScheduleManager: mode=%s, pool_size=%d, swap_every=%d episodes",
            mgr.mode.value, len(mgr._reward_specs), mgr.swap_every_n_episodes,
        )
        return mgr

    # ------------------------------------------------------------------
    # Reward creation
    # ------------------------------------------------------------------

    def _create_reward_from_spec(self, spec: dict[str, Any]):
        import adv_building_gym.rewards  # noqa: F401  (populate registry)
        cls = ComponentRegistry.get("reward", spec["class_name"])
        kwargs = {"weight": spec["weight"], **(spec.get("params") or {})}
        return cls(**kwargs)

    def create_all_rewards(self) -> list:
        return [self._create_reward_from_spec(s) for s in self._reward_specs]

    def create_active_rewards(self) -> list:
        return [self._create_reward_from_spec(s) for s in self._get_active_specs()]

    def create_eval_rewards(self) -> list:
        """Return reward instances for evaluation.

        OFF / FIX -> same set as create_active_rewards().
        RANDOM / GRAD_ADD / DIRICHLET -> entire reward pool with the
        original pool weights (ignore swap-time subsampling and sampled
        Dirichlet weights).
        """
        # TODO VP 2026.05.13.: For the RANDOM, GRAD_ADD, DIRICHLET modes: 
        # filter the pool to the on_rewards/reward_order set (if specified), but ignore the per-swap active subset / weights.
        if self.mode in (RewardScheduleMode.OFF, RewardScheduleMode.FIX):
            return self.create_active_rewards()
        return [self._create_reward_from_spec(s) for s in self._reward_specs]

    def get_active_reward_names(self) -> list[str]:
        return [s["class_name"] for s in self._get_active_specs()]

    # ------------------------------------------------------------------
    # Active-spec computation
    # ------------------------------------------------------------------

    def _get_active_specs(self) -> list[dict[str, Any]]:
        if self._active_specs_cache is not None:
            return self._active_specs_cache
        self._active_specs_cache = self._compute_active_specs()
        return self._active_specs_cache

    def _compute_active_specs(self) -> list[dict[str, Any]]:
        n = len(self._reward_specs)

        if self.mode in (RewardScheduleMode.OFF, RewardScheduleMode.FIX):
            return list(self._reward_specs)

        if self.mode is RewardScheduleMode.GRAD_ADD:
            # Add one group per swap (cap at total groups).
            count = min(self._swap_index + 1, len(self._grad_add_groups))
            active: list[dict[str, Any]] = []
            for g in self._grad_add_groups[:count]:
                active.extend(g)
            return active

        if self.mode is RewardScheduleMode.RANDOM:
            if self._random_active_indices is None:
                idx = self._rng.choice(n, size=self.random_active_count, replace=False)
                self._random_active_indices = sorted(int(i) for i in idx)
            return [self._reward_specs[i] for i in self._random_active_indices]

        if self.mode is RewardScheduleMode.DIRICHLET:
            if self._dir_weights is None:
                self._dir_weights = self._draw_dirichlet_weights(uniform=self._dir_uniform_start)
            return [
                {**spec, "weight": float(w)}
                for spec, w in zip(self._reward_specs, self._dir_weights)
            ]

        return list(self._reward_specs)

    # ------------------------------------------------------------------
    # advance()
    # ------------------------------------------------------------------

    def advance(self) -> bool:
        """Advance the schedule by one swap step.

        Returns True iff the active reward set / weights changed.
        """
        old_signature = self._signature()
        self._swap_index += 1

        if self.mode is RewardScheduleMode.RANDOM and self._random_active_indices is not None:
            self._apply_random_swap()
        elif self.mode is RewardScheduleMode.DIRICHLET:
            self._dir_weights = self._draw_dirichlet_weights(uniform=False)

        self._active_specs_cache = None
        new_signature = self._signature()
        changed = old_signature != new_signature
        if changed:
            specs = self._get_active_specs()
            logger.info(
                "Reward schedule advanced (swap_index=%d): %s",
                self._swap_index,
                [f"{s['class_name']}(w={s['weight']:.3g})" for s in specs],
            )
        return changed

    def _signature(self) -> tuple:
        """Return a hashable signature of the active reward set + weights."""
        return tuple((s["class_name"], float(s["weight"])) for s in self._get_active_specs())

    # ------------------------------------------------------------------
    # RANDOM swap
    # ------------------------------------------------------------------

    def _apply_random_swap(self) -> None:
        if self.random_swap_count <= 0:
            return
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

    # ------------------------------------------------------------------
    # DIRICHLET
    # ------------------------------------------------------------------

    def _draw_dirichlet_weights(self, *, uniform: bool) -> list[float]:
        n = len(self._reward_specs)
        if uniform:
            return [1.0] * n
        floors = np.array(
            [self._dir_w_low.get(s["class_name"], 0.0) for s in self._reward_specs],
            dtype=float,
        )

        for _ in range(128):
            sample = self._rng.dirichlet(np.ones(n)) * n
            if not self._dir_rejection or np.all(sample >= floors):
                return [float(x) for x in sample]

        # Fall through after retries — emit warning, return last sample.
        logger.warning("Dirichlet rejection sampling exceeded retry budget; using last draw.")

        return [float(x) for x in sample]
