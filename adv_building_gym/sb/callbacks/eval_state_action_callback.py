"""Eval-side logging callback for the SB3 driver — full parity with Ray.

This callback REPLACES SB3's stock ``EvalCallback``. It owns the eval
loop directly so the per-step ``infos``, ``actions``, ``observations``,
``rewards``, and ``dones`` are visible by ordinary Python iteration —
no ``locals()`` plumbing, no subclassing private internals.

What it produces, mirroring the Ray side ([ray/callbacks/eval_state_action_callback.py](adv_building_gym/ray/callbacks/eval_state_action_callback.py)):

1. ``eval/*`` scalars on the main TB log (via ``self.logger``):
   - ``eval/mean_reward``       (preserves SB3 EvalCallback's tag for tooling)
   - ``eval/mean_ep_length``
   - ``eval/achieved_reward``   (parity with ``rollout/achieved_reward``)
   - ``eval/reward_rate``       (parity with ``rollout/reward_rate``)
   - ``eval/cum_E_kWh``
   - ``eval/cum_price_EUR``
   - ``eval/reward/<component>``

2. Per-iteration TB SUB-RUN at::

      <eval_trajectories_root>/iter_NNNNNN_<trial_suffix>/

   with tags ``raw/<k>``, ``state/<k>``, ``action/<k>``, ``power/...``,
   ``reward/...`` — each emitted with ``/mean``, ``/min``, ``/max``
   reductions across the eval round's episodes, ``global_step`` = env
   step index within an episode (0…EPISODE_LENGTH-1). Identical on-disk
   shape to Ray, so ``start_tensorboard.sh`` works for both frameworks.

3. ``best/best_model.zip`` saved when ``mean_reward`` improves
   (replaces the stock ``EvalCallback``'s best-model save).

Why not subclass ``EvalCallback``? See the design discussion in
``.claude/plans/plan-it-first-how-magical-tower.md`` — short version:
the only per-step hook on the stock path is ``evaluate_policy(callback=...)``
which delivers state via a ``locals()`` dict pull, and ``EvalCallback``
wires that as ``_log_success_callback``. Owning the loop here gives
explicit data flow and a self-contained class.
"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


_TRIAL_SUFFIX_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitise_trial_name(name: Optional[str]) -> str:
    """Mirror Ray's trial-suffix sanitiser ([ray/callbacks/eval_state_action_callback.py:75](adv_building_gym/ray/callbacks/eval_state_action_callback.py#L75))."""
    if not name:
        return ""
    return _TRIAL_SUFFIX_RE.sub("_", name).strip("_") or "trial"


class _EvalRunStats:
    """Iteration-level accumulators populated during one eval round."""

    __slots__ = (
        "returns", "reward_rates", "cum_E_kWh", "cum_price_EUR",
        "lengths", "per_component_totals", "trajectory_buffer",
    )

    def __init__(self):
        self.returns: list[float] = []
        self.reward_rates: list[float] = []
        self.cum_E_kWh: list[float] = []
        self.cum_price_EUR: list[float] = []
        self.lengths: list[int] = []
        self.per_component_totals: list[dict[str, float]] = []
        # One entry per finished episode: dict[tag → list[float] (per step)]
        self.trajectory_buffer: list[dict[str, list[float]]] = []


class SBEvalStateActionCallback(BaseCallback):
    """Hand-rolled eval callback for the SB3 driver.

    Replaces ``stable_baselines3.common.callbacks.EvalCallback``. Owns
    the eval loop, emits parity-with-Ray TB tags, and saves the
    best-by-mean-reward model.

    Args:
        eval_env: A SB3 VecEnv (typically ``DummyVecEnv`` wrapped in
            ``VecMonitor``). Single-env is fine — eval episodes are
            stepped serially regardless. ``VecMonitor`` is REQUIRED so
            ``info["episode"]`` is set on done.
        best_model_save_path: Directory where ``best_model.zip`` is
            written when mean eval reward improves.
        eval_trajectories_root: Root dir for per-iter TB sub-runs.
            Typically ``ep_metrics/eval_trajectories/<stamp>_<jobid>/``.
        eval_freq: Cadence in training timesteps. Eval runs when
            ``self.n_calls % eval_freq == 0`` — same semantics as SB3's
            ``EvalCallback`` (per-env-step count after CallbackList
            propagates the tick). Exposed publicly so
            ``wrap_eval_callback_with_timer`` works unchanged.
        n_eval_episodes: Total episodes to roll across the VecEnv per
            eval round.
        trial_name: For the sub-run dir suffix (mirrors Ray's
            ``iter_NNNNNN_<trial_suffix>/`` naming).
        deterministic: Passed to ``self.model.predict``. Default True.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        *,
        best_model_save_path: str,
        eval_trajectories_root: str,
        eval_freq: int,
        n_eval_episodes: int = 2,
        trial_name: Optional[str] = None,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self._eval_env = eval_env
        self._best_dir = best_model_save_path
        self._eval_trajectories_root = eval_trajectories_root
        # Public attribute name — `wrap_eval_callback_with_timer` reads it.
        self.eval_freq = int(eval_freq)
        self._n_eval_episodes = max(1, int(n_eval_episodes))
        self._trial_suffix = _sanitise_trial_name(trial_name)
        self._deterministic = bool(deterministic)
        self._best_mean_reward: float = -np.inf
        self._eval_round = 0
        os.makedirs(self._best_dir, exist_ok=True)
        os.makedirs(self._eval_trajectories_root, exist_ok=True)

    # ---------------------------------------------------------------
    # SB3 callback hook
    # ---------------------------------------------------------------

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        stats = self._run_eval_round()
        if not stats.returns:
            # No episode finished — nothing to log / save. Keep training.
            return True

        self._eval_round += 1
        mean_reward = float(np.mean(stats.returns))
        self._emit_eval_scalars(stats, mean_reward)
        self._write_trajectory_subrun(stats.trajectory_buffer)
        self._maybe_save_best(mean_reward)
        return True

    # ---------------------------------------------------------------
    # Eval-loop owner
    # ---------------------------------------------------------------

    def _run_eval_round(self) -> _EvalRunStats:
        env = self._eval_env
        n_envs = env.num_envs
        # Split n_eval_episodes across envs (same distribution as evaluate_policy).
        target_per_env = (self._n_eval_episodes + np.arange(n_envs)) // n_envs
        episode_counts = np.zeros(n_envs, dtype=int)

        obs = env.reset()
        max_reward_sums = np.zeros(n_envs)
        breakdown_sums: list[dict[str, float]] = [defaultdict(float) for _ in range(n_envs)]
        in_flight: list[dict[str, list[float]]] = [defaultdict(list) for _ in range(n_envs)]
        stats = _EvalRunStats()

        while (episode_counts < target_per_env).any():
            action, _ = self.model.predict(obs, deterministic=self._deterministic)
            obs, _rewards, dones, infos = env.step(action)
            for env_idx, info in enumerate(infos):
                if episode_counts[env_idx] >= target_per_env[env_idx]:
                    continue
                self._capture_per_step(in_flight[env_idx], info)

                ms = info.get("max_reward_step")
                if isinstance(ms, (int, float)):
                    max_reward_sums[env_idx] += float(ms)
                for k, v in (info.get("reward_breakdown") or {}).items():
                    if isinstance(v, (int, float)):
                        breakdown_sums[env_idx][k] += float(v)

                if dones[env_idx] and "episode" in info:
                    ep_return = float(info["episode"].get("r", 0.0))
                    ep_length = int(info["episode"].get("l", 0))
                    max_total = max_reward_sums[env_idx]
                    stats.returns.append(ep_return)
                    stats.lengths.append(ep_length)
                    stats.reward_rates.append(ep_return / max_total if max_total > 0 else 0.0)
                    if "cum_E_kWh" in info:
                        stats.cum_E_kWh.append(float(info["cum_E_kWh"]))
                    if "cum_price_EUR" in info:
                        stats.cum_price_EUR.append(float(info["cum_price_EUR"]))
                    stats.per_component_totals.append(dict(breakdown_sums[env_idx]))
                    stats.trajectory_buffer.append(dict(in_flight[env_idx]))

                    max_reward_sums[env_idx] = 0.0
                    breakdown_sums[env_idx] = defaultdict(float)
                    in_flight[env_idx] = defaultdict(list)
                    episode_counts[env_idx] += 1

        return stats

    @staticmethod
    def _capture_per_step(ep_data: dict[str, list[float]], info: dict) -> None:
        """Mirror Ray's per-step capture ([eval_state_action_callback.py:108-147](adv_building_gym/ray/callbacks/eval_state_action_callback.py#L108-L147))."""
        # raw/* — denormalised physical values
        for key, val in (info.get("raw") or {}).items():
            try:
                ep_data[f"raw/{key}"].append(float(val))
            except (TypeError, ValueError):
                pass  # rare non-scalar raw entries — skip silently

        # state/* — requires log_full_info=True on the eval env
        state = info.get("state")
        if state is None:
            raise RuntimeError(
                "SBEvalStateActionCallback: info['state'] is missing. The eval "
                "env must set log_full_info=True (sb/env_creator.py does this "
                "for role='eval')."
            )
        for key, val in state.items():
            arr = np.atleast_1d(val)
            if arr.size == 1:
                ep_data[f"state/{key}"].append(float(arr.flat[0]))
            else:
                for dim, scalar in enumerate(arr.flat):
                    ep_data[f"state/{key}_{dim}"].append(float(scalar))

        # action/* — Dict action published by env.step (post-rescale, real units)
        for act_key, act_val in (info.get("action") or {}).items():
            arr = np.atleast_1d(act_val)
            for dim, scalar in enumerate(arr.flat):
                ep_data[f"action/{act_key}_{dim}"].append(float(scalar))

        # power/*
        if "net_power_kW" in info:
            ep_data["power/net_kW"].append(float(info["net_power_kW"]))
        for comp_name, comp_power in (info.get("power_breakdown") or {}).items():
            # power_breakdown values are (production_kW, consumption_kW) tuples
            # (per CLAUDE.md). net = production - consumption.
            if isinstance(comp_power, tuple):
                net_power = float(comp_power[0]) - float(comp_power[1])
            else:
                net_power = float(comp_power)
            ep_data[f"power/{comp_name}_kW"].append(net_power)

        # reward/*
        if "reward" in info:
            try:
                ep_data["reward/total"].append(float(info["reward"]))
            except (TypeError, ValueError):
                pass
        for rew_name, rew_val in (info.get("reward_breakdown") or {}).items():
            try:
                ep_data[f"reward/{rew_name}"].append(float(rew_val))
            except (TypeError, ValueError):
                pass

    # ---------------------------------------------------------------
    # Emit / persist
    # ---------------------------------------------------------------

    def _emit_eval_scalars(self, stats: _EvalRunStats, mean_reward: float) -> None:
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/mean_ep_length", float(np.mean(stats.lengths)))
        self.logger.record("eval/achieved_reward", mean_reward)
        self.logger.record("eval/reward_rate", float(np.mean(stats.reward_rates)))
        if stats.cum_E_kWh:
            self.logger.record("eval/cum_E_kWh", float(np.mean(stats.cum_E_kWh)))
        if stats.cum_price_EUR:
            self.logger.record("eval/cum_price_EUR", float(np.mean(stats.cum_price_EUR)))

        if stats.per_component_totals:
            all_keys: set[str] = set()
            for d in stats.per_component_totals:
                all_keys.update(d.keys())
            for k in sorted(all_keys):
                vals = [d.get(k, 0.0) for d in stats.per_component_totals]
                self.logger.record(f"eval/reward/{k}", float(np.mean(vals)))

        # Dump immediately so eval scalars appear on the eval cadence rather
        # than waiting for the next training-side dump (same pattern as SB3's
        # stock EvalCallback at callbacks.py:498-530).
        self.logger.dump(self.num_timesteps)

        if self.verbose:
            logger.info(
                "Eval round %d (n_eval_episodes=%d): mean_reward=%.4f reward_rate=%.4f",
                self._eval_round, self._n_eval_episodes,
                mean_reward, float(np.mean(stats.reward_rates)),
            )

    def _write_trajectory_subrun(self, ep_buffer: list[dict[str, list[float]]]) -> None:
        if not ep_buffer:
            return

        all_keys: set[str] = set()
        for ep in ep_buffer:
            all_keys.update(ep.keys())

        summarised: dict[str, list[float]] = {}
        for key in sorted(all_keys):
            arrays = [ep[key] for ep in ep_buffer if key in ep]
            if not arrays:
                continue
            min_len = min(len(a) for a in arrays)
            if min_len == 0:
                continue
            stacked = np.array([a[:min_len] for a in arrays])
            summarised[f"{key}/mean"] = np.mean(stacked, axis=0).tolist()
            summarised[f"{key}/min"] = np.min(stacked, axis=0).tolist()
            summarised[f"{key}/max"] = np.max(stacked, axis=0).tolist()

        suffix = f"_{self._trial_suffix}" if self._trial_suffix else ""
        run_dir = os.path.join(
            self._eval_trajectories_root,
            f"iter_{self._eval_round:06d}{suffix}",
        )
        with SummaryWriter(log_dir=run_dir) as writer:
            for key, vals in summarised.items():
                for step_idx, val in enumerate(vals):
                    writer.add_scalar(key, val, global_step=step_idx)

        if self.verbose:
            num_tags = len(summarised) // 3  # mean/min/max per logical tag
            num_steps = max((len(values) for values in summarised.values()), default=0)
            logger.info(
                "Eval trajectory round %d: %d tags x %d steps -> %s",
                self._eval_round, num_tags, num_steps, run_dir,
            )

    def _maybe_save_best(self, mean_reward: float) -> None:
        if mean_reward <= self._best_mean_reward:
            return
        self._best_mean_reward = mean_reward
        save_path = os.path.join(self._best_dir, "best_model")
        self.model.save(save_path)
        if self.verbose:
            logger.info(
                "New best eval mean reward %.4f → %s.zip",
                mean_reward, save_path,
            )
