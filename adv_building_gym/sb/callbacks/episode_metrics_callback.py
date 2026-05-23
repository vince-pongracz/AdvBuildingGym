"""Per-episode metrics for SB3, mirroring ``EpisodeMetricsCallback`` (Ray).

For every env that finishes an episode in this ``_on_step`` tick, sums
the per-step ``max_reward_step`` values from ``infos`` to compute
``reward_rate``, picks up the final ``cum_E_kWh`` / ``cum_price_EUR``,
and accumulates per-component reward totals from ``reward_breakdown``.

Writes to the SB3 Logger so TensorBoard sees:

    rollout/achieved_reward / _min / _max
    rollout/reward_rate     / _min / _max
    rollout/cum_E_kWh       / _min / _max
    rollout/cum_price_EUR   / _min / _max
    rollout/reward/<component_name>

(``rollout/`` is SB3's convention for env-rollout-derived metrics, the
same prefix it uses for ``ep_rew_mean`` / ``ep_len_mean``.)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class SBEpisodeMetricsCallback(BaseCallback):
    """SB3 port of ``adv_building_gym.ray.callbacks.episode_metrics_callback``.

    Per-step ``infos`` carry the episode budget keys; VecMonitor adds an
    ``"episode"`` entry on the terminal step with the standard
    ``r``/``l``/``t`` aggregates we use for the episode return.

    We can't rebuild the per-step max_reward_step accumulation from the
    terminal info alone (it's recomputed each step), so we keep a tiny
    per-env running total in this callback and reset it on done.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # Per-env in-flight accumulators (reset on done):
        #   max_reward_sum:        sum of info["max_reward_step"] over this episode
        #   reward_breakdown_sum:  dict[reward_name → sum over this episode]
        self._max_reward_sum: list[float] = []
        self._reward_breakdown_sum: list[dict[str, float]] = []
        self._n_envs = 0

    def _init_callback(self) -> None:
        self._n_envs = self.training_env.num_envs
        self._max_reward_sum = [0.0] * self._n_envs
        self._reward_breakdown_sum = [defaultdict(float) for _ in range(self._n_envs)]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones")
        if dones is None:
            dones = [False] * self._n_envs

        for env_idx, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            # Per-step accumulation (whether or not the episode ends).
            max_step = info.get("max_reward_step")
            if isinstance(max_step, (int, float)):
                self._max_reward_sum[env_idx] += float(max_step)

            breakdown = info.get("reward_breakdown")
            if isinstance(breakdown, dict):
                acc = self._reward_breakdown_sum[env_idx]
                for k, v in breakdown.items():
                    if isinstance(v, (int, float)):
                        acc[k] += float(v)

            # Episode end? VecMonitor publishes a top-level "episode" dict
            # at exactly the terminal step.
            if dones[env_idx] and "episode" in info:
                self._log_episode_end(env_idx, info)
                # Reset per-env buffers for the next episode.
                self._max_reward_sum[env_idx] = 0.0
                self._reward_breakdown_sum[env_idx] = defaultdict(float)

        return True

    def _log_episode_end(self, env_idx: int, info: dict) -> None:
        """Record terminal metrics for one episode."""
        ep_info: dict = info["episode"]
        # VecMonitor's keys: 'r' (cumulative reward), 'l' (length), 't' (wall).
        episode_return = float(ep_info.get("r", 0.0))
        ep_length = int(ep_info.get("l", 0))

        max_total = self._max_reward_sum[env_idx]
        reward_rate = episode_return / max_total if max_total > 0 else 0.0

        cum_E_kWh: Optional[float] = info.get("cum_E_kWh")
        cum_price_EUR: Optional[float] = info.get("cum_price_EUR")
        episode_count = info.get("episode_count")

        # SB3 Logger handles min/max/mean reductions via record_mean only —
        # we emit raw values plus min/max for parity with the Ray callback
        # which logs all three reductions. SB3 will accumulate them across
        # callback invocations within one ``dump`` window.
        self.logger.record_mean("rollout/achieved_reward", episode_return)
        self.logger.record("rollout/achieved_reward_last", episode_return)
        self.logger.record_mean("rollout/reward_rate", reward_rate)
        self.logger.record("rollout/reward_rate_last", reward_rate)

        if cum_E_kWh is not None:
            self.logger.record_mean("rollout/cum_E_kWh", float(cum_E_kWh))
        if cum_price_EUR is not None:
            self.logger.record_mean("rollout/cum_price_EUR", float(cum_price_EUR))

        for reward_key, total in self._reward_breakdown_sum[env_idx].items():
            self.logger.record_mean(f"rollout/reward/{reward_key}", float(total))

        ep_str = str(episode_count) if episode_count is not None else "?"
        if self.verbose:
            logger.info(
                "Episode %s (env %d) ended. Length: %d, Return: %.2f, "
                "Reward rate: %.4f, cum_E_kWh: %s",
                ep_str, env_idx, ep_length, episode_return, reward_rate,
                "%.3f" % cum_E_kWh if cum_E_kWh is not None else "n/a",
            )

    # Exposed so the checkpoint callback can read the most-recent reward_rate
    # without parsing TensorBoard files.
    @property
    def last_reward_rate(self) -> float:
        try:
            return float(self.logger.name_to_value.get("rollout/reward_rate_last", -np.inf))
        except Exception:
            return float("-inf")

    @property
    def last_achieved_reward(self) -> float:
        try:
            return float(self.logger.name_to_value.get("rollout/achieved_reward_last", -np.inf))
        except Exception:
            return float("-inf")
