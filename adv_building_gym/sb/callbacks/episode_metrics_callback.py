"""Per-episode metrics for SB3, mirroring Ray's ``EpisodeMetricsCallback``.

On each episode end: logs the return, final ``cum_E_kWh`` / ``cum_price_EUR``, and
per-component ``reward_breakdown`` totals to the SB3 Logger (``rollout/`` prefix):

    rollout/achieved_reward / _last ; rollout/cum_E_kWh ; rollout/cum_price_EUR ;
    rollout/reward/<component_name>
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class SBEpisodeMetricsCallback(BaseCallback):
    """SB3 port of Ray's episode_metrics_callback.

    VecMonitor adds an ``"episode"`` entry (r/l/t) on the terminal step. Per-component reward
    totals accumulate per env from ``reward_breakdown`` and reset on done.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # Per-env in-flight accumulator (reset on done):
        #   reward_breakdown_sum:  dict[reward_name → sum over this episode]
        self._reward_breakdown_sum: list[dict[str, float]] = []
        self._n_envs = 0

    def _init_callback(self) -> None:
        self._n_envs = self.training_env.num_envs
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
            breakdown = info.get("reward_breakdown")
            if isinstance(breakdown, dict):
                acc = self._reward_breakdown_sum[env_idx]
                for k, v in breakdown.items():
                    if isinstance(v, (int, float)):
                        acc[k] += float(v)

            # episode end: VecMonitor publishes a top-level "episode" dict at the terminal step
            if dones[env_idx] and "episode" in info:
                self._log_episode_end(env_idx, info)
                # Reset per-env buffer for the next episode.
                self._reward_breakdown_sum[env_idx] = defaultdict(float)

        return True

    def _log_episode_end(self, env_idx: int, info: dict) -> None:
        """Record terminal metrics for one episode."""
        ep_info: dict = info["episode"]
        # VecMonitor's keys: 'r' (cumulative reward), 'l' (length), 't' (wall).
        episode_return = float(ep_info.get("r", 0.0))
        ep_length = int(ep_info.get("l", 0))

        cum_E_kWh: Optional[float] = info.get("cum_E_kWh")
        cum_price_EUR: Optional[float] = info.get("cum_price_EUR")
        episode_count = info.get("episode_count")

        # SB3 Logger reduces via record_mean; emit raw + a _last value for Ray parity,
        # accumulated across calls within one dump window
        self.logger.record_mean("rollout/achieved_reward", episode_return)
        self.logger.record("rollout/achieved_reward_last", episode_return)

        if cum_E_kWh is not None:
            self.logger.record_mean("rollout/cum_E_kWh", float(cum_E_kWh))
        if cum_price_EUR is not None:
            self.logger.record_mean("rollout/cum_price_EUR", float(cum_price_EUR))

        for reward_key, total in self._reward_breakdown_sum[env_idx].items():
            self.logger.record_mean(f"rollout/reward/{reward_key}", float(total))

        ep_str = str(episode_count) if episode_count is not None else "?"
        if self.verbose:
            logger.info(
                "Episode %s (env %d) ended. Length: %d, Return: %.2f, cum_E_kWh: %s",
                ep_str, env_idx, ep_length, episode_return,
                "%.3f" % cum_E_kWh if cum_E_kWh is not None else "n/a",
            )
