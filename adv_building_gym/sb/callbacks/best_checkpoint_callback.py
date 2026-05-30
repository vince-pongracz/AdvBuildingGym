"""Best-by-metric checkpoint callback for SB3.

SB3's stock ``EvalCallback`` already saves a best-by-``mean_reward``
model, but the trial config also allows selecting ``achieved_reward`` or
``reward_rate`` (both come from our custom episode-metrics callback).
This callback keeps the top-N checkpoints, mirroring Ray Tune's
``CheckpointConfig``.

Scoring state is maintained INSIDE this callback (per-env in-flight
``max_reward_step`` sums + a rolling deque of recent episode returns),
not read out of ``logger.name_to_value``. SB3's ``dump_logs`` writes
``rollout/ep_rew_mean`` and immediately calls ``logger.dump()`` which
clears the dict, so no callback ``on_step`` can ever observe that key —
and ``achieved_reward_last`` / ``reward_rate_last`` set by
``SBEpisodeMetricsCallback`` were only reachable through fragile
callback-ordering luck. Doing the bookkeeping locally fixes all three
metric paths.

Fires on every episode completion but only writes when the new value
beats the worst kept checkpoint, gated by ``checkpoint_frequency_episodes``
so a SAC run with high UTD doesn't drown in disk I/O.
"""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


_VALID_METRICS = {
    "episode_return_mean",  # rolling mean over last `episode_return_mean_window` episodes
    "achieved_reward",      # sum-of-rewards over the most recent episode
    "reward_rate",          # achieved / max_possible over the most recent episode
}


class SBBestCheckpointCallback(BaseCallback):
    """Keep the top-N checkpoints by ``metric``, episode-aligned.

    Args:
        checkpoint_dir: Directory to write zip checkpoints into.
        metric: One of ``episode_return_mean`` / ``achieved_reward`` /
            ``reward_rate``. Scores are computed from per-env ``infos``
            and ``VecMonitor`` episode dicts seen in ``_on_step``.
        checkpoint_frequency_episodes: Only consider checkpointing every
            N episodes (counted across the VecEnv). Mirrors Ray's
            ``checkpoint_frequency_episodes``.
        episode_return_mean_window: Length of the rolling deque used when
            ``metric == "episode_return_mean"``. Sourced from
            ``TrainingParamConfig.episode_return_mean_window`` so it
            matches RLlib's ``metrics_num_episodes_for_smoothing``.
        num_to_keep: Maximum checkpoint count; the worst-by-score is
            evicted when this is exceeded.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        metric: str,
        checkpoint_frequency_episodes: int = 20,
        episode_return_mean_window: int = 30,
        num_to_keep: int = 3,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        if metric not in _VALID_METRICS:
            raise ValueError(f"metric {metric!r} must be one of {sorted(_VALID_METRICS)}")
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._metric = metric
        self._frequency = max(1, int(checkpoint_frequency_episodes))
        self._num_to_keep = max(1, int(num_to_keep))
        # (score, path) pairs — sorted by score ascending so worst is at index 0.
        self._kept: List[tuple[float, Path]] = []
        self._episodes_total = 0
        self._episodes_at_last_check = 0
        # Local scoring state — replaces logger.name_to_value polling.
        self._return_window: deque[float] = deque(maxlen=max(1, int(episode_return_mean_window)))
        self._last_achieved_reward: Optional[float] = None
        self._last_reward_rate: Optional[float] = None
        self._max_reward_sum: List[float] = []

    def _init_callback(self) -> None:
        self._max_reward_sum = [0.0] * self.training_env.num_envs

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones")
        if dones is None:
            dones = [False] * len(infos)

        for env_idx, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            max_step = info.get("max_reward_step")
            if isinstance(max_step, (int, float)):
                self._max_reward_sum[env_idx] += float(max_step)
            # Episode end? VecMonitor publishes "episode" at exactly the terminal step.
            if dones[env_idx] and "episode" in info:
                ep_return = float(info["episode"].get("r", 0.0))
                max_total = self._max_reward_sum[env_idx]
                self._last_reward_rate = (ep_return / max_total) if max_total > 0 else 0.0
                self._last_achieved_reward = ep_return
                self._return_window.append(ep_return)
                self._max_reward_sum[env_idx] = 0.0
                self._episodes_total += 1

        if self._episodes_total - self._episodes_at_last_check < self._frequency:
            return True
        self._episodes_at_last_check = self._episodes_total

        score = self._compute_score()
        if score is None or not np.isfinite(score):
            return True

        self._maybe_checkpoint(score)
        return True

    def _compute_score(self) -> Optional[float]:
        if self._metric == "episode_return_mean":
            return float(np.mean(self._return_window)) if self._return_window else None
        if self._metric == "achieved_reward":
            return self._last_achieved_reward
        if self._metric == "reward_rate":
            return self._last_reward_rate
        return None

    def _maybe_checkpoint(self, score: float) -> None:
        worst_kept_score = self._kept[0][0] if self._kept else -np.inf
        # If full and not strictly better, skip — no churn on disk.
        if len(self._kept) >= self._num_to_keep and score <= worst_kept_score:
            return

        episode = self._episodes_total
        timesteps = self.model.num_timesteps
        # NB: encode the score with `_` separating int/fraction so SB3's
        # `open_path` (which uses Path.suffix) doesn't misread "-1151.9565"
        # as a file extension and skip appending ".zip".
        score_token = f"{score:.4f}".replace(".", "_").replace("-", "n")
        fname = f"ckpt_ep{episode:06d}_ts{timesteps:08d}_{self._metric}{score_token}.zip"
        path = self._checkpoint_dir / fname

        self.model.save(str(path))
        self._kept.append((score, path))
        self._kept.sort(key=lambda t: t[0])  # ascending by score

        if len(self._kept) > self._num_to_keep:
            _, evict_path = self._kept.pop(0)
            try:
                os.remove(str(evict_path))
            except OSError as exc:
                logger.warning("Could not delete evicted checkpoint %s: %s", evict_path, exc)

        # Always rewrite a small metadata file pointing at the best.
        best_score, best_path = self._kept[-1]
        meta = {
            "metric": self._metric,
            "best_score": float(best_score),
            "best_checkpoint": str(best_path),
            "kept": [{"score": float(s), "path": str(p)} for s, p in self._kept],
            "episodes": int(episode),
            "timesteps": int(timesteps),
        }
        with open(self._checkpoint_dir / "best_checkpoint_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        if self.verbose:
            logger.info(
                "Checkpoint saved (episode %d, %s=%.4f, total kept=%d): %s",
                episode, self._metric, score, len(self._kept), path.name,
            )
