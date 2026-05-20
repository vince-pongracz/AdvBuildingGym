"""Best-by-metric checkpoint callback for SB3.

SB3's stock ``EvalCallback`` already saves a best-by-``mean_reward``
model, but the trial config also allows selecting ``achieved_reward`` or
``reward_rate`` (both come from our custom episode-metrics callback).
This callback consumes those metrics out of the SB3 Logger and keeps the
top-N checkpoints, mirroring Ray Tune's ``CheckpointConfig``.

Fires on every episode completion, but only writes when the new value
beats the worst kept checkpoint. Writes happen episode-aligned (not
timestep-aligned) so a SAC run with high UTD doesn't drown in disk I/O.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


_VALID_METRICS = {
    "episode_return_mean",  # SB3's mean reward — same as ep_rew_mean
    "achieved_reward",      # custom — sum-of-rewards over the most recent episode
    "reward_rate",          # custom — achieved / max_possible
}


class SBBestCheckpointCallback(BaseCallback):
    """Keep the top-N checkpoints by ``metric``, episode-aligned.

    Args:
        checkpoint_dir: Directory to write zip checkpoints into.
        metric: One of ``episode_return_mean`` / ``achieved_reward`` /
            ``reward_rate``. Reads the value from
            ``self.logger.name_to_value`` (keys written by
            ``SBEpisodeMetricsCallback`` and SB3's stock logger).
        checkpoint_frequency_episodes: Only consider checkpointing every
            N episodes (counted across the VecEnv). Mirrors Ray's
            ``checkpoint_frequency_episodes``.
        num_to_keep: Maximum checkpoint count; oldest-by-score is
            evicted when this is exceeded.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        metric: str,
        checkpoint_frequency_episodes: int = 20,
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
        # (score, path) pairs — kept sorted by score ascending so the
        # worst is at index 0.
        self._kept: List[tuple[float, Path]] = []
        self._episodes_total = 0
        self._episodes_at_last_check = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones") or []
        try:
            self._episodes_total += int(sum(bool(d) for d in dones))
        except TypeError:
            self._episodes_total += int(bool(dones))

        if self._episodes_total - self._episodes_at_last_check < self._frequency:
            return True
        self._episodes_at_last_check = self._episodes_total

        score = self._read_metric()
        if score is None or not np.isfinite(score):
            return True

        self._maybe_checkpoint(score)
        return True

    def _read_metric(self) -> Optional[float]:
        """Pull the latest metric value from the SB3 Logger."""
        name_to_value = getattr(self.logger, "name_to_value", {}) or {}

        if self._metric == "episode_return_mean":
            # SB3 publishes the rolling mean under rollout/ep_rew_mean.
            v = name_to_value.get("rollout/ep_rew_mean")
        elif self._metric == "achieved_reward":
            v = name_to_value.get("rollout/achieved_reward_last")
            if v is None:
                v = name_to_value.get("rollout/achieved_reward")
        elif self._metric == "reward_rate":
            v = name_to_value.get("rollout/reward_rate_last")
            if v is None:
                v = name_to_value.get("rollout/reward_rate")
        else:
            return None
        return float(v) if v is not None else None

    def _maybe_checkpoint(self, score: float) -> None:
        worst_kept_score = self._kept[0][0] if self._kept else -np.inf
        # If we already have N kept and this is no better than the worst,
        # skip — no churn on disk.
        if len(self._kept) >= self._num_to_keep and score <= worst_kept_score:
            return

        episode = self._episodes_total
        timesteps = self.model.num_timesteps
        fname = f"ckpt_ep{episode:06d}_ts{timesteps:08d}_{self._metric}{score:.4f}.zip"
        path = self._checkpoint_dir / fname

        self.model.save(str(path.with_suffix("")))  # SB3 .save appends .zip
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
