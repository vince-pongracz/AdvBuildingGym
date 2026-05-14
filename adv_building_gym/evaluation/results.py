"""Data structures for evaluation results.

Provides ``EpisodeStats`` (per-episode metrics) and ``EvalResults``
(aggregated summary with persistence helpers).
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from adv_building_gym.utils import CustomJSONEncoder

logger = logging.getLogger(__name__)


@dataclass
class EpisodeStats:
    """Metrics collected for a single evaluation episode."""

    episode: int
    length: int
    total_reward: float
    achieved_reward: float
    max_achievable_reward: float
    reward_rate: float
    seed: int
    data_variant: dict[str, str] | None = None
    episode_date: str | None = None

    def to_dict(self) -> dict:
        """Return a plain dict (JSON-serialisable)."""
        d = {
            "episode": self.episode,
            "length": self.length,
            "total_reward": self.total_reward,
            "achieved_reward": self.achieved_reward,
            "max_achievable_reward": self.max_achievable_reward,
            "reward_rate": self.reward_rate,
            "seed": self.seed,
        }
        if self.data_variant is not None:
            d["data_variant"] = self.data_variant
        if self.episode_date is not None:
            d["episode_date"] = self.episode_date
        return d


@dataclass
class EvalResults:
    """Aggregated evaluation results with persistence helpers.

    Use the :meth:`from_episodes` classmethod to construct from a list of
    ``EpisodeStats`` together with run metadata.
    """

    episodes: list[EpisodeStats]
    checkpoint_path: str
    trial_name: str
    algorithm: str | None
    seed: int
    eval_time_seconds: float
    error: str | None = None
    output_dir: str | None = None

    # Summary statistics (populated by from_episodes)
    mean_reward: float = 0.0
    std_reward: float = 0.0
    min_reward: float = 0.0
    max_reward: float = 0.0
    mean_reward_rate: float = 0.0
    std_reward_rate: float = 0.0

    @classmethod
    def from_episodes(
        cls,
        episodes: list[EpisodeStats],
        checkpoint_path: str,
        trial_name: str,
        algorithm: str | None,
        seed: int,
        eval_time_seconds: float,
        error: str | None = None,
    ) -> "EvalResults":
        """Construct ``EvalResults`` and compute summary statistics."""
        result = cls(
            episodes=episodes,
            checkpoint_path=checkpoint_path,
            trial_name=trial_name,
            algorithm=algorithm,
            seed=seed,
            eval_time_seconds=eval_time_seconds,
            error=error,
        )
        if episodes and error is None:
            rewards = [ep.total_reward for ep in episodes]
            rates = [ep.reward_rate for ep in episodes]
            result.mean_reward = float(np.mean(rewards))
            result.std_reward = float(np.std(rewards))
            result.min_reward = float(np.min(rewards))
            result.max_reward = float(np.max(rewards))
            result.mean_reward_rate = float(np.mean(rates))
            result.std_reward_rate = float(np.std(rates))
        return result

    def to_dict(self) -> dict:
        """Return a backward-compatible dict matching the old script output."""
        d: dict = {
            "checkpoint_path": self.checkpoint_path,
            "trial_name": self.trial_name,
            "algorithm": self.algorithm,
            "num_episodes": len(self.episodes),
            "seed": self.seed,
            "eval_time_seconds": self.eval_time_seconds,
        }
        if self.error is not None:
            d["error"] = self.error
        else:
            d.update({
                "mean_reward": self.mean_reward,
                "std_reward": self.std_reward,
                "min_reward": self.min_reward,
                "max_reward": self.max_reward,
                "mean_reward_rate": self.mean_reward_rate,
                "std_reward_rate": self.std_reward_rate,
            })
        d["episodes"] = [ep.to_dict() for ep in self.episodes]
        return d

    def save(self, output_dir: str) -> None:
        """Write JSON and CSV files to *output_dir*."""
        os.makedirs(output_dir, exist_ok=True)

        checkpoint_name = Path(self.checkpoint_path).name
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        results_file = os.path.join(
            output_dir, f"eval_{checkpoint_name}_{timestamp}.json"
        )
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, cls=CustomJSONEncoder, indent=4)
        logger.info("Results saved to: %s", results_file)

        csv_file = results_file.replace(".json", ".csv")
        df = pd.DataFrame([ep.to_dict() for ep in self.episodes])
        df.to_csv(csv_file, index=False)
        logger.info("Episode data saved to: %s", csv_file)

    def log_summary(self) -> None:
        """Log evaluation summary statistics."""
        logger.info("=" * 70)
        logger.info("Evaluation Summary")
        if self.error is None:
            logger.info(
                "  Mean Reward: %.2f +/- %.2f",
                self.mean_reward, self.std_reward,
            )
            logger.info(
                "  Mean Reward Rate: %.4f +/- %.4f",
                self.mean_reward_rate, self.std_reward_rate,
            )
            logger.info(
                "  Min/Max Reward: %.2f / %.2f",
                self.min_reward, self.max_reward,
            )
        else:
            logger.info("  Error: %s", self.error)
        logger.info(
            "  Evaluation time: %.2f seconds", self.eval_time_seconds,
        )
        logger.info("=" * 70)
