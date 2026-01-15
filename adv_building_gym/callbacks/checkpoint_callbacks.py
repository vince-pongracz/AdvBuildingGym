"""
Checkpoint callbacks for Ray RLlib training.

This module provides checkpoint management callbacks similar to Stable Baselines3's
checkpoint callbacks, but it is designed for Ray RLlib's callback system.
"""

import os
import json
import logging
import shutil
from typing import Optional, Type

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks

logger = logging.getLogger(__name__)


def make_checkpoint_callback_class(
    checkpoint_dir: str,
    checkpoint_frequency: int = 20,
    num_to_keep: int = 1,
    metric: str = "env_runners/reward_rate",
) -> Type["BestModelCheckpointCallback"]:
    """
    Factory function that returns a configured BestModelCheckpointCallback class.

    Important: RLlib's config.callbacks() expects a **class type**, not an instance or factory function.
    This function returns a class.

    Args:
        checkpoint_dir: Directory to save checkpoints
        checkpoint_frequency: Save checkpoint every N episodes (default: 20)
        num_to_keep: Number of best checkpoints to keep (default: 1)
        metric: Metric name to optimize (default: "env_runners/reward_rate")

    Returns:
        A BestModelCheckpointCallback subclass with parameters pre-configured

    Example:
        # Use with on_episode_end parameter for separate episode metrics callback
        config.callbacks(
            make_checkpoint_callback_class(
                checkpoint_dir="/path/to/checkpoints",
                checkpoint_frequency=20,
                metric="env_runners/reward_rate",
            ),
            on_episode_end=my_episode_end_callback,  # Separate callback
        )
    """

    class ConfiguredBestModelCheckpointCallback(BestModelCheckpointCallback):
        """Pre-configured BestModelCheckpointCallback with parameters from closure."""

        def __init__(self):
            super().__init__(
                checkpoint_dir=checkpoint_dir,
                checkpoint_frequency=checkpoint_frequency,
                num_to_keep=num_to_keep,
                metric=metric,
            )

    return ConfiguredBestModelCheckpointCallback


class BestModelCheckpointCallback(DefaultCallbacks):
    """
    Custom RLlib callback to save checkpoints every N episodes and keep only the best model.

    Similar to the SB3 BestModelCheckpointCallback, this callback:
    - Saves checkpoints every checkpoint_frequency episodes
    - Tracks the best model based on evaluation metric
    - Saves new best model first, then deletes old one
    - Keeps only num_to_keep best checkpoints

    Note: This runs on the callback level (like SB3), separate from Ray Tune's
    framework-level checkpointing (configured via CheckpointConfig)

    Important: on_train_result runs on the Algorithm actor (main process), while
    on_episode_end runs on EnvRunner actors (workers). State is NOT shared between
    these instances. We use episode counts from the result dict instead of self-tracking.

    For episode metrics logging, use config.callbacks() with the on_episode_end parameter:
        config.callbacks(
            CheckpointCallbackClass,
            on_episode_end=my_episode_metrics_callback,
        )

    Args:
        checkpoint_dir: Directory to save checkpoints
        checkpoint_frequency: Save checkpoint every N episodes (default: 20)
        num_to_keep: Number of best checkpoints to keep (default: 1)
        metric: Metric name to optimize (default: "env_runners/reward_rate")
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_frequency: int = 20,
        num_to_keep: int = 1,
        metric: str = "env_runners/reward_rate",
    ):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency
        self.num_to_keep = num_to_keep
        self.metric = metric

        # Internal state tracking (on Algorithm actor only)
        self.best_metric_value: float = -np.inf
        self.current_best_path: Optional[str] = None
        self.last_checkpoint_episode: int = 0

        # Note: Checkpoint directory (self.checkpoint_dir) will be created lazily on first checkpoint save
        logger.info(
            "BestModelCheckpointCallback initialized: checkpoint_dir=%s, frequency=%d episodes, metric=%s, num_to_keep=%d",
            checkpoint_dir,
            checkpoint_frequency,
            metric,
            num_to_keep,
        )

    def on_train_result(
        self, *, algorithm, metrics_logger=None, result: dict, **kwargs
    ):
        """
        Called after each training iteration with aggregated metrics.
        Check if we should save a checkpoint based on episode count from result dict.

        Note: We get episode count from result dict because on_train_result runs
        on Algorithm actor, while on_episode_end runs on EnvRunner workers.
        State is not shared between these different callback instances.
        """
        # Get total episode count from result dict (aggregated from all workers)
        # Use num_episodes_lifetime (cumulative total), NOT num_episodes (per-iteration)
        env_runners_data = result.get("env_runners", {})
        episode_count = env_runners_data.get(
            "num_episodes_lifetime",
            result.get("episodes_total", result.get("episodes_this_iter", 0))
        )

        # Check if we should checkpoint based on episode count threshold
        episodes_since_last = episode_count - self.last_checkpoint_episode
        if episode_count >= self.checkpoint_frequency and episodes_since_last >= self.checkpoint_frequency:
            self._save_checkpoint(algorithm, result, episode_count)
            self.last_checkpoint_episode = episode_count

    def _save_checkpoint(self, algorithm, result: dict, episode_count: int) -> None:
        """
        Save checkpoint if current model is better than previous best.

        Args:
            algorithm: The RLlib algorithm instance
            result: Dictionary containing training metrics
            episode_count: Current total episode count from result dict
        """

        # Extract the metric value from the result -- go "deeper" in the result dict with a loop
        # Metrics are nested: result["env_runners"]["reward_rate"]
        metric_path = self.metric.split("/")
        current_metric_value = result
        for key in metric_path:
            current_metric_value = current_metric_value.get(key, -np.inf)
            if not isinstance(current_metric_value, dict) and current_metric_value == -np.inf:
                break  # metric (something numeric) reached (hopefully)

        if not isinstance(current_metric_value, (int, float)):  # no value for metric yet
            current_metric_value = -np.inf

        logger.info(
            "Episode %d: Checking checkpoint (current_%s=%.4f, best_%s=%.4f)",
            episode_count,
            self.metric,
            current_metric_value,
            self.metric,
            self.best_metric_value,
        )

        # Check if this is a new best model
        if current_metric_value > self.best_metric_value:
            # Create checkpoint directory on first save (lazy creation)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            # Save old checkpoint path for deletion after new one is saved
            old_best_path = self.current_best_path

            # Save new best checkpoint 1st
            checkpoint_name = f"best_model_ep{episode_count}_{self.metric.replace('/', '_')}{current_metric_value:.4f}"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

            algorithm.save(checkpoint_path)
            self.best_metric_value = current_metric_value
            self.current_best_path = checkpoint_path

            logger.info(
                "New best model saved at episode %d with %s %.4f: %s",
                episode_count,
                self.metric,
                current_metric_value,
                checkpoint_path,
            )

            # Save metadata
            metadata = {
                "episode": episode_count,
                "metric": self.metric,
                "best_metric_value": float(current_metric_value),
                "num_timesteps": result.get("num_env_steps_sampled_lifetime", 0),
                "checkpoint_path": checkpoint_path,
            }
            metadata_path = os.path.join(
                self.checkpoint_dir, "best_checkpoint_metadata.json"
            )
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)

            # Now delete old best checkpoint AFTER new one is safely saved
            if old_best_path is not None and os.path.exists(old_best_path):
                logger.info("Deleting old best checkpoint: %s", old_best_path)
                try:
                    # RLlib saves checkpoints as directories, not single files
                    if os.path.isdir(old_best_path):
                        shutil.rmtree(old_best_path)
                    else:
                        os.remove(old_best_path)
                except OSError as e:
                    logger.warning("Failed to delete old checkpoint: %s", e)
