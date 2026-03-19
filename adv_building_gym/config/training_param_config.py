"""Training hyperparameter configuration.

Provides a dataclass that holds training hyperparameters and can be loaded
from a YAML file.
Fields are split by algorithm where their semantics differ.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class TrainingParamConfig:
    """Training hyperparameters, split by algorithm where semantics differ.

    PPO (on-policy) collects a batch of complete episodes before each policy
    update.  The batch size is therefore expressed in episodes
    (``ppo_episodes_per_iteration``), converted to timesteps via
    ``ppo_episodes_per_iteration × episode_length`` in ``select_model``.
    Within each update, SGD iterates over mini-batches of
    ``ppo_minibatch_size`` timesteps.

    SAC (off-policy) stores all experience in a replay buffer and samples
    ``sac_replay_batch_size`` transitions per gradient step, independent of
    episode boundaries.

    Attributes:
        learning_rate: Learning rate for optimiser(s).
        ppo_episodes_per_iteration: How many full episodes PPO collects
            before one policy update (on-policy batch, in episode units).
        ppo_minibatch_size: SGD mini-batch size within each PPO epoch
            (in timesteps).
        sac_replay_batch_size: Number of transitions sampled from the
            replay buffer per SAC gradient step.
    """

    learning_rate: float = 3e-4
    ppo_episodes_per_iteration: int = 25
    ppo_minibatch_size: int = 64
    sac_replay_batch_size: int = 256
    seed: int = 42

    @staticmethod
    def from_yaml(path: str | Path) -> "TrainingParamConfig":
        """Load training config from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            TrainingConfig populated from the file.
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return TrainingParamConfig(**data)
