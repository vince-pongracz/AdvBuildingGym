"""
Ray RLlib model configuration and selection.

This module provides functions for configuring and selecting RL algorithms
for training with Ray RLlib, including PPO and SAC.
"""

import logging
from pathlib import Path

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
# NOTE: TD3 was removed from RLlib in v2.7 (moved to rllib_contrib, then discontinued Nov 2024).
# Use SAC instead - similar off-policy algorithm with entropy regularization.
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

from adv_building_gym.config.training_config import TrainingConfig

logger = logging.getLogger(__name__)

# Default path to the bundled training config JSON
_DEFAULT_TRAINING_CONFIG = Path(__file__).resolve().parent.parent / "config" / "training_config.json"


def select_model(
    algorithm: str,
    episode_length: int,
    training_config: TrainingConfig | None = None,
):
    """
    Selects and configures algorithm-specific settings for training.

    This function creates the algorithm-specific configuration (e.g., PPO, SAC)
    with algorithm-specific hyperparameters and training settings. The returned
    config should be passed to common_model_config() by the caller for common
    configuration (environment, resources, callbacks, etc.).

    Args:
        algorithm: RL algorithm to use ("ppo" or "sac")
        episode_length: Episode length in timesteps
        training_config: Optional TrainingConfig with learning rate and batch
            size.  When *None* the bundled ``training_config.json`` is loaded.

    Returns:
        Algorithm-specific config (before common_model_config applied)
    """

    if training_config is None:
        training_config = TrainingConfig.from_json(_DEFAULT_TRAINING_CONFIG)

    learning_starts = 10 * episode_length

    # Algorithm-specific configuration
    if algorithm == "ppo":
        config = PPOConfig()
        # NOTE: These values should match SB3 PPO hyperparameters for fair comparison:
        #   - train_batch_size_per_learner: Total timesteps collected before training update
        #     SB3 equivalent: n_steps * num_envs = 288 * 4 = 1,152
        #     We use 4000 here to account for 2 env_runners collecting in parallel
        #   - minibatch_size: SGD minibatch size for gradient updates (like SB3's batch_size=64)
        #   - num_epochs: Number of passes over collected data (SB3 default is 10, we use 4)
        config.training(
            lr=training_config.learning_rate,
            train_batch_size_per_learner=training_config.batch_size,
            # Number of epochs per training iteration (typical for PPO)
            # --> num_epochs * batch_size steps in the env before policy update
            num_epochs=4,
            use_critic=True,
            use_gae=True,
            use_kl_loss=True,
            # NOTE VP 2026.01.12. : tune these and other hyperparameters later -- using tune
        )

    elif algorithm == "sac":
        config = SACConfig()
        # SAC is an off-policy actor-critic algorithm with entropy regularization
        # NOTE: SAC uses replay buffer instead of on-policy trajectories like PPO
        # New API stack (default in RLlib 2.7+) requires EpisodeReplayBuffer
        # New API stack requires separate learning rates for actor, critic, and alpha
        config.training(
            # NOTE VP 2026.02.11. : Actor critic methods SAC & PPO - blog
            # Link: https://joel-baptista.github.io/phd-weekly-report/posts/ac/
            actor_lr=training_config.learning_rate,  # LR of the policy network
            critic_lr=training_config.learning_rate,  # LR of the critic network
            alpha_lr=training_config.learning_rate,  # Influences weight of entropy -- and thus exploration
            replay_buffer_config={
                "type": "EpisodeReplayBuffer",
                "capacity": 100000,
            },
            # SAC-specific hyperparameters
            twin_q=True,  # Use twin Q-networks to reduce overestimation bias
            initial_alpha=1.0,  # Initial entropy coefficient (auto-tuned)
            target_network_update_freq=4,  # Update target networks every step
            tau=0.005,  # Soft update coefficient for target networks (at Polyak averaging)
            train_batch_size_per_learner=training_config.batch_size,  # Batch size sampled from replay buffer
            num_steps_sampled_before_learning_starts=learning_starts, # Number of steps to collect before starting learning (to fill up replay buffer)
        )
    # NOTE VP 2026.02.11. : Maybe add DreamerV3 -- but in that case drop the forecast states
    # DreamerV3 paper link: https://arxiv.org/pdf/2301.04104
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Supported: ppo, sac")

    # Common configuration for all algorithms
    # Network architecture: same for fair comparison across algorithms
    # TODO VP 2026.01.12. : look up rl_module config options, define own model -- in model_backbone module (?)
    config.rl_module(
        # Use new API to avoid RLModule(config=RLModuleConfig) deprecation warning
        # TODO VP 2026.01.12. : Use transformer model for better learning, it is a time series after all -- but does it really matter here?
        model_config=DefaultModelConfig(
            fcnet_activation='relu',
            fcnet_hiddens=[32, 32, 32],
            # Use LSTM to exploit temporal dependencies
            # use_lstm=True,
            # lstm_cell_size=5,
            # lstm_use_prev_action=True,
            # lstm_use_prev_reward=False,
        ),
    )

    return config
