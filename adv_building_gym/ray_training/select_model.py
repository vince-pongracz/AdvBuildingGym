"""
Ray RLlib model configuration and selection.

This module provides functions for configuring and selecting RL algorithms
for training with Ray RLlib, including PPO and SAC.
"""

import logging

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
# NOTE: TD3 was removed from RLlib in v2.7 (moved to rllib_contrib, then discontinued Nov 2024).
# Use SAC instead - similar off-policy algorithm with entropy regularization.
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

logger = logging.getLogger(__name__)


def select_model(
    algorithm: str,
    episode_length: int,
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

    Returns:
        Algorithm-specific config (before common_model_config applied)
    """

    # Common hyperparameters for consistent evaluation across algorithms
    learning_rate = 3e-4
    batch_size = 64
    # TODO VP 2026.01.12. : What about these? -- they does not seem like something important...
    learning_starts = 10 * episode_length
    n_steps = episode_length

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
            lr=learning_rate,
            # TODO VP 2026.01.07. : Use timesteps param?
            # ~14 episodes worth of data before each training update
            train_batch_size_per_learner=4000,
            minibatch_size=batch_size,  # 64 - SGD minibatch size (same as SB3)
            # Number of epochs per training iteration (typical for PPO)
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
            actor_lr=learning_rate,
            critic_lr=learning_rate,
            alpha_lr=learning_rate,
            replay_buffer_config={
                "type": "EpisodeReplayBuffer",
                "capacity": 100000,
            },
            # SAC-specific hyperparameters
            twin_q=True,  # Use twin Q-networks to reduce overestimation bias
            initial_alpha=1.0,  # Initial entropy coefficient (auto-tuned)
            target_network_update_freq=1,  # Update target networks every step
            tau=0.005,  # Soft update coefficient for target networks
            train_batch_size_per_learner=256,  # Batch size sampled from replay buffer
            num_steps_sampled_before_learning_starts=learning_starts,
        )

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Supported: ppo, sac")

    # Common configuration for all algorithms
    # Network architecture: same for fair comparison across algorithms
    # TODO VP 2026.01.12. : look up rl_module config options, define own model -- in model_backbone module (?)
    config.rl_module(
        # Use new API to avoid RLModule(config=RLModuleConfig) deprecation warning
        # TODO VP 2026.01.12. : Use transformer model for better learning, it is a time series after all
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
