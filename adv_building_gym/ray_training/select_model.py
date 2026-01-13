"""
Ray RLlib model configuration and selection.

This module provides functions for configuring and selecting RL algorithms
for training with Ray RLlib, including PPO, SAC, DDPG, TD3, and A2C.
"""

import logging

from ray.rllib.algorithms.ppo import PPOConfig
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
        algorithm: RL algorithm to use (e.g., "ppo")
        episode_length: Episode length in timesteps

    Returns:
        Algorithm-specific config (before common_model_config applied)
    """

    # Hyperparameters for consistent evaluation across algorithms
    learning_rate = 3e-4
    batch_size = 64
    # TODO VP 2026.01.12. : What about these? -- they does not seem like something important...
    learning_starts = 10 * episode_length
    n_steps = episode_length

    if algorithm == "ppo":
        config = PPOConfig()
        # NOTE: These values should match SB3 PPO hyperparameters for fair comparison:
        #   - train_batch_size_per_learner: Total timesteps collected before training update
        #     SB3 equivalent: n_steps * num_envs = 288 * 4 = 1,152
        #     We use 4000 here to account for 2 env_runners collecting in parallel
        #   - minibatch_size: SGD minibatch size for gradient updates (like SB3's batch_size=64)
        #   - num_epochs: Number of passes over collected data (SB3 default is 10, we use 4)
        # TODO VP 2026.01.12. : look up rl_module config options, define own model -- in model_backbone module (?)
        config.rl_module(
            # Use new API to avoid RLModule(config=RLModuleConfig) deprecation warning
            # TODO VP 2026.01.12. : Use transformer model for better learning, it is a time series after all
            model_config=DefaultModelConfig(
                fcnet_activation='relu',
                fcnet_hiddens=[32, 32, 32],
                # use_lstm=True,
                # lstm_cell_size=5,
                # lstm_use_prev_action=True,
                # lstm_use_prev_reward=False,
            ),
        )

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

    # TODO VP 2025.12.11. : add more RL algos
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return config
