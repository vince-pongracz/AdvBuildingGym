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

from adv_building_gym.config.training_param_config import TrainingParamConfig

logger = logging.getLogger(__name__)

# Default path to the bundled training config YAML
_DEFAULT_TRAINING_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "training_param_config.yaml"


def select_model(
    algorithm: str,
    episode_length: int,
    training_config: TrainingParamConfig | None = None,
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
            size.  When *None* the bundled ``training_config.yaml`` is loaded.

    Returns:
        Algorithm-specific config (before common_model_config applied)
    """

    if training_config is None:
        training_config = TrainingParamConfig.from_yaml(_DEFAULT_TRAINING_CONFIG)

    learning_starts = 10 * episode_length

    # Algorithm-specific configuration
    if algorithm == "ppo":
        config = PPOConfig()
        # PPO is on-policy: env runners collect a full batch of experience,
        # then the learner runs multiple SGD epochs over that batch.
        #
        # train_batch_size_per_learner = total timesteps collected per iteration.
        # We express this in episodes (ppo_episodes_per_iteration) and convert
        # to timesteps here, so the user thinks in episodes, not raw timesteps.
        #
        # minibatch_size = SGD mini-batch within each epoch (subset of the
        # collected batch).  Smaller than train_batch_size_per_learner.
        ppo_batch_timesteps = training_config.ppo_episodes_per_iteration * episode_length
        config.training(
            lr=training_config.learning_rate,
            train_batch_size_per_learner=ppo_batch_timesteps,
            minibatch_size=training_config.ppo_minibatch_size,
            num_epochs=4,
            use_critic=True,
            use_gae=True,
            use_kl_loss=True,
            # NOTE VP 2026.01.12. : tune these and other hyperparameters later -- using tune
        )

    elif algorithm == "sac":
        config = SACConfig()
        # SAC is off-policy: experience is stored in a replay buffer and the
        # learner samples sac_replay_batch_size transitions per gradient step,
        # independent of episode boundaries.  This is fundamentally different
        # from PPO's episode-based batching — the "batch size" here is just how
        # many transitions are drawn from the buffer, not how much new data is
        # collected per iteration.
        # New API stack (default in RLlib 2.7+) requires EpisodeReplayBuffer
        # and separate learning rates for actor, critic, and alpha.
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
            train_batch_size_per_learner=training_config.sac_replay_batch_size,
            num_steps_sampled_before_learning_starts=learning_starts, # Number of steps to collect before starting learning (to fill up replay buffer)
            # Gradient clipping mitigates but does NOT fully prevent NaN in
            # the policy network. If the loss itself is NaN/Inf (e.g. from
            # extreme Q-values caused by large reward spikes like the -2.0
            # harsh penalty in OperatorEnergyControlReward), NaN propagates
            # into weights before grad_clip can act.  The root fix is keeping
            # per-step rewards in a bounded range (ideally [-1, 1] total).
            # See: slurm job 1624328 — crash at iter 48 with
            # "normal expects all elements of std >= 0.0".
            grad_clip=1.0,
        )
    # NOTE VP 2026.02.11. : Maybe add DreamerV3 -- but in that case drop the forecasting states
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
            # NOTE VP 2026.03.10. : What is the NN structure which is needed to learn this task complexity?
            fcnet_hiddens=[256, 256],
            # [256, 256, 256]
            # Use LSTM to exploit temporal dependencies
            # use_lstm=True,
            # lstm_cell_size=5,
            # lstm_use_prev_action=True,
            # lstm_use_prev_reward=False,
        ),
    )

    return config
