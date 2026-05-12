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
            lr=training_config.learning_rate,  # RLlib default: 5e-5
            train_batch_size_per_learner=ppo_batch_timesteps,  # RLlib default: 4000
            minibatch_size=training_config.ppo_minibatch_size,  # RLlib default: 128
            num_epochs=training_config.ppo_num_epochs,  # RLlib default: 30
            use_critic=True,  # RLlib default
            use_gae=True,  # RLlib default
            lambda_=0.95, # GAE lambda, 0 means 1 step return, 1.0 means infinite step return, limited by the rollout length. # RLlib default: 1
            use_kl_loss=True,  # RLlib default
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
        # Collect complete episodes before returning to learner.
        # Without this, SAC defaults rollout_fragment_length to 1, causing
        # training episodes to be reported as length = 1 in callbacks.
        config.training(
            # NOTE VP 2026.02.11. : Actor critic methods SAC & PPO - blog
            # Link: https://joel-baptista.github.io/phd-weekly-report/posts/ac/
            # actor_lr=training_config.learning_rate,  # LR of the policy network. RLlib default: 3e-5
            # critic_lr=training_config.learning_rate,  # LR of the critic network. RLlib default: 3e-4
            # alpha_lr=training_config.learning_rate,  # Influences weight of entropy -- and thus exploration. RLlib default: 3e-4
            # PrioritizedEpisodeReplayBuffer crashes on Ray 2.52.1 with
            # KeyError in sum-tree when priorities degenerate to zero.
            # Use uniform EpisodeReplayBuffer until the bug is fixed upstream.
            # Link: https://github.com/ray-project/ray/issues/50966
            replay_buffer_config={
                "type": "EpisodeReplayBuffer",
                "capacity": episode_length * training_config.sac_episodes_to_keep_in_replay_buffer,
            },
            # SAC-specific hyperparameters
            twin_q=True,  # Use twin Q-networks to reduce overestimation bias. RLlib default
            initial_alpha=1.0,  # Initial entropy coefficient (auto-tuned via alpha_lr). RLlib default
            # target_network_update_freq=1,  # Update target networks every step. RLlib default: 0
            n_step=training_config.sac_n_step_return,  # RLlib default: 1
            tau=0.005,  # Soft update coefficient for target networks (at Polyak averaging). RLlib default
            train_batch_size_per_learner=training_config.sac_replay_batch_size,  # RLlib default: 256
            # training_intensity = replayed_steps / sampled_steps.
            # Without this, RLlib defaults to [1, 1] round-robin: only
            # 1 gradient update per ~864 sampled env steps (UTD ≈ 0.001).
            # Standard SAC uses UTD ≈ 1.0 (1 grad step per env step).
            # UTD = training_intensity / batch_size.
            # Link: https://arxiv.org/abs/1802.09477
            training_intensity=training_config.sac_training_intensity,  # RLlib default: None
            num_steps_sampled_before_learning_starts=training_config.sac_learning_starts_after_n_episodes * episode_length, # Warm up replay buffer with N episodes before learning starts.
            # Gradient clipping mitigates but does NOT fully prevent NaN in
            # the policy network. If the loss itself is NaN/Inf (e.g. from
            # extreme Q-values caused by large reward spikes like the -2.0
            # harsh penalty in OperatorEnergyControlReward), NaN propagates
            # into weights before grad_clip can act.  The root fix is keeping
            # per-step rewards in a bounded range (ideally [-1, 1] total).
            # See: slurm job 1624328 — crash at iter 48 with
            # "normal expects all elements of std >= 0.0".
            grad_clip=1.0,  # RLlib default: None
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
            fcnet_activation='tanh', # RLlib default: tanh
            # NOTE VP 2026.03.10. : What is the NN structure which is needed to learn this task complexity?
            fcnet_hiddens=[256, 256],  # RLlib default: [256, 256]
            # [256, 256, 256]
            # Use LSTM to exploit temporal dependencies
            # use_lstm=True,
            # lstm_cell_size=5,
            # lstm_use_prev_action=True,
            # lstm_use_prev_reward=False,
        ),
    )

    return config
