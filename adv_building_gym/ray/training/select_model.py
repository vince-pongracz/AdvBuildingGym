"""Ray RLlib algorithm selection/config (PPO, SAC, DreamerV3)."""

import logging
from pathlib import Path

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.dreamerv3 import DreamerV3Config
# NOTE: TD3 was removed from RLlib in v2.7 (moved to rllib_contrib, then discontinued Nov 2024).
# Use SAC instead - similar off-policy algorithm with entropy regularization.
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

from adv_building_gym.config.training.training_param_config import TrainingParamConfig

logger = logging.getLogger(__name__)

# TODO VP 2026.06.10.: Remove this
# Default path to the bundled training config YAML
_DEFAULT_TRAINING_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "training_param_config.yaml"


def select_model(
    algorithm: str,
    episode_length: int,
    training_config: TrainingParamConfig | None = None,
):
    """Build the algorithm-specific config (ppo/sac/dreamerv3) with its hyperparameters.

    Caller then passes it to common_model_setup() for env/resources/callbacks.
    ``training_config`` None → loads the bundled ``training_config.yaml``.
    """

    if training_config is None:
        training_config = TrainingParamConfig.from_yaml(_DEFAULT_TRAINING_CONFIG)

    # Algorithm-specific configuration
    if algorithm == "ppo":
        config = PPOConfig()
        # PPO on-policy: collect a full batch, then multiple SGD epochs over it.
        # train_batch_size_per_learner = timesteps/iteration (ppo_episodes_per_iteration ×
        # episode_length); minibatch_size = SGD mini-batch within each epoch.
        ppo_batch_timesteps = training_config.ppo_episodes_per_iteration * episode_length
        config.training(
            # lr left at RLlib default: 5e-5
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
        # SAC off-policy: samples sac_replay_batch_size transitions per gradient step from
        # the replay buffer (not episode-batched like PPO). New API stack needs
        # EpisodeReplayBuffer and separate actor/critic/alpha LRs. rollout_fragment_length
        # defaults to 1 otherwise, reporting episode length 1 in callbacks.
        config.training(
            # NOTE VP 2026.02.11. : Actor critic methods SAC & PPO - blog
            # Link: https://joel-baptista.github.io/phd-weekly-report/posts/ac/
            # actor_lr left at RLlib default 3e-5 (LR of the policy network)
            # critic_lr left at RLlib default 3e-4 (LR of the critic network)
            # alpha_lr left at RLlib default 3e-4 (weight of entropy -- exploration)
            alpha_lr = 0.0, # Keep it fixed (no auto-tuning)
            # PrioritizedEpisodeReplayBuffer crashes on Ray 2.52.1 (sum-tree KeyError at
            # zero priorities); use uniform EpisodeReplayBuffer until fixed.
            # Link: https://github.com/ray-project/ray/issues/50966
            replay_buffer_config={
                "type": "EpisodeReplayBuffer",
                "capacity": episode_length * training_config.sac_episodes_to_keep_in_replay_buffer,
            },
            # SAC-specific hyperparameters
            twin_q=True,  # Use twin Q-networks to reduce overestimation bias. RLlib default
            # NOTE VP 2026.06.04.: initial_alpha=0.2 is in the SB3 default
            initial_alpha=0.2,  # Initial entropy coefficient (auto-tuned via alpha_lr). RLlib default
            target_entropy=0.0,  # Target entropy for automatic alpha tuning. RLlib default: "auto" = -action_dim
            # target_network_update_freq=1,  # Update target networks every step. RLlib default: 0
            n_step=training_config.sac_n_step_return,  # RLlib default: 1
            tau=0.005,  # Soft update coefficient for target networks (at Polyak averaging). RLlib default
            train_batch_size_per_learner=training_config.sac_replay_batch_size,  # RLlib default: 256
            # training_intensity = replayed/sampled steps; UTD = intensity / batch_size.
            # Default [1, 1] round-robin gives UTD ≈ 0.001; standard SAC uses ≈ 1.0.
            # Link: https://arxiv.org/abs/1802.09477
            training_intensity=training_config.sac_training_intensity,  # RLlib default: None
            num_steps_sampled_before_learning_starts=training_config.sac_learning_starts_after_n_episodes * episode_length, # Warm up replay buffer with N episodes before learning starts.
            # grad_clip mitigates but doesn't fully prevent NaN: if the loss is NaN/Inf
            # (extreme Q from reward spikes) it reaches weights first. Root fix = bounded
            # per-step rewards (~[-1, 1]). See slurm job 1624328 (crash iter 48,
            # "normal expects all elements of std >= 0.0").
            # grad_clip=1.0,  # RLlib default: None
        )
    elif algorithm == "dreamerv3":
        config = DreamerV3Config()
        # DreamerV3 model-based off-policy: world model (RSSM) fit on batch_length_T sequences,
        # actor/critic on imagined horizon_H rollouts; training_ratio = UTD analogue. Ships its
        # own RLModule, so the DefaultModelConfig block below is skipped.
        # Link: https://arxiv.org/pdf/2301.04104
        config.training(
            model_size=training_config.dreamerv3_model_size,
            training_ratio=training_config.dreamerv3_training_ratio,
            batch_size_B=training_config.dreamerv3_batch_size_B,
            batch_length_T=training_config.dreamerv3_batch_length_T,
            horizon_H=training_config.dreamerv3_horizon_H,
            world_model_lr=training_config.dreamerv3_world_model_lr,
            actor_lr=training_config.dreamerv3_actor_lr,
            critic_lr=training_config.dreamerv3_critic_lr,
            replay_buffer_config={
                "type": "EpisodeReplayBuffer",
                "capacity": episode_length * training_config.dreamerv3_episodes_to_keep_in_replay_buffer,
            },
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Supported: ppo, sac, dreamerv3")

    # Common configuration for PPO/SAC. DreamerV3 ships its own RLModule and
    # ignores DefaultModelConfig.
    if algorithm in ("ppo", "sac"):
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
