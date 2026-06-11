"""SB3 algorithm builder — same :class:`TrainingParamConfig` drives PPO and SAC as the Ray side.

PPO: 
``n_steps = ppo_episodes_per_iteration × EPISODE_LENGTH / num_envs`` (SB3 n_steps is per-env),
``ppo_minibatch_size`` → batch_size, 
``ppo_num_epochs`` → n_epochs, 
GAE λ=0.95.
SAC: 
``sac_replay_batch_size`` → batch_size; 
buffer/learning_starts = episodes × EPISODE_LENGTH;
``train_freq=1, gradient_steps=⌈UTD⌉`` (UTD = sac_training_intensity / batch_size, <1 → 1).
``sac_n_step_return`` is NOT honoured (SB3 default ReplayBuffer); tau=0.005,
gamma from config.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch.nn as nn
from stable_baselines3 import PPO, SAC

from adv_building_gym.config.training.training_param_config import TrainingParamConfig

logger = logging.getLogger(__name__)


# net-arch parity with the Ray default ([256, 256], tanh)
_DEFAULT_POLICY_KWARGS: dict[str, Any] = {
    "net_arch": [256, 256],
    "activation_fn": nn.Tanh,
}


def sb_select_model(
    algorithm: str,
    *,
    vec_env,
    episode_length: int,
    num_envs: int,
    training_config: TrainingParamConfig,
    seed: int,
    device: str = "auto",
    tensorboard_log: str | None = None,
):
    """Build a configured SB3 model (``"ppo"``/``"sac"``; ``"dreamerv3"`` raises).

    ``episode_length`` / ``num_envs`` drive batch sizing; ``seed`` propagates to SB3's PRNG;
    ``tensorboard_log`` None disables TB.
    """
    if algorithm == "dreamerv3":
        raise ValueError(
            "SB3 does not ship DreamerV3 — use run_train_ray.py for that "
            "algorithm. The trial config 'algorithm' must be 'ppo' or 'sac' "
            "for the SB3 driver."
        )

    common_kwargs: dict[str, Any] = {
        "policy": "MultiInputPolicy",  # AdvBuildingGym has a Dict obs space
        "env": vec_env,
        "verbose": 1,
        "seed": seed,
        "device": device,
        "tensorboard_log": tensorboard_log,
        "policy_kwargs": dict(_DEFAULT_POLICY_KWARGS),
    }

    if algorithm == "ppo":
        # PPO total batch per update; SB3 n_steps is per-env, so divide by num_envs.
        # Link: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
        ppo_batch_timesteps = training_config.ppo_episodes_per_iteration * episode_length
        n_steps = max(1, ppo_batch_timesteps // max(1, num_envs))
        if n_steps * num_envs != ppo_batch_timesteps:
            logger.warning(
                "PPO: ppo_episodes_per_iteration*EPISODE_LENGTH=%d not evenly "
                "divisible by num_envs=%d; using n_steps=%d (effective batch %d).",
                ppo_batch_timesteps, num_envs, n_steps, n_steps * num_envs,
            )
        model = PPO(
            **common_kwargs,
            n_steps=n_steps,
            batch_size=training_config.ppo_minibatch_size,
            n_epochs=training_config.ppo_num_epochs,
            gae_lambda=0.95,
            gamma=training_config.gamma,  # parity with RLlib config.training(gamma=...)
        )
        logger.info(
            "PPO built: n_steps=%d (per env), batch_size=%d, n_epochs=%d",
            n_steps, training_config.ppo_minibatch_size,
            training_config.ppo_num_epochs,
        )
        return model

    if algorithm == "sac":
        buffer_size = episode_length * training_config.sac_episodes_to_keep_in_replay_buffer
        learning_starts = episode_length * training_config.sac_learning_starts_after_n_episodes

        # UTD parity with RLlib's training_intensity knob.
        # Link: https://arxiv.org/abs/1802.09477
        intensity = training_config.sac_training_intensity
        if intensity is None:
            gradient_steps = 1
        else:
            utd = float(intensity) / float(training_config.sac_replay_batch_size)
            gradient_steps = max(1, int(math.ceil(utd)))
            if utd < 1.0:
                logger.warning(
                    "SAC: requested UTD=%.3f rounded up to gradient_steps=1 "
                    "(SB3 has no fractional UTD knob).", utd,
                )

        if training_config.sac_n_step_return != 1:
            logger.warning(
                "SAC: sac_n_step_return=%d is NOT honoured by SB3's default "
                "ReplayBuffer (Ray RLlib supports n-step; SB3 does not).",
                training_config.sac_n_step_return,
            )

        model = SAC(
            **common_kwargs,
            batch_size=training_config.sac_replay_batch_size,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            train_freq=1,
            gradient_steps=gradient_steps,
            tau=0.005,
            gamma=training_config.gamma,  # parity with RLlib config.training(gamma=...)
            ent_coef="auto",
        )
        logger.info(
            "SAC built: batch_size=%d, buffer_size=%d, learning_starts=%d, "
            "train_freq=1, gradient_steps=%d",
            training_config.sac_replay_batch_size, buffer_size, learning_starts,
            gradient_steps,
        )
        return model

    raise ValueError(f"Unknown algorithm: {algorithm}. Supported: ppo, sac")
