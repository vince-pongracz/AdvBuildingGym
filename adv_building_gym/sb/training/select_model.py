"""SB3 algorithm builder.

Mirrors :func:`adv_building_gym.ray.training.select_model.select_model` so
the same :class:`TrainingParamConfig` (loaded from the same trial YAML)
drives PPO and SAC under Stable-Baselines3.

Algorithm mapping
-----------------

PPO (on-policy)
    * ``ppo_episodes_per_iteration × EPISODE_LENGTH / num_envs`` → ``n_steps``
      (SB3's ``n_steps`` is *per env*, so we divide by the VecEnv width
      to keep the same per-iteration batch as RLlib).
    * ``ppo_minibatch_size`` → ``batch_size``
    * ``ppo_num_epochs`` → ``n_epochs``
    * GAE λ pinned to 0.95 (matches RLlib config).

SAC (off-policy)
    * ``sac_replay_batch_size`` → ``batch_size``.
    * ``sac_episodes_to_keep_in_replay_buffer × EPISODE_LENGTH`` → ``buffer_size``.
    * ``sac_learning_starts_after_n_episodes × EPISODE_LENGTH`` → ``learning_starts``.
    * UTD: SB3 SAC samples ``gradient_steps`` minibatches every
      ``train_freq`` env steps. We pick ``train_freq=1, gradient_steps=⌈UTD⌉``
      where ``UTD = sac_training_intensity / sac_replay_batch_size``.
      (UTD < 1 is rounded up to 1 since SB3 doesn't support fractional
      ratios in one knob.)
    * ``sac_n_step_return`` is **not honoured** — SB3 SAC has no n-step
      return option in its default ReplayBuffer. A warning is logged.
    * ``tau`` left at the SB3 default (0.005 — same as the RLlib build);
      ``gamma`` is taken from ``training_config.gamma`` for parity with Ray.
    * ``gradient_clip`` plumbed through ``policy_kwargs``-equivalent path
      (SB3 doesn't expose a top-level grad-clip for SAC; we use
      ``optimizer_kwargs`` if needed in the future).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch.nn as nn
from stable_baselines3 import PPO, SAC

from adv_building_gym.config.training.training_param_config import TrainingParamConfig

logger = logging.getLogger(__name__)


# Network architecture parity with the Ray default (DefaultModelConfig
# fcnet_hiddens=[256, 256], activation=tanh). SB3 accepts a list for shared
# trunks; PPO/SAC both honour the policy_kwargs format below.
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
    """Build a configured SB3 model for ``algorithm``.

    Args:
        algorithm: ``"ppo"`` or ``"sac"``. ``"dreamerv3"`` raises.
        vec_env: The training ``VecEnv`` (already wrapped in VecMonitor).
        episode_length: ``env_config.EPISODE_LENGTH`` — drives batch sizing.
        num_envs: VecEnv width; PPO's ``n_steps`` divides by this.
        training_config: Already-loaded trial training params.
        seed: Reproducibility seed (propagated to SB3's PRNG).
        device: ``"auto"`` / ``"cuda"`` / ``"cpu"``.
        tensorboard_log: Directory for TB event files; ``None`` disables.

    Returns:
        A constructed ``stable_baselines3`` model.
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
        # PPO total batch (across all envs) per update.
        # Link: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
        ppo_batch_timesteps = training_config.ppo_episodes_per_iteration * episode_length
        # SB3's ``n_steps`` is per environment, so divide by num_envs.
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
