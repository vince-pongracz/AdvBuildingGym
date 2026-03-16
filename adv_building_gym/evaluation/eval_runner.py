"""Evaluation orchestrator for Ray/RLlib trained models.

Composes checkpoint loading, environment creation, inference, and results
aggregation into a single ``evaluate_model()`` entry point.
"""

import logging
import os
import signal
import time

import numpy as np
import ray

from adv_building_gym.data_combinator import DataCombinator
from adv_building_gym.envs import AdvBuildingGym
from adv_building_gym.ray_training.rl_module_inference import (
    flatten_observation,
    infer_action,
    load_rl_module,
)
from adv_building_gym.utils import TrajectoryCollector

from .results import EpisodeStats, EvalResults

logger = logging.getLogger(__name__)


def _timeout_handler(signum, frame):
    raise TimeoutError("Evaluation timed out")


def evaluate_model(
    checkpoint_path: str,
    active_config,
    num_episodes: int = 1,
    seed: int = 42,
    save_results: bool = True,
    output_dir: str = "eval_results",
    log_trajectories: bool = True,
    algorithm_hint: str | None = None,
    timeout_seconds: int = 300,
    data_combinator: DataCombinator | None = None,
) -> EvalResults:
    """Evaluate a Ray/RLlib trained model on AdvBuildingGym.

    Uses CPU-only inference to avoid GPU resource over-subscription.
    Policy weights are loaded directly via ``RLModule.from_checkpoint()``
    without spawning training actors.

    Args:
        checkpoint_path: Absolute path to the Ray checkpoint directory.
        active_config: Config object with infras, statesources, rewards,
            building_props.
        num_episodes: Number of evaluation episodes.
        seed: Random seed for reproducibility.
        save_results: Whether to persist results to disk.
        output_dir: Directory for result files.
        log_trajectories: Whether to save per-step trajectory JSON.
        algorithm_hint: Algorithm name for metadata (informational only).
        timeout_seconds: Maximum wall-clock seconds before aborting.
        data_combinator: Optional DataCombinator for variant scheduling.

    Returns:
        ``EvalResults`` with per-episode stats and summary.
    """
    logger.info("=" * 70)
    logger.info("Starting Ray model evaluation")
    logger.info("  Checkpoint: %s", checkpoint_path)
    logger.info("  Config: %s", active_config.config_name)
    logger.info("  Episodes: %d", num_episodes)
    logger.info("  Seed: %d", seed)
    logger.info("=" * 70)

    # Initialize Ray with minimal resources for CPU-only inference.
    # The [32,32,32] policy network runs fast enough on CPU.
    if not ray.is_initialized():
        # Silence Ray's future warning about overriding accelerator env var
        os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
        ray.init(
            num_cpus=2,
            num_gpus=0,
            ignore_reinit_error=True,
            logging_level=logging.WARNING,
        )

    # Load RLModule from checkpoint
    logger.info("Loading algorithm from checkpoint...")
    rl_module = load_rl_module(checkpoint_path)

    # Create evaluation environment
    logger.info("Creating evaluation environment...")
    env = AdvBuildingGym(
        infras=active_config.infras,
        statesources=active_config.statesources,
        rewards=active_config.rewards,
        building_props=active_config.building_props,
        training=False,
        data_combinator=data_combinator,
    )

    if log_trajectories:
        env.log_full_info = True

    collector = TrajectoryCollector(env) if log_trajectories else None
    max_reward_per_step = sum(
        r.weight * r.max_reward for r in active_config.rewards
    )

    episode_stats: list[EpisodeStats] = []
    start_time = time.time()

    # Set up timeout
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)

    error: str | None = None

    try:
        MAX_STEPS_PER_EPISODE = 1000
        for ep in range(num_episodes):
            logger.info("=" * 50)
            episode_seed = seed + ep
            logger.info("Episode %d/%d (seed: %d)", ep + 1, num_episodes, episode_seed)

            obs, reset_info = env.reset(seed=episode_seed)
            ep_data_variant = reset_info.get("data_variant")
            ep_episode_date = reset_info.get("episode_date")
            if ep_data_variant:
                logger.info(
                    "  Variant: %s | Date: %s", ep_data_variant, ep_episode_date,
                )
            done = False
            episode_reward = 0.0
            episode_length = 0
            episode_rewards: list[float] = []

            if collector is not None:
                collector.reset()
                collector.on_reset(reset_info)

            while not done and episode_length < MAX_STEPS_PER_EPISODE:
                flat_obs = flatten_observation(obs)
                raw_action = infer_action(rl_module, flat_obs)

                next_obs, reward, terminated, truncated, step_info = env.step(
                    raw_action,
                )

                if collector is not None:
                    collector.on_step(
                        step=episode_length,
                        obs=obs,
                        action=raw_action,
                        reward=reward,
                        info=step_info,
                        raw_policy_action=raw_action,
                    )

                episode_reward += reward
                episode_length += 1
                episode_rewards.append(reward)
                obs = next_obs
                
                done = terminated or truncated
                if done:
                    logger.info("Episode %d: DONE -- reward=%s", ep + 1, episode_reward)

            if episode_length >= MAX_STEPS_PER_EPISODE:
                logger.warning(
                    "Episode %d reached max steps (%d) without done=True.",
                    ep + 1, MAX_STEPS_PER_EPISODE,
                )

            achieved_reward = float(np.sum(episode_rewards))
            max_achievable_reward = episode_length * max_reward_per_step
            reward_rate = (
                achieved_reward / max_achievable_reward
                if max_achievable_reward > 0
                else 0.0
            )

            ep_stats = EpisodeStats(
                episode=ep + 1,
                length=episode_length,
                total_reward=float(episode_reward),
                achieved_reward=achieved_reward,
                max_achievable_reward=float(max_achievable_reward),
                reward_rate=float(reward_rate),
                seed=episode_seed,
                data_variant=ep_data_variant,
                episode_date=ep_episode_date,
            )

            if collector is not None and save_results:
                collector.on_episode_end(
                    episode_id=ep,
                    seed=episode_seed,
                    metadata={
                        "config_name": active_config.config_name,
                        "checkpoint_path": checkpoint_path,
                        "algorithm": algorithm_hint,
                    },
                )
                traj_file = os.path.join(
                    output_dir, f"{ep}_trajectory.json",
                )
                collector.save_json(traj_file)

            episode_stats.append(ep_stats)

            logger.info("  Length: %d", episode_length)
            logger.info("  Total Reward: %.2f", episode_reward)
            logger.info("  Achieved Reward: %.2f", achieved_reward)
            logger.info("  Max Achievable: %.2f", max_achievable_reward)
            logger.info("  Reward Rate: %.4f", reward_rate)
            logger.info("  Seed: %d", episode_seed)

    except TimeoutError as e:
        logger.error(
            "Evaluation timed out after %d seconds: %s", timeout_seconds, e,
        )
        error = "timeout"
    finally:
        signal.alarm(0)

    eval_time = time.time() - start_time

    results = EvalResults.from_episodes(
        episodes=episode_stats,
        checkpoint_path=checkpoint_path,
        config_name=active_config.config_name,
        algorithm=algorithm_hint,
        seed=seed,
        eval_time_seconds=eval_time,
        error=error,
    )

    results.log_summary()

    if save_results:
        results.save(output_dir)

    # Cleanup
    env.close()
    ray.shutdown()

    return results
