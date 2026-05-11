"""Evaluation orchestrator for Ray/RLlib trained models.

Composes checkpoint loading, environment creation, inference, and results
aggregation into a single ``evaluate_model()`` entry point.
"""

import datetime
import logging
import os
import signal
import time
from pathlib import Path

import numpy as np
import ray
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.env.single_agent_episode import SingleAgentEpisode

from adv_building_gym.config.env_config import EnvConfig
from adv_building_gym.config.training_param_config import TrainingParamConfig
from adv_building_gym.data_combinator import DataCombinator
from adv_building_gym.envs import AdvBuildingGym
from adv_building_gym.envs.env_creator import wrap_action_space
from adv_building_gym.rewards import SumRewardAggregator
from ray.rllib.connectors.common import AddObservationsFromEpisodesToBatch
from ray.rllib.connectors.env_to_module import FlattenObservations
from adv_building_gym.ray_training.rl_module_inference import (
    infer_action,
    load_rl_module,
)
from adv_building_gym.utils import TrajectoryCollector, check_space_compatibility

from .results import EpisodeStats, EvalResults

logger = logging.getLogger(__name__)


def _timeout_handler(signum, frame):
    raise TimeoutError("Evaluation timed out")


def evaluate_model(
    checkpoint_path: str,
    active_config: EnvConfig,
    trial_name: str,
    seed: int,
    num_episodes: int = 1,
    save_results: bool = True,
    output_dir: str = "eval_results",
    log_trajectories: bool = True,
    algorithm_hint: str | None = None,
    timeout_seconds: int = 300,
    data_combinator: DataCombinator | None = None,
    stochastic: bool = False,
) -> EvalResults:
    """Evaluate a Ray/RLlib trained model on AdvBuildingGym.

    Uses CPU-only inference to avoid GPU resource over-subscription.
    Policy weights are loaded directly via ``RLModule.from_checkpoint()``
    without spawning training actors.

    Args:
        checkpoint_path: Absolute path to the Ray checkpoint directory.
        active_config: Config object with infras, statesources, rewards.
        trial_name: Trial identifier (used for result metadata + log lines).
        num_episodes: Number of evaluation episodes.
        seed: Random seed for reproducibility.
        save_results: Whether to persist results to disk.
        output_dir: Directory for result files.
        log_trajectories: Whether to save per-step trajectory JSON.
        algorithm_hint: Algorithm name for metadata (informational only).
        timeout_seconds: Maximum wall-clock seconds before aborting.
        data_combinator: Optional DataCombinator for variant scheduling.
        stochastic: If True, sample actions from the squashed-Gaussian policy
            instead of taking ``tanh(mean)``. A ``torch.Generator`` is seeded
            per episode from ``seed + ep`` so runs stay reproducible.

    Returns:
        ``EvalResults`` with per-episode stats and summary.
    """
    # Load the default YAML training config if the caller didn't pass one —
    # eval MUST use the same hst settings as training or obs dimensions diverge.

    # Create a timestamped subdirectory so successive eval runs never collide
    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M") + "_eval"
    output_dir = os.path.join(output_dir, run_stamp)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Starting Ray model evaluation")
    logger.info("  Checkpoint: %s", checkpoint_path)
    logger.info("  Trial: %s", trial_name)
    logger.info("  Episodes: %d", num_episodes)
    logger.info("  Seed: %d", seed)
    logger.info("  Output: %s", output_dir)
    logger.info(
        "  Action mode: %s",
        "stochastic (squashed-Gaussian sample)"
        if stochastic
        else "deterministic (tanh(mean))",
    )
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
    
    # TODO VP 2026.03.16. : Train long term -- for 7 days, for 30 days, for 365 days -- episodes
    # --> Eval long term as well. Not only single day optimisation, long term optimisation learnt
    # On trial level it's already realised, but still have to try and test it

    # Create evaluation environment with action-space wrappers
    # (FlattenAction + RescaleAction) so the policy's flat [-1, 1] output
    # is correctly rescaled to each component's real bounds.
    logger.info("Creating evaluation environment...")
    
    # Built manually (not via adv_building_env_creator) because the factory
    # reads from the global env_config singleton, but eval uses an explicit
    # active_config passed by the caller (which may differ, e.g. loaded
    # from a JSON checkpoint).
    base_env = AdvBuildingGym(
        infras=active_config.infras,
        statesources=active_config.statesources,
        rewards=active_config.reward_config.rewards,
        data_combinator=data_combinator,
        reward_aggregator=SumRewardAggregator(),
        env_config=active_config,
    )

    if log_trajectories:
        base_env.log_full_info = True

    # TrajectoryCollector reads spaces from the unwrapped env
    collector = TrajectoryCollector(base_env) if log_trajectories else None

    env = wrap_action_space(base_env)

    # Mirrors the training-side connector pipeline so the flat obs dim
    # matches the checkpoint. StridedHistoryConnector is currently disabled
    # (see common_model_config.py); re-wire build_env_to_module_connectors
    # from history_connector.py here if HST is reinstated.
    # FlattenObservations needs the input spaces set at construction —
    # recompute_output_observation_space reads them from self, not its args.
    # FlattenObservations rewrites the episode's last obs to a flat tensor;
    # AddObservationsFromEpisodesToBatch then copies it into batch[OBS] for
    # the RLModule. Without the latter, batch[OBS] never gets populated.
    pipeline = [
        FlattenObservations(
            input_observation_space=base_env.observation_space,
            input_action_space=env.action_space,
        ),
        AddObservationsFromEpisodesToBatch(),
    ]

    # Run the space compatibility check *after* the pipeline is built so the
    # model's input dim is compared against the post-connector flat size (e.g.
    # with StridedHistoryConnector stacking obs history), not the raw env obs.
    check_space_compatibility(rl_module, env, pipeline=pipeline)

    episode_stats: list[EpisodeStats] = []
    start_time = time.time()

    # Set up timeout
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)

    error: str | None = None

    try:
        MAX_STEPS_PER_EPISODE = 1000
        for ep in range(num_episodes):
            episode_num = ep + 1
            logger.info("=" * 50)
            episode_seed = seed + ep
            logger.info("Episode %d/%d (seed: %d)", episode_num, num_episodes, episode_seed)

            # Per-episode torch RNG so stochastic action sampling is
            # reproducible across runs with the same --seed.
            action_generator: torch.Generator | None
            if stochastic:
                action_generator = torch.Generator().manual_seed(episode_seed)
            else:
                action_generator = None

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
            max_achievable_reward = 0.0

            if collector is not None:
                collector.reset()
                collector.on_reset(reset_info)

            # Persistent episode buffer — ``StridedHistoryConnector`` needs
            # the full observation/action lookback to materialise ``hst_*``
            # stacks at decision time.
            sa_episode = SingleAgentEpisode(
                observation_space=base_env.observation_space,
                action_space=base_env.action_space,
                observations=[obs],
            )

            while not done and episode_length < MAX_STEPS_PER_EPISODE:
                batch: dict = {}
                for connector in pipeline:
                    batch = connector(
                        rl_module=None,
                        batch=batch,
                        episodes=[sa_episode],
                        explore=False, # This has nothing to do with action stochasticity.
                        shared_data={},
                    )
                # ``add_batch_item`` stores: {Columns.OBS: {ep_id: [flat_obs]}}.
                obs_column = batch[Columns.OBS]
                flat_obs = next(iter(obs_column.values()))[-1]
                raw_action = infer_action(
                    rl_module,
                    flat_obs,
                    stochastic=stochastic,
                    generator=action_generator,
                )

                next_obs, reward, terminated, truncated, step_info = env.step(raw_action)

                sa_episode.add_env_step(
                    observation=next_obs,
                    action=raw_action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    infos=step_info,
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
                max_achievable_reward += step_info.get("max_reward_step", 0.0)
                obs = next_obs
                
                done = terminated or truncated
                if done:
                    logger.info("Episode %d: DONE -- reward=%s", episode_num, episode_reward)

            if episode_length >= MAX_STEPS_PER_EPISODE:
                logger.warning(
                    "Episode %d reached max steps (%d) without done=True.",
                    episode_num, MAX_STEPS_PER_EPISODE,
                )

            achieved_reward = float(np.sum(episode_rewards))
            reward_rate = (
                achieved_reward / max_achievable_reward
                if max_achievable_reward > 0
                else 0.0
            )

            ep_stats = EpisodeStats(
                episode=episode_num,
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
                    episode_id=episode_num,
                    seed=episode_seed,
                    metadata={
                        "trial_name": trial_name,
                        "checkpoint_path": checkpoint_path,
                        "algorithm": algorithm_hint,
                    },
                )
                traj_dir = os.path.join(output_dir, "trajectories")
                os.makedirs(traj_dir, exist_ok=True)
                traj_file = os.path.join(
                    traj_dir, f"{episode_num}_trajectory.json",
                )
                collector.save_json(traj_file)
                hdf5_path = os.path.join(output_dir, "trajectories.hdf5")
                collector.save_hdf5(hdf5_path, episode_id=str(episode_num))

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
        trial_name=trial_name,
        algorithm=algorithm_hint,
        seed=seed,
        eval_time_seconds=eval_time,
        error=error,
    )

    results.output_dir = output_dir
    results.log_summary()

    if save_results:
        results.save(output_dir)

    # Cleanup
    env.close()
    ray.shutdown()

    return results
