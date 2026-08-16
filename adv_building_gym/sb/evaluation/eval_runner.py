"""Evaluation orchestrator for Stable-Baselines3 trained models.

Mirrors :func:`adv_building_gym.ray.evaluation.eval_runner.evaluate_model`
but is Ray-free: the SB3 policy owns its own observation preprocessing, so
the eval loop is a plain ``model.predict → env.step`` roll-out over a single
(non-vectorised) env built with the *same* wrapper chain as training. That
identical chain is what keeps the loaded policy's observation/action spaces
compatible.

Outputs match the Ray side (so the shared plotting pipeline works): per-run
``eval_*.json`` / ``eval_*.csv`` summary, per-episode trajectory JSON +
``trajectories.hdf5``, ``provenance.yaml``, and a copy of the trial YAML +
the evaluated model ``.zip``.
"""

from __future__ import annotations

import datetime
import logging
import os
import shutil
import signal
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC

from adv_building_gym.config.env.env_config import EnvConfig
from adv_building_gym.config.data.data_combinator import DataCombinator
from adv_building_gym.core.env import AdvBuildingGym
from adv_building_gym.core.wrappers import wrap_action_space
from adv_building_gym.components.rewards import SumRewardAggregator
from adv_building_gym._common.trajectory_collector import TrajectoryCollector
from adv_building_gym._common.eval_results import EpisodeStat, EvalResults
from adv_building_gym._common.eval_provenance import copy_trial_yaml, write_provenance
from adv_building_gym._common.constants import MAX_STEPS_PER_EPISODE

logger = logging.getLogger(__name__)

# SB3 has no DreamerV3; PPO / SAC only (parity with sb/training/select_model.py).
_MODEL_CLASSES = {"ppo": PPO, "sac": SAC}


def _timeout_handler(signum, frame):
    raise TimeoutError("Evaluation timed out")


def _copy_sb_model(checkpoint_path: str, output_dir: str) -> None:
    """Copy the evaluated model ``.zip`` next to the results for provenance."""
    src = checkpoint_path if checkpoint_path.endswith(".zip") else f"{checkpoint_path}.zip"
    if not os.path.isfile(src):
        logger.warning("Model file %s not found — skipping model snapshot.", src)
        return
    dst = os.path.join(output_dir, os.path.basename(src))
    shutil.copy2(src, dst)
    logger.info("Saved model snapshot: %s", dst)


def evaluate_sb_model(
    checkpoint_path: str,
    active_config: EnvConfig,
    trial_name: str,
    algorithm: str,
    seed: int,
    num_episodes: int = 1,
    save_results: bool = True,
    output_dir: str = "eval_results",
    log_trajectories: bool = True,
    timeout_seconds: int = 300,
    data_combinator: DataCombinator | None = None,
    stochastic: bool = False,
    run_stamp: str | None = None,
    subdir: str | None = None,
    trial_yaml_path: str | Path | None = None,
) -> EvalResults:
    """Evaluate a Stable-Baselines3 trained model on AdvBuildingGym (CPU inference).

    Loads policy weights via ``PPO.load`` / ``SAC.load`` and rolls out
    ``num_episodes`` on a freshly built env. ``stochastic`` samples from the
    policy (``deterministic=False``) with a per-episode ``seed + ep`` reseed
    for reproducibility; otherwise the deterministic action is taken.

    Args:
        checkpoint_path: Path to the SB3 ``.zip`` model (extension optional).
        active_config: EnvConfig with materialised infras / statesources /
            reward_config (built by the caller, mirroring the Ray driver).
        trial_name: Trial identifier (result metadata + log lines).
        algorithm: ``"ppo"`` or ``"sac"`` — selects the SB3 loader class.
        seed: Base seed; episode ``ep`` resets with ``seed + ep``.
        num_episodes: Number of evaluation episodes.
        save_results: Whether to persist results to disk.
        output_dir: Directory for result files.
        log_trajectories: Whether to save per-step trajectory JSON / HDF5.
        timeout_seconds: Maximum wall-clock seconds before aborting.
        data_combinator: Optional DataCombinator for variant scheduling.
        stochastic: Sample from the policy instead of taking the deterministic action.
        run_stamp: Shared timestamped run dir (caller may reuse across passes).
        subdir: Optional per-config subdirectory nested under ``run_stamp``.
        trial_yaml_path: Trial YAML copied alongside results for provenance.

    Returns:
        ``EvalResults`` with per-episode stats and summary.
    """
    algo = algorithm.lower()
    if algo not in _MODEL_CLASSES:
        raise ValueError(f"SB3 evaluation supports {sorted(_MODEL_CLASSES)}, got '{algorithm}'.")

    # timestamped subdir so eval runs don't collide; caller may share a fixed
    # `run_stamp` across passes and nest each under `subdir`.
    if run_stamp is None:
        run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_eval"
        if stochastic:
            run_stamp += "_stoch"
    output_dir = os.path.join(output_dir, run_stamp)
    if subdir:
        output_dir = os.path.join(output_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)

    copy_trial_yaml(trial_yaml_path, Path(output_dir))
    write_provenance(Path(output_dir), trial_yaml_path, repo_dir=Path(__file__).resolve().parent)

    logger.info("=" * 70)
    logger.info("Starting SB3 model evaluation")
    logger.info("  Checkpoint: %s", checkpoint_path)
    logger.info("  Trial: %s", trial_name)
    logger.info("  Algorithm: %s", algo)
    logger.info("  Episodes: %d", num_episodes)
    logger.info("  Seed: %d", seed)
    logger.info("  Output: %s", output_dir)
    logger.info(
        "  Action mode: %s",
        "stochastic (policy sample)" if stochastic else "deterministic",
    )
    logger.info("=" * 70)

    # CPU inference: the [256, 256] policy is fast on CPU and eval needs no GPU.
    logger.info("Loading %s model from checkpoint...", algo.upper())
    model = _MODEL_CLASSES[algo].load(checkpoint_path, device="cpu")

    if save_results:
        _copy_sb_model(checkpoint_path, output_dir)

    # Built manually (not build_vec_env): eval uses the explicit active_config
    # passed by the caller, and a single non-vectorised env so the trajectory
    # collector reads one clean per-step info stream.
    logger.info("Creating evaluation environment...")
    # allow_reseed: this driver seeds per episode with `seed + ep` (see the reset below),
    # unlike the training envs, which latch on their single construction seed.
    base_env = AdvBuildingGym(
        infras=active_config.infras,
        statesources=active_config.statesources,
        rewards=active_config.reward_config.rewards,
        data_combinator=data_combinator,
        reward_aggregator=SumRewardAggregator(),
        env_config=active_config,
        allow_reseed=True,
    )

    if log_trajectories:
        base_env.log_full_info = True

    # TrajectoryCollector reads spaces from the unwrapped env
    collector = TrajectoryCollector(base_env) if log_trajectories else None

    # Same wrapper chain as training (sb/env_creator.make_sb_env_factory) so the
    # loaded policy's observation/action spaces line up exactly.
    env = base_env
    if active_config.hst.enabled:
        from adv_building_gym.core.history_wrapper import HistoryWrapper
        env = HistoryWrapper(
            env,
            tracked_keys=active_config.hst.tracked_keys,
            offsets=active_config.hst.offsets,
        )
        logger.info(
            "eval_runner: HistoryWrapper enabled (tracked_keys=%s, offsets=%s)",
            list(active_config.hst.tracked_keys), list(active_config.hst.offsets),
        )
    if active_config.forecast.enabled:
        from adv_building_gym.core.forecast_wrapper import ForecastWrapper
        env = ForecastWrapper(env, forecast_steps=active_config.forecast.steps)
        logger.info(
            "eval_runner: ForecastWrapper enabled (steps=%s)", list(active_config.forecast.steps),
        )
    env = wrap_action_space(env)

    if env.observation_space != model.observation_space:
        logger.warning(
            "Observation-space mismatch between env and loaded model — the "
            "eval wrapper chain must match training.\n  env:   %s\n  model: %s",
            env.observation_space, model.observation_space,
        )

    episode_stats: list[EpisodeStat] = []
    start_time = time.time()

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)

    error: str | None = None

    try:
        for ep in range(num_episodes):
            episode_num = ep + 1
            episode_seed = seed + ep
            logger.info("=" * 50)
            logger.info("Episode %d/%d (seed: %d)", episode_num, num_episodes, episode_seed)

            # Per-episode reseed so stochastic sampling is reproducible across runs.
            if stochastic:
                model.set_random_seed(episode_seed)

            obs, reset_info = env.reset(seed=episode_seed)
            ep_data_variant = reset_info.get("data_variant")
            ep_episode_date = reset_info.get("episode_date")
            if ep_data_variant:
                logger.info("  Variant: %s | Date: %s", ep_data_variant, ep_episode_date)

            if collector is not None:
                collector.reset()
                collector.on_reset(reset_info)

            done = False
            episode_reward = 0.0
            episode_length = 0
            episode_rewards: list[float] = []
            step_info: dict = {}

            while not done and episode_length < MAX_STEPS_PER_EPISODE:
                action, _ = model.predict(obs, deterministic=not stochastic)
                next_obs, reward, terminated, truncated, step_info = env.step(action)

                if collector is not None:
                    collector.on_step(
                        step=episode_length,
                        obs=obs,
                        action=action,
                        reward=reward,
                        info=step_info,
                        raw_policy_action=action,
                    )

                episode_reward += reward
                episode_length += 1
                episode_rewards.append(reward)
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

            ep_stat = EpisodeStat(
                episode=episode_num,
                length=episode_length,
                total_reward=float(episode_reward),
                achieved_reward=achieved_reward,
                seed=episode_seed,
                cum_E_kWh=float(step_info.get("cum_E_kWh", 0.0)),
                cum_price_EUR=float(step_info.get("cum_price_EUR", 0.0)),
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
                        "algorithm": algo,
                    },
                )
                traj_dir = os.path.join(output_dir, "trajectories")
                os.makedirs(traj_dir, exist_ok=True)
                collector.save_json(os.path.join(traj_dir, f"{episode_num}_trajectory.json"))
                collector.save_hdf5(
                    os.path.join(output_dir, "trajectories.hdf5"),
                    episode_id=str(episode_num),
                )

            episode_stats.append(ep_stat)

            logger.info("  Length: %d", episode_length)
            logger.info("  Total Reward: %.2f", episode_reward)
            logger.info("  Achieved Reward: %.2f", achieved_reward)
            logger.info("  Seed: %d", episode_seed)

    except TimeoutError as exc:
        logger.error("Evaluation timed out after %d seconds: %s", timeout_seconds, exc)
        error = "timeout"
    finally:
        signal.alarm(0)

    eval_time = time.time() - start_time

    results = EvalResults.from_episodes(
        episodes=episode_stats,
        checkpoint_path=checkpoint_path,
        trial_name=trial_name,
        algorithm=algo,
        seed=seed,
        eval_time_seconds=eval_time,
        error=error,
    )

    results.output_dir = output_dir
    results.log_summary()

    if save_results:
        results.save(output_dir)

    env.close()
    return results
