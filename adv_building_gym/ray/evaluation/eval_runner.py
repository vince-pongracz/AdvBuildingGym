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

import gymnasium
import numpy as np
import ray
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.env.single_agent_episode import SingleAgentEpisode

from adv_building_gym.config.env.env_config import EnvConfig
from adv_building_gym.config.training.training_param_config import TrainingParamConfig
from adv_building_gym.config.data.data_combinator import DataCombinator
from adv_building_gym.core.env import AdvBuildingGym
from adv_building_gym.ray.env_creator import wrap_action_space
from adv_building_gym.components.rewards import SumRewardAggregator
from ray.rllib.connectors.common import AddObservationsFromEpisodesToBatch
from adv_building_gym.ray.training.rl_module_inference import (
    infer_action,
    load_rl_module,
)
from adv_building_gym._common.trajectory_collector import TrajectoryCollector
from adv_building_gym._common.space_check import check_space_compatibility
from adv_building_gym._common.constants import MAX_STEPS_PER_EPISODE
from adv_building_gym._common.eval_results import EpisodeStat, EvalResults
from adv_building_gym._common.eval_provenance import copy_trial_yaml, write_provenance

from .utils import copy_rl_module

logger = logging.getLogger(__name__)


def _timeout_handler(signum, frame):
    raise TimeoutError("Evaluation timed out")


def _batch_state(state):
    """Add a leading batch dim to every tensor in a (nested) state dict
    (``get_initial_state()`` is unbatched); non-tensor leaves unchanged."""
    if isinstance(state, dict):
        return {k: _batch_state(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(_batch_state(v) for v in state)
    if isinstance(state, torch.Tensor):
        return state.unsqueeze(0)
    return state


def _detect_snapshot_root() -> Path | None:
    """Root of the snapshot bundle this code runs from, or None for the live repo.

    Snapshot mode is signalled by ``SNAPSHOT_DIR`` (exported by
    slurm_scripts/util/snapshot_mode.sh); direct invocations of snapshot code
    are detected by the ``snapshot.zip`` marker above the frozen ``code/`` dir.
    """
    env_root = os.environ.get("SNAPSHOT_DIR")
    if env_root:
        return Path(env_root).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "snapshot.zip").exists():
            return parent
    return None


def _enforce_checkpoint_containment(checkpoint_path: str) -> None:
    """Refuse checkpoint/code pairings across snapshot boundaries.

    The flat obs feature order is a property of the training CODE, not of the
    checkpoint (connector era = sorted keys, env-wrapper era = Dict key order),
    so a checkpoint is only valid with the code that produced it: snapshot code
    may only evaluate checkpoints under its own snapshot dir, and the live repo
    may not evaluate checkpoints that live inside any snapshot bundle.
    """
    ckpt = Path(checkpoint_path).resolve()
    snapshot_root = _detect_snapshot_root()
    if snapshot_root is not None:
        if snapshot_root not in ckpt.parents:
            raise ValueError(
                f"Snapshot code ({snapshot_root}) may only evaluate checkpoints from its "
                f"own runs, but got {ckpt}. Evaluate that checkpoint with the code that "
                "trained it (its own snapshot, or the live repo for live-trained runs)."
            )
    elif any((parent / "snapshot.zip").exists() for parent in ckpt.parents):
        raise ValueError(
            f"Live-repo eval refuses the snapshot checkpoint {ckpt}: the obs-flattening "
            "layout is a property of the training code, so this checkpoint must be "
            "evaluated by its own snapshot's code, e.g. "
            "python -m tools.snapshot.submit_snapshot --kind eval on that snapshot."
        )


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
    run_stamp: str | None = None,
    subdir: str | None = None,
    trial_yaml_path: str | Path | None = None,
) -> EvalResults:
    """Evaluate a Ray/RLlib trained model on AdvBuildingGym (CPU-only inference).

    Loads policy weights via ``RLModule.from_checkpoint()`` without training actors.
    ``stochastic`` samples from the squashed-Gaussian policy (per-episode ``seed + ep`` RNG)
    instead of ``tanh(mean)``. Returns ``EvalResults`` (per-episode stats + summary).

    Args:
        checkpoint_path: Absolute path to the Ray checkpoint directory.
        active_config: Config object with infras, statesources, rewards.
        trial_name: Trial identifier (used for result metadata + log lines).
        num_episodes: Number of evaluation episodes.
        seed: Random seed for reproducibility.
        save_results: Whether to persist results to disk.
        output_dir: Directory for result files.
        log_trajectories: Whether to save per-step trajectory JSON.
        algorithm_hint: Algorithm name for result metadata (obs flattening is
            env-side for all algorithms, so no per-algorithm branching here).
        timeout_seconds: Maximum wall-clock seconds before aborting.
        data_combinator: Optional DataCombinator for variant scheduling.
        stochastic: If True, sample actions from the squashed-Gaussian policy
            instead of taking ``tanh(mean)``. A ``torch.Generator`` is seeded
            per episode from ``seed + ep`` so runs stay reproducible.

    Returns:
        ``EvalResults`` with per-episode stats and summary.
    """
    # A checkpoint is only valid with the code that trained it (obs feature order
    # is a code property) — fail fast before any Ray/module loading.
    _enforce_checkpoint_containment(checkpoint_path)

    # eval MUST use the same hst settings as training or obs dimensions diverge.

    # timestamped subdir so eval runs don't collide; caller may share a fixed `run_stamp`
    # across passes and nest each under `subdir`
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
    logger.info("Starting Ray model evaluation")
    logger.info("  Checkpoint: %s", checkpoint_path)
    logger.info("  Trial: %s", trial_name)
    logger.info("  Episodes: %d", num_episodes)
    logger.info("  Seed: %d", seed)
    logger.info("  Output: %s", output_dir)
    logger.info(
        "  Action mode: %s",
        "stochastic (squashed-Gaussian sample)" if stochastic else "deterministic (tanh(mean))",
    )
    logger.info("=" * 70)

    # minimal Ray for CPU-only inference ([256, 256] policy is fast on CPU)
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

    if save_results:
        copy_rl_module(Path(checkpoint_path), Path(output_dir))
        logger.info("Copied RLModule files to output directory for provenance.")
    
    # TODO VP 2026.03.16. : Partially resolved -- Train long term -- for 7 days, for 30 days, for 365 days -- episodes
    # --> Eval long term as well. Not only single day optimisation, long term optimisation learnt
    # On trial level it's already realised, but still have to try and test it

    # eval env with FlattenAction + RescaleAction so the flat [-1,1] policy output
    # rescales to each component's real bounds.
    logger.info("Creating evaluation environment...")

    # built manually (not adv_building_env_creator): eval uses the explicit
    # active_config passed by the caller, not the global singleton
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

    if active_config.hst.enabled:
        from adv_building_gym.core.history_wrapper import HistoryWrapper
        base_env = HistoryWrapper(
            base_env,
            tracked_keys=active_config.hst.tracked_keys,
            offsets=active_config.hst.offsets,
        )
        logger.info(
            "eval_runner: HistoryWrapper enabled (tracked_keys=%s, offsets=%s)",
            list(active_config.hst.tracked_keys),
            list(active_config.hst.offsets),
        )

    if active_config.forecast.enabled:
        from adv_building_gym.core.forecast_wrapper import ForecastWrapper
        base_env = ForecastWrapper(
            base_env,
            forecast_steps=active_config.forecast.steps,
        )
        logger.info(
            "eval_runner: ForecastWrapper enabled (steps=%s)",
            list(active_config.forecast.steps),
        )

    env = wrap_action_space(base_env)

    # Mirrors the training env chain: obs flattening is env-side for all algorithms
    # (outermost FlattenObservation, Dict key order — see adv_building_env_creator),
    # so the pipeline only copies the already-flat obs into batch[OBS].
    env = gymnasium.wrappers.FlattenObservation(env)
    logger.info(
        "eval_runner: FlattenObservation applied (outermost; flat obs shape=%s)", env.observation_space.shape,
    )
    pipeline = [AddObservationsFromEpisodesToBatch()]

    # Era guard: checkpoints trained with the old connector-side FlattenObservations
    # carry an all-infinite module obs space (the connector recomputes Box(-inf, inf));
    # their feature order (sorted keys) is a permutation of the env-side order (Dict
    # key order), so the dim check below would pass while every feature is misplaced.
    module_obs_space = rl_module.observation_space
    if (
        isinstance(module_obs_space, gymnasium.spaces.Box)
        and np.all(np.isinf(module_obs_space.low))
        and np.all(np.isinf(module_obs_space.high))
        and not (np.all(np.isinf(env.observation_space.low)) and np.all(np.isinf(env.observation_space.high)))
    ):
        layout_mismatch_message = (
            "Checkpoint's module obs space has all-infinite bounds — it was trained "
            "with the old connector-side flattening (sorted key order). Env-side "
            "flattening uses the Dict key order, so features would be PERMUTED and "
            "eval results invalid. Evaluate this checkpoint with the code/snapshot "
            "that trained it."
        )
        if os.environ.get("ADVBG_ALLOW_OBS_LAYOUT_MISMATCH") == "1":
            logger.warning("%s (ADVBG_ALLOW_OBS_LAYOUT_MISMATCH=1 set — proceeding anyway)", layout_mismatch_message)
        else:
            raise ValueError(layout_mismatch_message + " Set ADVBG_ALLOW_OBS_LAYOUT_MISMATCH=1 to override.")

    # check spaces after the pipeline is built so the model dim is compared against the
    # post-connector flat size (incl. s_hst_<key>), not the raw env obs
    check_space_compatibility(rl_module, env, pipeline=pipeline)

    episode_stats: list[EpisodeStat] = []
    start_time = time.time()

    # Set up timeout
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)

    error: str | None = None

    try:
        for ep in range(num_episodes):
            episode_num = ep + 1
            logger.info("=" * 50)
            episode_seed = seed + ep
            logger.info("Episode %d/%d (seed: %d)", episode_num, num_episodes, episode_seed)

            # per-episode torch RNG so stochastic sampling is reproducible across runs
            action_generator: torch.Generator | None
            if stochastic:
                action_generator = torch.Generator().manual_seed(episode_seed)
            else:
                action_generator = None

            obs, reset_info = env.reset(seed=episode_seed)
            ep_data_variant = reset_info.get("data_variant")
            ep_episode_date = reset_info.get("episode_date")
            if ep_data_variant:
                logger.info("  Variant: %s | Date: %s", ep_data_variant, ep_episode_date)
            done = False
            episode_reward = 0.0
            episode_length = 0
            episode_rewards: list[float] = []

            if collector is not None:
                collector.reset()
                collector.on_reset(reset_info)

            # persistent episode buffer; the pipeline copies the latest (already flat)
            # obs into batch[OBS] each step.
            sa_episode = SingleAgentEpisode(
                observation_space=env.observation_space,
                action_space=base_env.action_space,
                observations=[obs],
            )

            # initial recurrent state ({} for stateless PPO/SAC); get_initial_state()
            # is unbatched, so add the batch dim
            initial_state = rl_module.get_initial_state() or {}
            state_in = _batch_state(initial_state)
            step_info = {}

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
                raw_action, state_in = infer_action(
                    rl_module,
                    flat_obs,
                    stochastic=stochastic,
                    generator=action_generator,
                    state_in=state_in,
                    is_first=(episode_length == 0),
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

            episode_stats.append(ep_stat)

            logger.info("  Length: %d", episode_length)
            logger.info("  Total Reward: %.2f", episode_reward)
            logger.info("  Achieved Reward: %.2f", achieved_reward)
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
