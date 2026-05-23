"""
Ray RLlib common configuration utilities.

This module provides the common configuration function for RLlib algorithms,
including environment setup, resource allocation, and callback configuration.
"""

import datetime
import logging

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from adv_building_gym.config.env.env_config import EnvConfig
# Per-key strided observation history lives in HistoryWrapper (an env wrapper
# applied in env_creator.py); the env-to-module / learner pipelines only need
# stock FlattenObservations to flatten the augmented Dict obs.
from ray.rllib.connectors.env_to_module import FlattenObservations

from adv_building_gym.ray.callbacks import (
    create_data_schedule_on_train_result_cb,
    create_infra_schedule_on_train_result_cb,
    create_iter_timing_on_train_result_cb,
    create_reward_switch_on_train_result_cb,
    make_episode_metrics_cb_class,
    make_eval_state_action_cb_class,
    make_trajectory_logging_cb_class,
)
from adv_building_gym.ray.callbacks.statesource_schedule_callback import (
    create_statesource_schedule_on_train_result_cb,
)
from adv_building_gym.config.training.exploration_reset import ExplorationResetConfig
from adv_building_gym.config.rewards.reward_schedule_manager import RewardScheduleManager, RewardScheduleMode
from adv_building_gym.config.training.training_param_config import TrainingParamConfig
from adv_building_gym.config.data.data_combinator import DataCombinator
from adv_building_gym.config.env.infra_combinator import InfraCombinator
from adv_building_gym.config.env.statesource_combinator import StatesourceCombinator
from adv_building_gym._common.resource_check_util import ResourceAllocation, SlurmResources, validate_resource_allocation

logger = logging.getLogger(__name__)


def _compose_on_train_result(*fns):
    """Compose multiple on_train_result callables into one.

    RLlib's ``config.callbacks()`` accepts a single ``on_train_result``
    callable.  This helper chains several so that both data scheduling
    and reward switching can coexist.
    """
    def composed_on_train_result(*, algorithm, result, **kwargs):
        for fn in fns:
            fn(algorithm=algorithm, result=result, **kwargs)
    return composed_on_train_result


def register_callbacks(
    config: AlgorithmConfig,
    num_env_runners: int,
    metrics_base_dir: str = "ep_metrics",
    data_combinator: DataCombinator | None = None,
    log_trajectories: bool = False,
    reward_schedule_manager: RewardScheduleManager | None = None,
    infra_combinator: InfraCombinator | None = None,
    statesource_combinator: StatesourceCombinator | None = None,
    exploration_reset: ExplorationResetConfig | None = None,
    exec_date: datetime.datetime | None = None,
) -> None:
    """Register episode-metric, trajectory, and scheduling callbacks on *config*.

    Mutates *config* in place via ``config.callbacks()``.  Intended to be
    called after :func:`common_model_setup` (or at the end of it) so that
    all algorithm-specific settings are already applied before callbacks
    are wired up.

    Args:
        config: Algorithm config object to register callbacks on.
        metrics_base_dir: Base directory for episode metrics.
        data_combinator: DataCombinator for data variant scheduling.
        log_trajectories: Save per-step trajectory JSON during evaluation.
        reward_schedule_manager: Optional reward schedule manager.
        infra_combinator: Optional infrastructure schedule combinator.
    """
    if data_combinator is None:
        data_combinator = DataCombinator()

    # Create callback classes for episode metrics and (optionally) trajectory logging.
    # Each factory returns a configured RLlibCallback subclass.
    # Link: https://docs.ray.io/en/latest/rllib/rllib-callback.html
    if exec_date is None:
        exec_date = datetime.datetime.now()

    episode_metrics_class = make_episode_metrics_cb_class(
        metrics_base_dir=f"{metrics_base_dir}/metrics",
        exec_date=exec_date,
        dump_metrics_json=False
    )

    eval_state_action_class = make_eval_state_action_cb_class(
        metrics_base_dir=metrics_base_dir,
        exec_date=exec_date,
    )

    callback_classes = [episode_metrics_class, eval_state_action_class]
    if log_trajectories:
        trajectory_class = make_trajectory_logging_cb_class(
            metrics_base_dir=f"{metrics_base_dir}/trajectories",
            exec_date=exec_date,
        )
        callback_classes.append(trajectory_class)
        logger.info("Trajectory logging enabled: per-step trajectory JSON will be saved for each episode.")

    # on_train_result callables — both data variant scheduling and reward
    # switching run at iteration boundaries.  RLlib accepts a single
    # on_train_result callable, so compose them when both are active.
    on_train_result_fns = [
        create_iter_timing_on_train_result_cb(),
        create_data_schedule_on_train_result_cb(
            data_combinator, num_env_runners=num_env_runners,
        ),
    ]
    logger.info(
        "DataScheduleCallback: swap_every_n_episodes=%d (num_env_runners=%d), %d variants",
        data_combinator.swap_every_n_episodes, num_env_runners, len(data_combinator.variants),
    )

    if (reward_schedule_manager is not None
            and reward_schedule_manager.mode is not RewardScheduleMode.OFF):
        on_train_result_fns.append(
            create_reward_switch_on_train_result_cb(
                reward_schedule_manager,
                num_env_runners=num_env_runners,
                exploration_reset=exploration_reset,
            ),
        )
        logger.info(
            "RewardSwitchCallback: mode=%s, swap_every_n_episodes=%d (num_env_runners=%d), active rewards: %s",
            reward_schedule_manager.mode,
            reward_schedule_manager.swap_every_n_episodes,
            num_env_runners,
            reward_schedule_manager.get_active_reward_names(),
        )

    if infra_combinator is not None and infra_combinator.is_enabled():
        on_train_result_fns.append(
            create_infra_schedule_on_train_result_cb(
                infra_combinator,
                num_env_runners=num_env_runners,
                exploration_reset=exploration_reset,
            ),
        )
        logger.info(
            "InfraScheduleCallback: mode=%s, swap_every_n_episodes=%d (num_env_runners=%d), %d configs in pool",
            infra_combinator.mode,
            infra_combinator.swap_every_n_episodes,
            num_env_runners,
            len(infra_combinator.config_paths),
        )

    if statesource_combinator is not None and statesource_combinator.is_enabled():
        on_train_result_fns.append(
            create_statesource_schedule_on_train_result_cb(
                statesource_combinator,
                num_env_runners=num_env_runners,
                exploration_reset=exploration_reset,
            ),
        )
        logger.info(
            "StatesourceScheduleCallback: mode=%s, swap_every_n_episodes=%d (num_env_runners=%d), %d configs in pool",
            statesource_combinator.mode,
            statesource_combinator.swap_every_n_episodes,
            num_env_runners,
            len(statesource_combinator.config_paths),
        )

    callback_kwargs = {
        "on_train_result": _compose_on_train_result(*on_train_result_fns),
    }

    # Register all callback classes + optional callable-based callbacks.
    # RLlib executes subclass callbacks in list order, then callables.
    config.callbacks(callbacks_class=callback_classes, **callback_kwargs)


def common_model_setup(
    config: AlgorithmConfig,
    slurm_resources: SlurmResources,
    training_config: TrainingParamConfig,
    env_config: EnvConfig,
    metrics_base_dir: str = "ep_metrics",
    data_combinator: DataCombinator | None = None,
    log_trajectories: bool = False,
    reward_schedule_manager: RewardScheduleManager | None = None,
    infra_combinator: InfraCombinator | None = None,
    statesource_combinator: StatesourceCombinator | None = None,
    exploration_reset: ExplorationResetConfig | None = None,
    exec_date: datetime.datetime | None = None,
):
    """
    Apply common RLlib configuration to an algorithm config.

    This function configures settings that are common across all algorithms:
    - API stack (RL module and learner, env runner and connector v2)
    - Environment configuration (retrieves action space from env_creator)
    - Debugging settings
    - Reporting settings
    - Framework configuration
    - Resource allocation (learners and env runners)
    - Learner resources
    - Env runner resources and connectors
    - Evaluation settings
    - Logger configuration
    - Callbacks (EpisodeMetricsCallback, TrajectoryLoggingCallback)
    - Resource validation

    Args:
        config: Algorithm config object (e.g., PPOConfig instance)
        slurm_resources: SLURM-allocated CPU/GPU resources
        metrics_base_dir: Base directory for episode metrics (default: "ep_metrics")
        data_combinator: DataCombinator for iteration-aligned variant
            scheduling via DataScheduleCallback (Approach D1). An empty
            DataCombinator() acts as a no-op (no variant swapping).
        log_trajectories: When True, save full per-step trajectory JSON
            during evaluation episodes (via episode callback).

    Returns:
        Configured algorithm config
    """
    if data_combinator is None:
        data_combinator = DataCombinator()

    # Resource allocation:
    # - local_learner=True (default): num_learners=0, Learner runs in the driver
    #   process. Driver's 1 CPU covers both, no separate learner CPU reservation.
    # - local_learner=False: one remote Learner per GPU, each with 1 GPU + 1 CPU.
    # - Env runners: remaining CPUs after learners and driver.
    local_learner = training_config.local_learner
    num_cpus_per_env_runner = 1
    driver_cpus = 1

    if local_learner:
        num_learners = 0
        num_gpus_per_learner = 1 if slurm_resources.num_gpus > 0 else 0
        num_cpus_per_learner = 0
        learner_total_cpus = 0
    else:
        num_learners = max(1, slurm_resources.num_gpus)
        num_gpus_per_learner = 1 if slurm_resources.num_gpus > 0 else 0
        num_cpus_per_learner = 1
        learner_total_cpus = num_learners * num_cpus_per_learner

    remaining_cpus = slurm_resources.num_cpus - learner_total_cpus - driver_cpus
    num_env_runners = max(1, remaining_cpus // num_cpus_per_env_runner)

    logger.info(
        "Resource allocation in rllib_config: learners=%d (gpus=%d, cpus=%d each), "
        "env_runners=%d (cpus=%d each), driver=%d CPU",
        num_learners, num_gpus_per_learner, num_cpus_per_learner,
        num_env_runners, num_cpus_per_env_runner, driver_cpus
    )

    # observation_space is intentionally omitted — FlattenObservations transforms
    # it automatically. 
    # action_space is also omitted — the env_creator wraps
    # the env with FlattenAction + RescaleAction so RLlib sees a flat Box(-1, 1).
    # Link: https://docs.ray.io/en/latest/rllib/env-to-module-connector.html

    # TODO VP 2026.03.18. : Check each setting here and at SAC/PPO
    config = config.api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    config.environment(
        env="AdvBuilding",
        clip_actions=training_config.clip_actions_to_env_bounds,  # RLlib default: False
    )
    config.debugging(
        # WARN: Reduces verbosity (suppress connector pipeline INFO messages)
        log_level="INFO",  # RLlib default: WARN
        seed=training_config.seed  # RLlib default: None
    )
    config.reporting(
        keep_per_episode_custom_metrics=True,  # RLlib default: False
        metrics_num_episodes_for_smoothing=25,  # RLlib default: 100
    )
    config.framework(
        framework="torch",  # RLlib default
        torch_skip_nan_gradients=True,
        # TODO VP 2026.03.18. : Torch dynamo backend -- what is it?
        # It runs on torch, not tensorflow
        tf_session_args={},
        local_tf_session_args={},
    )
    config.log_gradients = False # RLlib default: False
    # NOTE VP 2026.01.08. : about ray and rllib concept https://docs.ray.io/en/latest/rllib/key-concepts.html
    # Learning the NN, policy (gradient updates) -- needs GPU
    config.learners(
        num_learners=num_learners,
        num_gpus_per_learner=num_gpus_per_learner,
        num_cpus_per_learner=num_cpus_per_learner,
    )
    # Sampling actions (querying the env, using the policy, sample trajectories) -- no GPU needed
    # Per-key history stacking is handled inside HistoryWrapper (env wrapper);
    # the pipeline here only needs FlattenObservations and the default
    # episode_lookback_horizon (the wrapper owns its own rolling buffer).
    # PPO validates total_train_batch_size ≈ num_env_runners * rollout_fragment_length
    # (within 10%).  With rollout_fragment_length=EPISODE_LENGTH, we need
    # num_env_runners ≈ ppo_episodes_per_iteration * num_learners.  When the
    # SLURM-derived num_env_runners exceeds that, drop it down so episodes are
    # not over-collected on each iteration (the surplus CPUs go unused).
    is_ppo = type(config).__name__ == "PPOConfig"
    # Accessing train_batch_size_per_learner on non-PPO configs (e.g. DreamerV3)
    # can raise inside RLlib when train_batch_size is unset — only read it
    # when we actually need it for PPO's env-runner sizing heuristic.
    train_batch = getattr(config, "train_batch_size_per_learner", None) if is_ppo else None
    if is_ppo and train_batch:
        total_batch = train_batch * num_learners
        target_env_runners = max(1, total_batch // env_config.EPISODE_LENGTH)
        if num_env_runners > target_env_runners:
            logger.warning(
                "Reducing num_env_runners %d -> %d to match PPO total_train_batch_size=%d "
                "(per_learner=%d x num_learners=%d) at rollout_fragment_length=%d. "
                "Surplus CPUs will be left idle.",
                num_env_runners, target_env_runners, total_batch,
                train_batch, num_learners, env_config.EPISODE_LENGTH,
            )
            num_env_runners = target_env_runners
    config.env_runners(
        rollout_fragment_length=env_config.EPISODE_LENGTH, # Collect complete episodes before returning to learner.
        num_env_runners=num_env_runners,
        num_cpus_per_env_runner=num_cpus_per_env_runner,
        episode_lookback_horizon=training_config.episode_lookback_horizon_steps,  # RLlib default: 1
        env_to_module_connector=lambda env, spaces, device: [FlattenObservations()],  # type: ignore
    )
    # Mirror the env-to-module pipeline on the learner side so replayed (SAC)
    # or on-policy (PPO) batches flatten to the same obs dim.
    config.training(
        learner_connector=lambda obs_sp, act_sp: [FlattenObservations()],  # type: ignore
    )
    # Evaluation runs the current policy without exploration noise to provide
    # an unbiased performance signal for model selection (analogous to a
    # validation set).  It does NOT influence gradient updates.
    # Evaluation EnvRunners always get log_full_info=True so step() includes
    # a deep copy of named state in info["state"] — needed by the eval
    # trajectory callback (raw + normalised + actions) and trajectory logging.
    # Training EnvRunners are unaffected (no extra memory overhead).
    eval_env_config = {"log_full_info": True}

    # evaluation_interval > 1 means the `evaluation/env_runners/` keys are
    # absent from results on non-eval iterations.  Tune's strict metric check
    # would crash, so TUNE_DISABLE_STRICT_METRIC_CHECKING must be set in the
    # driver process (run_train_ray.py).
    config.evaluation(
        # evaluation_num_env_runners=1, # not important for now
        evaluation_interval=10,  # RLlib default: None
        evaluation_duration_unit="episodes",  # RLlib default
        evaluation_duration=2,  # RLlib default: 10
        evaluation_parallel_to_training=False,  # RLlib default
        evaluation_config=AlgorithmConfig.overrides(env_config=eval_env_config),
    )

    config.logger_config = {
        "type": "ray.tune.logger.UnifiedLogger", # Logging orchestrator
        "loggers": [
                "ray.tune.json.JsonLoggerCallback", # writes result.json files with train and eval metrics
                "ray.tune.csv.CSVLoggerCallback", # writes progress.csv files
                "ray.tune.tensorboardx.TBXLoggerCallback", # Writes tensorboard event files
        ],
    }

    register_callbacks(
        config,
        num_env_runners=num_env_runners,
        metrics_base_dir=metrics_base_dir,
        data_combinator=data_combinator,
        log_trajectories=log_trajectories,
        reward_schedule_manager=reward_schedule_manager,
        infra_combinator=infra_combinator,
        statesource_combinator=statesource_combinator,
        exploration_reset=exploration_reset,
        exec_date=exec_date,
    )

    # Validate resource allocation against SLURM constraints
    driver_cpus = 1
    learner_total_cpus = num_learners * num_cpus_per_learner
    total_cpu_usage = driver_cpus + learner_total_cpus + (num_env_runners * num_cpus_per_env_runner)
    unused_cpus = slurm_resources.num_cpus - total_cpu_usage

    allocation = ResourceAllocation(
        total_cpu_usage=total_cpu_usage,
        unused_cpus=unused_cpus,
        driver_cpus=driver_cpus,
        num_learners=num_learners,
        cpus_per_learner=num_cpus_per_learner,
        learner_total_cpus=learner_total_cpus,
        actual_env_runners=num_env_runners,
        cpus_per_env_runner=num_cpus_per_env_runner,
        slurm_cpus=slurm_resources.num_cpus,
        slurm_gpus=slurm_resources.num_gpus,
    )

    # Convert config to dict for validation
    param_space = config.to_dict()
    validate_resource_allocation(allocation, param_space)

    return config
