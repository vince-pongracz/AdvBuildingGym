"""
Ray RLlib common configuration utilities.

This module provides the common configuration function for RLlib algorithms,
including environment setup, resource allocation, and callback configuration.
"""

import datetime
import logging

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from adv_building_gym.ray_training.history_connector import build_env_to_module_connectors

from adv_building_gym.callbacks import (
    create_data_schedule_on_train_result_cb,
    create_infra_schedule_on_train_result_cb,
    create_reward_switch_on_train_result_cb,
    make_episode_metrics_cb_class,
    make_eval_state_action_cb_class,
    make_trajectory_logging_cb_class,
)
from adv_building_gym.config.reward_schedule_manager import RewardScheduleManager, RewardScheduleMode
from adv_building_gym.config.training_param_config import TrainingParamConfig
from adv_building_gym.data_combinator import DataCombinator
from adv_building_gym.infra_combinator import InfraCombinator
from adv_building_gym.utils import ResourceAllocation, SlurmResources, validate_resource_allocation

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
    metrics_base_dir: str = "ep_metrics",
    data_combinator: DataCombinator | None = None,
    log_trajectories: bool = False,
    reward_schedule_manager: RewardScheduleManager | None = None,
    infra_combinator: InfraCombinator | None = None,
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
        create_data_schedule_on_train_result_cb(
            data_combinator, data_combinator.swap_every_n_episodes,
        ),
    ]
    logger.info(
        "DataScheduleCallback: swap every %d iterations, %d variants",
        data_combinator.swap_every_n_episodes, len(data_combinator.variants),
    )

    if (reward_schedule_manager is not None
            and reward_schedule_manager.mode is not RewardScheduleMode.OFF):
        on_train_result_fns.append(
            create_reward_switch_on_train_result_cb(reward_schedule_manager),
        )
        logger.info(
            "RewardSwitchCallback: mode=%s, swap every %d iterations, "
            "active rewards: %s",
            reward_schedule_manager.mode,
            reward_schedule_manager.swap_every_n_iterations,
            reward_schedule_manager.get_active_reward_names(),
        )

    if infra_combinator is not None and infra_combinator.is_enabled():
        on_train_result_fns.append(
            create_infra_schedule_on_train_result_cb(infra_combinator),
        )
        logger.info(
            "InfraScheduleCallback: mode=%s, swap every %d iterations, "
            "%d configs in pool",
            infra_combinator.mode,
            infra_combinator.swap_every_n_iterations,
            len(infra_combinator.config_paths),
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
    metrics_base_dir: str = "ep_metrics",
    clip_actions: bool = True,
    data_combinator: DataCombinator | None = None,
    log_trajectories: bool = False,
    reward_schedule_manager: RewardScheduleManager | None = None,
    infra_combinator: InfraCombinator | None = None,
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
        clip_actions: Whether to clip actions to action space bounds
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
    # - Learners: one per GPU, each gets 1 GPU and 1 CPU -- it needs CPU for orchestration
    # - Driver: 1 CPU (taken from env_runners pool)
    # - Env runners: remaining CPUs after learners and driver
    num_learners = max(1, slurm_resources.num_gpus)  # At least 1 learner even without GPU
    num_gpus_per_learner = 1 if slurm_resources.num_gpus > 0 else 0
    num_cpus_per_learner = 1
    num_cpus_per_env_runner = 1

    driver_cpus = 1
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
        clip_actions=clip_actions,  # RLlib default: False
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
    config.log_gradients = True
    # NOTE VP 2026.01.08. : about ray and rllib concept https://docs.ray.io/en/latest/rllib/key-concepts.html
    # Learning the NN, policy (gradient updates) -- needs GPU
    config.learners(
        num_learners=num_learners,
        num_gpus_per_learner=num_gpus_per_learner,
        num_cpus_per_learner=num_cpus_per_learner,
    )
    # Sampling actions (querying the env, using the policy, sample trajectories) -- no GPU needed
    # Episode lookback horizon controls how far back the env-to-module connector
    # can index into the current episode; must cover the largest |offset| used
    # by StridedHistoryConnector, else mid-episode lookups silently fall back
    # to padding.
    # Link: https://docs.ray.io/en/latest/rllib/env-to-module-connector.html
    max_abs_offset = max((abs(offset) for offset in training_config.hst_offsets), default=0)
    effective_lookback = max(training_config.episode_lookback_horizon_steps, max_abs_offset)
    if effective_lookback > training_config.episode_lookback_horizon_steps:
        logger.warning(
            "episode_lookback_horizon_steps=%d is smaller than max(|hst.offsets|)=%d; "
            "raising episode_lookback_horizon to %d for StridedHistoryConnector.",
            training_config.episode_lookback_horizon_steps, max_abs_offset,
            effective_lookback
        )

    config.env_runners(
        num_env_runners=num_env_runners,
        num_cpus_per_env_runner=num_cpus_per_env_runner,
        episode_lookback_horizon=effective_lookback,  # RLlib default: 1
        # Dict obs is transformed by the shared connector factory:
        #   hst_tracked_keys set  →  [StridedHistoryConnector] (stack + flatten)
        #   otherwise             →  [FlattenObservations]
        # The two branches are mutually exclusive — StridedHistoryConnector
        # handles its own flattening. Action space flattening + rescaling
        # is handled by env wrappers (FlattenAction + RescaleAction) applied
        # in env_creator; the connector never reads the action space
        # (action history reaches the policy via <action_key>_prev obs keys
        # published by the env each step).
        # RLlib invokes this factory on every env runner. On remote workers
        # `env` is the wrapped env instance; on the driver-side local runner
        # `env` is None (no env is built there when num_env_runners >= 1 — see
        # env_runner_group.py:319 "local worker has no env"). The authoritative
        # spaces always live in the `spaces` dict under '__env_single__', so
        # prefer that and fall back to `env` only as a convenience.
        # Link: https://docs.ray.io/en/latest/rllib/connector.html
        env_to_module_connector=lambda env, spaces, device: build_env_to_module_connectors(
            training_config,
            env.observation_space if env is not None else spaces["__env_single__"][0],
            env.action_space if env is not None else spaces["__env_single__"][1],
            as_learner_connector=False,
        ),  # type: ignore
    )
    # The learner pipeline must mirror the env-to-module pipeline so that
    # replayed (SAC) or on-policy (PPO) episodes produce the same flat obs
    # dim the RLModule was built from.
    # Link: https://docs.ray.io/en/latest/rllib/learner-connector.html
    config.training(
        learner_connector=lambda obs_sp, act_sp: build_env_to_module_connectors(
            training_config,
            obs_sp,
            act_sp,
            as_learner_connector=True,
        ),  # type: ignore
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
        evaluation_interval=4,  # RLlib default: None
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
        metrics_base_dir=metrics_base_dir,
        data_combinator=data_combinator,
        log_trajectories=log_trajectories,
        reward_schedule_manager=reward_schedule_manager,
        infra_combinator=infra_combinator,
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
