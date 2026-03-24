"""
Ray RLlib common configuration utilities.

This module provides the common configuration function for RLlib algorithms,
including environment setup, resource allocation, and callback configuration.
"""

import datetime
import logging
from typing import List

from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from adv_building_gym.callbacks import (
    create_data_schedule_on_train_result,
    make_episode_metrics_callback_class,
    make_trajectory_logging_callback_class,
)
from adv_building_gym.config.training_param_config import TrainingParamConfig
from adv_building_gym.data_combinator import DataCombinator
from adv_building_gym.utils import ResourceAllocation, validate_resource_allocation

logger = logging.getLogger(__name__)


def common_model_setup(
    config: AlgorithmConfig,
    episode_length: int,
    num_cpus: int,
    num_gpus: int,
    training_config: TrainingParamConfig,
    # TODO VP 2026.01.13. : Improve checkpoint directory structure
    # save Policy NN in
    checkpoint_callback_class: type,
    env_id: str,
    rewards: List,
    metrics_base_dir: str = "ep_metrics",
    clip_actions: bool = True,
    data_combinator: DataCombinator | None = None,
    log_trajectories: bool = False,
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
    - Callbacks (EpisodeMetricsCallback, TrajectoryLoggingCallback, checkpoint)
    - Resource validation

    Args:
        config: Algorithm config object (e.g., PPOConfig instance)
        action_space: Flat Box action space for the RL module
        episode_length: Episode length in timesteps (used for rollout_fragment_length)
        num_cpus: Total CPUs available (from Ray/SLURM)
        num_gpus: Total GPUs available (from Ray/SLURM)
        checkpoint_callback_class: Callback class for checkpoint management
        env_id: Environment ID string for logging
        rewards: List of reward functions used in the environment
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
    # - Learners: one per GPU, each gets 1 GPU and 1 CPU
    # - Driver: 1 CPU (taken from env_runners pool)
    # - Env runners: remaining CPUs after learners and driver
    num_learners = max(1, num_gpus)  # At least 1 learner even without GPU
    num_gpus_per_learner = 1 if num_gpus > 0 else 0
    num_cpus_per_learner = 1
    num_cpus_per_env_runner = 1

    driver_cpus = 1
    learner_total_cpus = num_learners * num_cpus_per_learner
    remaining_cpus = num_cpus - learner_total_cpus - driver_cpus
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
        clip_actions=clip_actions,
    )
    config.debugging(
        # WARN: Reduces verbosity (suppress connector pipeline INFO messages)
        log_level="INFO",
        log_sys_usage=True,
        seed=training_config.seed
    )
    config.reporting(
        keep_per_episode_custom_metrics=True,
        metrics_num_episodes_for_smoothing=25,
    )
    config.framework(
        framework="torch",
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
    config.env_runners(
        num_env_runners=num_env_runners,
        num_envs_per_env_runner=1, # NOTE VP 2026.02.11. : Maybe worth running multiple envs on a single ray envrunner node...
        num_cpus_per_env_runner=num_cpus_per_env_runner,
        num_gpus_per_env_runner=0,
        # Collect complete episodes before returning to learner.
        # Without this, off-policy algorithms (SAC) default to 1, causing
        # training episodes to be reported as length = 1 in callbacks.
        rollout_fragment_length=episode_length,
        episode_lookback_horizon=training_config.episode_lookback_horizon_steps,
        # Flatten dict observation space into a single vector for the RL module.
        # Action space flattening + rescaling is handled by env wrappers
        # (FlattenAction + RescaleAction) applied in env_creator.
        env_to_module_connector=lambda env, spaces, device: FlattenObservations(),  # type: ignore
    )
    # NOTE VP 2026.03.23. : Eval during training does not really influence anything -- check whether the model checkpointing depends on this
    # Evaluation EnvRunners get log_full_info=True so step() includes a deep
    # copy of named state in info["state"] — needed by trajectory logging.
    # Training EnvRunners are unaffected (no extra memory overhead).
    eval_env_config = {"log_full_info": True} if log_trajectories else {}
    config.evaluation(
        # evaluation_interval=1 ensures `evaluation/env_runners/<metric>` is present in every
        # iteration result, which is required by tune.TuneConfig(metric=...) — it performs a strict
        # check and crashes if the metric is absent (as happens with interval > 1 before the first
        # eval run). Alternative workaround: set os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"]
        # = "1" and keep a higher interval, but that silences all metric validation.
        evaluation_interval=1,
        evaluation_duration_unit="episodes",
        evaluation_duration=2,  # e.g., 2 episodes
        # True only if `evaluation_num_env_runners` > 0
        evaluation_parallel_to_training=False,
        evaluation_config=AlgorithmConfig.overrides(env_config=eval_env_config),
    )

    # TODO VP 2026.02.11. : Check this out in HPC
    # config.training(gamma=0.995)

    config.logger_config = {
        "type": "ray.tune.logger.UnifiedLogger",
        "loggers": [
                "ray.tune.json.JsonLoggerCallback",
                "ray.tune.csv.CSVLoggerCallback",
                # TODO VP 2026.03.16. : Fire up tensorboard...
                "ray.tune.tensorboardx.TBXLoggerCallback",
        ],
    }

    # Create callback classes for episode metrics and (optionally) trajectory logging.
    # Each factory returns a configured RLlibCallback subclass.
    # Link: https://docs.ray.io/en/latest/rllib/rllib-callback.html
    exec_date = datetime.datetime.now()

    episode_metrics_class = make_episode_metrics_callback_class(
        env_id=env_id,
        rewards=rewards,
        metrics_base_dir=f"{metrics_base_dir}/metrics",
        exec_date=exec_date,
        dump_metrics_json=False
    )

    # Assemble the callbacks_class list: checkpoint + metrics (always), trajectory (optional)
    callback_classes = [checkpoint_callback_class, episode_metrics_class]
    if log_trajectories:
        trajectory_class = make_trajectory_logging_callback_class(
            rewards=rewards,
            metrics_base_dir=f"{metrics_base_dir}/trajectories",
            exec_date=exec_date,
        )
        callback_classes.append(trajectory_class)
        logger.info("Trajectory logging enabled: per-step trajectory JSON will be saved for each episode.")

    # on_train_result callable for iteration-aligned data variant scheduling (Approach D1)
    # DataCombinator is always present; an empty one (no variants) is a safe no-op
    # because create_data_schedule_on_train_result early-returns when variant is empty.
    
    callback_kwargs = {
        "on_train_result": create_data_schedule_on_train_result(
            data_combinator, data_combinator.swap_every_n_episodes,
        ),
    }
    logger.info(
        "DataScheduleCallback: swap every %d iterations, %d variants",
        data_combinator.swap_every_n_episodes, len(data_combinator.variants),
    )

    # Register all callback classes + optional callable-based callbacks.
    # RLlib executes subclass callbacks in list order, then callables.
    config.callbacks(callbacks_class=callback_classes, **callback_kwargs)

    # Validate resource allocation against SLURM constraints
    driver_cpus = 1
    learner_total_cpus = num_learners * num_cpus_per_learner
    total_cpu_usage = driver_cpus + learner_total_cpus + (num_env_runners * num_cpus_per_env_runner)
    unused_cpus = num_cpus - total_cpu_usage

    allocation = ResourceAllocation(
        total_cpu_usage=total_cpu_usage,
        unused_cpus=unused_cpus,
        driver_cpus=driver_cpus,
        num_learners=num_learners,
        cpus_per_learner=num_cpus_per_learner,
        learner_total_cpus=learner_total_cpus,
        actual_env_runners=num_env_runners,
        cpus_per_env_runner=num_cpus_per_env_runner,
        slurm_cpus=num_cpus,
        slurm_gpus=num_gpus,
    )

    # Convert config to dict for validation
    param_space = config.to_dict()
    validate_resource_allocation(allocation, param_space)

    return config
