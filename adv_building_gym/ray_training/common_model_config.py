"""
Ray RLlib common configuration utilities.

This module provides the common configuration function for RLlib algorithms,
including environment setup, resource allocation, and callback configuration.
"""

import datetime
import logging
from typing import Any, Callable, List

from ray.rllib.connectors.env_to_module import FlattenObservations

from adv_building_gym import create_on_episode_end_callback
from adv_building_gym.utils import ResourceAllocation, validate_resource_allocation
from .env_spaces import get_env_spaces

logger = logging.getLogger(__name__)


def common_model_config(
    config,
    seed: int,
    env_creator: Callable[[dict], Any],
    num_cpus: int,
    num_gpus: int,
    # TODO VP 2026.01.13. : Improve checkpoint directory structure
    # save Policy NN in
    checkpoint_callback_class: type,
    env_id: str,
    rewards: List,
    metrics_base_dir: str = "ep_metrics",
    clip_actions: bool = True,
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
    - Callbacks (creates on_episode_end_callback)
    - Resource validation

    Args:
        config: Algorithm config object (e.g., PPOConfig instance)
        seed: Random seed for reproducibility
        env_creator: Factory function that creates environment instances
        num_cpus: Total CPUs available (from Ray/SLURM)
        num_gpus: Total GPUs available (from Ray/SLURM)
        checkpoint_callback_class: Callback class for checkpoint management
        env_id: Environment ID string for logging
        rewards: List of reward functions used in the environment
        metrics_base_dir: Base directory for episode metrics (default: "ep_metrics")
        clip_actions: Whether to clip actions to action space bounds

    Returns:
        Configured algorithm config
    """
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

    # Get observation and action spaces from a temporary env instance
    # NOTE: We only provide action_space explicitly. The observation_space is
    # intentionally NOT provided because FlattenObservations connector transforms
    # the Dict obs space into a flat Box. If we provide the Dict space here,
    # RLlib's Catalog tries to build an encoder for Dict (unsupported) before
    # the connector can transform it.
    _obs_space, action_space = get_env_spaces(env_creator)
    logger.debug("Env action_space: %s (obs_space inferred after FlattenObservations)", action_space)

    config = config.api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    config.environment(
        env="AdvBuilding",
        # observation_space intentionally omitted - inferred after FlattenObservations
        action_space=action_space,
        normalize_actions=True,
        clip_actions=clip_actions,
    )
    config.debugging(
        # WARN: Reduces verbosity (suppress connector pipeline INFO messages)
        log_level="INFO",
        log_sys_usage=True,
        seed=seed
    )
    config.reporting(
        keep_per_episode_custom_metrics=True,
        metrics_num_episodes_for_smoothing=25,
    )
    config.framework(
        framework="torch",
        eager_tracing=True,
        eager_max_retraces=20,
        tf_session_args={},
        local_tf_session_args={},
    )
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
        num_envs_per_env_runner=1,
        num_cpus_per_env_runner=num_cpus_per_env_runner,
        num_gpus_per_env_runner=0,
        # Flatten dict observation space into a single vector for the RL module
        # NOTE VP 2026.01.05. : if other observation space needed for the policy, change the observations in the env.
        # (rather than using a new observation encoder -- that must be learnt as well, it overcomplicates things...)
        env_to_module_connector=lambda env, spaces, device: FlattenObservations(),  # type: ignore
    )
    config.evaluation(
        evaluation_interval=1,  # run evaluation every train() iteration
        evaluation_duration_unit="episodes",
        evaluation_duration=2,  # e.g., 2 episodes
        # True only if `evaluation_num_env_runners` > 0
        evaluation_parallel_to_training=False,
    )

    config.logger_config = {
        "type": "ray.tune.logger.UnifiedLogger",
        "loggers": [
                "ray.tune.json.JsonLoggerCallback",
                "ray.tune.csv.CSVLoggerCallback",
                "ray.tune.tensorboardx.TBXLoggerCallback",
        ],
    }

    # Create episode end callback for metrics logging
    exec_date = datetime.datetime.now()
    on_episode_end_callback = create_on_episode_end_callback(
        env_id=env_id,
        rewards=rewards,
        metrics_base_dir=metrics_base_dir,
        exec_date=exec_date
    )

    # Configure callbacks:
    # - on_episode_end: Logs episode metrics (reward_rate, achieved_reward, etc.)
    # - Checkpoint callback: Saves best model based on metric every N episodes
    #
    # RLlib calls on_episode_end callbacks first, then callback class methods -- ensures metrics are logged before checkpoint decisions are made.
    config.callbacks(
        checkpoint_callback_class,
        on_episode_end=on_episode_end_callback,  # Episode metrics callback (runs first)
    )

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
