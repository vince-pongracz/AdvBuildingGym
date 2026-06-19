"""Ray RLlib common configuration: env setup, resources, and callbacks."""

import datetime
import logging

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from adv_building_gym.config.env.env_config import EnvConfig
# History stacking is in HistoryWrapper (env wrapper); pipelines only need FlattenObservations.
from ray.rllib.connectors.env_to_module import FlattenObservations

from adv_building_gym.ray.callbacks import (
    create_eval_score_promote_on_train_result_cb,
    create_exploration_monitor_on_train_result_cb,
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
from adv_building_gym.config.env.infra_combinator import InfraCombinator
from adv_building_gym.config.env.statesource_combinator import StatesourceCombinator

logger = logging.getLogger(__name__)


def _compose_on_train_result(*fns):
    """Chain several on_train_result callables into one (RLlib accepts a single callable)."""
    def composed_on_train_result(*, algorithm, result, **kwargs):
        for fn in fns:
            fn(algorithm=algorithm, result=result, **kwargs)
    return composed_on_train_result


def register_callbacks(
    config: AlgorithmConfig,
    checkpoint_interval: int,
    metrics_base_dir: str = "ep_metrics",
    log_trajectories: bool = False,
    reward_schedule_manager: RewardScheduleManager | None = None,
    infra_combinator: InfraCombinator | None = None,
    statesource_combinator: StatesourceCombinator | None = None,
    exploration_reset: ExplorationResetConfig | None = None,
    exec_date: datetime.datetime | None = None,
    trial_name: str | None = None,
) -> None:
    """Register episode-metric, trajectory, and scheduling callbacks on *config* (in place).

    Must run after ``resource_setup`` (schedule callbacks read ``config.num_env_runners``
    to floor their swap window). Data-variant selection is env-side, not a callback here.
    """
    # env-runner count from resource_setup; read back so the swap-gate floor matches sampling
    num_env_runners = config.num_env_runners

    # callback classes for episode metrics (+ optional trajectory logging)
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
        trial_name=trial_name,
    )

    callback_classes = [episode_metrics_class, eval_state_action_class]
    if log_trajectories:
        trajectory_class = make_trajectory_logging_cb_class(
            metrics_base_dir=f"{metrics_base_dir}/trajectories",
            exec_date=exec_date,
        )
        callback_classes.append(trajectory_class)
        logger.info("Trajectory logging enabled: per-step trajectory JSON will be saved for each episode.")

    # on_train_result callables run at iteration boundaries (composed into one).
    # Data-variant schedule is NOT wired here — selection is env-side
    # (AdvBuildingGym.reset → DataVariantManager); an iteration-boundary push
    # was overwritten by the next reset(). See core/_data_variant_manager.py.
    on_train_result_fns = [
        # mirror eval return to a flat result key so Tune's checkpoint_score_attribute
        # can rank checkpoints (slashed keys are ignored); logs save/keep/evict each iter
        create_eval_score_promote_on_train_result_cb(checkpoint_interval=checkpoint_interval),
        # Warn when SAC's entropy temperature (alpha) collapses → exploration dies.
        create_exploration_monitor_on_train_result_cb(),
        create_iter_timing_on_train_result_cb(),
    ]

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

    # register class callbacks (run in list order) + callable callbacks
    config.callbacks(callbacks_class=callback_classes, **callback_kwargs)


def common_model_setup(
    config: AlgorithmConfig,
    training_config: TrainingParamConfig,
    env_config: EnvConfig,
    metrics_base_dir: str = "ep_metrics",
    log_trajectories: bool = False,
    reward_schedule_manager: RewardScheduleManager | None = None,
    infra_combinator: InfraCombinator | None = None,
    statesource_combinator: StatesourceCombinator | None = None,
    exploration_reset: ExplorationResetConfig | None = None,
    exec_date: datetime.datetime | None = None,
    trial_name: str | None = None,
):
    """Apply common, algorithm-independent RLlib config (API stack, env, debugging/reporting/
    framework, sampling connectors, evaluation, logger, callbacks).

    Must run AFTER ``resource_setup`` (schedule callbacks read the final ``config.num_env_runners``);
    resource allocation/validation live there, not here. Data-variant selection is env-side.

    Args:
        config: Algorithm config object (e.g., PPOConfig instance)
        training_config: Hyperparameters (seed, gamma, eval interval, ...)
        env_config: Provides EPISODE_LENGTH for the rollout fragment length.
        metrics_base_dir: Base directory for episode metrics (default: "ep_metrics")
        log_trajectories: When True, save full per-step trajectory JSON
            during evaluation episodes (via episode callback).

    Note:
        Data-variant selection is env-side (the combinator is wired to the
        envs by the env creator); there is no data-schedule callback here.

    Returns:
        Algorithm config
    """
    # obs_space omitted (FlattenObservations handles it); action_space omitted
    # (env_creator's FlattenAction + RescaleAction give RLlib a flat Box(-1, 1)).
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
        metrics_num_episodes_for_smoothing=training_config.episode_return_mean_window,  # RLlib default: 100
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
    config.training(gamma=training_config.gamma) # RLlib default: 0.99
    # Sampling (env queries, policy, trajectories) — no GPU. History stacking is in
    # HistoryWrapper, so the pipeline only needs FlattenObservations. Env-runner count
    # and resource shares are set later in resource_setup; here only sampling behaviour.
    config.env_runners(
        rollout_fragment_length=env_config.EPISODE_LENGTH, # Collect complete episodes before returning to learner.
        episode_lookback_horizon=training_config.episode_lookback_horizon_steps,  # RLlib default: 1
        env_to_module_connector=lambda env, spaces, device: [FlattenObservations()],  # type: ignore
    )
    # mirror the env-to-module pipeline on the learner side so batches flatten to the same dim
    config.training(
        learner_connector=lambda obs_sp, act_sp: [FlattenObservations()],  # type: ignore
    )
    # Eval runs the policy without exploration noise (unbiased selection signal; no gradients).
    # Eval EnvRunners get log_full_info=True (info["state"] for the eval trajectory callback)
    # and eval_mode=True (each episode draws a fresh random (variant, day)). Training unaffected.
    eval_env_config = {"log_full_info": True, "eval_mode": True}

    # evaluation_interval > 1 leaves evaluation/env_runners/ absent on non-eval iters,
    # so TUNE_DISABLE_STRICT_METRIC_CHECKING must be set in the driver (run_train_ray.py).
    config.evaluation(
        # evaluation_num_env_runners=1, # not important for now
        evaluation_interval=training_config.evaluation_interval,  # RLlib default: None
        evaluation_duration_unit="episodes",  # RLlib default
        evaluation_duration=training_config.evaluation_duration,  # RLlib default: 10
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

    # callbacks read num_env_runners (set by resource_setup), so run after it
    register_callbacks(
        config,
        checkpoint_interval=training_config.evaluation_interval,
        metrics_base_dir=metrics_base_dir,
        log_trajectories=log_trajectories,
        reward_schedule_manager=reward_schedule_manager,
        infra_combinator=infra_combinator,
        statesource_combinator=statesource_combinator,
        exploration_reset=exploration_reset,
        exec_date=exec_date,
        trial_name=trial_name,
    )

    return config
