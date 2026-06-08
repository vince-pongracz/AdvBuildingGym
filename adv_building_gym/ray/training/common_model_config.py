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
    """Register episode-metric, trajectory, and scheduling callbacks on *config*.

    Mutates *config* in place via ``config.callbacks()``. Must be called after
    ``resource_setup`` has set the env-runner count, since the schedule callbacks
    read ``config.num_env_runners`` to floor their swap window (one episode per
    runner between swaps).

    Data-variant selection is intentionally not a callback here — it is
    env-side (see :mod:`adv_building_gym.core._data_variant_manager`).

    Args:
        config: Algorithm config object to register callbacks on.
        metrics_base_dir: Base directory for episode metrics.
        log_trajectories: Save per-step trajectory JSON during evaluation.
        reward_schedule_manager: Optional reward schedule manager.
        infra_combinator: Optional infrastructure schedule combinator.
    """
    # The env-runner count was set earlier by resource_setup (algorithm-specific).
    # Read it back here so the swap-gate floor matches the actual sampling topology.
    num_env_runners = config.num_env_runners

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

    # on_train_result callables run at iteration boundaries. RLlib accepts a
    # single on_train_result callable, so compose them when several are active.
    #
    # NOTE: the data-variant schedule is intentionally NOT wired here. Variant
    # selection is fully env-side (AdvBuildingGym.reset -> DataVariantManager):
    # each runner picks its variant from episode_count // swap_every_n_episodes
    # (training) or a fresh random draw (eval). The old iteration-boundary push
    # was overwritten by the very next reset(), so it had no effect — see
    # core/_data_variant_manager.py.
    on_train_result_fns = [
        # Mirror the eval return to a flat top-level result key so Tune's
        # CheckpointConfig(checkpoint_score_attribute=...) can rank checkpoints
        # by best eval performance (a slashed key is silently ignored — see
        # eval_score_callback). Also logs the save/keep/evict decision per
        # checkpoint. Runs every iteration; carries forward on non-eval iters.
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

    # Register all callback classes + optional callable-based callbacks.
    # RLlib executes subclass callbacks in list order, then callables.
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
    """
    Apply common, algorithm-independent RLlib configuration.

    Configures the settings shared across all algorithms:
    - API stack (RL module and learner, env runner and connector v2)
    - Environment configuration (retrieves action space from env_creator)
    - Debugging / reporting / framework settings
    - Sampling config: rollout_fragment_length, episode_lookback_horizon, connectors
    - Evaluation settings
    - Logger configuration
    - Callbacks (episode metrics, eval trajectories, optional schedule callbacks)

    Must run AFTER ``resource_setup`` (which sets the learner / env-runner resources
    and the algorithm-specific ``num_env_runners``): the schedule callbacks read the
    final ``config.num_env_runners`` to floor their swap window. Resource allocation
    and validation themselves are intentionally NOT done here — they live in
    ``resource_setup``.

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
        Configured algorithm config
    """
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
    # Sampling actions (querying the env, using the policy, sample trajectories) -- no GPU needed.
    # Per-key history stacking is handled inside HistoryWrapper (env wrapper); the
    # pipeline here only needs FlattenObservations. The env-runner *count* and the
    # learner/env-runner resource shares are set later in resource_setup (they depend
    # on the SLURM budget and the algorithm); here we only set sampling behaviour.
    config.env_runners(
        rollout_fragment_length=env_config.EPISODE_LENGTH, # Collect complete episodes before returning to learner.
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
    # eval_mode=True makes each eval episode draw a fresh random data variant
    # (independent (variant, day) per episode) instead of following the
    # training swap cadence — see core/_data_variant_manager.select_variant.
    # Training EnvRunners are unaffected (no extra memory overhead).
    eval_env_config = {"log_full_info": True, "eval_mode": True}

    # evaluation_interval > 1 means the `evaluation/env_runners/` keys are
    # absent from results on non-eval iterations.  Tune's strict metric check
    # would crash, so TUNE_DISABLE_STRICT_METRIC_CHECKING must be set in the
    # driver process (run_train_ray.py).
    config.evaluation(
        # evaluation_num_env_runners=1, # not important for now
        evaluation_interval=training_config.evaluation_interval,  # RLlib default: None
        evaluation_duration_unit="episodes",  # RLlib default
        evaluation_duration=10,  # RLlib default: 10
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

    # Callbacks read the env-runner count from the config (set by resource_setup),
    # so this must run after resource_setup.
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
