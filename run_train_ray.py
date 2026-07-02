"""Ray RLlib training script for AdvBuildingGym environment.

Single CLI entry point: ``--trial <trial_cfg.yaml>``. The trial config
bundles algorithm + run control + env topology + training hyperparameters
+ data/reward/infra schedules. ``adv_building_gym.config.TrialConfig.load``
reads it once and resolves every sub-config so downstream modules receive
already-loaded objects.
"""

import os
import sys
import time
import datetime
import logging
from pathlib import Path

import json
import torch

import ray
from ray import tune

from ray.tune import CLIReporter
from ray.tune.registry import register_env

from adv_building_gym.ray.utils.warning_filters import setup_warning_filters
from adv_building_gym.config.trial_config import TrialConfig
from adv_building_gym.ray.env_creator import adv_building_env_creator, merge_env_context
from adv_building_gym.ray.training import common_model_setup, select_model, resource_setup
from adv_building_gym.ray.callbacks import EVAL_SCORE_KEY, CHECKPOINT_NUM_TO_KEEP
from adv_building_gym._common.json_encoder import CustomJSONEncoder
from adv_building_gym._common.resource_check_util import SlurmResources
from adv_building_gym.ray.utils.ray_utils import make_trial_dirname_creator
from adv_building_gym.ray.utils.early_stopping import build_stop_criteria
from adv_building_gym._common.startup_log import log_startup_banner

from run_train_util import parse_cli_args, trial_to_args_namespace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True,
)
logger = logging.getLogger("main")

# Environment variables to control Ray/RLlib behavior (must be set before ray.init).
# RAY_AIR_NEW_OUTPUT=0 keeps tune.RunConfig(progress_reporter=CLIReporter(...)) honoured.
# TUNE_DISABLE_STRICT_METRIC_CHECKING allows evaluation_interval > 1 without crashing on
# iterations that skip eval.
RUNTIME_ENV_VARS = {
    "PYTHONWARNINGS": "ignore::DeprecationWarning,ignore::UserWarning",
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "RAY_METRICS_SERVICE_ENABLED": "0",
    "RAY_event_stats": "0",
    "RAY_DEDUP_LOGS": "0",
    "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
    "RAY_AIR_NEW_OUTPUT": "0",
    "TUNE_DISABLE_STRICT_METRIC_CHECKING": "1",
    "RAY_COLOR_PREFIX": os.environ.get("RAY_COLOR_PREFIX", ""),
    "TERM": os.environ.get("TERM", ""),
}
os.environ.update(RUNTIME_ENV_VARS)

# Apply warning filters in the main process
setup_warning_filters()

logger.info("Runtime environment variables for Ray workers: %s", RUNTIME_ENV_VARS)


# ---------------------------------------------------------------------------
# Ray initialisation
# ---------------------------------------------------------------------------

def _init_ray(cpu_only: bool = False) -> SlurmResources:
    """Resolve SLURM resources and init Ray.

    Per-env seeding is handled by RLlib (``config.debugging(seed=...)`` →
    ``trial.seed + worker_index`` applied on each env's first reset), so no
    central RNG service is needed.
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    cpus = int(slurm_cpus) if slurm_cpus and slurm_cpus.isdigit() else 2

    if cpu_only:
        gpus = 0
        logger.warning("CPU-only smoke-test mode: running learner on CPU (--cpu).")
    else:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            gpus = len([x for x in cuda_visible.split(",") if x.strip() != ""])
            if not torch.cuda.is_available():
                logger.error(
                    "SLURM allocated GPUs (CUDA_VISIBLE_DEVICES=%s) but PyTorch "
                    "cannot access CUDA. Check driver/CUDA toolkit setup.",
                    cuda_visible,
                )
                sys.exit(1)
        else:
            logger.error(
                "No GPU allocated (CUDA_VISIBLE_DEVICES is not set). "
                "Training requires a GPU — submit with --gres=gpu:1, "
                "or pass --cpu for a CPU-only smoke test.",
            )
            sys.exit(1)

    slurm_resources = SlurmResources(num_cpus=cpus, num_gpus=gpus)
    logger.info(
        "Training on device: %s (%d GPU(s))",
        "cpu" if gpus == 0 else "cuda", slurm_resources.num_gpus,
    )

    logger.info(
        "Initializing Ray with cpus=%s gpus=%s (from SLURM/CUDA env)",
        slurm_resources.num_cpus, slurm_resources.num_gpus,
    )
    out = ray.init(
        num_cpus=slurm_resources.num_cpus,
        num_gpus=slurm_resources.num_gpus,
        ignore_reinit_error=True,
        runtime_env={"env_vars": RUNTIME_ENV_VARS},
        logging_level=logging.INFO,
    )

    return slurm_resources


# ---------------------------------------------------------------------------
# Algorithm config + checkpoint cadence
# ---------------------------------------------------------------------------

def _build_algo_config(args, trial: TrialConfig, slurm_resources, exec_date_dt):
    """Assemble the RLlib algorithm config and return ``(algo_config, param_space)``."""
    # 1. Algorithm-specific config (hyperparameters + RLModule).
    algo_config = select_model(
        algorithm=trial.algorithm,
        episode_length=trial.env_config.EPISODE_LENGTH,
        training_config=trial.training_param_config,
    )
    # 2. Resource-dependent config (learner/env-runner resources + count, validation).
    algo_config = resource_setup(
        config=algo_config,
        slurm_resources=slurm_resources,
        training_config=trial.training_param_config,
        env_config=trial.env_config,
    )
    # 3. Common, algorithm-independent config (env, connectors, eval, logger, callbacks).
    #    Callbacks read the env-runner count set by resource_setup above.
    algo_config = common_model_setup(
        config=algo_config,
        training_config=trial.training_param_config,
        env_config=trial.env_config,
        metrics_base_dir="ep_metrics",
        log_trajectories=trial.log_trajectories,
        reward_schedule_manager=trial.reward_manager,
        infra_combinator=trial.infra_combinator,
        statesource_combinator=trial.statesource_combinator,
        exploration_reset=trial.exploration_reset,
        exec_date=exec_date_dt,
        trial_name=trial.trial_name,
    )
    param_space = algo_config.to_dict()

    with open("param_space.json", "w", encoding="utf-8") as f:
        json.dump(param_space, f, cls=CustomJSONEncoder, indent=4)

    return algo_config, param_space


# ---------------------------------------------------------------------------
# Tuner construction
# ---------------------------------------------------------------------------

def _build_progress_reporter(algorithm: str) -> CLIReporter:
    """CLIReporter showing the algorithm-specific learner stats.

    Requires RAY_AIR_NEW_OUTPUT=0 — otherwise Tune's AIR output ignores
    ``metric_columns`` and renders its own default tables.
    """
    if algorithm == "ppo":
        algo_cols = {
            "learners/default_policy/policy_loss": "PolLoss",
            "learners/default_policy/vf_loss": "VfLoss",
            "learners/default_policy/mean_kl_loss": "KL",
            "learners/default_policy/mean_entropy": "Ent",
        }
    elif algorithm == "sac":
        algo_cols = {
            "learners/default_policy/critic_loss": "QLoss",
            "learners/default_policy/actor_loss": "PiLoss",
            "learners/default_policy/alpha_value": "Alpha",
            "learners/default_policy/td_error_mean": "TDErr",
        }
    elif algorithm == "dreamerv3":
        # TODO VP: Verify these metric paths against an actual DreamerV3
        # result.json (per memory feedback_rllib_new_api_stack.md). DreamerV3
        # learner logs world-model / actor / critic losses with names that
        # depend on the installed Ray version.
        algo_cols = {
            "learners/default_policy/WORLD_MODEL_total_loss": "WMLoss",
            "learners/default_policy/ACTOR_loss": "PiLoss",
            "learners/default_policy/CRITIC_L_total": "VfLoss",
        }
    else:
        algo_cols = {}

    return CLIReporter(
        metric_columns={
            "training_iteration": "Iter",
            "env_runners/num_episodes_lifetime": "Episodes",
            "time_total_s": "Time",
            "evaluation/env_runners/episode_return_mean": "EpReturnMean",
            "evaluation/env_runners/achieved_reward": "AchievedReward",
            **algo_cols,
        },
        max_report_frequency=30, # Report every 30 seconds
        print_intermediate_tables=True,
    )


def _build_tuner(trial: TrialConfig, metric: str, param_space, run_name, storage_path, 
                checkpoint_freq_iterations, trial_name: str | None = None):
    """Build the ``tune.Tuner`` for the chosen algorithm."""
    # Hard episode cap (1 episode = 1 day at 5-min control step) plus optional
    # episode-unit early stopping on the held-out eval metric. When early stopping is
    # disabled this is the legacy single-key dict; enabled → a CombinedStopper.
    stop_criteria = build_stop_criteria(trial.training_param_config, metric, mode="max")
    progress_reporter = _build_progress_reporter(trial.algorithm)

    # Ray's algorithm registry is case-sensitive: "PPO"/"SAC" are all-caps,
    # but DreamerV3 is mixed-case — see ray.rllib.algorithms.registry.
    trainable_name = {"ppo": "PPO", "sac": "SAC", "dreamerv3": "DreamerV3"}[trial.algorithm]

    return tune.Tuner(
        trainable_name,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            reuse_actors=True,
            max_concurrent_trials=1,
            # Metrics (auto-prefixed with env_runners/):
            #   - "episode_return_mean"  (default RLlib metric)
            #   - "achieved_reward" (custom: sum of rewards per episode)
            metric=metric,
            mode="max",
            trial_dirname_creator=make_trial_dirname_creator(trial_name),
        ),
        run_config=tune.RunConfig(
            name=run_name,
            storage_path=storage_path,
            stop=stop_criteria,
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=checkpoint_freq_iterations,
                num_to_keep=CHECKPOINT_NUM_TO_KEEP,
                # NOTE: a slashed key (e.g. "evaluation/env_runners/episode_return_mean")
                # is silently ignored by Tune's CheckpointManager (its insertion gate
                # tests membership against the un-flattened result dict) → retention
                # degrades to keep-most-recent. EVAL_SCORE_KEY is a flat top-level key
                # published every iteration by the eval-score promote callback.
                checkpoint_score_attribute=EVAL_SCORE_KEY,
                checkpoint_score_order="max",
            ),
            progress_reporter=progress_reporter,
            verbose=2, # 0=silent, 1=less, 2=default, 3=verbose
        ),
    )


# ---------------------------------------------------------------------------
# Result post-processing
# ---------------------------------------------------------------------------

def _log_best_result(results, metric: str, storage_path: str) -> None:
    """Log the best trial's path, checkpoint, and key metrics."""
    try:
        best_result = results.get_best_result(metric=metric, mode="max")
        if best_result:
            logger.info("=" * 70)
            logger.info("Best trial results:")
            logger.info("  Trial directory: %s", best_result.path)
            if best_result.checkpoint:
                logger.info("  Best checkpoint: %s", best_result.checkpoint.path)
            else:
                logger.warning("  No checkpoint available for best trial")
            logger.info("  Storage path: %s", storage_path)
            logger.info("=" * 70)

        if best_result and best_result.metrics:
            env_runners_metrics = best_result.metrics.get("env_runners", {})
            metrics_to_log = {
                "episode_return_mean": env_runners_metrics.get("episode_return_mean"),
                "achieved_reward": env_runners_metrics.get("achieved_reward"),
            }
            metric_key = metric.split("/")[-1]
            opt_value = metrics_to_log.get(metric_key)
            if opt_value is not None:
                logger.info("Optimized metric (%s): %.4f", metric, opt_value)
            for name, value in metrics_to_log.items():
                if value is not None and name != metric_key:
                    logger.info("  %s: %.4f", name, value)
        else:
            logger.warning("No best result or metrics available")
    except Exception as e:
        logger.error("Error retrieving best result: %s", str(e))


def _dump_all_results(results) -> None:
    """Write each trial's config + metrics + path to ``result_<i>.json``."""
    for i, res in enumerate(results._results):
        config_data = {}
        if res.config is not None:
            try:
                config_data = res.config.copy() if hasattr(res.config, "copy") else dict(res.config)
                if "_rl_module_spec" in config_data:
                    del config_data["_rl_module_spec"]
            except (TypeError, ValueError) as e:
                logger.warning("Could not serialize config for result %d: %s", i, e)
                config_data = {"error": str(e)}
        dump = {
            "config": config_data,
            "error": getattr(res, "error", None),
            "metrics": res.metrics if res.metrics is not None else {},
            "path": res.path,
        }
        with open(f"result_{i}.json", "w", encoding="utf-8") as f:
            json.dump(dump, f, cls=CustomJSONEncoder, indent=4)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Parse the trial path, load all configs upfront, and run training."""
    cli_args = parse_cli_args(
        description=(
            "Train an RL agent on AdvBuildingGym. The trial YAML "
            "bundles algorithm, env topology, hyperparameters, and schedules."
        ),
    )
    trial = TrialConfig.load(cli_args.trial)
    logger.info("Trial '%s' loaded from %s", trial.trial_name, trial.source_path)

    # Initialise singleton component instances exactly once (driver only).
    # Triggers CSV parsing here; Ray workers go through factory methods.
    trial.env_config.init_singletons()

    # Auto-prefix metric so it points at the eval env_runners by default.
    metric = trial.metric
    if not metric.startswith("evaluation/"):
        metric = f"evaluation/env_runners/{metric}"

    args = trial_to_args_namespace(trial)
    args.metric = metric  # banner uses the resolved metric

    slurm_resources = _init_ray(cpu_only=cli_args.cpu)

    exec_date_dt = datetime.datetime.now()
    exec_date = exec_date_dt.strftime("%Y%m%d_%H%M%S")
    run_name = f"{trial.algorithm}_seed{trial.seed}_{exec_date}"
    storage_path = os.path.abspath(f"models/{trial.trial_name}/ray/{trial.algorithm}")
    os.makedirs(storage_path, exist_ok=True)

    env_creator_config = {
        "seed": trial.seed,
        "env_config": trial.env_config,
        "data_combinator": trial.data_combinator,
        # Eval EnvRunners (eval_mode=True via the evaluation_config override) pick this
        # held-out combinator instead, so in-training eval rounds runs on the eval dataset.
        "eval_data_combinator": trial.eval_data_combinator,
        "reward_schedule_manager": trial.reward_manager,
    }
    register_env(
        "AdvBuilding",
        lambda cfg: adv_building_env_creator(merge_env_context(env_creator_config, cfg)),
    )

    _, algo_cfg_param_space = _build_algo_config(args, trial, slurm_resources, exec_date_dt)
    # Tie checkpoint cadence to the eval cadence so every checkpoint lands on a
    # fresh-eval iteration and can be ranked by eval return (best-N retention).
    tuner = _build_tuner(
        trial, metric, algo_cfg_param_space, run_name, storage_path,
        trial.training_param_config.evaluation.interval,
        trial_name=trial.trial_name,
    )

    log_startup_banner(
        args=args,
        env_config=trial.env_config,
        training_param_config=trial.training_param_config,
        reward_manager=trial.reward_manager,
        data_combinator=trial.data_combinator,
        infra_combinator=trial.infra_combinator,
        slurm_resources=slurm_resources,
        run_name=run_name,
        experiment_path=os.path.join(storage_path, run_name),
        storage_path=storage_path,
        seed=trial.seed,
        exec_date=exec_date_dt,
        eval_trajectories_path=os.path.abspath("ep_metrics/eval_trajectories"),
    )

    logger.info("Starting tuner.fit() for: %s", run_name)
    logger.info("Training progress will be displayed below:")
    logger.info("=" * 70)

    t0 = time.time()
    results = tuner.fit()
    elapsed_time = (time.time() - t0) / 60
    logger.info("=" * 70)
    logger.info("Tuner/training finished in %.2f min", elapsed_time)
    logger.info("=" * 70)

    _log_best_result(results, metric, storage_path)
    _dump_all_results(results)

    ray.shutdown()
    logger.info("Script completed")


if __name__ == "__main__":
    main()

# Usage examples:
# On slurm: sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/trial_cfg_1.yaml
#
# python run_train_ray.py --trial configs/trial_cfgs/trial_cfg_1.yaml
