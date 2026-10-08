"""Multi-agent Ray RLlib training script for AdvBuildingGym.

Implements Flavour A from ``docs/about_multi_agent.md``: each actuator
(``a_<name>``) becomes an agent with its own RLModule. Observations are
shared across agents; per-agent reward routing is configurable through
:class:`MultiAgentAdvBuildingGym` (cooperative by default — every agent
sees the global aggregated reward).

This is the multi-agent sibling of :mod:`run_train_ray`. The two scripts
deliberately share libraries (env config, callbacks, data combinator,
reward schedule manager) and only differ in orchestration:
    * env registered via :func:`adv_building_ma_env_creator`
    * RLlib's ``multi_agent(...)`` and ``MultiRLModuleSpec`` are wired
      per agent on top of the standard ``common_model_setup``.

Link: docs/about_multi_agent.md
Link: https://docs.ray.io/en/latest/rllib/multi-agent-envs.html
"""

# TODO VP 2026.09.30.: not used

import os
import sys
import time
import json
import datetime
import logging
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.registry import register_env

from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

from adv_building_gym.ray.utils.warning_filters import setup_warning_filters
from adv_building_gym.config.data.data_combinator import DataCombinator
from adv_building_gym.config.env.env_config_manager import EnvConfigManager
from adv_building_gym.config.env.env_config import EnvConfig
from adv_building_gym.config.data.data_config import load_data_combinator_config
from adv_building_gym.config.rewards.reward_schedule_manager import RewardScheduleManager, RewardScheduleMode

from adv_building_gym.config.training.training_param_config import TrainingParamConfig
from adv_building_gym.ray.ma_env import MultiAgentAdvBuildingGym
from adv_building_gym.ray.env_creator import adv_building_ma_env_creator, merge_env_context
from adv_building_gym.config.env.infra_combinator import InfraCombinator
from adv_building_gym.ray.training import common_model_setup, select_model, resource_setup
from adv_building_gym.ray.callbacks import EVAL_SCORE_KEY, CHECKPOINT_NUM_TO_KEEP
from adv_building_gym._common.json_encoder import CustomJSONEncoder
from adv_building_gym._common.resource_check_util import SlurmResources
from adv_building_gym.ray.utils.ray_utils import make_trial_dirname_creator
from adv_building_gym._common.startup_log import log_startup_banner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True,
)
logger = logging.getLogger("ma_main")

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
setup_warning_filters()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-agent training (per-actuator policies) for AdvBuildingGym."
    )
    parser.add_argument(
        "--trial", type=str, required=True,
        help="Path to trial config YAML (e.g. configs/trial_cfgs/trial_cfg_1.yaml)",
    )
    parser.add_argument(
        "--reward-partition", type=str, default=None,
        help="Optional JSON file mapping agent_id → list of reward names. "
             "When omitted every agent receives the global aggregated reward "
             "(cooperative MARL).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading (mirrors run_train_ray._load_configs)
# ---------------------------------------------------------------------------

@dataclass
class LoadedConfigs:
    training_param_config: TrainingParamConfig
    active_config: EnvConfig
    data_combinator: DataCombinator
    reward_manager: RewardScheduleManager
    infra_combinator: Optional[InfraCombinator]
    reward_partition: Optional[dict[str, list[str]]]


def _load_reward_partition(path: str | None) -> dict[str, list[str]] | None:
    if path is None:
        return None
    p = Path(path).expanduser().resolve()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"reward_partition at {p} must be a JSON object.")
    return {str(k): list(v) for k, v in data.items()}


def _load_configs(cli_args: argparse.Namespace) -> tuple[LoadedConfigs, "TrialConfig"]:
    """Load all configs upfront via TrialConfig and adapt to MARL-specific extras."""
    from adv_building_gym.config.trial_config import TrialConfig
    trial = TrialConfig.load(cli_args.trial)

    trial.env_config.init_singletons()

    return LoadedConfigs(
        training_param_config=trial.training_param_config,
        active_config=trial.env_config,
        data_combinator=trial.data_combinator,
        reward_manager=trial.reward_manager,
        infra_combinator=trial.infra_combinator,
        reward_partition=_load_reward_partition(cli_args.reward_partition),
    ), trial


# ---------------------------------------------------------------------------
# Ray + RNG bootstrap (identical contract to run_train_ray)
# ---------------------------------------------------------------------------

def _init_ray() -> SlurmResources:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    cpus = int(slurm_cpus) if slurm_cpus and slurm_cpus.isdigit() else 2

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        gpus = len([x for x in cuda_visible.split(",") if x.strip()])
        if not torch.cuda.is_available():
            logger.error("CUDA_VISIBLE_DEVICES=%s but torch.cuda unavailable.", cuda_visible)
            sys.exit(1)
    else:
        logger.error("No GPU allocated — submit with --gres=gpu:1.")
        sys.exit(1)

    slurm_resources = SlurmResources(num_cpus=cpus, num_gpus=gpus)
    ray.init(
        num_cpus=slurm_resources.num_cpus,
        num_gpus=slurm_resources.num_gpus,
        ignore_reinit_error=True,
        runtime_env={"env_vars": RUNTIME_ENV_VARS},
        logging_level=logging.INFO,
    )
    return slurm_resources


# ---------------------------------------------------------------------------
# Multi-agent wiring
# ---------------------------------------------------------------------------

def _probe_agents(configs: LoadedConfigs) -> tuple[list[str], dict, dict]:
    """Build a throwaway MA env once on the driver to read per-agent spaces.

    RLlib needs per-policy observation/action spaces upfront via
    ``MultiRLModuleSpec``. The cleanest source of truth is the env
    itself; we instantiate one driver-side, read ``possible_agents``
    plus the space dicts, and discard.
    """
    probe_env: MultiAgentAdvBuildingGym = adv_building_ma_env_creator({
        "env_config": configs.active_config,
        "reward_schedule_manager": configs.reward_manager,
        "data_combinator": configs.data_combinator,
        "reward_partition": configs.reward_partition,
    })
    agent_ids = list(probe_env.possible_agents)
    obs_spaces = dict(probe_env.observation_spaces)
    act_spaces = dict(probe_env.action_spaces)
    logger.info("Discovered %d agents: %s", len(agent_ids), agent_ids)
    return agent_ids, obs_spaces, act_spaces


def _apply_multi_agent_config(
    algo_config,
    agent_ids: list[str],
    obs_spaces: dict,
    act_spaces: dict,
):
    """Layer per-agent policies and RLModule specs onto a base config.

    Identity policy mapping (agent_id == policy_id) is the standard
    Flavour A choice — there is no policy reuse, each actuator has
    its own NN.
    """
    algo_config.multi_agent(
        policies=set(agent_ids),
        policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
    )

    per_policy_specs = {
        aid: RLModuleSpec(
            observation_space=obs_spaces[aid],
            action_space=act_spaces[aid],
            model_config=DefaultModelConfig(
                fcnet_hiddens=[256, 256],
                fcnet_activation="tanh",
            ),
        )
        for aid in agent_ids
    }
    algo_config.rl_module(
        rl_module_spec=MultiRLModuleSpec(rl_module_specs=per_policy_specs),
    )
    return algo_config


# ---------------------------------------------------------------------------
# Algorithm config + checkpoint cadence
# ---------------------------------------------------------------------------

def _build_algo_config(trial, configs: LoadedConfigs, slurm_resources, exec_date_dt,
                    agent_ids: list[str], obs_spaces, act_spaces):
    # 1. Algorithm-specific config (hyperparameters + RLModule).
    algo_config = select_model(
        algorithm=trial.algorithm,
        env_config=configs.active_config,
        training_config=configs.training_param_config,
    )
    # 2. Resource-dependent config (learner/env-runner resources + count, validation).
    algo_config = resource_setup(
        config=algo_config,
        slurm_resources=slurm_resources,
        training_config=configs.training_param_config,
        env_config=configs.active_config,
    )
    # 3. Common, algorithm-independent config (env, connectors, eval, logger, callbacks).
    #    Callbacks read the env-runner count set by resource_setup above.
    algo_config = common_model_setup(
        config=algo_config,
        training_config=configs.training_param_config,
        env_config=configs.active_config,
        metrics_base_dir="ep_metrics",
        log_trajectories=trial.log_trajectories,
        reward_schedule_manager=configs.reward_manager,
        infra_combinator=configs.infra_combinator,
        exec_date=exec_date_dt,
        trial_name=trial.trial_name,
    )
    algo_config = _apply_multi_agent_config(algo_config, agent_ids, obs_spaces, act_spaces)

    param_space = algo_config.to_dict()
    with open("param_space.json", "w", encoding="utf-8") as f:
        json.dump(param_space, f, cls=CustomJSONEncoder, indent=4)
    return algo_config, param_space



# ---------------------------------------------------------------------------
# Tuner
# ---------------------------------------------------------------------------

def _build_progress_reporter() -> CLIReporter:
    return CLIReporter(
        metric_columns={
            "training_iteration": "Iter",
            "env_runners/num_episodes_lifetime": "Episodes",
            "time_total_s": "Time",
            "evaluation/env_runners/episode_return_mean": "EpReturnMean",
            "evaluation/env_runners/achieved_reward": "AchievedReward",
        },
        max_report_frequency=30,
        print_intermediate_tables=True,
    )


def _build_tuner(trial, metric, param_space, run_name, storage_path, checkpoint_freq_iterations):
    return tune.Tuner(
        trial.algorithm.upper(),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            reuse_actors=True,
            max_concurrent_trials=1,
            metric=metric,
            mode="max",
            trial_dirname_creator=make_trial_dirname_creator(trial.trial_name),
        ),
        run_config=tune.RunConfig(
            name=run_name,
            storage_path=storage_path,
            stop={"env_runners/num_episodes_lifetime": trial.training_param_config.max_episodes_to_run},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=checkpoint_freq_iterations,
                num_to_keep=CHECKPOINT_NUM_TO_KEEP,
                # Flat top-level key published every iter by the eval-score promote
                # callback (wired via common_model_setup -> register_callbacks). A
                # slashed key would be silently ignored by Tune's CheckpointManager,
                # degrading retention to keep-most-recent.
                checkpoint_score_attribute=EVAL_SCORE_KEY,
                checkpoint_score_order="max",
            ),
            progress_reporter=_build_progress_reporter(),
            verbose=2,
        ),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cli_args = _parse_cli_args()
    configs, trial = _load_configs(cli_args)

    metric = trial.metric
    if not metric.startswith("evaluation/"):
        metric = f"evaluation/env_runners/{metric}"

    # Banner-friendly Namespace mirroring the legacy fields.
    banner_args = argparse.Namespace(
        algorithm=trial.algorithm,
        episodes=trial.training_param_config.max_episodes_to_run,
        seed=trial.seed,
        metric=metric,
        # Single cadence knob: eval + checkpoint share training_params.common.evaluation.interval.
        checkpoint_frequency_iterations=trial.training_param_config.evaluation.interval,
        log_trajectories=trial.log_trajectories,
        grad_train=trial.grad_train,
        trial_name=trial.trial_name,
        trial_path=str(trial.source_path) if trial.source_path else None,
    )
    logger.info("Trial '%s' loaded; reward_partition=%s",
                trial.trial_name, cli_args.reward_partition)

    slurm_resources = _init_ray()

    # Probe per-agent spaces BEFORE registering the env so the algo config
    # can reference them. The probe env is discarded after introspection.
    agent_ids, obs_spaces, act_spaces = _probe_agents(configs)

    exec_date_dt = datetime.datetime.now()
    exec_date = exec_date_dt.strftime("%Y%m%d_%H%M%S")
    run_name = f"ma_{trial.algorithm}_seed{trial.seed}_{exec_date}"
    storage_path = os.path.abspath(
        f"models/{trial.trial_name}/ray_ma/{trial.algorithm}"
    )
    os.makedirs(storage_path, exist_ok=True)

    env_creator_config = {
        "env_config": configs.active_config,
        "data_combinator": configs.data_combinator,
        "reward_schedule_manager": configs.reward_manager,
        "reward_partition": configs.reward_partition,
        # Anchor the per-env construction seed on the ENV axis (trial `env_seed:`, default
        # = `seed:`); the creator offsets it by worker/vector index. The learner keeps
        # `seed:` via config.debugging, so the two axes can be swept independently.
        "seed": trial.env_seed,
    }
    register_env(
        "AdvBuildingMA",
        lambda cfg: adv_building_ma_env_creator(merge_env_context(env_creator_config, cfg)),
    )

    # common_model_setup binds the env name "AdvBuilding" — override with
    # the MA env now that it's registered. clip_actions stays True (the
    # MA wrapper rescales unit actions internally; clipping ensures no
    # numerical drift past the [-1, 1] frame).
    algo_config, _ = _build_algo_config(
        trial, configs, slurm_resources, exec_date_dt,
        agent_ids, obs_spaces, act_spaces,
    )
    algo_config.environment(
        env="AdvBuildingMA",
        clip_actions=configs.training_param_config.clip_actions_to_env_bounds,
    )
    param_space = algo_config.to_dict()

    tuner = _build_tuner(trial, metric, param_space, run_name, storage_path, trial.training_param_config.evaluation.interval)

    experiment_path = os.path.join(storage_path, run_name)
    log_startup_banner(
        args=banner_args,
        env_config=configs.active_config,
        training_param_config=configs.training_param_config,
        reward_manager=configs.reward_manager,
        data_combinator=configs.data_combinator,
        infra_combinator=configs.infra_combinator,
        slurm_resources=slurm_resources,
        run_name=run_name,
        experiment_path=experiment_path,
        storage_path=storage_path,
        seed=trial.seed,
        exec_date=exec_date_dt,
    )

    logger.info("Starting MARL tuner.fit(): %s (agents=%s)", run_name, agent_ids)
    t0 = time.time()
    results = tuner.fit()
    logger.info("Training finished in %.2f min", (time.time() - t0) / 60)

    try:
        best = results.get_best_result(metric=metric, mode="max")
        if best:
            logger.info("Best trial: %s (ckpt: %s)",
                        best.path, best.checkpoint.path if best.checkpoint else "n/a")
    except Exception as e:  # noqa: BLE001
        logger.error("get_best_result failed: %s", e)

    ray.shutdown()


if __name__ == "__main__":
    main()

# Usage:
# python rl_ma_train.py --trial configs/trial_cfgs/trial_cfg_1.yaml
# python rl_ma_train.py --trial configs/trial_cfgs/trial_cfg_1.yaml --reward-partition configs/ma/reward_partition.json
