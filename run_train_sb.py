"""Stable-Baselines3 training driver for AdvBuildingGym.

CLI mirrors ``run_train_ray.py``:

    python run_train_sb.py --trial configs/trial_cfgs/trial_cfg_1_sac.yaml
    python run_train_sb.py --trial configs/trial_cfgs/trial_cfg_1_sac.yaml --cpu

The same trial YAML drives both drivers. SB3-side notes:

* ``algorithm`` must be ``ppo`` or ``sac`` (SB3 has no DreamerV3).
* GPU is required by default; pass ``--cpu`` for a CPU-only smoke test
  (mirrors the Ray driver's behaviour).
* ``training_param_config`` (PPO / SAC sections) maps to SB3 hyperparams
  in ``adv_building_gym.sb/sb_training/select_model.py``; some Ray-only
  knobs (e.g. ``sac_n_step_return``) emit a warning and fall through to
  the SB3 default.
* All four schedules (data / reward / infra / statesource) are
  supported via episode-counting callbacks that push to every sub-env
  through ``vec_env.env_method(...)``.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import time
from pathlib import Path

import torch

from adv_building_gym.config.trial_config import TrialConfig
from adv_building_gym._common.json_encoder import CustomJSONEncoder
from adv_building_gym._common.warning_filters import setup_warning_filters
from adv_building_gym._common.startup_log import log_startup_banner

from adv_building_gym.sb.training import sb_common_model_setup, sb_select_model

from run_train_util import parse_cli_args, trial_to_args_namespace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True,
)
logger = logging.getLogger("main")

setup_warning_filters()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cli_args = parse_cli_args(
        description=(
            "Train an RL agent on AdvBuildingGym with Stable-Baselines3. "
            "The trial YAML bundles algorithm, env topology, hyperparameters, "
            "and schedules — the same file that drives run_train_ray.py."
        ),
    )
    trial = TrialConfig.load(cli_args.trial)
    logger.info("Trial '%s' loaded from %s", trial.trial_name, trial.source_path)

    # Driver-only singleton initialisation. The trial dataclasses are
    # picklable and reach SB3 worker subprocesses via the env-factory
    # closure; component instances are constructed FRESH per worker in
    # the factory, so this call only seeds the driver-side singletons
    # used by log_startup_banner / introspection.
    trial.env_config.init_singletons()

    # Single exec_date drives both the run-dir name in sb_common_model_setup
    # AND the eval-trajectories sub-run dir (so the TB layout matches Ray's
    # exactly). Captured here so log_startup_banner shows the same stamp.
    exec_date_dt = datetime.datetime.now()
    slurm, paths, train_vec, eval_vec, callback_list = sb_common_model_setup(
        trial, cpu_only=cli_args.cpu, exec_date=exec_date_dt,
    )

    device = "cpu" if slurm.num_gpus == 0 else "cuda"
    logger.info(
        "Training on device: %s (%d GPU(s)); SLURM CPUs: %d",
        device, slurm.num_gpus, slurm.num_cpus,
    )

    model = sb_select_model(
        algorithm=trial.algorithm,
        vec_env=train_vec,
        episode_length=trial.env_config.EPISODE_LENGTH,
        num_envs=trial.num_envs,
        training_config=trial.training_param_config,
        seed=trial.seed,
        device=device,
        tensorboard_log=paths.log_dir,
    )

    # Total timesteps from episode budget. The trial config expresses the
    # stop criterion in episodes (matches Ray's ``num_episodes_lifetime``
    # criterion); SB3 takes raw timesteps.
    total_timesteps = (
        trial.training_param_config.max_episodes_to_run * trial.env_config.EPISODE_LENGTH
    )
    logger.info(
        "Training for %d episodes (= %d timesteps) with metric=%s, num_envs=%d",
        trial.training_param_config.max_episodes_to_run, total_timesteps,
        trial.metric, trial.num_envs,
    )

    log_startup_banner(
        args=trial_to_args_namespace(trial),
        env_config=trial.env_config,
        training_param_config=trial.training_param_config,
        reward_manager=trial.reward_manager,
        data_combinator=trial.data_combinator,
        infra_combinator=trial.infra_combinator,
        slurm_resources=slurm,
        run_name=Path(paths.model_dir).name,
        experiment_path=paths.model_dir,
        # SB3 writes TB events under <run>/tb/, not at the run root, so the
        # "this run only" launch line needs the subdir. The "all runs"
        # comparison is the parent (= models/<trial>/sb3/<algo>/) so sibling
        # runs across seeds appear side-by-side in TB.
        tensorboard_log_path=paths.log_dir,
        storage_path=str(Path(paths.model_dir).parent),
        # SBEvalStateActionCallback writes the eval-trajectories sub-runs
        # under ep_metrics/eval_trajectories/<stamp>_<jobid>/, mirroring
        # Ray's layout — so the banner block is meaningful now.
        eval_trajectories_path=os.path.abspath("ep_metrics/eval_trajectories"),
        seed=trial.seed,
        exec_date=exec_date_dt,
    )

    # Dump a JSON snapshot of the resolved config alongside the model,
    # symmetric with Ray's param_space.json so post-hoc comparisons are
    # easy.
    snapshot = {
        "trial_name": trial.trial_name,
        "algorithm": trial.algorithm,
        "seed": trial.seed,
        "metric": trial.metric,
        "num_envs": trial.num_envs,
        "max_episodes_to_run": trial.training_param_config.max_episodes_to_run,
        "EPISODE_LENGTH": trial.env_config.EPISODE_LENGTH,
        "CONTROL_STEP": trial.env_config.CONTROL_STEP,
        "active_rewards": trial.reward_manager.get_active_reward_names(),
        "infras": [str(i) for i in (trial.env_config.infras or [])],
        "statesources": [str(s) for s in (trial.env_config.statesources or [])],
        "device": device,
        "slurm_cpus": slurm.num_cpus,
        "slurm_gpus": slurm.num_gpus,
    }
    with open(os.path.join(paths.model_dir, "trial_snapshot.json"), "w") as f:
        json.dump(snapshot, f, cls=CustomJSONEncoder, indent=2)

    logger.info("=" * 70)
    logger.info("Starting SB3 model.learn() for: %s", Path(paths.model_dir).name)
    logger.info("=" * 70)

    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        progress_bar=False,  # progress_bar pulls in tqdm; logs are enough
    )
    elapsed_min = (time.time() - t0) / 60.0
    logger.info("=" * 70)
    logger.info("Training finished in %.2f min", elapsed_min)
    logger.info("=" * 70)

    final_path = os.path.join(paths.model_dir, "final_model")
    model.save(final_path)
    logger.info("Final model saved to: %s.zip", final_path)

    summary = {
        "training_time_min": elapsed_min,
        "final_model_path": f"{final_path}.zip",
        "best_dir": paths.best_dir,
        "checkpoint_dir": paths.checkpoint_dir,
        "total_timesteps": total_timesteps,
    }
    with open(os.path.join(paths.model_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    train_vec.close()
    eval_vec.close()
    logger.info("Script completed")


if __name__ == "__main__":
    main()

# Usage examples:
#   python run_train_sb.py --trial configs/trial_cfgs/trial_cfg_1_sac.yaml
#   python run_train_sb.py --trial configs/trial_cfgs/trial_cfg_1_ppo.yaml --cpu
#   sbatch slurm_scripts/slurm_train_sb.sh --trial configs/trial_cfgs/trial_cfg_1_sac.yaml
