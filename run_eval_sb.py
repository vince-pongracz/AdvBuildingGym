"""run_eval_sb.py – CLI wrapper for evaluating Stable-Baselines3 trained models.

Mirrors ``run_eval_ray.py``. 
Single CLI entry point: ``--trial <trial_cfg.yaml>``. The trial config bundles env
topology + reward schedule + (optional) data schedule; eval overrides for
``--checkpoint`` and a few presentation flags remain.

When the trial declares ``infra_schedule`` or ``statesource_schedule`` with
multiple ``configs.eval`` entries, this script iterates them: each entry is
evaluated for ``--episodes`` episodes, with results landing in its own
per-config subdirectory. Mutex guard: only one of the two axes can have
multi-entry ``configs.eval`` at a time (enforced by ``TrialConfig`` at load).
"""

import datetime
import logging
import sys

from adv_building_gym.config.trial_config import TrialConfig
from adv_building_gym.sb.evaluation import evaluate_sb_model, resolve_sb_checkpoint_path
from adv_building_gym._common.warning_filters import setup_warning_filters

from run_eval_util import parse_eval_args, generate_trajectory_plots

# Apply warning filters
setup_warning_filters()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True,
)
logger = logging.getLogger("main")


def main() -> None:
    """Parse arguments, load the trial config, and run evaluation."""
    args = parse_eval_args(
        description="Evaluate Stable-Baselines3 trained models on AdvBuildingGym",
        checkpoint_help=(
            "Path to an SB3 model .zip (extension optional), or 'best'/'latest'. "
            "If omitted, searches models/<trial>/sb3/<algo> for the best model."
        ),
        stochastic_help=(
            "Sample actions from the policy instead of taking the deterministic "
            "action. Per-episode SB3 RNG is reseeded from the trial seed."
        ),
        logger=logger,
    )

    # Eval may run with or without a data schedule.  For reward_schedule we
    # require it (the trial must declare which rewards to evaluate against).
    trial = TrialConfig.load(args.trial, require_data_schedule=False, is_training=False)

    if trial.algorithm == "dreamerv3":
        logger.error("SB3 has no DreamerV3 — use run_eval_ray.py to evaluate '%s'.", trial.trial_name)
        sys.exit(1)

    seed = trial.seed

    # Push the eval-mode rewards onto the env config (single source of truth):
    # OFF/FIX -> as-is, RANDOM/GRAD_ADD/DIRICHLET -> full pool with original weights.
    trial.env_config.reward_config.rewards = trial.reward_manager.create_eval_rewards()
    logger.info(
        "Eval rewards: %s",
        [type(r).__name__ for r in trial.env_config.reward_config.rewards],
    )

    data_combinator = trial.data_combinator
    if data_combinator is not None:
        if args.data_mode is not None:
            data_combinator.mode = args.data_mode
        if args.data_day is not None:
            data_combinator.day = args.data_day
        logger.info(
            "DataCombinator: %d variants, mode=%s, day=%s",
            len(data_combinator.variants), data_combinator.mode, data_combinator.day,
        )

    log_trajectories = trial.log_trajectories or args.plot or args.plot_all

    logger.info("Trial run params: %s", {
        "trial_name": trial.trial_name,
        "algorithm": trial.algorithm,
        "seed": seed,
        "episodes_cli": args.episodes,
        "log_trajectories": log_trajectories,
    })

    checkpoint_path = resolve_sb_checkpoint_path(
        checkpoint=args.checkpoint,
        trial_name=trial.trial_name,
        algorithm=trial.algorithm,
    )

    # ---- decide eval iteration axis ----
    # Only one axis can iterate (TrialConfig already enforced this). When
    # neither schedule has >1 eval entry, we run a single pass with whatever
    # infras/statesources the trial seeded onto env_config.
    infra_eval_n = trial.infra_combinator.eval_count() if trial.infra_combinator else 0
    ss_eval_n = trial.statesource_combinator.eval_count() if trial.statesource_combinator else 0

    iteration_axis: str | None = None
    n_iters = 1
    if infra_eval_n > 1:
        iteration_axis = "infra"
        n_iters = infra_eval_n
    elif ss_eval_n > 1:
        iteration_axis = "statesource"
        n_iters = ss_eval_n
    elif trial.infra_combinator is not None:
        # Single eval entry: still source it from the combinator so eval uses
        # the eval list rather than the train-seeded singleton.
        iteration_axis = "infra"
        n_iters = 1
    elif trial.statesource_combinator is not None:
        iteration_axis = "statesource"
        n_iters = 1

    # Shared timestamp so per-config dirs sit under one parent run dir.
    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_eval"
    if args.stochastic:
        run_stamp += "_stoch"

    try:
        all_results = []
        for idx in range(n_iters):
            subdir: str | None = None
            if iteration_axis == "infra":
                cfg_name = trial.infra_combinator.get_eval_config_name(idx)
                subdir = cfg_name if n_iters > 1 else None
                trial.env_config.infras = trial.infra_combinator.create_infras(idx, split="eval")
                logger.info(
                    "[eval %d/%d] infra config: %s", idx + 1, n_iters, cfg_name,
                )
            elif iteration_axis == "statesource":
                cfg_name = trial.statesource_combinator.get_eval_config_name(idx)
                subdir = cfg_name if n_iters > 1 else None
                trial.env_config.statesources = trial.statesource_combinator.create_statesources(idx, split="eval")
                logger.info(
                    "[eval %d/%d] statesource config: %s", idx + 1, n_iters, cfg_name,
                )

            # Materialise any remaining singletons (the inline-list case, or
            # the axis we did NOT swap this iteration).
            trial.env_config.init_singletons()

            results = evaluate_sb_model(
                checkpoint_path=checkpoint_path,
                active_config=trial.env_config,
                trial_name=trial.trial_name,
                algorithm=trial.algorithm,
                num_episodes=args.episodes,
                seed=seed,
                save_results=not args.no_save,
                output_dir=args.output_dir,
                log_trajectories=log_trajectories,
                timeout_seconds=300,
                data_combinator=data_combinator,
                stochastic=args.stochastic,
                run_stamp=run_stamp,
                subdir=subdir,
                trial_yaml_path=trial.source_path,
            )
            all_results.append((subdir, results))
            logger.info("Evaluation pass %d/%d completed.", idx + 1, n_iters)

            generate_trajectory_plots(results, args, logger)

        if n_iters > 1:
            logger.info("=" * 70)
            logger.info("Per-config eval summary (mean reward):")
            for subdir, r in all_results:
                logger.info(
                    "  %-40s  mean_reward=%.4f",
                    subdir, r.mean_reward,
                )
    except Exception as e:
        logger.error("Evaluation failed: %s", str(e), exc_info=True)
        logger.error("==================")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Usage:
# python run_eval_sb.py --trial configs/trial_cfgs/trial_cfg_1_sac.yaml --episodes 10
# python run_eval_sb.py --trial configs/trial_cfgs/trial_cfg_1_sac.yaml --checkpoint <path>.zip
