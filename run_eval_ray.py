"""run_eval_ray.py – CLI wrapper for evaluating Ray/RLlib trained models.

Single CLI entry point: ``--trial <trial_cfg.yaml>``. The trial config
bundles env topology + reward schedule + (optional) data schedule. Eval
overrides for ``--checkpoint`` and a few presentation flags remain.
"""

import argparse
import logging
import os
import sys

from adv_building_gym import TrialConfig, evaluate_model
from adv_building_gym.utils import resolve_checkpoint_path, RngService, setup_warning_filters

# Apply warning filters
setup_warning_filters()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True,
)
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Ray/RLlib trained models on AdvBuildingGym",
    )
    parser.add_argument(
        "--trial", type=str, required=True,
        help="Path to trial config YAML (e.g. configs/trial_cfgs/trial_cfg_1.yaml). "
             "For eval, the trial's reward_schedule should point at the eval rewards "
             "and data_schedule (if set) at the held-out eval data.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to Ray checkpoint directory. If not provided, searches for the best checkpoint.",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--output-dir", type=str, default="eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save results to file",
    )
    parser.add_argument(
        "--data-mode", type=str, default=None,
        choices=["cycle", "random"],
        help="Override variant selection mode (cycle=round-robin, random)",
    )
    parser.add_argument(
        "--data-day", type=str, default=None,
        help="Override day mode: 'each', 'random', or a date string like '2022-07-15'",
    )
    parser.add_argument(
        "--plot", action="store_true", default=False,
        help="Plot the best episode's trajectory after evaluation (implies trajectory logging)",
    )
    parser.add_argument(
        "--plot-all", action="store_true", default=False,
        help="Plot all episodes' trajectories after evaluation (implies trajectory logging)",
    )
    parser.add_argument(
        "--stochastic", action="store_true", default=False,
        help="Sample actions from the squashed-Gaussian policy instead of "
             "taking tanh(mean). Per-episode torch RNG is seeded from the trial seed.",
    )
    args = parser.parse_args()

    logger.info("CMD: %s", " ".join(sys.argv))
    return args


def main() -> None:
    """Parse arguments, load the trial config, and run evaluation."""
    args = parse_args()

    # Eval may run with or without a data schedule.  For reward_schedule we
    # require it (the trial must declare which rewards to evaluate against).
    trial = TrialConfig.load(args.trial, require_data_schedule=False)

    seed = trial.seed
    RngService.initialize(seed)

    # Push the loaded eval rewards onto the env config (single source of truth).
    trial.env_config.reward_config.rewards = trial.reward_manager.create_active_rewards()
    logger.info(
        "Eval rewards loaded from schedule: %s",
        trial.reward_manager.get_active_reward_names(),
    )

    trial.env_config.init_singletons()

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

    checkpoint_path = resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        config_name=trial.trial_name,
        algorithm=trial.algorithm,
    )

    try:
        results = evaluate_model(
            checkpoint_path=checkpoint_path,
            active_config=trial.env_config,
            trial_name=trial.trial_name,
            num_episodes=args.episodes,
            seed=seed,
            save_results=not args.no_save,
            output_dir=args.output_dir,
            log_trajectories=log_trajectories,
            algorithm_hint=trial.algorithm,
            timeout_seconds=300,
            data_combinator=data_combinator,
            stochastic=args.stochastic,
        )
        logger.info("Evaluation completed successfully!")

        # Generate trajectory plots if requested
        if (args.plot or args.plot_all) and not args.no_save:
            import h5py

            actual_output_dir = results.output_dir or args.output_dir
            hdf5_path = os.path.join(actual_output_dir, "trajectories.hdf5")
            if os.path.isfile(hdf5_path):
                from plotting.traj_plotting.trajectory_plot import generate_all_plots

                plot_dir = os.path.join(actual_output_dir, "plots")

                if args.plot_all:
                    with h5py.File(hdf5_path, "r") as hf:
                        episode_ids = list(hf.keys())
                    total_paths: list[str] = []
                    for ep_id in episode_ids:
                        ep_label = f"ep_{ep_id}"
                        ep_plot_dir = os.path.join(plot_dir, ep_label)
                        paths = generate_all_plots(
                            hdf5_path=hdf5_path,
                            episode_id=ep_id,
                            output_dir=ep_plot_dir,
                            file_prefix=ep_label,
                        )
                        total_paths.extend(paths)
                    logger.info(
                        "Generated %d plot files for %d episodes in %s",
                        len(total_paths), len(episode_ids), plot_dir,
                    )
                else:
                    paths = generate_all_plots(
                        hdf5_path=hdf5_path,
                        output_dir=plot_dir,
                        file_prefix="ep_best",
                    )
                    logger.info("Generated %d plot files in %s", len(paths), plot_dir)
            else:
                logger.warning("No trajectories.hdf5 found at %s — skipping plots.", hdf5_path)
    except Exception as e:
        logger.error("Evaluation failed: %s", str(e), exc_info=True)
        logger.error("==================")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Usage:
# python run_eval_ray.py --trial configs/trial_cfgs/trial_cfg_1.yaml --episodes 10
# python run_eval_ray.py --trial configs/trial_cfgs/trial_cfg_1.yaml --checkpoint <path>
