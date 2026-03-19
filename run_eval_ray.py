"""run_eval_ray.py – CLI wrapper for evaluating Ray/RLlib trained models.

All logic is delegated to the ``adv_building_gym`` package:
- ``utils.checkpoint_finder.resolve_checkpoint_path`` for checkpoint discovery
- ``evaluation.evaluate_model`` for the evaluation loop
"""

import argparse
import logging
import sys

from adv_building_gym import EnvConfigManager, evaluate_model
from adv_building_gym.config import config as default_config
from adv_building_gym.utils import resolve_checkpoint_path, setup_warning_filters

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
        "--algorithm", "-a", type=str, default="ppo",
        choices=["ppo", "sac"],
        help="RL algorithm to evaluate",
    )
    parser.add_argument(
        "-cn", "--config-name", type=str,
        help="Name of the configuration (used for checkpoint search path)",
    )
    parser.add_argument(
        "--load-config", type=str,
        help="Path to YAML config file to load (e.g., 'configs/my_config.yaml')",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to Ray checkpoint directory. If not provided, searches for best checkpoint.",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
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
        "--log-trajectories", action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-step trajectory JSON per episode (default: True)",
    )
    parser.add_argument(
        "--data-config", type=str, nargs="?", default=None,
        const="configs/eval_data_combinator_config.yaml",
        help="Path to data combinator YAML config. "
            "If given without a path, uses configs/eval_data_combinator_config.yaml.",
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
        help="Plot the best episode's trajectory after evaluation (implies --log-trajectories)",
    )
    parser.add_argument(
        "--plot-all", action="store_true", default=False,
        help="Plot all episodes' trajectories after evaluation (implies --log-trajectories)",
    )
    args = parser.parse_args()

    # --plot / --plot-all require trajectory data; force --log-trajectories on
    if args.plot or args.plot_all:
        if not args.log_trajectories:
            logger.info("--plot/--plot-all implies --log-trajectories; enabling trajectory logging.")
        args.log_trajectories = True

    return args


def main() -> None:
    """Parse arguments and run evaluation."""
    args = parse_args()

    # Load config from file if specified, otherwise use default
    if args.load_config:
        logger.info("Loading config from: %s", args.load_config)
        active_config = EnvConfigManager.load(args.load_config)
        logger.info("Config loaded successfully: %s", active_config.env_config_name)
    else:
        active_config = default_config

    # Initialise singleton component instances in the main process before use.
    active_config.init_singletons()

    config_name = (
        args.config_name
        if args.config_name is not None
        else active_config.env_config_name
    )

    # Build DataCombinator from YAML if specified
    data_combinator = None
    if args.data_config:
        from adv_building_gym.config.data_config import load_data_combinator_config

        data_combinator = load_data_combinator_config(args.data_config, seed_override=args.seed)
        if args.data_mode is not None:
            data_combinator.mode = args.data_mode
        if args.data_day is not None:
            data_combinator.day = args.data_day
        logger.info(
            "DataCombinator: %d variants, mode=%s, day=%s",
            len(data_combinator.variants), data_combinator.mode, data_combinator.day,
        )

    logger.info("Parsed arguments: %s", vars(args))

    # Resolve checkpoint path (auto-discovers if not provided)
    checkpoint_path = resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        config_name=config_name,
        algorithm=args.algorithm,
    )

    try:
        results = evaluate_model(
            checkpoint_path=checkpoint_path,
            active_config=active_config,
            num_episodes=args.episodes,
            seed=args.seed,
            save_results=not args.no_save,
            output_dir=args.output_dir,
            log_trajectories=args.log_trajectories,
            algorithm_hint=args.algorithm,
            timeout_seconds=300,
            data_combinator=data_combinator,
        )
        logger.info("Evaluation completed successfully!")

        # Generate trajectory plots if requested
        if (args.plot or args.plot_all) and not args.no_save:
            import os

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
                logger.warning(
                    "No trajectories.hdf5 found at %s — skipping plots.", hdf5_path,
                )
    except Exception as e:
        logger.error("Evaluation failed: %s", str(e), exc_info=True)
        logger.error("==================")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Usage examples:
# Evaluate best PPO model with default config
# python run_eval_ray.py --algorithm ppo --episodes 10 --seed 42
#
# Evaluate with a custom config file (matching training config)
# python run_eval_ray.py --algorithm ppo --load-config configs/my_config.yaml --episodes 10
#
# Evaluate latest SAC model
# python run_eval_ray.py --algorithm sac --config-name test1 --episodes 10
#
# Evaluate specific checkpoint
# python run_eval_ray.py --checkpoint models/test1/ray/ppo/checkpoints_ppo_seed42_20260106/best_model_ep100_...
#
# Evaluate without saving results
# python run_eval_ray.py --algorithm ppo --episodes 50 --no-save
