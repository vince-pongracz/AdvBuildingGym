"""run_eval_ray.py – CLI wrapper for evaluating Ray/RLlib trained models.

All logic is delegated to the ``adv_building_gym`` package:
- ``utils.checkpoint_finder.resolve_checkpoint_path`` for checkpoint discovery
- ``evaluation.evaluate_model`` for the evaluation loop
"""

import argparse
import logging
import sys

from adv_building_gym import ConfigManager, evaluate_model
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
        help="Path to JSON config file to load (e.g., 'configs/my_config.json')",
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
    # TODO VP 2026.03.11. : Check this out again -- run it
    parser.add_argument(
        "--data-config", type=str, default=None,
        help="Path to data combinator YAML config (e.g. configs/eval_data_combinator_config.yaml)",
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
    return parser.parse_args()


def main() -> None:
    """Parse arguments and run evaluation."""
    args = parse_args()

    # Load config from file if specified, otherwise use default
    if args.load_config:
        logger.info("Loading config from: %s", args.load_config)
        active_config = ConfigManager.load(args.load_config)
        logger.info("Config loaded successfully: %s", active_config.config_name)
    else:
        active_config = default_config

    # Initialise singleton component instances in the main process before use.
    active_config.init_singletons()

    config_name = (
        args.config_name
        if args.config_name is not None
        else active_config.config_name
    )

    # Build DataCombinator from YAML if specified
    data_combinator = None
    if args.data_config:
        from adv_building_gym.config.data_config import load_data_combinator

        data_combinator = load_data_combinator(args.data_config, seed_override=args.seed)
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
        evaluate_model(
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
    except Exception as e:
        logger.error("Evaluation failed: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

# Usage examples:
# Evaluate best PPO model with default config
# python run_eval_ray.py --algorithm ppo --episodes 10 --seed 42
#
# Evaluate with a custom config file (matching training config)
# python run_eval_ray.py --algorithm ppo --load-config configs/my_config.json --episodes 10
#
# Evaluate latest SAC model
# python run_eval_ray.py --algorithm sac --config-name test1 --episodes 10
#
# Evaluate specific checkpoint
# python run_eval_ray.py --checkpoint models/test1/ray/ppo/checkpoints_ppo_seed42_20260106/best_model_ep100_...
#
# Evaluate without saving results
# python run_eval_ray.py --algorithm ppo --episodes 50 --no-save
