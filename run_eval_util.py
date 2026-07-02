"""Shared CLI helpers for the evaluation drivers.

``run_eval_ray.py`` and ``run_eval_sb.py`` expose the same eval CLI surface
(``--trial`` plus presentation flags) and the same post-run trajectory-plot
rendering. Only three strings differ between the two — the argparse
``description`` and the ``--checkpoint`` / ``--stochastic`` help wording — so
those are injected by the caller and the parser + plot renderer live here once
instead of being duplicated per driver.

Mirrors ``run_train_util.py`` for the training drivers.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def parse_eval_args(
    *,
    description: str,
    checkpoint_help: str,
    stochastic_help: str,
    logger: logging.Logger,
) -> argparse.Namespace:
    """Parse the shared eval CLI; framework-specific help text is injected.

    The Ray and SB3 eval drivers differ only in ``description`` and the
    ``--checkpoint`` / ``--stochastic`` help wording; every flag and default is
    identical, so the parser is defined once here. ``logger`` is used only to
    echo the invocation (``CMD: ...``) so the line keeps the driver's logger name.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--trial", type=str, required=True,
        help="Path to trial config YAML (e.g. configs/trial_cfgs/<name>.yaml). "
            "For eval, the trial's reward_schedule should point at the eval rewards "
            "and data_schedule (if set) at the held-out eval data.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help=checkpoint_help,
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
        "--no-save", action="store_true", help="Don't save results to file",
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
        "--plot-all", action="store_true", default=True,
        help="Plot all episodes' trajectories after evaluation (implies trajectory logging)",
    )
    parser.add_argument(
        "--stochastic", action="store_true", default=False, help=stochastic_help,
    )
    args = parser.parse_args()

    logger.info("CMD: %s", " ".join(sys.argv))
    return args


def generate_trajectory_plots(results, args: argparse.Namespace, logger: logging.Logger) -> None:
    """Render trajectory plots for a just-completed eval run (Ray or SB).

    Reads ``<results.output_dir>/trajectories.hdf5`` and writes per-episode
    plots (plus a shared dashboard for ``--plot-all``). No-op when neither
    ``--plot`` nor ``--plot-all`` is set, or ``--no-save`` suppressed the HDF5.
    """
    if not (args.plot or args.plot_all) or args.no_save:
        return
    import h5py

    actual_output_dir = results.output_dir or args.output_dir
    hdf5_path = os.path.join(actual_output_dir, "trajectories.hdf5")
    if not os.path.isfile(hdf5_path):
        logger.warning("No trajectories.hdf5 found at %s — skipping plots.", hdf5_path)
        return

    from plotting.traj_plotting.trajectory_plot import generate_all_plots

    plot_dir = os.path.join(actual_output_dir, "plots")

    if args.plot_all:
        with h5py.File(hdf5_path, "r") as hf:
            episode_ids = list(hf.keys())
        dashboard_dir = os.path.join(plot_dir, "dashboards")
        total_paths: list[str] = []
        for ep_id in episode_ids:
            ep_label = f"ep_{ep_id}"
            ep_plot_dir = os.path.join(plot_dir, ep_label)
            paths = generate_all_plots(
                hdf5_path=hdf5_path,
                episode_id=ep_id,
                output_dir=ep_plot_dir,
                file_prefix=ep_label,
                dashboard_dir=dashboard_dir,
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
