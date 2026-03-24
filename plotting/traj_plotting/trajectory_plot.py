"""Orchestrate trajectory plotting from HDF5 evaluation files.

Generates four separate figures (states, actions, rewards, energy) for the
episode with the best achieved reward or a user-specified episode ID.

Usage:
    python -m plotting.traj_plotting.trajectory_plot [--hdf5 <path>] [--episode <id>]

When ``--hdf5`` is omitted the script auto-discovers the latest
``trajectories.hdf5`` under ``ep_metrics/trajectories/``.
"""

# TODO VP 2026.03.16. : Just as an idea -- Reward the reward smoothness?
# That would prevent immediate full charging actions, as if the EV is connected, it's a huge jump in rewards if charge is possible as well.

from __future__ import annotations

import argparse
import logging
import os

import plotly.graph_objects as go

from plotting.utils import (
    ensure_chrome_for_kaleido,
    find_latest_hdf5,
    get_output_root,
    load_episode,
    write_figure_list_html,
)
from .plot_states import plot_states
from .plot_actions import plot_actions
from .plot_rewards import plot_rewards
from .plot_energy import ENERGY_SIGN_CONVENTION_HTML, plot_energy
from .plot_raw import plot_raw

logger = logging.getLogger("trajectory_plot")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_all_plots(
    hdf5_path: str,
    episode_id: str | None = None,
    output_dir: str | None = None,
    control_step_seconds: int = 300,
    formats: list[str] = ["html"],
    select_by: str = "reward_rate",
    file_prefix: str | None = None,
) -> list[str]:
    """Load an episode, generate all four figures, save to output_dir.

    Args:
        hdf5_path: Path to trajectories.hdf5.
        episode_id: Episode to plot. None = best by ``select_by`` metric.
        output_dir: Output directory. Default: plotting/out/{episode_id}/.
        control_step_seconds: Timestep in seconds (default 300).
        formats: Output formats to produce. Default: ["html"].
        select_by: Summary metric for auto-selecting the best episode.
        file_prefix: Prefix for output filenames. Default: episode_id.

    Returns:
        List of saved file paths.
    """

    episode = load_episode(hdf5_path, episode_id, control_step_seconds, select_by)
    ep_id = file_prefix if file_prefix is not None else episode.episode_id

    if output_dir is None:
        output_dir = str(get_output_root() / ep_id)
    os.makedirs(output_dir, exist_ok=True)

    all_figures: dict[str, list[go.Figure]] = {
        "states": plot_states(episode),
        "actions": plot_actions(episode),
        "rewards": plot_rewards(episode),
        "energy": plot_energy(episode),
        "raw": plot_raw(episode),
    }

    # Per-figure-group footnotes rendered as separate HTML divs below the plots
    html_footnotes: dict[str, str] = {
        "energy": ENERGY_SIGN_CONVENTION_HTML,
    }

    saved: list[str] = []

    # Ensure Chrome/Chromium is available for static image export (svg/png/pdf).
    # Kaleido v1+ requires Chrome; this downloads it once to ~/.cache if missing.
    needs_static = any(f != "html" for f in formats)
    if needs_static:
        ensure_chrome_for_kaleido()

    for fmt in formats:
        # Static image formats go into a subdirectory (e.g. <ep_id>/svgs/)
        if fmt == "html":
            fmt_dir = output_dir
        else:
            fmt_dir = os.path.join(output_dir, f"{fmt}s")
            os.makedirs(fmt_dir, exist_ok=True)

        for name, figs in all_figures.items():
            if fmt == "html":
                filepath = os.path.join(fmt_dir, f"{ep_id}_{name}.html")
                footnote = html_footnotes.get(name, "")
                write_figure_list_html(figs, filepath, footnote=footnote)
                logger.info("Saved: %s", filepath)
                saved.append(filepath)
            else:
                for i, fig in enumerate(figs):
                    suffix = f"_{i}" if len(figs) > 1 else ""
                    filepath = os.path.join(
                        fmt_dir, f"{ep_id}_{name}{suffix}.{fmt}",
                    )
                    fig.write_image(filepath, width=1600, height=400)
                    logger.info("Saved: %s", filepath)
                    saved.append(filepath)

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for trajectory plotting."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Plot trajectory data from HDF5 evaluation files.",
    )
    parser.add_argument(
        "--hdf5", type=str, default=None,
        help="Path to trajectories.hdf5 file. Default: latest in ep_metrics/trajectories/.",
    )
    parser.add_argument(
        "--episode", type=str, default=None,
        help="Episode ID to plot. Default: best by --select-by metric.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory. Default: plotting/out/<episode_id>/.",
    )
    parser.add_argument(
        "--format", type=str, nargs="+", default=["html", "svg"],
        choices=["html", "png", "svg", "pdf"],
        help="Output format(s). Default: html svg.",
    )
    parser.add_argument(
        "--control-step", type=int, default=300,
        help="Control timestep in seconds. Default: 300 (5 min).",
    )
    parser.add_argument(
        "--select-by", type=str, default="reward_rate",
        choices=["reward_rate", "achieved_reward", "cum_E_kWh"],
        help="Summary metric for selecting the best episode. Default: reward_rate.",
    )
    args = parser.parse_args()

    hdf5_path = args.hdf5 if args.hdf5 else find_latest_hdf5()

    paths = generate_all_plots(
        hdf5_path=hdf5_path,
        episode_id=args.episode,
        output_dir=args.output_dir,
        control_step_seconds=args.control_step,
        formats=args.format,
        select_by=args.select_by,
    )

    for p in paths:
        print(f"Saved: {p}")


# TODO VP 2026.03.12. : Use float64 everywhere -- for training, for actions, etc... -- more precision is key

if __name__ == "__main__":
    main()