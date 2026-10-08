"""Orchestrate trajectory plotting from HDF5 evaluation files.

Generates four separate figures (states, actions, rewards, energy) for the
episode with the best achieved reward or a user-specified episode ID.

Usage:
    python -m plotting.traj_plotting.trajectory_plot [--hdf5 <path>] [--episode <id>]

When ``--hdf5`` is omitted the script auto-discovers the latest
``trajectories.hdf5`` under ``ep_metrics/trajectories/``.
"""

# That would prevent immediate full charging actions, as if the EV is connected, it's a huge jump in rewards if charge is possible as well.

from __future__ import annotations

import argparse
import logging
import os

import h5py
import plotly.graph_objects as go

from plotting.dashboard.build_dashboard import write_episode_dashboard_html
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
from .plot_price import PRICE_SIGN_CONVENTION_HTML, plot_price
from .plot_raw import plot_raw
from .plot_raw_policy_actions import plot_raw_policy_actions

logger = logging.getLogger("trajectory_plot")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_all_plots(
    hdf5_path: str,
    episode_id: str | None = None,
    output_dir: str | None = None,
    control_step_s: int = 300,
    formats: list[str] = ["html"],
    select_by: str = "achieved_reward",
    file_prefix: str | None = None,
    manifest_path: str | None = None,
    config_path: str | None = None,
    dashboard_dir: str | None = None,
    episode_plots: bool = True,
) -> list[str]:
    """Load an episode, generate all figure groups, save to output_dir.

    Args:
        hdf5_path: Path to trajectories.hdf5.
        episode_id: Episode to plot. None = best by ``select_by`` metric.
        output_dir: Output directory. Default: plotting/out/{episode_id}/.
        control_step_s: Control step duration in seconds (default 300).
        formats: Output formats to produce. Default: ["html"].
        select_by: Summary metric for auto-selecting the best episode.
        file_prefix: Prefix for output filenames. Default: episode_id.
        manifest_path: Snapshot manifest.json for the dashboard's right panel.
            None = auto-discover by walking up from ``output_dir``.
        config_path: Trial-config YAML for the dashboard's right panel.
            None = auto-discover from the eval-results dir.
        dashboard_dir: Directory for the aggregated dashboard HTML. None = write
            it inside ``output_dir`` (default). Multi-episode callers point this
            at a shared ``dashboards/`` dir sitting beside the ``ep_*`` dirs.
        episode_plots: Whether to also write the per-group figure files (states,
            actions, rewards, ...) into output_dir. The dashboard already embeds
            every figure, so callers that only need the dashboard can pass False
            to skip these redundant per-episode files entirely.

    Returns:
        List of saved file paths.
    """

    episode = load_episode(hdf5_path, episode_id, control_step_s, select_by)
    ep_id = file_prefix if file_prefix is not None else f"ep_{episode.episode_id}"

    if output_dir is None:
        output_dir = str(get_output_root() / ep_id)
    if episode_plots:
        os.makedirs(output_dir, exist_ok=True)

    all_figures: dict[str, list[go.Figure]] = {
        "states": plot_states(episode),
        "actions": plot_actions(episode),
        "raw_policy_actions": plot_raw_policy_actions(episode),
        "rewards": plot_rewards(episode),
        "energy": plot_energy(episode),
        "price": plot_price(episode),
        "raw": plot_raw(episode),
    }

    # Per-figure-group footnotes rendered as separate HTML divs below the plots
    html_footnotes: dict[str, str] = {
        "energy": ENERGY_SIGN_CONVENTION_HTML,
        "price": PRICE_SIGN_CONVENTION_HTML,
    }

    saved: list[str] = []

    # Aggregated self-contained dashboard (HTML only). Built BEFORE the per-group
    # writers run because write_figure_list_html mutates each figure's layout
    # size in place — serialising here captures the figures with their heights.
    if "html" in formats:
        dashboard_out = dashboard_dir if dashboard_dir is not None else output_dir
        os.makedirs(dashboard_out, exist_ok=True)
        dashboard_path = os.path.join(dashboard_out, f"{ep_id}_dashboard.html")
        write_episode_dashboard_html(
            all_figures, html_footnotes, episode, dashboard_path,
            output_dir=output_dir,
            manifest_path=manifest_path,
            config_path=config_path,
        )
        saved.append(dashboard_path)

    if not episode_plots:
        return saved

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
                    # Width/height come from each figure's layout (set via
                    # style_figure from traj_plot_config.yaml); keep static export
                    # consistent with the HTML render.
                    fig.write_image(filepath)
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
        "--all-episodes", action="store_true",
        help="Plot every episode in the HDF5 file (one subdir per episode). "
            "Mutually exclusive with --episode.",
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
        "--control-step", dest="control_step_s", type=int, default=300,
        help="Control step duration in seconds. Default: 300 (5 min).",
    )
    parser.add_argument(
        "--select-by", type=str, default="achieved_reward",
        choices=["achieved_reward", "cum_E_kWh"],
        help="Summary metric for selecting the best episode. Default: achieved_reward.",
    )
    args = parser.parse_args()

    if args.all_episodes and args.episode is not None:
        parser.error("--all-episodes and --episode are mutually exclusive.")

    hdf5_path = args.hdf5 if args.hdf5 else find_latest_hdf5()

    if args.all_episodes:
        with h5py.File(hdf5_path, "r") as hf:
            episode_ids = list(hf.keys())
        base_output_dir = args.output_dir or str(get_output_root())
        dashboard_dir = os.path.join(base_output_dir, "dashboards")
        paths: list[str] = []
        for ep_id in episode_ids:
            ep_label = f"ep_{ep_id}"
            ep_output_dir = os.path.join(base_output_dir, ep_label)
            paths.extend(generate_all_plots(
                hdf5_path=hdf5_path,
                episode_id=ep_id,
                output_dir=ep_output_dir,
                control_step_s=args.control_step_s,
                formats=args.format,
                select_by=args.select_by,
                file_prefix=ep_label,
                dashboard_dir=dashboard_dir,
            ))
        logger.info(
            "Generated %d plot files for %d episodes under %s",
            len(paths), len(episode_ids), base_output_dir,
        )
    else:
        paths = generate_all_plots(
            hdf5_path=hdf5_path,
            episode_id=args.episode,
            output_dir=args.output_dir,
            control_step_s=args.control_step_s,
            formats=args.format,
            select_by=args.select_by,
        )

    for p in paths:
        print(f"Saved: {p}")


# TODO noprio VP 2026.03.12. : Use float64 everywhere -- for training, for actions, etc... -- more precision is key

if __name__ == "__main__":
    main()