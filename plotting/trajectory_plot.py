"""Plot single-episode trajectory data from HDF5 evaluation files.

Generates four separate figures (states, actions, rewards, energy) for the
episode with the best achieved reward or a user-specified episode ID.

Usage:
    python -m plotting.trajectory_plot [--hdf5 <path>] [--episode <id>]

When ``--hdf5`` is omitted the script auto-discovers the latest
``trajectories.hdf5`` under ``ep_metrics/trajectories/``.
"""

import argparse
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Default output root relative to the repository
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "plotting" / "out"
_DEFAULT_METRICS_ROOT = _REPO_ROOT / "ep_metrics" / "trajectories"

# TODO VP 2026.03.10. : Split this up, refactor, relocate action, reward, state, energy plottings into standalone files, add plotting util for find_latest_hdf5

# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

def find_latest_hdf5(metrics_root: Path | None = None) -> str:
    """Return the path to the most recent ``trajectories.hdf5``.

    Scans ``metrics_root`` (default: ``ep_metrics/trajectories/``) for
    timestamped subdirectories (``YYYYMMDD_HHMMSS``) and returns the
    ``trajectories.hdf5`` inside the latest one.

    Raises:
        FileNotFoundError: If no trajectories.hdf5 is found.
    """
    root = metrics_root or _DEFAULT_METRICS_ROOT
    if not root.is_dir():
        raise FileNotFoundError(
            f"Metrics root directory does not exist: {root}"
        )

    # Timestamp directory names sort lexicographically → latest is last
    candidates = sorted(
        p for p in root.iterdir()
        if p.is_dir() and (p / "trajectories.hdf5").exists()
    )

    if not candidates:
        raise FileNotFoundError(
            f"No trajectories.hdf5 found under {root}"
        )

    latest = candidates[-1] / "trajectories.hdf5"
    logger.info("Auto-discovered latest HDF5: %s", latest)
    return str(latest)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_episode(
    hdf5_path: str,
    episode_id: str | None = None,
    control_step_seconds: int = 300,
    select_by: str = "reward_rate",
) -> dict:
    """Load one episode from an HDF5 trajectory file.

    Args:
        hdf5_path: Path to trajectories.hdf5.
        episode_id: Episode group name. If None, selects the best episode
            according to ``select_by``.
        control_step_seconds: Control timestep in seconds (default 300 = 5 min).
        select_by: Summary metric used to pick the best episode when
            ``episode_id`` is None. One of "reward_rate", "achieved_reward",
            "cum_E_kWh". Default: "reward_rate".

    Returns:
        Dict with keys: meta, summary, time_minutes, states, actions,
        rewards, reward_breakdown, step_power_kW, cum_E_kWh.
    """
    # cum_E_kWh: lower is better; all others: higher is better
    _minimize_metrics = {"cum_E_kWh"}

    with h5py.File(hdf5_path, "r") as f:
        if episode_id is None:
            pick_fn = min if select_by in _minimize_metrics else max
            episode_id = pick_fn(
                f.keys(),
                key=lambda eid: float(f[eid]["summary"].attrs.get(select_by, 0.0)),
            )
            best_val = float(f[episode_id]["summary"].attrs.get(select_by, 0.0))
            logger.info(
                "Auto-selected episode %s (best %s=%.3f)",
                episode_id, select_by, best_val,
            )

        if episode_id not in f:
            available = list(f.keys())
            raise KeyError(
                f"Episode '{episode_id}' not found. Available: {available}"
            )

        ep = f[episode_id]

        meta = {
            "episode_id": episode_id,
            "seed": int(ep.attrs.get("seed", 0)),
            "length": int(ep.attrs.get("length", 0)),
        }

        summary = {k: float(v) for k, v in ep["summary"].attrs.items()}

        traj = ep["trajectory"]
        steps = traj["step"][:].astype(np.float32)
        time_minutes = steps * (control_step_seconds / 60.0)

        # States
        states: dict[str, np.ndarray] = {}
        if "state" in traj:
            for key in traj["state"]:
                states[key] = traj["state"][key][:]

        # Actions
        actions: dict[str, np.ndarray] = {}
        if "action" in traj:
            for key in traj["action"]:
                actions[key] = traj["action"][key][:]

        # Rewards
        rewards = traj["reward"][:] if "reward" in traj else np.zeros_like(steps)

        # Reward breakdown
        reward_breakdown: dict[str, np.ndarray] = {}
        if "reward_breakdown" in traj:
            for key in traj["reward_breakdown"]:
                reward_breakdown[key] = traj["reward_breakdown"][key][:]

        # Energy
        cum_e = traj["cum_E_kWh"][:] if "cum_E_kWh" in traj else np.zeros_like(steps)
        power = (
            traj["step_power_kW"][:] if "step_power_kW" in traj
            else np.zeros_like(steps)
        )

    return {
        "meta": meta,
        "summary": summary,
        "time_minutes": time_minutes,
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "reward_breakdown": reward_breakdown,
        "step_power_kW": power,
        "cum_E_kWh": cum_e,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


def _title_suffix(episode: dict) -> str:
    """Return a short suffix with episode metadata for figure titles."""
    meta = episode["meta"]
    summary = episode["summary"]
    return (
        f"ep {meta['episode_id']}  |  "
        f"reward_rate {summary.get('reward_rate', 0):.3f}"
    )


def _apply_day_xaxis(
    fig: go.Figure,
    n_rows: int | None = None,
) -> None:
    """Configure x-axis as a 24-hour day with ticks every 5 minutes.

    Args:
        fig: Plotly figure.
        n_rows: If the figure uses make_subplots, pass the total row count
            so the label is placed on the bottom subplot. None for a plain
            figure without subplots.
    """
    # 24 h = 1440 min; ticks every 5 min
    tick_vals = list(range(0, 1441, 5))
    # Show HH:MM labels every 60 min, empty string for intermediate ticks
    tick_text = [
        f"{m // 60:02d}:{m % 60:02d}" if m % 60 == 0 else ""
        for m in tick_vals
    ]
    base_kwargs: dict = dict(
        range=[0, 1440],
        tickvals=tick_vals,
        ticktext=tick_text,
        tickangle=0,
    )
    if n_rows is not None:
        for row in range(1, n_rows + 1):
            fig.update_xaxes(
                **base_kwargs,
                title_text="Time (HH:MM)" if row == n_rows else None,
                row=row, col=1,
            )
    else:
        fig.update_xaxes(**base_kwargs, title_text="Time (HH:MM)")


def _style_figure(fig: go.Figure) -> go.Figure:
    """Apply consistent styling."""
    fig.update_layout(
        template="plotly_white",
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig


def _write_figure_list_html(figures: list[go.Figure], filepath: str) -> None:
    """Write a list of independent figures into a single HTML file."""
    parts: list[str] = [
        "<html><head><meta charset='utf-8'/></head><body>",
    ]
    for fig in figures:
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    parts.append("</body></html>")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


# ---------------------------------------------------------------------------
# Figure: States
# ---------------------------------------------------------------------------

def plot_states(episode: dict) -> list[go.Figure]:
    """One independent plot per state variable (or group of variables).

    Keys listed in ``_grouped_keys`` are merged into a single plot.
    Returns a list of figures to be rendered sequentially in one HTML file.
    """
    states = episode["states"]
    time = episode["time_minutes"]
    suffix = _title_suffix(episode)

    # State keys to exclude from plotting (constants or non-informative)
    _skip_keys = {"E_price_max"}

    # Groups of keys that share one plot
    _grouped_keys: list[list[str]] = [
        ["battery_pct", "battery_target_pct"],
        ["temp_in_norm", "desired_temp_in_norm"],
        ["ev_schedule_charger_eff", "ev_schedule_discharge_eff"],
        ["ev_schedule_start_soc", "ev_schedule_target_soc"],
    ]

    # Build ordered list of plot specs.  Each entry is a list of keys.
    grouped_flat = {k for group in _grouped_keys for k in group}
    plot_specs: list[list[str]] = []

    seen_groups: set[int] = set()
    for key, val in states.items():
        if key in _skip_keys or val.ndim != 1:
            continue
        if key in grouped_flat:
            for gi, group in enumerate(_grouped_keys):
                if key in group and gi not in seen_groups:
                    present = [k for k in group if k in states]
                    missing = [k for k in group if k not in states]
                    if missing:
                        logger.info(
                            "Grouped key(s) %s missing from data, skipping.",
                            missing,
                        )
                    if present:
                        plot_specs.append(present)
                        seen_groups.add(gi)
        else:
            plot_specs.append([key])

    # Multi-dim keys (2-D with few columns)
    for key, val in states.items():
        if key in _skip_keys or key in grouped_flat:
            continue
        if val.ndim == 2 and val.shape[1] <= 4:
            plot_specs.append([key])

    if not plot_specs:
        logger.warning("No plottable state variables found.")
        return []

    time_hhmm = [
        f"{int(m) // 60:02d}:{int(m) % 60:02d}" for m in time
    ]

    figures: list[go.Figure] = []
    trace_idx = 0
    for keys in plot_specs:
        fig = go.Figure()
        title = " + ".join(keys)
        all_data: list[np.ndarray] = []

        for key in keys:
            if key not in states:
                logger.info("State key '%s' missing from data, skipping.", key)
                continue
            arr = states[key]
            if arr.ndim == 1:
                fig.add_trace(go.Scatter(
                    x=time, y=arr, mode="lines",
                    name=key,
                    line=dict(color=_COLORS[trace_idx % len(_COLORS)]),
                    customdata=time_hhmm,
                    hovertemplate=(
                        f"{key}<br>"
                        "time: %{customdata}<br>"
                        "value: %{y:.4f}"
                        "<extra></extra>"
                    ),
                ))
                all_data.append(arr)
                trace_idx += 1
            elif arr.ndim == 2:
                for col_idx in range(arr.shape[1]):
                    label = f"{key}[{col_idx}]"
                    fig.add_trace(go.Scatter(
                        x=time, y=arr[:, col_idx], mode="lines",
                        name=label,
                        line=dict(
                            color=_COLORS[trace_idx % len(_COLORS)],
                        ),
                        customdata=time_hhmm,
                        hovertemplate=(
                            f"{label}<br>"
                            "time: %{customdata}<br>"
                            "value: %{y:.4f}"
                            "<extra></extra>"
                        ),
                    ))
                    trace_idx += 1
                all_data.append(arr)

        if all_data:
            combined = np.concatenate([d.ravel() for d in all_data])
            y_lo = -1.0 if float(np.min(combined)) < 0 else 0.0
            fig.update_yaxes(
                range=[min(y_lo, float(np.min(combined))),
                       max(1.0, float(np.max(combined)))],
            )

        _apply_day_xaxis(fig)
        fig.update_layout(title=f"{title}  —  {suffix}", height=350)
        figures.append(_style_figure(fig))

    return figures


# ---------------------------------------------------------------------------
# Figure: Actions
# ---------------------------------------------------------------------------

def plot_actions(episode: dict) -> list[go.Figure]:
    """One independent plot per action dimension over a 24-hour day.

    Multi-dimensional actions (e.g. HP_action with energy + mode) are
    expanded so each dimension gets its own plot.
    Returns a list of figures to be rendered sequentially in one HTML file.
    """
    actions = episode["actions"]
    time = episode["time_minutes"]
    suffix = _title_suffix(episode)

    if not actions:
        logger.warning("No action variables found.")
        return []

    # Dimension labels for multi-dim actions
    _dim_labels: dict[str, list[str]] = {
        "HP_action": ["energy", "mode"],
    }

    # Build flat list of (label, 1-D data)
    traces: list[tuple[str, np.ndarray]] = []
    for key, arr in actions.items():
        if arr.ndim == 2 and arr.shape[1] > 1:
            labels = _dim_labels.get(key, [])
            for col_idx in range(arr.shape[1]):
                dim_name = (
                    labels[col_idx] if col_idx < len(labels) else str(col_idx)
                )
                traces.append((f"{key} [{dim_name}]", arr[:, col_idx]))
        else:
            traces.append((key, arr.ravel()))

    time_hhmm = [
        f"{int(m) // 60:02d}:{int(m) % 60:02d}" for m in time
    ]

    figures: list[go.Figure] = []
    for i, (label, data) in enumerate(traces):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time, y=data, mode="lines",
            name=label,
            line=dict(color=_COLORS[i % len(_COLORS)]),
            customdata=time_hhmm,
            hovertemplate=(
                f"{label}<br>"
                "time: %{customdata}<br>"
                "value: %{y:.4f}"
                "<extra></extra>"
            ),
        ))
        _apply_day_xaxis(fig)
        fig.update_layout(title=f"{label}  —  {suffix}", height=350)
        figures.append(_style_figure(fig))

    return figures


# ---------------------------------------------------------------------------
# Figure: Rewards (stacked breakdown + total line)
# ---------------------------------------------------------------------------

def plot_rewards(episode: dict) -> go.Figure:
    """Stacked area for reward components, bold line for total reward."""
    time = episode["time_minutes"]
    breakdown = episode["reward_breakdown"]
    total = episode["rewards"]

    time_hhmm = [
        f"{int(m) // 60:02d}:{int(m) % 60:02d}" for m in time
    ]

    fig = go.Figure()

    # Stacked area traces for each reward component
    for i, (name, values) in enumerate(breakdown.items()):
        fig.add_trace(go.Scatter(
            x=time, y=values, mode="lines",
            name=name, stackgroup="rewards",
            line=dict(width=0.5, color=_COLORS[i % len(_COLORS)]),
            customdata=time_hhmm,
            hovertemplate=(
                f"{name}<br>"
                "time: %{customdata}<br>"
                "reward: %{y:.4f}"
                "<extra></extra>"
            ),
        ))

    # Total reward as a bold overlay line
    fig.add_trace(go.Scatter(
        x=time, y=total, mode="lines",
        name="total reward",
        line=dict(color="black", width=2.5),
        customdata=time_hhmm,
        hovertemplate=(
            "total reward<br>"
            "time: %{customdata}<br>"
            "reward: %{y:.4f}"
            "<extra></extra>"
        ),
    ))

    _apply_day_xaxis(fig)

    fig.update_layout(
        title=f"Rewards — {_title_suffix(episode)}",
        yaxis_title="Reward",
        height=450,
    )
    return _style_figure(fig)


# ---------------------------------------------------------------------------
# Figure: Energy (bar for power + line for cumulative)
# ---------------------------------------------------------------------------

def plot_energy(episode: dict) -> go.Figure:
    """Bar chart for instantaneous power, line for cumulative energy (dual y)."""
    time = episode["time_minutes"]
    power = episode["step_power_kW"]
    cum_e = episode["cum_E_kWh"]

    # HH:MM labels + both values for hover on each trace
    time_hhmm = [
        f"{int(m) // 60:02d}:{int(m) % 60:02d}" for m in time
    ]
    custom = list(zip(time_hhmm, power, cum_e))
    hover = (
        "time: %{customdata[0]}<br>"
        "power: %{customdata[1]:.3f} kW<br>"
        "cumulative: %{customdata[2]:.3f} kWh"
        "<extra></extra>"
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=time, y=power, name="Power (kW)",
            marker_color="#636EFA", opacity=0.7,
            customdata=custom, hovertemplate=hover,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=time, y=cum_e, mode="lines",
            name="Cumulative Energy (kWh)",
            line=dict(color="#EF553B", width=2.5),
            customdata=custom, hovertemplate=hover,
        ),
        secondary_y=True,
    )

    _apply_day_xaxis(fig)

    fig.update_layout(
        title=f"Energy — {_title_suffix(episode)}",
        height=450,
    )
    fig.update_yaxes(title_text="Power (kW)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Energy (kWh)", secondary_y=True)
    return _style_figure(fig)


# ---------------------------------------------------------------------------
# Kaleido / Chrome helper
# ---------------------------------------------------------------------------

def _ensure_chrome_for_kaleido() -> None:
    """Download Chrome for Kaleido if not already present.

    Kaleido v1+ requires a Chrome/Chromium binary for static image export
    (SVG, PNG, PDF).  This fetches it once into ``~/.cache/`` (user-local,
    no root required) so it works on headless HPC nodes.
    """
    # Suppress noisy kaleido / choreographer logs
    logging.getLogger("kaleido").setLevel(logging.WARNING)
    logging.getLogger("choreographer").setLevel(logging.WARNING)

    try:
        import kaleido
        kaleido.get_chrome_sync()
    except Exception as exc:
        logger.warning(
            "Could not ensure Chrome for static export: %s. "
            "Static image formats (svg/png/pdf) may fail. "
            "Run `kaleido_get_chrome` manually on a login node.",
            exc,
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_all_plots(
    hdf5_path: str,
    episode_id: str | None = None,
    output_dir: str | None = None,
    control_step_seconds: int = 300,
    formats: list[str] | None = None,
    select_by: str = "reward_rate",
) -> list[str]:
    """Load an episode, generate all four figures, save to output_dir.

    Args:
        hdf5_path: Path to trajectories.hdf5.
        episode_id: Episode to plot. None = best by ``select_by`` metric.
        output_dir: Output directory. Default: plotting/out/{episode_id}/.
        control_step_seconds: Timestep in seconds (default 300).
        formats: Output formats to produce. Default: ["html"].
        select_by: Summary metric for auto-selecting the best episode.

    Returns:
        List of saved file paths.
    """
    if formats is None:
        formats = ["html"]

    episode = load_episode(hdf5_path, episode_id, control_step_seconds, select_by)
    ep_id = episode["meta"]["episode_id"]

    if output_dir is None:
        output_dir = str(_DEFAULT_OUTPUT_ROOT / ep_id)
    os.makedirs(output_dir, exist_ok=True)

    # States and actions return lists of figures; rewards and energy are single
    multi_figures: dict[str, list[go.Figure]] = {
        "states": plot_states(episode),
        "actions": plot_actions(episode),
    }
    single_figures: dict[str, go.Figure] = {
        "rewards": plot_rewards(episode),
        "energy": plot_energy(episode),
    }

    saved: list[str] = []

    # Ensure Chrome/Chromium is available for static image export (svg/png/pdf).
    # Kaleido v1+ requires Chrome; this downloads it once to ~/.cache if missing.
    needs_static = any(f != "html" for f in formats)
    if needs_static:
        _ensure_chrome_for_kaleido()

    for fmt in formats:
        # Static image formats go into a subdirectory (e.g. <ep_id>/svgs/)
        if fmt == "html":
            fmt_dir = output_dir
        else:
            fmt_dir = os.path.join(output_dir, f"{fmt}s")
            os.makedirs(fmt_dir, exist_ok=True)

        for name, figs in multi_figures.items():
            if fmt == "html":
                filepath = os.path.join(fmt_dir, f"{ep_id}_{name}.html")
                _write_figure_list_html(figs, filepath)
            else:
                for i, fig in enumerate(figs):
                    filepath = os.path.join(fmt_dir, f"{ep_id}_{name}_{i}.{fmt}")
                    fig.write_image(filepath, width=1600, height=400)
                    logger.info("Saved: %s", filepath)
                    saved.append(filepath)
                continue
            logger.info("Saved: %s", filepath)
            saved.append(filepath)

        for name, fig in single_figures.items():
            filepath = os.path.join(fmt_dir, f"{ep_id}_{name}.{fmt}")
            if fmt == "html":
                fig.write_html(filepath)
            else:
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


if __name__ == "__main__":
    main()
