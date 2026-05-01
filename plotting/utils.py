"""Shared plotting utilities: data loading, discovery, and figure styling."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import plotly.graph_objects as go
import yaml

logger = logging.getLogger(__name__)

# Repository root (used for resolving relative paths in config)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_METRICS_ROOT = _REPO_ROOT / "ep_metrics" / "trajectories"

# Colour palette shared across all plot modules
COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]

# ---------------------------------------------------------------------------
# Plot configuration (loaded from plot_config.yaml)
# ---------------------------------------------------------------------------

_PLOT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "plot_config.yaml"
_plot_config_cache: dict[str, Any] | None = None


def load_plot_config() -> dict[str, Any]:
    """Load and cache ``plotting/config/plot_config.yaml``."""
    global _plot_config_cache
    if _plot_config_cache is None:
        with open(_PLOT_CONFIG_PATH, encoding="utf-8") as fh:
            _plot_config_cache = yaml.safe_load(fh)
        logger.debug("Loaded plot config from %s", _PLOT_CONFIG_PATH)
    return _plot_config_cache


def get_output_root() -> Path:
    """Return the trajectory output directory from plot_config.yaml."""
    cfg = load_plot_config()
    rel = cfg.get("output", {}).get("dir", "plotting/out/traj")
    return _REPO_ROOT / rel


# ---------------------------------------------------------------------------
# Episode data container
# ---------------------------------------------------------------------------

@dataclass
class EpisodeData:
    """Typed container for a single episode loaded from HDF5.

    Replaces the plain ``dict`` previously returned by ``load_episode``.
    All array fields use ``np.float32``.
    """

    episode_id: str
    seed: int
    length: int
    episode_date: str | None = None

    # Summary scalars (reward_rate, achieved_reward, cum_E_kWh, …)
    summary: dict[str, float] = field(default_factory=dict)

    # Time axis in minutes (float32)
    time_minutes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    # Observation / state variables  {key: 1-D or 2-D ndarray}
    states: dict[str, np.ndarray] = field(default_factory=dict)

    # Action variables  {key: 1-D or 2-D ndarray}
    actions: dict[str, np.ndarray] = field(default_factory=dict)

    # Total reward per timestep
    rewards: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    # Per-component reward breakdown  {name: 1-D ndarray}
    reward_breakdown: dict[str, np.ndarray] = field(default_factory=dict)

    # Instantaneous power per timestep (kW)
    step_power_kW: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    # Cumulative energy per timestep (kWh)
    cum_E_kWh: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    # Per-infrastructure power breakdown  {infra_name: 1-D ndarray (kW)}
    power_breakdown: dict[str, np.ndarray] = field(default_factory=dict)

    # Raw (unnormalised) physical values  {name: 1-D ndarray}
    # e.g. temp_out_raw (°C), desired_temp_in_raw (°C), temp_in_raw (°C)
    raw: dict[str, np.ndarray] = field(default_factory=dict)

    # -- convenience helpers ------------------------------------------------

    @property
    def time_hhmm(self) -> list[str]:
        """Return HH:MM strings for each timestep (for hover labels)."""
        return [
            f"{int(m) // 60:02d}:{int(m) % 60:02d}" for m in self.time_minutes
        ]

    def title_suffix(self) -> str:
        """Short suffix with episode metadata for figure titles.

        Format: ``ep {id}, date: {YYYY.MM.DD} | reward_rate {x} ; achieved_reward {y}``.
        Falls back gracefully when the date is missing.
        """
        date_str = self.episode_date or ""
        # Convention: dots between Y/M/D in titles, even if HDF5 stored "YYYY-MM-DD".
        if date_str:
            date_str = date_str.replace("-", ".")
            ep_part = f"ep {self.episode_id}, date: {date_str}"
        else:
            ep_part = f"ep {self.episode_id}"
        return (
            f"{ep_part}  |  "
            f"reward_rate {self.summary.get('reward_rate', 0):.3f} ; "
            f"achieved_reward {self.summary.get('achieved_reward', 0):.2f}"
        )


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
) -> EpisodeData:
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
        An ``EpisodeData`` instance.
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

        seed = int(ep.attrs.get("seed", 0))
        length = int(ep.attrs.get("length", 0))
        ep_date_attr = ep.attrs.get("episode_date")
        if isinstance(ep_date_attr, bytes):
            ep_date_attr = ep_date_attr.decode("utf-8")
        episode_date = str(ep_date_attr) if ep_date_attr is not None else None
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

        # Per-infrastructure power breakdown
        power_breakdown: dict[str, np.ndarray] = {}
        if "power_breakdown" in traj:
            for key in traj["power_breakdown"]:
                power_breakdown[key] = traj["power_breakdown"][key][:]

        # Raw (unnormalised) physical values
        raw: dict[str, np.ndarray] = {}
        if "raw" in traj:
            for key in traj["raw"]:
                raw[key] = traj["raw"][key][:]

    return EpisodeData(
        episode_id=episode_id,
        seed=seed,
        length=length,
        episode_date=episode_date,
        summary=summary,
        time_minutes=time_minutes,
        states=states,
        actions=actions,
        rewards=rewards,
        reward_breakdown=reward_breakdown,
        step_power_kW=power,
        cum_E_kWh=cum_e,
        power_breakdown=power_breakdown,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Figure styling helpers
# ---------------------------------------------------------------------------

def apply_day_xaxis(
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


def align_zero_dual_yaxes(fig: go.Figure, y1_data: list[float], y2_data: list[float]) -> None:
    """Set both y-axis ranges so that zero sits at the same vertical position."""
    min1, max1 = min(y1_data), max(y1_data)
    min2, max2 = min(y2_data), max(y2_data)

    # Add 10% padding
    pad1 = (max1 - min1) * 0.1 or 1.0
    pad2 = (max2 - min2) * 0.1 or 1.0
    min1, max1 = min1 - pad1, max1 + pad1
    min2, max2 = min2 - pad2, max2 + pad2

    # Compute the fraction of the range below zero for each axis
    frac1 = abs(min1) / (abs(min1) + abs(max1)) if (abs(min1) + abs(max1)) > 0 else 0.5
    frac2 = abs(min2) / (abs(min2) + abs(max2)) if (abs(min2) + abs(max2)) > 0 else 0.5

    # Use the larger zero-fraction so both axes have room
    frac = max(frac1, frac2)

    # Expand each axis so zero sits at the same relative position
    # range_below = frac * total_range, range_above = (1 - frac) * total_range
    span1 = max(abs(min1) / frac if frac > 0 else max1,
                abs(max1) / (1 - frac) if frac < 1 else abs(min1))
    span2 = max(abs(min2) / frac if frac > 0 else max2,
                abs(max2) / (1 - frac) if frac < 1 else abs(min2))

    fig.update_yaxes(range=[-frac * span1, (1 - frac) * span1], secondary_y=False)
    fig.update_yaxes(range=[-frac * span2, (1 - frac) * span2], secondary_y=True)


def style_figure(fig: go.Figure) -> go.Figure:
    """Apply consistent styling."""
    fig.update_layout(
        template="plotly_white",
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig


def write_figure_list_html(
    figures: list[go.Figure],
    filepath: str,
    footnote: str = "",
) -> None:
    """Write a list of independent figures into a single HTML file."""
    parts: list[str] = [
        "<html><head><meta charset='utf-8'/>"
        "</head><body>",
    ]
    # First figure embeds the bundled Plotly.js so the version always matches
    # the binary-encoded arrays that Plotly Python generates.
    for i, fig in enumerate(figures):
        include_js = True if i == 0 else False
        parts.append(fig.to_html(full_html=False, include_plotlyjs=include_js))
    if footnote:
        parts.append(footnote)
    parts.append("</body></html>")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def ensure_chrome_for_kaleido() -> None:
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
