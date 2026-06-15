"""Shared plotting utilities: data loading, discovery, and figure styling."""

from __future__ import annotations

import html
import logging
import math
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
# Plot configuration (loaded from traj_plot_config.yaml)
# ---------------------------------------------------------------------------

_PLOT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "traj_plot_config.yaml"
_plot_config_cache: dict[str, Any] | None = None


def load_plot_config() -> dict[str, Any]:
    """Load and cache ``plotting/config/traj_plot_config.yaml``."""
    global _plot_config_cache
    if _plot_config_cache is None:
        with open(_PLOT_CONFIG_PATH, encoding="utf-8") as fh:
            _plot_config_cache = yaml.safe_load(fh)
        logger.debug("Loaded plot config from %s", _PLOT_CONFIG_PATH)
    return _plot_config_cache


def get_output_root() -> Path:
    """Return the trajectory output directory from traj_plot_config.yaml."""
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

    # Summary scalars (achieved_reward, cum_E_kWh, …)
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

    # Instantaneous net power per timestep (kW)
    net_power_kW: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    # Cumulative energy per timestep (kWh)
    cum_E_kWh: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    # Cumulative electricity cost per timestep (EUR, positive = spent)
    cum_price_EUR: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    # Per-infrastructure power breakdown  {infra_name: 1-D ndarray (kW)}
    power_breakdown: dict[str, np.ndarray] = field(default_factory=dict)

    # Raw (unnormalised) physical values  {name: 1-D ndarray}
    # e.g. temp_out_raw (°C), desired_temp_in_raw (°C), temp_in_raw (°C)
    raw: dict[str, np.ndarray] = field(default_factory=dict)

    # Raw policy actions (pre-rescale, tanh-bounded) {raw_policy_action_<d>: 1-D ndarray}
    raw_policy_actions: dict[str, np.ndarray] = field(default_factory=dict)

    # -- convenience helpers ------------------------------------------------

    @property
    def time_hhmm(self) -> list[str]:
        """Return HH:MM strings for each timestep (for hover labels)."""
        return [
            f"{int(m) // 60:02d}:{int(m) % 60:02d}" for m in self.time_minutes
        ]

    def title_suffix(self) -> str:
        """Short suffix with episode metadata for figure titles.

        Format: ``ep {id}, date: {YYYY.MM.DD} | achieved_reward {y}``.
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
    select_by: str = "achieved_reward",
) -> EpisodeData:
    """Load one episode from an HDF5 trajectory file.

    Args:
        hdf5_path: Path to trajectories.hdf5.
        episode_id: Episode group name. If None, selects the best episode
            according to ``select_by``.
        control_step_seconds: Control timestep in seconds (default 300 = 5 min).
        select_by: Summary metric used to pick the best episode when
            ``episode_id`` is None. One of "achieved_reward",
            "cum_E_kWh". Default: "achieved_reward".

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
        cum_price = (
            traj["cum_price_EUR"][:] if "cum_price_EUR" in traj else np.zeros_like(steps)
        )
        # ``net_power_kW`` is the current key; older HDF5 files used
        # ``step_power_kW`` for the same quantity.
        if "net_power_kW" in traj:
            power = traj["net_power_kW"][:]
        elif "step_power_kW" in traj:
            power = traj["step_power_kW"][:]
        else:
            power = np.zeros_like(steps)

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

        # Raw policy actions (top-level datasets named raw_policy_action_<d>).
        # Pre-rescale tanh outputs from the policy network, useful for
        # diagnosing saturation. Dataset count = action-space dimensionality.
        raw_policy_actions: dict[str, np.ndarray] = {}
        for key in traj.keys():
            if key.startswith("raw_policy_action_") and isinstance(traj[key], h5py.Dataset):
                raw_policy_actions[key] = traj[key][:]

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
        net_power_kW=power,
        cum_E_kWh=cum_e,
        cum_price_EUR=cum_price,
        power_breakdown=power_breakdown,
        raw=raw,
        raw_policy_actions=raw_policy_actions,
    )


# ---------------------------------------------------------------------------
# Figure styling helpers
# ---------------------------------------------------------------------------

def _fig_cfg() -> dict[str, Any]:
    """Return the ``figure:`` subdict from traj_plot_config.yaml (with defaults)."""
    cfg = load_plot_config().get("figure", {}) or {}
    return {
        "width": int(cfg.get("width", 1100)),
        "tick_interval_min": int(cfg.get("tick_interval_min", 120)),
        "tick_angle": int(cfg.get("tick_angle", -60)),
        "base_top_margin": int(cfg.get("base_top_margin", 60)),
        "legend_row_px": int(cfg.get("legend_row_px", 22)),
        "legend_items_per_row": max(1, int(cfg.get("legend_items_per_row", 5))),
    }


def apply_day_xaxis(
    fig: go.Figure,
    n_rows: int | None = None,
) -> None:
    """Configure x-axis as a 24-hour day.

    Tick interval and rotation are loaded from ``traj_plot_config.yaml`` under the
    ``figure:`` section (``tick_interval_min`` / ``tick_angle``).

    Args:
        fig: Plotly figure.
        n_rows: If the figure uses make_subplots, pass the total row count
            so the label is placed on the bottom subplot. None for a plain
            figure without subplots.
    """
    cfg = _fig_cfg()
    interval = cfg["tick_interval_min"]
    tick_vals = list(range(0, 1441, interval))
    tick_text = [f"{m // 60:02d}:{m % 60:02d}" for m in tick_vals]
    base_kwargs: dict = dict(
        range=[0, 1440],
        tickvals=tick_vals,
        ticktext=tick_text,
        tickangle=cfg["tick_angle"],
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


def style_figure(
    fig: go.Figure,
    *,
    n_legend_items: int = 0,
    width_multiplier: float = 1.0,
) -> go.Figure:
    """Apply consistent styling.

    The top margin grows with ``n_legend_items`` so the title never overlaps
    the (top-anchored, horizontal) legend regardless of legend size. The
    title is pinned to the top of the paper, the legend stacked directly
    below it. Figure width is ``figure.width * width_multiplier`` (defaults
    to the configured ``figure.width``).
    """
    cfg = _fig_cfg()
    rows = math.ceil(n_legend_items / cfg["legend_items_per_row"]) if n_legend_items else 0
    legend_block = rows * cfg["legend_row_px"]
    top_margin = cfg["base_top_margin"] + legend_block
    width = int(cfg["width"] * width_multiplier)

    # Reserve the top ~30 px of the margin for the title, then place the
    # legend (top-anchored) just below it. Both title and legend use
    # container coordinates ([0, 1] across the FULL figure including
    # margins) — the default yref for legend is "paper" (plot area only),
    # which would put the legend inside the plot.
    height = fig.layout.height or 450
    title_band_px = 30
    title_y = 1 - (title_band_px / 2) / height       # centre of title band
    legend_y = 1 - (title_band_px + 6) / height       # top of legend, 6 px below title

    fig.update_layout(
        template="plotly_white",
        font=dict(size=12),
        width=width,
        title=dict(y=title_y, yanchor="middle", xanchor="left", x=0.02),
        legend=dict(
            orientation="h",
            yref="container", yanchor="top", y=legend_y,
            xref="container", xanchor="right", x=1 - 30 / width,
        ),
        margin=dict(l=60, r=30, t=top_margin, b=50),
    )
    return fig


def get_width_multiplier(group: str) -> float:
    """Return the configured width multiplier for a plot group (default 1.0)."""
    cfg = load_plot_config().get("figure", {}) or {}
    mults = cfg.get("width_multipliers", {}) or {}
    try:
        return float(mults.get(group, 1.0))
    except (TypeError, ValueError):
        return 1.0


# ---------------------------------------------------------------------------
# Shared dashboard / card-layout assets
# ---------------------------------------------------------------------------

_DASHBOARD_DIR = _REPO_ROOT / "plotting" / "dashboard"


def read_dashboard_asset(*parts: str) -> str:
    """Read a text asset bundled under ``plotting/dashboard/`` (vendored libs, templates)."""
    return (_DASHBOARD_DIR.joinpath(*parts)).read_text(encoding="utf-8")


def short_label_from_fig(fig: go.Figure) -> str:
    """Return the key/name part of a figure title, stripped of the decorated suffix.

    Figure titles follow ``"<keys> — ep <id>, date: … | achieved_reward …"``; the
    ticker labels and card headers only want the ``<keys>`` prefix before the em dash.
    """
    title = ""
    try:
        title = fig.layout.title.text or ""
    except AttributeError:
        title = ""
    # The em dash (U+2014) separates the key/name from the episode suffix; the
    # suffix never contains one, so splitting on the first dash is safe.
    label = title.split("—", 1)[0].strip() if title else ""
    return label or "plot"


# Card layout/behaviour live in standalone asset files so they can be edited as
# CSS/JS (with editor tooling) rather than as opaque Python strings. Both are
# shared by the per-group HTML files and the aggregated dashboard.
#   - plot_card.css : reorderable flex-card layout (cards carry their own
#       resize handle + a header bar that doubles as the SortableJS drag handle).
#   - card_resize.js: ResizeObserver re-flowing each Plotly graph on card resize;
#       the ``__IDS__`` placeholder is filled with the graph div ids at embed time.
PLOT_CARD_CSS = read_dashboard_asset("assets", "plot_card.css")
_CARD_RESIZE_JS = read_dashboard_asset("assets", "card_resize.js")


def write_figure_list_html(
    figures: list[go.Figure],
    filepath: str,
    footnote: str = "",
) -> None:
    """Write a list of independent figures into a single HTML file.

    Each figure becomes a reorderable card in a flexbox row (``flex-wrap: wrap``):
    drag a card's header bar to change its order; drag a card's bottom-right corner
    to resize it live (a ResizeObserver re-triggers ``Plotly.Plots.resize`` so axis
    ranges and tick density update). Reordering uses SortableJS with the header as
    the drag handle, so dragging *inside* a plot still zooms/pans normally.
    """
    import json as _json

    parts: list[str] = [
        "<html><head><meta charset='utf-8'/>",
        "<style>" + PLOT_CARD_CSS + "body{margin:12px;}</style>",
        "</head><body>",
        "<div class='plot-flex' id='plotFlex'>",
    ]
    div_ids: list[str] = []
    # First figure embeds the bundled Plotly.js so the version always matches
    # the binary-encoded arrays that Plotly Python generates.
    for i, fig in enumerate(figures):
        include_js = True if i == 0 else False
        initial_h = int(fig.layout.height) if fig.layout.height else 450
        label = short_label_from_fig(fig)
        # Make the figure fill its card body; the card drives sizing.
        fig.update_layout(autosize=True, width=None, height=None)
        div_id = f"adv_plot_{i}"
        div_ids.append(div_id)
        # +34 px reserves room for the header bar so the plot keeps its height.
        parts.append(f'<div class="plot-card" style="height:{initial_h + 34}px;">')
        parts.append(
            f'<div class="plot-card-header"><span class="grip">&#x283F;</span>'
            f'<span class="plot-card-title" title="{html.escape(label)}">{html.escape(label)}</span></div>'
        )
        parts.append('<div class="plot-body">')
        parts.append(fig.to_html(
            full_html=False,
            include_plotlyjs=include_js,
            div_id=div_id,
            default_width="100%",
            default_height="100%",
            config={"responsive": True},
        ))
        parts.append("</div></div>")
    parts.append("</div>")  # .plot-flex
    if footnote:
        parts.append(footnote)
    parts.append("<script>" + _CARD_RESIZE_JS.replace("__IDS__", _json.dumps(div_ids)) + "</script>")
    # SortableJS is an npm dependency read from node_modules and inlined here (see
    # README, 'Dashboard assets'). Imported lazily to avoid an import-time cycle
    # (plotting.dashboard imports utils). require() only checks — it never installs.
    from plotting.dashboard import vendor_assets
    vendor_assets.require(vendor_assets.SORTABLE)
    parts.append("<script>" + vendor_assets.read("sortable.js") + "</script>")
    parts.append(
        "<script>new Sortable(document.getElementById('plotFlex'),"
        "{handle:'.plot-card-header',animation:150,ghostClass:'sortable-ghost'});</script>"
    )
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
