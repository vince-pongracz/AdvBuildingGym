"""Plot raw (unnormalised) physical values from an episode trajectory.

These values are NOT part of the RL observation space.  They exist only
for human-readable visualisation of the actual physical quantities
(temperatures in °C, prices in €/kWh, etc.) that the normalised state
variables represent.
"""

from __future__ import annotations

import logging

import plotly.graph_objects as go

from plotting.utils import COLORS, EpisodeData, apply_day_xaxis, load_plot_config, style_figure

logger = logging.getLogger(__name__)

# Human-readable y-axis unit labels for known raw-value keys.
# Keys not listed here fall back to a generic "Value" label.
_UNIT_LABELS: dict[str, str] = {
    "temp_out_raw": "Temperature (°C)",
    "temp_in_raw": "Temperature (°C)",
    "desired_temp_in_raw": "Temperature (°C)",
    "E_price_raw": "Price (€/kWh)",
    "E_price_max_raw": "Price (€/kWh)",
}

# Suffix-based fallback for keys not in _UNIT_LABELS (auto-discovered _raw attrs).
_SUFFIX_UNITS: list[tuple[str, str]] = [
    ("_temp_", "Temperature (°C)"),
    ("_price_", "Price (€/kWh)"),
    ("_kW_", "Power (kW)"),
    ("_kWh_", "Energy (kWh)"),
]


def _unit_label(key: str) -> str:
    """Return a y-axis unit label for *key*, falling back to suffix heuristics."""
    label = _UNIT_LABELS.get(key)
    if label is not None:
        return label
    lower = key.lower()
    for fragment, unit in _SUFFIX_UNITS:
        if fragment in f"_{lower}_":
            return unit
    return "Value"


def plot_raw(episode: EpisodeData) -> list[go.Figure]:
    """One plot per raw physical value (or group of values) over a 24-hour day.

    Temperature keys are grouped into a single figure by default (they
    share units and are directly comparable).  Grouping is configurable
    via ``plot_config.yaml`` under the ``raw:`` section.

    Returns a list of figures to be rendered sequentially in one HTML file.
    """
    raw = episode.raw
    time = episode.time_minutes
    suffix = episode.title_suffix()
    time_hhmm = episode.time_hhmm

    if not raw:
        logger.info("No raw physical values found in episode data.")
        return []

    plot_cfg = load_plot_config().get("raw", {})
    grouped_keys: list[list[str]] = plot_cfg.get("grouped_keys", [
        ["temp_out_raw", "temp_in_raw", "desired_temp_in_raw"],
        ["E_price_raw", "E_price_max_raw"],
    ])
    skip_keys: set[str] = set(plot_cfg.get("skip_keys", []))

    # Build ordered list of plot specs (same pattern as plot_states.py)
    grouped_flat = {k for group in grouped_keys for k in group}
    plot_specs: list[list[str]] = []
    seen_groups: set[int] = set()

    for key in raw:
        if key in skip_keys:
            continue
        if key in grouped_flat:
            for gi, group in enumerate(grouped_keys):
                if key in group and gi not in seen_groups:
                    present = [k for k in group if k in raw]
                    if present:
                        plot_specs.append(present)
                        seen_groups.add(gi)
        else:
            plot_specs.append([key])

    figures: list[go.Figure] = []
    for keys in plot_specs:
        fig = go.Figure()
        title = " + ".join(keys)
        # Use the unit label of the first key in the group
        y_label = _unit_label(keys[0])

        for i, key in enumerate(keys):
            if key not in raw:
                continue
            arr = raw[key]
            fig.add_trace(go.Scatter(
                x=time, y=arr, mode="lines",
                name=key,
                line=dict(color=COLORS[i % len(COLORS)]),
                customdata=time_hhmm,
                hovertemplate=(
                    f"{key}<br>"
                    "time: %{customdata}<br>"
                    "value: %{y:.2f}"
                    "<extra></extra>"
                ),
            ))

        apply_day_xaxis(fig)
        fig.update_layout(
            title=f"{title}  —  {suffix}",
            yaxis_title=y_label,
            height=350,
        )
        figures.append(style_figure(fig))

    return figures
