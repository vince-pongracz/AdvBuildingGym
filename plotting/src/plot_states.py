"""Plot state variables from an episode trajectory."""

from __future__ import annotations

import logging

import numpy as np
import plotly.graph_objects as go

from .utils import COLORS, EpisodeData, apply_day_xaxis, style_figure

logger = logging.getLogger(__name__)

# State keys to exclude from plotting (constants or non-informative)
_SKIP_KEYS = {"E_price_max"}

# Groups of keys that share one plot
_GROUPED_KEYS: list[list[str]] = [
    ["battery_pct", "battery_target_pct"],
    ["temp_in_norm", "desired_temp_in_norm"],
    ["ev_schedule_charger_eff", "ev_schedule_discharge_eff"],
    ["ev_schedule_start_soc", "ev_schedule_target_soc"],
]


def plot_states(episode: EpisodeData) -> list[go.Figure]:
    """One independent plot per state variable (or group of variables).

    Keys listed in ``_GROUPED_KEYS`` are merged into a single plot.
    Returns a list of figures to be rendered sequentially in one HTML file.
    """
    states = episode.states
    time = episode.time_minutes
    suffix = episode.title_suffix()
    time_hhmm = episode.time_hhmm

    # Build ordered list of plot specs.  Each entry is a list of keys.
    grouped_flat = {k for group in _GROUPED_KEYS for k in group}
    plot_specs: list[list[str]] = []

    seen_groups: set[int] = set()
    for key, val in states.items():
        if key in _SKIP_KEYS or val.ndim != 1:
            continue
        if key in grouped_flat:
            for gi, group in enumerate(_GROUPED_KEYS):
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
        if key in _SKIP_KEYS or key in grouped_flat:
            continue
        if val.ndim == 2 and val.shape[1] <= 4:
            plot_specs.append([key])

    if not plot_specs:
        logger.warning("No plottable state variables found.")
        return []

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
                    line=dict(color=COLORS[trace_idx % len(COLORS)]),
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
                            color=COLORS[trace_idx % len(COLORS)],
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

        apply_day_xaxis(fig)
        fig.update_layout(title=f"{title}  —  {suffix}", height=350)
        figures.append(style_figure(fig))

    return figures
