"""Plot action dimensions from an episode trajectory."""

from __future__ import annotations

import logging

import numpy as np
import plotly.graph_objects as go

from plotting.utils import COLORS, EpisodeData, apply_day_xaxis, load_plot_config, style_figure

logger = logging.getLogger(__name__)


def plot_actions(episode: EpisodeData) -> list[go.Figure]:
    """One independent plot per action dimension over a 24-hour day.

    Multi-dimensional actions (e.g. HP_action with energy + mode) are
    expanded so each dimension gets its own plot.
    Returns a list of figures to be rendered sequentially in one HTML file.
    """
    actions = episode.actions
    time = episode.time_minutes
    suffix = episode.title_suffix()
    time_hhmm = episode.time_hhmm

    if not actions:
        logger.warning("No action variables found.")
        return []

    dim_labels: dict[str, list[str]] = load_plot_config().get("actions", {}).get("dim_labels", {})

    # Build flat list of (label, 1-D data)
    traces: list[tuple[str, np.ndarray]] = []
    for key, arr in actions.items():
        if arr.ndim == 2 and arr.shape[1] > 1:
            labels = dim_labels.get(key, [])
            for col_idx in range(arr.shape[1]):
                dim_name = (
                    labels[col_idx] if col_idx < len(labels) else str(col_idx)
                )
                traces.append((f"{key} [{dim_name}]", arr[:, col_idx]))
        else:
            traces.append((key, arr.ravel()))

    figures: list[go.Figure] = []
    for i, (label, data) in enumerate(traces):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time, y=data, mode="lines",
            name=label,
            line=dict(color=COLORS[i % len(COLORS)]),
            customdata=time_hhmm,
            hovertemplate=(
                f"{label}<br>"
                "time: %{customdata}<br>"
                "value: %{y:.4f}"
                "<extra></extra>"
            ),
        ))
        apply_day_xaxis(fig)
        fig.update_layout(title=f"{label}  —  {suffix}", height=350)
        figures.append(style_figure(fig, n_legend_items=1))

    return figures
