"""Plot raw policy actions (pre-rescale, tanh-bounded) from an episode trajectory.

These are the raw outputs of the policy network *before* ``FlattenAction`` and
``RescaleAction`` map them to physical action units. They live in ``[-1, 1]``
and are useful for diagnosing saturation / dead dimensions.
"""

from __future__ import annotations

import logging

import plotly.graph_objects as go

from plotting.utils import COLORS, EpisodeData, apply_day_xaxis, style_figure

logger = logging.getLogger(__name__)


def plot_raw_policy_actions(episode: EpisodeData) -> list[go.Figure]:
    """One independent plot per raw policy action dimension."""
    raw_actions = episode.raw_policy_actions
    time = episode.time_minutes
    suffix = episode.title_suffix()
    time_hhmm = episode.time_hhmm

    if not raw_actions:
        logger.info("No raw policy actions found in episode data.")
        return []

    # Sort by trailing numeric index so dimension order is stable.
    def _idx(key: str) -> int:
        try:
            return int(key.rsplit("_", 1)[-1])
        except ValueError:
            return 0

    sorted_keys = sorted(raw_actions.keys(), key=_idx)

    figures: list[go.Figure] = []
    for i, key in enumerate(sorted_keys):
        data = raw_actions[key]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time, y=data, mode="lines",
            name=key,
            line=dict(color=COLORS[i % len(COLORS)]),
            customdata=time_hhmm,
            hovertemplate=(
                f"{key}<br>"
                "time: %{customdata}<br>"
                "value: %{y:.4f}"
                "<extra></extra>"
            ),
        ))
        apply_day_xaxis(fig)
        fig.update_yaxes(range=[-1.05, 1.05])
        fig.update_layout(title=f"{key}  —  {suffix}", height=350)
        figures.append(style_figure(fig, n_legend_items=1))

    return figures
