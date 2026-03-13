"""Plot reward breakdown from an episode trajectory."""

from __future__ import annotations

import plotly.graph_objects as go

from .utils import COLORS, EpisodeData, apply_day_xaxis, style_figure


def plot_rewards(episode: EpisodeData) -> go.Figure:
    """Stacked area for reward components, bold line for total reward."""
    time = episode.time_minutes
    breakdown = episode.reward_breakdown
    total = episode.rewards
    time_hhmm = episode.time_hhmm

    fig = go.Figure()

    # Stacked area traces for each reward component
    for i, (name, values) in enumerate(breakdown.items()):
        fig.add_trace(go.Scatter(
            x=time, y=values, mode="lines",
            name=name, stackgroup="rewards",
            line=dict(width=0.5, color=COLORS[i % len(COLORS)]),
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

    apply_day_xaxis(fig)

    fig.update_layout(
        title=f"Rewards — {episode.title_suffix()}",
        yaxis_title="Reward",
        height=450,
    )
    return style_figure(fig)
