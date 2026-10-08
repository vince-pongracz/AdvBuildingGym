"""Plot reward breakdown from an episode trajectory."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from plotting.utils import COLORS, EpisodeData, apply_day_xaxis, style_figure


def plot_rewards(episode: EpisodeData) -> list[go.Figure]:
    """Per-step reward breakdown plus its cumulative counterpart.

    Returns two figures sharing the ``rewards`` group: the per-step stacked
    area (components) with a bold total-reward overlay, followed by the
    cumulative version whose total line ends at ``achieved_reward``.
    """
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
    n_legend = len(breakdown) + 1  # +1 for the total reward trace
    return [
        style_figure(fig, n_legend_items=n_legend),
        _cumulative_reward_figure(episode),
    ]


def _cumulative_reward_figure(episode: EpisodeData) -> go.Figure:
    """Cumulative reward: stacked per-component running sums + bold total line.

    Mirrors the per-step figure but accumulates with ``np.cumsum``. Because the
    per-step total equals the sum of its components and ``cumsum`` is linear, the
    stacked components sum exactly to the cumulative-total line at every step; the
    line's final value equals the episode's ``achieved_reward``.
    """
    time = episode.time_minutes
    breakdown = episode.reward_breakdown
    time_hhmm = episode.time_hhmm
    cum_total = np.cumsum(episode.rewards)

    fig = go.Figure()

    # Stacked area traces for each reward component's running sum
    for i, (name, values) in enumerate(breakdown.items()):
        cum_values = np.cumsum(values)
        # customdata carries (HH:MM, per-step contribution) so the hover shows
        # both the marginal reward and the running total at this step.
        customdata = list(zip(time_hhmm, values))
        fig.add_trace(go.Scatter(
            x=time, y=cum_values, mode="lines",
            name=name, stackgroup="cum_rewards",
            line=dict(width=0.5, color=COLORS[i % len(COLORS)]),
            customdata=customdata,
            hovertemplate=(
                f"{name}<br>"
                "time: %{customdata[0]}<br>"
                "step: %{customdata[1]:.4f}<br>"
                "cumulative: %{y:.4f}"
                "<extra></extra>"
            ),
        ))

    # Cumulative total reward as a bold overlay line
    total_custom = list(zip(time_hhmm, episode.rewards))
    fig.add_trace(go.Scatter(
        x=time, y=cum_total, mode="lines",
        name="cumulative total",
        line=dict(color="black", width=2.5),
        customdata=total_custom,
        hovertemplate=(
            "cumulative total<br>"
            "time: %{customdata[0]}<br>"
            "step: %{customdata[1]:.4f}<br>"
            "cumulative: %{y:.4f}"
            "<extra></extra>"
        ),
    ))

    apply_day_xaxis(fig)

    fig.update_layout(
        title=f"Cumulative Rewards — {episode.title_suffix()}",
        yaxis_title="Cumulative Reward",
        height=450,
    )
    n_legend = len(breakdown) + 1  # +1 for the cumulative total trace
    return style_figure(fig, n_legend_items=n_legend)
