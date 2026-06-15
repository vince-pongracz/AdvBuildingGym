"""Plot cumulative electricity cost from an episode trajectory."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plotting.utils import (
    EpisodeData,
    align_zero_dual_yaxes,
    apply_day_xaxis,
    get_width_multiplier,
    style_figure,
)


def plot_price(episode: EpisodeData) -> list[go.Figure]:
    """Per-step cost bars + cumulative electricity cost line (dual y).

    ``cum_price_EUR`` uses the consumption-positive convention (see
    ``core/_price_tracker.py``): positive = money spent drawing from the grid,
    negative = money earned by exporting. Per-step cost is the step-to-step
    difference of the cumulative series.
    """
    time = episode.time_minutes
    cum_price = episode.cum_price_EUR
    time_hhmm = episode.time_hhmm

    # Per-step incremental cost (EUR). The trajectory's step-0 row is the
    # zero-initialised initial condition, so a prepended-zero diff yields the
    # cost accrued during each step.
    step_cost = np.diff(cum_price, prepend=np.float32(0.0))

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    cost_custom = list(zip(time_hhmm, step_cost, cum_price))
    cost_hover = (
        "time: %{customdata[0]}<br>"
        "step cost: %{customdata[1]:.4f} EUR<br>"
        "cumulative: %{customdata[2]:.4f} EUR"
        "<extra></extra>"
    )

    fig.add_trace(
        go.Bar(
            x=time, y=step_cost, name="Step Cost (EUR)",
            marker_color="#FFA15A", opacity=0.7,
            customdata=cost_custom, hovertemplate=cost_hover,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=time, y=cum_price, mode="lines",
            name="Cumulative Cost (EUR)",
            line=dict(color="#EF553B", width=2.5),
            customdata=cost_custom, hovertemplate=cost_hover,
        ),
        secondary_y=True,
    )

    apply_day_xaxis(fig)

    fig.update_layout(
        title=f"Electricity Cost — {episode.title_suffix()}",
        height=400,
        bargap=0.25,
    )
    fig.update_yaxes(title_text="Step Cost (EUR)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Cost (EUR)", secondary_y=True)
    align_zero_dual_yaxes(fig, step_cost, cum_price)

    wm = get_width_multiplier("price")
    return [style_figure(fig, n_legend_items=2, width_multiplier=wm)]


# HTML footnote rendered as a separate div below the price plot
PRICE_SIGN_CONVENTION_HTML = (
    '<div style="max-width:900px; margin:0.8em auto; padding:0.6em 1em;'
    " background:#f8f8f8; border-left:3px solid #EF553B;"
    ' font-size:0.9em; line-height:1.5; font-family:sans-serif;">'

    "<b>Sign convention</b><br>"
    "<b>Cost (EUR):</b> consumption-positive (opposite of the power sign).<br>"
    "  - positive = money spent drawing energy from the grid<br>"
    "  - negative = money earned by exporting to the grid.<br>"

    "<b>Cumulative Cost (EUR):</b> running sum of per-step cost over the episode."
    "</div>"
)
