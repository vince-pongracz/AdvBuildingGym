"""Plot energy consumption from an episode trajectory."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import EpisodeData, apply_day_xaxis, style_figure


def plot_energy(episode: EpisodeData) -> go.Figure:
    """Bar chart for instantaneous power, line for cumulative energy (dual y)."""
    time = episode.time_minutes
    power = episode.step_power_kW
    cum_e = episode.cum_E_kWh
    time_hhmm = episode.time_hhmm

    # Both values for hover on each trace
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

    apply_day_xaxis(fig)

    fig.update_layout(
        title=f"Energy — {episode.title_suffix()}",
        height=450,
    )
    fig.update_yaxes(title_text="Power (kW)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Energy (kWh)", secondary_y=True)

    return style_figure(fig)


# HTML footnote rendered as a separate div below the energy plot
ENERGY_SIGN_CONVENTION_HTML = (
    '<div style="max-width:900px; margin:0.8em auto; padding:0.6em 1em;'
    " background:#f8f8f8; border-left:3px solid #636EFA;"
    ' font-size:0.9em; line-height:1.5; font-family:sans-serif;">'
    
    "<b>Sign convention</b><br>"
    "<b>Power (kW):</b><br>"
    "  - positive = electricity drawn from the grid (consumption)<br>"
    "  - negative = electricity fed back to the grid (generation / export).<br>"
    
    "<b>Cumulative Energy (kWh):</b> running sum of power over the episode.<br>"
    "  - positive = more energy consumed than exported so far<br>"
    "  - negative = more energy exported than consumed so far."
    "</div>"
)
