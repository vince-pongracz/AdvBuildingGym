"""Plot energy consumption from an episode trajectory."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plotting.utils import EpisodeData, apply_day_xaxis, style_figure


def plot_energy(episode: EpisodeData) -> list[go.Figure]:
    """Stacked bar chart for per-infra power, line for cumulative energy (dual y).

    When per-infrastructure breakdown is available, each timestep shows two
    bar groups side by side: a single net-power bar and a stacked breakdown bar.
    Falls back to a single bar when breakdown data is unavailable.
    """
    time = episode.time_minutes
    power = episode.step_power_kW
    cum_e = episode.cum_E_kWh
    time_hhmm = episode.time_hhmm
    power_breakdown = episode.power_breakdown

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Net power bar (always shown)
    net_custom = list(zip(time_hhmm, power, cum_e))
    net_hover = (
        "time: %{customdata[0]}<br>"
        "net power: %{customdata[1]:.3f} kW<br>"
        "cumulative: %{customdata[2]:.3f} kWh"
        "<extra></extra>"
    )
    fig.add_trace(
        go.Bar(
            x=time, y=power, name="Net Power (kW)",
            marker_color="#636EFA", opacity=0.7,
            customdata=net_custom, hovertemplate=net_hover,
            offsetgroup="net",
        ),
        secondary_y=False,
    )

    if power_breakdown:
        # Stacked bar — one trace per infrastructure, placed as a second
        # bar group next to the net power bar
        for infra_name, infra_power in power_breakdown.items():
            custom = list(zip(time_hhmm, infra_power, cum_e))
            hover = (
                "time: %{customdata[0]}<br>"
                f"{infra_name}: " + "%{y:.3f} kW<br>"
                "cumulative: %{customdata[2]:.3f} kWh"
                "<extra></extra>"
            )
            fig.add_trace(
                go.Bar(
                    x=time, y=infra_power, name=f"{infra_name} (kW)",
                    opacity=0.7,
                    customdata=custom, hovertemplate=hover,
                    offsetgroup="breakdown",
                ),
                secondary_y=False,
            )

    fig.update_layout(barmode="relative", bargap=0.5)

    # Cumulative energy line (always shown)
    cum_custom = list(zip(time_hhmm, power, cum_e))
    cum_hover = (
        "time: %{customdata[0]}<br>"
        "net power: %{customdata[1]:.3f} kW<br>"
        "cumulative: %{customdata[2]:.3f} kWh"
        "<extra></extra>"
    )
    fig.add_trace(
        go.Scatter(
            x=time, y=cum_e, mode="lines",
            name="Cumulative Energy (kWh)",
            line=dict(color="#EF553B", width=2.5),
            customdata=cum_custom, hovertemplate=cum_hover,
        ),
        secondary_y=True,
    )

    apply_day_xaxis(fig)

    fig.update_layout(
        title=f"Energy (full breakdown) — {episode.title_suffix()}",
        height=450,
    )
    fig.update_yaxes(title_text="Power (kW)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Energy (kWh)", secondary_y=True)

    # --- Second figure: net power + cumulative energy only ---
    fig_net = make_subplots(specs=[[{"secondary_y": True}]])

    fig_net.add_trace(
        go.Bar(
            x=time, y=power, name="Net Power (kW)",
            marker_color="#636EFA", opacity=0.7,
            customdata=net_custom, hovertemplate=net_hover,
        ),
        secondary_y=False,
    )

    fig_net.add_trace(
        go.Scatter(
            x=time, y=cum_e, mode="lines",
            name="Cumulative Energy (kWh)",
            line=dict(color="#EF553B", width=2.5),
            customdata=cum_custom, hovertemplate=cum_hover,
        ),
        secondary_y=True,
    )

    apply_day_xaxis(fig_net)

    fig_net.update_layout(
        title=f"Net Energy — {episode.title_suffix()}",
        height=400,
        bargap=0.5,
    )
    fig_net.update_yaxes(title_text="Net Power (kW)", secondary_y=False)
    fig_net.update_yaxes(title_text="Cumulative Energy (kWh)", secondary_y=True)

    return [style_figure(fig), style_figure(fig_net)]


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
