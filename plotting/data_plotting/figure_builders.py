"""Figure builder functions for day-data plots.

Each public function accepts pre-loaded DataFrames (keyed by trace label)
and returns one or more ``plotly.graph_objects.Figure`` instances ready
for export.  All figures share the same 24-hour x-axis via
``apply_day_xaxis``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plotting.utils import COLORS, apply_day_xaxis, style_figure

from .stats import add_stat_traces, compute_column_stats

logger = logging.getLogger(__name__)

# Weather columns to plot: (column, y-axis label, fallback column)
_WEATHER_COLS = [
    ("temp_amb", "Temperature (\u00b0C)", None),
    ("rel_humidity", "Relative humidity (%)", None),
    ("avg_wind_speed", "Wind speed (m/s)", None),
    ("sun_shine", "Global irradiance (J/cm\u00b2)", "direct_sun_shine"),
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def finalize_figure(fig: go.Figure, title: str, height: int = 350) -> None:
    """Apply title, sizing, and shared styling to a figure."""
    fig.update_layout(
        title_text=title, height=height, showlegend=True,
        title=dict(automargin=True, yref="container"),
    )
    style_figure(fig)
    fig.update_layout(margin=dict(t=90))


def _days_subtitle(day_labels: list[str]) -> str:
    """Build a subtitle listing the aggregated days."""
    return "<br><sub>Days: " + ", ".join(day_labels) + "</sub>"


def _add_day_traces(
    fig: go.Figure,
    day_frames: dict[str, pd.DataFrame],
    col: str,
    hover_label: str,
    hover_fmt: str = ".1f",
) -> None:
    """Add one coloured line trace per day for *col*."""
    for day_idx, (day_label, df) in enumerate(day_frames.items()):
        if col not in df.columns:
            continue
        color = COLORS[day_idx % len(COLORS)]
        minutes = df["minutes"]
        hhmm = [f"{int(m) // 60:02d}:{int(m) % 60:02d}" for m in minutes]
        fig.add_trace(go.Scatter(
            x=minutes,
            y=df[col],
            mode="lines",
            name=day_label,
            line=dict(color=color, width=1.5),
            customdata=np.column_stack([hhmm, [day_label] * len(minutes)]),
            hovertemplate=(
                "<b>%{customdata[1]}</b> %{customdata[0]}<br>"
                f"{hover_label}: " + "%{y:" + hover_fmt + "}"
                "<extra></extra>"
            ),
        ))


def _resolve_weather_cols(sample_df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return (column_name, label) pairs available in the data."""
    available = []
    for col, label, alt in _WEATHER_COLS:
        if col in sample_df.columns:
            available.append((col, label))
        elif alt and alt in sample_df.columns:
            available.append((alt, label))
    return available


# ---------------------------------------------------------------------------
# Generic overlay figure
# ---------------------------------------------------------------------------

def build_overlay_figure(
    day_frames: dict[str, pd.DataFrame],
    value_col: str,
    y_label: str,
    title: str,
    hover_label: str,
    hover_fmt: str = ".1f",
    stat_only: bool = False,
) -> go.Figure:
    """Build a single figure with one line trace per entry, plus stat bands.

    This is the common pattern used by weather sub-plots, desired-temperature,
    and household-consumption figures.
    """
    fig = go.Figure()
    is_multi = len(day_frames) > 1

    if not (is_multi and stat_only):
        _add_day_traces(fig, day_frames, value_col, hover_label, hover_fmt)

    if is_multi:
        stats = compute_column_stats(day_frames, value_col)
        if stats is not None:
            add_stat_traces(fig, stats, value_label=hover_label, value_fmt=hover_fmt)

    apply_day_xaxis(fig)
    fig.update_yaxes(title_text=y_label)

    if is_multi and stat_only:
        title += _days_subtitle(list(day_frames.keys()))

    finalize_figure(fig, title)
    return fig


# ---------------------------------------------------------------------------
# Source-specific builders
# ---------------------------------------------------------------------------

def build_weather_figures(
    day_frames: dict[str, pd.DataFrame],
    stat_only: bool = False,
) -> list[go.Figure]:
    """Create one standalone figure per weather variable, each with all days overlaid."""
    sample_df = next(iter(day_frames.values()))
    available = _resolve_weather_cols(sample_df)
    if not available:
        logger.warning("No plottable weather columns found.")
        return []

    return [
        build_overlay_figure(
            day_frames,
            value_col=col,
            y_label=label,
            title=label,
            hover_label=col,
            stat_only=stat_only,
        )
        for col, label in available
    ]


def build_price_figure(
    day_frames: dict[str, pd.DataFrame],
    stat_only: bool = False,
) -> go.Figure:
    """Create a single-panel figure with one price trace per day.

    Single-day plots use an area fill; multi-day plots use plain lines.
    """
    fig = go.Figure()
    day_labels = list(day_frames.keys())
    is_multi = len(day_labels) > 1

    if not (is_multi and stat_only):
        for day_idx, (day_label, df) in enumerate(day_frames.items()):
            color = COLORS[day_idx % len(COLORS)]
            minutes = df["minutes"]
            hhmm = [f"{int(m) // 60:02d}:{int(m) % 60:02d}" for m in minutes]

            fill = "tozeroy" if not is_multi else "none"
            fillcolor = "rgba(99,110,250,0.15)" if not is_multi else None

            fig.add_trace(go.Scatter(
                x=minutes,
                y=df["baseprice"],
                mode="lines",
                name=day_label,
                line=dict(color=color, width=2),
                fill=fill,
                fillcolor=fillcolor,
                customdata=np.column_stack([hhmm, [day_label] * len(minutes)]),
                hovertemplate=(
                    "<b>%{customdata[1]}</b> %{customdata[0]}<br>"
                    "price: %{y:.2f} ct/kWh"
                    "<extra></extra>"
                ),
            ))

    if is_multi:
        stats = compute_column_stats(day_frames, "baseprice")
        if stats is not None:
            add_stat_traces(fig, stats, value_label="price", value_fmt=".2f")

    apply_day_xaxis(fig)
    fig.update_yaxes(title_text="Energy price (ct/kWh)")
    title = (
        f"Energy price \u2014 {day_labels[0]}"
        if not is_multi
        else f"Energy price \u2014 {len(day_labels)} days"
    )
    if is_multi and stat_only:
        title += _days_subtitle(day_labels)

    finalize_figure(fig, title)
    return fig


def build_desired_temp_figure(
    profile_frames: dict[str, pd.DataFrame],
    value_col: str,
    stat_only: bool = False,
) -> go.Figure:
    """Create a figure overlaying all desired-temperature profiles."""
    n = len(profile_frames)
    return build_overlay_figure(
        profile_frames,
        value_col=value_col,
        y_label="Desired indoor temperature (\u00b0C)",
        title=f"Desired indoor temperature \u2014 {n} profile{'s' if n != 1 else ''}",
        hover_label="desired T",
        hover_fmt=".1f",
        stat_only=stat_only,
    )


def build_user_energy_need_figure(
    profile_frames: dict[str, pd.DataFrame],
    value_col: str,
    stat_only: bool = False,
) -> go.Figure:
    """Create a figure overlaying all household consumption profiles."""
    n = len(profile_frames)
    return build_overlay_figure(
        profile_frames,
        value_col=value_col,
        y_label="Household consumption (kW)",
        title=f"Household energy consumption \u2014 {n} profile{'s' if n != 1 else ''}",
        hover_label="consumption",
        hover_fmt=".3f",
        stat_only=stat_only,
    )


def build_ev_schedule_figure(
    profile_frames: dict[str, pd.DataFrame],
) -> go.Figure:
    """Create a timeline bar chart showing EV plug-in windows with SOC annotations."""
    fig = go.Figure()

    for idx, (label, df) in enumerate(profile_frames.items()):
        color = COLORS[idx % len(COLORS)]
        y_pos = idx  # vertical lane per EV profile

        # Parse plug-in / departure pairs from the event rows
        rows = df.sort_values("minutes").reset_index(drop=True)
        i = 0
        first_bar = True
        while i < len(rows):
            row = rows.iloc[i]
            has_data = pd.notna(row.get("max_cap_kWh"))
            if not has_data:
                i += 1
                continue

            plug_in_min = float(row["minutes"])
            start_soc = row.get("start_soc", np.nan)
            target_soc = row.get("target_soc", np.nan)
            cap_kwh = row.get("max_cap_kWh", np.nan)
            charge_kw = row.get("max_charging_kW", np.nan)
            duration_h = row.get("target_soc_reach_duration_h", np.nan)

            # Find departure: next row without data, or end of day
            depart_min = 1440.0
            if i + 1 < len(rows):
                next_row = rows.iloc[i + 1]
                if pd.isna(next_row.get("max_cap_kWh")):
                    depart_min = float(next_row["minutes"])
                    i += 1  # skip the departure row

            mid_min = (plug_in_min + depart_min) / 2.0
            hhmm_in = f"{int(plug_in_min) // 60:02d}:{int(plug_in_min) % 60:02d}"
            hhmm_out = f"{int(depart_min) // 60:02d}:{int(depart_min) % 60:02d}"

            soc_text = ""
            if pd.notna(start_soc) and pd.notna(target_soc):
                soc_text = f"SOC {start_soc:.0%}\u2192{target_soc:.0%}"

            hover_parts = [
                f"<b>{label}</b>",
                f"Plug-in: {hhmm_in}  Departure: {hhmm_out}",
            ]
            if pd.notna(cap_kwh):
                hover_parts.append(f"Capacity: {cap_kwh:.0f} kWh")
            if pd.notna(charge_kw):
                hover_parts.append(f"Max charge: {charge_kw:.1f} kW")
            if soc_text:
                hover_parts.append(soc_text)
            if pd.notna(duration_h):
                hover_parts.append(f"Target duration: {duration_h:.0f} h")

            fig.add_trace(go.Scatter(
                x=[plug_in_min, depart_min],
                y=[y_pos, y_pos],
                mode="lines",
                line=dict(color=color, width=16),
                name=label,
                legendgroup=label,
                showlegend=first_bar,
                hovertemplate="<br>".join(hover_parts) + "<extra></extra>",
            ))
            first_bar = False

            if soc_text:
                fig.add_annotation(
                    x=mid_min, y=y_pos,
                    text=soc_text,
                    showarrow=False,
                    font=dict(size=10, color="white"),
                    yshift=0,
                )

            i += 1

    ev_labels = list(profile_frames.keys())
    apply_day_xaxis(fig)
    fig.update_yaxes(
        tickvals=list(range(len(ev_labels))),
        ticktext=ev_labels,
        title_text="EV profile",
    )

    n = len(ev_labels)
    title = f"EV charging schedule \u2014 {n} profile{'s' if n != 1 else ''}"
    height = max(250, 80 + 50 * n)
    finalize_figure(fig, title, height=height)
    return fig
