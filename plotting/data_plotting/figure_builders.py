"""Figure builder functions for day-data plots.

Each public function accepts pre-loaded DataFrames (keyed by trace label)
and returns one or more ``plotly.graph_objects.Figure`` instances ready
for export.  All figures share the same 24-hour x-axis via
``apply_day_xaxis``.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plotting.utils import COLORS, apply_day_xaxis, style_figure

from .common import get_data_figure_config, syn_cfg_style
from .stats import add_stat_traces, compute_column_stats

logger = logging.getLogger(__name__)

# Weather columns to plot: (column, y-axis label, fallback column)
_WEATHER_COLS = [
    ("temp_amb", "Temperature (\u00b0C)", None),
    ("rel_humidity", "Relative humidity (%)", None),
    ("avg_wind_speed", "Wind speed (m/s)", None),
    ("sun_shine", "Global irradiance (W/m\u00b2)", "direct_sun_shine"),
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def finalize_figure(fig: go.Figure, title: str, height: int | None = None) -> None:
    """Apply title, sizing, and shared styling to a figure.

    When *height* is ``None`` the default is read from the ``figure.height``
    key in ``data_plot_config.yaml`` (fallback: 450 px).
    """
    if height is None:
        height = int(get_data_figure_config().get("height", 450))
    fig.update_layout(
        title_text=title, height=height, showlegend=True,
        title=dict(automargin=True, yref="container"),
    )
    style_figure(fig)
    fig.update_layout(margin=dict(t=90))


def _days_subtitle(day_labels: list[str]) -> str:
    """Build a single-line subtitle summarising the aggregated days.

    For ≤ 3 labels the days are listed verbatim (used by plot_day_data.py).
    For larger aggregates a compact summary keeps the subtitle on one line so
    it stays inside the title band reserved by ``style_figure``.
    """
    n = len(day_labels)
    if n == 0:
        return ""
    if n <= 3:
        return "<br><sub>Days: " + ", ".join(day_labels) + "</sub>"

    try:
        parsed = sorted(date.fromisoformat(d) for d in day_labels)
    except ValueError:
        return f"<br><sub>{n} days · {day_labels[0]} … {day_labels[-1]}</sub>"

    first, last = parsed[0], parsed[-1]
    contiguous = (last - first == timedelta(days=n - 1)) and (first.year, first.month) == (last.year, last.month)
    if contiguous:
        return f"<br><sub>{first.isoformat()} → {last.isoformat()} ({n} days)</sub>"
    return f"<br><sub>{n} days · {first.isoformat()} … {last.isoformat()}</sub>"


def _add_day_traces(
    fig: go.Figure,
    day_frames: dict[str, pd.DataFrame],
    col: str,
    hover_label: str,
    hover_fmt: str = ".1f",
    *,
    color: str | None = None,
    name_prefix: str = "",
    legendgroup: str | None = None,
    line_width: float = 1.5,
) -> None:
    """Add one coloured line trace per day for *col*.

    When *color* is provided every day shares that colour (used by syn_cfg
    overlays so all per-day lines of one cfg are visually grouped); pass a
    non-empty *name_prefix* / *legendgroup* to keep legends tidy.
    """
    for day_idx, (day_label, df) in enumerate(day_frames.items()):
        if col not in df.columns:
            continue
        line_color = color or COLORS[day_idx % len(COLORS)]
        minutes = df["minutes"]
        hhmm = [f"{int(m) // 60:02d}:{int(m) % 60:02d}" for m in minutes]
        trace_name = f"{name_prefix}{day_label}" if name_prefix else day_label
        fig.add_trace(go.Scatter(
            x=minutes,
            y=df[col],
            mode="lines",
            name=trace_name,
            legendgroup=legendgroup,
            line=dict(color=line_color, width=line_width),
            customdata=np.column_stack([hhmm, [trace_name] * len(minutes)]),
            hovertemplate=(
                "<b>%{customdata[1]}</b> %{customdata[0]}<br>"
                f"{hover_label}: " + "%{y:" + hover_fmt + "}"
                "<extra></extra>"
            ),
        ))


def _add_syn_overlays(
    fig: go.Figure,
    syn_frames: dict[str, dict[str, pd.DataFrame]],
    col: str,
    hover_label: str,
    hover_fmt: str,
    *,
    stat_only: bool,
) -> None:
    """Render every syn_cfg's traces on *fig* for column *col*.

    Behaviour matches plot_day_data's modes:
      - Single day per cfg: one solid line per cfg.
      - Multi-day, not stat_only: per-day lines (cfg colour, thinner) plus
        mean + ±1σ + min-max band in the same colour.
      - Multi-day, stat_only: only mean + ±1σ + min-max band.
    """
    for cfg_name, frames in syn_frames.items():
        if not frames:
            continue
        style = syn_cfg_style(cfg_name)
        is_multi = len(frames) > 1

        if not is_multi:
            # Single day: just one solid line per cfg.
            _add_day_traces(
                fig, frames, col, hover_label, hover_fmt,
                color=style["color"], name_prefix=f"{cfg_name} ",
                legendgroup=cfg_name, line_width=1.5,
            )
            continue

        if not stat_only:
            # Per-day lines (faint) so the stat overlay reads on top.
            _add_day_traces(
                fig, frames, col, hover_label, hover_fmt,
                color=style["color"], name_prefix=f"{cfg_name} ",
                legendgroup=cfg_name, line_width=0.8,
            )

        stats = compute_column_stats(frames, col)
        if stats is not None:
            add_stat_traces(
                fig, stats, value_label=hover_label, value_fmt=hover_fmt,
                color=style["color"], band_std=style["rgba_std"],
                band_mm=style["rgba_mm"],
                legend_prefix=cfg_name, legendgroup=cfg_name,
            )


def resolve_weather_cols(sample_df: pd.DataFrame) -> list[tuple[str, str]]:
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
    y_range: tuple[float, float] | None = None,
    syn_frames: dict[str, dict[str, pd.DataFrame]] | None = None,
    height: int | None = None,
) -> go.Figure:
    """Build a single figure with one line trace per entry, plus stat bands.

    This is the common pattern used by weather sub-plots, desired-temperature,
    and household-consumption figures. *y_range* fixes the y-axis to a
    shared (min, max) — used by monthly/cross-year sweeps so figures from
    different month or year subsets are visually comparable.
    """
    fig = go.Figure()
    is_multi = len(day_frames) > 1

    if not (is_multi and stat_only):
        _add_day_traces(fig, day_frames, value_col, hover_label, hover_fmt)

    if is_multi:
        stats = compute_column_stats(day_frames, value_col)
        if stats is not None:
            add_stat_traces(fig, stats, value_label=hover_label, value_fmt=hover_fmt)

    if syn_frames:
        _add_syn_overlays(
            fig, syn_frames, value_col, hover_label, hover_fmt,
            stat_only=stat_only,
        )

    apply_day_xaxis(fig)
    if y_range is not None:
        fig.update_yaxes(title_text=y_label, range=list(y_range))
    else:
        fig.update_yaxes(title_text=y_label)

    if is_multi and stat_only:
        title += _days_subtitle(list(day_frames.keys()))

    finalize_figure(fig, title, height=height)
    return fig


# ---------------------------------------------------------------------------
# Source-specific builders
# ---------------------------------------------------------------------------

def build_weather_figures(
    day_frames: dict[str, pd.DataFrame],
    stat_only: bool = False,
    y_ranges: dict[str, tuple[float, float]] | None = None,
    syn_frames: dict[str, dict[str, pd.DataFrame]] | None = None,
    height: int | None = None,
) -> list[go.Figure]:
    """Create one standalone figure per weather variable, each with all days overlaid.

    *y_ranges* maps column names to (min, max) bounds; when supplied, the
    matching figure's y-axis is fixed to that range. *syn_frames* layers
    synthesised configurations onto every figure that has the column.
    """
    sample_df = next(iter(day_frames.values()))
    available = resolve_weather_cols(sample_df)
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
            y_range=(y_ranges or {}).get(col),
            syn_frames=syn_frames,
            height=height,
        )
        for col, label in available
    ]


def build_price_figure(
    day_frames: dict[str, pd.DataFrame],
    stat_only: bool = False,
    y_range: tuple[float, float] | None = None,
    syn_frames: dict[str, dict[str, pd.DataFrame]] | None = None,
    height: int | None = None,
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

    if syn_frames:
        _add_syn_overlays(
            fig, syn_frames, "baseprice", "price", ".2f",
            stat_only=stat_only,
        )

    apply_day_xaxis(fig)
    if y_range is not None:
        fig.update_yaxes(title_text="Energy price (ct/kWh)", range=list(y_range))
    else:
        fig.update_yaxes(title_text="Energy price (ct/kWh)")
    title = (
        f"Energy price \u2014 {day_labels[0]}"
        if not is_multi
        else f"Energy price \u2014 {len(day_labels)} days"
    )
    if is_multi and stat_only:
        title += _days_subtitle(day_labels)

    finalize_figure(fig, title, height=height)
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
    fig_cfg = get_data_figure_config()
    min_h = int(fig_cfg.get("ev_min_height", 250))
    lane_px = int(fig_cfg.get("ev_lane_px", 50))
    height = max(min_h, 80 + lane_px * n)
    finalize_figure(fig, title, height=height)
    return fig
