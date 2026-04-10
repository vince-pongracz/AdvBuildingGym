"""Statistical overlay traces for day-data plots.

Provides ``compute_column_stats`` to aggregate a column across multiple
day-DataFrames onto a common minute grid, and ``add_stat_traces`` to
render the result as a mean line with +/-1 std and min-max bands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

STAT_COLOR = "black"
BAND_COLOR_MINMAX = "rgba(0,0,0,0.06)"
BAND_COLOR_STD = "rgba(0,0,0,0.15)"


@dataclass
class ColumnStats:
    """Per-timestep statistics for a single column across multiple days."""

    minutes: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    vmin: np.ndarray
    vmax: np.ndarray


def compute_column_stats(
    day_frames: dict[str, pd.DataFrame],
    col: str,
) -> ColumnStats | None:
    """Compute mean, std, min, and max for *col* across days on a common minute grid."""
    arrays: list[pd.Series] = []
    minute_index: pd.Index | None = None
    for df in day_frames.values():
        if col not in df.columns:
            continue
        s = pd.Series(df[col].values, index=df["minutes"].values, dtype=np.float64)
        arrays.append(s)
        if minute_index is None or len(df) > len(minute_index):
            minute_index = pd.Index(df["minutes"].values)

    if not arrays or minute_index is None:
        return None

    aligned = np.column_stack([
        s.reindex(minute_index, method="nearest", tolerance=3.0).values
        for s in arrays
    ])
    return ColumnStats(
        minutes=minute_index.values.astype(np.float32),
        mean=np.nanmean(aligned, axis=1).astype(np.float32),
        std=np.nanstd(aligned, axis=1).astype(np.float32),
        vmin=np.nanmin(aligned, axis=1).astype(np.float32),
        vmax=np.nanmax(aligned, axis=1).astype(np.float32),
    )


def add_stat_traces(
    fig: go.Figure,
    stats: ColumnStats,
    value_label: str,
    value_fmt: str = ".1f",
) -> None:
    """Add mean line, +/-1 std band (grey), and min-max band (lighter grey)."""
    minutes = stats.minutes
    hhmm = [f"{int(m) // 60:02d}:{int(m) % 60:02d}" for m in minutes]

    # --- min-max band (outer, lighter) ---
    fig.add_trace(go.Scatter(
        x=minutes, y=stats.vmax,
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=minutes, y=stats.vmin,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=BAND_COLOR_MINMAX,
        name="min\u2013max", showlegend=True, hoverinfo="skip",
    ))

    # --- +/-1 std band (inner, darker) ---
    upper_std = stats.mean + stats.std
    lower_std = stats.mean - stats.std
    fig.add_trace(go.Scatter(
        x=minutes, y=upper_std,
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=minutes, y=lower_std,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=BAND_COLOR_STD,
        name="\u00b11 std", showlegend=True, hoverinfo="skip",
    ))

    # --- mean line ---
    fig.add_trace(go.Scatter(
        x=minutes, y=stats.mean,
        mode="lines", name="mean",
        line=dict(color=STAT_COLOR, width=1.5),
        customdata=np.column_stack([
            hhmm,
            [f"{v:{value_fmt}}" for v in stats.vmin],
            [f"{v:{value_fmt}}" for v in stats.vmax],
        ]),
        hovertemplate=(
            "<b>mean</b> %{customdata[0]}<br>"
            f"{value_label}: " + "%{y:" + value_fmt + "}"
            " [%{customdata[1]} .. %{customdata[2]}]"
            "<extra></extra>"
        ),
    ))
