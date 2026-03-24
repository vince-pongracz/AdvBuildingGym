"""Plot weather and energy-price data for one or more dates.

Each day is overlaid as a separate trace in the same figure so that days
can be compared visually.  For multi-day plots an average curve with a
+/- 1 std-dev band is added automatically.

Usage:
    # Single day
    python -m plotting.data_plotting.plot_day_data 2020-07-15

    # Multiple explicit dates
    python -m plotting.data_plotting.plot_day_data 2020-07-15 2020-08-01 2021-01-10

    # Start date + N consecutive days
    python -m plotting.data_plotting.plot_day_data 2020-07-15 --days 7

    # Show only the statistical summary (mean + std band), hide individual days
    python -m plotting.data_plotting.plot_day_data 2020-07-15 --days 7 --stat

    # Mixed options (--days is ignored when multiple dates are given)
    python -m plotting.data_plotting.plot_day_data 2020-07-15 --days 3 --format html png
    python -m plotting.data_plotting.plot_day_data 2020-07-15 --config plotting/config/data_plot_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

from plotting.utils import (
    COLORS,
    apply_day_xaxis,
    ensure_chrome_for_kaleido,
    style_figure,
    write_figure_list_html,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "data_plot_config.yaml"

_STAT_COLOR = "black"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve_source(cfg_section: dict, section_name: str) -> dict:
    """Resolve a ``use`` selector inside a config section.

    The section is expected to contain a ``use`` key naming the active
    dataset, plus one sub-dict per dataset option.  Returns the sub-dict
    for the selected dataset.  Falls back to the flat layout (dir,
    file_pattern, timestamp_col at the top level) for backwards
    compatibility.
    """
    if "use" not in cfg_section:
        return cfg_section

    key = cfg_section["use"]
    if key not in cfg_section:
        available = [k for k in cfg_section if k != "use"]
        raise KeyError(
            f"{section_name}.use = {key!r} but available datasets are: "
            + ", ".join(available)
        )
    return cfg_section[key]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_day_csv(
    directory: Path,
    file_pattern: str,
    timestamp_col: str,
    date: datetime,
) -> pd.DataFrame:
    """Load CSV data for a single calendar day.

    Returns an empty DataFrame if the file does not exist or contains no
    data for the requested day.
    """
    year = date.year
    filename = file_pattern.format(year=year)
    filepath = directory / filename
    if not filepath.exists():
        logger.warning("File not found: %s", filepath)
        return pd.DataFrame()

    df = pd.read_csv(filepath, parse_dates=[timestamp_col])
    if df[timestamp_col].dt.tz is not None:
        df[timestamp_col] = df[timestamp_col].dt.tz_localize(None)

    day_start = pd.Timestamp(date)
    day_end = day_start + pd.Timedelta(days=1)
    mask = (df[timestamp_col] >= day_start) & (df[timestamp_col] < day_end)
    day_df = df.loc[mask].copy()

    if day_df.empty:
        logger.warning("No data for %s in %s", date.date(), filepath)
        return pd.DataFrame()

    day_df["minutes"] = (
        (day_df[timestamp_col] - day_start).dt.total_seconds() / 60.0
    ).astype(np.float32)

    return day_df


def _load_days(
    directory: Path,
    file_pattern: str,
    timestamp_col: str,
    dates: list[datetime],
) -> dict[str, pd.DataFrame]:
    """Load CSV data for multiple days. Returns {date_label: DataFrame}."""
    result: dict[str, pd.DataFrame] = {}
    for date in dates:
        df = _load_day_csv(directory, file_pattern, timestamp_col, date)
        if not df.empty:
            result[str(date.date())] = df
    return result


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

@dataclass
class _ColumnStats:
    """Per-timestep statistics for a single column across multiple days."""
    minutes: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    vmin: np.ndarray
    vmax: np.ndarray


def _compute_column_stats(
    day_frames: dict[str, pd.DataFrame],
    col: str,
) -> _ColumnStats | None:
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
    return _ColumnStats(
        minutes=minute_index.values.astype(np.float32),
        mean=np.nanmean(aligned, axis=1).astype(np.float32),
        std=np.nanstd(aligned, axis=1).astype(np.float32),
        vmin=np.nanmin(aligned, axis=1).astype(np.float32),
        vmax=np.nanmax(aligned, axis=1).astype(np.float32),
    )


_BAND_COLOR_MINMAX = "rgba(0,0,0,0.06)"
_BAND_COLOR_STD = "rgba(0,0,0,0.15)"


def _add_stat_traces(
    fig: go.Figure,
    stats: _ColumnStats,
    value_label: str,
    value_fmt: str = ".1f",
) -> None:
    """Add mean line, ±1 std band (grey), and min–max band (lighter grey)."""
    minutes = stats.minutes
    hhmm = [f"{int(m) // 60:02d}:{int(m) % 60:02d}" for m in minutes]

    # --- min–max band (outer, lighter) ---
    fig.add_trace(go.Scatter(
        x=minutes, y=stats.vmax,
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=minutes, y=stats.vmin,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=_BAND_COLOR_MINMAX,
        name="min\u2013max", showlegend=True, hoverinfo="skip",
    ))

    # --- ±1 std band (inner, darker) ---
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
        fill="tonexty", fillcolor=_BAND_COLOR_STD,
        name="\u00b11 std", showlegend=True, hoverinfo="skip",
    ))

    # --- mean line ---
    fig.add_trace(go.Scatter(
        x=minutes, y=stats.mean,
        mode="lines", name="mean",
        line=dict(color=_STAT_COLOR, width=1.5),
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


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

# Weather columns to plot: (column, y-axis label, fallback column)
_WEATHER_COLS = [
    ("temp_amb", "Temperature (°C)", None),
    ("rel_humidity", "Relative humidity (%)", None),
    ("avg_wind_speed", "Wind speed (m/s)", None),
    ("sun_shine", "Global irradiance (J/cm²)", "direct_sun_shine"),
]


def _resolve_weather_cols(sample_df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return (column_name, label) pairs available in the data."""
    available = []
    for col, label, alt in _WEATHER_COLS:
        if col in sample_df.columns:
            available.append((col, label))
        elif alt and alt in sample_df.columns:
            available.append((alt, label))
    return available


def _days_subtitle(day_labels: list[str]) -> str:
    """Build a subtitle listing the aggregated days."""
    return "<br><sub>Days: " + ", ".join(day_labels) + "</sub>"


def _build_weather_figures(
    day_frames: dict[str, pd.DataFrame],
    stat_only: bool = False,
) -> list[go.Figure]:
    """Create one standalone figure per weather variable, each with all days overlaid."""
    sample_df = next(iter(day_frames.values()))
    available = _resolve_weather_cols(sample_df)
    if not available:
        logger.warning("No plottable weather columns found.")
        return []

    is_multi = len(day_frames) > 1
    day_labels = list(day_frames.keys())
    figures: list[go.Figure] = []

    for col, label in available:
        fig = go.Figure()

        if not (is_multi and stat_only):
            for day_idx, (day_label, df) in enumerate(day_frames.items()):
                if col not in df.columns:
                    continue
                color = COLORS[day_idx % len(COLORS)]
                minutes = df["minutes"]
                hhmm = [f"{int(m) // 60:02d}:{int(m) % 60:02d}" for m in minutes]

                fig.add_trace(
                    go.Scatter(
                        x=minutes,
                        y=df[col],
                        mode="lines",
                        name=day_label,
                        line=dict(color=color, width=1.5),
                        customdata=np.column_stack([hhmm, [day_label] * len(minutes)]),
                        hovertemplate=(
                            "<b>%{customdata[1]}</b> %{customdata[0]}<br>"
                            f"{col}: " + "%{y:.1f}"
                            "<extra></extra>"
                        ),
                    )
                )

        if is_multi:
            stats = _compute_column_stats(day_frames, col)
            if stats is not None:
                _add_stat_traces(fig, stats, value_label=col)

        apply_day_xaxis(fig)
        fig.update_yaxes(title_text=label)
        title = label
        if is_multi and stat_only:
            title += _days_subtitle(day_labels)
        fig.update_layout(
            title_text=title,
            height=350,
            showlegend=True,
        )
        figures.append(style_figure(fig))

    return figures


def _build_price_figure(
    day_frames: dict[str, pd.DataFrame],
    stat_only: bool = False,
) -> go.Figure:
    """Create a single-panel figure with one price trace per day."""
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

            fig.add_trace(
                go.Scatter(
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
                )
            )

    if is_multi:
        stats = _compute_column_stats(day_frames, "baseprice")
        if stats is not None:
            _add_stat_traces(fig, stats, value_label="price", value_fmt=".2f")

    apply_day_xaxis(fig)
    fig.update_yaxes(title_text="Energy price (ct/kWh)")
    title = f"Energy price — {day_labels[0]}" if not is_multi else f"Energy price — {len(day_labels)} days"
    if is_multi and stat_only:
        title += _days_subtitle(day_labels)
    fig.update_layout(
        title_text=title,
        height=350,
        showlegend=True,
    )
    return style_figure(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_dates(date_strings: list[str], n_days: int) -> list[datetime]:
    """Build a sorted, deduplicated list of dates from CLI arguments.

    When a single date is given, ``n_days`` consecutive days starting from
    that date are used (default 1 = just that day).  When multiple dates
    are given explicitly, ``--days`` is ignored.
    """
    dates: list[datetime] = []
    for s in date_strings:
        try:
            dates.append(datetime.strptime(s, "%Y-%m-%d"))
        except ValueError:
            logger.error("Invalid date format: %r. Use YYYY-MM-DD.", s)
            sys.exit(1)

    if len(dates) == 1 and n_days > 1:
        start = dates[0]
        dates = [start + timedelta(days=i) for i in range(n_days)]

    dates.sort()
    return dates


def _make_base_name(dates: list[datetime], n_consecutive: int | None, stat_only: bool) -> str:
    """Build output file base name.

    - Single day:       day_YYYYMMDD
    - Consecutive days: days_YYYYMMDD_nN[_stat]
    - Explicit list:    days_YYYYMMDD_YYYYMMDD_...[_stat]
    """
    fmt = "%Y%m%d"
    if len(dates) == 1:
        return f"day_{dates[0].strftime(fmt)}"
    if n_consecutive is not None:
        name = f"days_{dates[0].strftime(fmt)}_n{n_consecutive}"
    else:
        name = "days_" + "_".join(d.strftime(fmt) for d in dates)
    if stat_only:
        name += "_stat"
    return name


def plot_days(
    dates: list[datetime],
    config_path: Path = _DEFAULT_CONFIG,
    output_formats: list[str] | None = None,
    n_consecutive: int | None = None,
    stat_only: bool = False,
) -> None:
    """Load data for *dates* and write combined weather + price plots."""
    cfg = _load_config(config_path)
    output_formats = output_formats or ["html"]

    weather_cfg = _resolve_source(cfg["weather"], "weather")
    price_cfg = _resolve_source(cfg["price"], "price")

    weather_dir = _REPO_ROOT / weather_cfg["dir"]
    price_dir = _REPO_ROOT / price_cfg["dir"]

    weather_frames = _load_days(
        weather_dir,
        weather_cfg["file_pattern"],
        weather_cfg["timestamp_col"],
        dates,
    )
    price_frames = _load_days(
        price_dir,
        price_cfg["file_pattern"],
        price_cfg["timestamp_col"],
        dates,
    )

    if not weather_frames and not price_frames:
        logger.error("No data found for any requested date. Nothing to plot.")
        sys.exit(1)

    figures: list[go.Figure] = []
    if weather_frames:
        figures.extend(_build_weather_figures(weather_frames, stat_only=stat_only))
    if price_frames:
        figures.append(_build_price_figure(price_frames, stat_only=stat_only))

    # Output
    out_dir = _REPO_ROOT / cfg["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = _make_base_name(dates, n_consecutive, stat_only)

    if "html" in output_formats:
        html_path = out_dir / f"{base_name}.html"
        write_figure_list_html(figures, str(html_path))
        logger.info("Wrote %s", html_path)

    static_formats = [fmt for fmt in output_formats if fmt != "html"]
    if static_formats:
        ensure_chrome_for_kaleido()
        for fmt in static_formats:
            for i, fig in enumerate(figures):
                title = fig.layout.title.text or f"fig{i}"
                tag = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
                img_path = out_dir / f"{base_name}_{tag}.{fmt}"
                fig.write_image(str(img_path))
                logger.info("Wrote %s", img_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot weather and energy-price data for one or more dates.",
    )
    parser.add_argument(
        "dates",
        nargs="+",
        type=str,
        help="One or more dates (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        metavar="N",
        help="Plot N consecutive days starting from the first date. "
             "Ignored when multiple dates are given explicitly. Default: 1.",
    )
    parser.add_argument(
        "--stat",
        action="store_true",
        default=False,
        help="Show only the mean curve and +/- 1 std band (hide individual day traces). "
             "Only effective for multi-day plots.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_DEFAULT_CONFIG),
        help="Path to data_plot_config.yaml (default: %(default)s).",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        default=["html"],
        choices=["html", "png", "svg", "pdf"],
        help="Output format(s). Default: html.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dates = _resolve_dates(args.dates, args.days)
    if not dates:
        parser.error("No valid dates provided.")

    # Determine whether this is a consecutive-days invocation
    is_consecutive = len(args.dates) == 1 and args.days > 1
    n_consecutive = args.days if is_consecutive else None

    plot_days(
        dates,
        config_path=Path(args.config),
        output_formats=args.format,
        n_consecutive=n_consecutive,
        stat_only=args.stat,
    )


if __name__ == "__main__":
    main()
