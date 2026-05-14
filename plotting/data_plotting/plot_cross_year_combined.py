"""Cross-year monthly plots with all datasets overlaid in a single figure.

For each of the 12 calendar months, collects every day across 2016-2026
and produces figures where each weather/price dataset appears as its own
stat band (mean +/- 1 std + min-max), colour-coded by dataset.

Weather figures: one per weather variable, with stat bands per weather
dataset (dwd, zenodo).  All price datasets feed into a single price
figure with one stat band per price source (awattar, e_charts).

Profile data (desired_temp_in, ev_schedule, user_energy_need) is
overlaid as before (date-independent).

Usage:
    python -m plotting.data_plotting.plot_cross_year_combined

    python -m plotting.data_plotting.plot_cross_year_combined --format html png

    python -m plotting.data_plotting.plot_cross_year_combined --start-year 2020 --end-year 2024
"""

from __future__ import annotations

import argparse
import calendar
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plotting.utils import apply_day_xaxis

from .common import (
    DATASET_STYLES,
    DEFAULT_CONFIG,
    PRICE_DATASETS,
    REPO_ROOT,
    WEATHER_DATASETS,
    add_shared_cli_args,
    apply_shared_cli_args,
    days_in_month,
    load_config,
    load_profiles_from_cfg,
    resolve_source,
    syn_cfg_style,
    write_output,
)
from .figure_builders import (
    resolve_weather_cols,
    build_desired_temp_figure,
    build_ev_schedule_figure,
    build_user_energy_need_figure,
    finalize_figure,
)
from .loaders import load_days, load_syn_cfg_days
from .stats import ColumnStats, compute_column_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multi-dataset stat band rendering
# ---------------------------------------------------------------------------

def _add_dataset_stat_traces(
    fig: go.Figure,
    stats: ColumnStats,
    dataset_name: str,
    value_label: str,
    value_fmt: str = ".1f",
) -> None:
    """Add mean line + std band + min-max band for one dataset, colour-coded."""
    # Synthesised variants are keyed "<base>:<cfg_name>" and get a
    # syn_cfg-palette colour so they stand apart from the base dataset.
    if dataset_name in DATASET_STYLES:
        style = DATASET_STYLES[dataset_name]
    elif ":" in dataset_name:
        style = syn_cfg_style(dataset_name.split(":", 1)[1])
    else:
        style = {
            "color": "#888",
            "rgba_std": "rgba(136,136,136,0.20)",
            "rgba_mm": "rgba(136,136,136,0.08)",
        }
    minutes = stats.minutes
    hhmm = [f"{int(m) // 60:02d}:{int(m) % 60:02d}" for m in minutes]

    # --- min-max band (outer, lighter) ---
    fig.add_trace(go.Scatter(
        x=minutes, y=stats.vmax,
        mode="lines", line=dict(width=0),
        legendgroup=dataset_name, showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=minutes, y=stats.vmin,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=style["rgba_mm"],
        legendgroup=dataset_name,
        name=f"{dataset_name} min\u2013max",
        showlegend=True, hoverinfo="skip",
    ))

    # --- +/-1 std band ---
    upper_std = stats.mean + stats.std
    lower_std = stats.mean - stats.std
    fig.add_trace(go.Scatter(
        x=minutes, y=upper_std,
        mode="lines", line=dict(width=0),
        legendgroup=dataset_name, showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=minutes, y=lower_std,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=style["rgba_std"],
        legendgroup=dataset_name,
        name=f"{dataset_name} \u00b11 std",
        showlegend=True, hoverinfo="skip",
    ))

    # --- mean line ---
    fig.add_trace(go.Scatter(
        x=minutes, y=stats.mean,
        mode="lines",
        legendgroup=dataset_name,
        name=f"{dataset_name} mean",
        line=dict(color=style["color"], width=2),
        customdata=np.column_stack([
            hhmm,
            [f"{v:{value_fmt}}" for v in stats.vmin],
            [f"{v:{value_fmt}}" for v in stats.vmax],
        ]),
        hovertemplate=(
            f"<b>{dataset_name} mean</b> " + "%{customdata[0]}<br>"
            f"{value_label}: " + "%{y:" + value_fmt + "}"
            " [%{customdata[1]} .. %{customdata[2]}]"
            "<extra></extra>"
        ),
    ))


# ---------------------------------------------------------------------------
# Combined figure builders
# ---------------------------------------------------------------------------

def _build_combined_weather_figures(
    datasets: dict[str, dict[str, object]],
    month_label: str,
    y_ranges: dict[str, tuple[float, float]] | None = None,
    height: int | None = None,
) -> list[go.Figure]:
    """Build one figure per weather variable with stat bands from each dataset.

    Resolves columns per-dataset so that equivalent variables with different
    column names (e.g. DWD ``sun_shine`` vs Zenodo ``direct_sun_shine``) are
    grouped under the same y-axis label.
    """
    # Resolve columns per dataset: {ds_name: [(col, label), ...]}
    ds_cols: dict[str, list[tuple[str, str]]] = {}
    for ds_name, day_frames in datasets.items():
        if not day_frames:
            continue
        sample = next(iter(day_frames.values()))
        ds_cols[ds_name] = resolve_weather_cols(sample)
    if not ds_cols:
        return []

    # Collect the union of y-axis labels in stable order
    seen_labels: set[str] = set()
    ordered_labels: list[str] = []
    for cols in ds_cols.values():
        for _, label in cols:
            if label not in seen_labels:
                seen_labels.add(label)
                ordered_labels.append(label)

    figures: list[go.Figure] = []
    for label in ordered_labels:
        fig = go.Figure()
        has_data = False
        for ds_name, cols in ds_cols.items():
            # Find the column name this dataset uses for this label
            col = next((c for c, lbl in cols if lbl == label), None)
            if col is None:
                continue
            stats = compute_column_stats(datasets[ds_name], col)
            if stats is not None:
                _add_dataset_stat_traces(fig, stats, ds_name, col)
                has_data = True

        if not has_data:
            continue

        apply_day_xaxis(fig)
        # Pick a y-range matching any column that resolves to this label.
        rng = None
        if y_ranges:
            for cols in ds_cols.values():
                col = next((c for c, lbl in cols if lbl == label), None)
                if col and col in y_ranges:
                    rng = y_ranges[col]
                    break
        if rng is not None:
            fig.update_yaxes(title_text=label, range=list(rng))
        else:
            fig.update_yaxes(title_text=label)
        title = f"{label} \u2014 {month_label} (all datasets)"
        finalize_figure(fig, title, height=height)
        figures.append(fig)

    return figures


def _build_combined_price_figure(
    datasets: dict[str, dict[str, object]],
    month_label: str,
    y_range: tuple[float, float] | None = None,
    height: int | None = None,
) -> go.Figure | None:
    """Build one price figure with stat bands from each price dataset."""
    fig = go.Figure()
    has_data = False
    for ds_name, day_frames in datasets.items():
        if not day_frames:
            continue
        stats = compute_column_stats(day_frames, "baseprice")
        if stats is not None:
            _add_dataset_stat_traces(fig, stats, ds_name, "price", ".2f")
            has_data = True

    if not has_data:
        return None

    apply_day_xaxis(fig)
    if y_range is not None:
        fig.update_yaxes(title_text="Energy price (ct/kWh)", range=list(y_range))
    else:
        fig.update_yaxes(title_text="Energy price (ct/kWh)")
    title = f"Energy price \u2014 {month_label} (all datasets)"
    finalize_figure(fig, title, height=height)
    return fig


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_multi_datasets(
    cfg_section: dict,
    section_name: str,
    dataset_keys: list[str],
    dates: list[datetime],
) -> dict[str, dict[str, object]]:
    """Load data from every dataset variant in *dataset_keys*.

    Additionally globs sibling ``*_syn_cfg_*.csv`` files for each dataset
    and adds them as pseudo-datasets keyed ``"<base>:<cfg_name>"`` so the
    existing per-dataset stat-band machinery renders them alongside the
    originals in a distinct colour.
    """
    result: dict[str, dict[str, object]] = {}
    for ds_key in dataset_keys:
        src = resolve_source(cfg_section, section_name, key=ds_key)
        if src is None:
            continue
        ds_dir = REPO_ROOT / src["dir"]
        frames = load_days(
            ds_dir, src["file_pattern"], src["timestamp_col"], dates,
        )
        if frames:
            result[ds_key] = frames
        syn_frames = load_syn_cfg_days(ds_dir, src["file_pattern"], src["timestamp_col"], dates)
        for cfg_name, cfg_frames in syn_frames.items():
            result[f"{ds_key}:{cfg_name}"] = cfg_frames
    return result


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def _aggregate_y_ranges(datasets: dict[str, dict[str, object]]) -> dict[str, tuple[float, float]]:
    """Compute global (min, max) per numeric column across all loaded datasets."""
    per_col: dict[str, list[np.ndarray]] = {}
    for frames in datasets.values():
        for df in frames.values():
            for col in df.columns:
                if col == "minutes" or not pd.api.types.is_numeric_dtype(df[col]):
                    continue

                arr = df[col].to_numpy(dtype=float)
                arr = arr[np.isfinite(arr)] # Removes nan and similars
                if arr.size:
                    per_col.setdefault(col, []).append(arr)
    # TODO VP 2026.05.03. : This could be performance critical...
    return {
        col: (float(np.concatenate(arrs).min()), float(np.concatenate(arrs).max()))
        for col, arrs in per_col.items()
    }


def run_combined(
    config_path: Path = DEFAULT_CONFIG,
    output_formats: list[str] | None = None,
    start_year: int = 2016,
    end_year: int = 2026,
) -> None:
    """Generate combined cross-year plots (all datasets per figure)."""
    output_formats = output_formats or ["html"]
    cfg = load_config(config_path)
    out_dir = REPO_ROOT / cfg["output"]["dir"] / "cross_year_combined"

    # Pre-pass: compute global y-ranges across every month so each figure
    # uses a consistent y-axis (overall min..max) regardless of which
    # calendar month it covers.
    full_dates: list[datetime] = []
    for m in range(1, 13):
        for y in range(start_year, end_year + 1):
            full_dates.extend(days_in_month(y, m))
    full_weather = _load_multi_datasets(cfg["weather"], "weather", WEATHER_DATASETS, full_dates)
    full_price = _load_multi_datasets(cfg["price"], "price", PRICE_DATASETS, full_dates)
    weather_y_ranges = _aggregate_y_ranges(full_weather)
    price_y_ranges = _aggregate_y_ranges(full_price)
    price_range = price_y_ranges.get("baseprice")

    def _slice_by_month(
        data: dict[str, dict[str, object]], labels: set[str],
    ) -> dict[str, dict[str, object]]:
        return {
            ds: {
                k: v 
                for k, v in frames.items() if k in labels
            }
            for ds, frames in data.items()
        }

    for month in range(1, 13):
        month_name = calendar.month_name[month]
        month_abbr = calendar.month_abbr[month]
        logger.info("=== %s ===", month_name)

        # Collect all days of this calendar month across all years
        all_dates: list[datetime] = []
        for year in range(start_year, end_year + 1):
            all_dates.extend(days_in_month(year, month))
        labels = {str(date.date()) for date in all_dates}

        # Slice pre-loaded weather/price data; reload date-independent profiles.
        weather_data = {ds: f for ds, f in _slice_by_month(full_weather, labels).items() if f}
        price_data = {ds: f for ds, f in _slice_by_month(full_price, labels).items() if f}
        profile_data = load_profiles_from_cfg(cfg, all_dates)

        if not weather_data and not price_data:
            logger.info("No weather or price data for %s — skipping.", month_name)
            continue

        # Build figures
        figures: list[go.Figure] = []
        month_label = f"{month_name} ({start_year}\u2013{end_year})"

        height = cfg.get("figure", {}).get("overview_height")
        if weather_data:
            figures.extend(_build_combined_weather_figures(
                weather_data, month_label, y_ranges=weather_y_ranges, height=height,
            ))

        if price_data:
            price_fig = _build_combined_price_figure(
                price_data, month_label, y_range=price_range, height=height,
            )
            if price_fig is not None:
                figures.append(price_fig)

        # Profile figures (not dataset-dependent)
        if profile_data.get("desired_temp_in"):
            figures.append(build_desired_temp_figure(
                profile_data["desired_temp_in"],
                cfg["desired_temp_in"]["value_col"],
                stat_only=True,
            ))

        if profile_data.get("user_energy_need"):
            figures.append(build_user_energy_need_figure(
                profile_data["user_energy_need"],
                cfg["user_energy_need"]["value_col"],
                stat_only=True,
            ))

        if profile_data.get("ev_schedule"):
            figures.append(build_ev_schedule_figure(profile_data["ev_schedule"]))

        if not figures:
            logger.info("No figures for %s — skipping.", month_name)
            continue

        base_name = f"all_years_{month:02d}_{month_abbr}_combined"
        write_output(figures, out_dir, base_name, output_formats)

    logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-year monthly plots with all weather/price datasets "
                    "overlaid in a single figure.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2016,
        help="First year to include (default: 2016).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2026,
        help="Last year to include (default: 2026).",
    )
    add_shared_cli_args(parser)
    args = parser.parse_args()

    apply_shared_cli_args(args, "plot_cross_year_combined")

    run_combined(
        config_path=Path(args.config),
        output_formats=args.format,
        start_year=args.start_year,
        end_year=args.end_year,
    )


if __name__ == "__main__":
    main()
