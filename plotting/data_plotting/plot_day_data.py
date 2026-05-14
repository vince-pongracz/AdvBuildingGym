"""Plot weather, energy-price, and profile data for one or more dates.

Each day is overlaid as a separate trace in the same figure so that days
can be compared visually.  For multi-day plots an average curve with a
+/- 1 std-dev band is added automatically.

Profile data sources (desired_temp_in, ev_schedule, user_energy_need)
overlay all available profiles in a single figure regardless of CLI dates.

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
from datetime import datetime, timedelta
from pathlib import Path

import plotly.graph_objects as go

from .common import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    add_shared_cli_args,
    apply_shared_cli_args,
    load_config,
    load_profiles_from_cfg,
    resolve_source,
    write_output,
)
from .figure_builders import (
    build_desired_temp_figure,
    build_ev_schedule_figure,
    build_price_figure,
    build_user_energy_need_figure,
    build_weather_figures,
)
from .loaders import load_days, load_syn_cfg_days

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Date helpers
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


# ---------------------------------------------------------------------------
# Data loading orchestration
# ---------------------------------------------------------------------------

def _load_all_sources(
    cfg: dict,
    dates: list[datetime],
) -> dict[str, dict[str, object]]:
    """Load every configured data source. Returns a dict of source name to frames."""
    sources: dict[str, dict[str, object]] = {}

    # Year-partitioned sources (weather, price)
    weather_cfg = resolve_source(cfg["weather"], "weather")
    weather_dir = REPO_ROOT / weather_cfg["dir"]
    sources["weather"] = load_days(
        weather_dir,
        weather_cfg["file_pattern"],
        weather_cfg["timestamp_col"],
        dates,
    )
    sources["weather_syn"] = load_syn_cfg_days(
        weather_dir,
        weather_cfg["file_pattern"],
        weather_cfg["timestamp_col"],
        dates,
    )

    price_cfg = resolve_source(cfg["price"], "price")
    price_dir = REPO_ROOT / price_cfg["dir"]
    sources["price"] = load_days(
        price_dir,
        price_cfg["file_pattern"],
        price_cfg["timestamp_col"],
        dates,
    )
    sources["price_syn"] = load_syn_cfg_days(
        price_dir,
        price_cfg["file_pattern"],
        price_cfg["timestamp_col"],
        dates,
    )

    # Profile sources (desired_temp_in, ev_schedule, user_energy_need)
    sources.update(load_profiles_from_cfg(cfg, dates))

    return sources


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def _build_figures(
    cfg: dict,
    sources: dict[str, dict],
    stat_only: bool,
    y_ranges: dict[str, tuple[float, float]] | None = None,
) -> list[go.Figure]:
    """Build all figures from loaded source data.

    *y_ranges* maps column names (e.g. ``temp_amb``, ``baseprice``) to a
    shared (min, max). When supplied, weather and price figures are drawn
    with that fixed y-axis so figures from different month/year subsets
    are directly comparable.
    """
    figures: list[go.Figure] = []
    y_ranges = y_ranges or {}

    if sources.get("weather"):
        figures.extend(build_weather_figures(
            sources["weather"], stat_only=stat_only, y_ranges=y_ranges,
            syn_frames=sources.get("weather_syn") or None,
        ))

    if sources.get("price"):
        figures.append(build_price_figure(
            sources["price"], stat_only=stat_only, y_range=y_ranges.get("baseprice"),
            syn_frames=sources.get("price_syn") or None,
        ))

    if sources.get("desired_temp_in"):
        figures.append(build_desired_temp_figure(
            sources["desired_temp_in"],
            cfg["desired_temp_in"]["value_col"],
            stat_only=stat_only,
        ))

    if sources.get("user_energy_need"):
        figures.append(build_user_energy_need_figure(
            sources["user_energy_need"],
            cfg["user_energy_need"]["value_col"],
            stat_only=stat_only,
        ))

    if sources.get("ev_schedule"):
        figures.append(build_ev_schedule_figure(sources["ev_schedule"]))

    return figures


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_days(
    dates: list[datetime],
    config_path: Path = DEFAULT_CONFIG,
    output_formats: list[str] | None = None,
    n_consecutive: int | None = None,
    stat_only: bool = False,
) -> None:
    """Load data for *dates* and write combined plots."""
    cfg = load_config(config_path)
    output_formats = output_formats or ["html"]

    sources = _load_all_sources(cfg, dates)

    if not any(sources.values()):
        logger.error("No data found for any requested date. Nothing to plot.")
        sys.exit(1)

    figures = _build_figures(cfg, sources, stat_only)

    out_dir = REPO_ROOT / cfg["output"]["dir"]
    base_name = _make_base_name(dates, n_consecutive, stat_only)
    write_output(figures, out_dir, base_name, output_formats)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot weather, energy-price, and profile data for one or more dates.",
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
    add_shared_cli_args(parser)
    args = parser.parse_args()

    apply_shared_cli_args(args, "plot_day_data")

    dates = _resolve_dates(args.dates, args.days)
    if not dates:
        parser.error("No valid dates provided.")

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
