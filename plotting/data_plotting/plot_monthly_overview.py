"""Generate month-level day-data plots for years 2016-2026.

Iterates over all year-month combinations and all weather/price dataset
variants, producing two kinds of output:

1. **Per year-month**: one plot per (year, month, dataset-combo) covering
   every day in that month with ``--stat`` (mean + std band).
   ~132 files per dataset combo.

2. **Per calendar month across all years**: one plot per (month, dataset-combo)
   that collects *every* occurrence of that month across 2016-2026 (up to ~341
   days for a 31-day month over 11 years).

Dataset combos are the cartesian product of weather sources x price sources
defined in the plot config.

Usage:
    # Default: html output
    python -m plotting.data_plotting.plot_monthly_overview

    # Also produce png
    python -m plotting.data_plotting.plot_monthly_overview --format html png

    # Restrict year range
    python -m plotting.data_plotting.plot_monthly_overview --start-year 2020 --end-year 2024

    # Use specific config
    python -m plotting.data_plotting.plot_monthly_overview --config plotting/config/data_plot_config.yaml
"""

from __future__ import annotations

import argparse
import calendar
import copy
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    DEFAULT_CONFIG,
    PRICE_DATASETS,
    REPO_ROOT,
    WEATHER_DATASETS,
    add_shared_cli_args,
    apply_shared_cli_args,
    days_in_month,
    load_config,
    write_output,
)
from .plot_day_data import _build_figures, _load_all_sources

logger = logging.getLogger(__name__)


def _make_config_variant(
    cfg: dict,
    weather_key: str,
    price_key: str,
) -> dict | None:
    """Return a deep-copied config with the given weather/price selectors.

    Returns ``None`` if the selected dataset is not present in the config.
    """
    cfg = copy.deepcopy(cfg)

    if weather_key not in cfg["weather"]:
        logger.debug("Weather dataset %r not in config, skipping.", weather_key)
        return None
    cfg["weather"]["use"] = weather_key

    if price_key not in cfg["price"]:
        logger.debug("Price dataset %r not in config, skipping.", price_key)
        return None
    cfg["price"]["use"] = price_key

    return cfg


_DATE_KEYED_SOURCES = ("weather", "price")


def _slice_sources(
    sources: dict[str, dict],
    date_labels: set[str],
) -> dict[str, dict]:
    """Filter year-partitioned sources to a subset of date labels.

    Profile sources (desired_temp_in, ev_schedule, user_energy_need) are
    date-independent and passed through unchanged.
    """
    sliced: dict[str, dict] = {}
    for name, frames in sources.items():
        if name in _DATE_KEYED_SOURCES:
            sliced[name] = {k: v for k, v in frames.items() if k in date_labels}
        else:
            sliced[name] = frames
    return sliced


def _compute_y_ranges(
    sources: dict[str, dict],
) -> dict[str, tuple[float, float]]:
    """Compute global (min, max) per numeric column across all weather/price data.

    The result is fed to ``_build_figures`` so that the y-axis on every
    per-month and cross-year figure spans the same overall range — making
    figures from different months/years visually comparable.
    """
    per_col: dict[str, list[np.ndarray]] = {}
    for name in _DATE_KEYED_SOURCES:
        for df in (sources.get(name) or {}).values():
            for col in df.columns:
                if col == "minutes" or not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                arr = df[col].to_numpy(dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    per_col.setdefault(col, []).append(arr)

    ranges: dict[str, tuple[float, float]] = {}
    for col, arrs in per_col.items():
        all_vals = np.concatenate(arrs)
        ranges[col] = (float(all_vals.min()), float(all_vals.max()))
    return ranges


def _run_monthly(
    cfg: dict,
    sources: dict[str, dict],
    out_dir: Path,
    base_name: str,
    output_formats: list[str],
    y_ranges: dict[str, tuple[float, float]] | None = None,
    stat_only: bool = True,
) -> None:
    """Build figures and write output for a pre-sliced source bundle."""
    has_data = any(
        bool(v) for k, v in sources.items()
        if k not in ("desired_temp_in", "ev_schedule")
    )
    if not has_data:
        logger.info("No data for %s — skipping.", base_name)
        return

    figures = _build_figures(cfg, sources, stat_only, y_ranges=y_ranges)
    if not figures:
        logger.info("No figures for %s — skipping.", base_name)
        return

    write_output(figures, out_dir, base_name, output_formats)


def run_overview(
    config_path: Path = DEFAULT_CONFIG,
    output_formats: list[str] | None = None,
    start_year: int = 2016,
    end_year: int = 2026,
) -> None:
    """Generate all per-month and cross-year plots."""
    output_formats = output_formats or ["html"]
    base_cfg = load_config(config_path)
    out_root = REPO_ROOT / base_cfg["output"]["dir"] / "monthly_overview"

    weather_keys = [k for k in WEATHER_DATASETS if k in base_cfg["weather"]]
    price_keys = [k for k in PRICE_DATASETS if k in base_cfg["price"]]

    if not weather_keys:
        logger.error("No known weather datasets found in config.")
        sys.exit(1)
    if not price_keys:
        logger.error("No known price datasets found in config.")
        sys.exit(1)

    total_combos = len(weather_keys) * len(price_keys)
    logger.info(
        "Datasets: weather=%s, price=%s (%d combos)",
        weather_keys, price_keys, total_combos,
    )

    # Full date sweep (all months × all years) — used both for global
    # y-range computation and as the data pool that gets sliced per figure.
    full_dates: list[datetime] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            full_dates.extend(days_in_month(year, month))

    for w_key in weather_keys:
        for p_key in price_keys:
            combo_tag = f"{w_key}_{p_key}"
            cfg = _make_config_variant(base_cfg, w_key, p_key)
            if cfg is None:
                continue

            combo_dir = out_root / combo_tag
            logger.info("=== Dataset combo: %s ===", combo_tag)

            all_sources = _load_all_sources(cfg, full_dates)
            y_ranges = _compute_y_ranges(all_sources)

            # --- Part 1: per year-month plots ---
            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    dates = days_in_month(year, month)
                    labels = {str(d.date()) for d in dates}
                    sources = _slice_sources(all_sources, labels)
                    base_name = f"{year}-{month:02d}"
                    _run_monthly(
                        cfg, sources,
                        combo_dir / "per_month",
                        base_name,
                        output_formats,
                        y_ranges=y_ranges,
                    )

            # --- Part 2: cross-year per calendar month ---
            for month in range(1, 13):
                month_dates: list[datetime] = []
                for year in range(start_year, end_year + 1):
                    month_dates.extend(days_in_month(year, month))
                labels = {str(d.date()) for d in month_dates}
                sources = _slice_sources(all_sources, labels)

                month_name = calendar.month_abbr[month]
                base_name = f"all_years_{month:02d}_{month_name}"
                _run_monthly(
                    cfg, sources,
                    combo_dir / "cross_year",
                    base_name,
                    output_formats,
                    y_ranges=y_ranges,
                )

    logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate monthly overview plots for years 2016-2026, "
                    "iterating over all weather/price dataset combinations.",
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

    apply_shared_cli_args(args, "plot_monthly_overview")

    run_overview(
        config_path=Path(args.config),
        output_formats=args.format,
        start_year=args.start_year,
        end_year=args.end_year,
    )


if __name__ == "__main__":
    main()
