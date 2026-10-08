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
    load_profiles_from_cfg,
    resolve_source,
    write_output,
)
from .loaders import load_days, load_syn_cfg_days
from .plot_day_data import _build_figures

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
_DATE_KEYED_SYN_SOURCES = ("weather_syn", "price_syn")


def _slice_sources(
    sources: dict[str, dict],
    date_labels: set[str],
) -> dict[str, dict]:
    """Filter year-partitioned sources to a subset of date labels.

    Profile sources (desired_temp_in, ev_schedule, user_energy_need) are
    date-independent and passed through unchanged. ``*_syn`` entries are
    nested ``{cfg_name: {date_label: df}}`` and get sliced one level deeper.
    """
    sliced: dict[str, dict] = {}
    for name, frames in sources.items():
        if name in _DATE_KEYED_SOURCES:
            sliced[name] = {k: v for k, v in frames.items() if k in date_labels}
        elif name in _DATE_KEYED_SYN_SOURCES:
            sliced[name] = {
                cfg: {k: v for k, v in cfg_frames.items() if k in date_labels}
                for cfg, cfg_frames in frames.items()
            }
            # Drop fully-empty cfgs for tidy figures.
            sliced[name] = {c: f for c, f in sliced[name].items() if f}
        else:
            sliced[name] = frames
    return sliced


def _compute_y_ranges(
    sources: dict[str, dict],
) -> dict[str, tuple[float, float]]:
    """Compute global (min, max) per numeric column across all weather/price data.

    The result is fed to ``_build_figures`` so that the y-axis on every
    per-month and cross-year figure spans the same overall range — making
    figures from different months/years visually comparable. Numeric
    columns are resolved once per source bundle and ``np.isfinite`` is
    applied once on the concatenated array, avoiding per-frame Python
    overhead that dominated the per-day path.
    """
    per_col: dict[str, list[np.ndarray]] = {}

    def _ingest(frames: dict[str, pd.DataFrame]) -> None:
        if not frames:
            return
        sample = next(iter(frames.values()))
        cols = [
            c for c in sample.columns
            if c != "minutes" and pd.api.types.is_numeric_dtype(sample[c])
        ]
        for col in cols:
            arrs = [
                df[col].to_numpy(dtype=float, copy=False)
                for df in frames.values() if col in df.columns
            ]
            if arrs:
                per_col.setdefault(col, []).extend(arrs)

    for name in _DATE_KEYED_SOURCES:
        _ingest(sources.get(name) or {})
    # Include syn_cfg frames so the shared y-axis encompasses their range too.
    for name in _DATE_KEYED_SYN_SOURCES:
        for cfg_frames in (sources.get(name) or {}).values():
            _ingest(cfg_frames)

    ranges: dict[str, tuple[float, float]] = {}
    for col, arrs in per_col.items():
        all_vals = np.concatenate(arrs)
        all_vals = all_vals[np.isfinite(all_vals)]
        if all_vals.size:
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

    height = cfg.get("figure", {}).get("overview_height")
    figures = _build_figures(cfg, sources, stat_only, y_ranges=y_ranges, height=height)
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

    # Preload each weather/price dataset once across all combos. With
    # |weather_keys| × |price_keys| combos, the naive per-combo load would
    # parse each weather dataset |price_keys|× and each price dataset
    # |weather_keys|×; caching here cuts that to one load per dataset.
    # y-ranges are computed at the same time since they depend only on
    # the dataset, not on which combo it appears in.
    weather_cache: dict[str, dict[str, dict]] = {}
    weather_y_ranges: dict[str, dict[str, tuple[float, float]]] = {}
    for w_key in weather_keys:
        src = resolve_source(base_cfg["weather"], "weather", key=w_key)
        w_dir = REPO_ROOT / src["dir"]
        weather_cache[w_key] = {
            "weather": load_days(w_dir, src["file_pattern"], src["timestamp_col"], full_dates),
            "weather_syn": load_syn_cfg_days(w_dir, src["file_pattern"], src["timestamp_col"], full_dates),
        }
        weather_y_ranges[w_key] = _compute_y_ranges(weather_cache[w_key])

    price_cache: dict[str, dict[str, dict]] = {}
    price_y_ranges: dict[str, dict[str, tuple[float, float]]] = {}
    for p_key in price_keys:
        src = resolve_source(base_cfg["price"], "price", key=p_key)
        p_dir = REPO_ROOT / src["dir"]
        price_cache[p_key] = {
            "price": load_days(p_dir, src["file_pattern"], src["timestamp_col"], full_dates),
            "price_syn": load_syn_cfg_days(p_dir, src["file_pattern"], src["timestamp_col"], full_dates),
        }
        price_y_ranges[p_key] = _compute_y_ranges(price_cache[p_key])

    # Profile sources are dataset-independent; load once for all combos.
    profile_sources = load_profiles_from_cfg(base_cfg, full_dates)

    for w_key in weather_keys:
        for p_key in price_keys:
            combo_tag = f"{w_key}_{p_key}"
            cfg = _make_config_variant(base_cfg, w_key, p_key)
            if cfg is None:
                continue

            combo_dir = out_root / combo_tag
            logger.info("=== Dataset combo: %s ===", combo_tag)

            all_sources = {
                **weather_cache[w_key],
                **price_cache[p_key],
                **profile_sources,
            }
            # Weather y-ranges and price y-ranges have disjoint column keys.
            y_ranges = {**weather_y_ranges[w_key], **price_y_ranges[p_key]}

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
