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

from .common import (
    DEFAULT_CONFIG,
    PRICE_DATASETS,
    REPO_ROOT,
    WEATHER_DATASETS,
    days_in_month,
    load_config,
    write_output,
)
from .loaders import set_warn_future_data
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


def _run_monthly(
    cfg: dict,
    dates: list[datetime],
    out_dir: Path,
    base_name: str,
    output_formats: list[str],
    stat_only: bool = True,
) -> None:
    """Load data, build figures, and write output for a list of dates."""
    sources = _load_all_sources(cfg, dates)

    # Check if any year-partitioned source returned data
    has_data = any(
        bool(v) for k, v in sources.items()
        if k not in ("desired_temp_in", "ev_schedule")
    )
    if not has_data:
        logger.info("No data for %s — skipping.", base_name)
        return

    figures = _build_figures(cfg, sources, stat_only)
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

    for w_key in weather_keys:
        for p_key in price_keys:
            combo_tag = f"{w_key}_{p_key}"
            cfg = _make_config_variant(base_cfg, w_key, p_key)
            if cfg is None:
                continue

            combo_dir = out_root / combo_tag
            logger.info("=== Dataset combo: %s ===", combo_tag)

            # --- Part 1: per year-month plots ---
            for year in range(start_year, end_year + 1):
                for month in range(1, 13):
                    dates = days_in_month(year, month)
                    base_name = f"{year}-{month:02d}"
                    _run_monthly(
                        cfg, dates,
                        combo_dir / "per_month",
                        base_name,
                        output_formats,
                    )

            # --- Part 2: cross-year per calendar month ---
            for month in range(1, 13):
                all_dates: list[datetime] = []
                for year in range(start_year, end_year + 1):
                    all_dates.extend(days_in_month(year, month))

                month_name = calendar.month_abbr[month]
                base_name = f"all_years_{month:02d}_{month_name}"
                _run_monthly(
                    cfg, all_dates,
                    combo_dir / "cross_year",
                    base_name,
                    output_formats,
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
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to data_plot_config.yaml (default: %(default)s).",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        default=["html"],
        choices=["html", "png", "svg", "pdf"],
        help="Output format(s). Default: html.",
    )
    parser.add_argument(
        "--warn-future-data",
        action="store_true",
        default=False,
        help="Emit 'No data for <date>' warnings for dates that have not yet "
             "occurred. Off by default — future dates are silently skipped.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info(
        "plot_monthly_overview started at %s",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    set_warn_future_data(args.warn_future_data)
    if not args.warn_future_data:
        logger.info(
            "Future-date warnings are OFF: 'No data for <date>' messages "
            "will be suppressed for dates after today. Pass --warn-future-data to enable."
        )

    run_overview(
        config_path=Path(args.config),
        output_formats=args.format,
        start_year=args.start_year,
        end_year=args.end_year,
    )


if __name__ == "__main__":
    main()
