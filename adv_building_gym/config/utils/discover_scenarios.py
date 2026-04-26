"""Utilities for discovering synthesised data scenarios on disk."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# (directory, per-year glob pattern) — use {year} as a placeholder.
DEFAULT_WEATHER_SOURCES: list[tuple[str, str]] = [
    ("data/weather/dwd/preprocessed", "{year}_merged_*_syn_cfg_*.csv"),
    ("data/weather/zenodo/csvs_weather", "{year}_weather_syn_cfg_*.csv"),
]


def discover_synthetic_scenarios(
    years: range,
    weather_dir: str | list[tuple[str, str]] = "data/weather/dwd/preprocessed",
    price_dirs: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Glob for synthesised yearly CSVs and pair weather + price by year.

    Args:
        years: Year range to search.
        weather_dir: Either a single DWD-style directory (for backwards-compat;
            uses the DWD glob `{year}_merged_*_syn_cfg_*.csv`) or an explicit
            list of ``(dir, pattern)`` pairs. ``pattern`` may use ``{year}`` as
            a placeholder. If left at its default the function automatically
            also picks up Zenodo synthesised CSVs (see ``DEFAULT_WEATHER_SOURCES``).
        price_dirs: Map ``source_name → directory`` for price CSVs.

    Only adds a scenario when both a synthesised weather and a synthesised
    price file exist for a given year.
    """
    if price_dirs is None:
        price_dirs = {
            "awattar": "data/e_price/awattar",
            "e_charts": "data/e_price/e_charts",
        }

    if isinstance(weather_dir, str):
        if weather_dir == "data/weather/dwd/preprocessed":
            weather_sources = DEFAULT_WEATHER_SOURCES
        else:
            weather_sources = [(weather_dir, "{year}_merged_*_syn_cfg_*.csv")]
    else:
        weather_sources = weather_dir

    scenarios: list[dict[str, str]] = []

    for year in years:
        # Collect synthesised weather files for this year across all sources
        syn_weather_paths: list[Path] = []
        for w_dir, w_pattern in weather_sources:
            syn_weather_paths.extend(
                sorted(Path(w_dir).glob(w_pattern.format(year=year)))
            )
        if not syn_weather_paths:
            continue

        for _, source_dir in price_dirs.items():
            syn_price_paths = sorted(Path(source_dir).glob(f"price_data_{year}_syn_cfg_*.csv"))
            if not syn_price_paths:
                continue

            # Pair each synthesised weather file with each synthesised price file
            for weather_path in syn_weather_paths:
                for price_path in syn_price_paths:
                    scenarios.append({
                        "weather": str(weather_path),
                        "E_price": str(price_path),
                    })

    if scenarios:
        logger.info("Discovered %d synthesised scenario(s)", len(scenarios))
    return scenarios
