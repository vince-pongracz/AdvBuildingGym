"""Utilities for discovering augmented data scenarios on disk."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def discover_augmented_scenarios(
    years: range,
    weather_dir: str = "data/weather/dwd/preprocessed",
    price_dirs: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Glob for augmented yearly CSVs and pair weather + price by year.

    Only adds a scenario when both the augmented weather and augmented
    price file exist for a given year and source.
    """
    if price_dirs is None:
        price_dirs = {
            "awattar": "data/e_price/awattar",
            "e_charts": "data/e_price/e_charts",
        }

    weather_path = Path(weather_dir)
    scenarios: list[dict[str, str]] = []

    for year in years:
        # Find any augmented weather file for this year (any seed)
        aug_weather = sorted(weather_path.glob(f"{year}_merged_*_aug_*.csv"))
        if not aug_weather:
            continue

        for source_name, source_dir in price_dirs.items():
            aug_prices = sorted(Path(source_dir).glob(f"price_data_{year}_aug*.csv"))
            if not aug_prices:
                continue

            # Pair each augmented weather file with each augmented price file
            for w in aug_weather:
                for p in aug_prices:
                    scenarios.append({
                        "weather": str(w),
                        "E_price": str(p),
                    })

    if scenarios:
        logger.info("Discovered %d augmented scenario(s)", len(scenarios))
    return scenarios
