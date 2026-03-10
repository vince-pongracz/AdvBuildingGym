"""Generic CSV data augmentation via additive Gaussian noise.

Works on any CSV with numeric columns. Thin wrappers for price and
weather data provide domain-specific defaults and output naming.

Noise configuration is loaded from ``preproc/augment_config.yaml``.  The YAML is
read once at import time.  If the file is missing, augmentation functions
log an error and return without modifying any data.

Examples:
    python preproc/augment.py price data/e_price/e_charts/price_data_2023.csv
    python preproc/augment.py weather data/weather/dwd/preprocessed/2023_merged_04177.csv
    python preproc/augment.py price data/e_price/awattar/price_data_2023.csv --noise-std 0.5 --seed 18
    python preproc/augment.py weather data/weather/dwd/preprocessed/2023_merged_04177.csv --noise-std '{"temp_amb":0.3,"avg_wind_speed":0.1}'
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load noise configuration from configs/augment_config.yaml
# ---------------------------------------------------------------------------
_PREPROC_DIR: Path = Path(__file__).resolve().parent
_NOISE_CONFIG_PATH: Path = _PREPROC_DIR / "augment_config.yaml"


def _load_noise_config(path: Path = _NOISE_CONFIG_PATH) -> dict | None:
    """Read the noise YAML config. Returns None if file is missing."""
    if not path.exists():
        logger.error("Noise config not found at %s — augmentation disabled", path)
        return None
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


_CFG: dict | None = _load_noise_config()

DEFAULT_SEED: int = _CFG["seed"] if _CFG else 42
PRICE_COLUMNS: list[str] = _CFG["price"]["columns"] if _CFG else []
PRICE_NOISE_STD: float = _CFG["price"]["noise_std"]["baseprice"] if _CFG else 0.0
WEATHER_COLUMNS: list[str] = _CFG["weather"]["columns"] if _CFG else []
WEATHER_NOISE_STD: dict[str, float] = _CFG["weather"]["noise_std"] if _CFG else {}
WEATHER_CLIP_MIN: dict[str, float] = _CFG["weather"]["clip_min"] if _CFG else {}


def augment_csv(
    input_path: str,
    output_path: str,
    columns: list[str],
    noise_std: float | dict[str, float] = 0.3,
    seed: int | None = None,
    clip_min: dict[str, float] | None = None,
) -> pd.DataFrame | None:
    """Add Gaussian noise to specified numeric columns of a CSV.

    Args:
        input_path: Path to input CSV.
        output_path: Path for augmented output CSV.
        columns: Column names to augment. Columns not present in the CSV are
            silently skipped.
        noise_std: Standard deviation of noise. Either a single float applied
            to all columns, or a dict mapping column names to per-column std.
        seed: Random seed for reproducibility (None = non-deterministic).
        clip_min: Optional per-column minimum value to clip after adding noise
            (e.g. ``{"avg_wind_speed": 0.0}`` to prevent negative wind speeds).

    Returns:
        The augmented DataFrame (also written to *output_path*), or None if
        the noise config is missing.
    """
    if _CFG is None:
        logger.error("Cannot augment — preproc/augment_config.yaml not found")
        return None

    df = pd.read_csv(input_path)
    logger.info("Loaded %d records from %s", len(df), input_path)

    rng = np.random.default_rng(seed)

    present_columns = [c for c in columns if c in df.columns]
    missing = set(columns) - set(present_columns)
    if missing:
        logger.info("Columns not in CSV (skipped): %s", sorted(missing))

    for col in present_columns:
        std = noise_std[col] if isinstance(noise_std, dict) else noise_std
        noise = rng.normal(loc=0.0, scale=std, size=len(df)).astype(np.float64)
        df[col] = df[col] + noise

        if clip_min and col in clip_min:
            df[col] = df[col].clip(lower=clip_min[col])

        logger.info(
            "  %s: noise_std=%.4f, range after augmentation: %.4f – %.4f",
            col, std, df[col].min(), df[col].max(),
        )

    df.to_csv(output_path, index=False)
    logger.info("Saved %d augmented records to %s", len(df), output_path)
    return df


# ---------------------------------------------------------------------------
# Domain wrappers
# ---------------------------------------------------------------------------

def augment_prices(
    input_path: str,
    output_path: str,
    noise_std: float = PRICE_NOISE_STD,
    seed: int | None = None,
    normalize: bool = False,
) -> None:
    """Add Gaussian noise to preprocessed price data.

    Args:
        input_path: Path to preprocessed price CSV (5-min resolution, ct/kWh).
        output_path: Path for the augmented output CSV.
        noise_std: Standard deviation of the Gaussian noise in ct/kWh.
        seed: Random seed for reproducibility (None = non-deterministic).
        normalize: If True, recompute price_normalized column after augmentation.
    """
    df = augment_csv(
        input_path, output_path,
        columns=PRICE_COLUMNS,
        noise_std=noise_std,
        seed=seed,
    )
    if df is None:
        return

    if normalize:
        # Reload to add normalization column and rewrite
        abs_max = df["baseprice"].abs().max()
        if abs_max > 0:
            df["price_normalized"] = df["baseprice"] / abs_max
        else:
            logger.warning("All prices zero after augmentation, setting normalized to 0.0")
            df["price_normalized"] = 0.0
        df = df[["start", "baseprice", "unit", "hour", "price_normalized"]]
        df.to_csv(output_path, index=False)


def augment_weather(
    input_path: str,
    output_path: str,
    noise_std: float | dict[str, float] | None = None,
    seed: int | None = None,
    columns: list[str] | None = None,
) -> None:
    """Add Gaussian noise to DWD weather data.

    Args:
        input_path: Path to preprocessed DWD weather CSV.
        output_path: Path for the augmented output CSV.
        noise_std: Noise std per column (dict) or single float for all columns.
            Defaults to WEATHER_NOISE_STD dict.
        seed: Random seed for reproducibility.
        columns: Columns to augment. Defaults to WEATHER_COLUMNS.
    """
    if noise_std is None:
        noise_std = WEATHER_NOISE_STD
    if columns is None:
        columns = WEATHER_COLUMNS

    augment_csv(
        input_path, output_path,
        columns=columns,
        noise_std=noise_std,
        seed=seed,
        clip_min=WEATHER_CLIP_MIN,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_noise_std(value: str) -> float | dict[str, float]:
    """Parse noise-std as either a float or a JSON dict."""
    try:
        return float(value)
    except ValueError:
        return json.loads(value)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Add Gaussian noise to CSV data for augmentation",
    )
    parser.add_argument(
        "domain",
        choices=["price", "weather"],
        help="Data domain (determines default columns and noise levels)",
    )
    parser.add_argument("input", help="Path to input CSV")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output CSV path (default: <input>_aug_seed<seed>.csv)",
    )
    parser.add_argument(
        "--noise-std",
        type=_parse_noise_std,
        default=None,
        help="Noise std: float for all columns, or JSON dict for per-column "
             "(e.g. '{\"temp_amb\":0.3,\"avg_wind_speed\":0.1}'). "
             "Defaults to domain-specific values.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="(price only) Recompute price_normalized column after augmentation",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = re.sub(r"\.csv$", f"_aug_seed{args.seed}.csv", args.input)

    if args.domain == "price":
        augment_prices(
            args.input, output_path,
            noise_std=args.noise_std if args.noise_std is not None else PRICE_NOISE_STD,
            seed=args.seed,
            normalize=args.normalize,
        )
    else:
        augment_weather(
            args.input, output_path,
            noise_std=args.noise_std,
            seed=args.seed,
        )
