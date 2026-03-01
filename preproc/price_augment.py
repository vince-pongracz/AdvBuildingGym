"""Add noise to preprocessed electricity price data for data augmentation.

Applies additive Gaussian noise to the baseprice column and recomputes
the normalized price. Preserves the original CSV format so the output
can be used directly by the EnergyPrice statesource.

Input/output format (same as awattar_price_preproc.py output):
    start, baseprice, unit, hour, price_normalized
"""

# Usage: python preproc/price_augment.py data/price_data_2023_norm.csv -o data/price_data_2023_aug.csv
#        python preproc/price_augment.py data/price_data_2023_norm.csv --noise-std 0.5 --seed 42

import argparse
import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_NOISE_STD: float = 0.3
SEED: int = 42


def augment_prices(
    input_path: str,
    output_path: str,
    noise_std: float = DEFAULT_NOISE_STD,
    seed: int | None = None,
) -> None:
    """Add Gaussian noise to preprocessed price data and renormalize.

    Args:
        input_path: Path to preprocessed price CSV (5-min resolution, ct/kWh).
        output_path: Path for the augmented output CSV.
        noise_std: Standard deviation of the Gaussian noise in ct/kWh.
        seed: Random seed for reproducibility (None = non-deterministic).
    """
    df = pd.read_csv(input_path)
    logger.info("Loaded %d records from %s", len(df), input_path)

    rng = np.random.default_rng(seed)

    # Add Gaussian noise to baseprice (ct/kWh)
    noise = rng.normal(loc=0.0, scale=noise_std, size=len(df)).astype(np.float64)
    df["baseprice"] = df["baseprice"] + noise

    # Renormalize: absolute-max normalization, sign-preserving, range [-1, 1]
    abs_max = df["baseprice"].abs().max()
    if abs_max > 0:
        df["price_normalized"] = df["baseprice"] / abs_max
    else:
        logger.warning("All prices zero after augmentation, setting normalized to 0.0")
        df["price_normalized"] = 0.0

    price_min = df["baseprice"].min()
    price_max = df["baseprice"].max()

    # Preserve column order
    df = df[["start", "baseprice", "unit", "hour", "price_normalized"]]

    df.to_csv(output_path, index=False)
    logger.info(
        "Saved %d augmented records to %s (noise_std=%.2f ct/kWh, price range: %.2f–%.2f ct/kWh)",
        len(df), output_path, noise_std, price_min, price_max,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Add noise to preprocessed price data for augmentation",
    )
    parser.add_argument("input", help="Path to preprocessed price CSV (price_data_*_norm.csv)")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output CSV path (default: replaces '_norm' with '_norm_aug' in input filename)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=DEFAULT_NOISE_STD,
        help=f"Noise standard deviation in ct/kWh (default: {DEFAULT_NOISE_STD})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for reproducibility (default: {SEED})",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        # Replace '_norm' suffix with '_aug', or append '_aug' before extension
        if "_norm" in args.input:
            output_path = args.input.replace("_norm", f"_norm_aug_seed{SEED}")
        else:
            output_path = re.sub(r"\.csv$", "_aug.csv", args.input)

    augment_prices(args.input, output_path, noise_std=args.noise_std, seed=args.seed)
