"""Preprocess fetched aWATTar electricity price data for use in AdvBuildingGym.

Converts hourly aWATTar market data (Eur/MWh, UTC) to 5-minute resolution
data (ct/kWh, naive timestamps) matching the price_data format used by
the environment's EnergyPrice statesource.

Input format (aWATTar fetch output):
    start_timestamp, end_timestamp, marketprice, unit, marketprice_eur_per_kwh

Output format (environment-compatible):
    start, baseprice, unit, hour[, price_normalized]
"""

# NOTE VP 2026.02.28. : Usage: python preproc/e_price/awattar_price_preproc.py data/e_price/2023_prices.csv [-o data/e_price/2023_prices_preproc.csv]

import argparse
import logging

import pandas as pd

logger = logging.getLogger(__name__)

RESAMPLE_INTERVAL: str = "5min"
OUTPUT_UNIT: str = "ct/kWh"


def preprocess_prices(input_path: str, output_path: str, normalize: bool = False) -> None:
    """Convert aWATTar hourly price data to 5-min resolution environment format.

    Args:
        input_path: Path to fetched aWATTar CSV (hourly, Eur/MWh).
        output_path: Path for the preprocessed output CSV.
        normalize: If True, add a price_normalized column (abs-max, [-1, 1]).
    """
    df = pd.read_csv(input_path, parse_dates=["start_timestamp", "end_timestamp"])
    logger.info("Loaded %d hourly records from %s", len(df), input_path)

    # Detect missing hourly intervals in the source data
    df_sorted = df.sort_values("start_timestamp")
    time_diffs = df_sorted["start_timestamp"].diff()
    expected_interval = pd.Timedelta(hours=1)
    gaps = time_diffs[time_diffs > expected_interval]
    if gaps.empty:
        logger.info("No missing intervals detected in source data")
    else:
        logger.warning("Detected %d gap(s) in source data:", len(gaps))
        for idx in gaps.index:
            gap_start = df_sorted["start_timestamp"].iloc[idx - 1]
            gap_end = df_sorted["start_timestamp"].iloc[idx]
            missing_hours = int(gaps[idx] / expected_interval) - 1
            logger.warning("  Missing %d hour(s): %s to %s", missing_hours, gap_start, gap_end)

    # Log rows with missing values
    null_counts = df[["start_timestamp", "marketprice"]].isnull().sum()
    if null_counts.any():
        logger.warning("Null values found — start_timestamp: %d, marketprice: %d",
                        null_counts["start_timestamp"], null_counts["marketprice"])

    # Convert Eur/MWh to ct/kWh (1 Eur/MWh = 0.1 ct/kWh)
    df["baseprice"] = df["marketprice"] / 10.0

    # Strip timezone info, keep values as-is
    df["start"] = df["start_timestamp"].dt.tz_localize(None)

    # Extract hour from start timestamp
    df["hour"] = df["start"].dt.hour

    # Upsample from 1-hour to 5-minute resolution using forward-fill
    df = df.set_index("start")
    df = df[["baseprice", "hour"]]
    df = df.resample(RESAMPLE_INTERVAL).ffill()

    # The last hourly row gets forward-filled for its 5-min slots, but the
    # final generated timestamp (next hour boundary) is an artifact — drop it
    # if it extends beyond the original data range.
    df = df.iloc[:-1] if len(df) > 0 else df

    df = df.reset_index()

    # Recalculate hour after resampling (forward-fill may not update it correctly
    # at hour boundaries where the index crosses but the ffilled hour value lags)
    df["hour"] = df["start"].dt.hour

    price_min = df["baseprice"].min()
    price_max = df["baseprice"].max()

    df["unit"] = OUTPUT_UNIT

    if normalize:
        # Absolute max normalization: divide by max(|price|) so sign is preserved
        # Negative prices stay negative, positive stay positive, range is [-1, 1]
        abs_max = df["baseprice"].abs().max()
        if abs_max > 0:
            df["price_normalized"] = df["baseprice"] / abs_max
        else:
            logger.warning("All prices zero, setting normalized to 0.0")
            df["price_normalized"] = 0.0
        df = df[["start", "baseprice", "unit", "hour", "price_normalized"]]
    else:
        df = df[["start", "baseprice", "unit", "hour"]]

    df.to_csv(output_path, index=False)
    logger.info(
        "Saved %d records to %s (price range: %.2f–%.2f %s, normalized: %s)",
        len(df), output_path, price_min, price_max, OUTPUT_UNIT, normalize,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Preprocess aWATTar price data for AdvBuildingGym")
    parser.add_argument("input", help="Path to fetched aWATTar CSV file")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output CSV path (default: data/e_price/price_data_<year>.csv)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Add price_normalized column (abs-max normalization to [-1, 1])",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        df_peek = pd.read_csv(args.input, nrows=1, parse_dates=["start_timestamp"])
        year = df_peek["start_timestamp"].iloc[0].year
        suffix = "_norm" if args.normalize else ""
        output_path = f"data/e_price/price_data_{year}{suffix}.csv"

    preprocess_prices(args.input, output_path, normalize=args.normalize)
