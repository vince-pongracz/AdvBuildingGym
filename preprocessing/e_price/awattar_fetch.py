"""Fetch electricity market price data from the aWATTar API.

Retrieves hourly EPEX Spot prices for a given date range and saves them as CSV.
API docs: https://www.awattar.at/services/api
"""

# NOTE VP 2026.02.28. : Usage: python preprocessing/e_price/awattar_fetch.py

# aWATTar fair-use policy: max 100 requests/day
# Link: https://www.awattar.at/services/api

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT_STR = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_STR)

import pandas as pd
import requests

from preprocessing.utils import fetch_with_retry

logger = logging.getLogger(__name__)

API_URL: str = "https://api.awattar.de/v1/marketdata"

YEAR: int = 2023
START_DATE: str = f"{YEAR}-01-01T00:00:00Z"
END_DATE: str = f"{YEAR}-12-31T23:59:59Z"

OUTPUT_PATH: str = f"data/e_price/awattar/{YEAR}_prices.csv"


def _to_epoch_ms(iso_date: str) -> int:
    """Convert ISO 8601 date string to Unix epoch milliseconds."""
    dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


def fetch_market_data(start_date: str, end_date: str, api_url: str = API_URL) -> pd.DataFrame:
    """Fetch hourly electricity prices from the aWATTar API for a date range.

    Args:
        start_date: ISO 8601 start date (e.g. "2024-01-01T00:00:00Z").
        end_date: ISO 8601 end date (e.g. "2024-12-31T23:59:59Z").
        api_url: aWATTar API endpoint URL.

    Returns:
        DataFrame with columns: start_timestamp, end_timestamp, marketprice, unit.
    """
    start_ms = _to_epoch_ms(start_date)
    end_ms = _to_epoch_ms(end_date)

    logger.info(
        "Fetching %s to %s",
        datetime.fromtimestamp(
            start_ms / 1000, tz=timezone.utc).isoformat(),
        datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).isoformat(),
    )

    try:
        response = fetch_with_retry(
            api_url,
            params={"start": start_ms, "end": end_ms},
            timeout=30,
        )
    except requests.exceptions.RequestException as exc:
        logger.warning(
            "Fetch failed for %s to %s after retries: %s — returning empty DataFrame",
            start_date, end_date, exc,
        )
        return pd.DataFrame()

    data = response.json().get("data", [])
    logger.info("Received %d records", len(data))

    df = pd.DataFrame(data)
    if df.empty:
        logger.warning("No data returned from API")
        return df

    # Convert epoch ms timestamps to datetime
    df["start_timestamp"] = pd.to_datetime(df["start_timestamp"], unit="ms", utc=True)
    df["end_timestamp"] = pd.to_datetime(df["end_timestamp"], unit="ms", utc=True)

    # marketprice is left in Eur/MWh; the Eur/MWh -> ct/kWh conversion happens
    # once downstream in awattar_price_preproc.preprocess_prices (baseprice).
    df = df.sort_values("start_timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Fetching aWATTar market data from %s to %s", START_DATE, END_DATE)
    prices_df = fetch_market_data(START_DATE, END_DATE)

    if not prices_df.empty:
        prices_df.to_csv(OUTPUT_PATH, index=False)
        
        logger.info("Sample of fetched data:")
        logger.info("=" * 80)
        logger.info(prices_df.head(10))
        logger.info("=" * 80)
        
        logger.info("Saved %d records to %s", len(prices_df), OUTPUT_PATH)
    else:
        logger.error("No data fetched, CSV not written")
