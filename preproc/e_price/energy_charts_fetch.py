"""Fetch electricity market price data from the Energy Charts API.

Retrieves EPEX Spot day-ahead prices for a given date range and saves them
as CSV. Output format matches awattar_fetch.py so the same preprocessing
pipeline (awattar_price_preproc.py) can be used downstream.

API docs: https://api.energy-charts.info/#/prices/day_ahead_price_price_get
OpenAPI spec: https://api.energy-charts.info/openapi.json
License: CC BY 4.0 from Bundesnetzagentur | SMARD.de (for DE-LU)
"""

# Usage: python preproc/e_price/energy_charts_fetch.py
# Link: https://api.energy-charts.info/#/prices/day_ahead_price_price_get

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

_PROJECT_ROOT_STR = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_STR)

import pandas as pd
import requests

from preproc.utils import fetch_with_retry

logger = logging.getLogger(__name__)

API_URL: str = "https://api.energy-charts.info/price"

# The API handles up to ~one year per request. We use half-year chunks
# to stay well within limits while minimising the number of HTTP calls.
CHUNK_DAYS: int = 183

YEAR: int = 2023
BZN: str = "DE-LU"
START_DATE: str = f"{YEAR}-01-01"
END_DATE: str = f"{YEAR}-12-31"

OUTPUT_PATH: str = f"data/e_price/e_charts/{YEAR}_15m_prices.csv"


def _fetch_chunk(
    start: str,
    end: str,
    bzn: str = BZN,
    api_url: str = API_URL,
) -> pd.DataFrame:
    """Fetch day-ahead prices for a single date range from the Energy Charts API.

    Args:
        start: Start date as ISO 8601 date string (e.g. "2023-01-01").
        end: End date as ISO 8601 date string (e.g. "2023-06-30").
        bzn: Bidding zone identifier (default: DE-LU).
        api_url: Energy Charts API endpoint URL.

    Returns:
        DataFrame with columns: start_timestamp, marketprice, unit.
        Empty DataFrame if no data is returned.
    """
    # The API accepts ISO date strings directly (interpreted as local time)
    # Link: https://api.energy-charts.info/openapi.json
    response = fetch_with_retry(
        api_url,
        params={"bzn": bzn, "start": start, "end": end},
        timeout=60,
    )

    payload = response.json()
    unix_seconds = payload.get("unix_seconds", [])
    prices = payload.get("price", [])
    unit = payload.get("unit", "Eur/MWh")

    if not unix_seconds:
        logger.warning("No data returned for %s to %s", start, end)
        return pd.DataFrame()

    df = pd.DataFrame({
        "start_timestamp": pd.to_datetime(unix_seconds, unit="s", utc=True),
        "marketprice": prices,
        "unit": unit,
    })

    # Drop rows where price is null (can occur at DST transitions or missing data)
    null_count = df["marketprice"].isna().sum()
    if null_count > 0:
        logger.warning("Dropping %d rows with null prices (%s to %s)", null_count, start, end)
        df = df.dropna(subset=["marketprice"])

    return df


def fetch_market_data(
    start_date: str,
    end_date: str,
    bzn: str = BZN,
    api_url: str = API_URL,
) -> pd.DataFrame:
    """Fetch day-ahead prices in half-year chunks.

    Args:
        start_date: Start date (e.g. "2023-01-01").
        end_date: End date (e.g. "2023-12-31").
        bzn: Bidding zone identifier (default: DE-LU).
        api_url: Energy Charts API endpoint URL.

    Returns:
        DataFrame with columns matching aWATTar fetch output:
        start_timestamp, end_timestamp, marketprice, unit, marketprice_eur_per_kwh.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    chunks: list[pd.DataFrame] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS - 1), end)
        logger.info("Fetching %s to %s (bzn=%s)", chunk_start, chunk_end, bzn)

        try:
            df_chunk = _fetch_chunk(
                str(chunk_start), str(chunk_end), bzn=bzn, api_url=api_url,
            )
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Fetch failed for %s to %s (bzn=%s): %s — skipping chunk",
                chunk_start, chunk_end, bzn, exc,
            )
            chunk_start = chunk_end + timedelta(days=1)
            continue
        if not df_chunk.empty:
            chunks.append(df_chunk)

        chunk_start = chunk_end + timedelta(days=1)

    if not chunks:
        logger.warning("No data fetched for the entire range %s to %s", start_date, end_date)
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["start_timestamp"]).sort_values("start_timestamp").reset_index(drop=True)

    # Derive end_timestamp from actual data intervals (hourly or quarter-hourly)
    if len(df) >= 2:
        interval = df["start_timestamp"].iloc[1] - df["start_timestamp"].iloc[0]
    else:
        interval = pd.Timedelta(hours=1)
    df["end_timestamp"] = df["start_timestamp"] + interval

    # Convert Eur/MWh to Eur/kWh (1 Eur/MWh = 0.001 Eur/kWh)
    df["marketprice_eur_per_kwh"] = df["marketprice"] / 1000.0

    # Match aWATTar fetch output column order:
    # start_timestamp, end_timestamp, marketprice, unit, marketprice_eur_per_kwh
    df = df[["start_timestamp", "end_timestamp", "marketprice", "unit", "marketprice_eur_per_kwh"]]

    logger.info(
        "Fetched %d records total (%s to %s, interval=%s)",
        len(df),
        df["start_timestamp"].iloc[0],
        df["start_timestamp"].iloc[-1],
        interval,
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info(
        "Fetching Energy Charts day-ahead prices: %s to %s, bzn=%s",
        START_DATE, END_DATE, BZN,
    )
    prices_df = fetch_market_data(START_DATE, END_DATE, bzn=BZN)

    if not prices_df.empty:
        prices_df.to_csv(OUTPUT_PATH, index=False)

        logger.info("Sample of fetched data:")
        logger.info("=" * 80)
        logger.info(prices_df.head(10))
        logger.info("=" * 80)

        logger.info("Saved %d records to %s", len(prices_df), OUTPUT_PATH)
    else:
        logger.error("No data fetched, CSV not written")
