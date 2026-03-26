"""
Preprocessing module for data exploration and transformation.
"""

from .e_price.awattar_fetch import fetch_market_data as fetch_awattar_market_data
from .e_price.awattar_price_preproc import preprocess_prices
from .e_price.energy_charts_fetch import fetch_market_data as fetch_energy_charts_market_data
from .augment import augment_prices
from .weather.explore_hdf5 import explore_hdf5_file
from .weather.extract_sfh_csv import extract_sfh_data
from .weather.extract_weather_csv import extract_weather_data
from .weather.preproc_types import SFHExtractionStats, WeatherExtractionStats
from .utils import (
    DWD_MISSING_VALUE,
    ensure_datetime_index,
    fetch_with_retry,
    get_measurement_columns,
    is_missing,
    parse_timestamp_column,
    parse_year_from_filename,
    resolve_path,
    select_columns,
)

__all__ = [
    "augment_prices",
    "DWD_MISSING_VALUE",
    "ensure_datetime_index",
    "fetch_with_retry",
    "get_measurement_columns",
    "is_missing",
    "parse_timestamp_column",
    "parse_year_from_filename",
    "select_columns",
    "explore_hdf5_file",
    "extract_sfh_data",
    "extract_weather_data",
    "fetch_awattar_market_data",
    "fetch_energy_charts_market_data",
    "preprocess_prices",
    "resolve_path",
    "SFHExtractionStats",
    "WeatherExtractionStats",
]
