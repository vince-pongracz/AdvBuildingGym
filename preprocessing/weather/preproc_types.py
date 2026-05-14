"""Type definitions for preprocessing module."""

from typing import TypedDict


class SFHExtractionStats(TypedDict):
    """Statistics from SFH data extraction."""

    total_buildings: int
    successful: int
    failed: int
    files_created: list[str]
    errors: list[dict[str, str]]


class WeatherExtractionStats(TypedDict):
    """Statistics from weather data extraction."""

    variables_found: int
    variables_loaded: int
    rows_before_dropna: int
    rows_after_dropna: int
    columns: list[str]
    output_file: str | None
    errors: list[dict[str, str]]
