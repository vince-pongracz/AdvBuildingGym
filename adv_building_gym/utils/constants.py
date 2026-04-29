"""Project-wide named constants.

Single source of truth for unit conversions and sentinel values that
were previously inlined as magic numbers across the codebase.
"""

SECONDS_PER_HOUR: int = 3600
SECONDS_PER_DAY: int = 86400

# kJ -> kWh: power(kW) * (control_step_s / SECONDS_PER_HOUR) = energy(kWh)
KJ_PER_KWH: int = 3600

# Raw weather CSVs use -999 as the missing-value sentinel; derived
# columns (e.g. cloud cover -> sunshine) propagate it as -1998.
WEATHER_SENTINEL: float = -999.0
WEATHER_SENTINEL_DERIVED: float = -1998.0