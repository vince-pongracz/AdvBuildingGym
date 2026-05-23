"""Project-wide named constants.

Single source of truth for unit conversions and sentinel values that
were previously inlined as magic numbers across the codebase.
"""

SECONDS_PER_HOUR: int = 3600
SECONDS_PER_DAY: int = 86400

# kJ -> kWh: power(kW) * (control_step_s / SECONDS_PER_HOUR) = energy(kWh)
KJ_PER_KWH: int = 3600

# kW -> W
W_PER_KW: float = 1000.0

# Dynamical slowdown applied to the 1R1C building thermal update.
# Inherited from the LLEC parent project; multiplies dT/dt in both the
# HP and heat-loss paths so that YAML envelope parameters (K [W/K] in
# the tens, mC [J/K] in the hundreds) produce realistic, hour-scale
# thermal time constants instead of the seconds-scale dynamics that
# strict SI would imply for those magnitudes.
# Reference: KIT-IAI/LLECBuildingGym base_building_gym.py::update_Tin.
SLOWDOWN_TERM: float = 0.001
