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
KW_TO_W: float = W_PER_KW

# Safety cap against non-terminating eval episodes. Shared by the RLlib eval
# runner and the rule-based eval driver so both abort runaway episodes alike.
MAX_STEPS_PER_EPISODE: int = 1000

# Dynamical slowdown on the 1R1C thermal update (HP + heat-loss paths). From the LLEC
# parent project; gives hour-scale time constants from tens-K / hundreds-mC YAML params
# instead of the seconds-scale strict-SI dynamics.
# Reference: KIT-IAI/LLECBuildingGym base_building_gym.py::update_Tin.
SLOWDOWN_TERM: float = 0.001
