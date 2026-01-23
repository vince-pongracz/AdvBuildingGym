"""EV specification dataclass for configuring EV charger parameters."""

from dataclasses import dataclass


@dataclass
class EvSpec:
    """Specification for an Electric Vehicle's battery and charging parameters.

    To update EV charger when a new vehicle connects.
    """
    max_cap_kWh: float
    max_charging_kW: float
    charger_efficiency: float
    discharge_efficiency: float
    v2g_enabled: bool
    start_soc: float = 0.05
    target_soc: float = 0.9
