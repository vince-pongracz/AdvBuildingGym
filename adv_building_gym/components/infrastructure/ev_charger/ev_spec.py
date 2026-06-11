"""EV specification dataclass for configuring EV charger parameters."""

from dataclasses import dataclass


@dataclass
class EvSpec:
    """EV battery + charging parameters; applied when a new vehicle connects."""
    max_cap_kWh: float
    max_charging_kW: float
    charger_efficiency: float
    discharge_efficiency: float
    v2g_enabled: bool
    start_soc: float = 0.05
    target_soc: float = 0.9
    # Hard deadline (hours from connect) to reach target_soc — the user-observable
    # charging contract. Drives the LinearEVCharger corridor:
    #   * s_ev_soc_min: back-from-target line reaching target_soc by the deadline at
    #     max rate; dropping below it makes the target unreachable → EVChargingReward ends the episode.
    #   * s_ev_soc_max: forward-from-start line at max rate, capped at 1.0 (upper envelope of reachable SoC).
    charge_to_target_in_hrs: float = 8.0
