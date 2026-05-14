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
    # Hard deadline (hours from connect) by which the session must reach
    # target_soc.  This is the user-observable charging contract -- the actual
    # disconnect time may be later but is not assumed observable.  Drives the
    # per-session corridor in LinearEVCharger:
    #   * s_ev_soc_min: lazy back-from-target line that just reaches
    #     target_soc by this deadline at max charge rate.  Falling below it
    #     means the target is no longer reachable and EVChargingReward
    #     terminates the episode.
    #   * s_ev_soc_max: forward-from-start line at max charge rate from
    #     connect, capped at 1.0; the upper envelope of physically reachable
    #     SoC at each step.
    charge_to_target_in_hrs: float = 8.0
