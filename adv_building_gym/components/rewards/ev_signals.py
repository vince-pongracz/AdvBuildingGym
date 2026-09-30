"""Shared EV signal accessors for the EV reward functions.

``LinearEVCharger`` no longer republishes the EV-spec fields that ``EVState`` already
publishes as ``ctxt_ev_schedule_*`` (``max_cap_kWh``, ``charger_efficiency``,
``target_soc``) and no longer publishes a dedicated connection flag. The rewards read
what they still need through these accessors, so the source of each signal is stated in
exactly one place.

The schedule keys are NOT interchangeable with the charger's view: ``EVState`` is an
exogenous statesource updated after the iteration counter advances, so it flips one step
before the charger applies the connect/disconnect in ``exec_action``.
"""

from typing import Dict


def is_ev_connected(states: Dict) -> bool:
    """Whether an EV is plugged in, with the charger's timing.

    Read from ``ctxt_evc_max_charging_kW``, which is ``min(charger rating, EV acceptance)``
    and therefore 0 exactly when no EV is attached
    (``LinearEVCharger.effective_max_charging_kW`` returns 0.0 for ``ev_spec is None``).
    Being a plain observation key it is available on both ``state`` and ``next_state``, so
    the disconnect edge is still detectable from the (s, s') pair.
    """
    return float(states["ctxt_evc_max_charging_kW"][0]) > 0.0


def session_target_soc(info: dict) -> float:
    """Target SoC of the current — or just-ended — charging session.

    Latched by ``LinearEVCharger`` and deliberately not cleared on detach, so the
    disconnect verdict can still judge the achieved SoC against it on the step the charger
    releases the EV.
    """
    return float(info.get("evc_target_soc", 0.0))
