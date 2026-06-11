"""EV schedule state source that drives EV charger connection events from CSV profiles."""

import logging
from typing import ClassVar, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from adv_building_gym._common.constants import SECONDS_PER_HOUR

from ..base import StateSource
from ..csv_loader import CsvLoader
from adv_building_gym.components.registry import ComponentRegistry
from ...infrastructure.ev_charger.ev_spec import EvSpec

logger = logging.getLogger(__name__)


class EVState(StateSource):
    """Reads an EV usage profile CSV and drives charger connect/disconnect events.

    The CSV has the following columns:
        start, max_cap_kWh, max_charging_kW, charger_efficiency,
        discharge_efficiency, v2g_enabled, start_soc, target_soc,
        target_soc_reach_duration_h
    Populated ``max_cap_kWh`` = connect (full EvSpec); NaN = disconnect. Each timestamp's
    time-of-day maps to an iteration via ``control_step``.

    Writes ``ev_schedule_connected`` and EV spec fields into the shared dict;
    LinearEVCharger reads them in ``exec_action`` and calls ``set_ev_connected``.
    """

    _context_params: ClassVar[Set[str]] = {'control_step'}
    _exclude_params: ClassVar[Set[str]] = {
        '_events', '_event_lookup',
        '_ev_connected', '_current_spec',
    }

    # State keys written by this source (read by LinearEVCharger)
    KEY_CONNECTED = "ev_schedule_connected"
    KEY_MAX_CAP = "ev_schedule_max_cap_kWh"
    KEY_MAX_CHARGE = "ev_schedule_max_charging_kW"
    KEY_CHARGE_EFF = "ev_schedule_charger_eff"
    KEY_DISCHARGE_EFF = "ev_schedule_discharge_eff"
    KEY_V2G = "ctxt_ev_schedule_v2g"
    KEY_START_SOC = "ctxt_ev_schedule_start_soc"
    KEY_TARGET_SOC = "ctxt_ev_schedule_target_soc"
    # Hours from connect to hit target_soc — the user contract, deadline for the
    # LinearEVCharger corridor (not the actual disconnect time, not assumed observable).
    KEY_CHARGE_TO_TARGET_HRS = "ev_schedule_charge_to_target_hrs"

    def __init__(
        self,
        name: str,
        ds_path: str | None = None,
        control_step: float = 300.0,
    ) -> None:
        super().__init__(name=name, control_step=control_step)

        self._ev_connected: bool = False
        self._current_spec: Optional[EvSpec] = None

        # Pre-compute event list from CSV
        self._events: List[Tuple[int, bool, Optional[EvSpec]]] = []
        # Lookup dict for O(1) access: iteration_index -> (is_connect, ev_spec)
        self._event_lookup: Dict[int, Tuple[bool, Optional[EvSpec]]] = {}

        self.loader = CsvLoader(ds_path, on_reload=self._run_post_load)

    def _post_load_data_processing(self) -> None:
        """Parse events and reset runtime state after CSV load / reload."""
        self._ev_connected = False
        self._current_spec = None
        self._events = []
        self._event_lookup = {}
        self._parse_events()

    def _parse_events(self) -> None:
        """Parse the CSV DataFrame into a sorted event list."""
        self.ts["start"] = pd.to_datetime(self.ts["start"])

        for _, row in self.ts.iterrows():
            ts = row["start"]
            seconds_from_midnight = ts.hour * SECONDS_PER_HOUR + ts.minute * 60 + ts.second
            iteration_index = int(seconds_from_midnight // self.control_step)

            if pd.notna(row.get("max_cap_kWh")):
                ev_spec = EvSpec(
                    max_cap_kWh=float(row["max_cap_kWh"]),
                    max_charging_kW=float(row["max_charging_kW"]),
                    charger_efficiency=float(row["charger_efficiency"]),
                    discharge_efficiency=float(row["discharge_efficiency"]),
                    v2g_enabled=str(row["v2g_enabled"]).strip().lower() == "true",
                    start_soc=float(row["start_soc"]),
                    target_soc=float(row["target_soc"]),
                    charge_to_target_in_hrs=float(row["target_soc_reach_duration_h"]),
                )
                self._events.append((iteration_index, True, ev_spec))
                self._event_lookup[iteration_index] = (True, ev_spec)
            else:
                self._events.append((iteration_index, False, None))
                self._event_lookup[iteration_index] = (False, None)

        self._events.sort(key=lambda e: e[0])

        logger.debug(
            "EVState '%s': parsed %d events from %s",
            self.name, len(self._events), self.ds_path,
        )
        for it, is_connect, spec in self._events:
            if is_connect:
                logger.debug(
                    "  iter %d: CONNECT (cap=%.1f kWh, charge=%.1f kW, soc=%.2f->%.2f)",
                    it, spec.max_cap_kWh, spec.max_charging_kW,
                    spec.start_soc, spec.target_soc,
                )
            else:
                logger.debug("  iter %d: DISCONNECT", it)


    def setup_spaces(self, state_spaces, action_spaces):
        """Register the bounded [0, 1] EV schedule keys; unbounded/raw fields go to info
        instead (see ``update_state``)."""
        if self.KEY_V2G not in state_spaces:
            state_spaces[self.KEY_V2G] = Box(low=0, high=1, shape=(1,), dtype=np.float32,)
        if self.KEY_START_SOC not in state_spaces:
            state_spaces[self.KEY_START_SOC] = Box(low=0, high=1, shape=(1,), dtype=np.float32,)
        if self.KEY_TARGET_SOC not in state_spaces:
            state_spaces[self.KEY_TARGET_SOC] = Box(low=0, high=1, shape=(1,), dtype=np.float32,)

        return state_spaces, action_spaces

    def update_state(self, states, info=None) -> None:
        """Check for EV events at the current iteration and write schedule."""
        event = self._event_lookup.get(self.iteration)
        if event is not None:
            is_connect, ev_spec = event
            self._ev_connected = is_connect
            self._current_spec = ev_spec if is_connect else None

        # Bounded [0, 1] keys → states; 
        # zero everything when EV is disconnected (avoid stale spec values)
        if self._ev_connected and self._current_spec is not None:
            states[self.KEY_V2G][0] = np.float32(1.0 if self._current_spec.v2g_enabled else 0.0)
            states[self.KEY_START_SOC][0] = np.float32(self._current_spec.start_soc)
            states[self.KEY_TARGET_SOC][0] = np.float32(self._current_spec.target_soc)
        else:
            states[self.KEY_V2G][0] = np.float32(0.0)
            states[self.KEY_START_SOC][0] = np.float32(0.0)
            states[self.KEY_TARGET_SOC][0] = np.float32(0.0)

        # Unbounded / raw keys → info (inter-component communication only)
        # TODO VP 2026.06.10.: Why are these in the info dict, they should be in the state.
        if info is not None:
            connected = self._ev_connected and self._current_spec is not None
            info[self.KEY_CONNECTED] = 1.0 if self._ev_connected else 0.0
            info[self.KEY_MAX_CAP] = self._current_spec.max_cap_kWh if connected else 0.0
            info[self.KEY_MAX_CHARGE] = self._current_spec.max_charging_kW if connected else 0.0
            # mirror bounded keys so LinearEVCharger reads all spec fields from info
            info[self.KEY_CHARGE_EFF] = self._current_spec.charger_efficiency if connected else 0.0
            info[self.KEY_DISCHARGE_EFF] = self._current_spec.discharge_efficiency if connected else 0.0
            info[self.KEY_V2G] = (1.0 if self._current_spec.v2g_enabled else 0.0) if connected else 0.0
            info[self.KEY_START_SOC] = self._current_spec.start_soc if connected else 0.0
            info[self.KEY_TARGET_SOC] = self._current_spec.target_soc if connected else 0.0
            info[self.KEY_CHARGE_TO_TARGET_HRS] = self._current_spec.charge_to_target_in_hrs if connected else 0.0


ComponentRegistry.register('statesource', EVState)
