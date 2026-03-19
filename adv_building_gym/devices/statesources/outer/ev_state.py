"""EV schedule state source that drives EV charger connection events from CSV profiles."""

import logging
from typing import ClassVar, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from ..base import StateSource
from adv_building_gym.config.utils.serializable import ComponentRegistry
from ...infrastructure.ev_charger.ev_spec import EvSpec

logger = logging.getLogger(__name__)

# TODO VP 2026.02.12. : Check this, as this one is still provisional


class EVState(StateSource):
    """State source that reads an EV usage profile CSV and generates
    connect/disconnect signals for the LinearEVCharger via the shared states dict.

    The CSV must have the following columns:
        start, max_cap_kWh, max_charging_kW, charger_efficiency,
        discharge_efficiency, v2g_enabled, start_soc, target_soc,
        target_soc_reach_duration_h

    Rows where ``max_cap_kWh`` is populated represent *connect* events (with
    full EvSpec parameters).  Rows where ``max_cap_kWh`` is NaN represent
    *disconnect* events.

    Events are matched to environment iterations by converting each
    timestamp's time-of-day to an iteration index via ``control_step``.

    Communication with the charger:
        This source writes ``ev_schedule_connected`` and EV spec fields
        (``ev_schedule_max_cap_kWh``, etc.) into the shared states dict.
        The ``LinearEVCharger`` reads these in its ``exec_action`` and calls
        ``set_ev_connected`` on itself when a change is detected.
    """

    _context_params: ClassVar[Set[str]] = {'control_step'}
    _exclude_params: ClassVar[Set[str]] = {
        'iteration', 'ts', '_events', '_event_lookup',
        '_ev_connected', '_current_spec',
    }

    # State keys written by this source (read by LinearEVCharger)
    KEY_CONNECTED = "ev_schedule_connected"
    KEY_MAX_CAP = "ev_schedule_max_cap_kWh"
    KEY_MAX_CHARGE = "ev_schedule_max_charging_kW"
    KEY_CHARGE_EFF = "ev_schedule_charger_eff"
    KEY_DISCHARGE_EFF = "ev_schedule_discharge_eff"
    KEY_V2G = "ev_schedule_v2g"
    KEY_START_SOC = "ev_schedule_start_soc"
    KEY_TARGET_SOC = "ev_schedule_target_soc"

    def __init__(
        self,
        name: str,
        ds_path: str | None = None,
        control_step: float = 300.0,
    ) -> None:
        super().__init__(name, ds_path, control_step)

        self._ev_connected: bool = False
        self._current_spec: Optional[EvSpec] = None

        # Pre-compute event list from CSV
        self._events: List[Tuple[int, bool, Optional[EvSpec]]] = []
        # Lookup dict for O(1) access: iteration_index -> (is_connect, ev_spec)
        self._event_lookup: Dict[int, Tuple[bool, Optional[EvSpec]]] = {}

        if self.ts is not None:
            self._post_load_data_processing()

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
            seconds_from_midnight = ts.hour * 3600 + ts.minute * 60 + ts.second
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
                )
                self._events.append((iteration_index, True, ev_spec))
                self._event_lookup[iteration_index] = (True, ev_spec)
            else:
                self._events.append((iteration_index, False, None))
                self._event_lookup[iteration_index] = (False, None)

        self._events.sort(key=lambda e: e[0])

        logger.info(
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
        """Register bounded EV schedule keys in the observation space.

        Unbounded keys (``max_cap_kWh``, ``max_charging_kW``) and the
        connection flag are written to the shared info dict instead (see
        ``update_state``).  The bounded [0, 1] keys below are useful for
        the control policy and safe for the neural network.
        """
        if self.KEY_CHARGE_EFF not in state_spaces:
            state_spaces[self.KEY_CHARGE_EFF] = Box(
                low=0, high=1, shape=(1,), dtype=np.float32,
            )
        if self.KEY_DISCHARGE_EFF not in state_spaces:
            state_spaces[self.KEY_DISCHARGE_EFF] = Box(
                low=0, high=1, shape=(1,), dtype=np.float32,
            )
        if self.KEY_V2G not in state_spaces:
            state_spaces[self.KEY_V2G] = Box(
                low=0, high=1, shape=(1,), dtype=np.float32,
            )
        if self.KEY_START_SOC not in state_spaces:
            state_spaces[self.KEY_START_SOC] = Box(
                low=0, high=1, shape=(1,), dtype=np.float32,
            )
        if self.KEY_TARGET_SOC not in state_spaces:
            state_spaces[self.KEY_TARGET_SOC] = Box(
                low=0, high=1, shape=(1,), dtype=np.float32,
            )

        return state_spaces, action_spaces

    def update_state(self, states, info=None) -> None:
        """Check for EV events at the current iteration and write schedule."""
        event = self._event_lookup.get(self.iteration)
        if event is not None:
            is_connect, ev_spec = event
            self._ev_connected = is_connect
            self._current_spec = ev_spec if is_connect else None

        # Bounded [0, 1] keys → observation space (states)
        # Zero everything when EV is disconnected so the agent sees a clean
        # signal instead of stale spec values from the previous session.
        if self._ev_connected and self._current_spec is not None:
            states[self.KEY_CHARGE_EFF][0] = np.float32(self._current_spec.charger_efficiency)
            states[self.KEY_DISCHARGE_EFF][0] = np.float32(self._current_spec.discharge_efficiency)
            states[self.KEY_V2G][0] = np.float32(1.0 if self._current_spec.v2g_enabled else 0.0)
            states[self.KEY_START_SOC][0] = np.float32(self._current_spec.start_soc)
            states[self.KEY_TARGET_SOC][0] = np.float32(self._current_spec.target_soc)
        else:
            states[self.KEY_CHARGE_EFF][0] = np.float32(0.0)
            states[self.KEY_DISCHARGE_EFF][0] = np.float32(0.0)
            states[self.KEY_V2G][0] = np.float32(0.0)
            states[self.KEY_START_SOC][0] = np.float32(0.0)
            states[self.KEY_TARGET_SOC][0] = np.float32(0.0)

        # Unbounded / raw keys → info (inter-component communication only)
        if info is not None:
            connected = self._ev_connected and self._current_spec is not None
            info[self.KEY_CONNECTED] = 1.0 if self._ev_connected else 0.0
            info[self.KEY_MAX_CAP] = self._current_spec.max_cap_kWh if connected else 0.0
            info[self.KEY_MAX_CHARGE] = self._current_spec.max_charging_kW if connected else 0.0
            # Mirror bounded keys so LinearEVCharger can read all EV spec
            # fields from a single source.
            info[self.KEY_CHARGE_EFF] = self._current_spec.charger_efficiency if connected else 0.0
            info[self.KEY_DISCHARGE_EFF] = self._current_spec.discharge_efficiency if connected else 0.0
            info[self.KEY_V2G] = (1.0 if self._current_spec.v2g_enabled else 0.0) if connected else 0.0
            info[self.KEY_START_SOC] = self._current_spec.start_soc if connected else 0.0
            info[self.KEY_TARGET_SOC] = self._current_spec.target_soc if connected else 0.0


ComponentRegistry.register('statesource', EVState)
