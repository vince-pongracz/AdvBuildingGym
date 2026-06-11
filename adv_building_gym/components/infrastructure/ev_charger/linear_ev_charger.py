"""Linear EV charger infrastructure component."""

import logging
from typing import ClassVar, Dict, Optional, Set

import numpy as np
from gymnasium.spaces import Box

from adv_building_gym._common.constants import SECONDS_PER_HOUR

from ..base import Infrastructure
from adv_building_gym.components.registry import ComponentRegistry
from .ev_spec import EvSpec

logger = logging.getLogger(__name__)


class LinearEVCharger(Infrastructure):
    """EV charging station with controllable rate and optional V2G.

    Action ``a_lin_ev_charger`` in [-1, 1] (V2G) or [0, 1]: positive=charge (consume),
    negative=V2G export. The connected EV is an :class:`EvSpec` (``None`` = unplugged);
    EVState pushes connect/disconnect events via info, and :meth:`_check_schedule`
    rebuilds the EvSpec and calls :meth:`set_ev_connected`.
    """
    # TODO VP 2026.06.10.: Connect disconnect event publish mechanism review.

    # Runtime v2g_enabled flag still gates export; see max_production_kW override.
    POWER_FLOW = "bidirectional"

    # control_step comes from config context
    _context_params: ClassVar[Set[str]] = {'control_step'}

    # Internal state variables - don't serialize
    _exclude_params: ClassVar[Set[str]] = {'iteration', 'soc', 'ev_spec', 'charge_to_target_in_hrs', 'actual_power_kW'}

    # Info-dict keys for the per-session corridor exposed to rewards.
    INFO_JUST_DISCONNECTED: ClassVar[str] = "ev_just_disconnected"
    INFO_SESSION_TARGET_SOC: ClassVar[str] = "ev_session_target_soc"
    INFO_SESSION_ACTIVE: ClassVar[str] = "ev_session_active"

    def __init__(self,
                name: str,
                max_power_kW: float,
                control_step: int,
                max_charge_time_hrs: float = 24.0,
                v2g_enabled: bool = True,
                v2g_playroom: float = 0.1,
                ) -> None:
        """Initialize EV Charger infrastructure.

        Args:
            name: Component identifier.
            max_power_kW: Charger hardware electrical rating in kW.
            control_step: Control timestep in seconds.
            max_charge_time_hrs: Upper bound on the per-session deadline; also
                the normaliser for ``s_ev_charge_to_target_hrs_norm``.
            v2g_enabled: Charger-side V2G capability. Drives the action-space
                lower bound at construction. Effective V2G at runtime requires
                both this flag and ``ev_spec.v2g_enabled``.
            v2g_playroom: SoC headroom above target below which V2G is
                disallowed even when both V2G flags are set.
        """
        super().__init__(name, max_power_kW)

        self.control_step = control_step
        self.max_charge_time_hrs = max_charge_time_hrs
        self.v2g_enabled = v2g_enabled
        self.v2g_playroom = v2g_playroom

        # Connected EV; None ⇔ no EV currently plugged in.
        self.ev_spec: Optional[EvSpec] = None

        self.soc: float = 0.0
        self.charge_to_target_in_hrs: float = 0.0
        self.actual_power_kW: float = 0.0

        # Per-session corridor bookkeeping; snapshot at connect, held through
        # the disconnect step so rewards keep the original target.
        self._session_step: int = 0
        self._session_total_steps: int = 0
        self._session_start_soc: float = 0.0
        self._session_target_soc: float = 0.0
        self._session_step_soc_gain: float = 0.0
        self._session_active: bool = False
        self._just_disconnected: bool = False

    # ----- Connection state helpers -----

    @property
    def is_connected(self) -> bool:
        return self.ev_spec is not None

    @property
    def target_soc(self) -> float:
        return self.ev_spec.target_soc if self.ev_spec is not None else 0.0

    @property
    def effective_max_charging_kW(self) -> float:
        # Charger hardware cap and EV acceptance both bind.
        if self.ev_spec is None:
            return 0.0
        return min(self.max_power_kW, self.ev_spec.max_charging_kW)

    @property
    def effective_v2g(self) -> bool:
        return self.v2g_enabled and (self.ev_spec.v2g_enabled if self.ev_spec is not None else False)

    @property
    def max_production_kW(self) -> float:
        # v2g_enabled is a per-instance gate that class-level POWER_FLOW can't express.
        return self.max_power_kW if self.v2g_enabled else 0.0

    def setup_spaces(self, state_spaces, action_spaces):
        """Setup observation and action spaces for EV charger.

        Action convention: positive = consumption (charging from grid), negative = production (V2G to grid).

        Action: charging/discharging rate
        - Range: [-1, 1] if V2G enabled, [0, 1] if V2G disabled
        - Positive: charging EV from grid (consuming energy)
        - Negative: V2G discharge to grid (providing energy, if enabled)
        """

        # Actions
        if "a_lin_ev_charger" not in action_spaces.keys():
            low: float = -1.0 if self.v2g_enabled else 0.0
            action_spaces["a_lin_ev_charger"] = Box(low=low, high=1, shape=(1,), dtype=np.float32)

        # States
        if "s_ev_soc" not in state_spaces.keys():
            state_spaces["s_ev_soc"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        if "s_ev_target_soc" not in state_spaces.keys():
            state_spaces["s_ev_target_soc"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        if "s_ev_connected" not in state_spaces.keys():
            # Binary: 0 = not connected, 1 = connected
            state_spaces["s_ev_connected"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        if "s_ev_charge_to_target_hrs_norm" not in state_spaces.keys():
            # Normalised: 0 = none/disconnected, 1 = max_charge_time_hrs remaining
            state_spaces["s_ev_charge_to_target_hrs_norm"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Max charging power (kW) — changes per connected EV.
        if "ctxt_ev_max_charging_kW" not in state_spaces.keys():
            state_spaces["ctxt_ev_max_charging_kW"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        # Effective V2G = charger.v2g_enabled AND ev_spec.v2g_enabled. Unlike
        # ctxt_ev_schedule_v2g (EV-side only), this is the actionable composite
        # (0 if unplugged or either side forbids V2G).
        if "ctxt_ev_v2g_effective" not in state_spaces.keys():
            state_spaces["ctxt_ev_v2g_effective"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Per-session SoC corridor (zero when disconnected). Deadline is
        # ``charge_to_target_in_hrs`` (the user's contract), not the actual
        # disconnect time (not assumed observable).
        # Min: back-from-target line hitting target_soc by the deadline at max
        # rate; below it the target is unreachable → EVChargingReward terminates.
        # Max: forward-from-start line at max rate, capped at 1.0.
        if "s_ev_soc_min" not in state_spaces.keys():
            state_spaces["s_ev_soc_min"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        if "s_ev_soc_max" not in state_spaces.keys():
            state_spaces["s_ev_soc_max"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        return state_spaces, action_spaces

    def set_ev_connected(self, connected: bool, ev_spec: Optional[EvSpec] = None) -> None:
        """Apply an EV connect/disconnect transition.

        On connect, ``self.soc`` = ``ev_spec.start_soc`` so the corridor anchors
        on the real start SoC; on disconnect, ``self.soc`` is cleared.
        """
        if not connected:
            self.ev_spec = None
            self.charge_to_target_in_hrs = 0.0
            self.soc = 0.0
        elif ev_spec is not None:
            self.ev_spec = ev_spec
            self.soc = ev_spec.start_soc
            # Cap charge_to_target_in_hrs at max_charge_time_hrs
            self.charge_to_target_in_hrs = min(ev_spec.charge_to_target_in_hrs, self.max_charge_time_hrs)

    def _snapshot_session(self) -> None:
        """Snapshot corridor params at connect (after set_ev_connected, so self.soc
        is the start SoC). ``_session_active`` is False when target is unreachable
        from start at max rate — corridor keys still computed, but EVChargingReward
        suppresses the min-curve termination then.
        """
        assert self.ev_spec is not None, "_snapshot_session requires a connected EV"

        self._session_step = 0
        self._session_start_soc = self.soc
        self._session_target_soc = self.target_soc

        # Step budget for the corridor from charge_to_target_in_hrs (deadline to
        # hit target_soc). At least one step so it's meaningful for short windows.
        self._session_total_steps = max(1, int(round(self.charge_to_target_in_hrs * SECONDS_PER_HOUR / self.control_step)))

        # Max SoC gain per step (charging only, not V2G), using the connected EV's
        # capacity/efficiency so the corridor tracks the actual session.
        if self.ev_spec.max_cap_kWh > 0:
            self._session_step_soc_gain = (
                self.effective_max_charging_kW * self.ev_spec.charger_efficiency
                * self.control_step / SECONDS_PER_HOUR / self.ev_spec.max_cap_kWh
            )
        else:
            self._session_step_soc_gain = 0.0

        soc_gap = max(0.0, self._session_target_soc - self._session_start_soc)
        max_reachable_gain = self._session_step_soc_gain * self._session_total_steps
        self._session_active = max_reachable_gain >= soc_gap

    def _check_schedule(self, info: Dict) -> None:
        """Apply EV connect/disconnect from EVState's info dict.

        Reads ``ev_schedule_connected`` (+ spec fields); on a change calls
        :meth:`set_ev_connected`.
        """
        # reset one-shot flag; re-armed below only on a 1->0 transition
        self._just_disconnected = False

        if "ev_schedule_connected" not in info:
            return  # No EVState in this config

        scheduled_connected = bool(info["ev_schedule_connected"] > 0.5)

        if scheduled_connected == self.is_connected:
            return  # No change

        if scheduled_connected:
            # build EvSpec from info fields
            ev_spec = EvSpec(
                max_cap_kWh=float(info["ev_schedule_max_cap_kWh"]),
                max_charging_kW=float(info["ev_schedule_max_charging_kW"]),
                charger_efficiency=float(info["ev_schedule_charger_eff"]),
                discharge_efficiency=float(info["ev_schedule_discharge_eff"]),
                v2g_enabled=bool(info["ctxt_ev_schedule_v2g"] > 0.5),
                start_soc=float(info["ctxt_ev_schedule_start_soc"]),
                target_soc=float(info["ctxt_ev_schedule_target_soc"]),
                charge_to_target_in_hrs=float(info.get("ev_schedule_charge_to_target_hrs", 8.0)),
            )
            logger.debug("EV schedule: CONNECT (cap=%.1f, soc=%.2f->%.2f)",
                        ev_spec.max_cap_kWh, ev_spec.start_soc, ev_spec.target_soc)
            self.set_ev_connected(connected=True, ev_spec=ev_spec)
            self._snapshot_session()
        else:
            logger.debug("EV schedule: DISCONNECT")
            self.set_ev_connected(connected=False)
            # mark transition for one step; keep the session snapshot so
            # EVChargingReward can judge the disconnect against the target
            self._just_disconnected = True

    def exec_action(self, actions: Dict, states: Dict, info=None) -> None:
        """Execute charging/discharging action."""
        # apply any EV schedule change first
        self._check_schedule(info or {})

        if not self.is_connected:
            # no EV → no action
            actions["a_lin_ev_charger"][0] = 0.0
            self.actual_power_kW = 0.0
            return

        action = float(np.atleast_1d(actions["a_lin_ev_charger"])[0])

        # Allow discharge (V2G) only when both sides allow it AND SoC >= target - playroom;
        # discharging a car that still needs charging defeats the session.
        if not self.effective_v2g or self.soc < self.target_soc - self.v2g_playroom:
            action = max(0.0, action)
            actions["a_lin_ev_charger"][0] = action

        # action in [-1, 1] → ±effective_max_charging_kW (= min(charger, EV cap)); energy in kWh
        eff_max_kW = self.effective_max_charging_kW
        power_kW = action * eff_max_kW
        energy_kWh = power_kW * (self.control_step / SECONDS_PER_HOUR)

        # efficiency from the connected EV's spec
        if action > 0:
            # charging: grid energy * efficiency = battery energy
            soc_change = (energy_kWh * self.ev_spec.charger_efficiency) / self.ev_spec.max_cap_kWh
        else:
            # V2G: battery energy / efficiency = grid energy (already negative)
            soc_change = (energy_kWh / self.ev_spec.discharge_efficiency) / self.ev_spec.max_cap_kWh

        # tentative new SoC
        new_soc = self.soc + soc_change

        # if clipped, back-calculate the action that reaches the bound
        if new_soc > 1.0 or new_soc < 0.0:
            if new_soc > 1.0:
                actual_soc_change = 1.0 - self.soc
            else:  # new_soc < 0.0
                actual_soc_change = -self.soc

            # invert soc_change → energy
            if action > 0:
                # charging: energy = soc_change * capacity / efficiency
                actual_energy_kWh = (actual_soc_change * self.ev_spec.max_cap_kWh) / self.ev_spec.charger_efficiency
            else:
                # V2G: energy = soc_change * capacity * efficiency
                actual_energy_kWh = actual_soc_change * self.ev_spec.max_cap_kWh * self.ev_spec.discharge_efficiency

            # energy → action (energy = action * eff_max_kW * time)
            time_hours = self.control_step / SECONDS_PER_HOUR
            actual_power_kW = actual_energy_kWh / time_hours if time_hours > 0 else 0.0
            action = actual_power_kW / eff_max_kW if eff_max_kW > 0 else 0.0

            self.soc = 1.0 if new_soc > 1.0 else 0.0
        else:
            self.soc = new_soc

        # write clipped action back; store actual power
        actions["a_lin_ev_charger"][0] = np.float32(action)
        self.actual_power_kW = action * eff_max_kW

    def update_state(self, states: Dict, info=None) -> None:
        """Update observable state."""
        super().update_state(states, info)
        states["s_ev_soc"][0] = np.float32(self.soc)
        states["s_ev_target_soc"][0] = np.float32(self.target_soc)
        states["s_ev_connected"][0] = np.float32(1.0 if self.is_connected else 0.0)
        states["ctxt_ev_max_charging_kW"][0] = np.float32(self.effective_max_charging_kW)
        states["ctxt_ev_v2g_effective"][0] = np.float32(1.0 if self.effective_v2g else 0.0)

        # decrement deadline by control_step (s→h)
        if self.is_connected and self.charge_to_target_in_hrs > 0:
            time_step_hrs = self.control_step / SECONDS_PER_HOUR
            self.charge_to_target_in_hrs = max(0.0, self.charge_to_target_in_hrs - time_step_hrs)

        # normalise to [0, 1]
        normalized_time = self.charge_to_target_in_hrs / self.max_charge_time_hrs if self.max_charge_time_hrs > 0 else 0.0
        states["s_ev_charge_to_target_hrs_norm"][0] = np.float32(np.clip(normalized_time, 0.0, 1.0))

        # corridor envelopes (zero when disconnected)
        soc_min, soc_max = 0.0, 0.0
        if self.is_connected:
            self._session_step += 1
            steps_to_deadline = max(0, self._session_total_steps - self._session_step)
            soc_max = float(np.clip(
                self._session_start_soc + self._session_step_soc_gain * self._session_step,
                0.0, 1.0,
            ))
            if steps_to_deadline > 0:
                soc_min = float(np.clip(
                    self._session_target_soc - self._session_step_soc_gain * steps_to_deadline,
                    0.0, 1.0,
                ))
            else:
                soc_min = float(np.clip(self._session_target_soc, 0.0, 1.0))
        states["s_ev_soc_min"][0] = np.float32(soc_min)
        states["s_ev_soc_max"][0] = np.float32(soc_max)

        # publish per-EV params to info for rewards (e.g. EVChargingOnTimeReward)
        # that need them without an infra reference
        if info is not None:
            info["ctxt_ev_max_charging_kW"] = self.effective_max_charging_kW
            info["ctxt_ev_max_cap_kWh"] = self.ev_spec.max_cap_kWh if self.is_connected else 0.0
            info["ctxt_ev_charger_efficiency"] = self.ev_spec.charger_efficiency if self.is_connected else 0.0
            info["ctxt_ev_max_charge_time_hrs"] = self.max_charge_time_hrs
            info["ctxt_ev_v2g_effective"] = bool(self.effective_v2g)

            # corridor signals for EVChargingReward; session_target_soc keeps the
            # snapshot through the disconnect step (EVState zeros s_ev_target_soc).
            info[self.INFO_JUST_DISCONNECTED] = bool(self._just_disconnected)
            info[self.INFO_SESSION_ACTIVE] = bool(self._session_active and self.is_connected)
            info[self.INFO_SESSION_TARGET_SOC] = float(self._session_target_soc if (self.is_connected or self._just_disconnected) else 0.0)

    def reset(self, states: Dict, info=None) -> None:
        """Reset to a fresh, disconnected state; EVState re-connects on step 1 if needed."""
        self.ev_spec = None
        self.soc = 0.0
        self.charge_to_target_in_hrs = 0.0
        self.actual_power_kW = 0.0
        self._session_step = 0
        self._session_total_steps = 0
        self._session_start_soc = 0.0
        self._session_target_soc = 0.0
        self._session_step_soc_gain = 0.0
        self._session_active = False
        self._just_disconnected = False
        super().reset(states, info)

    def get_E(self, actions: Dict) -> tuple[float, float]:
        # self.actual_power_kW is positive when the EV charges -- consumes energy
        if self.actual_power_kW > 0.0:
            # production, consumption
            return 0.0, self.actual_power_kW
        else:
            # self.actual_power_kW is negative when the EV discharges -- produces energy to the others
            return -1.0 * self.actual_power_kW, 0.0


# register with ComponentRegistry
ComponentRegistry.register('infrastructure', LinearEVCharger)
