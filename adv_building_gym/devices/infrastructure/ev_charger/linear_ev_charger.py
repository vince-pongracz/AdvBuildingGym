"""Linear EV charger infrastructure component."""

import logging
from typing import ClassVar, Dict, Optional, Set

import numpy as np
from gymnasium.spaces import Box

from adv_building_gym.utils.constants import SECONDS_PER_HOUR

from ..base import Infrastructure
from adv_building_gym.utils.serializable import ComponentRegistry
from .ev_spec import EvSpec

logger = logging.getLogger(__name__)


class LinearEVCharger(Infrastructure):
    """Electric Vehicle Charger infrastructure component.

    Action convention: positive = consumption (charging from grid), negative = production (V2G to grid).
    EV charger action is in [-1, 1] if V2G enabled, [0, 1] if V2G disabled.

    Models an EV charging station with controllable charging rate.
    Supports vehicle-to-grid (V2G) when action is negative -- in this case it behaves like a battery.

    The EV availability can be controlled via CSV or synthetic schedule.
    """

    # Runtime v2g_enabled flag still gates export; see max_production_kW override.
    POWER_FLOW = "bidirectional"

    # control_step comes from config context
    _context_params: ClassVar[Set[str]] = {'control_step'}

    # Internal state variables - don't serialize
    _exclude_params: ClassVar[Set[str]] = {'iteration', 'soc', 'ev_connected', 'charge_to_target_in_hrs', 'max_cap_kWh', 'actual_power_kW'}

    # Info-dict keys for the per-session corridor exposed to rewards.
    INFO_JUST_DISCONNECTED: ClassVar[str] = "ev_just_disconnected"
    INFO_SESSION_TARGET_SOC: ClassVar[str] = "ev_session_target_soc"
    INFO_SESSION_ACTIVE: ClassVar[str] = "ev_session_active"

    def __init__(self,
                name: str,
                max_power_kW: float,
                max_charging_kW: float,
                control_step: int,
                max_cap_kWh: float = 60.0,
                charger_efficiency: float = 0.92,
                discharge_efficiency: float = 0.92,
                v2g_enabled: bool = True,
                history_length: int = 4,
                start_soc: float = 0.3,
                target_soc: float = 0.9,
                max_charge_time_hrs: float = 24.0,
                v2g_playroom: float = 0.1
                ) -> None:
        """Initialize EV Charger infrastructure.

        Args:
            name: Component identifier
            max_power_kW: Maximum power consumption in kW

            max_charging_kW: Maximum charging power in kW
            max_cap_kWh: EV battery capacity in kWh
            charger_efficiency: Charging efficiency [0, 1]
            discharge_efficiency: Discharging efficiency for V2G [0, 1]
            v2g_enabled: Whether vehicle-to-grid discharge is allowed

            control_step: Control timestep in seconds
            history_length: Number of historical SoC values to track
            start_soc: Initial state of charge [0, 1]
            target_soc: Target state of charge [0, 1]
            max_charge_time_hrs: Maximum charging time in hours (for capping and normalization)
        """
        super().__init__(name, max_power_kW)

        self.max_charging_kW = max_charging_kW
        self.max_cap_kWh = max_cap_kWh
        self.charger_efficiency = charger_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.v2g_enabled = v2g_enabled
        self.v2g_playroom = v2g_playroom
        self.control_step = control_step
        self.history_length = history_length
        self.max_charge_time_hrs = max_charge_time_hrs

        # State variables
        self.soc = start_soc
        self.target_soc = target_soc
        self.ev_connected = False  # Whether EV is connected to charger
        self.charge_to_target_in_hrs = 0.0  # Time remaining to reach target SoC
        self.actual_power_kW = 0.0  # Track actual electric consumption for reporting

        # Per-session corridor bookkeeping. Snapshot at connect; held through
        # the disconnect step so rewards can judge against the original target
        # even after EVState zeros the obs key.
        self._session_step: int = 0
        self._session_total_steps: int = 0
        self._session_start_soc: float = 0.0
        self._session_target_soc: float = 0.0
        self._session_step_soc_gain: float = 0.0
        self._session_active: bool = False
        self._just_disconnected: bool = False

        # Capture initial values so reset() can restore the EV to the same
        # starting condition each episode rather than inheriting whatever
        # state the previous episode (or schedule) left behind.
        self._initial_max_cap_kWh = max_cap_kWh
        self._initial_max_charging_kW = max_charging_kW
        self._initial_charger_efficiency = charger_efficiency
        self._initial_discharge_efficiency = discharge_efficiency
        self._initial_v2g_enabled = v2g_enabled
        self._initial_start_soc = start_soc
        self._initial_target_soc = target_soc

        if charger_efficiency <= 0 or charger_efficiency > 1:
            raise ValueError("charger_efficiency must be in (0, 1].")
        if discharge_efficiency <= 0 or discharge_efficiency > 1:
            raise ValueError("discharge_efficiency must be in (0, 1].")

    @property
    def max_production_kW(self) -> float:
        # Override needed because v2g_enabled is a runtime (per-instance) gate,
        # which POWER_FLOW (class-level) cannot express.
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
            low:float = -1.0 if self.v2g_enabled else 0.0
            action_spaces["a_lin_ev_charger"] = Box(low=low, high=1, shape=(1,), dtype=np.float32)

        # States
        if "s_ev_soc" not in state_spaces.keys():
            state_spaces["s_ev_soc"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        if "s_ev_target_soc" not in state_spaces.keys():
            state_spaces["s_ev_target_soc"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        if "s_ev_connected" not in state_spaces.keys():
            # Binary: 0 = not connected, 1 = connected
            state_spaces["s_ev_connected"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        # Policy-side history of s_ev_soc is assembled by
        # StridedHistoryConnector; env no longer stores it in obs.
        if "s_ev_charge_to_target_hrs_norm" not in state_spaces.keys():
            # Normalized: 0 = no time left or disconnected, 1 = max_charge_time_hrs remaining
            state_spaces["s_ev_charge_to_target_hrs_norm"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Raw maximum charging power (kW) — changes only when a new EV
        # connects with different specs (via EvSpec).
        if "ctxt_ev_max_charging_kW" not in state_spaces.keys():
            state_spaces["ctxt_ev_max_charging_kW"] = Box(
                low=0, high=np.inf, shape=(1,), dtype=np.float32
            )

        # Per-session SoC corridor envelopes (zero when disconnected).  The
        # deadline driving both curves is ``charge_to_target_in_hrs`` from
        # the EvSpec -- the user's hard contract for hitting target_soc.  We
        # deliberately do NOT use the actual disconnect time: that is not
        # assumed observable in deployment, only the requested charging
        # window is.
        #
        # Min curve: lazy back-from-target line that just reaches target_soc
        # by ``charge_to_target_in_hrs`` at max charge rate.  Falling below
        # it means the target is no longer reachable within the contract --
        # EVChargingReward terminates.
        # Max curve: forward-from-start line at max charge rate from connect,
        # capped at 1.0.  Once the agent has been charging at full power
        # since connect, its SoC sits on this curve.
        if "s_ev_soc_min" not in state_spaces.keys():
            state_spaces["s_ev_soc_min"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        if "s_ev_soc_max" not in state_spaces.keys():
            state_spaces["s_ev_soc_max"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        return state_spaces, action_spaces

    def set_target(self, target: float) -> None:
        """Set target state of charge."""
        self.target_soc = float(np.clip(target, 0.0, 1.0))

    def set_ev_connected(self, connected: bool, ev_spec: Optional[EvSpec] = None) -> None:
        """Set EV connection status and update parameters when new EV connects.

        Args:
            connected: Whether EV is connected
            ev_spec: Optional EV specifications to update charger parameters
        """
        self.ev_connected = connected

        if not connected:
            # EV disconnected: reset charge time to 0
            self.charge_to_target_in_hrs = 0.0
        elif ev_spec is not None:
            self.max_cap_kWh = ev_spec.max_cap_kWh
            self.max_charging_kW = ev_spec.max_charging_kW
            self.charger_efficiency = ev_spec.charger_efficiency
            self.discharge_efficiency = ev_spec.discharge_efficiency
            self.v2g_enabled = ev_spec.v2g_enabled
            self.soc = ev_spec.start_soc
            self.target_soc = ev_spec.target_soc
            # Cap charge_to_target_in_hrs at max_charge_time_hrs
            self.charge_to_target_in_hrs = min(
                ev_spec.charge_to_target_in_hrs,
                self.max_charge_time_hrs
            )

    def _snapshot_session(self) -> None:
        """Capture per-session corridor parameters at the moment of connect.

        Called from :meth:`_check_schedule` after :meth:`set_ev_connected`
        has applied the new EvSpec, so ``self.soc`` already equals the EV's
        start SoC.  ``_session_active`` is False when the target is
        unreachable from start at max charge rate -- the corridor obs keys
        are still computed but EVChargingReward suppresses the min-curve
        termination in that case.
        """
        self._session_step = 0
        self._session_start_soc = float(self.soc)
        self._session_target_soc = float(self.target_soc)

        # Convert charge_to_target_in_hrs into a step-budget for the corridor.
        # This is the deadline by which target_soc must be met -- the actual
        # disconnect time may be later, but only the user-requested charging
        # window is assumed observable, so the corridor anchors on it.  At
        # least one step so the deadline is meaningful even for very short
        # charge windows.
        self._session_total_steps = max(1, int(round(self.charge_to_target_in_hrs * SECONDS_PER_HOUR / self.control_step)))

        # Max SoC gain per step in normalised SoC units (charging side; the
        # corridor reflects only charging capability, not V2G).
        if self.max_cap_kWh > 0:
            self._session_step_soc_gain = self.max_charging_kW * self.charger_efficiency * self.control_step / SECONDS_PER_HOUR / self.max_cap_kWh
        else:
            self._session_step_soc_gain = 0.0

        soc_gap = max(0.0, self._session_target_soc - self._session_start_soc)
        max_reachable_gain = self._session_step_soc_gain * self._session_total_steps
        self._session_active = max_reachable_gain >= soc_gap

    def _check_schedule(self, info: Dict) -> None:
        """Check the shared info dict for EV schedule changes from EVState.

        Reads ``ev_schedule_connected`` (and EV spec fields) written by the
        EVState source.  When the schedule differs from the current connection
        state, calls :meth:`set_ev_connected` to apply the transition.
        """
        # Reset the one-shot transition flag every step; it is re-armed
        # below only on a 1->0 transition.
        self._just_disconnected = False

        if "ev_schedule_connected" not in info:
            return  # No EVState in this config

        scheduled_connected = bool(info["ev_schedule_connected"] > 0.5)

        if scheduled_connected == self.ev_connected:
            return  # No change

        if scheduled_connected:
            # Build EvSpec from schedule fields in info dict
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
            # Mark the transition for one step; do NOT clear the session
            # snapshot yet -- EVChargingReward needs the original target
            # to judge the disconnect.
            self._just_disconnected = True

    def exec_action(self, actions: Dict, states: Dict, info=None) -> None:
        """Execute charging/discharging action."""
        # Check for EV schedule changes before acting
        self._check_schedule(info or {})

        if not self.ev_connected:
            # EV not connected --> no action
            actions["a_lin_ev_charger"][0] = 0.0
            self.actual_power_kW = 0.0
            return

        action = float(np.atleast_1d(actions["a_lin_ev_charger"])[0])

        # Clip action based on V2G capability — only allow discharge when the
        # EV has enough charge.  Discharging a car that still needs charging
        # defeats the purpose of the charging session.
        # V2G is permitted only when SoC >= (target - playroom).
        if not self.v2g_enabled or self.soc < self.target_soc - self.v2g_playroom:
            action = max(0.0, action)
            actions["a_lin_ev_charger"][0] = action

        # Calculate energy transfer in kWh for this timestep
        # action in [-1, 1] maps to [-max_charging_kW, +max_charging_kW]
        power_kW = action * self.max_charging_kW
        energy_kWh = power_kW * (self.control_step / SECONDS_PER_HOUR)  # Convert to hours

        # Apply efficiency
        if action > 0:
            # Charging: grid energy * efficiency = battery energy
            soc_change = (energy_kWh * self.charger_efficiency) / self.max_cap_kWh
        else:
            # Discharging (V2G): battery energy / efficiency = grid energy
            # More battery energy consumed than delivered to grid due to efficiency loss
            soc_change = (energy_kWh / self.discharge_efficiency) / self.max_cap_kWh  # Already negative

        # Calculate potential new SoC
        new_soc = self.soc + soc_change

        # If SoC would be clipped, back-calculate the actual action needed
        if new_soc > 1.0 or new_soc < 0.0:
            # Calculate actual SoC change to reach the limit
            if new_soc > 1.0:
                actual_soc_change = 1.0 - self.soc
            else:  # new_soc < 0.0
                actual_soc_change = -self.soc

            # Back-calculate energy and action from actual SoC change
            if action > 0:
                # Charging: soc_change = (energy * efficiency) / capacity
                # => energy = (soc_change * capacity) / efficiency
                actual_energy_kWh = (actual_soc_change * self.max_cap_kWh) / self.charger_efficiency
            else:
                # Discharging: soc_change = (energy / efficiency) / capacity
                # => energy = soc_change * capacity * efficiency
                actual_energy_kWh = actual_soc_change * self.max_cap_kWh * self.discharge_efficiency

            # Back-calculate action from energy
            # energy = power * time, power = action * max_charging_kW
            time_hours = self.control_step / SECONDS_PER_HOUR
            actual_power_kW = actual_energy_kWh / time_hours if time_hours > 0 else 0.0
            action = actual_power_kW / self.max_charging_kW if self.max_charging_kW > 0 else 0.0

            self.soc = 1.0 if new_soc > 1.0 else 0.0
        else:
            self.soc = new_soc

        # Write adjusted action back and store actual power for consumption reporting
        actions["a_lin_ev_charger"][0] = np.float32(action)
        self.actual_power_kW = action * self.max_charging_kW

    def update_state(self, states: Dict, info=None) -> None:
        """Update observable state."""
        super().update_state(states, info)
        states["s_ev_soc"][0] = np.float32(self.soc)
        states["s_ev_target_soc"][0] = np.float32(self.target_soc)
        states["s_ev_connected"][0] = np.float32(1.0 if self.ev_connected else 0.0)
        states["ctxt_ev_max_charging_kW"][0] = np.float32(self.max_charging_kW)

        # Decrement charge_to_target_in_hrs by control_step (convert seconds to hours)
        if self.ev_connected and self.charge_to_target_in_hrs > 0:
            time_step_hrs = self.control_step / SECONDS_PER_HOUR
            self.charge_to_target_in_hrs = max(0.0, self.charge_to_target_in_hrs - time_step_hrs)

        # Normalize charge_to_target_in_hrs to [0, 1] for state space
        normalized_time = self.charge_to_target_in_hrs / self.max_charge_time_hrs if self.max_charge_time_hrs > 0 else 0.0
        states["s_ev_charge_to_target_hrs_norm"][0] = np.float32(np.clip(normalized_time, 0.0, 1.0))

        # Per-session corridor envelopes (zero when disconnected so the obs
        # signal is clean between sessions).
        soc_min, soc_max = 0.0, 0.0
        if self.ev_connected:
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

        # Publish static charger params into info for downstream consumers
        # (e.g. EVChargingOnTimeReward) that need them without holding an
        # infrastructure reference.
        if info is not None:
            info["ctxt_ev_max_charging_kW"] = self.max_charging_kW
            info["ctxt_ev_max_cap_kWh"] = self.max_cap_kWh
            info["ctxt_ev_charger_efficiency"] = self.charger_efficiency
            info["ctxt_ev_max_charge_time_hrs"] = self.max_charge_time_hrs

            # Per-session corridor signals consumed by EVChargingReward.
            # session_target_soc retains the snapshot through the disconnect
            # step (EVState zeros s_ev_target_soc when the schedule fires).
            info[self.INFO_JUST_DISCONNECTED] = bool(self._just_disconnected)
            info[self.INFO_SESSION_ACTIVE] = bool(self._session_active and self.ev_connected)
            info[self.INFO_SESSION_TARGET_SOC] = float(self._session_target_soc if (self.ev_connected or self._just_disconnected) else 0.0)

    def reset(self, states: Dict, info=None) -> None:
        """Restore the charger to its constructor configuration.

        Without this, an EV connection (and its CSV-driven spec overrides)
        from one episode would persist into the next.  EVState will
        re-trigger set_ev_connected on the first step if the new episode
        actually starts with the EV plugged in.
        """
        self.max_cap_kWh = self._initial_max_cap_kWh
        self.max_charging_kW = self._initial_max_charging_kW
        self.charger_efficiency = self._initial_charger_efficiency
        self.discharge_efficiency = self._initial_discharge_efficiency
        self.v2g_enabled = self._initial_v2g_enabled
        self.soc = self._initial_start_soc
        self.target_soc = self._initial_target_soc
        self.ev_connected = False
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

    def get_penalisable_consumption(self, actions: Dict, states: Dict) -> float:
        """Exempt charging when EV is connected and below target SoC."""
        _, consumed_power = self.get_E(actions)
        if consumed_power > 0 and self.ev_connected and self.soc < self.target_soc:
            return 0.0
        return consumed_power

    def get_E(self, actions: Dict) -> tuple[float, float]:
        # self.actual_power_kW is positive when the EV charges -- consumes energy
        if self.actual_power_kW > 0.0:
            # production, consumption
            return 0.0, self.actual_power_kW
        else:
            # self.actual_power_kW is negative when the EV discharges -- produces energy to the others
            return -1.0 * self.actual_power_kW, 0.0


# Register EvCharger with the component registry
ComponentRegistry.register('infrastructure', LinearEVCharger)
