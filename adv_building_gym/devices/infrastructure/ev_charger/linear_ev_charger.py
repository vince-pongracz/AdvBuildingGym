"""Linear EV charger infrastructure component."""

import logging
from typing import ClassVar, Dict, Optional, Set

import numpy as np
from gymnasium.spaces import Box

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

    # control_step comes from config context
    _context_params: ClassVar[Set[str]] = {'control_step'}

    # Internal state variables - don't serialize
    _exclude_params: ClassVar[Set[str]] = {'iteration', 'soc', 'ev_connected', 'charge_to_target_in_hrs', 'max_cap_kWh'}

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

        if charger_efficiency <= 0 or charger_efficiency > 1:
            raise ValueError("charger_efficiency must be in (0, 1].")
        if discharge_efficiency <= 0 or discharge_efficiency > 1:
            raise ValueError("discharge_efficiency must be in (0, 1].")

    @property
    def max_export_kW(self) -> float:
        return self.max_power_kW if self.v2g_enabled else 0.0

    def setup_spaces(self,
                     state_spaces,
                     action_spaces
                     ):
        """Setup observation and action spaces for EV charger.

        Action convention: positive = consumption (charging from grid), negative = production (V2G to grid).

        Action: charging/discharging rate
        - Range: [-1, 1] if V2G enabled, [0, 1] if V2G disabled
        - Positive: charging EV from grid (consuming energy)
        - Negative: V2G discharge to grid (providing energy, if enabled)
        """

        # Actions
        if "lin_ev_charger_action" not in action_spaces.keys():
            low:float = -1.0 if self.v2g_enabled else 0.0
            action_spaces["lin_ev_charger_action"] = Box(low=low, high=1, shape=(1,), dtype=np.float32)

        # States
        if "ev_soc" not in state_spaces.keys():
            state_spaces["ev_soc"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        if "ev_target_soc" not in state_spaces.keys():
            state_spaces["ev_target_soc"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        if "ev_connected" not in state_spaces.keys():
            # Binary: 0 = not connected, 1 = connected
            state_spaces["ev_connected"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        if "ev_soc_hist" not in state_spaces.keys():
            state_spaces["ev_soc_hist"] = Box(low=0, high=1, shape=(self.history_length,), dtype=np.float32)
        if "ev_charge_to_target_hrs_norm" not in state_spaces.keys():
            # Normalized: 0 = no time left or disconnected, 1 = max_charge_time_hrs remaining
            state_spaces["ev_charge_to_target_hrs_norm"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

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

    def _check_schedule(self, info: Dict) -> None:
        """Check the shared info dict for EV schedule changes from EVState.

        Reads ``ev_schedule_connected`` (and EV spec fields) written by the
        EVState source.  When the schedule differs from the current connection
        state, calls :meth:`set_ev_connected` to apply the transition.
        """
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
                v2g_enabled=bool(info["ev_schedule_v2g"] > 0.5),
                start_soc=float(info["ev_schedule_start_soc"]),
                target_soc=float(info["ev_schedule_target_soc"]),
                charge_to_target_in_hrs=float(info.get("ev_schedule_charge_to_target_hrs", 8.0)),
            )
            logger.debug("EV schedule: CONNECT (cap=%.1f, soc=%.2f->%.2f)",
                        ev_spec.max_cap_kWh, ev_spec.start_soc, ev_spec.target_soc)
            self.set_ev_connected(connected=True, ev_spec=ev_spec)
        else:
            logger.debug("EV schedule: DISCONNECT")
            self.set_ev_connected(connected=False)

    def exec_action(self, actions: Dict, states: Dict, info=None) -> None:
        """Execute charging/discharging action."""
        # Check for EV schedule changes before acting
        self._check_schedule(info or {})

        if not self.ev_connected:
            # EV not connected --> no action
            actions["lin_ev_charger_action"][0] = 0.0
            return

        action = float(np.atleast_1d(actions["lin_ev_charger_action"])[0])

        # Clip action based on V2G capability — only allow discharge when the
        # EV has enough charge.  Discharging a car that still needs charging
        # defeats the purpose of the charging session.
        # V2G is permitted only when SoC >= (target - playroom).
        if not self.v2g_enabled or self.soc < self.target_soc - self.v2g_playroom:
            action = max(0.0, action)
            actions["lin_ev_charger_action"][0] = action

        # Calculate energy transfer in kWh for this timestep
        # action in [-1, 1] maps to [-max_charging_kW, +max_charging_kW]
        power_kW = action * self.max_charging_kW
        energy_kWh = power_kW * (self.control_step / 3600.0)  # Convert to hours

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
            time_hours = self.control_step / 3600.0
            actual_power_kW = actual_energy_kWh / time_hours if time_hours > 0 else 0.0
            action = actual_power_kW / self.max_charging_kW if self.max_charging_kW > 0 else 0.0

            self.soc = 1.0 if new_soc > 1.0 else 0.0
        else:
            self.soc = new_soc

        # Write adjusted action back
        actions["lin_ev_charger_action"][0] = np.float32(action)

    def update_state(self, states: Dict, info=None) -> None:
        """Update observable state."""
        super().update_state(states, info)
        states["ev_soc"][0] = np.float32(self.soc)
        states["ev_target_soc"][0] = np.float32(self.target_soc)
        states["ev_connected"][0] = np.float32(1.0 if self.ev_connected else 0.0)

        # Decrement charge_to_target_in_hrs by control_step (convert seconds to hours)
        if self.ev_connected and self.charge_to_target_in_hrs > 0:
            time_step_hrs = self.control_step / 3600.0
            self.charge_to_target_in_hrs = max(0.0, self.charge_to_target_in_hrs - time_step_hrs)

        # Normalize charge_to_target_in_hrs to [0, 1] for state space
        normalized_time = self.charge_to_target_in_hrs / self.max_charge_time_hrs if self.max_charge_time_hrs > 0 else 0.0
        states["ev_charge_to_target_hrs_norm"][0] = np.float32(np.clip(normalized_time, 0.0, 1.0))

        # Update SoC history (rolling window)
        history = states["ev_soc_hist"]
        history[:-1] = history[1:]
        history[-1] = np.float32(self.soc)

        # Publish static charger params into info for downstream consumers
        # (e.g. EVChargingOnTimeReward) that need them without holding an
        # infrastructure reference.
        if info is not None:
            info["ev_max_charging_kW"] = self.max_charging_kW
            info["ev_max_cap_kWh"] = self.max_cap_kWh
            info["ev_charger_efficiency"] = self.charger_efficiency
            info["ev_max_charge_time_hrs"] = self.max_charge_time_hrs

    def get_electric_consumption(self, actions: Dict) -> float:
        """Get current electric energy consumption from EV charger.

        Sign convention: positive = consumption from grid, negative = production to grid.

        Returns:
            Positive value when charging EV (consuming from grid).
            Negative value when V2G discharging (providing to grid).
        """
        if "lin_ev_charger_action" not in actions or not self.ev_connected:
            return 0.0

        action = float(np.atleast_1d(actions["lin_ev_charger_action"])[0])

        # V2G discharge only allowed when SOC >= target - playroom (consistent with exec_action)
        if not self.v2g_enabled or self.soc < self.target_soc - self.v2g_playroom:
            action = max(0.0, action)

        # Power consumption in kW
        return action * self.max_charging_kW


# Register EvCharger with the component registry
ComponentRegistry.register('infrastructure', LinearEVCharger)
