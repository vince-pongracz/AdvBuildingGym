"""Linear EV charger infrastructure component."""

import logging
from typing import ClassVar, Dict, Optional, Set

import numpy as np
from gymnasium.spaces import Box

from ..base import Infrastructure
from adv_building_gym.config.utils.serializable import ComponentRegistry
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
    _exclude_params: ClassVar[Set[str]] = {'iteration', 'soc', 'ev_connected'}

    def __init__(self,
                 name: str,
                 Q_electric_max: float,
                 max_charging_kW: float,
                 max_cap_kWh: float,
                 charger_efficiency: float = 0.92,
                 discharge_efficiency: float = 0.92,
                 v2g_enabled: bool = True,
                 control_step: int = 300,
                 history_length: int = 4,
                 start_soc: float = 0.3,
                 target_soc: float = 0.9,
                 ) -> None:
        """Initialize EV Charger infrastructure.

        Args:
            name: Component identifier
            Q_electric_max: Maximum power consumption in kW

            max_charging_kW: Maximum charging power in kW
            max_cap_kWh: EV battery capacity in kWh
            charger_efficiency: Charging efficiency [0, 1]
            discharge_efficiency: Discharging efficiency for V2G [0, 1]
            v2g_enabled: Whether vehicle-to-grid discharge is allowed

            control_step: Control timestep in seconds
            history_length: Number of historical SoC values to track
            start_soc: Initial state of charge [0, 1]
            target_soc: Target state of charge [0, 1]
        """
        super().__init__(name, Q_electric_max)

        self.max_charging_kW = max_charging_kW
        self.max_cap_kWh = max_cap_kWh
        self.charger_efficiency = charger_efficiency
        self.discharge_efficiency = discharge_efficiency
        self.v2g_enabled = v2g_enabled
        self.control_step = control_step
        self.history_length = history_length

        # State variables
        self.soc = start_soc
        self.target_soc = target_soc
        self.ev_connected = True  # Whether EV is connected to charger

        if charger_efficiency <= 0 or charger_efficiency > 1:
            raise ValueError("charger_efficiency must be in (0, 1].")
        if discharge_efficiency <= 0 or discharge_efficiency > 1:
            raise ValueError("discharge_efficiency must be in (0, 1].")

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

        if connected and ev_spec is not None:
            self.max_cap_kWh = ev_spec.max_cap_kWh
            self.max_charging_kW = ev_spec.max_charging_kW
            self.charger_efficiency = ev_spec.charger_efficiency
            self.discharge_efficiency = ev_spec.discharge_efficiency
            self.v2g_enabled = ev_spec.v2g_enabled
            self.soc = ev_spec.start_soc
            self.target_soc = ev_spec.target_soc

    def exec_action(self, actions: Dict, states: Dict) -> None:
        """Execute charging/discharging action."""

        if not self.ev_connected:
            # EV not connected --> no action
            actions["lin_ev_charger_action"][0] = 0.0
            return

        action = float(np.atleast_1d(actions["lin_ev_charger_action"])[0])

        # Clip action based on V2G capability -- and write it back to actions
        if not self.v2g_enabled:
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

    def update_state(self, states: Dict) -> None:
        """Update observable state."""
        states["ev_soc"][0] = np.float32(self.soc)
        states["ev_target_soc"][0] = np.float32(self.target_soc)
        states["ev_connected"][0] = np.float32(1.0 if self.ev_connected else 0.0)

        # Update SoC history (rolling window)
        history = states["ev_soc_hist"]
        history[:-1] = history[1:]
        history[-1] = np.float32(self.soc)

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

        # Clip action based on V2G capability
        if not self.v2g_enabled:
            action = max(0.0, action)

        # Power consumption in kW
        return action * self.max_charging_kW


# Register EvCharger with the component registry
ComponentRegistry.register('infrastructure', LinearEVCharger)
