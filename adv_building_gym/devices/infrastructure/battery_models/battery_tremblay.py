import logging
from typing import ClassVar, Dict, Optional, Set

import numpy as np
from gymnasium.spaces import Box

from ..base import Infrastructure
from adv_building_gym.config.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)

# NOTE VP 2026.01.15. : Seems easier to use the Tremblay model than the PyBaMM
# However, PyBaMM seems more complex (maybe more accurate phyisically) for battery simulation, has more options to configure.

# NOTE VP 2026.01.15. : Battery discharge modelling: https://dl.acm.org/doi/pdf/10.1145/3592149.3592156, 
# Evaluation and Extension of ns-3 Battery Framework, 
# ALBERTO GALLEGOS RAMONET, Tokushima University, Tokushima, Tokushima, Japan, 
# ALEXANDER GUZMAN-URBINA, Tohoku University, Sendai, Miyagi, Japan, 
# KAZUHIKO KINOSHITA, Tokushima University, Tokushima, Tokushima, Japan

# NOTE VP 2026.01.15. : Battery modelling framework: PyBaMM (Python Battery Mathematical Modelling)
# https://pybamm.readthedocs.io/en/latest/
# https://github.com/pybamm-team/PyBaMM/blob/develop/docs/source/examples/notebooks/getting_started/tutorial-5-run-experiments.ipynb

# NOTE VP 2026.01.15. : Trembley battery model reference
# Paper: A Generic Battery Model for the Dynamic Simulation of Hybrid Electric Vehicles
# Link: https://ieeexplore.ieee.org/document/4544139
# "Shepherd developed an equation to describe the electrochemical behaviour of a 
# battery directly in terms of terminal voltage, open circuit voltage, 
# internal resistance, discharge current and state-of-charge" -- from the Tremblay paper

# NOTE VP 2026.01.23. : Tremblay / Shepard model
# Paper: Modeling Stationary Lithium-Ion Batteries for Optimization and Predictive Control
# Link: https://ieeexplore.ieee.org/document/7935755

# NOTE VP 2026.01.15. : Battery modelling approaches: 
# Tremblay, Shepherd, DNF, SPM

# NOTE VP 2026.01.20. : It is optional to add this in a .py class
# - PyBaMM (more complex, maybe keep it as an option later, but do not implement that)

# TODO VP 2026.01.15. : Implement a reward function, which rewards shorter fully charged and fully drained times of the battery.


class BatteryTremblay(Infrastructure):
    """Battery infrastructure using Tremblay model with series-parallel cell configuration.

    Tremblay model (IEEE 2008) describes battery terminal voltage as a function of
    state-of-charge, current, and internal parameters. This provides more realistic
    charge/discharge behavior than simple linear models.

    Tremblay discharge equation (per cell):
        E = E0 - K*(Q/(Q-it))*it - K*(Q/(Q-it))*i + A*exp(-B*it)

    Where:
        E0 = battery constant voltage (V per cell)
        K = polarization constant (V/Ah)
        Q = cell capacity (Ah)
        it = extracted capacity (Ah) = (1-SoC)*Q
        i = cell current (A)
        A = exponential zone amplitude (V)
        B = exponential zone time constant inverse (1/Ah)

    Series-Parallel Configuration (NsNp):
        - n_series cells in series per string (voltages add)
        - n_parallel strings in parallel (capacities/currents add)
        - Pack voltage: V_pack = n_series * V_cell
        - Pack capacity: Q_pack = n_parallel * Q_cell
        - Pack resistance: R_pack = (n_series * R_cell) / n_parallel
        - Cell current: I_cell = I_pack / n_parallel
    """

    # control_step comes from config context
    _context_params: ClassVar[Set[str]] = {'control_step'}

    # Internal state variables - don't serialize (derived from cell configuration)
    _exclude_params: ClassVar[Set[str]] = {
        'iteration', 'soc', 'current_amps', 'actual_voltage',
        'actual_power_kW', 'nominal_voltage', 'max_cap_Ah', 'max_cap_kWh', 'R_pack'
    }

    def __init__(self, name: str,
                 Q_electric_max: float = 19.0,  # Max charge/discharge power in kW (400V × 48A)
                 cell_capacity_Ah: float = 3.5,  # Single cell capacity in Ah (typical 21700)
                 max_charge_amps: float = 48.0,  # Max pack current in A
                 max_charge_voltage: float = 420.0,  # Max pack voltage in V
                 start_soc_percentage: float = 0.3,  # Initial SoC [0, 1]
                 target_soc: float = 1.0,  # Target SoC
                 control_step: int = 300,  # Timesteps in seconds
                 max_charge_rate: float = 1.5,  # C-rate limit
                 history_length: int = 4,  # Number of past SoC values to track
                 # Tremblay model parameters (Li-ion LFP defaults, per cell)
                 E0: float = 3.2,  # Constant voltage (V per cell)
                 K: float = 0.009,  # Polarization constant (V/Ah)
                 A: float = 0.468,  # Exponential zone amplitude (V)
                 B: float = 3.529,  # Exponential zone time constant inverse (1/Ah)
                 R_cell: float = 0.01,  # Single cell internal resistance (Ohms)
                 # Cell configuration (NsNp topology) - defaults for ~14 kWh pack
                 # 125s10p: 125 × 3.2V = 400V, 10 × 3.5Ah = 35Ah → 14 kWh
                 n_series: int = 125,  # Number of cells in series per string
                 n_parallel: int = 10,  # Number of parallel strings
                 # Efficiency parameters
                 charge_efficiency: float = 0.95,  # Coulombic efficiency for charging
                 discharge_efficiency: float = 0.95,  # Coulombic efficiency for discharging
                 # Operating limits -- prevent battery damage
                 soc_min: float = 0.1,  # Minimum SoC to prevent damage
                 soc_max: float = 0.95,  # Maximum SoC to prevent damage
                 ) -> None:
        super().__init__(name, Q_electric_max)

        self.cell_capacity_Ah = cell_capacity_Ah
        self.max_charge_amps = max_charge_amps
        self.max_charge_voltage = max_charge_voltage

        # State of Charge (SoC) as a percentage [0, 1]
        self.soc = start_soc_percentage
        self.target_soc = target_soc
        self.history_length = history_length
        self.control_step = control_step
        # Link: https://www.batterypowertips.com/how-to-read-battery-discharge-curves-faq/
        # Charge Rate (C‐rate) is the rate of charge or discharge of a battery relative to its rated capacity.
        # For example, a 1C rate will fully charge or discharge a battery in 1 hour.
        # At a discharge rate of 0.5C, a battery will be fully discharged in 2 hours.
        self.max_charge_rate = max_charge_rate

        # Tremblay model parameters (per cell)
        self.E0 = E0
        self.K = K
        self.A = A
        self.B = B
        self.R_cell = R_cell

        # Cell configuration (NsNp topology)
        self.n_series = n_series
        self.n_parallel = n_parallel

        # Derived pack properties from cell configuration
        # Pack voltage = series cells × cell voltage
        self.nominal_voltage = E0 * n_series
        # Pack capacity = parallel strings × cell capacity
        self.max_cap_Ah = n_parallel * cell_capacity_Ah
        # Pack resistance = (series cells × cell resistance) / parallel strings
        self.R_pack = (n_series * R_cell) / n_parallel
        # Pack energy capacity in kWh
        self.max_cap_kWh = (self.max_cap_Ah * self.nominal_voltage) / 1000.0

        # Efficiency parameters
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency

        # Operating limits
        self.soc_min = soc_min
        self.soc_max = soc_max

        # Track actual current for voltage calculation
        self.current_amps = 0.0
        self.actual_voltage = self._calculate_terminal_voltage(self.soc, 0.0)
        self.actual_power_kW = 0.0  # Track actual power for consumption reporting


    def setup_spaces(self,
                     state_spaces,
                     action_spaces
                     ):
        # Action
        if "battery_action" not in action_spaces.keys():
            action_spaces["battery_action"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # States
        if "battery_pct" not in state_spaces.keys():
            state_spaces["battery_pct"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        if "battery_target_pct" not in state_spaces.keys():
            state_spaces["battery_target_pct"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        if "battery_pct_hist" not in state_spaces.keys():
            state_spaces["battery_pct_hist"] = Box(low=0, high=1, shape=(self.history_length,), dtype=np.float32)

        return state_spaces, action_spaces

    # TODO VP 2026.01.13. : How to set it dinamically, at eval?
    def set_target(self, target: Optional[float] = None) -> None:
        self.target_soc = target

    def _calculate_terminal_voltage(self, soc: float, pack_current: float) -> float:
        """Calculate battery terminal voltage using Tremblay model at cell level.

        The Tremblay equations are applied per cell, then scaled to pack level.
        Pack current is divided among parallel strings to get cell current.

        For discharge (current > 0):
            E = E0 - K*(Q/(Q-it))*it - K*(Q/(Q-it))*i + A*exp(-B*it)
        For charge (current < 0):
            E = E0 - K*(Q/(it+0.1*Q))*it - K*(Q/(Q-it))*|i| + A*exp(-B*it)

        Terminal voltage: V = E - R*i (discharge) or V = E + R*|i| (charge)

        Args:
            soc: State of charge [0, 1]
            pack_current: Pack current in Amps (positive = discharge, negative = charge)

        Returns:
            Terminal voltage of the battery pack in Volts
        """
        # Cell current = pack current divided among parallel strings
        cell_current = pack_current / self.n_parallel if self.n_parallel > 0 else pack_current

        # Extracted capacity (Ah) at cell level - how much has been discharged
        it = (1.0 - soc) * self.cell_capacity_Ah
        Q = self.cell_capacity_Ah

        # Prevent division by zero at extreme SoC values
        epsilon = 0.001 * Q

        if cell_current >= 0:  # Discharge
            # Tremblay discharge equation (per cell)
            denom = max(Q - it, epsilon)
            E_cell = (self.E0
                      - self.K * (Q / denom) * it
                      - self.K * (Q / denom) * cell_current
                      + self.A * np.exp(-self.B * it))
            # Cell terminal voltage with IR drop
            V_cell = E_cell - self.R_cell * cell_current
        else:  # Charge
            # Tremblay charge equation (modified for charging, per cell)
            abs_current = abs(cell_current)
            denom_charge = it + 0.1 * Q
            denom_discharge = max(Q - it, epsilon)
            E_cell = (self.E0
                      - self.K * (Q / denom_charge) * it
                      - self.K * (Q / denom_discharge) * abs_current
                      + self.A * np.exp(-self.B * it))
            # Cell terminal voltage rises during charging
            V_cell = E_cell + self.R_cell * abs_current

        # Scale to pack voltage: series cells multiply voltage
        V_pack = V_cell * self.n_series
        return float(np.clip(V_pack, 0.0, self.max_charge_voltage * 1.2))

    def _calculate_max_current(self, is_charging: bool) -> float:
        """Calculate maximum allowable pack current based on C-rate and current limits.

        C-rate is applied at cell level, then scaled by parallel strings for pack current.
        Pack current limit = n_parallel × cell C-rate limit

        Args:
            is_charging: True if charging, False if discharging

        Returns:
            Maximum pack current in Amps
        """
        # C-rate limit per cell: max_charge_rate * cell_capacity gives max cell current
        cell_c_rate_limit = self.max_charge_rate * self.cell_capacity_Ah

        # Pack current limit = parallel strings × cell current limit
        pack_c_rate_limit = self.n_parallel * cell_c_rate_limit

        if is_charging:
            return min(self.max_charge_amps, pack_c_rate_limit)
        else:
            # Discharge can also be limited by C-rate
            return min(self.max_charge_amps, pack_c_rate_limit)

    def exec_action(self, actions: Dict, states: Dict) -> None:
        """Execute battery charge/discharge action using Tremblay model.
    
        Action is in [-1, 1]:
            - Positive: charge battery (consume power from grid)
            - Negative: discharge battery (provide power to grid)

        The action represents fraction of max power (Q_electric_max).
        """
        action = float(np.atleast_1d(actions["battery_action"])[0])

        # Determine if charging or discharging
        is_charging = action > 0

        # Calculate requested power in kW
        requested_power_kW = abs(action) * self.Q_electric_max

        # Get current terminal voltage for power-to-current conversion
        # Use small test current in the right direction to estimate voltage
        test_current = 1.0 if not is_charging else -1.0
        estimated_voltage = self._calculate_terminal_voltage(self.soc, test_current)

        # Convert power to current: P = V * I, so I = P / V
        if estimated_voltage > 0:
            requested_current = (requested_power_kW * 1000) / estimated_voltage
        else:
            requested_current = 0.0

        # Apply current limits based on C-rate and max charge current
        max_current = self._calculate_max_current(is_charging)
        actual_current = min(requested_current, max_current)

        # Calculate energy transferred in this control step
        # Energy (Ah) = Current (A) * Time (h)
        time_hours = self.control_step / 3600.0
        delta_Ah = actual_current * time_hours

        # Apply efficiency losses
        if is_charging:
            delta_Ah_effective = delta_Ah * self.charge_efficiency
        else:
            delta_Ah_effective = delta_Ah / self.discharge_efficiency

        # Convert Ah change to SoC change
        delta_soc = delta_Ah_effective / self.max_cap_Ah if self.max_cap_Ah > 0 else 0.0

        # Apply SoC change and clip to valid range
        old_soc = self.soc
        new_soc = self.soc + delta_soc if is_charging else self.soc - delta_soc
        self.soc = float(np.clip(new_soc, self.soc_min, self.soc_max))
        actual_delta_soc = abs(self.soc - old_soc)

        # Calculate actual current based on actual delta_soc (for voltage calculation)
        if is_charging:
            actual_Ah = actual_delta_soc * self.max_cap_Ah / self.charge_efficiency
        else:
            actual_Ah = actual_delta_soc * self.max_cap_Ah * self.discharge_efficiency

        self.current_amps = actual_Ah / time_hours if time_hours > 0 else 0.0
        if not is_charging:
            self.current_amps = -self.current_amps  # Convention: positive = discharge

        # Update terminal voltage with actual current
        self.actual_voltage = self._calculate_terminal_voltage(
            self.soc, self.current_amps if not is_charging else -self.current_amps
        )

        # Calculate actual power for consumption reporting (kW)
        self.actual_power_kW = (self.actual_voltage * abs(self.current_amps)) / 1000.0
        if not is_charging:
            self.actual_power_kW = -self.actual_power_kW

        # Update the action dict to reflect actual (clipped) action
        actual_action = self.actual_power_kW / self.Q_electric_max if self.Q_electric_max > 0 else 0.0
        actions["battery_action"] = np.array([np.float32(actual_action)], dtype=np.float32)

    def update_state(self, states: Dict) -> None:
        # Ensure float32 dtype for all updates
        states["battery_pct"][0] = np.float32(self.soc)
        states["battery_target_pct"][0] = np.float32(self.target_soc)

        history = states["battery_pct_hist"]
        # Shift all rows up (drop oldest)
        history[:-1] = history[1:]
        # Insert new state at the end
        history[-1] = np.float32(self.soc)

    def get_electric_consumption(self, actions: Dict) -> float:
        """Get current electric energy consumption from battery in kW.

        Returns the actual power calculated from the Tremblay model,
        accounting for voltage variations and efficiency losses.

        Returns:
            Positive value when charging (consuming from grid),
            negative value when discharging (providing to grid).
        """
        return self.actual_power_kW

    def _get_actual_battery_charge_kW(self) -> float:
        """Calculate actual charge in kW based on current battery percentage and max capacity."""
        return self.soc * self.max_cap_kWh


# Register BatteryTremblay with the component registry
ComponentRegistry.register('infrastructure', BatteryTremblay)
