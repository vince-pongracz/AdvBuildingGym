import logging
from typing import ClassVar, Dict, Optional, Set

import numpy as np
from gymnasium.spaces import Box

from adv_building_gym._common.constants import SECONDS_PER_HOUR

from ..base import Infrastructure
from adv_building_gym.components.registry import ComponentRegistry

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

# TODO noprio VP / IDEA 2026.01.15. : Implement a reward function, which rewards shorter fully charged and fully drained times of the battery.


class BatteryTremblay(Infrastructure):
    """Battery using the Tremblay model with series-parallel (NsNp) cell config.

    Action ``a_battery`` in [-1, 1]: +1 = max charge (consume), -1 = max discharge (export).
    Tremblay (IEEE 2008) gives terminal voltage from SoC, current and cell params —
    more realistic than a linear model.

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

    POWER_FLOW = "bidirectional"

    # control_step comes from config context
    _context_params: ClassVar[Set[str]] = {'control_step'}

    # Internal state variables - don't serialize (derived from cell configuration)
    _exclude_params: ClassVar[Set[str]] = {
        'iteration', 'soc', 'current_amps', 'actual_voltage',
        'actual_power_kW', 'nominal_voltage', 'max_cap_Ah', 'max_cap_kWh', 'R_pack'
    }

    def __init__(self, name: str,
                 control_step: int,  # Timesteps in seconds
                 max_power_kW: float,  # Max charge/discharge power in kW (e.g. 400V × 48A ≈ 19.0)
                 cell_capacity_Ah: float,  # Single cell capacity in Ah (typical 21700: 3.5)
                 max_charge_amps: float,  # Max pack current in A (e.g. 48)
                 max_charge_voltage: float,  # Max pack voltage in V (e.g. 420)
                 start_soc_percentage: float,  # Initial SoC [0, 1]
                 max_charge_rate: float,  # C-rate limit (e.g. 1.5)
                 history_length: int,  # Number of past SoC values to track
                 # Tremblay model parameters (per cell). Li-ion LFP reference values:
                 #   E0=3.2 V, K=0.009 V/Ah, A=0.468 V, B=3.529 1/Ah, R_cell=0.01 Ω
                 E0: float,  # Constant voltage (V per cell)
                 K: float,  # Polarization constant (V/Ah)
                 A: float,  # Exponential zone amplitude (V)
                 B: float,  # Exponential zone time constant inverse (1/Ah)
                 R_cell: float,  # Single cell internal resistance (Ohms)
                 # Cell configuration (NsNp topology). Reference 125s10p ≈ 14 kWh pack:
                 #   125 × 3.2V = 400V, 10 × 3.5Ah = 35Ah → 14 kWh
                 n_series: int,  # Number of cells in series per string
                 n_parallel: int,  # Number of parallel strings
                 # Efficiency parameters
                 charge_efficiency: float,  # Coulombic efficiency for charging [0, 1]
                 discharge_efficiency: float,  # Coulombic efficiency for discharging [0, 1]
                 # Operating limits -- prevent battery damage
                 soc_min: float,  # Hardware minimum SoC (clipping floor)
                 soc_max: float,  # Hardware maximum SoC (clipping ceiling)
                 ) -> None:
        super().__init__(name, max_power_kW)

        self.cell_capacity_Ah = cell_capacity_Ah
        self.max_charge_amps = max_charge_amps
        self.max_charge_voltage = max_charge_voltage

        # State of Charge (SoC) as a percentage [0, 1]
        self.start_soc_percentage = start_soc_percentage
        self.soc = start_soc_percentage
        self.history_length = history_length
        self.control_step = control_step
        # C-rate = charge/discharge rate vs rated capacity (1C ≈ full in 1h).
        # Link: https://www.batterypowertips.com/how-to-read-battery-discharge-curves-faq/
        self.max_charge_rate = max_charge_rate

        # Tremblay model parameters (per cell)
        self.E0 = E0
        self.K = K
        self.A = A
        self.B = B
        self.R_cell = R_cell

        # Cell configuration (NsNp topology) -- N_series serial, N_parallel parallel circuit of cells
        self.n_series = n_series
        self.n_parallel = n_parallel

        # Derived pack properties (NsNp scaling)
        self.nominal_voltage = E0 * n_series             # series cells add voltage
        self.max_cap_Ah = n_parallel * cell_capacity_Ah  # parallel strings add capacity
        self.R_pack = (n_series * R_cell) / n_parallel
        self.max_cap_kWh = (self.max_cap_Ah * self.nominal_voltage) / 1000.0
        # TODO VP 2026.05.31.: Define max_power_kW -- check this battery model again
        self.max_power_kW = max_charge_voltage * max_charge_amps / 1000.0  # Convert W to kW

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

    def setup_spaces(self, state_spaces, action_spaces):
        """Register a_battery [-1, 1] (positive=charge, negative=discharge) plus SoC/context states."""
        # a_battery in [-1, 1]: positive=charge (consume), negative=discharge (export)
        if "a_battery" not in action_spaces.keys():
            action_spaces["a_battery"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # States
        if "s_battery_soc" not in state_spaces.keys():
            state_spaces["s_battery_soc"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Capacity (kWh) — constant hardware parameter.
        if "ctxt_battery_capacity_kWh" not in state_spaces.keys():
            state_spaces["ctxt_battery_capacity_kWh"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        if "ctxt_battery_max_power_kW" not in state_spaces.keys():
            state_spaces["ctxt_battery_max_power_kW"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        return state_spaces, action_spaces

    def _calculate_terminal_voltage(self, soc: float, pack_current: float) -> float:
        """Tremblay terminal voltage (pack), computed per cell then scaled.

        Discharge (i≥0): E = E0 - K*(Q/(Q-it))*it - K*(Q/(Q-it))*i + A*exp(-B*it).
        Charge (i<0):  E = E0 - K*(Q/(it+0.1*Q))*it - K*(Q/(Q-it))*|i| + A*exp(-B*it).
        V = E - R*i (discharge) / E + R*|i| (charge).
        pack_current in A: positive=discharge, negative=charge. Returns pack voltage (V).
        """
        # cell current = pack current / parallel strings
        cell_current = pack_current / self.n_parallel if self.n_parallel > 0 else pack_current

        # extracted capacity (Ah) at cell level = (1-SoC)*Q
        it = (1.0 - soc) * self.cell_capacity_Ah
        Q = self.cell_capacity_Ah

        # guard against division by zero at extreme SoC
        epsilon = 0.001 * Q

        if cell_current >= 0:  # discharge
            denom = max(Q - it, epsilon)
            E_cell = (self.E0
                      - self.K * (Q / denom) * it
                      - self.K * (Q / denom) * cell_current
                      + self.A * np.exp(-self.B * it))
            V_cell = E_cell - self.R_cell * cell_current  # IR drop
        else:  # charge (modified equation)
            abs_current = abs(cell_current)
            denom_charge = it + 0.1 * Q
            denom_discharge = max(Q - it, epsilon)
            E_cell = (self.E0
                      - self.K * (Q / denom_charge) * it
                      - self.K * (Q / denom_discharge) * abs_current
                      + self.A * np.exp(-self.B * it))
            V_cell = E_cell + self.R_cell * abs_current  # voltage rises while charging

        # series cells multiply voltage
        V_pack = V_cell * self.n_series
        return float(np.clip(V_pack, 0.0, self.max_charge_voltage * 1.2))

    def _calculate_max_current(self, is_charging: bool) -> float:
        """Max allowable pack current (A) from C-rate and current limits.

        Pack limit = n_parallel × (max_charge_rate × cell_capacity), capped by max_charge_amps.
        """
        cell_c_rate_limit = self.max_charge_rate * self.cell_capacity_Ah
        pack_c_rate_limit = self.n_parallel * cell_c_rate_limit

        # same C-rate cap for charge and discharge
        return min(self.max_charge_amps, pack_c_rate_limit)

    def exec_action(self, actions: Dict, states: Dict, info=None) -> None:
        """Charge/discharge via Tremblay model. action in [-1, 1] (fraction of max_power_kW):
        positive=charge (consume), negative=discharge (export)."""
        action = float(np.atleast_1d(actions["a_battery"])[0])

        is_charging = action > 0

        # requested power (kW)
        requested_power_kW = abs(action) * self.max_power_kW

        # estimate voltage with a small test current to convert power → current
        test_current = 1.0 if not is_charging else -1.0
        estimated_voltage = self._calculate_terminal_voltage(self.soc, test_current)

        # I = P / V
        if estimated_voltage > 0:
            requested_current = (requested_power_kW * 1000) / estimated_voltage
        else:
            requested_current = 0.0

        # clip to C-rate / max-current limit
        max_current = self._calculate_max_current(is_charging)
        actual_current = min(requested_current, max_current)

        # energy: Ah = A * h
        time_hours = self.control_step / SECONDS_PER_HOUR
        delta_Ah = actual_current * time_hours

        # efficiency losses
        if is_charging:
            delta_Ah_effective = delta_Ah * self.charge_efficiency
        else:
            delta_Ah_effective = delta_Ah / self.discharge_efficiency

        # Ah → SoC change
        delta_soc = delta_Ah_effective / self.max_cap_Ah if self.max_cap_Ah > 0 else 0.0

        # apply and clip
        old_soc = self.soc
        new_soc = self.soc + delta_soc if is_charging else self.soc - delta_soc
        self.soc = float(np.clip(new_soc, self.soc_min, self.soc_max))
        actual_delta_soc = abs(self.soc - old_soc)

        # current from realised delta_soc (for voltage calc)
        if is_charging:
            actual_Ah = actual_delta_soc * self.max_cap_Ah / self.charge_efficiency
        else:
            actual_Ah = actual_delta_soc * self.max_cap_Ah * self.discharge_efficiency

        self.current_amps = actual_Ah / time_hours if time_hours > 0 else 0.0
        if not is_charging:
            self.current_amps = -self.current_amps  # convention: positive = discharge

        # terminal voltage at actual current; helper wants positive=discharge,
        # so negate current_amps (negative for discharge here).
        self.actual_voltage = self._calculate_terminal_voltage(
            self.soc, -self.current_amps
        )

        # actual power (kW), for reporting
        self.actual_power_kW = (self.actual_voltage * abs(self.current_amps)) / 1000.0
        if not is_charging:
            self.actual_power_kW = -self.actual_power_kW

        # rewrite action to the clipped fraction
        actual_action = self.actual_power_kW / self.max_power_kW if self.max_power_kW > 0 else 0.0
        actions["a_battery"] = np.array([np.float32(actual_action)], dtype=np.float32)

    def update_state(self, states: Dict, info=None) -> None:
        super().update_state(states, info)
        states["s_battery_soc"][0] = np.float32(self.soc)
        states["ctxt_battery_capacity_kWh"][0] = np.float32(self.max_cap_kWh)
        states["ctxt_battery_max_power_kW"][0] = np.float32(self.max_power_kW)

    def reset(self, states: Dict, info=None) -> None:
        """Reset SoC and derived voltage/current/power to __init__ values each episode
        (base would carry self.soc over)."""
        self.soc = self.start_soc_percentage
        self.current_amps = 0.0
        self.actual_voltage = self._calculate_terminal_voltage(self.soc, 0.0)
        self.actual_power_kW = 0.0
        super().reset(states, info)

    def get_E(self, actions: Dict) -> tuple[float, float]:
        # actual_power_kW > 0 = charging (consumes); returns (production, consumption)
        if self.actual_power_kW > 0.0:
            return 0.0, self.actual_power_kW
        else:
            # < 0 = discharging (produces to others)
            return -1.0 * self.actual_power_kW, 0.0


# register with ComponentRegistry
ComponentRegistry.register('infrastructure', BatteryTremblay)
