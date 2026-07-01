import logging
from typing import ClassVar, Dict, Set

import numpy as np
from gymnasium.spaces import Box

from adv_building_gym._common.constants import SECONDS_PER_HOUR, KW_TO_W

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
        E = E0 - K * (Q/(Q-it)) * it - K * (Q/(Q-it)) * i + A * exp(-B*it)

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
        'iteration', 'soc', 'cap_Ah', 'cap_min_Ah', 'cap_max_Ah',
        'actual_current_A', 'actual_V', 'actual_power_kW', 'nominal_V',
        'max_cap_Ah', 'max_cap_kWh', 'R_pack',
        'max_power_kW', 'max_current_A', 'max_terminal_V'
    }

    def __init__(self, name: str,
                 control_step: int,  # Timesteps in seconds
                 cell_capacity_Ah: float,  # Single cell capacity in Ah (typical 21700: 3.5)
                 max_charge_A: float,  # Max pack current in A (e.g. 48)
                 start_soc: float,  # Initial SoC [0, 1]
                 max_charge_rate: float,  # C-rate limit (e.g. 1.5)
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
                 emit_ctxt: bool = False,  # publish policy-only ctxt_* (generalisation runs)
                 ) -> None:
        # Rated pack power is DERIVED from the pack's own limits, not passed in.
        # P_rated = V_nominal * I_max, where V_nominal = E0 * n_series and I_max is
        # the C-rate / wiring-limited pack current (the same cap exec_action applies).
        # Using the nominal operating voltage
        # keeps action=±1 aligned with the largest current the battery can actually
        # sustain, so the top of the action range is not a saturated dead zone.
        self.nominal_V = E0 * n_series           # series cells add voltage
        self.max_current_A = min(max_charge_A, n_parallel * cell_capacity_Ah * max_charge_rate) # C-rate / wiring-limited pack current (A)
        # NOTE VP 2026.06.14.: C-rate [1/h], cell_capacity_Ah [Ah]
        # → max current (A) limit from chemistry and wiring.
        super().__init__(name, self.nominal_V * self.max_current_A / 1000.0)
        self.emit_ctxt = emit_ctxt

        self.cell_capacity_Ah = cell_capacity_Ah
        self.max_charge_A = max_charge_A

        # State of Charge (SoC) as a percentage [0, 1]. The stored charge in Ah
        # (self.cap_Ah) is the primary state; self.soc is derived from it. cap_Ah is
        # set below once max_cap_Ah is known.
        self.start_soc = start_soc
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
        self.max_cap_Ah = n_parallel * cell_capacity_Ah  # parallel strings add capacity
        self.R_pack = (n_series * R_cell) / n_parallel
        self.max_cap_kWh = (self.max_cap_Ah * self.nominal_V) / 1000.0
        # self.max_power_kW (rated pack power) is derived above and set by super().__init__.

        # Primary state: charge stored in the pack (Ah). SoC is DERIVED from it
        # (soc = cap_Ah / max_cap_Ah) so the Tremblay law runs off the real extracted
        # capacity rather than a re-scaled SoC. soc_min / soc_max become Ah clip
        # thresholds once here; the clip is applied on cap_Ah in exec_action.
        self.cap_min_Ah = soc_min * self.max_cap_Ah
        self.cap_max_Ah = soc_max * self.max_cap_Ah
        self.cap_Ah = start_soc * self.max_cap_Ah

        # Efficiency parameters
        self.charge_efficiency = charge_efficiency
        self.discharge_efficiency = discharge_efficiency

        # Operating limits
        self.soc_min = soc_min
        self.soc_max = soc_max

        # Terminal-voltage clamp ceiling, DERIVED from the cells (replaces the former
        # passed-in `max_charge_V`). The highest terminal voltage the pack can reach is
        # at full charge current; the 1.2x margin
        # keeps the guard from clipping legitimate values while still catching numeric
        # spikes from the (Q - it) denominator at SoC extremes.
        self.max_terminal_V = self._terminal_V_raw(1.0, -self.max_current_A) * 1.2

        # Track actual current for voltage calculation
        self.actual_current_A = 0.0
        self.actual_V = self._calculate_terminal_V(self.soc, 0.0)
        self.actual_power_kW = 0.0  # Track actual power for consumption reporting

    @property
    def soc(self) -> float:
        """SoC derived from the stored charge (cap_Ah / max_cap_Ah). The model's
        primary state is the Ah charge, not SoC; this keeps s_battery_soc consistent."""
        return self.cap_Ah / self.max_cap_Ah if self.max_cap_Ah > 0 else 0.0

    def _directional_max_power_kW(self, soc: float, is_charging: bool) -> float:
        """Largest pack power (kW) the battery can sustain in one direction at this SoC.

        Set by the C-rate / wiring current cap (``max_current_A``) at the terminal
        voltage that current actually produces (P = V * I), with V taken from the
        realised Tremblay law -- NOT the fixed ``nominal_V``. The terminal voltage is
        SoC- and direction-dependent (OCV drift plus the +R|i| charge overpotential /
        -R*i discharge sag), so this ceiling moves with the operating point.
        Tremblay sign convention: pack current is +discharge / -charge.
        """
        pack_current = -self.max_current_A if is_charging else self.max_current_A
        terminal_V = self._calculate_terminal_V(soc, pack_current)
        return terminal_V * self.max_current_A / KW_TO_W

    @property
    def max_consumption_kW(self) -> float:
        """Real charge-power ceiling at the current SoC (V*I at max charge current).

        Published into ``info`` each step via the base power-bound bookkeeping; replaces
        the fixed nominal_V nameplate so the bound never understates what a charge step
        can actually draw (the charge terminal voltage exceeds nominal_V at high SoC)."""
        return self._directional_max_power_kW(self.soc, is_charging=True)

    @property
    def max_production_kW(self) -> float:
        """Real discharge-power ceiling at the current SoC (V*I at max discharge current)."""
        return self._directional_max_power_kW(self.soc, is_charging=False)

    def setup_spaces(self, state_spaces, action_spaces):
        """Register a_battery [-1, 1] (positive=charge, negative=discharge) plus SoC/context states."""
        # a_battery in [-1, 1]: positive=charge (consume), negative=discharge (export)
        if "a_battery" not in action_spaces.keys():
            action_spaces["a_battery"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # States
        if "s_battery_soc" not in state_spaces.keys():
            state_spaces["s_battery_soc"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Capacity (kWh) — constant hardware parameter.
        # Capacity (kWh) and power (kW) — policy-only conditioning, gated by emit_ctxt.
        self._publish_ctxt(state_spaces, "ctxt_battery_capacity_kWh",
                            Box(low=0, high=np.inf, shape=(1,), dtype=np.float32))
        self._publish_ctxt(state_spaces, "ctxt_battery_max_power_kW",
                            Box(low=0, high=np.inf, shape=(1,), dtype=np.float32))

        return state_spaces, action_spaces

    def _terminal_V_raw(self, soc: float, pack_current: float) -> float:
        """Unclamped Tremblay terminal voltage (pack), computed per cell then scaled.

        Discharge (i≥0): E = E0 - K*(Q/(Q-it))*i - K*(Q/(Q-it))*it + A*exp(-B*it).
        Charge (i<0):  E = E0 + K*(Q/(it+0.1*Q))*|i| - K*(Q/(Q-it))*it + A*exp(-B*it).
        V = E - R*i (discharge) / E + R*|i| (charge).
        pack_current in A: positive=discharge, negative=charge. Returns pack voltage (V).
        """
        # cell current = pack current / parallel strings
        cell_current = pack_current / self.n_parallel if self.n_parallel > 0 else pack_current

        # extracted capacity (it -- Ah) at cell level = (1-SoC)*Q
        Q = self.cell_capacity_Ah
        it = (1.0 - soc) * Q

        # guard against division by zero at extreme SoC
        epsilon = 0.001 * Q

        if cell_current >= 0:  # discharge
            denom = max(Q - it, epsilon)
            E_cell = (self.E0
                      - self.K * (Q / denom) * it
                      - self.K * (Q / denom) * cell_current
                      + self.A * np.exp(-self.B * it))
            V_cell: float = E_cell - self.R_cell * cell_current  # IR drop
        else:  # charge (i* < 0)
            # Tremblay Li-ion charge branch. Relative to discharge, only the
            # polarization-RESISTANCE term (the one in the current i*) changes: its
            # denominator becomes (it + 0.1*Q) and it flips sign (i* < 0 on charge),
            # which produces the end-of-charge voltage overshoot. The polarization-
            # VOLTAGE term (the one in it) keeps the (Q - it) denominator, as in discharge.
            abs_current = abs(cell_current)
            denom_charge = it + 0.1 * Q              # polarization-resistance denom (charge)
            denom_discharge = max(Q - it, epsilon)   # polarization-voltage denom (as discharge)
            E_cell = (self.E0
                      + self.K * (Q / denom_charge) * abs_current
                      - self.K * (Q / denom_discharge) * it
                      + self.A * np.exp(-self.B * it))
            V_cell: float = E_cell + self.R_cell * abs_current  # +R|i|: overpotential raises V on charge

        # series cells multiply voltage
        return V_cell * self.n_series

    def _calculate_terminal_V(self, soc: float, pack_current: float) -> float:
        """Terminal voltage clamped to a physical range. The floor keeps reported
        voltage non-negative; the ceiling (`max_terminal_V`, derived from the cells)
        guards numeric spikes from the (Q - it) denominator at SoC extremes."""
        return float(np.clip(self._terminal_V_raw(soc, pack_current), 0.0, self.max_terminal_V))

    def exec_action(self, actions: Dict, states: Dict, info: dict) -> None:
        """Charge/discharge via the Tremblay model. action in [-1, 1] (fraction of the
        per-step directional power ceiling): positive=charge (consume), negative=discharge
        (export).

        Stored charge ``cap_Ah`` (Ah) is the primary state and is clipped to the Ah
        thresholds derived from soc_min/soc_max; ``self.soc`` is derived from it.
        Sign convention: power is +consume / -export, while the Tremblay current is
        +discharge / -charge (opposite), matching ``_terminal_V_raw``.
        """
        action = float(np.atleast_1d(actions["a_battery"])[0])

        is_charging = action > 0
        # Power sign: +1 charging (consume), -1 discharging (export) -- matches get_E.
        power_sign = 1.0 if is_charging else -1.0

        # Per-step power ceiling derived from the realised terminal voltage (V*I at
        # max_current_A in this direction), not the fixed nominal_V. action=+/-1 maps to
        # the power the pack can truly sustain at the present SoC, so the realised power
        # stays within the published max_consumption_kW / max_production_kW bound.
        max_power_this_step_kW = self._directional_max_power_kW(self.soc, is_charging)

        # requested power magnitude (kW)
        requested_power_kW = abs(action) * max_power_this_step_kW

        # Convert power -> current with the terminal voltage carried over from the
        # previous step (stored as self.actual_V), rather than a fresh test current;
        # falls back to the zero-current terminal voltage on the first step. I = P / V.
        estimated_V = self.actual_V if self.actual_V > 0 else self._calculate_terminal_V(self.soc, 0.0)
        requested_A = (requested_power_kW * KW_TO_W) / estimated_V if estimated_V > 0 else 0.0

        # clip the current magnitude to the C-rate / max-current limit (cached at init)
        current_magnitude_A = min(requested_A, self.max_current_A)

        # charge moved this step (Ah) before efficiency: Ah = A * h
        time_hours = self.control_step / SECONDS_PER_HOUR
        delta_Ah = current_magnitude_A * time_hours

        # Coulombic efficiency: charging stores less than is drawn; discharging pulls
        # more out of the cells than is delivered.
        if is_charging:
            delta_cap_Ah = delta_Ah * self.charge_efficiency
        else:
            delta_cap_Ah = delta_Ah / self.discharge_efficiency

        # Track the stored charge directly in Ah and clip to the Ah thresholds:
        # cap_Ah(t) = cap_Ah(t-1) +/- i(t) * dt. SoC is derived (see the soc property).
        old_cap_Ah = self.cap_Ah
        new_cap_Ah = self.cap_Ah + delta_cap_Ah if is_charging else self.cap_Ah - delta_cap_Ah
        self.cap_Ah = float(np.clip(new_cap_Ah, self.cap_min_Ah, self.cap_max_Ah))
        realised_cap_Ah = abs(self.cap_Ah - old_cap_Ah)  # may be threshold-limited

        # Back out the realised cell-side current magnitude (undo efficiency).
        if is_charging:
            realised_Ah = realised_cap_Ah / self.charge_efficiency
        else:
            realised_Ah = realised_cap_Ah * self.discharge_efficiency
        current_magnitude_A = realised_Ah / time_hours if time_hours > 0 else 0.0

        # Signed current for the Tremblay voltage law (+discharge / -charge), i.e.
        # opposite sign to the power convention.
        self.actual_current_A = -power_sign * current_magnitude_A

        # terminal voltage at the realised (signed) current
        self.actual_V = self._calculate_terminal_V(self.soc, self.actual_current_A)

        # Signed pack power (kW): +charge (consume) / -discharge (export), so get_E
        # reports discharge as production.
        self.actual_power_kW = power_sign * self.actual_V * current_magnitude_A * (1 / KW_TO_W)

        # rewrite action to the realised (signed) fraction of the SAME per-step ceiling
        # used to size the request, clipped to the a_battery Box. The ceiling is taken at
        # the pre-update SoC while the realised power uses the post-update SoC, so the
        # ratio can edge just past 1 at the clip boundary -- keep the label in [-1, 1].
        actual_action = self.actual_power_kW / max_power_this_step_kW if max_power_this_step_kW > 0 else 0.0
        actual_action = float(np.clip(actual_action, -1.0, 1.0))
        actions["a_battery"] = np.array([np.float32(actual_action)], dtype=np.float32)

    def update_state(self, states: Dict, info: dict) -> None:
        super().update_state(states, info)
        states["s_battery_soc"][0] = np.float32(self.soc)
        # Publish the real computed energy / power at the realised terminal voltage
        # (self.actual_V), not the fixed nominal_V nameplate: capacity = max_cap_Ah * V,
        # power ceiling = V * max_current_A. Both track the live operating point.
        # NOTE VP 2026.06.14.: They are not static anymore
        self._write_ctxt(states, "ctxt_battery_capacity_kWh", np.float32(self.max_cap_Ah * self.actual_V / KW_TO_W))
        self._write_ctxt(states, "ctxt_battery_max_power_kW", np.float32(self.actual_V * self.max_current_A / KW_TO_W))

    def reset(self, states: Dict, info: dict) -> None:
        """Reset stored charge and derived voltage/current/power to __init__ values each
        episode (base would carry the charge over)."""
        self.cap_Ah = self.start_soc * self.max_cap_Ah
        self.actual_current_A = 0.0
        self.actual_V = self._calculate_terminal_V(self.soc, 0.0)
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
