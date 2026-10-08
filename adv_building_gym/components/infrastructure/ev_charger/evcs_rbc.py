"""Rule-based EV charging station — passive infrastructure, no policy action."""

import logging
from typing import ClassVar, Dict, Optional, Set

import numpy as np
from gymnasium.spaces import Box

from adv_building_gym._common.constants import SECONDS_PER_HOUR

from ..base import Infrastructure
from adv_building_gym.components.registry import ComponentRegistry
from .ev_spec import EvSpec

logger = logging.getLogger(__name__)


class EvcsRbc(Infrastructure):
    """EV charging station driven by its own rule instead of the policy.

    Registers no action key. Connect/disconnect is read from the ``ctxt_ev_schedule_*``
    observation keys exactly as :class:`LinearEVCharger` does (an EV is attached iff
    ``ctxt_ev_schedule_max_cap_kWh`` > 0), and the same per-session corridor is published.

    Rule: charge at the constant rate that just reaches ``target_soc`` by the session
    deadline — the remaining SoC gap spread over the hours left to the target — saturated
    at ``min(charger rating, EV acceptance)``. A gap too wide for the remaining time simply
    leaves the charger at full power and is re-evaluated on the next step; the rate drops
    to zero once the target is met. A Gaussian actuator disturbance is added on top of a
    non-zero rate (an idle or unplugged charger stays exactly at zero).

    Charge-only: no V2G, so this is a pure consumer. It publishes the same ``s_evc_*`` /
    ``ctxt_evc_*`` keys and info fields as :class:`LinearEVCharger`, so swapping the two
    changes only the action space — and for that reason they must not be configured
    together.
    """

    POWER_FLOW = "consumer"

    # control_step comes from config context
    _context_params: ClassVar[Set[str]] = {'control_step'}

    # Internal state variables - don't serialize
    _exclude_params: ClassVar[Set[str]] = {
        'iteration', 'soc', 'ev_spec', 'charge_to_target_in_hrs', 'actual_power_kW',
        'rbc_action', '_rng',
    }

    def __init__(self,
                name: str,
                max_power_kW: float,
                control_step: int,
                max_charge_time_hrs: float = 24.0,
                action_noise_std: float = 0.05,
                ) -> None:
        """Initialise the rule-based EV charger.

        Args:
            name: Component identifier.
            max_power_kW: Charger hardware electrical rating in kW.
            control_step: Control step duration in seconds.
            max_charge_time_hrs: Upper bound on the per-session deadline; also
                the normaliser for ``s_evc_charge_to_target_hrs_norm``.
            action_noise_std: Standard deviation of the Gaussian disturbance added to the
                charging rate (in action units, i.e. fractions of the effective maximum
                charging power). Applied only to a non-zero rate.

        No ``ctxt_keys`` argument: both ``ctxt_evc_*`` keys this component publishes are
        unconditional (the EV rewards read them), so there is nothing to gate.
        """
        super().__init__(name, max_power_kW)

        self.control_step = control_step
        self.max_charge_time_hrs = max_charge_time_hrs
        self.action_noise_std = action_noise_std

        if self.max_charge_time_hrs <= 0:
            raise ValueError("max_charge_time_hrs must be positive.")
        if self.action_noise_std < 0:
            raise ValueError("action_noise_std must be non-negative.")

        # Connected EV; None ⇔ no EV currently plugged in.
        self.ev_spec: Optional[EvSpec] = None

        self.soc: float = 0.0
        self.charge_to_target_in_hrs: float = 0.0
        self.actual_power_kW: float = 0.0
        self.rbc_action: float = 0.0  # applied charging rate in [0, 1] (post-noise, post-clip)

        # Per-session corridor bookkeeping; snapshot at connect.
        self._session_step: int = 0
        self._session_total_steps: int = 0
        self._session_start_soc: float = 0.0
        self._session_target_soc: float = 0.0
        self._session_step_soc_gain: float = 0.0
        self._session_active: bool = False

        # Per-episode RNG for the actuator noise; rebound to the env rng (info["_rng"])
        # on reset(). Standalone default until the first reset.
        self._rng = np.random.default_rng()

    # ----- Connection state helpers -----

    @property
    def is_connected(self) -> bool:
        return self.ev_spec is not None

    @property
    def target_soc(self) -> float:
        return self.ev_spec.target_soc if self.is_connected else 0.0  # type: ignore

    @property
    def effective_max_charging_kW(self) -> float:
        # Charger hardware cap and EV acceptance both bind.
        if self.ev_spec is None:
            return 0.0
        return min(self.max_power_kW, self.ev_spec.max_charging_kW)

    @property
    def max_consumption_kW(self) -> float:
        # Only a connected EV can draw power; bound by charger AND EV acceptance
        # (effective_max_charging_kW is 0 when unplugged).
        # Overrides the class-level default
        return self.effective_max_charging_kW

    def setup_spaces(self, state_spaces, action_spaces):
        """Register the observation keys only — the charging rate is rule-based, no action.

        The key set matches :class:`LinearEVCharger` (minus the V2G-only
        ``ctxt_evc_v2g_effective``), so the EV rewards and the trajectory plots read the
        same signals for either charger, plus ``ar_evcs``: the applied charging rate,
        mirroring what ``a_lin_ev_charger`` would be if the policy drove this charger. It
        shares that action's [-1, 1] range (V2G export being the negative half) even though
        this charge-only component never leaves [0, 1], and is an observation rather than an
        action so a policy controlling the rest of the building sees the EV load it faces.
        """
        if "ar_evcs" not in state_spaces.keys():
            state_spaces["ar_evcs"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # States. No dedicated connection flag: ctxt_evc_max_charging_kW is 0 exactly when
        # no EV is attached, so it carries the flag with the charger's own timing
        # (see components/rewards/ev_signals.is_ev_connected).
        if "s_evc_soc" not in state_spaces.keys():
            state_spaces["s_evc_soc"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        if "s_evc_charge_to_target_hrs_norm" not in state_spaces.keys():
            # Normalised: 0 = none/disconnected, 1 = max_charge_time_hrs remaining
            state_spaces["s_evc_charge_to_target_hrs_norm"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Max charging power (kW) — changes per connected EV.
        if "ctxt_evc_max_charging_kW" not in state_spaces.keys():
            state_spaces["ctxt_evc_max_charging_kW"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        if "ctxt_evc_max_charge_time_hrs" not in state_spaces.keys():
            state_spaces["ctxt_evc_max_charge_time_hrs"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        # Per-session SoC corridor (zero when disconnected); see LinearEVCharger for the
        # envelope definitions. Kept here so the observation space is charger-agnostic.
        if "s_evc_soc_min" not in state_spaces.keys():
            state_spaces["s_evc_soc_min"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        if "s_evc_soc_max" not in state_spaces.keys():
            state_spaces["s_evc_soc_max"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Per-session target feasibility (snapshotted at connect):
        # 1.0 reachable, 0.0 unreachable, 0.5 neutral when no EV.
        if "s_evc_session_target_feasible" not in state_spaces.keys():
            state_spaces["s_evc_session_target_feasible"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

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
        """Snapshot boundary params at connect (after set_ev_connected, so self.soc is the
        start SoC). ``_session_active`` is False when the target is unreachable from the
        start at max rate — the rule then simply charges at full power throughout.
        """
        assert self.ev_spec is not None, "_snapshot_session requires a connected EV"

        self._session_step = 0
        self._session_start_soc = self.soc
        self._session_target_soc = self.target_soc

        # Step budget for the corridor from charge_to_target_in_hrs (deadline to
        # hit target_soc). At least one step so it's meaningful for short windows.
        self._session_total_steps = max(1, int(round(self.charge_to_target_in_hrs * SECONDS_PER_HOUR / self.control_step)))

        # Max SoC gain per step, using the connected EV's capacity/efficiency so the
        # corridor tracks the actual session.
        if self.ev_spec.max_cap_kWh > 0:
            self._session_step_soc_gain = (
                self.effective_max_charging_kW * self.ev_spec.charger_efficiency * self.control_step / SECONDS_PER_HOUR /
                self.ev_spec.max_cap_kWh
            )
        else:
            self._session_step_soc_gain = 0.0

        soc_gap = max(0.0, self._session_target_soc - self._session_start_soc)
        max_reachable_soc_gain = self._session_step_soc_gain * self._session_total_steps
        self._session_active = max_reachable_soc_gain >= soc_gap

    def _check_schedule(self, states: Dict) -> None:
        """Apply an EV connect/disconnect transition from the schedule observation.

        Connection is read from ``ctxt_ev_schedule_max_cap_kWh`` > 0 (no real EV has zero
        capacity); on a connect the EvSpec is rebuilt from the ``ctxt_ev_schedule_*``
        observation keys. On a change calls :meth:`set_ev_connected`.
        """
        if "ctxt_ev_schedule_max_cap_kWh" not in states:
            return  # No EVState in this config

        scheduled_connected = float(states["ctxt_ev_schedule_max_cap_kWh"][0]) > 0.0

        if scheduled_connected == self.is_connected:
            return  # No change

        if scheduled_connected:
            # build EvSpec from the ctxt_ev_schedule_* observation keys
            ev_spec = EvSpec(
                max_cap_kWh=float(states["ctxt_ev_schedule_max_cap_kWh"][0]),
                max_charging_kW=float(states["ctxt_ev_schedule_max_charging_kW"][0]),
                charger_efficiency=float(states["ctxt_ev_schedule_charger_eff"][0]),
                discharge_efficiency=float(states["ctxt_ev_schedule_discharge_eff"][0]),
                v2g_enabled=bool(states["ctxt_ev_schedule_v2g"][0] > 0.5),
                start_soc=float(states["ctxt_ev_schedule_start_soc"][0]),
                target_soc=float(states["ctxt_ev_schedule_target_soc"][0]),
                charge_to_target_in_hrs=float(states["ctxt_ev_schedule_charge_to_target_hrs"][0]),
            )
            logger.debug("EV schedule: CONNECT (cap=%.1f, soc=%.2f->%.2f)",
                        ev_spec.max_cap_kWh, ev_spec.start_soc, ev_spec.target_soc)
            self.set_ev_connected(connected=True, ev_spec=ev_spec)
            self._snapshot_session()
        else:
            logger.debug("EV schedule: DISCONNECT")
            self.set_ev_connected(connected=False)

    def _charging_rate(self) -> float:
        """Rule-based charging rate in [0, 1] (fraction of ``effective_max_charging_kW``).

        The remaining SoC gap is spread evenly over the steps left to the session deadline;
        the resulting grid-side power is expressed as a fraction of the effective maximum
        and saturated at 1.0, so an unreachable target just means full power this step.
        Requires a connected EV.
        """
        assert self.ev_spec is not None, "_charging_rate requires a connected EV"

        soc_gap = self._session_target_soc - self.soc
        eff_max_kW = self.effective_max_charging_kW
        if soc_gap <= 0.0 or eff_max_kW <= 0.0 or self.ev_spec.max_cap_kWh <= 0:
            return 0.0

        # _session_step is advanced in update_state, so it counts the steps already
        # charged: on the first step after connect the full budget is still available.
        steps_to_deadline = max(1, self._session_total_steps - self._session_step)
        remaining_hrs = steps_to_deadline * self.control_step / SECONDS_PER_HOUR

        # Grid-side energy needed: battery-side gap divided by the charging efficiency.
        required_energy_kWh = soc_gap * self.ev_spec.max_cap_kWh / self.ev_spec.charger_efficiency
        required_power_kW = required_energy_kWh / remaining_hrs

        return float(np.clip(required_power_kW / eff_max_kW, 0.0, 1.0))

    def _apply_noise(self, rate: float) -> float:
        """Gaussian actuator disturbance on a non-zero rate, clipped to [0, 1]."""
        if rate == 0.0 or self.action_noise_std == 0.0:
            return rate
        noisy = rate + float(self._rng.normal(loc=0.0, scale=self.action_noise_std))
        return float(np.clip(noisy, 0.0, 1.0))

    def exec_action(self, actions: Dict, states: Dict, info: dict) -> None:
        """Derive and apply the rule-based charging rate; ``actions`` is untouched."""
        # apply any EV schedule change first (read from the ctxt_ev_schedule_* obs keys)
        self._check_schedule(states)

        if not self.is_connected:
            # no EV → no charging
            self.actual_power_kW = 0.0
            self.rbc_action = 0.0
            return

        action = self._apply_noise(self._charging_rate())
        if action == 0.0:
            self.actual_power_kW = 0.0
            self.rbc_action = 0.0
            return

        # rate in [0, 1] → effective_max_charging_kW (= min(charger, EV cap)); energy in kWh
        eff_max_kW = self.effective_max_charging_kW
        power_kW = action * eff_max_kW
        energy_kWh = power_kW * (self.control_step / SECONDS_PER_HOUR)

        # charging: grid energy * efficiency = battery energy
        soc_change = (energy_kWh * self.ev_spec.charger_efficiency) / self.ev_spec.max_cap_kWh
        new_soc = self.soc + soc_change

        # if clipped at a full battery, back-calculate the rate that reaches the bound
        if new_soc > 1.0:
            actual_soc_change = 1.0 - self.soc
            # invert soc_change → grid-side energy → power → rate
            actual_energy_kWh = (actual_soc_change * self.ev_spec.max_cap_kWh) / self.ev_spec.charger_efficiency
            time_hours = self.control_step / SECONDS_PER_HOUR
            actual_power_kW = actual_energy_kWh / time_hours if time_hours > 0 else 0.0
            action = actual_power_kW / eff_max_kW if eff_max_kW > 0 else 0.0
            self.soc = 1.0
        else:
            self.soc = new_soc

        self.rbc_action = action
        self.actual_power_kW = action * eff_max_kW

    def update_state(self, states: Dict, info: dict) -> None:
        """Update observable state; identical publication contract to LinearEVCharger."""
        super().update_state(states, info)
        states["s_evc_soc"][0] = np.float32(self.soc)
        # Applied charging rate, so the policy sees the rule-based charger's own action.
        states["ar_evcs"][0] = np.float32(self.rbc_action)
        states["ctxt_evc_max_charging_kW"][0] = np.float32(self.effective_max_charging_kW)
        states["ctxt_evc_max_charge_time_hrs"][0] = np.float32(self.max_charge_time_hrs)

        # EV-spec values the EV rewards consume but the policy already sees via EVState's
        # ctxt_ev_schedule_* keys — routed to info rather than duplicated in the obs space.
        # evc_target_soc is the LATCHED session target: _session_target_soc is written at
        # connect and never cleared on detach, so the disconnect verdict can still judge
        # the achieved SoC against it on the step the charger releases the EV.
        info["evc_target_soc"] = float(self._session_target_soc)
        info["evc_max_cap_kWh"] = float(self.ev_spec.max_cap_kWh) if self.ev_spec is not None else 0.0
        info["evc_charger_efficiency"] = float(self.ev_spec.charger_efficiency) if self.ev_spec is not None else 0.0

        # Corridor envelopes and the remaining-time obs share one step budget
        # (_session_step / _session_total_steps) so the deadline has a single source of
        # truth; all zero when disconnected.
        soc_min, soc_max, normalized_time = 0.0, 0.0, 0.0
        if self.is_connected:
            self._session_step += 1
            steps_to_deadline = max(0, self._session_total_steps - self._session_step)

            # remaining hours to the deadline, normalised by max_charge_time_hrs to [0, 1]
            remaining_hrs = steps_to_deadline * self.control_step / SECONDS_PER_HOUR
            normalized_time = remaining_hrs / self.max_charge_time_hrs if self.max_charge_time_hrs > 0 else 0.0

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
        states["s_evc_charge_to_target_hrs_norm"][0] = np.float32(np.clip(normalized_time, 0.0, 1.0))
        states["s_evc_soc_min"][0] = np.float32(soc_min)
        states["s_evc_soc_max"][0] = np.float32(soc_max)

        # Session target feasibility (held from the connect snapshot): 1.0 reachable,
        # 0.0 unreachable, 0.5 neutral when no EV.
        feasible = (1.0 if self._session_active else 0.0) if self.is_connected else 0.5
        states["s_evc_session_target_feasible"][0] = np.float32(feasible)

    def reset(self, states: Dict, info: dict) -> None:
        """Reset to a fresh, disconnected state; EVState re-connects on step 1 if needed."""
        self.ev_spec = None
        self.soc = 0.0
        self.charge_to_target_in_hrs = 0.0
        self.actual_power_kW = 0.0
        self.rbc_action = 0.0
        self._session_step = 0
        self._session_total_steps = 0
        self._session_start_soc = 0.0
        self._session_target_soc = 0.0
        self._session_step_soc_gain = 0.0
        self._session_active = False
        # Bind to the env rng (info["_rng"]) so the actuator noise shares the
        # deterministic per-worker stream; standalone fallback otherwise.
        self._rng = info.get("_rng") or np.random.default_rng()
        super().reset(states, info)

    def get_raw_values(self) -> dict[str, float]:
        return {
            "raw_evc_kW": self.actual_power_kW,
            "raw_evcs_rbc_action": self.rbc_action,
        }

    def get_E(self, actions: Dict) -> tuple[float, float]:
        """Charge-only: (production, consumption) in kW, consumption always positive."""
        return 0.0, self.actual_power_kW


# register with ComponentRegistry
ComponentRegistry.register('infrastructure', EvcsRbc)
