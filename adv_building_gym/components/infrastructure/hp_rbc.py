"""Rule-based heat pump — passive infrastructure, no policy action."""

from collections import OrderedDict
import logging
from typing import ClassVar, Dict, Set

import numpy as np
from gymnasium.spaces import Box

from .base import Infrastructure
from adv_building_gym.components.registry import ComponentRegistry
from adv_building_gym._common.constants import SLOWDOWN_TERM, W_PER_KW, TEMP_ABS_MAX_CELSIUS

logger = logging.getLogger(__name__)


class HPRbc(Infrastructure):
    """Heat pump driven by its own rule-based controller instead of the policy.

    Registers no action key: the control level is recomputed every step from the comfort
    error ``s_temp_error_norm`` (indoor - setpoint, published by ``InsideTemperature``).

    Rule: inside a ``deadband_C`` band around the setpoint the unit idles; outside it the
    thermal power that would null the error within ONE control step is requested and
    saturated at the rated power, so a large gap is closed over several steps rather than
    tracked by a tuned gain. A Gaussian actuator disturbance is added on top of a non-zero
    control level (an idle unit stays exactly idle).

    Thermal effect, ctxt/raw keys and the ``info["temp_in_norm"]`` integration variable are
    identical to :class:`HP` — this component is a drop-in replacement for it, and the two
    must not be configured together (both own ``info["temp_in_norm"]``).
    """

    POWER_FLOW = "consumer"

    # control_step from env context; mC read from info["ctxt_building_mC"] at runtime.
    _context_params: ClassVar[Set[str]] = {'control_step'}

    # Internal state - not serialised
    _exclude_params: ClassVar[Set[str]] = {
        'iteration', 'temp_in_norm_change', 'control_step', 'actual_power_kW',
        'temp_in_raw', 'rbc_action', '_rng',
    }

    def __init__(self,
                name: str,
                max_power_kW: float,
                control_step: int,
                cop_heat: float = 1.0,
                cop_cool: float = 1.0,
                deadband_C: float = 0.5,
                action_noise_std: float = 0.05,
                ctxt_keys: list[str] | None = None,
                ) -> None:
        """Initialise the rule-based heat pump.

        Args:
            name: Component identifier.
            max_power_kW: Electric rating in kW.
            control_step: Control step duration in seconds.
            cop_heat: Heating coefficient of performance [-].
            cop_cool: Cooling coefficient of performance [-].
            deadband_C: Half-width of the idle band around the setpoint (°C). The unit
                stays off while the comfort error is inside it.
            action_noise_std: Standard deviation of the Gaussian disturbance added to the
                control level (in action units, i.e. fractions of rated power). Applied
                only to a non-zero control level.
            ctxt_keys: Allow-list of ``ctxt_*`` observation keys to publish.
        """
        super().__init__(name, max_power_kW)
        self.ctxt_keys = list(ctxt_keys) if ctxt_keys is not None else None

        # NOTE VP 2026.01.20. : COP, link: https://en.wikipedia.org/wiki/Coefficient_of_performance
        # COP = Q_thermal / P_electric => Q_thermal = P_electric * COP
        self.cop_heat = cop_heat  # [-] heating COP
        self.cop_cool = cop_cool  # [-] cooling COP
        self.control_step = control_step
        self.deadband_C = deadband_C
        self.action_noise_std = action_noise_std

        self.temp_in_norm_change = 0.0
        self.actual_power_kW = 0.0  # actual electric draw (kW), for reporting
        self.temp_in_raw = 0.0  # Denormalised indoor temp after this component's update (°C)
        self.rbc_action = 0.0  # applied control level in [-1, 1] (post-noise, post-clip)

        if self.cop_heat <= 0 or self.cop_cool <= 0:
            raise ValueError("cop_heat and cop_cool must be positive.")
        if self.deadband_C < 0:
            raise ValueError("deadband_C must be non-negative.")
        if self.action_noise_std < 0:
            raise ValueError("action_noise_std must be non-negative.")

        # Per-episode RNG for the actuator noise; rebound to the env rng (info["_rng"])
        # on reset(). Standalone default until the first reset.
        self._rng = np.random.default_rng()

    def setup_spaces(self,
                    state_spaces: OrderedDict,
                    action_spaces: OrderedDict) -> tuple[OrderedDict, OrderedDict]:
        """Publish the applied control level (and the rated-power context key); no action key.

        ``ar_hp`` mirrors what ``a_hp`` would be if the policy drove this unit: the applied,
        post-noise, post-clip level in [-1, 1] (negative = cool, positive = heat). It is an
        observation rather than an action so a policy controlling the rest of the building
        can see what the rule-based heat pump is doing to the load.
        """
        # Indoor/outdoor temperature are not observations — indoor temp is the shared
        # integration variable on info["temp_in_norm"]; the policy sees the comfort
        # error (s_temp_error_norm), which InsideTemperature publishes.
        if "ar_hp" not in state_spaces.keys():
            state_spaces["ar_hp"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Raw electric capacity (kW) — policy-only conditioning, published only when listed in ctxt_keys.
        self._publish_ctxt(state_spaces, "ctxt_hp_max_power_kW", Box(low=0, high=np.inf, shape=(1,), dtype=np.float32))

        return state_spaces, action_spaces

    def _control_level(self, states: Dict, mC: float, temp_abs_max: float) -> float:
        """Rule-based control level in [-1, 1]: negative = cool, positive = heat.

        Requests the thermal power that closes the whole comfort error within one control
        step (the 1R1C update of :meth:`exec_action` inverted), saturated at rated power —
        a gap wider than one step's reach leaves the unit at full power and is re-evaluated
        on the next step. Zero inside the deadband.
        """
        if "s_temp_error_norm" not in states:
            raise RuntimeError(
                f"HPRbc '{self.name}': no 's_temp_error_norm' observation — an "
                "InsideTemperature statesource is required to drive the rule-based control."
            )

        # Comfort error published by InsideTemperature: indoor - setpoint, normalised.
        error_norm = float(states["s_temp_error_norm"][0])
        error_C = error_norm * temp_abs_max
        if abs(error_C) <= self.deadband_C:
            return 0.0

        # Invert the 1R1C update: the temperature change needed is -error, so
        # dT_raw [K] = -error_C = SLOWDOWN_TERM * dt * q_required_W / mC.
        required_dT_raw = -error_C
        required_q_W = required_dT_raw * mC / (SLOWDOWN_TERM * self.control_step)
        required_q_kW = required_q_W / W_PER_KW

        cop = self.cop_heat if required_q_kW > 0 else self.cop_cool
        denominator = self.max_power_kW * cop
        if denominator <= 0:
            return 0.0
        return float(np.clip(required_q_kW / denominator, -1.0, 1.0))

    def _apply_noise(self, control_level: float) -> float:
        """Gaussian actuator disturbance on a non-zero control level.

        Clipped into the half-range of the rule's own direction ([0, 1] when heating,
        [-1, 0] when cooling), so the disturbance can only scale the demanded level — never
        turn a heating request into a cooling one (or the reverse).
        """
        if control_level == 0.0 or self.action_noise_std == 0.0:
            return control_level
        noisy = control_level + float(self._rng.normal(loc=0.0, scale=self.action_noise_std))
        
        low, high = (0.0, 1.0) if control_level > 0 else (-1.0, 0.0)
        return float(np.clip(noisy, low, high))

    def exec_action(self, actions: Dict, states: Dict, info: dict) -> None:
        """Derive and apply the rule-based control level; ``actions`` is untouched."""
        # Thermal mass owned by BuildingHeatLoss; shared on the info channel
        # (inter-component), so it is available regardless of the obs-space ctxt_keys.
        mC = float(info["ctxt_building_mC"])
        # Fixed temperature normalisation scale from the info channel (WeatherDataSource).
        temp_abs_max = float(info["temp_abs_max"]) if "temp_abs_max" in info else TEMP_ABS_MAX_CELSIUS

        hp_action = self._apply_noise(self._control_level(states, mC, temp_abs_max))
        energy = abs(hp_action)

        # NOTE VP 2026.01.20. : Thermal model is 1R1C, same as links below
        # Thermal power Q_thermal = energy * max_power_kW * COP
        # Sign of q_hp follows the control level: positive = heating, negative = cooling
        if hp_action < 0:
            cop = self.cop_cool
            q_hp = -energy * self.max_power_kW * cop  # heat removed from building
        elif hp_action > 0:
            cop = self.cop_heat
            q_hp = energy * self.max_power_kW * cop  # heat added to building
        else:
            self.temp_in_norm_change = 0.0
            self.actual_power_kW = 0.0
            self.rbc_action = 0.0
            return

        # NOTE VP 2026.01.20. : Heat loss Q_transfer is handled by the BuildingHeatLoss
        # statesource (a continuous effect); the heat pump applies only its own effect.
        # NOTE VP 2026.01.20. : Thermal model (1R1C) -- lumped-parameter models
        # paper1: Particle Swarm Optimization and Kalman Filtering for Demand Prediction of Commercial Buildings
        # Link: https://www.researchgate.net/publication/301310479_Particle_Swarm_Optimization_and_Kalman_Filtering_for_Demand_Prediction_of_Commercial_Buildings
        # paper2: EKF based self-adaptive thermal model for a passive house
        # Link: https://www.sciencedirect.com/science/article/pii/S0378778812003039?via%3Dihub

        # 1R1C update (SI): dT_raw [K] = SLOWDOWN_TERM * dt * q_hp_W / mC.
        # q_hp converted kW->W; SLOWDOWN_TERM is the dynamical slowdown (see constants.py).
        # Indoor temperature is normalised, so divide the raw °C change by temp_abs_max.
        q_hp_W = q_hp * W_PER_KW
        dT_raw = SLOWDOWN_TERM * self.control_step * q_hp_W / mC
        dTemp_norm = dT_raw / temp_abs_max if temp_abs_max > 0 else 0.0

        # Indoor temperature is the shared integration variable on info["temp_in_norm"];
        # check whether the change would clip at the ±1 normalised bounds.
        current_temp_norm = float(info.get("temp_in_norm", 0.0))
        new_temp_norm = current_temp_norm + dTemp_norm

        if new_temp_norm > 1.0 or new_temp_norm < -1.0:
            # temp change needed to reach the limit
            if new_temp_norm > 1.0:
                actual_dTemp = 1.0 - current_temp_norm
            else:  # new_temp_norm < -1.0
                actual_dTemp = -1.0 - current_temp_norm

            # Invert the forward path: dTemp(norm) -> dT_raw -> q_hp_W -> q_hp(kW) -> energy
            actual_dT_raw = actual_dTemp * temp_abs_max
            actual_q_hp_W = actual_dT_raw * mC / (SLOWDOWN_TERM * self.control_step)
            actual_q_hp_kW = actual_q_hp_W / W_PER_KW
            actual_energy = abs(actual_q_hp_kW) / (self.max_power_kW * cop) if (self.max_power_kW * cop) > 0 else 0.0
            actual_energy = float(np.clip(actual_energy, 0.0, 1.0))

            # preserve sign (cool/heat)
            sign = -1.0 if hp_action < 0 else 1.0
            self.rbc_action = sign * actual_energy
            self.temp_in_norm_change = actual_dTemp
            self.actual_power_kW = actual_energy * self.max_power_kW
        else:
            # no clip needed
            self.rbc_action = hp_action
            self.temp_in_norm_change = dTemp_norm
            self.actual_power_kW = energy * self.max_power_kW

    def update_state(self, states: Dict, info: dict) -> None:
        super().update_state(states, info)

        # Apply the thermal effect to the shared indoor temperature on info.
        current_temp_norm = float(info.get("temp_in_norm", 0.0))
        new_temp: float = current_temp_norm + self.temp_in_norm_change  # clip ensured in exec_action
        info["temp_in_norm"] = new_temp
        # Applied control level, so the policy sees the rule-based unit's own action.
        states["ar_hp"][0] = np.float32(self.rbc_action)
        self._write_ctxt(states, "ctxt_hp_max_power_kW", np.float32(self.max_power_kW))

        # Cache raw indoor temp after this component's heat. BuildingHeatLoss
        # re-derives it later; collected after infras, so its value wins.
        temp_abs_max = float(info["temp_abs_max"]) if "temp_abs_max" in info else TEMP_ABS_MAX_CELSIUS
        self.temp_in_raw = new_temp * temp_abs_max

    def reset(self, states: Dict, info: dict) -> None:
        """Clear per-episode transient state and bind the env rng before publishing obs."""
        self.temp_in_norm_change = 0.0
        self.actual_power_kW = 0.0
        self.temp_in_raw = 0.0
        self.rbc_action = 0.0
        # Bind to the env rng (info["_rng"]) so the actuator noise shares the
        # deterministic per-worker stream; standalone fallback otherwise.
        self._rng = info.get("_rng") or np.random.default_rng()
        super().reset(states, info)

    def get_raw_values(self) -> dict[str, float]:
        return {
            "raw_hp_kW": self.actual_power_kW,
            "raw_temp_in": self.temp_in_raw,
            "raw_hp_rbc_action": self.rbc_action,
        }

    def get_E(self, actions: Dict) -> tuple[float, float]:
        """Electric consumption (kW), always positive. Uses post-clip power from exec_action."""
        return 0.0, self.actual_power_kW


# register with ComponentRegistry
ComponentRegistry.register('infrastructure', HPRbc)
