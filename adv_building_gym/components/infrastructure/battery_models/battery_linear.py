import logging
from typing import ClassVar, Dict, Set

import numpy as np
from gymnasium.spaces import Box

from adv_building_gym._common.constants import SECONDS_PER_HOUR
from ..base import Infrastructure
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class BatteryLinear(Infrastructure):
    """Bidirectional battery as an ideal linear store (no losses/voltage variation).

    Action ``a_battery`` in [-1, 1] → [-max_power_kW, max_power_kW]: positive=charge
    (consume), negative=discharge (export).

    Per ``exec_action``: delta_E = action*max_power_kW*control_step/3600;
    delta_SoC = delta_E/max_cap_kWh; soc = clip(soc+delta_SoC, soc_min, soc_max).
    After clipping, ``actual_power_kW`` and ``actions["a_battery"]`` are
    back-calculated from the realised SoC change.

    Publishes ``s_battery_soc``, ``ctxt_battery_capacity_kWh``, ``ctxt_battery_power_kW``.
    ``reset`` restores SoC to ``start_soc_percentage`` ± uniform ``start_soc_jitter`` (env rng), clipped.
    """

    POWER_FLOW = "bidirectional"

    _context_params: ClassVar[Set[str]] = {'control_step'}

    _exclude_params: ClassVar[Set[str]] = { 'iteration', 'soc', 'actual_power_kW' }

    def __init__(self, name: str,
                max_power_kW: float,
                max_cap_kWh: float,
                control_step: int,
                start_soc_percentage: float,
                soc_min: float,
                soc_max: float,
                start_soc_jitter: float = 0.0,
                emit_ctxt: bool = False,
                ) -> None:
        """Initialize linear battery model.

        Args:
            name: Component identifier
            max_power_kW: Maximum charge/discharge power in kW
            max_cap_kWh: Battery capacity in kWh
            control_step: Timestep duration in seconds
            start_soc_percentage: Initial state of charge [0, 1]
            soc_min: Hardware minimum SoC (clipping floor)
            soc_max: Hardware maximum SoC (clipping ceiling)
            start_soc_jitter: Half-width of the uniform per-episode offset
                applied to the initial SoC for generalisation. 0.0 (default)
                keeps the deterministic ``start_soc_percentage``.
        """
        super().__init__(name, max_power_kW)
        self.emit_ctxt = emit_ctxt

        if start_soc_jitter < 0.0:
            raise ValueError("start_soc_jitter must be non-negative.")

        self.max_cap_kWh = max_cap_kWh
        # Attribute name matches the ctor param so Serializable.to_dict round-trips it.
        self.start_soc_percentage = start_soc_percentage
        self.start_soc_jitter = start_soc_jitter
        self.soc = start_soc_percentage
        self.control_step = control_step
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.actual_power_kW = 0.0

    def setup_spaces(self, state_spaces, action_spaces):
        if "a_battery" not in action_spaces.keys():
            action_spaces["a_battery"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        if "s_battery_soc" not in state_spaces.keys():
            state_spaces["s_battery_soc"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Capacity (kWh) and power (kW) — policy-only conditioning, gated by emit_ctxt.
        self._publish_ctxt(state_spaces, "ctxt_battery_capacity_kWh",
                        Box(low=0, high=np.inf, shape=(1,), dtype=np.float32))
        self._publish_ctxt(state_spaces, "ctxt_battery_power_kW",
                        Box(low=0, high=np.inf, shape=(1,), dtype=np.float32))

        return state_spaces, action_spaces

    def exec_action(self, actions: Dict, states: Dict, info: dict) -> None:
        """Linear charge/discharge. action in [-1, 1] (fraction of max_power_kW):
        positive=charge (consume), negative=discharge (export)."""
        action = float(np.atleast_1d(actions["a_battery"])[0])

        # requested power (kW)
        requested_power_kW = action * self.max_power_kW

        # energy change: E[kWh] = P[kW] * t[h]
        time_duration_in_hours = self.control_step / SECONDS_PER_HOUR
        delta_energy_kWh = requested_power_kW * time_duration_in_hours

        # energy → SoC change
        delta_soc = delta_energy_kWh / self.max_cap_kWh if self.max_cap_kWh > 0 else 0.0

        # apply and clip
        old_soc = self.soc
        new_soc = self.soc + delta_soc
        self.soc = float(np.clip(new_soc, self.soc_min, self.soc_max))
        if not np.isclose(self.soc,new_soc):
            info["action_overstep"] = info.get("action_overstep", 0) + 1

        # actual energy transferred (may be SoC-limited)
        actual_delta_soc = self.soc - old_soc
        actual_energy_kWh = actual_delta_soc * self.max_cap_kWh

        # actual power, for reporting
        self.actual_power_kW = actual_energy_kWh / time_duration_in_hours if time_duration_in_hours > 0 else 0.0

        # rewrite action to the clipped fraction
        actual_action = self.actual_power_kW / self.max_power_kW if self.max_power_kW > 0 else 0.0
        actions["a_battery"] = np.array([np.float32(actual_action)], dtype=np.float32)


    def update_state(self, states: Dict, info: dict) -> None:
        super().update_state(states, info)
        states["s_battery_soc"][0] = np.float32(self.soc)
        self._write_ctxt(states, "ctxt_battery_capacity_kWh", np.float32(self.max_cap_kWh))
        self._write_ctxt(states, "ctxt_battery_power_kW", np.float32(self.max_power_kW))

    def reset(self, states: Dict, info: dict) -> None:
        """Reset SoC to start_soc each episode (base would carry it over).

        With ``start_soc_jitter`` > 0, perturb by a uniform offset from the env rng
        (info["_rng"], deterministic per-worker; standalone fallback), then clip.
        """
        self.soc = self.start_soc_percentage
        if self.start_soc_jitter > 0.0:
            rng = info.get("_rng") or np.random.default_rng()
            self.soc += rng.uniform(-self.start_soc_jitter, self.start_soc_jitter)
        self.soc = float(np.clip(self.soc, self.soc_min, self.soc_max))
        
        self.actual_power_kW = 0.0
        super().reset(states, info)

    def get_E(self, actions: Dict) -> tuple[float, float]:
        # actual_power_kW > 0 = charging (consumes); returns (production, consumption)
        if self.actual_power_kW > 0.0:
            return 0.0, self.actual_power_kW
        else:
            # < 0 = discharging (produces to others)
            return -1.0 * self.actual_power_kW, 0.0


ComponentRegistry.register('infrastructure', BatteryLinear)
