import logging
from collections import OrderedDict
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from ..base import StateSource
from adv_building_gym.components.registry import ComponentRegistry
from adv_building_gym._common.constants import SLOWDOWN_TERM, TEMP_ABS_MAX_CELSIUS

logger = logging.getLogger(__name__)

# Inner statesource: deterministic physics update of s_temp_in_norm (no action).
# Sole owner of envelope params (K, mC): both exposable to the policy as 
# ctxt_building_K / ctxt_building_mC via ctxt_keys; 
# mC is also shared on the info channel for HP.

class BuildingHeatLoss(StateSource):
    """Building heat loss/gain from indoor-outdoor temperature difference.

    1R1C envelope transfer ``Q_transfer = K * (T_out - T_in)`` applied to s_temp_in_norm.
    No time-series data needed — it only updates the indoor temperature state.
    """

    # timestep comes from env_meta control_step; K and mC are explicit YAML params.
    _context_params: ClassVar[Set[str]] = {'timestep'}

    # Endogenous physics: runs before the reward under the observed exogenous row
    # (see StateSource.UPDATE_PHASE / env._update_endogenous).
    UPDATE_PHASE: ClassVar[str] = "endogenous"

    def __init__(self,
                name: str,
                K: float,
                mC: float,
                timestep: float = 300,
                ctxt_keys: list[str] | None = None) -> None:
        """K: heat transfer coefficient [W/K]; mC: thermal mass [J/K]; timestep: seconds."""
        super().__init__(name=name)
        self.ctxt_keys = list(ctxt_keys) if ctxt_keys is not None else None
        self.K = K
        self.mC = mC
        self.timestep = timestep
        self.temp_in_raw = 0.0  # Denormalised indoor temp after this component's update (°C)

    def setup_spaces(self,
                    state_spaces: OrderedDict,
                    action_spaces: OrderedDict) -> tuple[OrderedDict, OrderedDict]:
        """Register the building K/mC context only.

        Indoor and outdoor temperature are both shared on the info channel
        (info["temp_in_norm"] / info["temp_out_norm"]), not observations — the policy
        sees the comfort error (s_temp_error_norm, published by InsideTemperature) instead.
        """
        # Static building physics — context, changes only on a building swap.
        # Both K and mC are policy-facing ctxt keys (added to the obs space only when listed
        # in ctxt_keys). mC is additionally shared on the info channel in update_state — HP
        # reads it there for its 1R1C update regardless of ctxt_keys.
        self._publish_ctxt(state_spaces, "ctxt_building_K", Box(low=0, high=np.inf, shape=(1,), dtype=np.float32))
        self._publish_ctxt(state_spaces, "ctxt_building_mC", Box(low=0, high=np.inf, shape=(1,), dtype=np.float32))

        return state_spaces, action_spaces

    def update_state(self, states: OrderedDict, info: dict) -> None:
        """Update indoor temperature with envelope heat loss/gain (after infra exec_action)."""

        # NOTE VP 2026.01.14. : Reference to the 1R1C thermal model
        # Paper: EKF based self-adaptive thermal model for a passive house
        # Link: https://www.sciencedirect.com/science/article/pii/S0378778812003039?via%3Dihub
        # 1R1C (SI): Q_transfer[W] = K*(Tout_raw - Tin_raw); dT_raw[K] = SLOWDOWN_TERM*dt*Q/mC.
        # Indoor and outdoor temperature both live on the info channel (normalised by
        # temp_abs_max); denormalise, apply physics, then renormalise the increment.
        Tin_norm = float(info.get("temp_in_norm", 0.0))
        Tout_norm = float(info.get("temp_out_norm", 0.0))
        # Fixed temperature normalisation scale from the info channel (WeatherDataSource).
        temp_abs_max = float(info["temp_abs_max"]) if "temp_abs_max" in info else TEMP_ABS_MAX_CELSIUS

        Tin_raw = Tin_norm * temp_abs_max
        Tout_raw = Tout_norm * temp_abs_max

        # Heat transfer -- drawn from inside to the outside
        Q_transfer = self.K * (Tout_raw - Tin_raw)

        # Temperature change due to heat loss (raw °C), then renormalise.
        dT_raw = SLOWDOWN_TERM * self.timestep * Q_transfer / self.mC
        dTemp_norm = dT_raw / temp_abs_max if temp_abs_max > 0 else 0.0

        # Apply heat loss to indoor temperature, clip to the ±1 normalised bounds.
        new_temp = float(np.clip(Tin_norm + dTemp_norm, -1.0, 1.0))
        info["temp_in_norm"] = new_temp

        # Cache raw indoor temp after heat loss. Runs after HP, so this final value
        # wins in RawStateTracker (infras collected before statesources).
        self.temp_in_raw = new_temp * temp_abs_max

        # K and mC are policy-facing ctxt keys (obs iff listed in ctxt_keys); mC is also
        # shared on the info channel so HP can read the thermal mass regardless of ctxt_keys.
        self._write_ctxt(states, "ctxt_building_K", np.float32(self.K))
        self._write_ctxt(states, "ctxt_building_mC", np.float32(self.mC))
        info["ctxt_building_mC"] = float(self.mC)

    def get_raw_values(self) -> dict[str, float]:
        return {"raw_temp_in": self.temp_in_raw}


# register with ComponentRegistry
ComponentRegistry.register('statesource', BuildingHeatLoss)
