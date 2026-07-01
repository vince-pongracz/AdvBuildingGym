import logging
from typing import ClassVar, Dict, Set

import numpy as np
from gymnasium.spaces import Box

from .base import Infrastructure
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class HouseholdEnergyConsumers(Infrastructure):
    """Passive household consumer (no policy action).

    Reads the normalised ``s_desired_energy_need`` signal (in [0, 1], published by
    DesiredUserEnergyNeed) and scales it to physical kW (``current_consumption_kW``),
    exposed via ``get_E`` as the consumption term. ``update_state`` republishes the
    normalised signal as the ``s_desired_energy_need`` observation. Positive =
    consumption drawn from grid.

    The peak scale is the dataset max published by DesiredUserEnergyNeed as
    ``ctxt_hh_consumption_max`` (``kW = norm * data_max``), which varies per data
    variant. ``peak_consumption_kW`` is only a fallback peak used when no
    ``ctxt_hh_consumption_max`` is published; a synthetic time-of-day profile likewise
    replaces ``s_desired_energy_need`` when that signal is absent.
    """

    POWER_FLOW = "consumer"

    # Internal state variables — don't serialize
    _exclude_params: ClassVar[Set[str]] = {
        'iteration', 'consumption_norm', 'current_consumption_kW', '_rng',
        '_effective_peak_kW',
    }

    def __init__(self, name: str, peak_consumption_kW: float) -> None:
        """peak_consumption_kW: fallback peak (kW) used only when no
        DesiredUserEnergyNeed source publishes ``ctxt_hh_consumption_max``."""
        super().__init__(name, peak_consumption_kW)

        self.peak_consumption_kW = peak_consumption_kW

        # State variables
        self.consumption_norm = 0.0  # Normalized consumption [0, 1]
        self.current_consumption_kW = 0.0  # Actual consumption in kW
        # Effective peak (kW) used to scale the normalised signal: the published
        # data max when available, else the configured fallback. Refreshed per episode.
        self._effective_peak_kW = peak_consumption_kW

        # Per-episode RNG for the synthetic fallback; rebound to env rng
        # (info["_rng"]) on reset(). Standalone default until first reset.
        self._rng = np.random.default_rng()

    def setup_spaces(self,
                    state_spaces,
                    action_spaces):
        """Register state space only — consumption is set by DesiredUserEnergyNeed, no action.

        The peak scale (``ctxt_hh_consumption_max``) is owned/published by
        DesiredUserEnergyNeed, so it is not declared here.
        """
        if "s_desired_energy_need" not in state_spaces:
            state_spaces["s_desired_energy_need"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        return state_spaces, action_spaces

    def exec_action(self, actions: Dict, states: Dict, info: dict) -> None:
        """Compute and store the current consumption.

        Reads ``s_desired_energy_need`` (DesiredUserEnergyNeed); falls back to a
        synthetic time-of-day profile if absent. Scales the normalised signal to kW
        (``current_consumption_kW``) by the effective peak from ``_resolve_peak_kW``.
        """
        # read normalised consumption signal
        if "s_desired_energy_need" in states:
            self.consumption_norm = float(states["s_desired_energy_need"][0])
        else:
            # Fallback: synthetic time-based profile
            self.consumption_norm = self._synthetic_consumption(states)

        # Scale normalised signal to physical kW by the dataset's own max (recovers
        # raw kW since DesiredUserEnergyNeed uses abs-min-max scaling); fall back to
        # the configured peak when no such source is present.
        self._effective_peak_kW = self._resolve_peak_kW(states)
        self.current_consumption_kW = self.consumption_norm * self._effective_peak_kW

    def update_state(self, states: Dict, info: dict) -> None:
        """Write current normalized consumption into states for observation."""
        states["s_desired_energy_need"] = np.array([self.consumption_norm], dtype=np.float32)

    def reset(self, states: Dict, info: dict) -> None:
        """Clear per-episode consumption readouts and refresh the effective peak.

        Statesources reset before infras, so ``ctxt_hh_consumption_max`` is already
        published here when a DesiredUserEnergyNeed source exists.
        """
        self.consumption_norm = 0.0
        self.current_consumption_kW = 0.0
        self._effective_peak_kW = self._resolve_peak_kW(states)
        # Bind to env rng (info["_rng"]) so fallback noise shares the
        # deterministic per-worker stream; standalone fallback otherwise.
        self._rng = info.get("_rng") or np.random.default_rng()
        super().reset(states, info)

    @property
    def max_consumption_kW(self) -> float:
        """Max grid draw (kW) = effective peak (data max when published, else fallback)."""
        return self._effective_peak_kW

    def _resolve_peak_kW(self, states: Dict) -> float:
        """Physical peak load (kW): the dataset max published by DesiredUserEnergyNeed
        (``ctxt_hh_consumption_max``) when present and positive, else the configured fallback."""
        if "ctxt_hh_consumption_max" in states:
            published = float(states["ctxt_hh_consumption_max"][0])
            if published > 0.0:
                return published
        return self.peak_consumption_kW

    def _synthetic_consumption(self, states: Dict) -> float:
        """Stepped time-of-day profile plus Gaussian noise."""
        # s_sim_hour is published by the env in [0, 1] (hour-of-day / 24).
        sim_hour = float(states.get("s_sim_hour", np.zeros(1, dtype=np.float32))[0]) * 24.0

        if sim_hour < 6:
            base = 0.2   # Low demand during night
        elif sim_hour < 9:
            base = 0.6   # Morning peak
        elif sim_hour < 17:
            base = 0.4   # Daytime moderate
        elif sim_hour < 21:
            base = 0.8   # Evening peak
        else:
            base = 0.3   # Late evening

        noise = self._rng.normal(loc=0.0, scale=0.05)
        return float(np.clip(base + noise, 0.0, 1.0))
    
    def get_raw_values(self) -> Dict[str, float]:
        return {
            "raw_current_consumption_kW": self.current_consumption_kW
        }

    def get_E(self, actions: Dict) -> tuple[float, float]:
        """Returns (production, consumption); positive consumption drawn from grid (kW)."""
        return 0.0, self.current_consumption_kW


# register with ComponentRegistry
ComponentRegistry.register('infrastructure', HouseholdEnergyConsumers)
