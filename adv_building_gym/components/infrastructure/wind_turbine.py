import logging
from typing import ClassVar, Dict, Set

import numpy as np
from gymnasium.spaces import Box

from .base import Infrastructure
from ..statesources.forecastable import Forecastable
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class WindTurbine(Infrastructure, Forecastable):
    """Wind turbine (uncontrolled generator).

    Converts raw wind speed (m/s, read from ``info["raw_wind_speed_ms"]`` published by
    WeatherDataSource) to power via a cubic power curve with cut-in/rated/cut-out
    thresholds; no policy input. The policy observes only the resulting
    ``s_wind_power_norm`` (production as a fraction of ``rated_power_kW``).
    Link: https://en.wikipedia.org/wiki/Wind_turbine_design#Power_curve

    Forecastable: ``s_fc_wind_power_norm`` applies the same power curve to the weather
    source's future wind speed (obtained via ``info["_weather_source"]``).
    """

    POWER_FLOW = "generator"

    _exclude_params: ClassVar[Set[str]] = {
        'iteration', 'wind_speed_raw', 'current_production_kW'
    }

    def __init__(self,
                name: str,
                max_power_kW: float,
                rated_power_kW: float = 5.0,
                cut_in_speed_ms: float = 3.0,
                rated_speed_ms: float = 10.0,
                cut_out_speed_ms: float = 25.0,
                emit_ctxt: bool = False,
                ) -> None:
        """Initialize wind turbine infrastructure.

        Args:
            name: Component identifier.
            max_power_kW: Maximum power export in kW (typically = rated_power_kW).
            rated_power_kW: Rated electrical output at rated wind speed (kW).
            cut_in_speed: Minimum wind speed for power generation (m/s).
                Typical range for small turbines: 2.5–4.0 m/s.
                Link: https://en.wikipedia.org/wiki/Cut-in_speed
            rated_speed: Wind speed at which rated power is reached (m/s).
                Typical range for small turbines: 10–14 m/s.
            cut_out_speed: Wind speed above which turbine shuts down for
                safety (m/s). IEC 61400-2 standard value for small turbines.
                Link: https://webstore.iec.ch/en/publication/5433
        """
        super().__init__(name, max_power_kW)
        self.emit_ctxt = emit_ctxt

        if cut_in_speed_ms >= rated_speed_ms:
            raise ValueError("cut_in_speed must be less than rated_speed.")
        if rated_speed_ms >= cut_out_speed_ms:
            raise ValueError("rated_speed must be less than cut_out_speed.")

        self.rated_power_kW = rated_power_kW
        self.cut_in_speed = cut_in_speed_ms
        self.rated_speed = rated_speed_ms
        self.cut_out_speed = cut_out_speed_ms

        # State variables
        self.wind_speed_raw = 0.0          # Wind speed (m/s, from info)
        self.current_production_kW = 0.0   # Power produced (kW)
        # Weather source captured from info each update_state; used for power forecasts.
        self._weather_source = None

    def setup_spaces(self,
                    state_spaces,
                    action_spaces):
        """Register normalised production + rated-power context; no action (wind-driven)."""
        # Normalised production [0, 1] (fraction of rated_power_kW) — the only
        # weather-derived observation the policy sees for wind.
        if "s_wind_power_norm" not in state_spaces.keys():
            state_spaces["s_wind_power_norm"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        # Raw rated power (kW) — policy-only conditioning, gated by emit_ctxt;
        # lets the policy recover absolute power from the normalised obs.
        self._publish_ctxt(state_spaces, "ctxt_wind_rated_power_kW",
                        Box(low=0, high=np.inf, shape=(1,), dtype=np.float32))

        return state_spaces, action_spaces

    def exec_action(self, actions: Dict, states: Dict, info=None) -> None:
        """Read raw wind speed (m/s) from info, then apply the cubic power curve."""
        self.wind_speed_raw = float(info.get("raw_wind_speed_ms", 0.0)) if info is not None else 0.0

        # cubic curve P ∝ v³ between cut-in and rated (Betz's law)
        # Link: https://en.wikipedia.org/wiki/Betz%27s_law
        self.current_production_kW = self._power_curve(self.wind_speed_raw)

    def _power_curve(self, wind_speed: float) -> float:
        """Cubic power curve (kW) from wind speed (m/s).

        Three operating regimes:
        - Below cut-in: no generation
        - Cut-in to rated: cubic ramp  P = P_rated * ((v - v_ci) / (v_r - v_ci))³
        - Rated to cut-out: full rated power
        - Above cut-out: shutdown (0 power)

        Link: https://en.wikipedia.org/wiki/Wind_turbine_design#Power_curve

        Args:
            wind_speed: Wind speed in m/s.

        Returns:
            Power output in kW.
        """
        if wind_speed < self.cut_in_speed:
            return 0.0
        elif wind_speed < self.rated_speed:
            # cubic ramp cut-in → rated
            fraction = (wind_speed - self.cut_in_speed) / (self.rated_speed - self.cut_in_speed)
            return self.rated_power_kW * fraction ** 3
        elif wind_speed <= self.cut_out_speed:
            return self.rated_power_kW
        else:
            # safety shutdown above cut-out
            return 0.0

    def update_state(self, states: Dict, info=None) -> None:
        """Publish normalised production, rated power, and power bounds."""
        super().update_state(states, info)

        # Capture the weather source for forecast() (future wind-speed look-ahead).
        if info is not None:
            self._weather_source = info.get("_weather_source")
        wind_power_norm = self.current_production_kW / self.rated_power_kW if self.rated_power_kW > 0 else 0.0
        states["s_wind_power_norm"][0] = np.float32(np.clip(wind_power_norm, 0.0, 1.0))
        self._write_ctxt(states, "ctxt_wind_rated_power_kW", np.float32(self.rated_power_kW))

    def reset(self, states: Dict, info=None) -> None:
        """Clear per-episode wind/production readouts."""
        self.wind_speed_raw = 0.0
        self.current_production_kW = 0.0
        super().reset(states, info)

    
    def get_E(self, actions: Dict) -> tuple[float, float]:
        """Returns (production, consumption); wind only produces."""
        return self.current_production_kW, 0.0

    def get_raw_values(self) -> dict[str, float]:
        return {
            "raw_wind_speed": self.wind_speed_raw,
            "raw_wind_production_kW": self.current_production_kW,
        }

    def forecast_keys(self) -> tuple[str, ...]:
        return ("s_fc_wind_power_norm",)

    def forecast(self, selected_future_steps: list[int]) -> dict[str, list[float]]:
        """Normalised wind production at the future weather rows (same curve as the live obs)."""
        n = len(selected_future_steps)
        if self._weather_source is None or self.rated_power_kW <= 0:
            return {"s_fc_wind_power_norm": [0.0] * n}
        future_wind = self._weather_source.raw_weather_forecast(selected_future_steps)["wind_ms"]
        out = [self._power_curve(float(v)) / self.rated_power_kW for v in future_wind]
        return {"s_fc_wind_power_norm": [float(np.clip(p, 0.0, 1.0)) for p in out]}


# register with ComponentRegistry
ComponentRegistry.register('infrastructure', WindTurbine)
