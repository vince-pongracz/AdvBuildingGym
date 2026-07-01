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
    WeatherDataSource) to power via a v^k power curve (cubic by default) with cut-in/rated/cut-out
    thresholds; no policy input. The policy observes only the resulting
    ``s_wind_power_norm`` (production as a fraction of ``max_power_kW``).
    Link: https://en.wikipedia.org/wiki/Wind_turbine_design#Power_curve

    Forecastable: ``s_fc_wind_power_norm`` applies the same power curve to the future wind
    speed that WeatherDataSource publishes on ``info["raw_weather_forecast"]`` (a plain data
    channel — no reference to the weather component).
    """

    POWER_FLOW = "generator"

    # Exponent k of the cut-in→rated power curve P = a + b·vᵏ (3 → cubic, P ∝ v³).
    POWER_CURVE_EXPONENT_K: ClassVar[float] = 3.0

    _exclude_params: ClassVar[Set[str]] = {
        'iteration', 'wind_speed_raw', 'current_production_kW'
    }

    def __init__(self,
                name: str,
                max_power_kW: float,
                cut_in_speed_ms: float = 3.0,
                rated_speed_ms: float = 10.0,
                cut_out_speed_ms: float = 25.0,
                emit_ctxt: bool = False,
                ) -> None:
        """Initialize wind turbine infrastructure.

        Args:
            name: Component identifier.
            max_power_kW: Maximum power export = rated electrical output at rated wind speed (kW).
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

        self.cut_in_speed_ms = cut_in_speed_ms
        self.rated_speed_ms = rated_speed_ms
        self.cut_out_speed_ms = cut_out_speed_ms

        # Pre-compute the curve coefficients a, b so P(v_c)=0 and P(v_r)=P_max.
        # a = P_max·v_cᵏ / (v_cᵏ − v_rᵏ),  b = P_max / (v_rᵏ − v_cᵏ)
        cut_in_k = self.cut_in_speed_ms ** self.POWER_CURVE_EXPONENT_K
        rated_k = self.rated_speed_ms ** self.POWER_CURVE_EXPONENT_K
        self._curve_a = self.max_power_kW * cut_in_k / (cut_in_k - rated_k)
        self._curve_b = self.max_power_kW / (rated_k - cut_in_k)

        # State variables
        self.wind_speed_raw = 0.0          # Wind speed (m/s, from info)
        self.current_production_kW = 0.0   # Power produced (kW)
        # Reference to WeatherDataSource's stable forecast channel (info["raw_weather_forecast"]),
        # captured each update_state — NOT the whole info dict. forecast() reads the future raw
        # wind speed from it.
        self._weather_forecast: dict[str, list[float]] | None = None

    def setup_spaces(self,
                    state_spaces,
                    action_spaces):
        """Register normalised production + rated-power context; no action (wind-driven)."""
        # Normalised production [0, 1] (fraction of max_power_kW) — the only
        # weather-derived observation the policy sees for wind.
        if "s_wind_power_norm" not in state_spaces.keys():
            state_spaces["s_wind_power_norm"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        # Raw rated power (kW) — policy-only conditioning, gated by emit_ctxt;
        # lets the policy recover absolute power from the normalised obs.
        self._publish_ctxt(state_spaces, "ctxt_wind_rated_power_kW",
                        Box(low=0, high=np.inf, shape=(1,), dtype=np.float32))

        return state_spaces, action_spaces

    def exec_action(self, actions: Dict, states: Dict, info: dict) -> None:
        """Read raw wind speed (m/s) from info, then apply the power curve."""
        self.wind_speed_raw = float(info.get("raw_wind_speed_ms", 0.0))

        # P = a + b*v^k between cut-in and rated (k=3 -> cubic by default)
        # Critical analysis of methods for mathematical modelling of wind turbines
        # Link: https://www.sciencedirect.com/science/article/pii/S0960148111001303
        self.current_production_kW = self._power_curve(self.wind_speed_raw)

    def _power_curve(self, wind_speed: float) -> float:
        """Power curve (kW) from wind speed (m/s).

        Four operating regimes (vᵏ model, coefficients a, b pre-computed in __init__):
        - Below cut-in (v < v_c) or above cut-out (v > v_f): no generation
        - Cut-in to rated (v_c ≤ v ≤ v_r): P = a + b·vᵏ
        - Rated to cut-out (v_r < v ≤ v_f): full rated power

        with  a = P_max·v_cᵏ / (v_cᵏ − v_rᵏ)  and  b = P_max / (v_rᵏ − v_cᵏ).
        Critical analysis of methods for mathematical modelling of wind turbines
        Link: https://www.sciencedirect.com/science/article/pii/S0960148111001303

        Args:
            wind_speed: Wind speed in m/s.

        Returns:
            Power output in kW.
        """
        if wind_speed < self.cut_in_speed_ms:
            return 0.0
        elif wind_speed <= self.rated_speed_ms:
            # vᵏ ramp cut-in → rated
            return self._curve_a + self._curve_b * wind_speed ** self.POWER_CURVE_EXPONENT_K
        elif wind_speed <= self.cut_out_speed_ms:
            return self.max_power_kW
        else:
            # safety shutdown above cut-out
            return 0.0

    def update_state(self, states: Dict, info: dict) -> None:
        """Publish normalised production, rated power, and power bounds."""
        super().update_state(states, info)

        # Capture only the weather-forecast channel for forecast() -- not the whole info dict.
        # WeatherDataSource updates this dict in place later in the step, so the reference
        # reflects the row[t+1] look-ahead by the time forecast() runs.
        self._weather_forecast = info.get("raw_weather_forecast")
        wind_power_norm = self.current_production_kW / self.max_power_kW if self.max_power_kW > 0 else 0.0
        states["s_wind_power_norm"][0] = np.float32(np.clip(wind_power_norm, 0.0, 1.0))
        self._write_ctxt(states, "ctxt_wind_rated_power_kW", np.float32(self.max_power_kW))

    def reset(self, states: Dict, info: dict) -> None:
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
        """Normalised wind production at the future weather rows (same curve as the live obs).

        Reads the future raw wind speed from ``info["raw_weather_forecast"]["wind_ms"]``
        (published by WeatherDataSource over info channel), so the count matches
        ``selected_future_steps`` by construction.
        """
        n = len(selected_future_steps)
        forecast = self._weather_forecast
        if forecast is None or self.max_power_kW <= 0:
            return {"s_fc_wind_power_norm": [0.0] * n}
        future_wind = forecast.get("wind_ms", [])
        if len(future_wind) != n:
            raise ValueError(
                f"WindTurbine '{self.name}': published weather forecast has "
                f"{len(future_wind)} step(s) but {n} were requested — "
                "info['forecast_steps'] must equal the ForecastWrapper step set."
            )
        out = [self._power_curve(float(v)) / self.max_power_kW for v in future_wind]
        return {"s_fc_wind_power_norm": [float(np.clip(p, 0.0, 1.0)) for p in out]}


# register with ComponentRegistry
ComponentRegistry.register('infrastructure', WindTurbine)
