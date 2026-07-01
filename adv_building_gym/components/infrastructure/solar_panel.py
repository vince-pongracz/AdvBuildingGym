import logging
from typing import ClassVar, Dict, Set

import numpy as np
from gymnasium.spaces import Box

from .base import Infrastructure
from ..statesources.forecastable import Forecastable
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)

class SolarPanel(Infrastructure, Forecastable):
    """Solar PV (uncontrolled generator).

    Production is a pure function of irradiance — no policy action. 
    Raw irradiance (W/m²) read from the shared info channel (``info["raw_solar_irradiance_W_m2"]``,
    published by ``WeatherDataSource``); 
    output ``P[kW] = G*A*η/1000`` clipped to ``max_power_kW``. 
    The policy observes only the resulting ``s_pv_power_norm`` (the production as a fraction of ``max_power_kW``), not the raw weather feature.

    Forecastable: ``s_fc_pv_power_norm`` applies the same curve to the future irradiance
    that WeatherDataSource publishes on ``info["raw_weather_forecast"]`` (a plain data
    channel).
    """

    POWER_FLOW = "generator"

    # Internal state variables - don't serialize
    _exclude_params: ClassVar[Set[str]] = {
        'iteration', 'irradiance_W_m2', 'current_production_kW'
    }

    def __init__(self,
                name: str,
                max_power_kW: float,
                control_step: int,
                # Link: https://www.ise.fraunhofer.de/content/dam/ise/de/documents/publications/studies/Photovoltaics-Report.pdf
                pv_efficiency: float,
                panel_area_m2: float,
                emit_ctxt: bool = False
                ) -> None:
        """max_power_kW: peak output under standard test conditions (STC)."""
        # NOTE VP 2026.01.24. : Inverter efficiency is not considered,
        # max power means peak output power, produced by the solar panel.
        super().__init__(name, max_power_kW)
        self.emit_ctxt = emit_ctxt

        # State variables
        self.irradiance_W_m2 = 0.0  # Global irradiance in W/m² (raw, denormalised)
        self.current_production_kW = 0.0  # Actual power production in kW
        self.pv_efficiency = pv_efficiency
        self.panel_area_m2 = panel_area_m2
        self.control_step = control_step
        # Reference to WeatherDataSource's stable forecast channel (info["raw_weather_forecast"]),
        # captured each update_state — NOT the whole info dict. forecast() reads the future raw
        # irradiance from it.
        self._weather_forecast: dict[str, list[float]] | None = None

    def _power_from_irradiance(self, irradiance_W_m2: float) -> float:
        """P[kW] = G[W/m²] * A[m²] * η / 1000, clipped to max_power_kW."""
        p = irradiance_W_m2 * self.panel_area_m2 * self.pv_efficiency / 1000.0
        return float(np.clip(p, 0.0, self.max_power_kW))

    def setup_spaces(self,
                    state_spaces,
                    action_spaces):
        """Register state space only — no action (production is irradiance-driven)."""

        # Normalised PV production [0, 1] (fraction of max_power_kW) — the only
        # weather-derived observation the policy sees for solar.
        if "s_pv_power_norm" not in state_spaces.keys():
            state_spaces["s_pv_power_norm"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Raw peak power capacity (kW) — policy-only conditioning, gated by emit_ctxt;
        # lets the policy recover absolute production from the normalised obs.
        self._publish_ctxt(state_spaces, "ctxt_solar_max_power_kW", Box(low=0, high=np.inf, shape=(1,), dtype=np.float32))

        return state_spaces, action_spaces


    def exec_action(self, actions: Dict, states: Dict, info: dict) -> None:
        """Compute solar production from raw irradiance (W/m², from info; no policy input)."""

        # Raw global irradiance (W/m²) handed over by WeatherDataSource on the info channel.
        # Irradiance is a flux, so power is independent of control_step.
        self.irradiance_W_m2 = float(info.get("raw_solar_irradiance_W_m2", 0.0))
        self.current_production_kW = self._power_from_irradiance(self.irradiance_W_m2)

    def update_state(self, states: Dict, info: dict) -> None:
        """Publish normalised production + static peak power into the observable state."""
        super().update_state(states, info)
        # Capture only the weather-forecast channel for forecast() (not the whole info dict).
        # WeatherDataSource updates this dict in place later in the step, so the reference
        # reflects the row[t+1] look-ahead by the time forecast() runs.
        self._weather_forecast = info.get("raw_weather_forecast")
        # Production as a fraction of capacity; efficiency/area are baked into this value.
        pv_power_norm = self.current_production_kW / self.max_power_kW if self.max_power_kW > 0 else 0.0
        states["s_pv_power_norm"][0] = np.float32(np.clip(pv_power_norm, 0.0, 1.0))
        self._write_ctxt(states, "ctxt_solar_max_power_kW", np.float32(self.max_power_kW))

    def reset(self, states: Dict, info: dict) -> None:
        """Clear per-episode irradiance/production readouts."""
        self.irradiance_W_m2 = 0.0
        self.current_production_kW = 0.0
        super().reset(states, info)
    
    def get_raw_values(self) -> Dict[str, float]:
        return {
            "raw_pv_prod_kW": self.current_production_kW,
            "raw_pv_max_kW": self.max_power_kW
        }

    def get_E(self, actions: Dict) -> tuple[float, float]:
        """Returns (production, consumption); solar only produces."""
        return self.current_production_kW, 0.0

    def forecast_keys(self) -> tuple[str, ...]:
        return ("s_fc_pv_power_norm",)

    def forecast(self, selected_future_steps: list[int]) -> dict[str, list[float]]:
        """Normalised PV production at the future weather rows (same curve as the live obs).

        Reads the future raw irradiance from ``info["raw_weather_forecast"]["solar_W_m2"]``
        (published by WeatherDataSource over ``info["forecast_steps"]``), so the count matches
        ``selected_future_steps`` by construction.
        """
        n = len(selected_future_steps)
        forecast = self._weather_forecast
        if forecast is None or self.max_power_kW <= 0:
            return {"s_fc_pv_power_norm": [0.0] * n}
        future_irradiance = forecast.get("solar_W_m2", [])
        if len(future_irradiance) != n:
            raise ValueError(
                f"SolarPanel '{self.name}': published weather forecast has "
                f"{len(future_irradiance)} step(s) but {n} were requested — "
                "info['forecast_steps'] must equal the ForecastWrapper step set."
            )
        out = [self._power_from_irradiance(float(g)) / self.max_power_kW for g in future_irradiance]
        return {"s_fc_pv_power_norm": [float(np.clip(p, 0.0, 1.0)) for p in out]}


# register with ComponentRegistry
ComponentRegistry.register('infrastructure', SolarPanel)
