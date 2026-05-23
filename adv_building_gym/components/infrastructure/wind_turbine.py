import logging
from typing import ClassVar, Dict, Set

import numpy as np
from gymnasium.spaces import Box

from .base import Infrastructure
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class WindTurbine(Infrastructure):
    """Wind turbine infrastructure (uncontrolled generator).

    Reads normalised wind speed from the WeatherDataSource
    (``avg_wind_speed_norm`` in states) and converts it to electrical
    power output using a cubic power curve with cut-in/rated/cut-out
    thresholds. Production is fully determined by wind speed — there
    is no policy-controlled input (mirrors the SolarPanel design).

    Power curve reference:
    Link: https://en.wikipedia.org/wiki/Wind_turbine_design#Power_curve
    """

    POWER_FLOW = "generator"

    _exclude_params: ClassVar[Set[str]] = {
        'iteration', 'wind_speed_raw',
        'current_production_kW', 'wind_speed_abs_max'
    }

    def __init__(self,
                name: str,
                max_power_kW: float,
                rated_power_kW: float = 5.0,
                cut_in_speed_ms: float = 3.0,
                rated_speed_ms: float = 10.0,
                cut_out_speed_ms: float = 25.0,
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

        if cut_in_speed_ms >= rated_speed_ms:
            raise ValueError("cut_in_speed must be less than rated_speed.")
        if rated_speed_ms >= cut_out_speed_ms:
            raise ValueError("rated_speed must be less than cut_out_speed.")

        self.rated_power_kW = rated_power_kW
        self.cut_in_speed = cut_in_speed_ms
        self.rated_speed = rated_speed_ms
        self.cut_out_speed = cut_out_speed_ms

        # Scale factor for denormalising avg_wind_speed_norm back to m/s.
        # Read at runtime from states["ctxt_wind_speed_abs_max"] (published by WeatherDataSource).
        self.wind_speed_abs_max: float = 1.0

        # State variables
        self.wind_speed_raw = 0.0          # Denormalised wind speed (m/s)
        self.current_production_kW = 0.0   # Power produced (kW)

    def setup_spaces(self,
                    state_spaces,
                    action_spaces):
        """Setup observation space for wind turbine.

        Wind speed observation is registered by WeatherDataSource
        (``avg_wind_speed_norm``). No action is added — production is
        fully determined by wind speed.
        """
        # Raw rated power (kW) — static context variable, only
        # changes between episodes if the config is swapped.
        if "ctxt_wind_rated_power_kW" not in state_spaces.keys():
            state_spaces["ctxt_wind_rated_power_kW"] = Box(
                low=0, high=np.inf, shape=(1,), dtype=np.float32
            )

        return state_spaces, action_spaces

    def exec_action(self, actions: Dict, states: Dict, info=None) -> None:
        """Compute wind power output from the current wind speed.

        1. Denormalise wind speed from ``avg_wind_speed_norm``.
        2. Apply cubic power curve with cut-in/rated/cut-out thresholds.
        """
        # Read normalised wind speed from state
        wind_norm = 0.0
        if "s_avg_wind_speed_norm" in states:
            wind_norm = float(states["s_avg_wind_speed_norm"][0])

        # Denormalise to m/s using scale factor from WeatherDataSource
        self.wind_speed_raw = wind_norm * self.wind_speed_abs_max

        # Compute power from cubic power curve
        # P ∝ v³ between cut-in and rated speed (Betz's law)
        # Link: https://en.wikipedia.org/wiki/Betz%27s_law
        self.current_production_kW = self._power_curve(self.wind_speed_raw)

    def _power_curve(self, wind_speed: float) -> float:
        """Compute available power from wind speed using a cubic power curve.

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
            # Cubic ramp from cut-in to rated speed
            fraction = (wind_speed - self.cut_in_speed) / (self.rated_speed - self.cut_in_speed)
            return self.rated_power_kW * fraction ** 3
        elif wind_speed <= self.cut_out_speed:
            return self.rated_power_kW
        else:
            # Safety shutdown above cut-out speed
            return 0.0

    def update_state(self, states: Dict, info=None) -> None:
        """Read wind speed scale factor from state and publish power bounds."""
        super().update_state(states, info)

        if "ctxt_wind_speed_abs_max" in states:
            self.wind_speed_abs_max = float(states["ctxt_wind_speed_abs_max"][0])
        states["ctxt_wind_rated_power_kW"][0] = np.float32(self.rated_power_kW)

    def reset(self, states: Dict, info=None) -> None:
        """Clear per-episode wind/production readouts."""
        self.wind_speed_raw = 0.0
        self.current_production_kW = 0.0
        super().reset(states, info)

    
    def get_E(self, actions: Dict) -> tuple[float, float]:
        """Get current electric energy production, consumption from wind turbine.
        """
        return self.current_production_kW, 0.0

    def get_raw_values(self) -> dict[str, float]:
        return {
            "raw_wind_speed": self.wind_speed_raw,
            "raw_wind_production_kW": self.current_production_kW,
        }


# Register WindTurbine with the component registry
ComponentRegistry.register('infrastructure', WindTurbine)
