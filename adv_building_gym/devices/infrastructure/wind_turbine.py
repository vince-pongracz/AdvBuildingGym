import logging
from typing import ClassVar, Dict, Set

import numpy as np
from gymnasium.spaces import Box

from .base import Infrastructure
from adv_building_gym.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)


class WindTurbine(Infrastructure):
    """Wind turbine infrastructure with policy-controlled curtailment.

    Reads normalised wind speed from the WeatherDataSource
    (``avg_wind_speed_norm`` in states) and converts it to electrical
    power output using a cubic power curve with cut-in/rated/cut-out
    thresholds.

    The agent controls a continuous curtailment action in [0, 1]:
    0 = fully curtailed (no power fed in), 1 = full utilisation.

    Power curve reference:
    Link: https://en.wikipedia.org/wiki/Wind_turbine_design#Power_curve

    wind_action value: 1 = full rated production, 0 = no production.
    """

    _exclude_params: ClassVar[Set[str]] = {
        'iteration', 'wind_speed_raw', 'available_power_kW',
        'current_production_kW', 'wind_speed_abs_max'
    }

    def __init__(self,
                name: str,
                max_power_kW: float,
                rated_power_kW: float = 5.0,
                cut_in_speed: float = 3.0,
                rated_speed: float = 12.0,
                cut_out_speed: float = 25.0,
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

        if cut_in_speed >= rated_speed:
            raise ValueError("cut_in_speed must be less than rated_speed.")
        if rated_speed >= cut_out_speed:
            raise ValueError("rated_speed must be less than cut_out_speed.")

        self.rated_power_kW = rated_power_kW
        self.cut_in_speed = cut_in_speed
        self.rated_speed = rated_speed
        self.cut_out_speed = cut_out_speed

        # Scale factor for denormalising avg_wind_speed_norm back to m/s.
        # Set at runtime from info["_wind_speed_abs_max"] (published by WeatherDataSource).
        self.wind_speed_abs_max: float = 1.0

        # State variables
        self.wind_speed_raw = 0.0          # Denormalised wind speed (m/s)
        self.available_power_kW = 0.0      # Power before curtailment (kW)
        self.current_production_kW = 0.0   # Power after curtailment (kW)

    @property
    def max_consumption_kW(self) -> float:
        return 0.0

    @property
    def max_export_kW(self) -> float:
        return self.max_power_kW

    def setup_spaces(self,
                    state_spaces,
                    action_spaces):
        """Setup action space for wind turbine curtailment.

        Wind speed observation is already registered by WeatherDataSource
        (``avg_wind_speed_norm``).  Only the curtailment action is added here.
        """
        action_spaces["wind_curtailment"] = Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )

        # Raw rated power (kW) — static context variable, only
        # changes between episodes if the config is swapped.
        if "wind_rated_power_kW" not in state_spaces.keys():
            state_spaces["wind_rated_power_kW"] = Box(
                low=0, high=np.inf, shape=(1,), dtype=np.float32
            )

        return state_spaces, action_spaces

    def exec_action(self, actions: Dict, states: Dict, info=None) -> None:
        """Compute wind power output and apply curtailment.

        1. Denormalise wind speed from ``avg_wind_speed_norm``.
        2. Apply cubic power curve with cut-in/rated/cut-out thresholds.
        3. Multiply by the policy's curtailment action.
        """
        # Read normalised wind speed from state
        wind_norm = 0.0
        if "avg_wind_speed_norm" in states:
            wind_norm = float(states["avg_wind_speed_norm"][0])

        # Denormalise to m/s using scale factor from WeatherDataSource
        self.wind_speed_raw = wind_norm * self.wind_speed_abs_max

        # Compute available power from cubic power curve
        # P ∝ v³ between cut-in and rated speed (Betz's law)
        # Link: https://en.wikipedia.org/wiki/Betz%27s_law
        self.available_power_kW = self._power_curve(self.wind_speed_raw)

        # Apply curtailment action
        curtailment = float(np.atleast_1d(actions.get(
            "wind_curtailment", np.array([1.0])
        ))[0])
        curtailment = float(np.clip(curtailment, 0.0, 1.0))

        self.current_production_kW = self.available_power_kW * curtailment

        # Write normalised production as output (1 = full production, 0 = none)
        wind_action = self.current_production_kW / self.rated_power_kW if self.rated_power_kW > 0 else 0.0
        wind_action = float(np.clip(wind_action, 0.0, 1.0))

        if "wind_action" not in actions:
            actions["wind_action"] = np.array([wind_action], dtype=np.float32)
        else:
            actions["wind_action"][0] = wind_action

    def _power_curve(self, wind_speed: float) -> float:
        """Compute available power from wind speed using a cubic power curve.

        Three operating regimes:
        - Below cut-in: no generation
        - Cut-in to rated: cubic ramp  P = P_rated × ((v - v_ci) / (v_r - v_ci))³
        - Rated to cut-out: full rated power
        - Above cut-out: shutdown (0 power)

        Link: https://en.wikipedia.org/wiki/Wind_turbine_design#Power_curve

        Args:
            wind_speed: Wind speed in m/s.

        Returns:
            Available power in kW (before curtailment).
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
        """Read wind speed scale factor from info and publish power bounds."""
        wind_abs_max = (info or {}).get("_wind_speed_abs_max")
        if wind_abs_max is not None:
            self.wind_speed_abs_max = float(wind_abs_max)
        super().update_state(states, info)
        states["wind_rated_power_kW"][0] = np.float32(self.rated_power_kW)

    def get_electric_consumption(self, actions: Dict) -> float:
        """Get current electric energy consumption (production) from wind turbine.

        Sign convention: positive = consumption, negative = production.
        Wind turbines produce energy, so this returns a negative value.
        """
        return -self.current_production_kW

    def get_penalisable_consumption(self, actions: Dict, states: Dict) -> float:
        """Generation source — always exempt from energy penalty."""
        return 0.0

    def get_raw_values(self) -> dict[str, float]:
        return {
            "wind_speed_raw": self.wind_speed_raw,
            "wind_available_kW": self.available_power_kW,
            "wind_production_kW": self.current_production_kW,
        }


# Register WindTurbine with the component registry
ComponentRegistry.register('infrastructure', WindTurbine)
