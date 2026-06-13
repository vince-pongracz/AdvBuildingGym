"""Base class and shared helpers for rule-based control strategies.

Strategies are heuristic baselines (no RL): they emit native Dict actions on
the unwrapped ``AdvBuildingGym`` and read measured power flows from the
previous step's ``info["power_breakdown"]`` — a one-step measurement lag,
acceptable for rule-based control (step 0 falls back to the zero action).

Unlike RL policies, strategies may hold component references (battery, price
statesource): a real HEMS measures battery SoC directly and day-ahead prices
are public, so this is not privileged information in the modelled sense.
"""

import logging
from typing import ClassVar

import numpy as np

from adv_building_gym._common.constants import SECONDS_PER_HOUR
from adv_building_gym.components.infrastructure.battery_models.battery_linear import BatteryLinear
from adv_building_gym.components.infrastructure.battery_models.battery_tremblay import BatteryTremblay
from adv_building_gym.components.infrastructure.solar_panel import SolarPanel
from adv_building_gym.components.infrastructure.wind_turbine import WindTurbine
from adv_building_gym.components.statesources.outer.energy_price import EnergyPriceDataSource
from adv_building_gym.core.env import AdvBuildingGym
from . import _strategy_utils as utils

logger = logging.getLogger(__name__)

BATTERY_CLASSES = (BatteryLinear, BatteryTremblay)
RENEWABLE_CLASSES = (SolarPanel, WindTurbine)


def in_time_window(hour: float, start: float, end: float) -> bool:
    """Half-open window check [start, end) in hours, wrap-around safe (start > end spans midnight)."""
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


class RuleBasedStrategy:
    """Rule-based Dict-action controller for ``AdvBuildingGym``.

    Subclasses implement ``_decide(obs, last_info)``; all concrete strategies
    extend this base directly (flat hierarchy, shared logic lives in helpers).
    """

    name: ClassVar[str] = "base"
    requires_battery: ClassVar[bool] = False
    requires_price: ClassVar[bool] = False

    def __init__(self, env: AdvBuildingGym, *, preserve_start_soc: bool = True):
        """preserve_start_soc: forbid ending an episode with less SoC than it started
        (caps every discharge at the episode-start SoC floor)."""
        self.env = env
        self.preserve_start_soc = preserve_start_soc
        self.control_step_s = int(env.env_config.CONTROL_STEP)
        self.episode_length = int(env.env_config.EPISODE_LENGTH)
        self._action_spaces = dict(env.action_space.spaces)

        self.battery = next((i for i in env.infras if isinstance(i, BATTERY_CLASSES)), None)
        self.renewable_names = [i.name for i in env.infras if isinstance(i, RENEWABLE_CLASSES)]
        self.price_source = next(
            (ds for ds in env.statesources if isinstance(ds, EnergyPriceDataSource)), None,
        )

        if self.requires_battery and self.battery is None:
            raise ValueError(
                f"Strategy '{self.name}' requires a battery infrastructure; "
                f"trial has: {[i.name for i in env.infras]}"
            )
        if self.requires_price and self.price_source is None:
            raise ValueError(
                f"Strategy '{self.name}' requires an EnergyPriceDataSource; "
                f"trial has: {[type(ds).__name__ for ds in env.statesources]}"
            )

        self._step = 0
        self.start_soc = 0.0

    # ------------------------------------------------------------------ lifecycle
    def reset(self, obs: dict) -> None:
        """Per-episode reset; call right after ``env.reset()``."""
        self._step = 0
        if self.battery is not None:
            # battery.reset() already applied the per-episode start-SoC jitter
            self.start_soc = float(self.battery.soc)

    def act(self, obs: dict, last_info: dict | None) -> dict[str, np.ndarray]:
        """Return a fresh Dict action (components back-calculate clipped values in place)."""
        action = self._decide(obs, last_info)
        self._step += 1
        return action

    def _decide(self, obs: dict, last_info: dict | None) -> dict[str, np.ndarray]:
        raise NotImplementedError

    # ------------------------------------------------------------------ action helpers
    # Logic lives in _strategy_utils (pure, instance-free); these wrappers just
    # bind the strategy's action spaces / measured state.
    def zero_action(self) -> dict[str, np.ndarray]:
        """No-op Dict action: 0 for every key, clipped into the space bounds."""
        return utils.zero_action(self._action_spaces)

    def _battery_action(self, value: float) -> dict[str, np.ndarray]:
        return utils.battery_action(self._action_spaces, value)

    # ------------------------------------------------------------------ measurements
    def _renewable_surplus_kW(self, last_info: dict | None) -> float:
        """Renewable production minus all non-battery consumption, from the
        previous step's power_breakdown {name: (production_kW, consumption_kW)}.
        Positive = surplus available for charging; negative = deficit."""
        battery_name = self.battery.name if self.battery is not None else None
        return utils.renewable_surplus_kW(last_info, self.renewable_names, battery_name)

    def _hour_of_day(self) -> float:
        """Hour-of-day derived from step count × control step; assumes the
        episode starts at midnight (standard config: 288 × 300 s = one day)."""
        return utils.hour_of_day(self._step, self.control_step_s)

    # ------------------------------------------------------------------ SoC headrooms
    def _soc_floor(self) -> float:
        floor = self.battery.soc_min
        if self.preserve_start_soc:
            floor = max(floor, self.start_soc)
        return floor

    def _discharge_headroom_kW(self) -> float:
        """Max discharge power this step without dropping below the SoC floor."""
        dt_h = self.control_step_s / SECONDS_PER_HOUR
        return max(0.0, (self.battery.soc - self._soc_floor()) * self.battery.max_cap_kWh / dt_h)

    def _charge_headroom_kW(self) -> float:
        """Max charge power this step without exceeding soc_max."""
        dt_h = self.control_step_s / SECONDS_PER_HOUR
        return max(0.0, (self.battery.soc_max - self.battery.soc) * self.battery.max_cap_kWh / dt_h)

    def _charge_value(self, power_kW: float) -> float:
        """a_battery in [0, 1] charging min(power, headroom); surplus beyond it exports to the grid."""
        if self.battery.max_power_kW <= 0:
            return 0.0
        power_kW = min(power_kW, self._charge_headroom_kW())
        return float(np.clip(power_kW / self.battery.max_power_kW, 0.0, 1.0))

    def _discharge_value(self, power_kW: float) -> float:
        """a_battery in [-1, 0] discharging min(power, SoC-floor headroom)."""
        if self.battery.max_power_kW <= 0:
            return 0.0
        power_kW = min(power_kW, self._discharge_headroom_kW())
        return float(-np.clip(power_kW / self.battery.max_power_kW, 0.0, 1.0))
