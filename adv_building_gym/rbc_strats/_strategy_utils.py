"""Pure helpers for rule-based strategies.

Two concerns, both kept free of any strategy/env instance state so they are
independently testable and reusable across strategies:
  * action construction from the Dict action space, and
  * measurement extraction from the env's info channel / step clock.
"""

from collections.abc import Collection, Mapping

import numpy as np
from gymnasium.spaces import Space

from adv_building_gym._common.constants import SECONDS_PER_HOUR


# ---------------------------------------------------------------- action construction
def zero_action(action_spaces: Mapping[str, Space]) -> dict[str, np.ndarray]:
    """No-op Dict action: 0 for every key, clipped into the space bounds."""
    return {
        key: np.clip(np.zeros(space.shape, dtype=np.float32), space.low, space.high)
        for key, space in action_spaces.items()
    }


def battery_action(action_spaces: Mapping[str, Space], value: float) -> dict[str, np.ndarray]:
    """``zero_action`` with the ``a_battery`` key set to ``value``."""
    action = zero_action(action_spaces)
    action["a_battery"] = np.array([np.float32(value)], dtype=np.float32)
    return action


# ---------------------------------------------------------------- measurements
def renewable_surplus_kW(
    last_info: dict | None,
    renewable_names: Collection[str],
    battery_name: str | None,
) -> float:
    """Renewable production minus all non-battery consumption, from the
    previous step's power_breakdown {name: (production_kW, consumption_kW)}.
    Positive = surplus available for charging; negative = deficit."""
    breakdown = (last_info or {}).get("power_breakdown")
    if not breakdown:
        return 0.0
    production_kW = sum(
        prod for name, (prod, _cons) in breakdown.items() if name in renewable_names
    )
    consumption_kW = sum(
        cons for name, (_prod, cons) in breakdown.items() if name != battery_name
    )
    return production_kW - consumption_kW


def hour_of_day(step: int, control_step_s: int) -> float:
    """Hour-of-day derived from step count × control step; assumes the
    episode starts at midnight (standard config: 288 × 300 s = one day)."""
    return (step * control_step_s / SECONDS_PER_HOUR) % 24.0
