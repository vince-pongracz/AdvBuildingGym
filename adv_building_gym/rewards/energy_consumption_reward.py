import logging

import numpy as np

from .base import RewardFunction
from adv_building_gym.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)

# TODO VP 2026.01.14. : Add battery life saving reward

class MinimiseEnergyConsumptionReward(RewardFunction):
    """Energy consumption-based reward function.

    Penalises total energy consumption using ``power_breakdown`` from
    the info dict (published by the environment from each infrastructure's
    ``get_electric_consumption``).  This decouples the reward from action
    semantics — only physical power matters.

    Exempts *necessary* charging: when the battery or EV has not yet
    reached its target SoC, that device's consumption is excluded from
    the penalty.  This prevents the reward from conflicting with the
    battery/EV target rewards.

    Normalises by ``max_consumption_kW`` (also from info) so the reward
    stays in [-1, 0].
    """

    def __init__(self, weight: float, name: str = "E_consumption_reward") -> None:
        super().__init__(weight, name)

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        max_step = self.weight * self.max_reward

        if info is None:
            logger.warning("MinimiseEnergyConsumptionReward: info dict is None, returning 0")
            return 0.0, max_step

        power_breakdown = info.get("power_breakdown")
        max_consumption_kW = info.get("max_consumption_kW")

        if power_breakdown is None or max_consumption_kW is None:
            logger.warning("MinimiseEnergyConsumptionReward: missing power data in info, returning 0")
            return 0.0, max_step

        e_consumption_kW: float = 0.0

        for name, power_kW in power_breakdown.items():
            # Non-controllable load — exempt from penalty
            # TODO VP 2026.03.23. : Really like this?
            if name == "hh_consumers":
                continue

            # Exempt necessary battery charging (below target SoC)
            if name == "battery" and power_kW > 0:
                battery_pct = float(states["battery_pct"][0])
                battery_target = float(states["battery_target_pct"][0])
                if battery_pct < battery_target:
                    continue

            # Exempt necessary EV charging (connected and below target SoC)
            if name == "ev_charger" and power_kW > 0:
                ev_connected = float(states["ev_connected"][0])
                ev_soc = float(states["ev_soc"][0])
                ev_target = float(states["ev_target_soc"][0])
                if ev_connected > 0.5 and ev_soc < ev_target:
                    continue

            e_consumption_kW += power_kW

        if max_consumption_kW <= 0:
            return 0.0, max_step

        # Normalise to [-1, 0]: full consumption = -1, zero consumption = 0
        reward = float(np.clip(-e_consumption_kW / max_consumption_kW, -1.0, 0.0))

        return float(self.weight * reward), max_step


# Register MinimiseEnergyConsumption_Reward with the component registry
ComponentRegistry.register('reward', MinimiseEnergyConsumptionReward)
