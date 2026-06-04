"""EV charging progress reward function (V0)."""

import logging
from typing import Dict

import numpy as np

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class EVChargingOnTimeRewardV0(RewardFunction):
    """On-time EV charging reward (V0), bounded per step to ``[-1, 1]``.

    Logic identical to :class:`EVChargingOnTimeReward` except
    ``harsh_penalty`` default is ``-1.0`` (was ``-5.0``):

    - EV not connected → ``(0, 0)``
    - target met (``SoC >= target``) → ``+1.0``
    - no time left & target unmet → ``harsh_penalty``
    - else → ``max(0, 1 - energy_needed / energy_achievable)``, zeroed when
      not actively charging.
    """

    def __init__(self,
                weight: float,
                name: str = "ev_charging_ontime_reward_v0",
                harsh_penalty: float = -1.0
                ) -> None:
        super().__init__(weight, name)
        self.harsh_penalty = harsh_penalty

    def get_reward(self, actions: Dict, states: Dict, info: dict | None = None) -> tuple[float, float]:
        ev_connected = states["s_ev_connected"][0]

        if ev_connected < 0.5:
            return 0.0, 0.0

        if info is None:
            logger.warning("EVChargingOnTimeRewardV0: info dict is None, returning 0")
            return 0.0, 0.0

        max_step = self.weight * self.max_reward_in_step
        current_soc = states["s_ev_soc"][0]
        target_soc = states["s_ev_target_soc"][0]

        if current_soc >= target_soc:
            return self.weight * 1.0, max_step

        max_charging_kW = info["ctxt_ev_max_charging_kW"]
        max_cap_kWh = info["ctxt_ev_max_cap_kWh"] or 0.0
        charger_efficiency = info["ctxt_ev_charger_efficiency"]
        max_charge_time_hrs = info["ctxt_ev_max_charge_time_hrs"]

        normalized_time = states["s_ev_charge_to_target_hrs_norm"][0]
        remaining_hrs = normalized_time * max_charge_time_hrs

        energy_needed = (target_soc - current_soc) * max_cap_kWh
        energy_achievable = max_charging_kW * charger_efficiency * remaining_hrs

        if energy_achievable <= 0:
            return self.weight * self.harsh_penalty, max_step

        ratio = energy_needed / energy_achievable
        reward = max(0.0, 1.0 - ratio)

        ev_action = float(np.atleast_1d(actions.get("a_lin_ev_charger", [0]))[0])
        if ev_action > 0.0:
            return self.weight * reward, max_step
        else:
            return self.weight * 0.0, max_step


ComponentRegistry.register('reward', EVChargingOnTimeRewardV0)
