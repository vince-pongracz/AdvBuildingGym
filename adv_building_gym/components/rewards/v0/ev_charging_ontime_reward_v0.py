"""EV charging progress reward function (V0)."""

from typing import Dict

import numpy as np

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry


class EVChargingOnTimeRewardV0(RewardFunction):
    """On-time EV charging reward (V0).

    Scores how well charging keeps pace with the time left to reach the target
    SoC. ``get_reward`` returns ``self.weight`` times a raw reward in
    ``[harsh_penalty, 1]`` (so ``[-1, 1]`` with the default
    ``harsh_penalty = -1.0``):

    - EV not connected (``s_evc_connected < 0.5``), or ``info`` is ``None``
      → ``0.0`` (returned unscaled by ``weight``).
    - Target already met (``s_evc_soc >= s_evc_target_soc``) → ``weight * 1.0``.
    - No charging headroom left (``energy_achievable <= 0``)
      → ``weight * harsh_penalty``.
    - Otherwise → ``weight * max(0, 1 - energy_needed / energy_achievable)``,
      but only while actively charging (``a_lin_ev_charger > 0``); when not
      charging the reward is ``weight * 0.0``.

    where ``energy_needed = (target_soc - soc) * ctxt_evc_max_cap_kWh`` and
    ``energy_achievable = ctxt_evc_max_charging_kW *
    ctxt_evc_charger_efficiency * remaining_hrs``, with
    ``remaining_hrs = s_evc_charge_to_target_hrs_norm *
    ctxt_evc_max_charge_time_hrs``.
    """

    def __init__(self,
                weight: float,
                name: str = "ev_charging_ontime_reward_v0",
                harsh_penalty: float = -1.0
                ) -> None:
        super().__init__(weight, name)
        self.harsh_penalty = harsh_penalty

    def get_reward(self, actions: Dict, state: Dict, next_state: Dict, info: dict) -> float:
        ev_connected = next_state["s_evc_connected"][0]

        if ev_connected < 0.5:
            return 0.0

        current_soc = next_state["s_evc_soc"][0]
        target_soc = next_state["s_evc_target_soc"][0]

        if current_soc >= target_soc:
            return self.weight * 1.0

        max_charging_kW = float(next_state["ctxt_evc_max_charging_kW"][0])
        max_cap_kWh = float(next_state["ctxt_evc_max_cap_kWh"][0])
        charger_efficiency = float(next_state["ctxt_evc_charger_efficiency"][0])
        max_charge_time_hrs = float(next_state["ctxt_evc_max_charge_time_hrs"][0])

        normalized_time = next_state["s_evc_charge_to_target_hrs_norm"][0]
        remaining_hrs = normalized_time * max_charge_time_hrs

        energy_needed = (target_soc - current_soc) * max_cap_kWh
        energy_achievable = max_charging_kW * charger_efficiency * remaining_hrs

        if energy_achievable <= 0:
            return self.weight * self.harsh_penalty

        ratio = energy_needed / energy_achievable
        reward = max(0.0, 1.0 - ratio)

        ev_action = float(np.atleast_1d(actions.get("a_lin_ev_charger", [0]))[0])
        if ev_action > 0.0:
            return self.weight * reward
        else:
            return self.weight * 0.0


ComponentRegistry.register('reward', EVChargingOnTimeRewardV0)
