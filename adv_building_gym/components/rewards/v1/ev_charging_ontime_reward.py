"""EV charging progress reward function."""

from typing import Dict

import numpy as np

from ..base import RewardFunction
from ..ev_signals import is_ev_connected, session_target_soc
from adv_building_gym.components.registry import ComponentRegistry


class EVChargingOnTimeReward(RewardFunction):
    """Reward for EV charging progress vs time remaining to target SoC.

    Charger params come from two channels, both written by LinearEVCharger in the same
    ``update_state``: ``max_charging_kW`` / ``max_charge_time_hrs`` are observation keys,
    ``max_cap_kWh`` / ``charger_efficiency`` / the session target are on ``info`` (the
    policy already sees the same numbers via EVState's ``ctxt_ev_schedule_*`` keys).

    Not connected → 0 (max 0); soc ≥ target → 1; else max(0, 1 - energy_needed/energy_achievable),
    with energy_needed = (target - soc)*max_cap_kWh, energy_achievable = max_charging_kW*efficiency*remaining_hrs.
    """

    def __init__(self,
                weight: float,
                name: str = "ev_charging_ontime_reward",
                harsh_penalty: float = -5.0
                ) -> None:
        """harsh_penalty: applied when no time remains and the target is unmet."""
        super().__init__(weight, name)
        self.harsh_penalty = harsh_penalty
        
    # TODO noprio VP 2026.03.25. : Check whether pydispatcher could be used instead of info objects...

    def get_reward(self, actions: Dict, state: Dict, next_state: Dict, info: dict) -> float:
        """EV charging progress reward. EV signals (SoC, target, remaining time) from
        ``next_state``; charger params from ``info``. Returns 0 if unplugged."""
        if not is_ev_connected(next_state):
            return 0.0

        current_soc = next_state["s_evc_soc"][0]
        target_soc = session_target_soc(info)

        # Max reward if target already achieved
        if current_soc >= target_soc:
            return self.weight * 1.0

        # Charger params: the rate limit stays an observation, the EV-spec values come off
        # the info channel (both written by LinearEVCharger in the same update_state).
        max_charging_kW = float(next_state["ctxt_evc_max_charging_kW"][0])
        max_cap_kWh = float(info.get("evc_max_cap_kWh", 0.0))
        charger_efficiency = float(info.get("evc_charger_efficiency", 0.0))
        max_charge_time_hrs = float(next_state["ctxt_evc_max_charge_time_hrs"][0])

        # remaining time [0,1] → hours
        normalized_time = next_state["s_evc_charge_to_target_hrs_norm"][0]
        remaining_hrs = normalized_time * max_charge_time_hrs

        # energy needed (kWh)
        energy_needed = (target_soc - current_soc) * max_cap_kWh

        # energy achievable = power * efficiency * time (kWh)
        energy_achievable = max_charging_kW * charger_efficiency * remaining_hrs

        # Avoid division by zero
        if energy_achievable <= 0:
            # no time left, target unmet → harsh penalty
            return self.weight * self.harsh_penalty

        # needed/achievable ratio; reward 1 at ratio=0, clipped to [0, 1]
        ratio = energy_needed / energy_achievable
        reward = max(0.0, 1.0 - ratio)

        # gate on actually charging — without it inaction (time left) would be rewarded
        ev_action = float(np.atleast_1d(actions.get("a_lin_ev_charger", [0]))[0])
        if ev_action > 0.0:
            return self.weight * reward
        else:
            return self.weight * 0.0


# register with ComponentRegistry
ComponentRegistry.register('reward', EVChargingOnTimeReward)
