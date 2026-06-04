"""EV charging progress reward function."""

import logging
from typing import Dict

import numpy as np

from ..base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class EVChargingOnTimeReward(RewardFunction):
    """Reward function for EV charging progress.

    Rewards the agent for making good charging progress relative to the
    time remaining to reach the target SoC. The reward is based on whether
    the target can be achieved given the remaining time and charging capacity.

    EV charger parameters (max_charging_kW, max_cap_kWh, charger_efficiency,
    max_charge_time_hrs) are read from the ``info`` dict at each step,
    published by the LinearEVCharger infrastructure.

    Reward calculation:
    - If EV not connected: reward = 0 (max reward also 0)
    - If target SoC already achieved (soc >= target_soc): reward = 1
    - Otherwise: reward = max(0, 1 - (energy_needed / energy_achievable))

    Where:
    - energy_needed = (target_soc - current_soc) * max_cap_kWh
    - energy_achievable = max_charging_kW * charger_efficiency * remaining_hours
    """

    def __init__(self,
                weight: float,
                name: str = "ev_charging_ontime_reward",
                harsh_penalty: float = -5.0
                ) -> None:
        """Initialize EVChargingOnTimeReward.

        Args:
            weight: Reward weight for multi-objective optimization
            name: Reward function identifier
            harsh_penalty: Penalty applied when no time remains and target
                is not met.
        """
        super().__init__(weight, name)
        self.harsh_penalty = harsh_penalty
        
    # TODO noprio VP 2026.03.25. : Check whether pydispatcher could be used instead of info objects...

    def get_reward(self, actions: Dict, states: Dict, info: dict | None = None) -> tuple[float, float]:
        """Calculate EV charging progress reward.

        Args:
            actions: Dictionary of actions taken by the agent.
            states: Dictionary of current environment states.
            info: Shared inter-component dict containing EV charger params
                (ctxt_ev_max_charging_kW, ctxt_ev_max_cap_kWh,
                ctxt_ev_charger_efficiency, ctxt_ev_max_charge_time_hrs).

        Returns:
            Tuple of (reward, max_reward_for_this_step):
            - (0, 0) if EV not connected
            - (weight, weight) if target achieved or on track
        """
        ev_connected = states["s_ev_connected"][0]

        if ev_connected < 0.5:
            return 0.0, 0.0

        if info is None:
            logger.warning("EVChargingOnTimeReward: info dict is None, returning 0")
            return 0.0, 0.0

        max_step = self.weight * self.max_reward_in_step
        current_soc = states["s_ev_soc"][0]
        target_soc = states["s_ev_target_soc"][0]

        # Max reward if target already achieved
        if current_soc >= target_soc:
            return self.weight * 1.0, max_step

        # Read EV charger parameters from info dict (published by LinearEVCharger)
        max_charging_kW = info["ctxt_ev_max_charging_kW"]
        max_cap_kWh = info["ctxt_ev_max_cap_kWh"] or 0.0  # avoid None
        charger_efficiency = info["ctxt_ev_charger_efficiency"]
        max_charge_time_hrs = info["ctxt_ev_max_charge_time_hrs"]

        # Denormalize remaining time from [0, 1] to hours
        normalized_time = states["s_ev_charge_to_target_hrs_norm"][0]
        remaining_hrs = normalized_time * max_charge_time_hrs

        # Calculate energy needed to reach target (in kWh)
        energy_needed = (target_soc - current_soc) * max_cap_kWh

        # Calculate energy achievable in remaining time (in kWh)
        # energy = power * efficiency * time
        energy_achievable = max_charging_kW * charger_efficiency * remaining_hrs

        # Avoid division by zero
        if energy_achievable <= 0:
            # No time left, target not met — apply harsh penalty
            return self.weight * self.harsh_penalty, max_step

        # Calculate ratio of needed vs achievable energy
        ratio = energy_needed / energy_achievable

        # Reward: 1 when ratio=0 (target achieved), decreasing as ratio increases
        # Clipped to [0, 1] - no negative rewards
        reward = max(0.0, 1.0 - ratio)

        # Only give reward if the agent is actually charging at least a tiny bit.
        # Without this gate the reward rewards inaction (having time left) instead
        # of rewarding charging progress.
        ev_action = float(np.atleast_1d(actions.get("a_lin_ev_charger", [0]))[0])
        if ev_action > 0.0:
            return self.weight * reward, max_step
        else:
            return self.weight * 0.0, max_step


# Register EVChargingReward with the component registry
ComponentRegistry.register('reward', EVChargingOnTimeReward)
