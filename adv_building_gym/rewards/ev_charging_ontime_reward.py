"""EV charging progress reward function."""

from typing import TYPE_CHECKING, ClassVar, Dict, Set

from .base import RewardFunction
from adv_building_gym.config.utils.serializable import ComponentRegistry

# Import only for type checkers to avoid circular imports at runtime
if TYPE_CHECKING:
    from adv_building_gym.devices.infrastructure.ev_charger import LinearEVCharger


class EVChargingOnTimeReward(RewardFunction):
    """Reward function for EV charging progress.

    Rewards the agent for making good charging progress relative to the
    time remaining to reach the target SoC. The reward is based on whether
    the target can be achieved given the remaining time and charging capacity.

    Reward calculation:
    - If EV not connected: reward = 1.0 (neutral, no penalty -- this objective is fulfilled when EV is not present)
    - If target SoC already achieved (soc >= target_soc): reward = 1
    - Otherwise: reward = max(0, 1 - (energy_needed / energy_achievable))

    Where:
    - energy_needed = (target_soc - current_soc) * max_cap_kWh
    - energy_achievable = max_charging_kW * charger_efficiency * remaining_hours
    """

    # Parameters derived from context (charger instance)
    _context_params: ClassVar[Set[str]] = {'ev_charger'}

    def __init__(self,
                 weight: float,
                 name: str = "ev_charging_reward",
                 ev_charger: "LinearEVCharger | None" = None) -> None:
        """Initialize EVChargingReward.

        Args:
            weight: Reward weight for multi-objective optimization
            name: Reward function identifier
            ev_charger: LinearEVCharger instance to extract parameters from
        """
        super().__init__(weight, name)

        if ev_charger is None:
            raise ValueError("ev_charger must be provided to EVChargingReward")

        self.max_charging_kW = ev_charger.max_charging_kW
        self.max_cap_kWh = ev_charger.max_cap_kWh
        self.charger_efficiency = ev_charger.charger_efficiency
        self.max_charge_time_hrs = ev_charger.max_charge_time_hrs

    def get_reward(self, _actions: Dict, states: Dict) -> float:
        """Calculate EV charging progress reward.

        Args:
            _actions: Dictionary of actions (unused)
            states: Dictionary of current environment states

        Returns:
            Weighted reward in range [0, weight]:
            - 0 if EV not connected or behind schedule
            - weight if target achieved or on track
        """
        ev_connected = states["ev_connected"][0]

        # No reward if EV not connected
        if ev_connected < 0.5:
            # Neutral reward when EV is not present, as this objective is fulfilled when EV is not present
            return self.weight * 1.0 

        current_soc = states["ev_soc"][0]
        target_soc = states["ev_target_soc"][0]

        # Max reward if target already achieved
        if current_soc >= target_soc:
            return self.weight * 1.0

        # Denormalize remaining time from [0, 1] to hours
        normalized_time = states["ev_charge_to_target_hrs_norm"][0]
        remaining_hrs = normalized_time * self.max_charge_time_hrs

        # Calculate energy needed to reach target (in kWh)
        energy_needed = (target_soc - current_soc) * self.max_cap_kWh

        # Calculate energy achievable in remaining time (in kWh)
        # energy = power * efficiency * time
        energy_achievable = self.max_charging_kW * self.charger_efficiency * remaining_hrs

        # Avoid division by zero
        if energy_achievable <= 0:
            # No time left, can't achieve target
            return 0.0

        # Calculate ratio of needed vs achievable energy
        ratio = energy_needed / energy_achievable

        # Reward: 1 when ratio=0 (target achieved), decreasing as ratio increases
        # Clipped to [0, 1] - no negative rewards
        reward = max(0.0, 1.0 - ratio)

        return self.weight * reward


# Register EVChargingReward with the component registry
ComponentRegistry.register('reward', EVChargingOnTimeReward)
