import numpy as np

from .base import RewardFunction
from adv_building_gym.utils.serializable import ComponentRegistry


# TODO VP 2026.01.14. : Add battery life saving reward

class MinimiseEnergyConsumptionReward(RewardFunction):
    """Energy consumption-based reward function.

    Penalises total energy consumption across all actions, but exempts
    *necessary* charging: when the battery or EV has not yet reached its
    target SoC, the positive (charging) portion of that device's action
    is excluded from the penalty.  This prevents the reward from
    conflicting with the battery/EV target rewards.
    """

    def __init__(self, weight: float, name: str = "E_consumption_reward") -> None:
        super().__init__(weight, name)

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        e_consumption: float = 0
        n_actions: int = 0

        for key, v in actions.items():
            # TODO VP 2026.03.23. : Really like this?
            if key == "hh_consumption_action":
                continue  # Non-controllable load — exempt from penalty

            n_actions += 1

            if key == "HP_action":
                # HP_action is 2D: [energy, mode], only use energy (index 0)
                e_consumption += float(np.atleast_1d(v)[0])

            elif key == "battery_action":
                action_val = float(np.atleast_1d(v)[0])
                battery_pct = float(states["battery_pct"][0])
                battery_target = float(states["battery_target_pct"][0])
                if action_val > 0 and battery_pct < battery_target:
                    # Necessary charging — exempt from penalty
                    continue
                e_consumption += action_val

            elif key == "lin_ev_charger_action":
                action_val = float(np.atleast_1d(v)[0])
                ev_connected = float(states["ev_connected"][0])
                ev_soc = float(states["ev_soc"][0])
                ev_target = float(states["ev_target_soc"][0])
                if action_val > 0 and ev_connected > 0.5 and ev_soc < ev_target:
                    # Necessary charging — exempt from penalty
                    continue
                e_consumption += action_val

            else:
                e_consumption += np.sum(v, axis=0)

        reward = -1.0 * e_consumption / n_actions if n_actions > 0 else 0.0

        return float(self.weight * reward), self.weight * self.max_reward


# Register MinimiseEnergyConsumption_Reward with the component registry
ComponentRegistry.register('reward', MinimiseEnergyConsumptionReward)
