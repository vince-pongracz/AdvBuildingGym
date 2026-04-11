"""Battery state-of-charge dead-zone reward function."""

from .base import RewardFunction
from adv_building_gym.utils.serializable import ComponentRegistry


class BatteryTargetReward(RewardFunction):
    """Guardrail reward keeping battery SoC inside a safe operating band.

    A small positive reward inside ``[safe_low, safe_high]`` acts as a mild
    incentive to stay in the healthy operating range, while a quadratic
    penalty outside the band scales up to ``max_penalty`` at the hard SoC
    limits (0 and 1).  The in-band reward is intentionally small so
    economic arbitrage can freely shape the within-band trajectory; the
    out-of-band penalty is intentionally large so depletion and overflow
    are strongly avoided.

    Shape (continuous across the band edges):
        SoC ∈ [safe_low, safe_high]      → reward = in_band_reward
        SoC < safe_low                   → reward = in_band_reward
                                                    + (max_penalty - in_band_reward)
                                                    · ((safe_low - SoC) / safe_low)²
        SoC > safe_high                  → reward = in_band_reward
                                                    + (max_penalty - in_band_reward)
                                                    · ((SoC - safe_high) / (1 - safe_high))²

    With the defaults (in_band_reward=0.3, max_penalty=-3.0), the reward
    lives in [-3.0, 0.3].  ``self.max_reward`` is set to ``in_band_reward``
    so ``reward_rate`` logging reflects the true best-case contribution.
    """

    def __init__(self,
                weight: float,
                name: str = "battery_target_reward",
                safe_low: float = 0.15,
                safe_high: float = 0.85,
                in_band_reward: float = 0.3,
                max_penalty: float = -3.0) -> None:
        """
        Args:
            weight: Reward weight for multi-objective optimisation.
            name: Reward function identifier.
            safe_low: Lower bound of the neutral band (fractional SoC).
            safe_high: Upper bound of the neutral band (fractional SoC).
            in_band_reward: Small positive reward granted on every step the
                SoC lies inside ``[safe_low, safe_high]``.
            max_penalty: Most negative reward reached at the hard SoC
                limits (SoC = 0 or SoC = 1).  Must be ≤ 0.
        """
        super().__init__(weight, name)
        if not 0.0 <= safe_low < safe_high <= 1.0:
            raise ValueError("require 0 <= safe_low < safe_high <= 1")
        if in_band_reward < 0.0:
            raise ValueError("in_band_reward must be >= 0")
        if max_penalty > 0.0:
            raise ValueError("max_penalty must be <= 0")
        self.safe_low = float(safe_low)
        self.safe_high = float(safe_high)
        self.in_band_reward = float(in_band_reward)
        self.max_penalty = float(max_penalty)
        self.max_reward = float(in_band_reward)

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        soc = float(states["battery_pct"][0])

        if self.safe_low <= soc <= self.safe_high:
            reward = self.in_band_reward
        else:
            if soc < self.safe_low:
                breach = (self.safe_low - soc) / self.safe_low
            else:
                breach = (soc - self.safe_high) / (1.0 - self.safe_high)
            # Blend from in_band_reward at the band edge to max_penalty at
            # the hard SoC limit, so the reward is continuous at safe_low /
            # safe_high instead of stepping by ``in_band_reward``.
            reward = self.in_band_reward + (self.max_penalty - self.in_band_reward) * (breach ** 2)

        if reward < self.max_penalty:
            reward = self.max_penalty

        return self.weight * reward, self.weight * self.max_reward


ComponentRegistry.register('reward', BatteryTargetReward)
