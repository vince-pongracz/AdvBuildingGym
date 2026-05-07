"""Battery state-of-charge dead-zone reward function."""

from .base import RewardFunction
from adv_building_gym.utils.serializable import ComponentRegistry

# TODO VP 2026.05.06.: Rewrite docsstring
class BatteryTargetReward(RewardFunction):
    """Guardrail reward keeping battery SoC inside a safe operating band.

    A small positive reward inside ``[min_pct, max_pct]`` acts as a mild
    incentive to stay in the healthy operating range, while a quadratic
    penalty outside the band scales up to ``max_penalty`` at the hard SoC
    limits (0 and 1).  The in-band reward is intentionally small so
    economic arbitrage can freely shape the within-band trajectory; the
    out-of-band penalty is intentionally large so depletion and overflow
    are strongly avoided.

    The band is owned by this reward (constructor arguments) — the
    battery component is unaware of it.  Reads ``s_battery_pct`` from the
    observation dict.

    Shape (continuous across the band edges):
        SoC ∈ [min_pct, max_pct]      → reward = in_band_reward
        SoC < min_pct                 → reward = in_band_reward
                                                 + (max_penalty - in_band_reward)
                                                 · ((min_pct - SoC) / min_pct)²
        SoC > max_pct                 → reward = in_band_reward
                                                 + (max_penalty - in_band_reward)
                                                 · ((SoC - max_pct) / (1 - max_pct))²

    With the defaults (in_band_reward=1.0, max_penalty=-3.0), the reward
    lives in [-3.0, 1.0].  ``self.max_reward`` is set to ``in_band_reward``
    so ``reward_rate`` logging reflects the true best-case contribution.
    """

    def __init__(self,
                weight: float,
                min_pct: float,
                max_pct: float,
                name: str = "battery_target_reward") -> None:
        """
        Args:
            weight: Reward weight for multi-objective optimisation.
            name: Reward function identifier.
            min_pct: Lower edge of the healthy SoC band (fractional SoC).
            max_pct: Upper edge of the healthy SoC band (fractional SoC).
            in_band_reward: Small positive reward granted on every step the
                SoC lies inside ``[min_pct, max_pct]``.
        """
        super().__init__(weight, name)
        if not 0.0 <= min_pct < max_pct <= 1.0:
            raise ValueError("require 0 <= min_pct < max_pct <= 1")

        self.min_pct = float(min_pct)
        self.max_pct = float(max_pct)

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        soc = float(states["s_battery_pct"][0])

        if self.min_pct <= soc <= self.max_pct:
            return 0.0, 0.0
        else:
            # if soc < self.min_pct:
            #     # Width of the lower unsafe interval is min_pct itself
            #     # (from SoC=0 up to the band edge).  Guard against the
            #     # degenerate min_pct=0 case where the lower skirt vanishes.
            #     breach = (self.min_pct - soc) / self.min_pct if self.min_pct > 0.0 else 0.0
            # else:
            #     upper_width = 1.0 - self.max_pct
            #     breach = (soc - self.max_pct) / upper_width if upper_width > 0.0 else 0.0
            # # Blend from in_band_reward at the band edge to max_penalty at
            # # the hard SoC limit, so the reward is continuous at min_pct /
            # # max_pct instead of stepping by ``in_band_reward``.
            # reward = self.max_penalty * (breach ** 2)
            reward = -1.0

        return self.weight * reward, self.weight * self.max_reward


ComponentRegistry.register('reward', BatteryTargetReward)
