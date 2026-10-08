import logging
import math

from .economic_reward_v0 import EconomicRewardV0
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class EconomicSellFactorRewardV0(EconomicRewardV0):
    """``EconomicRewardV0`` with a discounted sell (feed-in) price.

    Identical per-step formula, except that on a net-EXPORT step (``net_power_kW`` > 0)
    the price signal is multiplied by ``sell_price_factor``; net-import steps keep the
    full price. The factor scales the signed price, so an export at a negative price is
    penalised by ``sell_price_factor · |price|``.

    The factor is published on the info channel as ``info["sell_price_factor"]`` at every
    reset, so the env's ``PriceTracker`` bills exports at the same discounted price and
    ``cum_price_EUR`` follows what this reward sees.
    """

    def __init__(self, weight: float, reference_power_kW: float = 15.0,
                sell_price_factor: float = 0.7,
                name: str = "economic_sell_factor_reward_v0") -> None:
        super().__init__(weight, reference_power_kW=reference_power_kW, name=name)
        if not math.isfinite(sell_price_factor) or sell_price_factor < 0:
            raise ValueError("sell_price_factor must be finite and non-negative.")
        self.sell_price_factor = float(sell_price_factor)

    def on_reset(self, states, info: dict) -> None:
        super().on_reset(states, info)
        # env clears the info channel on reset, so republish each episode for the PriceTracker
        info["sell_price_factor"] = self.sell_price_factor

    def _tariff_factor(self, net_power_kW: float) -> float:
        # Canonical: net > 0 means export (selling), net < 0 means consumption (buying).
        return self.sell_price_factor if net_power_kW > 0 else 1.0


ComponentRegistry.register('reward', EconomicSellFactorRewardV0)
