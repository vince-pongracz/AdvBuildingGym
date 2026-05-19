"""Rewards module for building environment reward functions."""

from .base import RewardFunction
from .aggregator import RewardAggregator, SumRewardAggregator
from .action_smoothness_reward import ActionSmoothnessReward
from .action_smoothness_reward_v0 import ActionSmoothnessRewardV0
from .battery_target_reward import BatteryTargetReward
from .battery_target_reward_v0 import BatteryTargetRewardV0
from .battery_mgmt_reward import BatteryMgmtReward
from .battery_mgmt_reward_v0 import BatteryMgmtRewardV0
from .economic_reward import EconomicReward
from .economic_reward_v0 import EconomicRewardV0
from .long_term_economic_reward import LongTermEconomicReward
from .long_term_economic_reward_v0 import LongTermEconomicRewardV0
from .energy_consumption_reward import MinimiseEnergyConsumptionReward
from .energy_consumption_reward_v0 import MinimiseEnergyConsumptionRewardV0
from .ev_charging_ontime_reward import EVChargingOnTimeReward
from .ev_charging_ontime_reward_v0 import EVChargingOnTimeRewardV0
from .ev_charging_reward import EVChargingReward
from .ev_charging_reward_v0 import EVChargingRewardV0
from .operator_energy_control_reward import OperatorEnergyControlReward
from .operator_energy_control_reward_v0 import OperatorEnergyControlRewardV0
from .temp_reward import TempReward
from .temp_reward_v0 import TempRewardV0

__all__ = [
    "RewardFunction",
    "RewardAggregator",
    "SumRewardAggregator",
    "ActionSmoothnessReward",
    "ActionSmoothnessRewardV0",
    "BatteryTargetReward",
    "BatteryTargetRewardV0",
    "BatteryMgmtReward",
    "BatteryMgmtRewardV0",
    "EconomicReward",
    "EconomicRewardV0",
    "LongTermEconomicReward",
    "LongTermEconomicRewardV0",
    "EVChargingOnTimeReward",
    "EVChargingOnTimeRewardV0",
    "EVChargingReward",
    "EVChargingRewardV0",
    "MinimiseEnergyConsumptionReward",
    "MinimiseEnergyConsumptionRewardV0",
    "OperatorEnergyControlReward",
    "OperatorEnergyControlRewardV0",
    "TempReward",
    "TempRewardV0",
]
