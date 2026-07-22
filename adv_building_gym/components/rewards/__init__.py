"""Rewards module for building environment reward functions."""

from .base import RewardFunction
from .aggregator import RewardAggregator, SumRewardAggregator
from .v1.action_smoothness_reward import ActionSmoothnessReward
from .v0.action_smoothness_reward_v0 import ActionSmoothnessRewardV0
from .v1.battery_target_reward import BatteryTargetReward
from .v0.battery_target_reward_v0 import BatteryTargetRewardV0
from .v1.bes_regulator_reward import BESRegulatorReward
from .v1.battery_mgmt_reward import BatteryMgmtReward
from .v0.battery_mgmt_reward_v0 import BatteryMgmtRewardV0
from .v0.battery_range_mgmt_reward_v0 import BatteryRangeMgmtRewardV0
from .v1.economic_reward import EconomicReward
from .v0.economic_reward_v0 import EconomicRewardV0
from .v1.long_term_economic_reward import LongTermEconomicReward
from .v0.long_term_economic_reward_v0 import LongTermEconomicRewardV0
from .v1.energy_consumption_reward import MinimiseEnergyConsumptionReward
from .v0.energy_consumption_reward_v0 import MinimiseEnergyConsumptionRewardV0
from .v1.ev_charging_ontime_reward import EVChargingOnTimeReward
from .v0.ev_charging_ontime_reward_v0 import EVChargingOnTimeRewardV0
from .v1.ev_charging_reward import EVChargingReward
from .v0.ev_charging_reward_v0 import EVChargingRewardV0
from .v1.ev_charging_session_reward import EVChargingSessionReward
from .v1.ev_regulator_reward import EVRegulatorReward
from .v1.operator_energy_control_reward import OperatorEnergyControlReward
from .v0.operator_energy_control_reward_v0 import OperatorEnergyControlRewardV0
from .v1.temp_reward import TempReward
from .v0.temp_reward_v0 import TempRewardV0

__all__ = [
    "RewardFunction",
    "RewardAggregator",
    "SumRewardAggregator",
    "ActionSmoothnessReward",
    "ActionSmoothnessRewardV0",
    "BatteryTargetReward",
    "BatteryTargetRewardV0",
    "BESRegulatorReward",
    "BatteryMgmtReward",
    "BatteryMgmtRewardV0",
    "BatteryRangeMgmtRewardV0",
    "EconomicReward",
    "EconomicRewardV0",
    "LongTermEconomicReward",
    "LongTermEconomicRewardV0",
    "EVChargingOnTimeReward",
    "EVChargingOnTimeRewardV0",
    "EVChargingReward",
    "EVChargingRewardV0",
    "EVChargingSessionReward",
    "EVRegulatorReward",
    "MinimiseEnergyConsumptionReward",
    "MinimiseEnergyConsumptionRewardV0",
    "OperatorEnergyControlReward",
    "OperatorEnergyControlRewardV0",
    "TempReward",
    "TempRewardV0",
]
