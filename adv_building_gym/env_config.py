

from dataclasses import dataclass, field
from typing import List

from adv_building_gym.envs.building_adv import BuildingProps

from .devices.infrastructure import Infrastructure, HP, Battery
from .devices.data_sources import DataSource, EnergyPriceDataSource, WeatherDataSource
from .rewards.rewards import RewardFunction, TempReward, EconomicReward, EnergyConsumptionReward



@dataclass
class Config:
    config_name: str = "test1"
    
    EPISODE_LENGTH: int = 288

    infras: List[Infrastructure] = field(default_factory=lambda: [
        HP("HP", 1000.0, 800.0, 1.0, 1.0),
        Battery("battery", 1000, 50, 100),
    ])

    datasources: List[DataSource] = field(default_factory=lambda: [
        EnergyPriceDataSource("E_price"),
        WeatherDataSource("weather"),
    ])

    rewards: List[RewardFunction] = field(default_factory=lambda: [
        TempReward(1),
        EconomicReward(1),
        EnergyConsumptionReward(1),
    ])

    building_props: BuildingProps = field(default_factory=lambda:
        BuildingProps(mC=300, K=20)
    )

# default/config instance
config = Config()
