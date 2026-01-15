
from dataclasses import dataclass, field
from typing import List, Optional

from adv_building_gym.envs.utils import BuildingProps

from .devices.infrastructure import Infrastructure, HP, Battery
from .devices.datasources import (
    DataSource, BuildingHeatLoss, DesiredUserEnergyNeed,
    EnergyPriceDataSource, WeatherDataSource
)
from .rewards import RewardFunction, TempReward, EconomicReward, MinimiseEnergyConsumption_Reward, UserEnergyNeedReward

# TODO VP 2026.01.07. : Create serialisation or dump for config, dump and load back from specific file (json, yaml, etc.)
# Do not serialise such params of datasources or infrastructures, which are e.g coming from the BuildingProps during initialisation

@dataclass
class Config:
    config_name: str = "test1"

    EPISODE_LENGTH: int = 288
    control_step: int = 300  # seconds (5 minutes)

    building_props: BuildingProps = field(default_factory=lambda:
        BuildingProps(mC=300, K=20)
    )

    # Initialisation is later, it depends on the building_props
    infras: Optional[List[Infrastructure]] = None

    # Note: datasources list is initialized in __post_init__ to use building_props
    datasources: Optional[List[DataSource]] = None

    rewards: List[RewardFunction] = field(default_factory=lambda: [
        TempReward(1),
        EconomicReward(1),
        MinimiseEnergyConsumption_Reward(1),
    ])

    def __post_init__(self):
        """Initialize infras and datasources using building_props if not provided."""
        if self.datasources is None:
            self.datasources = [
                EnergyPriceDataSource("E_price", ds_path="data/price_data_2025.csv"),
                WeatherDataSource("weather", ds_path="data/LLEC_outdoor_temperature_5min_data.csv"),
                DesiredUserEnergyNeed("user_energy_need"),
                BuildingHeatLoss(
                    name="building_heat_loss",
                    K=self.building_props.K,
                    mC=self.building_props.mC,
                    timestep=self.control_step
                ),
            ]

        if self.infras is None:
            self.infras = [
                HP(
                    name="HP",
                    Q_electric_max=1000.0,
                    Q_hp_max=800.0,
                    K=self.building_props.K,
                    mC=self.building_props.mC,
                    cop_heat=3.0,  # Typical heating COP for heat pumps
                    cop_cool=2.5   # Typical cooling COP for heat pumps
                ),
                Battery("battery", 1000, 50, 100),
            ]

# default/config instance
config = Config()
