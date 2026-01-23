
from dataclasses import dataclass, field
from typing import List, Optional

from adv_building_gym.envs.utils import BuildingProps

from adv_building_gym.devices.infrastructure import Infrastructure, HP, BatteryTremblay, SolarPanel, LinearEVCharger
from adv_building_gym.devices.statesources import (
    StateSource, BuildingHeatLoss, DesiredUserEnergyNeed, InsideTemperature,
    EnergyPriceDataSource, WeatherDataSource
)
from adv_building_gym.rewards import RewardFunction, TempReward, EconomicReward, MinimiseEnergyConsumption_Reward, UserEnergyNeedReward, OperatorEnergyControlReward


# TODO VP 2026.01.13. : How to learn more days during training? -- solve consecutive days from data sources

@dataclass
class Config:
    """
    Config serialisation -- handled by ConfigManager.
    Use ConfigManager.save(config, path) to save and ConfigManager.load(path) to load configurations
    """
    config_name: str = "test1"

    EPISODE_LENGTH: int = 288
    control_step: int = 300  # seconds (5 minutes)

    building_props: BuildingProps = field(default_factory=lambda:
        BuildingProps(mC=300, K=20)
    )

    # Initialisation is later, it depends on the building_props
    infras: Optional[List[Infrastructure]] = None

    # Note: statesources list is initialized in __post_init__ to use building_props
    statesources: Optional[List[StateSource]] = None

    # Note: rewards list is initialized in __post_init__ to use infras
    rewards: Optional[List[RewardFunction]] = None

    def __post_init__(self):
        """Initialize infras, statesources, and rewards using building_props if not provided."""
        if self.statesources is None:
            self.statesources = [
                # TODO VP 2026.01.23. : Statesources also need a passed control_step...
                EnergyPriceDataSource("E_price", ds_path="data/price_data_2025.csv"),
                WeatherDataSource("weather", ds_path="data/LLEC_outdoor_temperature_5min_data.csv"),
                InsideTemperature("desired_temp_in"),
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
                    K=self.building_props.K,
                    mC=self.building_props.mC,
                    cop_heat=3.0,  # Typical heating COP for heat pumps
                    cop_cool=2.5,  # Typical cooling COP for heat pumps
                    control_step=self.control_step
                ),
                BatteryTremblay("battery", control_step=self.control_step),  # 14 kWh, 48A defaults
                LinearEVCharger("ev_charger", Q_electric_max=7.0, max_cap_kWh=60.0, max_charging_kW=7.0, control_step=self.control_step),
                SolarPanel("solar", Q_electric_max=5.0, peak_power_kW=5.0, control_step=self.control_step),
            ]

        if self.rewards is None:
            self.rewards = [
                TempReward(1),
                EconomicReward(1),
                MinimiseEnergyConsumption_Reward(1),
                OperatorEnergyControlReward(self.infras, weight=1),
            ]

# default/config instance
config = Config()
