
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

from adv_building_gym.envs.utils import BuildingProps

from adv_building_gym.devices.infrastructure import Infrastructure, HP, BatteryTremblay, SolarPanel, LinearEVCharger
from adv_building_gym.devices.statesources import (
    StateSource, BuildingHeatLoss, DesiredUserEnergyNeed, EVState, InsideTemperature,
    EnergyPriceDataSource, WeatherDataSource
)
from adv_building_gym.rewards import (
    RewardFunction, TempReward, EconomicReward, 
    EVChargingOnTimeReward, MinimiseEnergyConsumption_Reward, 
    UserEnergyNeedReward, OperatorEnergyControlReward
)


# TODO VP 2026.01.13. : How to learn more days during training? -- solve consecutive days from data sources
# TODO VP 2026.02.11. : Use shorter state source .csv-s, retrieve data row based on the control step -- if no such step given, keep the previous state value -- works for EV target state source

@dataclass
class Config:
    """
    Config serialisation -- handled by ConfigManager.
    Use ConfigManager.save(config, path) to save and ConfigManager.load(path) to load configurations

    IMPORTANT: Use the factory methods (create_infras, create_statesources, create_rewards)
    when creating environment instances to ensure each env gets independent component instances.
    Direct access to self.infras/statesources/rewards returns shared singletons and should
    only be used for inspection, not for passing to AdvBuildingGym in parallel environments.
    """
    config_name: str = "test1"

    seed: int = 42
    EPISODE_LENGTH: int = 288
    control_step: int = 300  # seconds (5 minutes)

    building_props: BuildingProps = field(default_factory=lambda:
        BuildingProps(mC=300, K=20)
    )

    # Cached singleton instances (for backward compatibility and inspection)
    # WARNING: Do not pass these to parallel environments - use factory methods instead
    infras: Optional[List[Infrastructure]] = None
    statesources: Optional[List[StateSource]] = None
    rewards: Optional[List[RewardFunction]] = None

    def create_statesources(self) -> List[StateSource]:
        """
        Factory method to create fresh StateSource instances.

        Each call returns NEW independent instances, safe for parallel environments.
        Components have their own iteration counter and state.

        Returns:
            List of newly created StateSource instances.
        """
        return [
            EnergyPriceDataSource("E_price", ds_path=str(_DATA_DIR / "price_data_2025.csv")),
            WeatherDataSource("weather", ds_path=str(_DATA_DIR / "LLEC_outdoor_temperature_5min_data.csv")),
            InsideTemperature("desired_temp_in"),
            DesiredUserEnergyNeed("user_energy_need"),
            BuildingHeatLoss(
                name="building_heat_loss",
                K=self.building_props.K,
                mC=self.building_props.mC,
                timestep=self.control_step
            ),
            EVState("ev_schedule", ds_path=str(_DATA_DIR / "ev_usage_profiles/ev_1.csv")),
        ]

    def create_infras(self) -> List[Infrastructure]:
        """
        Factory method to create fresh Infrastructure instances.

        Each call returns NEW independent instances, safe for parallel environments.
        Components have their own iteration counter and state.

        Returns:
            List of newly created Infrastructure instances.
        """
        return [
            HP(
                name="HP",
                Q_electric_max=5.0,  # kW (consistent with battery 19 kW, EV 7 kW, solar 5 kW)
                K=self.building_props.K,
                mC=self.building_props.mC,
                cop_heat=3.0,
                cop_cool=2.5,
                control_step=self.control_step
            ),
            BatteryTremblay("battery", control_step=self.control_step),
            LinearEVCharger(
                "ev_charger",
                Q_electric_max=7.0,
                max_cap_kWh=60.0,
                max_charging_kW=7.0,
                control_step=self.control_step
            ),
            SolarPanel(
                "solar",
                Q_electric_max=5.0,
                peak_power_kW=5.0,
                seed=self.seed,
                control_step=self.control_step
            ),
        ]

    def create_rewards(self, infras: List[Infrastructure]) -> List[RewardFunction]:
        """
        Factory method to create fresh RewardFunction instances.

        Each call returns NEW independent instances, safe for parallel environments.

        Args:
            infras: List of Infrastructure instances (from create_infras) to link
                    rewards that depend on infrastructure state (e.g., EV charger).

        Returns:
            List of newly created RewardFunction instances.
        """
        # TODO VP 2026.02.19. : Pass full infras list to EVChargingOnTimeReward and let it find the EV charger itself, instead of passing ev_charger separately
        ev_charger = next(
            (infra for infra in infras if isinstance(infra, LinearEVCharger)),
            None
        )
        return [
            TempReward(weight=1),
            EconomicReward(infras, weight=1),
            MinimiseEnergyConsumption_Reward(weight=1),
            OperatorEnergyControlReward(infras, weight=1),
            EVChargingOnTimeReward(weight=1, ev_charger=ev_charger),
        ]

    def __post_init__(self):
        """Initialize cached singleton instances for backward compatibility.

        WARNING: These cached instances should NOT be used for parallel environments.
        Use create_infras(), create_statesources(), create_rewards() instead.
        """
        if self.statesources is None:
            self.statesources = self.create_statesources()

        if self.infras is None:
            self.infras = self.create_infras()

        if self.rewards is None:
            self.rewards = self.create_rewards(self.infras)

# default/config instance
config = Config()
