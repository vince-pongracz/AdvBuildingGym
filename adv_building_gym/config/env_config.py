
from dataclasses import dataclass, field
from typing import List, Optional

# TODO VP 2026.02.20. : Simplyfy env config somehow, too much code here, too little declarative stuff...

from adv_building_gym.config.data_combinator import DataCombinator
from adv_building_gym.envs.utils import BuildingProps

from adv_building_gym.devices.infrastructure import (
    Infrastructure, HP, BatteryTremblay, 
    SolarPanel, LinearEVCharger
)

from adv_building_gym.devices.statesources import (
    StateSource, BuildingHeatLoss, DesiredUserEnergyNeed, EVState, InsideTemperature,
    EnergyPriceDataSource, WeatherDataSource
)

from adv_building_gym.rewards import (
    RewardFunction, TempReward, EconomicReward, 
    EVChargingOnTimeReward, MinimiseEnergyConsumption_Reward, 
    UserEnergyNeedReward, OperatorEnergyControlReward
)


@dataclass
class Config:
    """
    Config serialisation -- by ConfigManager.
    - Save config: ConfigManager.save(config, path)
    - Load config: ConfigManager.load(path)

    **IMPORTANT**: Use the factory methods (create_infras, create_statesources, create_rewards)
    when creating env instances to ensure each env gets independent component instances.
    Direct access to self.infras/statesources/rewards returns shared singletons and should
    only be used for inspection, not for passing to AdvBuildingGym in parallel environments.
    """
    config_name: str = "test1"

    seed: int = 42

    EPISODE_LENGTH: int = 288 # a day
    CONTROL_STEP: int = 300  # seconds (5 minutes)

    building_props: BuildingProps = field(default_factory=lambda:
        BuildingProps(mC=300, K=20)
    )

    # How to schedule backbone data sources along the training run.
    # Manage that different configurations are seen during the training
    # --> Help generalisation
    # NOTE VP 2026.02.23. : Default data combinator -- just an example
    data_combinator: DataCombinator = field(default_factory=lambda: DataCombinator(
        scenarios=[
            {
                "weather": "data/test1/LLEC_outdoor_temperature_5min_data.csv",
                "E_price": "data/test1/price_data_2025_1.csv",
            },
        ],
        variable={
            "ev_schedule": [
                "data/ev_usage_profiles/ev_0.csv",
                "data/ev_usage_profiles/ev_1.csv",
                "data/ev_usage_profiles/ev_2.csv",
                "data/ev_usage_profiles/ev_3.csv",
                "data/ev_usage_profiles/ev_4.csv",
                "data/ev_usage_profiles/ev_5.csv",
            ],
        },
        swap_every_n_episodes=10,
        mode="cycle",
    ))

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
            EnergyPriceDataSource("E_price"),
            WeatherDataSource("weather"),
            InsideTemperature("desired_temp_in"),
            DesiredUserEnergyNeed("user_energy_need"),
            BuildingHeatLoss(
                name="building_heat_loss",
                K=self.building_props.K,
                mC=self.building_props.mC,
                timestep=self.CONTROL_STEP
            ),
            EVState("ev_schedule"),
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
                control_step=self.CONTROL_STEP
            ),
            BatteryTremblay("battery", control_step=self.CONTROL_STEP),
            LinearEVCharger(
                "ev_charger",
                Q_electric_max=7.0,
                max_cap_kWh=60.0,
                max_charging_kW=7.0,
                control_step=self.CONTROL_STEP
            ),
            SolarPanel(
                "solar",
                Q_electric_max=5.0,
                peak_power_kW=5.0,
                seed=self.seed,
                control_step=self.CONTROL_STEP
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
        return [
            TempReward(weight=1),
            EconomicReward(infras, weight=1),
            MinimiseEnergyConsumption_Reward(weight=1),
            OperatorEnergyControlReward(infras, weight=1),
            EVChargingOnTimeReward(infrastructures=infras, weight=1),
        ]

    def __post_init__(self):
        """Lightweight post-init — does NOT eagerly call factory methods.

        Singleton fields (infras, statesources, rewards) are left as None to
        avoid unnecessary CSV parsing in every Ray worker subprocess that
        imports this module.  Call init_singletons() explicitly in the main
        process where those fields are actually needed.
        """

    def init_singletons(self) -> None:
        """Initialise the cached singleton component instances.

        Call this once in the main process after creating / loading a Config,
        before accessing self.infras / self.statesources / self.rewards.
        Not needed in Ray worker subprocesses — they call the factory methods
        (create_infras, create_statesources, create_rewards) directly via
        adv_building_env_creator.

        WARNING: Do not pass these singleton instances to parallel environments
        — use the factory methods instead.
        """
        if self.statesources is None:
            self.statesources = self.create_statesources()

        if self.infras is None:
            self.infras = self.create_infras()

        if self.rewards is None:
            self.rewards = self.create_rewards(self.infras)

# default/config instance
config = Config()
