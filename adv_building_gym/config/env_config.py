
import logging
from dataclasses import dataclass, field
from typing import List, Optional

# TODO VP 2026.02.20. : Simplyfy env config somehow, too much code here, too little declarative stuff...
from adv_building_gym.envs.utils import BuildingProps
from adv_building_gym.config.utils.loggable_config import LoggableConfig

logger = logging.getLogger(__name__)

from adv_building_gym.devices.infrastructure import (
    Infrastructure, HP, BatteryTremblay,
    SolarPanel, WindTurbine, LinearEVCharger, HouseholdEnergyConsumers
)

from adv_building_gym.devices.statesources import (
    StateSource, BuildingHeatLoss, DesiredUserEnergyNeed, EVState, InsideTemperature,
    EnergyPriceDataSource, WeatherDataSource
)

from adv_building_gym.config.reward_config import RewardConfig


@dataclass
class EnvConfig(LoggableConfig):
    """Environment topology configuration — state sources, infrastructure, and physics.

    Config serialisation -- by ConfigManager.
    - Save config: ConfigManager.save(config, path)
    - Load config: ConfigManager.load(path)

    **IMPORTANT**: Use the factory methods (create_infras, create_statesources)
    when creating env instances to ensure each env gets independent component instances.
    Direct access to self.infras/statesources returns shared singletons and should
    only be used for inspection, not for passing to AdvBuildingGym in parallel environments.

    Reward composition is managed by ``reward_config`` (RewardConfig).
    """
    env_config_name: str = "env_test1_small"

    EPISODE_LENGTH: int = 288 # a day
    CONTROL_STEP: int = 300  # seconds (5 minutes)
    ACTION_HISTORY_LENGTH: int = 15  # rolling window of past actions kept in env for reward functions (not in obs)

    building_props: BuildingProps = field(default_factory=lambda:
        BuildingProps(mC=300, K=20)
    )

    # Cached singleton instances (for backward compatibility and inspection)
    # WARNING: Do not pass these to parallel environments - use factory methods instead
    infras: Optional[List[Infrastructure]] = None
    statesources: Optional[List[StateSource]] = None

    # Reward composition — separate config
    reward_config: RewardConfig = field(default_factory=RewardConfig)

    def create_statesources(self) -> List[StateSource]:
        """
        Factory method to create fresh StateSource instances.

        Each call returns NEW independent instances, safe for parallel environments.
        Components have their own iteration counter and state.

        Returns:
            List of newly created StateSource instances.
        """
        
        # TODO VP 2026.04.24. : How are the statesources created in the envs? Are they always created here, or are they specified in yamls?
        # Statesources are outer components -- usually we can't really change them (except the s_temp_in_norm)
        return [
            EnergyPriceDataSource("E_price"),
            WeatherDataSource("weather"),
            InsideTemperature("desired_temp_in"),
            DesiredUserEnergyNeed("user_energy_need"),
            BuildingHeatLoss(
                name="building_heat_loss",
                K=self.building_props.K,
                mC=self.building_props.mC,
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
                max_power_kW=5.0,  # kW (consistent with battery 19 kW, EV 7 kW, solar 5 kW)
                K=self.building_props.K,
                mC=self.building_props.mC,
                cop_heat=3.0,
                cop_cool=2.5,
                control_step=self.CONTROL_STEP
            ),
            BatteryTremblay(
                "battery",
                control_step=self.CONTROL_STEP,
                max_power_kW=19.0,
                cell_capacity_Ah=3.5,
                max_charge_amps=48.0,
                max_charge_voltage=420.0,
                start_soc_percentage=0.3,
                max_charge_rate=1.5,
                history_length=4,
                E0=3.2,
                K=0.009,
                A=0.468,
                B=3.529,
                R_cell=0.01,
                n_series=125,
                n_parallel=10,
                charge_efficiency=0.95,
                discharge_efficiency=0.95,
                soc_min=0.1,
                soc_max=0.95,
            ),
            LinearEVCharger(
                "ev_charger",
                max_power_kW=7.0,
                max_charging_kW=7.0,
                control_step=self.CONTROL_STEP
            ),
            SolarPanel(
                "solar",
                max_power_kW=5.0,
            ),
            WindTurbine(
                "wind_turbine",
                max_power_kW=5.0,
                rated_power_kW=5.0,
            ),
            HouseholdEnergyConsumers(
                "hh_consumers",
                peak_consumption_kW=8.0,
            ),
        ]

    def __post_init__(self):
        """Lightweight post-init — does NOT eagerly call factory methods.

        Singleton fields (infras, statesources) are left as None to
        avoid unnecessary CSV parsing in every Ray worker subprocess that
        imports this module.  Call init_singletons() explicitly in the main
        process where those fields are actually needed.
        """

    def init_singletons(self) -> None:
        """Initialise the cached singleton component instances.

        Call this once in the main process after creating / loading a Config,
        before accessing self.infras / self.statesources / self.reward_config.rewards.
        Not needed in Ray worker subprocesses — they call the factory methods
        (create_infras, create_statesources) directly via adv_building_env_creator.
        Rewards are populated by ``RewardScheduleManager`` before this is called.

        WARNING: Do not pass these singleton instances to parallel environments
        — use the factory methods instead.
        """
        if self.statesources is None:
            self.statesources = self.create_statesources()

        if self.infras is None:
            self.infras = self.create_infras()

    def _log_label(self) -> str:
        return "EnvConfig"

    def log_values(self) -> None:
        """Log config values, showing component names instead of object repr."""
        def _names(components: list | None) -> str:
            if components is None:
                return "None"
            return "[" + ", ".join(c.name for c in components) + "]"

        lines = [
            f"  env_config_name = {self.env_config_name}",
            f"  EPISODE_LENGTH = {self.EPISODE_LENGTH}",
            f"  CONTROL_STEP = {self.CONTROL_STEP}",
            f"  ACTION_HISTORY_LENGTH = {self.ACTION_HISTORY_LENGTH}",
            f"  building_props = mC={self.building_props.mC}, K={self.building_props.K}",
            f"  infras = {_names(self.infras)}",
            f"  statesources = {_names(self.statesources)}",
        ]
        logger.info("%s:\n%s", self._log_label(), "\n".join(lines))
        self.reward_config.log_values()

# default/config instance
config = EnvConfig()
