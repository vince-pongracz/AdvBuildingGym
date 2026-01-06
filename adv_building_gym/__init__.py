

# Gymnasium registration for custom environments
import gymnasium as gym
from gymnasium.envs.registration import register

from .envs import BaseBuildingGym, Building, AdvBuildingGym
from .controllers import FuzzyController, MPCController, PIController, PIDController
from .env_config import config
from .callbacks import create_on_episode_end_callback

# Register environment with temperature reward mode
register(
    id="LLEC-HeatPumpHouse-1R1C-Temperature-v0",
    entry_point="adv_building_gym.envs:BaseBuildingGym",
    max_episode_steps=288,
    kwargs={
        # temperature-based reward
        "reward_mode": "temperature",
        "render_mode": None,
    },
)

# Register environment with combined reward mode
register(
    id="LLEC-HeatPumpHouse-1R1C-Combined-v0",
    entry_point="adv_building_gym.envs:BaseBuildingGym",
    max_episode_steps=288,
    kwargs={
        # temperature and economic reward
        "reward_mode": "combined",
        "temperature_weight": 1.0,
        "economic_weight": 1.0,
        "render_mode": None,
    },
)

# Register advanced environment -- for stable baselines (SB)
register(
    id="AdvBuilding",
    entry_point="adv_building_gym.envs:AdvBuildingGym",
    max_episode_steps=288,
    kwargs={
        "infras": config.infras,
        "datasources": config.datasources,
        "rewards": config.rewards,
        "building_props": config.building_props,
        "render_mode": None,
    },
)

# Exported components of the adv_building_gym package
__all__ = [
    "BaseBuildingGym",
    "Building",
    "AdvBuildingGym",
    "config",
    "FuzzyController",
    "MPCController",
    "PIController",
    "PIDController",
    "create_on_episode_end_callback",
]
