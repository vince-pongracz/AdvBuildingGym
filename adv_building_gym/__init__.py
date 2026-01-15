

# Gymnasium registration for custom environments
import gymnasium as gym
from gymnasium.envs.registration import register

from .envs import AdvBuildingGym
from .controllers import FuzzyController, MPCController, PIController, PIDController
from .env_config import config
from .callbacks import (
    create_on_episode_end_callback,
    make_checkpoint_callback_class,
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
    "AdvBuildingGym",
    "config",
    "FuzzyController",
    "MPCController",
    "PIController",
    "PIDController",
    "create_on_episode_end_callback",
    "make_checkpoint_callback_class",
]
