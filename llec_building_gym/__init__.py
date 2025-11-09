# llec_building_gym/__init__.py

# Gymnasium registration for custom environments
import gymnasium as gym
from gymnasium.envs.registration import register
from .envs import BaseBuildingGym, Building
from .controllers import FuzzyController, MPCController, PIController, PIDController

# Register environment with temperature reward mode
register(
    id="LLEC-HeatPumpHouse-1R1C-Temperature-v0",
    entry_point="llec_building_gym.envs:BaseBuildingGym",
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
    entry_point="llec_building_gym.envs:BaseBuildingGym",
    max_episode_steps=288,
    kwargs={
        # temperature and economic reward
        "reward_mode": "combined",
        "temperature_weight": 1.0,
        "economic_weight": 1.0,
        "render_mode": None,
    },
)

# Exported components of the llec_building_gym package
__all__ = [
    "BaseBuildingGym",
    "Building",
    "FuzzyController",
    "MPCController",
    "PIController",
    "PIDController",
]
