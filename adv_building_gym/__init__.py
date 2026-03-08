

# Gymnasium registration for custom environments
import gymnasium as gym
from gymnasium.envs.registration import register

from .envs import AdvBuildingGym
from .controllers import FuzzyController, MPCController, PIController, PIDController
from .config import config, ConfigManager
from .callbacks import (
    make_episode_metrics_callback_class,
    make_trajectory_logging_callback_class,
    make_checkpoint_callback_class,
)
from .evaluation import evaluate_model, EvalResults


def _make_adv_building(**kwargs):
    """Entry-point factory for gym.make('AdvBuilding').

    Calls factory methods at make() time so that components are never
    captured as None at import time (singletons are lazily initialised).
    Each gym.make() call gets independent component instances.
    """
    from .config import config as env_config
    infras = env_config.create_infras()
    statesources = env_config.create_statesources()
    rewards = env_config.create_rewards(infras)
    return AdvBuildingGym(
        infras=infras,
        statesources=statesources,
        rewards=rewards,
        building_props=env_config.building_props,
    )


# Register advanced environment -- for stable baselines (SB3)
register(
    id="AdvBuilding",
    entry_point=_make_adv_building,
    max_episode_steps=288,
)

# Exported components of the adv_building_gym package
__all__ = [
    "AdvBuildingGym",
    "config",
    "ConfigManager",
    "FuzzyController",
    "MPCController",
    "PIController",
    "PIDController",
    "make_episode_metrics_callback_class",
    "make_trajectory_logging_callback_class",
    "make_checkpoint_callback_class",
    "evaluate_model",
    "EvalResults",
]
