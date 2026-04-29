"""Configuration module for AdvBuildingGym.

This module provides configuration data structures and serialization utilities.

Previously, this module used ``__getattr__`` lazy imports to work around a
circular dependency: ``config/__init__`` → ``env_config`` → devices →
``config.utils.serializable`` → ``config/__init__`` (still initializing).

That cycle was resolved by moving ``Serializable`` and ``ComponentRegistry``
to ``adv_building_gym.utils.serializable``, which devices and rewards import
directly. All config sub-modules can now be imported eagerly.
"""

from .env_config import EnvConfig
from .reward_config import RewardConfig
from .env_config_manager import EnvConfigManager
from .reward_schedule_manager import RewardScheduleManager
from .reward_config_serializer import RewardConfigSerializer
from .training_param_config import TrainingParamConfig
from .data_config import load_data_combinator_config

__all__ = [
    "EnvConfig",
    "RewardConfig",
    "EnvConfigManager",
    "RewardScheduleManager",
    "RewardConfigSerializer",
    "TrainingParamConfig",
    "load_data_combinator_config",
]
