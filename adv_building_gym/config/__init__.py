"""Configuration module for AdvBuildingGym.

This module provides configuration data structures and serialization utilities.

Note: Config and config are imported lazily to avoid circular imports.
Use:
    from adv_building_gym.config import Config
    from adv_building_gym.config import config
    from adv_building_gym.config import ConfigManager
"""
# TODO VP 2026.03.10. : Is this lazyness still needed at the imports?

from __future__ import annotations

from typing import TYPE_CHECKING

# These don't cause circular imports - import directly
from .utils import Serializable, ComponentRegistry
from .training_config import TrainingConfig

if TYPE_CHECKING:
    from .env_config import Config as Config, config as config
    from .config_manager import ConfigManager as ConfigManager


def __getattr__(name):
    """Lazy import for Config, config, ConfigManager, and DataCombinator to avoid circular imports."""
    if name == "Config":
        from .env_config import Config
        return Config
    elif name == "config":
        from .env_config import config
        return config
    elif name == "ConfigManager":
        from .config_manager import ConfigManager
        return ConfigManager
    elif name == "DataCombinator":
        from adv_building_gym.data_combinator import DataCombinator
        return DataCombinator
    elif name == "load_data_combinator":
        from .data_config import load_data_combinator
        return load_data_combinator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Config",
    "config",
    "ConfigManager",
    "DataCombinator",
    "load_data_combinator",
    "Serializable",
    "ComponentRegistry",
    "TrainingConfig",
]
