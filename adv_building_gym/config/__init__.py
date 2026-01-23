"""Configuration module for AdvBuildingGym.

This module provides configuration data structures and serialization utilities.

Note: Config and config are imported lazily to avoid circular imports.
Use:
    from adv_building_gym.config import Config
    from adv_building_gym.config import config
    from adv_building_gym.config import ConfigManager
"""

# These don't cause circular imports - import directly
from .utils import Serializable, ComponentRegistry


def __getattr__(name):
    """Lazy import for Config, config, and ConfigManager to avoid circular imports."""
    if name == "Config":
        from .env_config import Config
        return Config
    elif name == "config":
        from .env_config import config
        return config
    elif name == "ConfigManager":
        from .config_manager import ConfigManager
        return ConfigManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Config",
    "config",
    "ConfigManager",
    "Serializable",
    "ComponentRegistry",
]
