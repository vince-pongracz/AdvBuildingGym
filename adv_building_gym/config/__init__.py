"""Configuration module for AdvBuildingGym.

This module provides configuration data structures and serialization utilities.

Lazy imports via ``__getattr__`` are required for EnvConfig, config,
EnvConfigManager, DataCombinator and load_data_combinator_config to break
a circular dependency chain:

    config/__init__  (eager import of env_config)
  → env_config       (imports devices/rewards at module level)
  → devices/statesources/base.py
  → config.utils.serializable  (triggers config/__init__.py again
                                 while it is still partially initialized)

Because ``from adv_building_gym.config.utils.serializable import …`` causes
Python to execute ``config/__init__.py`` the first time the config package
is touched, any eager import of ``env_config`` (which in turn imports
device/reward classes) creates the cycle above.  The ``__getattr__`` hook
defers these imports until they are actually requested, by which time
``config/__init__.py`` has finished initializing.

The ``TYPE_CHECKING`` block provides the real types to static analysers
(Pylance / mypy) without triggering the runtime cycle.

Use:
    from adv_building_gym.config import EnvConfig
    from adv_building_gym.config import config
    from adv_building_gym.config import EnvConfigManager
    from adv_building_gym.config import load_data_combinator_config
"""

# TODO VP 2026.03.19. : Check and resolve this circular dependency thingy

from __future__ import annotations

from typing import TYPE_CHECKING

# These don't cause circular imports - import directly
from .utils import Serializable, ComponentRegistry
from .training_param_config import TrainingParamConfig

if TYPE_CHECKING:
    from .env_config import EnvConfig as EnvConfig, config as config
    from .env_config_manager import EnvConfigManager as EnvConfigManager
    from .data_config import load_data_combinator_config as load_data_combinator_config


def __getattr__(name):
    """Lazy import to avoid circular imports (see module docstring)."""
    if name == "EnvConfig":
        from .env_config import EnvConfig
        return EnvConfig
    elif name == "config":
        from .env_config import config
        return config
    elif name == "EnvConfigManager":
        from .env_config_manager import EnvConfigManager
        return EnvConfigManager
    elif name == "DataCombinator":
        from adv_building_gym.data_combinator import DataCombinator
        return DataCombinator
    elif name == "load_data_combinator_config":
        from .data_config import load_data_combinator_config
        return load_data_combinator_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EnvConfig",
    "config",
    "EnvConfigManager",
    "DataCombinator",
    "load_data_combinator_config",
    "Serializable",
    "ComponentRegistry",
    "TrainingParamConfig",
]
