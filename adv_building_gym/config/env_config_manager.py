"""Config serialization and management utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any

import yaml
import logging

if TYPE_CHECKING:
    from adv_building_gym.config.env_config import EnvConfig

from adv_building_gym.envs.utils import BuildingProps

logger = logging.getLogger(__name__)

class EnvConfigManager:
    """Handles serialization and deserialization of EnvConfig objects.

    Uses the flexible serialization system where each component (Infrastructure,
    StateSource, RewardConfig) knows how to serialize itself.
    """

    @staticmethod
    def to_dict(config: EnvConfig) -> Dict[str, Any]:
        """
        Serialize config to dictionary.

        Uses each component's to_dict() method for flexible serialization
        that doesn't hard-code specific attributes.

        Args:
            config: Config object to serialize

        Returns:
            Dictionary representation of config
        """
        config_dict = {
            "env_config_name": config.env_config_name,
            "EPISODE_LENGTH": config.EPISODE_LENGTH,
            "control_step": config.CONTROL_STEP,
            "building_props": {
                "mC": config.building_props.mC,
                "K": config.building_props.K,
            },
        }

        # Serialize infrastructures using their to_dict() method
        if config.infras is not None:
            config_dict["infras"] = [infra.to_dict() for infra in config.infras]

        # Serialize statesources using their to_dict() method
        if config.statesources is not None:
            config_dict["statesources"] = [source.to_dict() for source in config.statesources]


        return config_dict

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> EnvConfig:
        """
        Deserialize config from dictionary.

        Uses the ComponentRegistry to find the correct class for each component
        and passes appropriate context (building_props, control_step, infras)
        for derived parameters.

        Args:
            config_dict: Dictionary representation of config

        Returns:
            Config object
        """
        from adv_building_gym.config.env_config import EnvConfig
        from adv_building_gym.config.reward_config import RewardConfig
        from adv_building_gym.devices.infrastructure import Infrastructure
        from adv_building_gym.devices.statesources import StateSource

        # Create BuildingProps
        building_props_dict = config_dict.get("building_props", {})
        building_props = BuildingProps(
            mC=building_props_dict.get("mC", 300),
            K=building_props_dict.get("K", 20),
        )

        control_step = config_dict.get("control_step", 300)

        # Create Config with basic params (don't trigger __post_init__ defaults)
        config = EnvConfig(
            env_config_name=config_dict.get("env_config_name", "loaded_config"),
            EPISODE_LENGTH=config_dict.get("EPISODE_LENGTH", 288),
            CONTROL_STEP=control_step,
            building_props=building_props,
            # Set to empty lists to prevent __post_init__ from creating defaults
            infras=[],
            statesources=[],
            reward_config=RewardConfig(),
        )

        # Build context for infrastructure deserialization
        infra_context = {
            "K": building_props.K,
            "mC": building_props.mC,
            "control_step": control_step,
        }

        # Reconstruct infrastructures
        if "infras" in config_dict:
            infras = []
            for infra_dict in config_dict["infras"]:
                infra = Infrastructure.from_dict(infra_dict, infra_context)
                infras.append(infra)
            config.infras = infras

        # Build context for statesource deserialization
        statesource_context = {
            "K": building_props.K,
            "mC": building_props.mC,
            "timestep": control_step,
        }

        # Reconstruct statesources
        if "statesources" in config_dict:
            statesources = []
            for source_dict in config_dict["statesources"]:
                source = StateSource.from_dict(source_dict, statesource_context)
                statesources.append(source)
            config.statesources = statesources


        return config

    @staticmethod
    def save(config, path: str | Path) -> None:
        """
        Save config to YAML file.

        Args:
            config: Config object to save
            path: File path where config should be saved (e.g., 'configs/my_config.yaml')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = EnvConfigManager.to_dict(config)

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def load(path: str | Path) -> EnvConfig:
        """
        Load config from YAML file.

        Args:
            path: File path to load config from (e.g., 'configs/my_config.yaml')

        Returns:
            Config object reconstructed from file
        """
        path = Path(path)

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        config = EnvConfigManager.from_dict(config_dict)
        config.log_values()

        logger.info("Config loaded successfully: %s", config.env_config_name)

        return config
