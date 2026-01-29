"""Config serialization and management utilities."""

import json
from pathlib import Path
from typing import Dict, Any

from adv_building_gym.envs.utils import BuildingProps


class ConfigManager:
    """Handles serialization and deserialization of Config objects.

    Uses the flexible serialization system where each component (Infrastructure,
    StateSource, RewardFunction) knows how to serialize itself via the Serializable
    mixin. This allows adding new component types without modifying ConfigManager.
    """

    @staticmethod
    def to_dict(config) -> Dict[str, Any]:
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
            "config_name": config.config_name,
            "seed": config.seed,
            "EPISODE_LENGTH": config.EPISODE_LENGTH,
            "control_step": config.control_step,
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

        # Serialize rewards using their to_dict() method
        if config.rewards is not None:
            config_dict["rewards"] = [reward.to_dict() for reward in config.rewards]

        return config_dict

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]):
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
        from adv_building_gym.config.env_config import Config
        from adv_building_gym.devices.infrastructure import Infrastructure
        from adv_building_gym.devices.statesources import StateSource
        from adv_building_gym.rewards import RewardFunction

        # Create BuildingProps
        building_props_dict = config_dict.get("building_props", {})
        building_props = BuildingProps(
            mC=building_props_dict.get("mC", 300),
            K=building_props_dict.get("K", 20),
        )

        control_step = config_dict.get("control_step", 300)
        seed = config_dict.get("seed", 42)

        # Create Config with basic params (don't trigger __post_init__ defaults)
        config = Config(
            config_name=config_dict.get("config_name", "loaded_config"),
            seed=seed,
            EPISODE_LENGTH=config_dict.get("EPISODE_LENGTH", 288),
            control_step=control_step,
            building_props=building_props,
            # Set to empty lists to prevent __post_init__ from creating defaults
            infras=[],
            statesources=[],
            rewards=[],
        )

        # Build context for infrastructure deserialization
        infra_context = {
            "K": building_props.K,
            "mC": building_props.mC,
            "control_step": control_step,
            "seed": seed,
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

        # Build context for reward deserialization (includes infrastructures reference)
        reward_context = {
            "infrastructures": config.infras,
        }

        # Reconstruct rewards
        if "rewards" in config_dict:
            rewards = []
            for reward_dict in config_dict["rewards"]:
                reward = RewardFunction.from_dict(reward_dict, reward_context)
                rewards.append(reward)
            config.rewards = rewards

        return config

    @staticmethod
    def save(config, path: str | Path) -> None:
        """
        Save config to JSON file.

        Args:
            config: Config object to save
            path: File path where config should be saved (e.g., 'configs/my_config.json')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = ConfigManager.to_dict(config)

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @staticmethod
    def load(path: str | Path):
        """
        Load config from JSON file.

        Args:
            path: File path to load config from (e.g., 'configs/my_config.json')

        Returns:
            Config object reconstructed from file
        """
        path = Path(path)

        with open(path, 'r') as f:
            config_dict = json.load(f)

        return ConfigManager.from_dict(config_dict)
