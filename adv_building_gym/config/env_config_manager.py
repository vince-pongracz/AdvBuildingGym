"""Config serialization and management utilities.

Configuration is split across three concerns, glued by a wrapper YAML:

    configs/env/<name>.yaml         # wrapper — references the three files below
    configs/infras/<name>.yaml      # building_props + infras list
    configs/statesources/<name>.yaml # statesources list
    configs/env_meta/<name>.yaml    # EPISODE_LENGTH, control_step

The wrapper format is::

    env_config_name: env_name
    infras: configs/infras/test1_small.yaml
    statesources: configs/statesources/default.yaml
    env_meta: configs/env_meta/default.yaml
"""

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
    StateSource) knows how to serialize itself.
    """

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def from_dict(
        wrapper: Dict[str, Any],
        infras_doc: Dict[str, Any],
        statesources_doc: Dict[str, Any],
        env_meta_doc: Dict[str, Any],
    ) -> EnvConfig:
        """Reconstruct an EnvConfig from the four parsed YAML documents.

        Stores the raw component specs on the config; the env's factory
        methods (``create_infras`` / ``create_statesources``) are the
        single deserialisation path.
        """
        from adv_building_gym.config.env_config import EnvConfig
        from adv_building_gym.config.reward_config import RewardConfig

        bp_dict = infras_doc.get("building_props", {})
        building_props = BuildingProps(
            mC=bp_dict.get("mC", 300),
            K=bp_dict.get("K", 20),
        )

        control_step = env_meta_doc.get("control_step", 300)
        episode_length = env_meta_doc.get("EPISODE_LENGTH", 288)

        infra_specs = list(infras_doc.get("infras", []))
        statesource_specs = list(statesources_doc.get("statesources", []))

        config = EnvConfig(
            env_config_name=wrapper.get("env_config_name", "loaded_config"),
            EPISODE_LENGTH=episode_length,
            CONTROL_STEP=control_step,
            building_props=building_props,
            infra_specs=infra_specs,
            statesource_specs=statesource_specs,
            infras=None,
            statesources=None,
            reward_config=RewardConfig(),
        )
        return config

    @staticmethod
    def load(path: str | Path) -> EnvConfig:
        """Load an env config from a wrapper YAML.

        Resolves the three referenced files (infras, statesources, env_meta)
        relative to the project root (current working directory) so wrapper
        files can use repo-relative paths like ``configs/infras/...``.
        """
        wrapper_path = Path(path)
        wrapper = EnvConfigManager._load_yaml(wrapper_path)

        for key in ("infras", "statesources", "env_meta"):
            if key not in wrapper:
                raise ValueError(
                    f"Wrapper config {wrapper_path} is missing required key '{key}'. "
                    f"Expected: env_config_name, infras, statesources, env_meta."
                )

        infras_doc = EnvConfigManager._load_yaml(Path(wrapper["infras"]))
        statesources_doc = EnvConfigManager._load_yaml(Path(wrapper["statesources"]))
        env_meta_doc = EnvConfigManager._load_yaml(Path(wrapper["env_meta"]))

        config = EnvConfigManager.from_dict(wrapper, infras_doc, statesources_doc, env_meta_doc)
        config.log_values()

        logger.info("Config loaded successfully: %s", config.env_config_name)
        return config

    @staticmethod
    def save(
        config,
        wrapper_path: str | Path,
        infras_path: str | Path,
        statesources_path: str | Path,
        env_meta_path: str | Path,
    ) -> None:
        """Save an EnvConfig as the four-file split layout.

        All four paths must be supplied; the wrapper is written with
        repo-relative references to the other three (as given).
        """
        wrapper_path = Path(wrapper_path)
        infras_path = Path(infras_path)
        statesources_path = Path(statesources_path)
        env_meta_path = Path(env_meta_path)

        for p in (wrapper_path, infras_path, statesources_path, env_meta_path):
            p.parent.mkdir(parents=True, exist_ok=True)

        # Prefer the raw specs (round-trip from load) over re-serialising live
        # instances; if specs are absent, fall back to the live components.
        infra_specs = list(config.infra_specs) if config.infra_specs \
            else [i.to_dict() for i in (config.infras or [])]
        statesource_specs = list(config.statesource_specs) if config.statesource_specs \
            else [s.to_dict() for s in (config.statesources or [])]
        infras_doc = {
            "building_props": {
                "mC": config.building_props.mC,
                "K": config.building_props.K,
            },
            "infras": infra_specs,
        }
        statesources_doc = {
            "statesources": statesource_specs,
        }
        env_meta_doc = {
            "EPISODE_LENGTH": config.EPISODE_LENGTH,
            "control_step": config.CONTROL_STEP,
        }
        wrapper_doc = {
            "env_config_name": config.env_config_name,
            "infras": str(infras_path),
            "statesources": str(statesources_path),
            "env_meta": str(env_meta_path),
        }

        for path, doc in (
            (infras_path, infras_doc),
            (statesources_path, statesources_doc),
            (env_meta_path, env_meta_doc),
            (wrapper_path, wrapper_doc),
        ):
            with path.open("w") as f:
                yaml.dump(doc, f, default_flow_style=False, sort_keys=False)
