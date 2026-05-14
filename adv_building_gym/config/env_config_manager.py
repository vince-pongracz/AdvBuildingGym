"""Config serialization and management utilities.

Environment topology is split across three concerns, all referenced by
the trial config (``configs/trial_cfgs/<name>.yaml``):

    configs/infra_cfgs/<name>.yaml        # infras list (building envelope on BuildingHeatLoss)
    configs/statesource_cfgs/<name>.yaml  # statesources list
    configs/env_meta/<name>.yaml          # EPISODE_LENGTH, control_step
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any

import yaml
import logging

if TYPE_CHECKING:
    from adv_building_gym.config.env_config import EnvConfig

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
        infras_doc: Dict[str, Any],
        statesources_doc: Dict[str, Any],
        env_meta_doc: Dict[str, Any],
    ) -> EnvConfig:
        """Reconstruct an EnvConfig from the three parsed YAML documents.

        Stores the raw component specs on the config; the env's factory
        methods (``create_infras`` / ``create_statesources``) are the
        single deserialisation path.
        """
        from adv_building_gym.config.env_config import EnvConfig
        from adv_building_gym.config.reward_config import RewardConfig

        control_step = env_meta_doc.get("control_step", 300)
        episode_length = env_meta_doc.get("EPISODE_LENGTH", 288)
        allow_early_termination = bool(env_meta_doc.get("allow_early_termination", True))

        hst_cfg = env_meta_doc.get("hst_env_wrapper") or {}
        hst_enabled = bool(hst_cfg.get("enabled", False))
        hst_len = int(hst_cfg.get("hst_len", 0))
        if hst_enabled and hst_len <= 0:
            raise ValueError("env_meta.hst_env_wrapper: hst_len must be > 0 when enabled=true")

        infra_specs = list(infras_doc.get("infras", []))
        statesource_specs = list(statesources_doc.get("statesources", []))

        config = EnvConfig(
            EPISODE_LENGTH=episode_length,
            CONTROL_STEP=control_step,
            allow_early_termination=allow_early_termination,
            hst_env_wrapper_enabled=hst_enabled,
            hst_env_wrapper_hst_len=hst_len,
            infra_specs=infra_specs,
            statesource_specs=statesource_specs,
            infras=None,
            statesources=None,
            reward_config=RewardConfig(),
        )
        return config


    @staticmethod
    def save(
        config,
        infras_path: str | Path,
        statesources_path: str | Path,
        env_meta_path: str | Path,
    ) -> None:
        """Save an EnvConfig as the three-file split layout."""
        infras_path = Path(infras_path)
        statesources_path = Path(statesources_path)
        env_meta_path = Path(env_meta_path)

        for p in (infras_path, statesources_path, env_meta_path):
            p.parent.mkdir(parents=True, exist_ok=True)

        infra_specs = list(config.infra_specs) if config.infra_specs \
            else [i.to_dict() for i in (config.infras or [])]
        statesource_specs = list(config.statesource_specs) if config.statesource_specs \
            else [s.to_dict() for s in (config.statesources or [])]
        infras_doc = {"infras": infra_specs}
        statesources_doc = {"statesources": statesource_specs}
        env_meta_doc = {
            "EPISODE_LENGTH": config.EPISODE_LENGTH,
            "control_step": config.CONTROL_STEP,
            "allow_early_termination": config.allow_early_termination,
        }
        if config.hst_env_wrapper_enabled:
            env_meta_doc["hst_env_wrapper"] = {
                "enabled": True,
                "hst_len": config.hst_env_wrapper_hst_len,
            }

        for path, doc in (
            (infras_path, infras_doc),
            (statesources_path, statesources_doc),
            (env_meta_path, env_meta_doc),
        ):
            with path.open("w") as f:
                yaml.dump(doc, f, default_flow_style=False, sort_keys=False)
