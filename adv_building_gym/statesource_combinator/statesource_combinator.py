"""Schedule statesource bundle YAML files for curriculum training.

Mirrors ``InfraCombinator`` but cycles statesource bundles instead of infras.
Companion callback: ``statesource_schedule_callback.py``.
"""

import logging
from pathlib import Path
from typing import Literal

import yaml

from adv_building_gym.devices.statesources.base import StateSource

logger = logging.getLogger(__name__)


class _ParsedConfig:
    __slots__ = ("name", "statesource_dicts", "context")

    def __init__(self, name: str, statesource_dicts: list[dict], control_step: int) -> None:
        self.name = name
        self.statesource_dicts = statesource_dicts
        self.context = {"control_step": control_step}


class StatesourceCombinator:
    """Cycle statesource bundle YAMLs (configs/statesource_cfgs/**/*.yaml)."""

    def __init__(
        self,
        config_paths: list[str],
        control_step: int,
        swap_every_n_episodes: int = 300,
        mode: Literal["cycle", "off"] = "cycle",
    ) -> None:
        self.config_paths = config_paths
        self.control_step = control_step
        self.swap_every_n_episodes = swap_every_n_episodes
        self.mode = mode
        self._swap_index: int = 0

        self._configs: list[_ParsedConfig] = [self._load_config(p, control_step) for p in config_paths]
        
        logger.info(
            "StatesourceCombinator: %d configs loaded, mode=%s, swap_every_n_episodes=%d",
            len(self._configs), self.mode, self.swap_every_n_episodes,
        )
        
        for i, cfg in enumerate(self._configs):
            logger.info("  [%d] %s", i, cfg.name)

    @staticmethod
    def _load_config(path_str: str, control_step: int) -> _ParsedConfig:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"StatesourceCombinator: config file not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        if "infras" in raw or "EPISODE_LENGTH" in raw or "control_step" in raw:
            raise ValueError(
                f"StatesourceCombinator: {path} is not a pure statesource YAML. "
                f"Schedule entries must point at statesource-only YAMLs."
            )
        return _ParsedConfig(
            name=path.stem,
            statesource_dicts=raw.get("statesources", []),
            control_step=control_step,
        )

    def create_statesources(self, swap_index: int) -> list[StateSource]:
        cfg = self._configs[swap_index % len(self._configs)]
        return [StateSource.from_dict(spec, cfg.context) for spec in cfg.statesource_dicts]

    def advance(self) -> bool:
        if not self.is_enabled() or len(self._configs) < 2:
            return False
        self._swap_index += 1
        return True

    def get_active_config_name(self) -> str:
        if not self._configs:
            return "<none>"
        return self._configs[self._swap_index % len(self._configs)].name

    def is_enabled(self) -> bool:
        return self.mode != "off" and len(self._configs) > 0

    @classmethod
    def from_dict(cls, raw: dict, control_step: int) -> "StatesourceCombinator":
        if not raw:
            raise ValueError("StatesourceCombinator: empty schedule dict")
        return cls(
            config_paths=list(raw.get("configs", [])),
            control_step=control_step,
            swap_every_n_episodes=raw["swap_every_n_episodes"],
            mode=raw.get("mode", "cycle"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path, control_step: int) -> "StatesourceCombinator":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"StatesourceCombinator schedule file not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw, control_step)
