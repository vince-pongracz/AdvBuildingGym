"""Schedule statesource bundle YAML files for curriculum training.

Mirrors ``InfraCombinator`` but cycles statesource bundles instead of infras.
Companion callback: ``statesource_schedule_callback.py``.

Schedule YAML carries separate train and eval lists (see
``InfraCombinator`` for the layout).
"""

import logging
from pathlib import Path
from typing import Literal

import yaml

from adv_building_gym.devices.statesources.base import StateSource
from adv_building_gym.utils.serialization import from_dict as component_from_dict

logger = logging.getLogger(__name__)


Split = Literal["train", "eval"]


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
        train_config_paths: list[str],
        eval_config_paths: list[str],
        control_step: int,
        swap_every_n_episodes: int = 300,
        mode: Literal["cycle", "off"] = "cycle",
    ) -> None:
        if not train_config_paths:
            raise ValueError("StatesourceCombinator: configs.train must be a non-empty list")
        if not eval_config_paths:
            raise ValueError("StatesourceCombinator: configs.eval must be a non-empty list")

        self.control_step = control_step
        self.swap_every_n_episodes = swap_every_n_episodes
        self.mode = mode
        self._swap_index: int = 0

        self._configs: dict[Split, list[_ParsedConfig]] = {
            "train": [self._load_config(p, control_step) for p in train_config_paths],
            "eval":  [self._load_config(p, control_step) for p in eval_config_paths],
        }
        self._config_paths: dict[Split, list[str]] = {
            "train": list(train_config_paths),
            "eval":  list(eval_config_paths),
        }

        logger.info(
            "StatesourceCombinator: %d train / %d eval configs, mode=%s, "
            "swap_every_n_episodes=%d",
            len(self._configs["train"]), len(self._configs["eval"]),
            self.mode, self.swap_every_n_episodes,
        )

        for split in ("train", "eval"):
            for i, cfg in enumerate(self._configs[split]):
                logger.info("  [%s %d] %s", split, i, cfg.name)

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

    def create_statesources(self, swap_index: int, split: Split = "train") -> list[StateSource]:
        cfgs = self._configs[split]
        cfg = cfgs[swap_index % len(cfgs)]
        return [component_from_dict(spec, "statesource", cfg.context) for spec in cfg.statesource_dicts]

    def advance(self) -> bool:
        if not self.is_enabled() or len(self._configs["train"]) < 2:
            return False
        self._swap_index += 1
        return True

    def get_active_config_name(self, split: Split = "train", swap_index: int | None = None) -> str:
        cfgs = self._configs[split]
        if not cfgs:
            return "<none>"
        idx = self._swap_index if swap_index is None else swap_index
        return cfgs[idx % len(cfgs)].name

    def is_enabled(self) -> bool:
        return self.mode != "off" and len(self._configs["train"]) > 0

    def train_count(self) -> int:
        return len(self._configs["train"])

    def eval_count(self) -> int:
        return len(self._configs["eval"])

    def get_eval_config_name(self, idx: int) -> str:
        return self._configs["eval"][idx].name

    @property
    def config_paths(self) -> list[str]:
        """Backwards-compatible alias for the training paths."""
        return list(self._config_paths["train"])

    @classmethod
    def from_dict(cls, raw: dict, control_step: int) -> "StatesourceCombinator":
        if not raw:
            raise ValueError("StatesourceCombinator: empty schedule dict")
        configs = raw.get("configs")
        if not isinstance(configs, dict) or "train" not in configs or "eval" not in configs:
            raise ValueError(
                "StatesourceCombinator: 'configs' must be a dict with 'train' "
                "and 'eval' keys (lists of statesource YAML paths)."
            )
        return cls(
            train_config_paths=list(configs.get("train") or []),
            eval_config_paths=list(configs.get("eval") or []),
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
