"""Schedule infrastructure config YAML files for curriculum training.

Cycles through infra YAMLs (``configs/infra_cfgs/**/*.yaml``) during training so
the agent generalises across many building configurations.

The swap is synchronised across all Ray workers via the companion
``infra_schedule_callback`` (episode-budget-aligned ``on_train_result``).

Schedule YAML carries separate train and eval lists::

    mode: cycle
    swap_every_n_episodes: 300
    configs:
      train:
        - configs/infra_cfgs/.../foo_1.yaml
        - configs/infra_cfgs/.../foo_2.yaml
      eval:
        - configs/infra_cfgs/.../foo_1.yaml
        - configs/infra_cfgs/.../foo_2.yaml

Training cycles ``configs.train``; ``run_eval_ray.py`` iterates each entry
of ``configs.eval`` for the requested number of eval episodes.
"""

import logging
from pathlib import Path
from typing import Literal

import yaml

from adv_building_gym.components.infrastructure.base import Infrastructure
from adv_building_gym.components.registry import from_dict as component_from_dict

logger = logging.getLogger(__name__)


Split = Literal["train", "eval"]


class _ParsedConfig:
    """One parsed infra YAML (infras only)."""

    __slots__ = ("name", "infra_dicts", "context")

    def __init__(self, name: str, infra_dicts: list[dict], control_step: int) -> None:
        self.name = name
        self.infra_dicts = infra_dicts
        # Deserialization context (same key as EnvConfigManager).
        self.context = {"control_step": control_step}


class InfraCombinator:
    """Schedule infra YAMLs for infrastructure curriculum training.

    Two ordered lists of infra-only YAMLs (``configs/infra_cfgs/**/*.yaml``): train (cycled by
    the swap callback) and eval (iterated by the orchestrator).

    Args:
        train_config_paths: Ordered list of infra YAML file paths for training.
        eval_config_paths:  Ordered list of infra YAML file paths for evaluation.
        control_step: Control step (seconds) of the active env config; used
            as deserialisation context for all entries.
        swap_every_n_episodes: Hold each train config for N episodes collected
            across all env_runners. The scheduler callback clamps this
            to ``max(N, num_env_runners)`` at registration time.
        mode: ``"cycle"`` for round-robin, ``"off"`` to disable swapping
            (training only -- eval always iterates ``eval_config_paths``).
    """

    def __init__(
        self,
        train_config_paths: list[str],
        eval_config_paths: list[str],
        control_step: int,
        swap_every_n_episodes: int = 300,
        mode: Literal["cycle", "off"] = "cycle",
    ) -> None:
        if not train_config_paths:
            raise ValueError("InfraCombinator: configs.train must be a non-empty list")
        if not eval_config_paths:
            raise ValueError("InfraCombinator: configs.eval must be a non-empty list")

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
            "InfraCombinator: %d train / %d eval configs, mode=%s, "
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
            raise FileNotFoundError(f"InfraCombinator: config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        if "statesources" in raw or "EPISODE_LENGTH" in raw or "control_step" in raw \
                or "building_props" in raw:
            raise ValueError(
                f"InfraCombinator: {path} is not a pure infra YAML. "
                f"Infra schedule entries must point at infra-only YAMLs "
                f"(configs/infra_cfgs/**/*.yaml) containing only 'infras'."
            )

        return _ParsedConfig(
            name=path.stem,
            infra_dicts=raw.get("infras", []),
            control_step=control_step,
        )

    # ------------------------------------------------------------------
    # Runtime API
    # ------------------------------------------------------------------

    def create_infras(self, swap_index: int, split: Split = "train") -> list[Infrastructure]:
        """Create fresh Infrastructure instances for *split* at *swap_index*."""
        cfgs = self._configs[split]
        cfg = cfgs[swap_index % len(cfgs)]
        return [
            component_from_dict(spec, "infrastructure", cfg.context)
            for spec in cfg.infra_dicts
        ]

    def advance(self) -> bool:
        """Advance the *training* cycle to the next config. Returns True if changed."""
        if not self.is_enabled() or len(self._configs["train"]) < 2:
            return False
        self._swap_index += 1
        return True

    def get_active_config_name(self, split: Split = "train", swap_index: int | None = None) -> str:
        """Return the name (file stem) of the currently active YAML in *split*."""
        cfgs = self._configs[split]
        if not cfgs:
            return "<none>"
        idx = self._swap_index if swap_index is None else swap_index
        return cfgs[idx % len(cfgs)].name

    def is_enabled(self) -> bool:
        """Return True if (training-side) infra scheduling is active."""
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

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: dict, control_step: int) -> "InfraCombinator":
        """Build a combinator from an inlined schedule dict (mode / swap_every_n_episodes / configs.{train,eval}).

        Expected layout::

            mode: cycle
            swap_every_n_episodes: 300
            configs:
              train: [<path>, ...]
              eval:  [<path>, ...]
        """
        if not raw:
            raise ValueError("InfraCombinator: empty schedule dict")
        configs = raw.get("configs")
        if not isinstance(configs, dict) or "train" not in configs or "eval" not in configs:
            raise ValueError(
                "InfraCombinator: 'configs' must be a dict with 'train' and "
                "'eval' keys (lists of infra YAML paths)."
            )
        return cls(
            train_config_paths=list(configs.get("train") or []),
            eval_config_paths=list(configs.get("eval") or []),
            control_step=control_step,
            swap_every_n_episodes=raw["swap_every_n_episodes"],
            mode=raw.get("mode", "cycle"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path, control_step: int) -> "InfraCombinator":
        """Load an infra schedule from a YAML config file (thin wrapper)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"InfraCombinator schedule file not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw, control_step)
