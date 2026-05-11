"""Schedule infrastructure config YAML files for curriculum training.

Cycles through infra YAMLs (``configs/infra_cfgs/**/*.yaml``) during training so
the agent generalises across many building configurations.

The swap is synchronised across all Ray workers via the companion
``infra_schedule_callback`` (iteration-aligned ``on_train_result``).
"""

import logging
from pathlib import Path
from typing import Literal

import yaml

from adv_building_gym.devices.infrastructure.base import Infrastructure

logger = logging.getLogger(__name__)


class _ParsedConfig:
    """One parsed infra YAML (infras only)."""

    __slots__ = ("name", "infra_dicts", "context")

    def __init__(self, name: str, infra_dicts: list[dict], control_step: int) -> None:
        self.name = name
        self.infra_dicts = infra_dicts
        # Deserialization context (same key as EnvConfigManager).
        self.context = {"control_step": control_step}


class InfraCombinator:
    """Schedule infra YAML files for infrastructure curriculum training.

    Loads a sequence of infra YAMLs (``configs/infra_cfgs/**/*.yaml``) and cycles
    through them during training.  Each file declares only the ``infras``
    list -- statesources, timing, building envelope, and rewards are
    managed separately.

    Args:
        config_paths: Ordered list of infra YAML file paths.
        control_step: Control step (seconds) of the active env config; used
            as deserialisation context for all entries.
        swap_every_n_iterations: Hold each config for N training iterations.
        mode: ``"cycle"`` for round-robin, ``"off"`` to disable swapping.
    """

    def __init__(
        self,
        config_paths: list[str],
        control_step: int,
        swap_every_n_iterations: int = 300,
        mode: Literal["cycle", "off"] = "cycle",
    ) -> None:
        self.config_paths = config_paths
        self.control_step = control_step
        self.swap_every_n_iterations = swap_every_n_iterations
        self.mode = mode
        self._swap_index: int = 0

        self._configs: list[_ParsedConfig] = [
            self._load_config(p, control_step) for p in config_paths
        ]

        logger.info(
            "InfraCombinator: %d configs loaded, mode=%s, "
            "swap_every_n_iterations=%d",
            len(self._configs), self.mode, self.swap_every_n_iterations,
        )
        for i, cfg in enumerate(self._configs):
            logger.info("  [%d] %s", i, cfg.name)

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

    def create_infras(self, swap_index: int) -> list[Infrastructure]:
        """Create fresh Infrastructure instances for the config at *swap_index*."""
        cfg = self._configs[swap_index % len(self._configs)]
        return [
            Infrastructure.from_dict(spec, cfg.context)
            for spec in cfg.infra_dicts
        ]

    def advance(self) -> bool:
        """Advance to the next config. Returns True if changed."""
        if not self.is_enabled() or len(self._configs) < 2:
            return False
        self._swap_index += 1
        return True

    def get_active_config_name(self) -> str:
        """Return the name (file stem) of the currently active YAML."""
        if not self._configs:
            return "<none>"
        return self._configs[self._swap_index % len(self._configs)].name

    def is_enabled(self) -> bool:
        """Return True if infra scheduling is active."""
        return self.mode != "off" and len(self._configs) > 0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: dict, control_step: int) -> "InfraCombinator":
        """Build a combinator from an inlined schedule dict.

        Expected layout::

            mode: cycle
            swap_every_n_iterations: 300
            configs:
              - configs/infra_cfgs/test1_small.yaml
              - configs/infra_cfgs/test1_mid.yaml
        """
        if not raw:
            raise ValueError("InfraCombinator: empty schedule dict")
        return cls(
            config_paths=list(raw.get("configs", [])),
            control_step=control_step,
            swap_every_n_iterations=raw.get("swap_every_n_iterations", 300),
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
