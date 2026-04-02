"""Schedule infrastructure config YAML files for curriculum training.

Cycles through env config YAML files during training so the agent
generalises across many building configurations.

The swap is synchronised across all Ray workers via the companion
``infra_schedule_callback`` (iteration-aligned ``on_train_result``).
"""

import logging
from pathlib import Path
from typing import Literal

import yaml

from adv_building_gym.devices.infrastructure.base import Infrastructure
from adv_building_gym.envs.utils import BuildingProps

logger = logging.getLogger(__name__)


class _ParsedConfig:
    """One parsed env config YAML (infras + building_props only)."""

    __slots__ = ("name", "infra_dicts", "building_props", "context")

    def __init__(self, name: str, infra_dicts: list[dict],
                building_props: BuildingProps, control_step: int) -> None:
        self.name = name
        self.infra_dicts = infra_dicts
        self.building_props = building_props
        # Deserialization context (same keys as EnvConfigManager)
        self.context = {
            "K": building_props.K,
            "mC": building_props.mC,
            "control_step": control_step,
        }


class InfraCombinator:
    """Schedule env config YAML files for infrastructure curriculum training.

    Loads a sequence of env config YAMLs and cycles through them during
    training.  Only the ``infras`` and ``building_props`` sections are
    extracted -- statesources and rewards are managed separately.

    Args:
        config_paths: Ordered list of env config YAML file paths.
        swap_every_n_iterations: Hold each config for N training iterations.
        mode: ``"cycle"`` for round-robin, ``"off"`` to disable swapping.
    """

    def __init__(
        self,
        config_paths: list[str],
        swap_every_n_iterations: int = 300,
        mode: Literal["cycle", "off"] = "cycle",
    ) -> None:
        self.config_paths = config_paths
        self.swap_every_n_iterations = swap_every_n_iterations
        self.mode = mode
        self._swap_index: int = 0

        self._configs: list[_ParsedConfig] = []
        for path_str in config_paths:
            self._configs.append(self._load_config(path_str))

        self._validate_control_step_consistency()

        logger.info(
            "InfraCombinator: %d configs loaded, mode=%s, "
            "swap_every_n_iterations=%d",
            len(self._configs), self.mode, self.swap_every_n_iterations,
        )
        for i, cfg in enumerate(self._configs):
            logger.info("  [%d] %s", i, cfg.name)

    @staticmethod
    def _load_config(path_str: str) -> _ParsedConfig:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"InfraCombinator: config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        bp_dict = raw.get("building_props", {})
        return _ParsedConfig(
            name=raw.get("env_config_name", path.stem),
            infra_dicts=raw.get("infras", []),
            building_props=BuildingProps(
                mC=bp_dict.get("mC", 300),
                K=bp_dict.get("K", 20),
            ),
            control_step=raw.get("control_step", 300),
        )

    def _validate_control_step_consistency(self) -> None:
        """All configs must share ``control_step``."""
        steps = {cfg.context["control_step"] for cfg in self._configs}
        if len(steps) > 1:
            raise ValueError(
                f"InfraCombinator: all configs must share the same "
                f"control_step. Found: {sorted(steps)}"
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

    def get_building_props(self, swap_index: int) -> BuildingProps:
        """Return the BuildingProps for the config at *swap_index*."""
        return self._configs[swap_index % len(self._configs)].building_props

    def advance(self) -> bool:
        """Advance to the next config. Returns True if changed."""
        if not self.is_enabled() or len(self._configs) < 2:
            return False
        self._swap_index += 1
        return True

    def get_active_config_name(self) -> str:
        """Return the ``env_config_name`` of the currently active YAML."""
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
    def from_yaml(cls, path: str | Path) -> "InfraCombinator":
        """Load an infra schedule from a YAML config file.

        Expected format::

            mode: cycle
            swap_every_n_iterations: 300
            configs:
            - configs/env_cfg/env_test1_small.yaml
            - configs/env_cfg/env_test1_mid.yaml
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"InfraCombinator schedule file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        return cls(
            config_paths=raw.get("configs", []),
            swap_every_n_iterations=raw.get("swap_every_n_iterations", 300),
            mode=raw.get("mode", "cycle"),
        )
