
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from adv_building_gym.config.utils.loggable_config import LoggableConfig

logger = logging.getLogger(__name__)

from adv_building_gym.components.infrastructure import Infrastructure
from adv_building_gym.components.statesources import StateSource
from adv_building_gym.components.registry import from_dict as component_from_dict
from adv_building_gym.config.rewards.reward_config import RewardConfig


@dataclass
class HstConfig(LoggableConfig):
    """Per-key strided observation history (``HistoryWrapper``).

    Adds ``s_hst_<key>`` of shape ``(len(offsets), *shape)`` per tracked key; offsets ≤ 0
    (0 = current). Originals pass through. See ``core/history_wrapper.py``.
    """
    enabled: bool = False
    tracked_keys: tuple[str, ...] = ()
    offsets: tuple[int, ...] = ()

    def _log_label(self) -> str:
        return "HstConfig"


@dataclass
class ForecastConfig(LoggableConfig):
    """Forecast wrapper (``ForecastWrapper``): adds ``s_fc_<var>`` arrays of future
    statesource values at ``steps`` (positive control-step offsets). See ``core/forecast_wrapper.py``.
    """
    enabled: bool = False
    steps: tuple[int, ...] = ()

    def _log_label(self) -> str:
        return "ForecastConfig"


@dataclass
class EnvConfig(LoggableConfig):
    """Environment topology — statesources, infrastructure, physics.

    Components come from YAML via ``EnvConfigManager.from_dict``, populating
    ``infra_specs`` / ``statesource_specs``. ``create_infras`` / ``create_statesources``
    deserialise fresh instances per call — use them for env creation. ``self.infras`` /
    ``self.statesources`` are shared singletons (``init_singletons()``), for inspection only.
    Reward composition lives in ``reward_config``.
    """
    EPISODE_LENGTH: int = 288 # a day
    CONTROL_STEP: int = 300  # seconds (5 minutes)
    # NOTE VP 2026.06.08.: It's not used anymore, but keep it, in the future may be useful
    ACTION_HISTORY_LENGTH: int = 4  # rolling window of past actions kept in env for reward functions (not in obs)
    # When False, should_terminate hooks and rewards' terminal-only branches are skipped
    # (soft, non-terminating); True = terminate on hard breach.
    allow_early_termination: bool = False

    # Env-wrapper sub-configs (see core/history_wrapper.py / core/forecast_wrapper.py).
    hst: HstConfig = field(default_factory=HstConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)

    # Raw YAML component specs — source of truth for the factory methods below.
    # Envelope params (K, mC) live on BuildingHeatLoss, not here.
    infra_specs: List[Dict[str, Any]] = field(default_factory=list)
    statesource_specs: List[Dict[str, Any]] = field(default_factory=list)

    # Cached singleton instances populated by init_singletons().
    # Do NOT pass these to parallel environments — use the factory methods.
    infras: Optional[List[Infrastructure]] = None
    statesources: Optional[List[StateSource]] = None

    # Reward composition — separate config
    reward_config: RewardConfig = field(default_factory=RewardConfig)

    def _infra_context(self) -> Dict[str, Any]:
        return {"control_step": self.CONTROL_STEP}

    def _statesource_context(self) -> Dict[str, Any]:
        return {"timestep": self.CONTROL_STEP, "episode_length": self.EPISODE_LENGTH}

    def create_statesources(self) -> List[StateSource]:
        """Fresh StateSource instances from ``statesource_specs`` (new each call, parallel-safe).
        Raises RuntimeError if no specs were loaded."""
        if not self.statesource_specs:
            raise RuntimeError(
                "EnvConfig.create_statesources: no statesource_specs loaded. "
                "Statesources must be declared in a YAML file referenced by "
                "the trial config (configs/trial_cfgs/<name>.yaml → statesources)."
            )
        ctx = self._statesource_context()
        return [component_from_dict(spec, "statesource", ctx) for spec in self.statesource_specs]

    def create_infras(self) -> List[Infrastructure]:
        """Fresh Infrastructure instances from ``infra_specs`` (new each call, parallel-safe).
        Raises RuntimeError if no specs were loaded."""
        if not self.infra_specs:
            raise RuntimeError(
                "EnvConfig.create_infras: no infra_specs loaded. "
                "Infras must be declared in a YAML file referenced by "
                "the trial config (configs/trial_cfgs/<name>.yaml → infras)."
            )
        ctx = self._infra_context()
        return [component_from_dict(spec, "infrastructure", ctx) for spec in self.infra_specs]

    def __post_init__(self):
        """Lightweight post-init: leaves singletons as None to avoid CSV parsing in
        every Ray worker; call init_singletons() in the main process when needed."""

    def init_singletons(self) -> None:
        """Populate the cached singleton instances (call once in the main process before
        accessing self.infras / self.statesources). Ray workers use the factory methods instead.
        WARNING: never pass these singletons to parallel envs.
        """
        if self.statesources is None:
            self.statesources = self.create_statesources()

        if self.infras is None:
            self.infras = self.create_infras()

    def _log_label(self) -> str:
        return "EnvConfig"

    def log_values(self) -> None:
        """Log config values, showing component names instead of object repr."""

        lines = [
            f"  EPISODE_LENGTH = {self.EPISODE_LENGTH}",
            f"  CONTROL_STEP = {self.CONTROL_STEP}",
            f"  ACTION_HISTORY_LENGTH = {self.ACTION_HISTORY_LENGTH}",
            f"  allow_early_termination = {self.allow_early_termination}",
        ]
        logger.info("%s:\n%s", self._log_label(), "\n".join(lines))

        self.hst.log_values()
        self.forecast.log_values()
        self.reward_config.log_values()
