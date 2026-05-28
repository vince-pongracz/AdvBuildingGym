
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
class EnvConfig(LoggableConfig):
    """Environment topology configuration — state sources, infrastructure, and physics.

    Components are always declared in YAML (``configs/infra_cfgs/**/*.yaml`` and
    ``configs/statesource_cfgs/*.yaml``) and reach this dataclass via
    ``EnvConfigManager.from_dict(...)``, which populates ``infra_specs`` and
    ``statesource_specs`` (the raw component dicts).  The factory methods
    ``create_infras`` / ``create_statesources`` deserialise those specs into
    fresh component instances per call.

    **IMPORTANT**: Use the factory methods (``create_infras``,
    ``create_statesources``) when creating env instances to ensure each env
    gets independent component instances.  Direct access to ``self.infras`` /
    ``self.statesources`` returns shared singletons populated by
    ``init_singletons()`` and is intended for inspection only.

    Reward composition is managed by ``reward_config`` (``RewardConfig``).
    """
    EPISODE_LENGTH: int = 288 # a day
    CONTROL_STEP: int = 300  # seconds (5 minutes)
    ACTION_HISTORY_LENGTH: int = 15  # rolling window of past actions kept in env for reward functions (not in obs)
    # When False, reward should_terminate hooks are bypassed and rewards skip
    # their terminal-only branches (huge penalties / success bonuses), keeping
    # only the regular reward/penalty curves. Lets a single env config family
    # toggle between "terminate on hard breach" and "soft, non-terminating".
    allow_early_termination: bool = True

    # Per-key strided observation history applied as an env wrapper.
    # When enabled, HistoryWrapper adds an s_hst_<key> Dict entry with shape
    # (len(offsets), *original_shape) for every tracked key. Offset 0 (the
    # current step) is always prepended; remaining offsets must be negative
    # (lag in env steps). Originals pass through unchanged. See
    # envs/history_wrapper.py.
    hst_env_wrapper_enabled: bool = False
    hst_env_wrapper_tracked_keys: tuple[str, ...] = ()
    hst_env_wrapper_offsets: tuple[int, ...] = ()

    # Forecast wrapper: augments the Dict observation space with s_fc_<var>
    # arrays carrying future values of deterministic CSV-backed statesources
    # at the offsets listed in forecast_env_wrapper_steps (positive ints, in
    # control-step units). See envs/forecast_wrapper.py.
    forecast_env_wrapper_enabled: bool = False
    forecast_env_wrapper_steps: tuple[int, ...] = ()

    # Raw component specs as parsed from YAML.  Source of truth for the
    # factory methods below; never reach into hardcoded defaults.
    # Building envelope params (K, mC) live on BuildingHeatLoss directly —
    # there is no separate building_props on the env config.
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
        """Deserialise fresh StateSource instances from ``statesource_specs``.

        Each call returns NEW independent instances, safe for parallel
        environments.  Raises if no specs were loaded — every env config
        must declare its statesources via YAML.
        """
        if not self.statesource_specs:
            raise RuntimeError(
                "EnvConfig.create_statesources: no statesource_specs loaded. "
                "Statesources must be declared in a YAML file referenced by "
                "the trial config (configs/trial_cfgs/<name>.yaml → statesources)."
            )
        ctx = self._statesource_context()
        return [component_from_dict(spec, "statesource", ctx) for spec in self.statesource_specs]

    def create_infras(self) -> List[Infrastructure]:
        """Deserialise fresh Infrastructure instances from ``infra_specs``.

        Each call returns NEW independent instances, safe for parallel
        environments.  Raises if no specs were loaded — every env config
        must declare its infras via YAML.
        """
        if not self.infra_specs:
            raise RuntimeError(
                "EnvConfig.create_infras: no infra_specs loaded. "
                "Infras must be declared in a YAML file referenced by "
                "the trial config (configs/trial_cfgs/<name>.yaml → infras)."
            )
        ctx = self._infra_context()
        return [component_from_dict(spec, "infrastructure", ctx) for spec in self.infra_specs]

    def __post_init__(self):
        """Lightweight post-init — does NOT eagerly call factory methods.

        Singleton fields (infras, statesources) are left as None to
        avoid unnecessary CSV parsing in every Ray worker subprocess that
        imports this module.  Call init_singletons() explicitly in the main
        process where those fields are actually needed.
        """

    def init_singletons(self) -> None:
        """Initialise the cached singleton component instances.

        Call this once in the main process after creating / loading a Config,
        before accessing self.infras / self.statesources / self.reward_config.rewards.
        Not needed in Ray worker subprocesses — they call the factory methods
        (create_infras, create_statesources) directly via adv_building_env_creator.
        Rewards are populated by ``RewardScheduleManager`` before this is called.

        WARNING: Do not pass these singleton instances to parallel environments
        — use the factory methods instead.
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
            f"  hst_env_wrapper_enabled = {self.hst_env_wrapper_enabled}",
            f"  hst_env_wrapper_tracked_keys = {list(self.hst_env_wrapper_tracked_keys)}",
            f"  hst_env_wrapper_offsets = {list(self.hst_env_wrapper_offsets)}",
            f"  forecast_env_wrapper_enabled = {self.forecast_env_wrapper_enabled}",
            f"  forecast_env_wrapper_steps = {list(self.forecast_env_wrapper_steps)}",
        ]
        logger.info("%s:\n%s", self._log_label(), "\n".join(lines))

        self.reward_config.log_values()
