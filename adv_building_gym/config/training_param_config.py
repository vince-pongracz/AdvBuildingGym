"""Training hyperparameter configuration.

Provides a dataclass that holds training hyperparameters and can be loaded
from a YAML file.
Fields are split by algorithm where their semantics differ.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from adv_building_gym.config.utils.loggable_config import LoggableConfig

import yaml

logger = logging.getLogger(__name__)

@dataclass
class TrainingParamConfig(LoggableConfig):
    """Training hyperparameters, split by algorithm where semantics differ.

    PPO (on-policy) collects a batch of complete episodes before each policy
    update.  The batch size is therefore expressed in episodes
    (``ppo_episodes_per_iteration``), converted to timesteps via
    ``ppo_episodes_per_iteration × episode_length`` in ``select_model``.
    Within each update, SGD iterates over mini-batches of
    ``ppo_minibatch_size`` timesteps.

    SAC (off-policy) stores all experience in a replay buffer and samples
    ``sac_replay_batch_size`` transitions per gradient step, independent of
    episode boundaries.

    Attributes:
        learning_rate: Learning rate for optimiser(s).
        ppo_episodes_per_iteration: How many full episodes PPO collects
            before one policy update (on-policy batch, in episode units).
        ppo_minibatch_size: SGD mini-batch size within each PPO epoch
            (in timesteps).
        sac_replay_batch_size: Number of transitions sampled from the
            replay buffer per SAC gradient step.
        sac_training_intensity: Ratio of replayed steps to sampled steps.
            Controls how many gradient updates SAC performs per sampling
            round.  Standard UTD ≈ training_intensity / batch_size.
            With batch_size=256 and rollout_fragment_length=288 (3 workers),
            training_intensity=128 gives UTD≈0.5 (432 gradient steps
            per iteration instead of 1).
            Link: https://arxiv.org/abs/1802.09477
    """

    learning_rate: float = 3e-4
    episode_lookback_horizon_steps: int = 120
    seed: int = 42
    max_episodes_to_run:int = 10000
    clip_actions_to_env_bounds: bool = True
    
    ppo_episodes_per_iteration: int = 25
    ppo_minibatch_size: int = 128 # Rllib default
    ppo_num_epochs: int = 20
    
    sac_replay_batch_size: int = 256 # NOTE VP 2026.05.08.: Rllib default
    sac_episodes_to_keep_in_replay_buffer: int = 100
    sac_training_intensity: float | None = None
    sac_n_step_return: int = 1
    sac_learning_starts_after_n_episodes: int = 50 # Warm up replay buffer with 50 episodes before learning starts.

    # Per-key strided history (see docs/hst_mgmt.md).
    # tracked_keys: obs-space keys fed to StridedHistoryConnector; each
    # tracked key's obs entry is replaced with a stack along the offsets.
    # For action history, list the "<action_key>_prev" obs entries the env
    # publishes each step — those are plain obs keys.
    # offsets: step lags (negative = past). Offset 0 is prepended automatically
    # so the current step is always the first slice of the stack.
    # Empty tracked_keys disables the connector (flatten-only pipeline).
    hst_tracked_keys: list[str] = field(default_factory=list)
    hst_offsets: list[int] = field(default_factory=list)

    _source_file: str | None = field(default=None, repr=False)

    @staticmethod
    def from_dict(doc: dict, default_seed: int | None = None, *, source_label: str = "<inline>") -> "TrainingParamConfig":
        """Build from an already-parsed dict (mirrors from_yaml's prefix flattening)."""
        tparam_cfg = dict(doc) if doc else {}
        flat: dict = {}
        for cfg_section in ("common", "ppo", "sac", "hst"):
            section_data = tparam_cfg.pop(cfg_section, {}) or {}
            section_prefix = "" if cfg_section == "common" else f"{cfg_section}_"
            for key, value in section_data.items():
                flat[f"{section_prefix}{key}"] = value
        if "seed" not in flat:
            if default_seed is None:
                raise ValueError(
                    f"TrainingParamConfig {source_label} omits 'common.seed' "
                    f"and no default_seed was supplied"
                )
            flat["seed"] = default_seed
        config = TrainingParamConfig(**flat)
        config._source_file = source_label
        config.log_values()
        return config

    @staticmethod
    def from_yaml(path: str | Path, default_seed: int | None = None) -> "TrainingParamConfig":
        """Load training config from a YAML file.

        The YAML is organised into ``common``, ``ppo``, and ``sac`` sections.
        Keys inside ``ppo`` / ``sac`` are prefixed with the algorithm name
        (e.g. ``ppo.episodes_per_iteration`` → ``ppo_episodes_per_iteration``)
        before being passed to the dataclass constructor.

        Seed resolution: if ``common.seed`` is present in the YAML it wins;
        otherwise ``default_seed`` is used (typically the trial seed). This
        makes the trial config the single source of truth while still
        letting a training-params YAML opt out with its own seed.

        Args:
            path: Path to the YAML config file.
            default_seed: Fallback seed when the YAML omits ``common.seed``.

        Returns:
            TrainingParamConfig populated from the file.
        """
        with open(path, "r") as cfg_file:
            tparam_cfg = yaml.safe_load(cfg_file)
        return TrainingParamConfig.from_dict(
            tparam_cfg, default_seed=default_seed, source_label=Path(path).name
        )

    def _log_label(self) -> str:
        if self._source_file:
            return f"TrainingParamConfig (src file: {self._source_file})"
        return "TrainingParamConfig"
