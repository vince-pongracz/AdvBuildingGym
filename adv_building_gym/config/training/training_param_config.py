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
class EvaluationConfig:
    """Evaluation cadence shared by both drivers."""

    # Eval cadence in iterations (RLlib evaluation_interval); checkpoints align to it
    # so each lands on a fresh-eval iteration (ranked by eval return).
    interval: int = 10
    # Episodes per evaluation round (RLlib evaluation_duration, unit=episodes; also the
    # SB driver's n_eval_episodes), so both drivers average over the same number of episodes.
    duration: int = 10


@dataclass
class EarlyStoppingConfig:
    """Early stopping on the held-out eval metric, in episode units. 
    Disabled unless ``patience_episodes > 0``. 
    A running (EMA/Polyak) average of the eval scores is kept;
    stop when no fresh eval beats that average (by > ``min_delta``) for ``patience_episodes``
    episodes. Because the eval metric only refreshes each eval round, set ``patience_episodes``
    to span several rounds. See adv_building_gym/_common/early_stopping.py.
    """

    patience_episodes: int = 0
    min_delta: float = 0.0
    grace_episodes: int = 0
    # EMA span in eval rounds (alpha = 2/(smoothing_window+1)); also the warmup sample count.
    smoothing_window: int = 30


@dataclass
class TrainingParamConfig(LoggableConfig):
    """Training hyperparameters, split by algorithm where semantics differ.

    PPO (on-policy): batch in episodes (``ppo_episodes_per_iteration`` × episode_length
    timesteps), SGD over ``ppo_minibatch_size`` mini-batches.
    SAC (off-policy): samples ``sac_replay_batch_size`` transitions per gradient step.
    ``sac_training_intensity`` sets UTD ≈ intensity / batch_size (e.g. 128 with batch 256 → UTD≈0.5).
    """

    episode_lookback_horizon_steps: int = 120
    seed: int = 42
    max_episodes_to_run:int = 10000
    gamma: float = 0.99
    clip_actions_to_env_bounds: bool = True
    # When True, num_learners=0 → Learner runs in the driver (no remote actor),
    # freeing one CPU for an extra EnvRunner.
    local_learner: bool = True
    # Rolling window for episode_return_mean smoothing (RLlib metrics_num_episodes_for_smoothing
    # and the SB3 best-by-metric deque), so both drivers score over the same recent episodes.
    episode_return_mean_window: int = 30
    # Eval cadence (interval in iterations, duration in episodes), shared by both drivers.
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    # Early stopping on the held-out eval metric (disabled unless patience_episodes > 0).
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

    # Per-algorithm rollout_fragment_length override 
    # (steps each EnvRunner collects per sample call). 
    # None → fall back to EPISODE_LENGTH (the codebase invariant, see
    # ray/callbacks/_swap_trigger.py). Set only to deviate from full-episode fragments.
    ppo_rollout_length: int | None = None
    sac_rollout_length: int | None = None
    dreamerv3_rollout_length: int | None = None

    ppo_episodes_per_iteration: int = 25
    ppo_minibatch_size: int = 128 # Rllib default
    ppo_num_epochs: int = 10

    sac_replay_batch_size: int = 256 # Rllib default
    sac_episodes_to_keep_in_replay_buffer: int = 100
    sac_training_intensity: float | None = None
    sac_n_step_return: int = 1
    sac_learning_starts_after_n_episodes: int = 50 # Warm up replay buffer with 50 episodes before learning starts.

    # DreamerV3 (model-based, off-policy): world model trained on batch_length_T sequences
    # from the replay buffer; actor/critic on imagined horizon_H rollouts; training_ratio = UTD analogue.
    # Link: https://arxiv.org/pdf/2301.04104
    dreamerv3_model_size: str = "XS"  # one of "XS", "S", "M", "L", "XL"
    dreamerv3_batch_size_B: int = 16  # RLlib default
    dreamerv3_batch_length_T: int = 64  # RLlib default
    dreamerv3_training_ratio: float = 10.0 # Not RLlib default, default is 1024
    dreamerv3_horizon_H: int = 15  # imagination horizon. RLlib default
    dreamerv3_episodes_to_keep_in_replay_buffer: int = 500
    dreamerv3_world_model_lr: float = 1e-4  # RLlib default
    dreamerv3_actor_lr: float = 3e-5  # RLlib default
    dreamerv3_critic_lr: float = 3e-5  # RLlib default

    _source_file: str | None = field(default=None, repr=False)

    @staticmethod
    def from_dict(doc: dict, default_seed: int | None = None, *, source_label: str = "<inline>") -> "TrainingParamConfig":
        """Build from an already-parsed dict (mirrors from_yaml's prefix flattening)."""
        tparam_cfg = dict(doc) if doc else {}
        flat: dict = {}
        for cfg_section in ("common", "ppo", "sac", "dreamerv3"):
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
        # ``common.evaluation`` / ``common.early_stop`` arrive as nested dicts; build their
        # dataclasses (an unknown sub-key raises TypeError loudly). Absent → dataclass default.
        eval_section = flat.pop("evaluation", None)
        if eval_section is not None:
            flat["evaluation"] = EvaluationConfig(**eval_section)
        early_stop_section = flat.pop("early_stop", None)
        if early_stop_section is not None:
            flat["early_stopping"] = EarlyStoppingConfig(**early_stop_section)
        config = TrainingParamConfig(**flat)
        config._source_file = source_label
        # config.log_values()  # Uncomment if log_values is a method
        return config

    @staticmethod
    def from_yaml(path: str | Path, default_seed: int | None = None) -> "TrainingParamConfig":
        """Load training config from a YAML file.

        Sections ``common``/``ppo``/``sac`` are flattened with the algorithm prefix
        (``ppo.episodes_per_iteration`` → ``ppo_episodes_per_iteration``). The nested
        ``common.evaluation`` / ``common.early_stop`` blocks build ``EvaluationConfig`` /
        ``EarlyStoppingConfig``. Seed: YAML's ``common.seed`` wins, else ``default_seed``.
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
