"""Trial config loader.

A single trial config (``configs/trial_cfgs/<name>.yaml``) describes a complete
training or evaluation run: env topology, training hyperparameters, data /
reward / infra schedules, plus run-control fields (algorithm, episodes, seed,
metric, etc.).

``TrialConfig.load(path)`` parses the trial YAML and resolves every referenced
sub-config in one upfront pass, returning a ``TrialConfig`` bundle. Downstream
modules (env creator, RLlib config builders, callbacks) receive already-loaded
objects — they never re-parse YAML.

Trial YAML schema::

    trial_name: <str>                       # used for run dir naming

    # algorithm + run control
    algorithm: ppo                          # ppo | sac
    episodes: 3500                          # overrides training_params.max_episodes_to_run
    seed: 42                                # default seed for every sub-config that
                                            # doesn't pin its own (single source of truth)
    metric: reward_rate                     # episode_return_mean | achieved_reward | reward_rate
    checkpoint_frequency_episodes: 250
    log_trajectories: false
    num_envs: 1
    grad_train: false                       # honour reward schedule mode (else forced OFF)

    # env topology (replaces the old configs/env/*.yaml wrapper).  The
    # ``trial_name`` above is what gets used as the run identifier
    # (checkpoint dir = ``models/<trial_name>/<algo>/...``).
    infras:        configs/infra_cfgs/test1_small.yaml
    statesources:  configs/statesource_cfgs/default.yaml
    env_meta:      configs/env_meta/default.yaml

    # rewards: which rewards exist (required)
    rewards: configs/reward_cfgs/rewards.yaml

    # training params and schedules
    training_params: configs/training_param_cfgs/default.yaml
    data_schedule:   configs/schedules/data/train.yaml
    # reward_schedule: how the rewards above are scheduled (optional —
    # omit / null for "all rewards live, no swapping")
    reward_schedule: configs/schedules/reward/train.yaml
    infra_schedule:  null                   # or configs/schedules/infra/train.yaml

    # optional inline overrides (deep-merged onto the referenced YAMLs)
    overrides:
      training_params:
        common: { learning_rate: 1.0e-4 }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from adv_building_gym.config.data_config import load_data_combinator_config
from adv_building_gym.config.env_config import EnvConfig
from adv_building_gym.config.env_config_manager import EnvConfigManager
from adv_building_gym.config.reward_schedule_manager import (
    RewardScheduleManager,
    RewardScheduleMode,
)
from adv_building_gym.config.training_param_config import TrainingParamConfig
from adv_building_gym.data_combinator import DataCombinator
from adv_building_gym.infra_combinator import InfraCombinator

logger = logging.getLogger(__name__)


# Run-control keys with sane defaults if a trial cfg omits them.
_RUN_DEFAULTS: dict[str, Any] = {
    "algorithm": "ppo",
    "episodes": None,
    "metric": "reward_rate",
    "checkpoint_frequency_episodes": 20,
    "log_trajectories": False,
    "num_envs": 1,
    "grad_train": False,
    "infra_schedule": None,
    "data_schedule": None,
    "reward_schedule": None,
    "overrides": None,
}

# Required top-level keys.
_REQUIRED = (
    "trial_name", "seed", "infras", "statesources", "env_meta",
    "training_params", "rewards",
)


@dataclass
class TrialConfig:
    """Bundle of fully-loaded run + sub-configs for a trial."""

    # run control
    trial_name: str
    algorithm: str
    episodes: Optional[int]
    seed: Optional[int]
    metric: str
    checkpoint_frequency_episodes: int
    log_trajectories: bool
    num_envs: int
    grad_train: bool

    # loaded sub-configs
    training_param_config: TrainingParamConfig
    env_config: EnvConfig
    data_combinator: Optional[DataCombinator]
    reward_manager: RewardScheduleManager
    infra_combinator: Optional[InfraCombinator]

    # raw trial dict (kept for diagnostics)
    raw: dict[str, Any] = field(default_factory=dict)
    source_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Loader
    # ------------------------------------------------------------------

    @staticmethod
    def load(
        trial_yaml_path: str | Path,
        *,
        require_data_schedule: bool = True,
    ) -> "TrialConfig":
        """Parse a trial YAML and resolve every referenced sub-config.

        Args:
            trial_yaml_path: Path to ``configs/trial_cfgs/<name>.yaml``.
            require_data_schedule: If True, ``data_schedule`` must be present.
                Eval contexts may pass False to allow no data combinator.
        """
        path = Path(trial_yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Trial config not found: {path}")

        with path.open("r") as f:
            raw = yaml.safe_load(f) or {}

        for key in _REQUIRED:
            if key not in raw:
                raise ValueError(
                    f"Trial config {path.name} missing required key '{key}'"
                )

        # Resolve run-control with defaults
        run = {**_RUN_DEFAULTS, **{k: v for k, v in raw.items() if k in _RUN_DEFAULTS}}
        if run["algorithm"] not in ("ppo", "sac"):
            raise ValueError(
                f"Trial config {path.name}: algorithm must be 'ppo' or 'sac', got '{run['algorithm']}'"
            )

        # Trial seed is the single source of truth.  Each sub-loader receives
        # it as the fallback default; sub-config YAMLs only need to declare
        # ``seed:`` if they deliberately diverge.
        trial_seed: int = int(raw["seed"])
        overrides = run["overrides"] or {}

        # ---- training params ----
        training_param_config = TrainingParamConfig.from_yaml(
            raw["training_params"], default_seed=trial_seed,
        )
        _apply_overrides_dataclass(training_param_config, overrides.get("training_params"))

        if run["episodes"] is None:
            run["episodes"] = training_param_config.max_episodes_to_run

        # ---- env topology ----
        env_config = EnvConfigManager.load(
            infras_path=raw["infras"],
            statesources_path=raw["statesources"],
            env_meta_path=raw["env_meta"],
        )

        # ---- data combinator ----
        data_combinator: Optional[DataCombinator] = None
        if run["data_schedule"]:
            data_combinator = load_data_combinator_config(
                cfg_yaml_path=run["data_schedule"],
                default_seed=trial_seed,
            )
        elif require_data_schedule:
            raise ValueError(
                f"Trial config {path.name} missing 'data_schedule' (required for training)"
            )

        # ---- rewards (required) + optional reward schedule ----
        # Rewards file (required) declares which rewards exist; the optional
        # schedule controls how/when they are swapped during training.  When
        # no schedule is given, mode defaults to OFF (all rewards live).
        reward_manager = RewardScheduleManager.load(
            rewards_path=raw["rewards"],
            schedule_path=run["reward_schedule"],
            default_seed=trial_seed,
        )
        if run["reward_schedule"] and not run["grad_train"]:
            reward_manager.mode = RewardScheduleMode.OFF
            logger.info(
                "grad_train=False — reward schedule mode forced to OFF "
                "(all configured rewards active from start)",
            )
        elif run["reward_schedule"]:
            logger.info(
                "grad_train=True — reward schedule mode=%s, swap every %d iterations",
                reward_manager.mode, reward_manager.swap_every_n_iterations,
            )

        # ---- infra schedule (optional) ----
        infra_combinator: Optional[InfraCombinator] = None
        if run["infra_schedule"]:
            infra_combinator = InfraCombinator.from_yaml(
                run["infra_schedule"], control_step=env_config.CONTROL_STEP,
            )
            logger.info(
                "Infra schedule enabled: mode=%s, %d configs, swap every %d iterations",
                infra_combinator.mode, len(infra_combinator.config_paths),
                infra_combinator.swap_every_n_iterations,
            )

        trial = TrialConfig(
            trial_name=raw["trial_name"],
            algorithm=run["algorithm"],
            episodes=run["episodes"],
            seed=trial_seed,
            metric=run["metric"],
            checkpoint_frequency_episodes=int(run["checkpoint_frequency_episodes"]),
            log_trajectories=bool(run["log_trajectories"]),
            num_envs=int(run["num_envs"]),
            grad_train=bool(run["grad_train"]),
            training_param_config=training_param_config,
            env_config=env_config,
            data_combinator=data_combinator,
            reward_manager=reward_manager,
            infra_combinator=infra_combinator,
            raw=raw,
            source_path=path,
        )

        logger.info(
            "Loaded trial '%s' from %s "
            "[algorithm=%s, seed=%s, episodes=%s, metric=%s]",
            trial.trial_name, path, trial.algorithm, trial.seed,
            trial.episodes, trial.metric,
        )
        return trial


def _apply_overrides_dataclass(target: Any, overrides: Any) -> None:
    """Apply a possibly-nested overrides dict onto a dataclass instance.

    Sections (``common``/``ppo``/``sac``/``hst``) get prefix-flattened the
    same way TrainingParamConfig.from_yaml flattens them, so trial cfgs
    can write::

        overrides:
          training_params:
            common: {learning_rate: 1.0e-4}
            ppo:    {minibatch_size: 128}
    """
    if not overrides:
        return
    flat: dict[str, Any] = {}
    for section, data in overrides.items():
        if isinstance(data, dict):
            section_prefix = "" if section == "common" else f"{section}_"
            for key, value in data.items():
                flat[f"{section_prefix}{key}"] = value
        else:
            flat[section] = data
    for key, value in flat.items():
        if hasattr(target, key):
            setattr(target, key, value)
        else:
            logger.warning(
                "TrialConfig override key '%s' is not on %s — ignored",
                key, type(target).__name__,
            )
