"""Trial config loader.

A single trial config (``configs/trial_cfgs/<name>.yaml``) describes a complete
training or evaluation run. Every sub-config is **inlined** in the trial YAML
except ``data_schedule`` (which references descriptor YAMLs).

Schema (top-level keys, ordered):

    trial_name: <str>

    # run control
    algorithm: ppo|sac
    seed: 42
    metric: episode_return_mean | achieved_reward | reward_rate
    checkpoint_frequency_episodes: 20
    log_trajectories: false
    num_envs: 1
    grad_train: false

    # env topology — inlined
    env_meta:        {EPISODE_LENGTH: 288, control_step: 300, allow_early_termination: false}
    training_params: {common: {...}, ppo: {...}, sac: {...}}
    statesources:    [<spec>, ...]   # null when statesource_schedule is set
    infras:          [<spec>, ...]   # null when infra_schedule is set
    rewards:         [{class_name, weight, params}, ...]

    # schedules — inlined; data_schedule keeps its path-based form
    infra_schedule:        {mode, swap_every_n_episodes, configs: {train: [...], eval: [...]}} | null
    statesource_schedule:  {mode, swap_every_n_episodes, configs: {train: [...], eval: [...]}} | null
    reward_schedule:       {mode, swap_every_n_episodes, ...} | null
    data_schedule:         {train: <path>, eval: <path>}

    # standalone exploration reset
    exploration_reset:     {enabled, trigger, ...}

Mutex rules: exactly one of (``infras``, ``infra_schedule``) is non-null;
same for (``statesources``, ``statesource_schedule``).
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
from adv_building_gym.config.exploration_reset import ExplorationResetConfig
from adv_building_gym.config.reward_schedule_manager import (
    RewardScheduleManager,
    RewardScheduleMode,
)
from adv_building_gym.config.training_param_config import TrainingParamConfig
from adv_building_gym.data_combinator import DataCombinator
from adv_building_gym.infra_combinator import InfraCombinator
from adv_building_gym.statesource_combinator import StatesourceCombinator

logger = logging.getLogger(__name__)


_RUN_DEFAULTS: dict[str, Any] = {
    "algorithm": "ppo",
    "metric": "reward_rate",
    "checkpoint_frequency_episodes": 20,
    "log_trajectories": False,
    "num_envs": 1,
    "grad_train": False,
    "overrides": None,
}

# Required top-level keys (always present, possibly None for nullable ones).
_REQUIRED = (
    "trial_name", "seed", "env_meta", "training_params", "rewards",
)


@dataclass
class TrialConfig:
    """Bundle of fully-loaded run + sub-configs for a trial."""

    # run control
    trial_name: str
    algorithm: str
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
    statesource_combinator: Optional[StatesourceCombinator]
    exploration_reset: ExplorationResetConfig

    raw: dict[str, Any] = field(default_factory=dict)
    source_path: Optional[Path] = None

    @staticmethod
    def load(
        trial_yaml_path: str | Path,
        *,
        require_data_schedule: bool = True,
        is_training: bool = True,
    ) -> "TrialConfig":
        """Parse a trial YAML and resolve every referenced sub-config."""
        path = Path(trial_yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Trial config not found: {path}")
        with path.open("r") as f:
            trial_dict = yaml.safe_load(f) or {}
        return TrialConfig._from_dict(
            trial_dict,
            source_path=path,
            require_data_schedule=require_data_schedule,
            is_training=is_training,
        )

    @staticmethod
    def _from_dict(
        trial_dict: dict[str, Any],
        *,
        source_path: Optional[Path] = None,
        require_data_schedule: bool = True,
        is_training: bool = True,
    ) -> "TrialConfig":
        label = source_path.name if source_path else "<inline>"
        for key in _REQUIRED:
            if key not in trial_dict:
                raise ValueError(f"Trial config {label} missing required key '{key}'")

        run = {**_RUN_DEFAULTS, **{k: v for k, v in trial_dict.items() if k in _RUN_DEFAULTS}}
        if run["algorithm"] not in ("ppo", "sac"):
            raise ValueError(f"Trial config {label}: algorithm must be 'ppo' or 'sac', got '{run['algorithm']}'")

        trial_seed: int = int(trial_dict["seed"])
        overrides = run["overrides"] or {}

        # ---- training params (inlined) ----
        tparam_doc = trial_dict["training_params"]
        if not isinstance(tparam_doc, dict):
            raise ValueError(
                f"Trial config {label}: 'training_params' must be inlined (dict), "
                f"got {type(tparam_doc).__name__}"
            )
        training_param_config = TrainingParamConfig.from_dict(
            tparam_doc, default_seed=trial_seed, source_label=label,
        )
        _apply_overrides_dataclass(training_param_config, overrides.get("training_params"))

        # ---- env topology mutex validation ----
        infras_inline = trial_dict.get("infras")
        infra_sch_inline = trial_dict.get("infra_schedule")
        ss_inline = trial_dict.get("statesources")
        ss_sch_inline = trial_dict.get("statesource_schedule")

        _validate_mutex(label, "infras", infras_inline, "infra_schedule", infra_sch_inline)
        _validate_mutex(label, "statesources", ss_inline, "statesource_schedule", ss_sch_inline)

        env_meta_doc = trial_dict["env_meta"]
        if not isinstance(env_meta_doc, dict):
            raise ValueError(f"Trial config {label}: 'env_meta' must be inlined (dict)")

        # When infras / statesources are deferred to a schedule, seed the
        # env config with empty lists; the swap callback will set_infras /
        # set_statesources before training begins.
        env_config = EnvConfigManager.from_dict(
            infras_doc={"infras": list(infras_inline)} if infras_inline else {"infras": []},
            statesources_doc={"statesources": list(ss_inline)} if ss_inline else {"statesources": []},
            env_meta_doc=env_meta_doc,
        )

        # ---- data combinator (path-based train/eval) ----
        data_combinator: Optional[DataCombinator] = None
        data_schedule = trial_dict.get("data_schedule")
        if data_schedule:
            split = "train" if is_training else "eval"
            data_schedule_path = data_schedule.get(split)
            if not data_schedule_path:
                raise ValueError(
                    f"Trial config {label}: data_schedule.{split} not set"
                )
            data_combinator = load_data_combinator_config(
                cfg_yaml_path=data_schedule_path, default_seed=trial_seed,
            )
        elif require_data_schedule:
            raise ValueError(
                f"Trial config {label} missing 'data_schedule' (required for training)"
            )

        # ---- rewards (required) + optional reward schedule (inlined) ----
        rewards_pool = trial_dict["rewards"]
        if not isinstance(rewards_pool, list) or not rewards_pool:
            raise ValueError(
                f"Trial config {label}: 'rewards' must be a non-empty list"
            )
        reward_sch_cfg = trial_dict.get("reward_schedule")
        if reward_sch_cfg is not None and not isinstance(reward_sch_cfg, dict):
            raise ValueError(
                f"Trial config {label}: 'reward_schedule' must be inlined (dict) or null"
            )
        reward_manager = RewardScheduleManager.from_dict(
            rewards=rewards_pool,
            reward_sch_cfg=reward_sch_cfg,
            default_seed=trial_seed,
        )
        if reward_sch_cfg and not run["grad_train"]:
            reward_manager.mode = RewardScheduleMode.OFF
            logger.info(
                "grad_train=False — reward schedule mode forced to OFF "
                "(all configured rewards active from start)",
            )
        elif reward_sch_cfg:
            logger.info(
                "grad_train=True — reward schedule mode=%s, swap every %d episodes",
                reward_manager.mode, reward_manager.swap_every_n_episodes,
            )

        # ---- infra schedule (inlined) ----
        infra_combinator: Optional[InfraCombinator] = None
        if infra_sch_inline:
            if not isinstance(infra_sch_inline, dict):
                raise ValueError(
                    f"Trial config {label}: 'infra_schedule' must be a dict (inlined)"
                )
            infra_combinator = InfraCombinator.from_dict(
                infra_sch_inline, control_step=env_config.CONTROL_STEP,
            )
            # Seed env_config with the first train (or eval, when not training) entry
            # so env startup works before the first hot-swap.
            seed_split = "train" if is_training else "eval"
            seed_cfgs = infra_combinator._configs[seed_split]
            if seed_cfgs:
                env_config.infra_specs = list(seed_cfgs[0].infra_dicts)
            logger.info(
                "Infra schedule enabled: mode=%s, %d train / %d eval configs, "
                "swap every %d episodes",
                infra_combinator.mode,
                infra_combinator.train_count(), infra_combinator.eval_count(),
                infra_combinator.swap_every_n_episodes,
            )

        # ---- statesource schedule (inlined) ----
        statesource_combinator: Optional[StatesourceCombinator] = None
        if ss_sch_inline:
            if not isinstance(ss_sch_inline, dict):
                raise ValueError(
                    f"Trial config {label}: 'statesource_schedule' must be a dict (inlined)"
                )
            statesource_combinator = StatesourceCombinator.from_dict(
                ss_sch_inline, control_step=env_config.CONTROL_STEP,
            )
            seed_split = "train" if is_training else "eval"
            seed_cfgs = statesource_combinator._configs[seed_split]
            if seed_cfgs:
                env_config.statesource_specs = list(seed_cfgs[0].statesource_dicts)
            logger.info(
                "Statesource schedule enabled: mode=%s, %d train / %d eval configs, "
                "swap every %d episodes",
                statesource_combinator.mode,
                statesource_combinator.train_count(), statesource_combinator.eval_count(),
                statesource_combinator.swap_every_n_episodes,
            )

        # ---- multi-axis eval guard ----
        if not is_training and infra_combinator is not None and statesource_combinator is not None:
            if infra_combinator.eval_count() > 1 and statesource_combinator.eval_count() > 1:
                raise ValueError(
                    f"Trial config {label}: cannot iterate both "
                    "infra_schedule.configs.eval and statesource_schedule.configs.eval "
                    "during evaluation; choose one axis."
                )

        # ---- exploration reset (standalone) ----
        exploration_reset = ExplorationResetConfig.from_dict(trial_dict.get("exploration_reset"))

        trial = TrialConfig(
            trial_name=trial_dict["trial_name"],
            algorithm=run["algorithm"],
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
            statesource_combinator=statesource_combinator,
            exploration_reset=exploration_reset,
            raw=trial_dict,
            source_path=source_path,
        )
        logger.info(
            "Loaded trial '%s' from %s [algorithm=%s, seed=%s, episodes=%s, metric=%s]",
            trial.trial_name, source_path or "<inline>", trial.algorithm,
            trial.seed, training_param_config.max_episodes_to_run, trial.metric,
        )
        return trial


def _validate_mutex(
    label: str,
    inline_key: str,
    inline_value: Any,
    schedule_key: str,
    schedule_value: Any,
) -> None:
    has_inline = inline_value not in (None, [], {})
    has_schedule = schedule_value not in (None, [], {})
    if has_inline and has_schedule:
        raise ValueError(
            f"Trial config {label}: '{inline_key}' and '{schedule_key}' are "
            f"mutually exclusive — set exactly one."
        )
    if not has_inline and not has_schedule:
        raise ValueError(
            f"Trial config {label}: at least one of '{inline_key}' / "
            f"'{schedule_key}' must be set."
        )


def _apply_overrides_dataclass(target: Any, overrides: Any) -> None:
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
