"""Trial config loader.

A single trial config (``configs/trial_cfgs/<name>.yaml``) describes a complete
training or evaluation run. Every sub-config is **inlined** in the trial YAML
except ``data_schedule`` (which references descriptor YAMLs).

Schema (top-level keys, ordered):

    trial_name: <str>

    # run control
    algorithm: ppo|sac|dreamerv3
    seed: 42        # LEARNER axis: NN init, exploration noise, global torch/numpy RNGs.
    env_seed: 42    # optional; ENV axis: data variant / episode day / per-episode component
                    # draws of the TRAINING envs. Defaults to `seed`. Pin it while varying
                    # `seed` to sweep network init over an identical data traversal.
    eval_seed: 42   # optional; ENV axis of the IN-TRAINING EVAL env runner. Defaults to
                    # `env_seed`. Pin it while varying `seed` to run a seed sweep against an
                    # identical eval episode sequence (see adv_building_gym/ray/env_creator.py).
    metric: episode_return_mean | achieved_reward
    # Eval + checkpoint cadence is a single knob: training_params.common.evaluation.interval
    # (checkpoints are taken on eval iterations; see run_train_ray._build_tuner).
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

from adv_building_gym._common.lifecycle import Reloadable
from adv_building_gym.components.registry import ComponentRegistry
from adv_building_gym.config.data.data_config import load_data_combinator_config
from adv_building_gym.config.env.env_config import EnvConfig
from adv_building_gym.config.env.env_config_manager import EnvConfigManager
from adv_building_gym.config.training.exploration_reset import ExplorationResetConfig
from adv_building_gym.config.rewards.reward_schedule_manager import (
    RewardScheduleManager,
    RewardScheduleMode,
)
from adv_building_gym.config.training.training_param_config import TrainingParamConfig
from adv_building_gym.config.data.data_combinator import DataCombinator
from adv_building_gym.config.env.infra_combinator import InfraCombinator
from adv_building_gym.config.env.statesource_combinator import StatesourceCombinator

logger = logging.getLogger(__name__)


_RUN_DEFAULTS: dict[str, Any] = {
    "algorithm": "ppo",
    "env_seed": None,  # None → fall back to the seed
    "eval_seed": None, # None → fall back to the env_seed
    "metric": "episode_return_mean",
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
    # Env-stochasticity seed (data variant / day / per-episode component draws) of the
    # training envs; equals `seed` unless the YAML sets it. Independent of the learner seed.
    env_seed: int
    # Same axis for the in-training eval env only; equals `env_seed` unless the YAML sets it.
    eval_seed: int
    metric: str
    log_trajectories: bool
    num_envs: int
    grad_train: bool

    # loaded sub-configs
    training_param_config: TrainingParamConfig
    env_config: EnvConfig
    data_combinator: Optional[DataCombinator]
    # Held-out eval-split combinator, loaded only when is_training=True so the
    # in-training evaluation rounds run on the eval dataset (see env_creator).
    eval_data_combinator: Optional[DataCombinator]
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
        if run["algorithm"] not in ("ppo", "sac", "dreamerv3"):
            raise ValueError(f"Trial config {label}: algorithm must be 'ppo', 'sac', or 'dreamerv3', got '{run['algorithm']}'")

        trial_seed: int = int(trial_dict["seed"])
        # Two independent axes. `seed` is the LEARNER axis (NN init, exploration, global
        # torch/numpy RNGs). `env_seed` is the ENV axis: every env is seeded exactly once, at
        # construction by the env creators, and AdvBuildingGym._maybe_reseed then ignores the
        # framework's own (learner-derived) reset seeds — so sweeping `seed` with `env_seed`
        # pinned varies network init over an identical data traversal.
        env_seed: int = trial_seed if run["env_seed"] is None else int(run["env_seed"])
        # `eval_seed` isolates the in-training eval env from a seed sweep the same way; it
        # follows the env axis, not the learner axis.
        eval_seed: int = env_seed if run["eval_seed"] is None else int(run["eval_seed"])
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

        # deferred to a schedule → seed empty lists; swap callback sets them before training
        env_config = EnvConfigManager.from_dict(
            infras_doc={"infras": list(infras_inline)} if infras_inline else {"infras": []},
            statesources_doc={"statesources": list(ss_inline)} if ss_inline else {"statesources": []},
            env_meta_doc=env_meta_doc,
        )

        # ---- rewards (required) + optional reward schedule (inlined) ----
        rewards_pool = trial_dict["rewards"]
        if not isinstance(rewards_pool, list) or not rewards_pool:
            raise ValueError(f"Trial config {label}: 'rewards' must be a non-empty list")
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

        # ---- data combinator (path-based train/eval) ----
        # Built after the statesource axis is known: variant keys with no consuming
        # reloadable statesource are excluded from the pool, so a shared data-schedule
        # YAML cannot inflate the pool with combinations the trial never loads.
        # With a statesource schedule, the per-split union keeps an axis alive if ANY
        # scheduled config consumes it (matching the swap-through-configs semantics).
        if statesource_combinator is not None:
            active_sources: dict[str, frozenset[str]] = {
                split: _reloadable_statesource_names(
                    [spec for cfg_specs in statesource_combinator.spec_dicts(split) for spec in cfg_specs]
                )
                for split in ("train", "eval")
            }
        else:
            inline_names = _reloadable_statesource_names(env_config.statesource_specs)
            active_sources = {"train": inline_names, "eval": inline_names}

        data_combinator: Optional[DataCombinator] = None
        eval_data_combinator: Optional[DataCombinator] = None
        data_schedule = trial_dict.get("data_schedule")
        if data_schedule:
            split = "train" if is_training else "eval"
            data_schedule_path = data_schedule.get(split)
            if not data_schedule_path:
                raise ValueError(f"Trial config {label}: data_schedule.{split} not set")
            # env axis: the combinator's seed only shuffles the variant pool, which is part of
            # the data traversal — it must follow `env_seed`, not the learner seed.
            data_combinator = load_data_combinator_config(
                cfg_yaml_path=data_schedule_path, default_seed=env_seed,
                active_source_names=active_sources[split],
            )
            # During training, also build the eval-split combinator so the in-training
            # evaluation rounds sample from the held-out eval dataset (wired to the eval
            # EnvRunners by the env creator, gated on eval_mode). When is_training=False
            # the standalone eval driver already loads the eval split as data_combinator.
            if is_training:
                eval_schedule_path = data_schedule.get("eval")
                if eval_schedule_path:
                    eval_data_combinator = load_data_combinator_config(
                        cfg_yaml_path=eval_schedule_path, default_seed=eval_seed,
                        active_source_names=active_sources["eval"],
                    )
                else:
                    logger.warning(
                        "Trial config %s: data_schedule.eval not set — in-training "
                        "evaluation falls back to the training dataset.", label,
                    )
        elif require_data_schedule:
            raise ValueError(f"Trial config {label} missing 'data_schedule' (required for training)")

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
            env_seed=env_seed,
            eval_seed=eval_seed,
            metric=run["metric"],
            log_trajectories=bool(run["log_trajectories"]),
            num_envs=int(run["num_envs"]),
            grad_train=bool(run["grad_train"]),
            training_param_config=training_param_config,
            env_config=env_config,
            data_combinator=data_combinator,
            eval_data_combinator=eval_data_combinator,
            reward_manager=reward_manager,
            infra_combinator=infra_combinator,
            statesource_combinator=statesource_combinator,
            exploration_reset=exploration_reset,
            raw=trial_dict,
            source_path=source_path,
        )
        logger.info(
            "Loaded trial '%s' from %s [algorithm=%s, seed(learner)=%s, env_seed=%s%s, "
            "eval_seed=%s%s, episodes=%s, metric=%s]",
            trial.trial_name, source_path or "<inline>", trial.algorithm,
            trial.seed,
            trial.env_seed, "" if run["env_seed"] is not None else " (inherited from seed)",
            trial.eval_seed, "" if run["eval_seed"] is not None else " (inherited from env_seed)",
            training_param_config.max_episodes_to_run, trial.metric,
        )
        return trial


def _reloadable_statesource_names(specs: list[dict]) -> frozenset[str]:
    """Names of the configured statesources whose class is ``Reloadable``.

    Statically mirrors what ``ReloadDispatcher.register_all`` does at runtime with
    the instantiated sources (``isinstance(source, Reloadable)``), so the data
    combinator can exclude variant keys no configured statesource would consume.
    Only the trial's spec dicts count — a class merely registered in the
    ``ComponentRegistry`` contributes nothing. Unresolvable class names are kept
    (never silently drop data on a lookup error).
    """
    names: set[str] = set()
    for spec in specs:
        name = spec.get("name")
        if not name:
            continue
        class_name = spec.get("class")
        try:
            statesource_class = ComponentRegistry.get("statesource", class_name)
        except ValueError:
            logger.warning(
                "Statesource class %r not in ComponentRegistry — keeping %r in the "
                "active data-source set.", class_name, name,
            )
            names.add(name)
            continue
        if issubclass(statesource_class, Reloadable):
            names.add(name)
    return frozenset(names)


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
