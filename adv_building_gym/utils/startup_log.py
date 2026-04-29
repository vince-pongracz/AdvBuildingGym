"""Organised startup banner for training runs.

Prints a structured block that summarises everything a reader
needs to reconstruct the run from scratch: invocation, tensorboard, eval, 
env config, training setup, env topology: infrastructure, 
statesources, rewards + schedules.

The same block is optionally written to ``<run_dir>/startup.txt`` so the
snapshot travels with the checkpoint and can be diffed against other runs.

Call once from the training script, before learning/training starts.
"""

from __future__ import annotations

import datetime
import logging
import os
import socket
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

logger = logging.getLogger("startup")

_HORIZONTAL_LINE = "=" * 70

# Each section builder returns one or more (title, body_lines) tuples.
# Numbering is applied at render time from the order of the list below, so
# reordering sections (or adding/removing one) cannot desync the labels.
Section = tuple[str, list[str]]


def _header(title: str) -> str:
    return f"{_HORIZONTAL_LINE}\n{title}\n{_HORIZONTAL_LINE}"


def _fmt_infra(infra: Any) -> str:
    """One-line summary of an Infrastructure component."""
    name = getattr(infra, "name", type(infra).__name__)
    cls = type(infra).__name__
    fields: list[str] = []
    for attr in (
        "max_power_kW", "rated_power_kW", "max_charging_kW",
        "peak_consumption_kW", "cop_heat", "cop_cool", "capacity_kWh",
    ):
        if hasattr(infra, attr):
            val = getattr(infra, attr)
            if isinstance(val, (int, float)):
                fields.append(f"{attr}={val}")
    tail = "  " + "  ".join(fields) if fields else ""
    return f"  - {name:<14} [{cls}]{tail}"


def _fmt_statesource(src: Any) -> str:
    name = getattr(src, "name", type(src).__name__)
    cls = type(src).__name__
    ds = getattr(src, "ds_path", None)
    tail = f"  ds_path={Path(ds).name}" if ds else ""
    return f"  - {name:<20} [{cls}]{tail}"


def _section_invocation(seed: int, config_path: str | None = None) -> list[Section]:
    job_id = os.environ.get("SLURM_JOB_ID", "n/a")
    host = socket.gethostname()
    cmd = " ".join(sys.argv)
    return [(
        "INVOCATION",
        [
            f"  CMD       : {cmd}",
            f"  HOST/JOB  : {host} / SLURM {job_id}",
            f"  SEED      : {seed}",
            f"  ENV CFG   : {config_path or '(not set)'}",
            f"  PYTHON    : {sys.version.split()[0]}",
        ],
    )]


def _section_tensorboard(
    experiment_path: str,
    storage_path: str,
    exec_date: datetime.datetime | None,
) -> list[Section]:
    eval_trajectories_root = os.path.abspath("ep_metrics/eval_trajectories")
    body = [
        "  Training + eval curves share the same logdir; eval metrics are",
        "  nested under evaluation/env_runners/ in the TB UI.",
        "",
        "  This run only:",
        f"    ./start_tensorboard.sh {experiment_path}",
        "",
        "  All runs for this algorithm (compare across seeds):",
        f"    ./start_tensorboard.sh {storage_path}",
        "",
        "  Eval trajectories (all runs):",
        f"    ./start_tensorboard.sh {eval_trajectories_root}",
    ]
    if exec_date is not None:
        # Matches the directory created in make_eval_state_action_cb_class
        # (ep_metrics/eval_trajectories/<YYYYmmdd_HHMMSS>/) using the same
        # exec_date stamp this run uses end-to-end.
        run_eval_trajectories_path = os.path.join(
            eval_trajectories_root, exec_date.strftime("%Y%m%d_%H%M%S"),
        )
        body.extend([
            "",
            "  Eval trajectories (this run only):",
            f"    ./start_tensorboard.sh {run_eval_trajectories_path}",
        ])
    return [("TENSORBOARD", body)]


def _section_eval(
    args: Namespace, env_config: Any, experiment_path: str, seed: int
) -> list[Section]:
    algo = args.algorithm
    load_config = getattr(args, "load_config", None) or "<env_config.yaml>"

    # Forward the reward schedule training used so the printed eval command
    # reproduces training's reward definition. run_eval_ray.py defaults
    # --data-config to the eval YAML, so we don't need to forward it here.
    # Infra schedules are training-only.
    reward_schedule = getattr(args, "reward_schedule", None)
    reward_flag = f"--reward-schedule {reward_schedule}" if reward_schedule else None

    explicit_lines = [
        "  Explicit checkpoint path (this run):",
        "    sbatch slurm_scripts/slurm_eval_ray.sh \\",
        f"        --algorithm {algo} --seed {seed} \\",
        f"        --load-config {load_config} \\",
    ]
    if reward_flag:
        explicit_lines.append(f"        {reward_flag} \\")

    explicit_lines.append(
        f"        --checkpoint {experiment_path}/best_model_ep{{checkpoint_serial}}"
    )

    auto_lines = [
        "  Auto-resolve best checkpoint for this env config:",
        "    sbatch slurm_scripts/slurm_eval_ray.sh \\",
        f"        --algorithm {algo} --seed {seed} --load-config {load_config}"
        + (" \\" if reward_flag else ""),
    ]
    if reward_flag:
        auto_lines.append(f"        {reward_flag}")

    return [(
        "EVAL",
        [
            "  Replace {checkpoint_serial} with the episode number of the",
            "  checkpoint to load (e.g. best_model_ep500_...).",
            "",
            *explicit_lines,
            "",
            *auto_lines,
        ],
    )]


def _section_env_config(env_config: Any) -> list[Section]:
    return [(
        f"ENV CONFIG  ({env_config.env_config_name})",
        [
            f"  EPISODE_LENGTH   = {env_config.EPISODE_LENGTH} steps",
            f"  CONTROL_STEP     = {env_config.CONTROL_STEP} s",
            f"  ACTION_HISTORY   = {env_config.ACTION_HISTORY_LENGTH}",
        ],
    )]


def _section_training_setup(
    args: Namespace,
    training_param_config: Any,
    slurm_resources: Any,
    run_name: str,
    experiment_path: str,
) -> list[Section]:
    algo = args.algorithm.upper()
    tpc = training_param_config
    if args.algorithm == "ppo":
        hp = (
            f"lr={tpc.learning_rate}  "
            f"episodes_per_iter={tpc.ppo_episodes_per_iteration}  "
            f"minibatch={tpc.ppo_minibatch_size}  "
            f"epochs={tpc.ppo_num_epochs}"
        )
    elif args.algorithm == "sac":
        hp = (
            f"lr={tpc.learning_rate}  "
            f"replay_batch={tpc.sac_replay_batch_size}  "
            f"days_in_buffer={tpc.sac_episodes_to_keep_in_replay_buffer}  "
            f"train_intensity={tpc.sac_training_intensity}  "
        )
    else:
        hp = f"lr={tpc.learning_rate}"

    return [(
        "TRAINING SETUP",
        [
            f"  Algorithm       : {algo}  (new API stack)",
            f"  Hyperparams     : {hp}",
            f"  Episodes        : {args.episodes}",
            f"  Ray resources   : cpus={slurm_resources.num_cpus}  gpus={slurm_resources.num_gpus}",
            f"  Metric          : {args.metric}  (mode=max)",
            f"  Checkpoint freq : every {args.checkpoint_frequency_episodes} episodes",
            f"  Trajectories    : {'enabled' if args.log_trajectories else 'disabled'}",
            f"  Run name        : {run_name}",
            f"  Run dir         : {experiment_path}",
        ],
    )]


def _section_components(env_config: Any) -> list[Section]:
    infras = env_config.infras or []
    statesources = env_config.statesources or []
    infra_body = [_fmt_infra(i) for i in infras] if infras else ["  (none)"]
    src_body = [_fmt_statesource(s) for s in statesources] if statesources else ["  (none)"]
    return [
        (f"INFRASTRUCTURE  ({len(infras)} components)", infra_body),
        (f"STATESOURCES  ({len(statesources)} components)", src_body),
    ]


def _section_rewards(reward_manager: Any) -> list[Section]:
    specs = list(getattr(reward_manager, "_reward_specs", []))
    body: list[str] = []
    if specs:
        for s in specs:
            weight = s.get("weight", 1.0)
            params = s.get("params") or {}
            params_str = f"  {params}" if params else ""
            body.append(f"  - {s['class_name']:<32} w={weight}{params_str}")
    else:
        body.append("  (no rewards configured)")
    return [("REWARDS", body)]


def _section_schedules(
    reward_manager: Any,
    data_combinator: Any,
    infra_combinator: Any,
    grad_train: bool,
) -> list[Section]:
    body: list[str] = []

    reward_mode = getattr(reward_manager, "mode", "?")
    reward_swap = getattr(reward_manager, "swap_every_n_iterations", "?")
    body.append(
        f"  Reward schedule : mode={reward_mode}  swap every {reward_swap} iter  "
        f"(grad_train={grad_train})"
    )

    body.append("")
    if infra_combinator is not None:
        body.append(
            f"  Infra schedule  : mode={infra_combinator.mode}  "
            f"{len(infra_combinator.config_paths)} configs  "
            f"swap every {infra_combinator.swap_every_n_iterations} iter"
        )
    else:
        body.append("  Infra schedule  : none  (single static infra config)")

    body.append("")
    if data_combinator is not None:
        n_variants = len(getattr(data_combinator, "variants", []) or [])
        n_scenarios = len(getattr(data_combinator, "scenarios", []) or [])
        variable = getattr(data_combinator, "variable", {}) or {}
        n_var_combos = 1
        for v in variable.values():
            n_var_combos *= max(1, len(v))
        body.append(
            f"  Data schedule   : {n_variants} variants  "
            f"({n_scenarios} scenarios x {n_var_combos} variable combos)  "
            f"swap every {data_combinator.swap_every_n_episodes} ep  "
            f"mode={data_combinator.mode}  day={data_combinator.day}"
        )
    else:
        body.append("  Data schedule   : none")
    return [("SCHEDULES", body)]


def log_startup_banner(
    *,
    args: Namespace,
    env_config: Any,
    training_param_config: Any,
    reward_manager: Any,
    data_combinator: Any,
    infra_combinator: Any,
    slurm_resources: Any,
    run_name: str,
    experiment_path: str,
    storage_path: str,
    seed: int,
    exec_date: datetime.datetime | None = None,
    write_to_disk: bool = True,
) -> None:
    """Log an organised startup banner.

    Sections are numbered automatically from their order in the list below;
    reordering or adding sections does not require touching any `[i/N]` label.
    """
    sections: list[Section] = [
        *_section_invocation(seed, getattr(args, "load_config", None)),
        *_section_tensorboard(experiment_path, storage_path, exec_date),
        *_section_eval(args, env_config, experiment_path, seed),
        *_section_env_config(env_config),
        *_section_training_setup(
            args, training_param_config, slurm_resources, run_name, experiment_path
        ),
        *_section_components(env_config),
        *_section_rewards(reward_manager),
        *_section_schedules(
            reward_manager, data_combinator, infra_combinator, args.grad_train
        ),
    ]

    total = len(sections)
    rendered: list[str] = []
    for idx, (title, body) in enumerate(sections, start=1):
        rendered.append(_header(f"[{idx}/{total}] {title}"))
        rendered.extend(body)

    banner = "\n".join(rendered)
    # Single logger.info call so the timestamp prefix appears only once and
    # the box borders stay vertically aligned.
    logger.info("\n%s\nStartup summary:\n%s\n%s", _HORIZONTAL_LINE, banner, _HORIZONTAL_LINE)

    if write_to_disk:
        try:
            os.makedirs(experiment_path, exist_ok=True)
            out = Path(experiment_path) / "startup.txt"
            out.write_text(banner + "\n", encoding="utf-8")
            logger.info("Startup snapshot written to %s\n%s", out, _HORIZONTAL_LINE)
        except OSError as e:
            logger.warning("Could not write startup.txt: %s", e)
