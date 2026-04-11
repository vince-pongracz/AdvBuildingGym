"""Organised startup banner for training runs.

Prints a single, well-structured block that summarises everything a reader
needs to reconstruct the run from scratch: invocation, environment topology,
infrastructure, statesources, rewards + schedules, data schedule, training
setup, and TensorBoard commands.

The same block is optionally written to ``<run_dir>/startup.txt`` so the
snapshot travels with the checkpoint and can be diffed against other runs.

Call once from the training script, right before ``tuner.fit()``.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

logger = logging.getLogger("startup")

_HR = "=" * 70


def _header(title: str) -> str:
    return f"{_HR}\n{title}\n{_HR}"


def _fmt_infra(infra: Any) -> str:
    """One-line summary of an Infrastructure component."""
    name = getattr(infra, "name", type(infra).__name__)
    cls = type(infra).__name__
    fields: list[str] = []
    for attr in (
        "max_power_kW", "peak_power_kW", "rated_power_kW", "max_charging_kW",
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


def _section_invocation(seed: int) -> list[str]:
    job_id = os.environ.get("SLURM_JOB_ID", "n/a")
    host = socket.gethostname()
    cmd = " ".join(sys.argv)
    return [
        _header("[1/6] INVOCATION"),
        f"  CMD       : {cmd}",
        f"  HOST/JOB  : {host} / SLURM {job_id}",
        f"  SEED      : {seed}",
        f"  PYTHON    : {sys.version.split()[0]}",
    ]


def _section_env_config(env_config: Any) -> list[str]:
    bp = getattr(env_config, "building_props", None)
    mC = getattr(bp, "mC", "?")
    K = getattr(bp, "K", "?")
    return [
        _header(f"[2/6] ENV CONFIG  ({env_config.env_config_name})"),
        f"  EPISODE_LENGTH   = {env_config.EPISODE_LENGTH} steps",
        f"  CONTROL_STEP     = {env_config.CONTROL_STEP} s",
        f"  ACTION_HISTORY   = {env_config.ACTION_HISTORY_LENGTH}",
        f"  building_props   : mC={mC}  K={K}",
    ]


def _section_components(env_config: Any) -> list[str]:
    infras = env_config.infras or []
    statesources = env_config.statesources or []
    lines = [_header(f"[3/6] INFRASTRUCTURE  ({len(infras)} components)")]
    lines.extend(_fmt_infra(i) for i in infras) if infras else lines.append("  (none)")
    lines.append("")
    lines.append(_header(f"[4/6] STATESOURCES  ({len(statesources)} components)"))
    lines.extend(_fmt_statesource(s) for s in statesources) if statesources else lines.append("  (none)")
    return lines


def _section_rewards_and_schedules(
    reward_manager: Any,
    data_combinator: Any,
    infra_combinator: Any,
    grad_train: bool,
) -> list[str]:
    lines = [_header("[5/6] REWARDS & SCHEDULES")]

    specs = list(getattr(reward_manager, "_reward_specs", []))
    mode = getattr(reward_manager, "mode", "?")
    swap = getattr(reward_manager, "swap_every_n_iterations", "?")
    lines.append(
        f"  Reward schedule : mode={mode}  swap every {swap} iter  "
        f"(grad_train={grad_train})"
    )
    if specs:
        for s in specs:
            weight = s.get("weight", 1.0)
            params = s.get("params") or {}
            params_str = f"  {params}" if params else ""
            lines.append(f"    - {s['class_name']:<32} w={weight}{params_str}")
    else:
        lines.append("    (no rewards configured)")

    lines.append("")
    if infra_combinator is not None:
        lines.append(
            f"  Infra schedule  : mode={infra_combinator.mode}  "
            f"{len(infra_combinator.config_paths)} configs  "
            f"swap every {infra_combinator.swap_every_n_iterations} iter"
        )
    else:
        lines.append("  Infra schedule  : none  (single static infra config)")

    lines.append("")
    if data_combinator is not None:
        n_variants = len(getattr(data_combinator, "variants", []) or [])
        n_scenarios = len(getattr(data_combinator, "scenarios", []) or [])
        variable = getattr(data_combinator, "variable", {}) or {}
        n_var_combos = 1
        for v in variable.values():
            n_var_combos *= max(1, len(v))
        lines.append(
            f"  Data schedule   : {n_variants} variants  "
            f"({n_scenarios} scenarios x {n_var_combos} variable combos)  "
            f"swap every {data_combinator.swap_every_n_episodes} ep  "
            f"mode={data_combinator.mode}  day={data_combinator.day}"
        )
    else:
        lines.append("  Data schedule   : none")
    return lines


def _section_training_setup(
    args: Namespace,
    training_param_config: Any,
    slurm_resources: Any,
    run_name: str,
    experiment_path: str,
) -> list[str]:
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
            f"days_in_buffer={tpc.sac_days_to_keep_in_replay_buffer}  "
            f"train_intensity={tpc.sac_training_intensity}  "
            f"rollout_fragment={tpc.sac_rollout_fragment_length}"
        )
    else:
        hp = f"lr={tpc.learning_rate}"

    return [
        _header("[6/6] TRAINING SETUP"),
        f"  Algorithm       : {algo}  (new API stack)",
        f"  Hyperparams     : {hp}",
        f"  Episodes        : {args.episodes}  (-> {args.timesteps} timesteps)",
        f"  Ray resources   : cpus={slurm_resources.num_cpus}  gpus={slurm_resources.num_gpus}",
        f"  Metric          : {args.metric}  (mode=max)",
        f"  Checkpoint freq : every {args.checkpoint_frequency_episodes} episodes",
        f"  Trajectories    : {'enabled' if args.log_trajectories else 'disabled'}",
        f"  Run name        : {run_name}",
        f"  Run dir         : {experiment_path}",
    ]


def _section_tensorboard(experiment_path: str, storage_path: str) -> list[str]:
    return [
        _header("TENSORBOARD"),
        "  Training + eval curves share the same logdir; eval metrics are",
        "  nested under evaluation/env_runners/ in the TB UI.",
        "",
        "  This run only:",
        f"    tensorboard --logdir {experiment_path} --port 6006",
        "",
        "  All runs for this algorithm (compare across seeds):",
        f"    tensorboard --logdir {storage_path} --port 6007",
        "",
        "  Side-by-side training vs eval split (explicit tag filter):",
        f"    tensorboard --logdir_spec train:{experiment_path},eval:{experiment_path} --port 6008",
    ]


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
    write_to_disk: bool = True,
) -> None:
    """Log an organised, six-section startup banner.

    Prints via ``logger.info`` (one line per entry so timestamps stay aligned)
    and optionally writes the full banner to ``<experiment_path>/startup.txt``.
    """
    sections: list[list[str]] = [
        _section_invocation(seed),
        _section_env_config(env_config),
        _section_components(env_config),
        _section_rewards_and_schedules(
            reward_manager, data_combinator, infra_combinator, args.grad_train
        ),
        _section_training_setup(
            args, training_param_config, slurm_resources, run_name, experiment_path
        ),
        _section_tensorboard(experiment_path, storage_path),
    ]

    banner = "\n".join("\n".join(section) for section in sections)
    # Single logger.info call so the timestamp prefix appears only once and
    # the box borders stay vertically aligned.
    logger.info("Startup summary:\n%s", banner)

    if write_to_disk:
        try:
            os.makedirs(experiment_path, exist_ok=True)
            out = Path(experiment_path) / "startup.txt"
            out.write_text(banner + "\n", encoding="utf-8")
            logger.info("Startup snapshot written to %s", out)
        except OSError as e:
            logger.warning("Could not write startup.txt: %s", e)
