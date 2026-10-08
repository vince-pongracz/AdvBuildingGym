"""Organised startup banner for training runs.

Summarises the run (invocation, tensorboard, eval, env config, training setup, infra,
statesources, rewards + schedules), optionally written to ``<run_dir>/startup.txt``.
Call once before training starts.
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

# Section builders return (title, body_lines); numbering is applied at render time
# from list order, so reordering can't desync labels.
Section = tuple[str, list[str]]


def _header(title: str) -> str:
    return f"{_HORIZONTAL_LINE}\n{title}\n{_HORIZONTAL_LINE}"


def _fmt_numeric_params(component: Any) -> str:
    """Numeric constructor params via the component's own ``to_dict()`` 
    (keeps only numeric scalars)."""
    to_dict = getattr(component, "to_dict", None)
    if not callable(to_dict):
        return ""
    params = to_dict()
    fields = [
        f"{key}={val}"
        for key, val in params.items()
        if key != "class" and isinstance(val, (int, float)) and not isinstance(val, bool)
    ]
    return "  " + "  ".join(fields) if fields else ""


def _fmt_infra(infra: Any) -> str:
    """One-line summary of an Infrastructure component."""
    name = getattr(infra, "name", type(infra).__name__)
    cls = type(infra).__name__
    return f"  - {name:<14} [{cls}]{_fmt_numeric_params(infra)}"


def _fmt_statesource(src: Any) -> str:
    name = getattr(src, "name", type(src).__name__)
    cls = type(src).__name__
    ds = getattr(src, "ds_path", None)
    ds_tail = f"  ds_path={Path(ds).name}" if ds else ""
    return f"  - {name:<20} [{cls}]{ds_tail}{_fmt_numeric_params(src)}"


def _section_invocation(seed: int, trial_path: str | None = None) -> list[Section]:
    job_id = os.environ.get("SLURM_JOB_ID", "n/a")
    host = socket.gethostname()
    cmd = " ".join(sys.argv)
    return [(
        "INVOCATION",
        [
            f"  CMD       : {cmd}",
            f"  HOST/JOB  : {host} / SLURM {job_id}",
            f"  SEED      : {seed}",
            f"  TRIAL CFG : {trial_path or '(not set)'}",
            f"  PYTHON    : {sys.version.split()[0]}",
        ],
    )]


def _section_tensorboard(
    tensorboard_log_path: str,
    tensorboard_root: str,
    *,
    eval_trajectories_path: str | None = None,
    exec_date: datetime.datetime | None = None,
) -> list[Section]:
    body = [
        "  Training + eval curves share the same logdir.",
        "",
        "  This run only:",
        f"    ./start_tensorboard.sh {tensorboard_log_path}",
        "",
        "  All runs for this algorithm (compare across seeds):",
        f"    ./start_tensorboard.sh {tensorboard_root}",
    ]
    if eval_trajectories_path is not None:
        body.extend([
            "",
            "  Eval trajectories (all runs):",
            f"    ./start_tensorboard.sh {eval_trajectories_path}",
        ])
        if exec_date is not None:
            # matches make_eval_state_action_cb_class dir:
            # ep_metrics/eval_trajectories/<YYYYmmdd_HHMMSS>_<job|pid>/ (avoids same-second collisions)
            job_suffix = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"
            run_dir_name = f"{exec_date.strftime('%Y%m%d_%H%M%S')}_{job_suffix}"
            run_eval_trajectories_path = os.path.join(eval_trajectories_path, run_dir_name)
            body.extend([
                "",
                "  Eval trajectories (this run only):",
                f"    ./start_tensorboard.sh {run_eval_trajectories_path}",
            ])
    return [("TENSORBOARD", body)]


def _section_eval(
    args: Namespace, env_config: Any, experiment_path: str, seed: int
) -> list[Section]:
    trial_path = getattr(args, "trial_path", None) or "<trial.yaml>"

    explicit_lines = [
        "  Explicit checkpoint path (this run):",
        "    sbatch slurm_scripts/slurm_eval_ray.sh \\",
        f"        --trial {trial_path} \\",
        f"        --checkpoint {experiment_path}/best_model_ep{{checkpoint_serial}}",
    ]

    auto_lines = [
        "  Auto-resolve best checkpoint for this env config:",
        f"    sbatch slurm_scripts/slurm_eval_ray.sh --trial {trial_path}",
    ]

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


def _section_env_config(env_config: Any, trial_name: str | None = None) -> list[Section]:
    label = f"ENV CONFIG  ({trial_name})" if trial_name else "ENV CONFIG"
    return [(
        label,
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
    exec_date: datetime.datetime | None = None,
) -> list[Section]:
    algo = args.algorithm.upper()
    tpc = training_param_config
    if args.algorithm == "ppo":
        hyperparams = (
            f"episodes_per_iter={tpc.ppo_episodes_per_iteration}  "
            f"minibatch={tpc.ppo_minibatch_size}  "
            f"epochs={tpc.ppo_num_epochs}"
        )
    elif args.algorithm == "sac":
        hyperparams = (
            f"replay_batch={tpc.sac_replay_batch_size}  "
            f"days_in_buffer={tpc.sac_episodes_to_keep_in_replay_buffer}  "
            f"train_intensity={tpc.sac_training_intensity}  "
        )

    else:
        hyperparams = ""

    body = [
        f"  Algorithm       : {algo}  (new API stack)",
        f"  Hyperparams     : {hyperparams}",
        f"  Episodes        : {args.episodes}",
        f"  Ray resources   : cpus={slurm_resources.num_cpus}  gpus={slurm_resources.num_gpus}",
        f"  Metric          : {args.metric}  (mode=max)",
        f"  Checkpoint freq : every {args.checkpoint_frequency_iterations} iterations",
        f"  Trajectories    : {'enabled' if args.log_trajectories else 'disabled'}",
    ]

    if args.log_trajectories:
        # mirrors common_model_setup: metrics_base_dir/trajectories/<exec_date>
        stamp = (exec_date or datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
        traj_dir = (Path.cwd() / "ep_metrics" / "trajectories" / stamp).as_posix()
        body.append(f"  Trajectory dir  : {traj_dir}")

    body.extend([
        f"  Run name        : {run_name}",
        f"  Run dir         : {experiment_path}",
    ])

    return [("TRAINING SETUP", body)]


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
    reward_swap = getattr(reward_manager, "swap_every_n_episodes", "?")
    body.append(
        f"  Reward schedule : mode={reward_mode}  swap every {reward_swap} episodes  "
        f"(grad_train={grad_train})"
    )

    body.append("")
    if infra_combinator is not None:
        body.append(
            f"  Infra schedule  : mode={infra_combinator.mode}  "
            f"{len(infra_combinator.config_paths)} configs  "
            f"swap every {infra_combinator.swap_every_n_episodes} episodes"
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
    tensorboard_log_path: str | None = None,
    eval_trajectories_path: str | None = None,
    write_to_disk: bool = True,
) -> None:
    """Log an organised startup banner (sections auto-numbered from list order).

    ``storage_path``: parent of all runs (cross-seed TB target). 
    ``tensorboard_log_path``: TB events root (defaults to ``experiment_path``; SB3 passes the ``tb/`` subdir).
    ``eval_trajectories_path``: root of the Ray ``EvalStateActionCallback`` sub-runs;
    ``None`` skips that block (e.g. SB3).
    """
    sections: list[Section] = [
        *_section_invocation(seed, getattr(args, "load_config", None)),
        *_section_tensorboard(
            tensorboard_log_path or experiment_path,
            storage_path,
            eval_trajectories_path=eval_trajectories_path,
            exec_date=exec_date,
        ),
        *_section_eval(args, env_config, experiment_path, seed),
        *_section_env_config(env_config, getattr(args, "trial_name", None)),
        *_section_training_setup(
            args, training_param_config, slurm_resources, run_name, experiment_path, exec_date=exec_date,
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
    # single logger.info so the timestamp prefix appears once and borders stay aligned
    logger.info("\n%s\nStartup summary:\n%s\n%s", _HORIZONTAL_LINE, banner, _HORIZONTAL_LINE)

    if write_to_disk:
        try:
            os.makedirs(experiment_path, exist_ok=True)
            out = Path(experiment_path) / "startup.txt"
            out.write_text(banner + "\n", encoding="utf-8")
            logger.info("Startup snapshot written to %s\n%s", out, _HORIZONTAL_LINE)
        except OSError as e:
            logger.warning("Could not write startup.txt: %s", e)
