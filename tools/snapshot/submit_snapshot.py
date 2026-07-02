"""Snapshot-aware SLURM submitter.

Two modes:

1. Create a fresh snapshot and submit a SLURM job against it::

       python -m tools.snapshot.submit_snapshot \
           --trial configs/trial_cfgs/<x>.yaml \
           --kind train \
           [--sbatch "--time=02:00:00 ..."] \
           [-- <args passed through to run_train_ray.py>]

2. Re-run an existing snapshot (no new snapshot is built)::

       python -m tools.snapshot.submit_snapshot \
           --snapshot snapshots/<existing> \
           --kind eval \
           [--checkpoint <abs path>] \
           [-- --episodes 20 --plot-all]

3. Run a snapshot locally without sbatch (foreground bash exec)::

       python -m tools.snapshot.submit_snapshot \
           --snapshot snapshots/<existing> \
           --kind eval \
           --local \
           [-- --episodes 20]

Submission injects ``SNAPSHOT_DIR``, ``SNAPSHOT_RUN_ID`` and
``LIVE_REPO_ROOT`` via ``sbatch --export=`` (or directly into the
child env in ``--local`` mode); the snapshot-aware wrappers in
``slurm_scripts/`` consume those env vars.

Outputs land under ``<snapshot>/runs/<kind>_<timestamp>/``.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import logging
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.snapshot.make_snapshot import make_snapshot  # noqa: E402

logger = logging.getLogger("submit_snapshot")


# Mapping from --kind to (slurm wrapper basename, entry-point .py inside snapshot).
KIND_SPECS: dict[str, tuple[str, str]] = {
    "train":        ("slurm_train_ray.sh",     "run_train_ray.py"),
    "train-cpu":    ("slurm_train_ray_cpu.sh", "run_train_ray.py"),
    "train-ma":     ("slurm_train_ma.sh",      "rl_ma_train.py"),
    "train-sb":     ("slurm_train_sb.sh",      "run_train_sb.py"),
    "train-sb-cpu": ("slurm_train_sb_cpu.sh",  "run_train_sb.py"),
    "eval":         ("slurm_eval_ray.sh",      "run_eval_ray.py"),
    "eval-rbc":     ("slurm_eval_rbc.sh",      "run_eval_rule_based.py"),
    "eval-sb":      ("slurm_eval_sb.sh",       "run_eval_sb.py"),
}


@dataclass
class SubmissionPlan:
    """The state we hand off to sbatch."""

    snapshot_dir: Path
    run_id: str
    run_dir: Path
    kind: str
    wrapper_path: Path
    entry_script: str           # basename relative to <snapshot>/code/
    trial_path_in_snap: str     # relative to <snapshot>/code/
    extra_args: list[str]
    seed_override: int | None = None


# ---------------------------------------------------------------------------
# Snapshot inspection helpers
# ---------------------------------------------------------------------------

def _read_manifest(snapshot_dir: Path) -> dict:
    import json
    mf = snapshot_dir / "manifest.json"
    if not mf.exists():
        raise FileNotFoundError(f"manifest.json not found in {snapshot_dir}")
    with mf.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_latest_ray_train_checkpoint(snapshot_dir: Path) -> str | None:
    """Look for a Ray checkpoint inside this snapshot's train run(s).

    Returns the absolute path of the most recent checkpoint root (marked by
    rllib_checkpoint.json) under ``<snapshot>/runs/train_*/models/``, or
    None if no training run exists yet.
    """
    runs_dir = snapshot_dir / "runs"
    if not runs_dir.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for train_dir in sorted(runs_dir.glob("train*")):
        models_dir = train_dir / "models"
        if not models_dir.exists():
            continue
        for root, _, files in os.walk(models_dir):
            if "rllib_checkpoint.json" in files or ".is_checkpoint" in files:
                candidates.append((os.path.getmtime(root), Path(root)))
    if not candidates:
        return None
    candidates.sort(key=lambda e: e[0], reverse=True)
    return str(candidates[0][1].resolve())


def _find_latest_sb_train_model(snapshot_dir: Path) -> str | None:
    """Look for an SB3 model (.zip) inside this snapshot's SB train run(s).

    Prefers the most recent ``best/best_model.zip`` under
    ``<snapshot>/runs/train_sb*/models/`` (what the SB eval callback saves),
    else the most recent ``.zip`` there. Returns None if no SB training run
    exists yet. Mirrors ``_find_latest_train_checkpoint`` for the Ray side.
    """
    runs_dir = snapshot_dir / "runs"
    if not runs_dir.exists():
        return None
    best_models: list[tuple[float, Path]] = []
    other_zips: list[tuple[float, Path]] = []
    for train_dir in sorted(runs_dir.glob("train_sb*")):
        models_dir = train_dir / "models"
        if not models_dir.exists():
            continue
        for root, _, files in os.walk(models_dir):
            for name in files:
                if not name.endswith(".zip"):
                    continue
                path = Path(root) / name
                bucket = best_models if name == "best_model.zip" else other_zips
                bucket.append((os.path.getmtime(path), path))
    # Prefer best_model.zip; fall back to any checkpoint/final .zip.
    pool = best_models or other_zips
    if not pool:
        return None
    pool.sort(key=lambda e: e[0], reverse=True)
    return str(pool[0][1].resolve())


# ---------------------------------------------------------------------------
# Seed-override helpers
# ---------------------------------------------------------------------------

def _extract_snapshot_if_needed(snapshot_dir: Path) -> None:
    """Eagerly extract snapshot.zip → <snapshot>/code/ if not already done.

    Mirrors slurm_scripts/util/snapshot_mode.sh's lazy extraction. We need
    code/configs/ on disk at submit time when --seed is given so we can copy
    it into the per-run dir. Uses the same flock as snapshot_mode.sh so
    concurrent submitters don't race.
    """
    code_dir = snapshot_dir / "code"
    if code_dir.exists():
        return
    zip_path = snapshot_dir / "snapshot.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"snapshot.zip missing from {snapshot_dir}")
    lock_path = snapshot_dir / ".code.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if not code_dir.exists():
            logger.info("Extracting snapshot.zip into %s/", code_dir)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(code_dir)


_TOP_LEVEL_SEED_REGEX = re.compile(r"^seed:\s*-?\d+(?P<trail>\s*#.*)?$", re.MULTILINE)


def _rewrite_top_level_seed(trial_yaml_path: Path, new_seed: int) -> None:
    """Rewrite the top-level ``seed:`` value in a trial YAML in place.

    Matches a single line at indent 0 of the form ``seed: <int>`` (allowing a
    trailing comment). The trailing comment, if any, is preserved verbatim.
    Raises if zero or multiple matches are found so we never silently change
    the wrong scalar — nested ``seed:`` keys (e.g. ``data_combinator.seed``)
    live at non-zero indent and won't match this MULTILINE start-of-line
    pattern.
    """
    text = trial_yaml_path.read_text(encoding="utf-8")
    matches = _TOP_LEVEL_SEED_REGEX.findall(text)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one top-level `seed:` line in {trial_yaml_path}; "
            f"found {len(matches)}."
        )

    def _replace(m: re.Match) -> str:
        trail = m.group("trail") or ""
        return f"seed: {new_seed}{trail}"

    new_text = _TOP_LEVEL_SEED_REGEX.sub(_replace, text, count=1)
    trial_yaml_path.write_text(new_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

def _build_plan(
    *,
    snapshot_dir: Path,
    kind: str,
    extra_args: list[str],
    checkpoint_override: str | None,
    seed_override: int | None = None,
    dry_run: bool = False,
) -> SubmissionPlan:
    if kind not in KIND_SPECS:
        raise ValueError(f"Unknown --kind {kind!r}; valid: {sorted(KIND_SPECS)}")
    wrapper_basename, entry_script = KIND_SPECS[kind]
    wrapper_path = REPO_ROOT / "slurm_scripts" / wrapper_basename
    if not wrapper_path.exists():
        raise FileNotFoundError(f"SLURM wrapper not found: {wrapper_path}")

    manifest = _read_manifest(snapshot_dir)
    trial_path_in_snap = manifest["source_trial_path"]
    if not (snapshot_dir / "snapshot.zip").exists():
        raise FileNotFoundError(f"snapshot.zip missing from {snapshot_dir}")

    # Absolute path to the trial YAML inside the snapshot's code/ dir. The
    # SLURM wrapper changes CWD to <snapshot>/runs/<run_id>/, so a relative
    # path would resolve there (wrong); use the absolute form so the entry
    # script picks up the frozen YAML.
    trial_abs_in_snap = snapshot_dir / "code" / trial_path_in_snap

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{kind.replace('-', '_')}_{timestamp}"
    if seed_override is not None:
        run_id = f"{run_id}_seed{seed_override}"
    run_dir = snapshot_dir / "runs" / run_id

    args = list(extra_args)
    effective_trial_path = trial_abs_in_snap

    # --seed: pre-create the run dir with a copied configs/ tree and rewrite
    # the seed in the per-run copy. The original snapshot stays untouched so
    # multiple seeds can fan out from one snapshot in isolated run dirs.
    # source_trial_path is repo-relative (e.g. "configs/trial_cfgs/.../x.yaml");
    # the copytree puts it under <run_dir>/configs/trial_cfgs/.../x.yaml.
    if seed_override is not None:
        trial_rel_to_configs = Path(trial_path_in_snap).relative_to("configs")
        effective_trial_path = run_dir / "configs" / trial_rel_to_configs
        if dry_run:
            logger.info(
                "[dry-run] would extract snapshot, copy configs/ into %s, "
                "and rewrite seed → %d in %s",
                run_dir, seed_override, effective_trial_path,
            )
        else:
            _extract_snapshot_if_needed(snapshot_dir)
            run_dir.mkdir(parents=True, exist_ok=False)
            shutil.copytree(snapshot_dir / "code" / "configs", run_dir / "configs")
            _rewrite_top_level_seed(effective_trial_path, seed_override)
            logger.info(
                "Seed override applied: %s (rewrote %s)",
                seed_override, effective_trial_path,
            )

    # If the user didn't supply --trial in pass-through args, inject the
    # snapshot-internal trial path (or the per-run copy when --seed is set)
    # so the entry script always reads the frozen YAML (never the live repo's
    # copy).
    if "--trial" not in args:
        args = ["--trial", str(effective_trial_path), *args]

    # For eval re-runs without --checkpoint, auto-discover the latest training
    # checkpoint inside this snapshot. The eval scripts would otherwise search
    # from CWD=<run_dir>/models/ (empty). Ray eval wants a checkpoint dir
    # (rllib_checkpoint.json marker); SB eval wants a .zip model.
    if kind in ("eval", "eval-sb") and "--checkpoint" not in args:
        if checkpoint_override:
            ckpt = checkpoint_override
        elif kind == "eval-sb":
            ckpt = _find_latest_sb_train_model(snapshot_dir)
        else:
            ckpt = _find_latest_ray_train_checkpoint(snapshot_dir)
        if ckpt:
            args = [*args, "--checkpoint", ckpt]
        else:
            logger.warning(
                "No --checkpoint provided and no %s train run found in %s; the "
                "eval script will need to find a checkpoint itself.",
                "SB" if kind == "eval-sb" else "Ray", snapshot_dir,
            )

    return SubmissionPlan(
        snapshot_dir=snapshot_dir,
        run_id=run_id,
        run_dir=run_dir,
        kind=kind,
        wrapper_path=wrapper_path,
        entry_script=entry_script,
        trial_path_in_snap=trial_path_in_snap,
        extra_args=args,
        seed_override=seed_override,
    )


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def _shlex_split(s: str) -> list[str]:
    import shlex
    return shlex.split(s) if s else []


# Matches a #SBATCH time directive in a wrapper script: --time=X, --time X, -t X.
_WRAPPER_TIME_REGEX = re.compile(
    r"^#SBATCH\s+(?:--time(?:=|\s+)|-t\s+)(?P<time>\S+)", re.MULTILINE
)


def _resolve_time_limit(extra_sbatch_args: list[str], wrapper_path: Path) -> str:
    """Return the effective sbatch wall-clock limit for this submission.

    CLI --sbatch flags override the wrapper's #SBATCH directive (matching how
    sbatch itself resolves precedence), so a --time/-t in extra_sbatch_args
    wins; otherwise fall back to the wrapper's #SBATCH --time=. Returns
    "unknown" if neither specifies one.
    """
    # Walk the CLI args; the last --time/-t wins (later flags override earlier).
    resolved: str | None = None
    i = 0
    while i < len(extra_sbatch_args):
        arg = extra_sbatch_args[i]
        if arg.startswith("--time="):
            resolved = arg.split("=", 1)[1]
        elif arg in ("--time", "-t") and i + 1 < len(extra_sbatch_args):
            resolved = extra_sbatch_args[i + 1]
            i += 1
        i += 1
    if resolved is not None:
        return resolved

    try:
        text = wrapper_path.read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    m = _WRAPPER_TIME_REGEX.search(text)
    return m.group("time") if m else "unknown"


def _build_sbatch_cmd(
    plan: SubmissionPlan,
    *,
    extra_sbatch_args: list[str],
    dry_run: bool,
) -> list[str]:
    # %j is expanded by sbatch to the assigned job id, so the files land as
    # slurm_<jobid>.{out,err} — easy to correlate with `squeue` / `sacct`.
    out_file = plan.run_dir / "slurm_%j.out"
    err_file = plan.run_dir / "slurm_%j.err"

    # --export with embedded commas/spaces is tricky; sbatch accepts a single
    # comma-separated string. None of our values contain commas because they
    # are absolute paths and a timestamped id.
    export_pairs = [
        "ALL",
        f"SNAPSHOT_DIR={plan.snapshot_dir.resolve()}",
        f"SNAPSHOT_RUN_ID={plan.run_id}",
        f"LIVE_REPO_ROOT={REPO_ROOT.resolve()}",
    ]
    cmd = [
        "sbatch",
        f"--output={out_file}",
        f"--error={err_file}",
        f"--export={','.join(export_pairs)}",
        *extra_sbatch_args,
        str(plan.wrapper_path),
        *plan.extra_args,
    ]
    if dry_run:
        # Print and return — caller will skip the actual sbatch call.
        return cmd
    return cmd


def _submit(cmd: list[str]) -> str:
    """Run sbatch and return the parsed job id ('?' if it cannot be extracted)."""
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"sbatch failed with exit code {proc.returncode}")
    m = re.search(r"Submitted batch job (\d+)", proc.stdout)
    return m.group(1) if m else "?"


def _build_local_cmd(plan: SubmissionPlan) -> tuple[list[str], dict[str, str]]:
    """Build the bash invocation + env overrides for --local mode.

    Mirrors what sbatch --export would have given the wrapper: SNAPSHOT_DIR,
    SNAPSHOT_RUN_ID, LIVE_REPO_ROOT. The wrapper's #SBATCH directives are
    ignored by bash, so the script runs inline with whatever resources the
    current shell has.

    SLURM_SUBMIT_DIR is also set here to REPO_ROOT. sbatch normally exports
    this automatically (= the dir where sbatch was invoked); the wrappers
    use ${SLURM_SUBMIT_DIR:-$PWD} to locate sibling helpers like
    scratch_monitor.sh and print_env_info.py, and the $PWD fallback breaks
    once snapshot_mode.sh cd's into the run dir. Setting it upfront keeps
    the wrappers' path resolution consistent across sbatch and --local.
    """
    repo_root = str(REPO_ROOT.resolve())
    env_overrides = {
        "SNAPSHOT_DIR": str(plan.snapshot_dir.resolve()),
        "SNAPSHOT_RUN_ID": plan.run_id,
        "LIVE_REPO_ROOT": repo_root,
        "SLURM_SUBMIT_DIR": repo_root,
    }
    cmd = ["bash", str(plan.wrapper_path), *plan.extra_args]
    return cmd, env_overrides


def _run_local(cmd: list[str], env_overrides: dict[str, str]) -> int:
    """Exec the wrapper script in the current shell; stdio inherits."""
    env = os.environ.copy()
    env.update(env_overrides)
    proc = subprocess.run(cmd, env=env, check=False)
    return proc.returncode


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Snapshot-aware SLURM submitter. Either creates a new snapshot "
            "(--trial) or reuses an existing one (--snapshot), then submits "
            "a SLURM job whose outputs land inside the snapshot."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Anything after `--` is passed through to the SLURM wrapper, "
            "and onward to the entry script (run_train_ray.py / run_eval_ray.py / ...)."
        ),
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--trial",
        help="Path to a trial config YAML — creates a new snapshot before submitting.",
    )
    src.add_argument(
        "--snapshot",
        help="Path to an existing snapshot directory — re-submits without snapshotting.",
    )
    parser.add_argument(
        "--kind", required=True, choices=sorted(KIND_SPECS),
        help="Which SLURM wrapper to submit against.",
    )
    parser.add_argument(
        "--sbatch", default="",
        help="Extra sbatch flags (quoted), e.g. --sbatch=\"--time=02:00:00 --cpus-per-task=8\".",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Eval-only: explicit checkpoint path. Bypasses snapshot-internal auto-discovery.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the trial YAML's top-level seed for this run. Copies "
            "configs/ into the run dir, rewrites the seed in the copy, and "
            "appends _seed{N} to the run id. Original snapshot is untouched, "
            "so multiple seeds can fan out from a single snapshot in isolated "
            "run dirs.",
    )
    parser.add_argument(
        "--note", default=None,
        help="Snapshot creation only: free-text note stored in manifest.json.",
    )
    parser.add_argument(
        "--out", default="snapshots",
        help="Snapshot creation only: where to put the new snapshot dir.",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Run the wrapper script directly in this shell instead of "
            "submitting it via sbatch. Snapshot resolution and run-dir setup "
            "are still done; SNAPSHOT_DIR / SNAPSHOT_RUN_ID / LIVE_REPO_ROOT "
            "are exported so the wrapper's snapshot-mode helper still kicks "
            "in. Stdout/stderr stream straight to the console; the wrapper's "
            "#SBATCH directives are inert under bash. --sbatch flags are "
            "ignored in this mode.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the sbatch command (or local bash command with --local) "
            "instead of running it.",
    )
    # Everything after the first bare `--` becomes pass-through args. argparse
    # already supports REMAINDER, but we use parse_known_args so users don't
    # have to put `--` before unrecognised flags.
    args, extras = parser.parse_known_args(argv)
    # parse_known_args keeps a literal `--` separator token; drop it so it
    # doesn't get forwarded to sbatch / the entry script.
    if extras and extras[0] == "--":
        extras = extras[1:]
    args.pass_through = extras
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    args = _parse_args(argv)

    # --seed rewrites the per-run trial YAML copy and points --trial at it.
    # A pass-through --trial would bypass the rewrite, so the seed override
    # would silently do nothing — reject up front instead.
    if args.seed is not None and "--trial" in args.pass_through:
        logger.error(
            "--seed cannot be combined with a pass-through --trial; "
            "--seed rewrites the snapshot's trial YAML and must own the --trial flag."
        )
        return 1

    # Phase 1: snapshot resolution.
    if args.trial:
        if args.dry_run:
            # No durable state on dry-run; report a placeholder path.
            snapshot_dir = Path(args.out).resolve() / "<would-be-created-from-trial>"
            logger.info("[dry-run] would snapshot %s into %s/", args.trial, args.out)
        else:
            result = make_snapshot(args.trial, out_root=args.out, note=args.note)
            snapshot_dir = result.snapshot_dir
            logger.info("Snapshot created at %s", snapshot_dir)
    else:
        snapshot_dir = Path(args.snapshot).resolve()
        if not snapshot_dir.exists():
            logger.error("Snapshot directory does not exist: %s", snapshot_dir)
            return 1

    # Phase 2: build a submission plan and create the run dir.
    if args.dry_run and args.trial:
        # Build a synthetic plan from the trial YAML so we can still print
        # the sbatch command shape. _build_plan needs an existing manifest;
        # short-circuit with a minimal stand-in.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{args.kind.replace('-', '_')}_{timestamp}"
        if args.seed is not None:
            run_id = f"{run_id}_seed{args.seed}"
        wrapper_basename, entry_script = KIND_SPECS[args.kind]
        trial_abs = Path(args.trial).resolve()
        trial_rel = (
            str(trial_abs.relative_to(REPO_ROOT))
            if trial_abs.is_relative_to(REPO_ROOT) else str(trial_abs)
        )
        # Mirror the real flow: --trial points at the absolute path inside the
        # would-be snapshot's code/ dir, because the wrapper cd's into runs/.
        # When --seed is set, it would instead point at the per-run copy
        # under <run_dir>/configs/. Dry-run only previews the path — no
        # extraction, no copy, no rewrite.
        run_dir = snapshot_dir / "runs" / run_id
        if args.seed is not None:
            trial_rel_to_configs = Path(trial_rel).relative_to("configs")
            effective_trial_path = run_dir / "configs" / trial_rel_to_configs
        else:
            effective_trial_path = snapshot_dir / "code" / trial_rel
        plan = SubmissionPlan(
            snapshot_dir=snapshot_dir,
            run_id=run_id,
            run_dir=run_dir,
            kind=args.kind,
            wrapper_path=REPO_ROOT / "slurm_scripts" / wrapper_basename,
            entry_script=entry_script,
            trial_path_in_snap=trial_rel,
            extra_args=["--trial", str(effective_trial_path), *args.pass_through],
            seed_override=args.seed,
        )
    else:
        try:
            plan = _build_plan(
                snapshot_dir=snapshot_dir,
                kind=args.kind,
                extra_args=args.pass_through,
                checkpoint_override=args.checkpoint,
                seed_override=args.seed,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            logger.error("Could not build submission plan: %s", exc)
            return 1
        # _build_plan already created the run dir when --seed is set (to
        # land the configs copytree). Otherwise create it here. Dry-run
        # never touches disk.
        if not args.dry_run and plan.seed_override is None:
            plan.run_dir.mkdir(parents=True, exist_ok=False)

    # Phase 3: sbatch — or local bash exec when --local is set.
    if args.local:
        if args.sbatch:
            logger.warning("--sbatch flags are ignored in --local mode: %s", args.sbatch)
        local_cmd, local_env = _build_local_cmd(plan)
    else:
        extra_sbatch = _shlex_split(args.sbatch)
        cmd = _build_sbatch_cmd(plan, extra_sbatch_args=extra_sbatch, dry_run=args.dry_run)

    logger.info("Submission plan:")
    logger.info("  snapshot dir : %s", plan.snapshot_dir)
    logger.info("  run id       : %s", plan.run_id)
    logger.info("  run dir      : %s", plan.run_dir)
    logger.info("  kind         : %s", plan.kind)
    logger.info("  wrapper      : %s", plan.wrapper_path)
    logger.info("  entry script : %s", plan.entry_script)
    logger.info("  trial path   : %s (inside snapshot)", plan.trial_path_in_snap)
    if plan.seed_override is not None:
        logger.info("  seed override: %d", plan.seed_override)
    logger.info("  pass-through : %s", " ".join(plan.extra_args) or "(none)")
    logger.info("  mode         : %s", "local (bash)" if args.local else "sbatch")

    if args.local:
        if args.dry_run:
            env_preview = " ".join(f"{k}={v}" for k, v in local_env.items())
            print(f"{env_preview} {' '.join(local_cmd)}")
            return 0
        logger.info("Running locally — output streams to this terminal.")
        rc = _run_local(local_cmd, local_env)
        if rc != 0:
            logger.error("Wrapper script exited with code %d", rc)
        else:
            logger.info("Wrapper script completed; outputs under %s", plan.run_dir)
        print(plan.run_dir)
        return rc

    if args.dry_run:
        print(" ".join(cmd))
        return 0

    job_id = _submit(cmd)
    # Persist the job id inside the run dir so the run can later be cancelled
    # (`scancel $(cat <run_dir>/slurm_job_id)`) without grepping squeue/sacct.
    # The second line records the effective wall-clock limit for reference.
    time_limit = _resolve_time_limit(extra_sbatch, plan.wrapper_path)
    job_id_file = plan.run_dir / "slurm_job_id"
    job_id_file.write_text(
        f"Slurm job ID: {job_id}\nTime limit: {time_limit}\n", encoding="utf-8"
    )
    logger.info("Submitted job %s — outputs will appear under %s", job_id, plan.run_dir)
    logger.info(
        "  job id recorded in %s (stop with: scancel %s); time limit: %s",
        job_id_file, job_id, time_limit,
    )
    print(plan.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
