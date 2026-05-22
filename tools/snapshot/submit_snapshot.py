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

Submission injects ``SNAPSHOT_DIR``, ``SNAPSHOT_RUN_ID`` and
``LIVE_REPO_ROOT`` via ``sbatch --export=``; the snapshot-aware
wrappers in ``slurm_scripts/`` consume those env vars.

Outputs land under ``<snapshot>/runs/<kind>_<timestamp>/``.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.snapshot.make_snapshot import make_snapshot  # noqa: E402

logger = logging.getLogger(__name__)


# Mapping from --kind to (slurm wrapper basename, entry-point .py inside snapshot).
KIND_SPECS: dict[str, tuple[str, str]] = {
    "train":    ("slurm_train_ray.sh", "run_train_ray.py"),
    "eval":     ("slurm_eval_ray.sh",  "run_eval_ray.py"),
    "train-ma": ("slurm_train_ma.sh",  "rl_ma_train.py"),
    "train-sb": ("slurm_train_sb.sh",  "run_train_sb.py"),
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


def _find_latest_train_checkpoint(snapshot_dir: Path) -> str | None:
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


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------

def _build_plan(
    *,
    snapshot_dir: Path,
    kind: str,
    extra_args: list[str],
    checkpoint_override: str | None,
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
    run_dir = snapshot_dir / "runs" / run_id

    args = list(extra_args)

    # If the user didn't supply --trial in pass-through args, inject the
    # snapshot-internal trial path so the entry script always reads the
    # frozen YAML (never the live repo's copy).
    if "--trial" not in args:
        args = ["--trial", str(trial_abs_in_snap), *args]

    # For eval re-runs without --checkpoint, auto-discover the latest
    # training checkpoint inside this snapshot. resolve_checkpoint_path()
    # in the eval script would otherwise search from CWD=<run_dir>/models/
    # which is empty.
    if kind == "eval" and "--checkpoint" not in args:
        ckpt = checkpoint_override or _find_latest_train_checkpoint(snapshot_dir)
        if ckpt:
            args = [*args, "--checkpoint", ckpt]
        else:
            logger.warning(
                "No --checkpoint provided and no train run found in %s; the "
                "eval script will need to find a checkpoint itself.", snapshot_dir,
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
    )


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def _shlex_split(s: str) -> list[str]:
    import shlex
    return shlex.split(s) if s else []


def _build_sbatch_cmd(
    plan: SubmissionPlan,
    *,
    extra_sbatch_args: list[str],
    dry_run: bool,
) -> list[str]:
    out_file = plan.run_dir / "slurm.out"
    err_file = plan.run_dir / "slurm.err"

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
        "--note", default=None,
        help="Snapshot creation only: free-text note stored in manifest.json.",
    )
    parser.add_argument(
        "--out", default="snapshots",
        help="Snapshot creation only: where to put the new snapshot dir.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the sbatch command instead of running it.",
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
        wrapper_basename, entry_script = KIND_SPECS[args.kind]
        trial_abs = Path(args.trial).resolve()
        trial_rel = (
            str(trial_abs.relative_to(REPO_ROOT))
            if trial_abs.is_relative_to(REPO_ROOT) else str(trial_abs)
        )
        # Mirror the real flow: --trial points at the absolute path inside the
        # would-be snapshot's code/ dir, because the wrapper cd's into runs/.
        trial_abs_in_snap = snapshot_dir / "code" / trial_rel
        plan = SubmissionPlan(
            snapshot_dir=snapshot_dir,
            run_id=run_id,
            run_dir=snapshot_dir / "runs" / run_id,
            kind=args.kind,
            wrapper_path=REPO_ROOT / "slurm_scripts" / wrapper_basename,
            entry_script=entry_script,
            trial_path_in_snap=trial_rel,
            extra_args=["--trial", str(trial_abs_in_snap), *args.pass_through],
        )
    else:
        try:
            plan = _build_plan(
                snapshot_dir=snapshot_dir,
                kind=args.kind,
                extra_args=args.pass_through,
                checkpoint_override=args.checkpoint,
            )
        except Exception as exc:
            logger.error("Could not build submission plan: %s", exc)
            return 1
        plan.run_dir.mkdir(parents=True, exist_ok=False)

    # Phase 3: sbatch.
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
    logger.info("  pass-through : %s", " ".join(plan.extra_args) or "(none)")

    if args.dry_run:
        print(" ".join(cmd))
        return 0

    job_id = _submit(cmd)
    logger.info("Submitted job %s — outputs will appear under %s", job_id, plan.run_dir)
    print(plan.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
