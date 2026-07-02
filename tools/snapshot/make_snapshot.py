"""Create an immutable snapshot of the code + configs feeding a single trial.

A snapshot is a self-contained ``snapshots/<timestamp>_<trial_name>/`` directory
holding:
    snapshot.zip   — code + the trial YAML + every YAML it transitively references
    manifest.json  — provenance metadata (git SHA, dirty flag, sha256, file list)

The zip is the immutable artifact. The companion SLURM wrapper extracts it
lazily into ``<snapshot>/code/`` on first run and reuses it on subsequent runs.

Data CSVs are NOT bundled — configs reference them by relative templates that
resolve against whatever is on disk at run time.

CLI:
    python -m tools.snapshot.make_snapshot --trial configs/trial_cfgs/<x>.yaml [--note "..."]

Importable:
    from tools.snapshot.make_snapshot import make_snapshot
    snap_dir = make_snapshot(trial_yaml_path, note="...")
"""

from __future__ import annotations

import argparse
import datetime
import getpass
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Make the project root importable so ``from adv_building_gym ...`` works
# whether this file is run as a module or as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adv_building_gym.config.trial_config import TrialConfig  # noqa: E402  (path setup above)

logger = logging.getLogger("make_snapshot")


# ---------------------------------------------------------------------------
# What the snapshot contains
# ---------------------------------------------------------------------------

# Top-level paths that are bundled wholesale (subject to the exclusion filter).
# Anything not in this list and not pulled in via the trial's config closure
# is left out. plotting/ is deliberately NOT bundled — see docs/snapshot plan.
WHITELIST_DIRS = (
    "adv_building_gym",
    "slurm_scripts/util",
)
WHITELIST_FILES = (
    "pyproject.toml",
    "run_train_ray.py",
    "run_eval_ray.py",
    "run_eval_rule_based.py",
    "rl_ma_train.py",
    "run_train_sb.py",
    "run_eval_sb.py",
)

# Names skipped anywhere inside the bundled tree.
EXCLUDED_DIR_NAMES = frozenset({"__pycache__"})
EXCLUDED_FILE_SUFFIXES = (".pyc", ".pyo")
EXCLUDED_DIR_SUFFIXES = (".egg-info",)


@dataclass
class SnapshotResult:
    """What ``make_snapshot()`` returns."""

    snapshot_dir: Path
    zip_path: Path
    manifest_path: Path
    included_files: list[tuple[str, int]] = field(default_factory=list)
    zip_sha256: str = ""
    zip_size_bytes: int = 0


# ---------------------------------------------------------------------------
# Config closure
# ---------------------------------------------------------------------------

def _collect_referenced_configs(trial_yaml_path: Path, raw: dict) -> list[Path]:
    """Resolve every YAML path the trial transitively references.

    Includes the trial YAML itself, the data-schedule train/eval YAMLs, and
    every infra/statesource schedule YAML listed in the trial. Paths are kept
    as their original relative form (relative to ``REPO_ROOT``) so they
    resolve from inside the snapshot just like they do in the live repo.
    """
    found: list[Path] = [trial_yaml_path]

    data_schedule = raw.get("data_schedule") or {}
    for split in ("train", "eval"):
        ds_path = data_schedule.get(split)
        if ds_path:
            found.append(Path(ds_path))

    for schedule_key in ("infra_schedule", "statesource_schedule"):
        sched = raw.get(schedule_key) or {}
        configs = (sched.get("configs") or {}) if isinstance(sched, dict) else {}
        for split in ("train", "eval"):
            for schedule_path in configs.get(split) or []:
                if schedule_path:
                    found.append(Path(schedule_path))

    # Dedupe while preserving order, drop missing-on-disk entries with a warning.
    seen: set[Path] = set()
    closure: list[Path] = []
    for schedule_path in found:
        abs_p = (REPO_ROOT / schedule_path).resolve() if not schedule_path.is_absolute() else schedule_path.resolve()
        if abs_p in seen:
            continue
        seen.add(abs_p)
        if not abs_p.exists():
            logger.warning("Referenced config not found on disk, skipping: %s", abs_p)
            continue
        closure.append(abs_p)
    return closure


# ---------------------------------------------------------------------------
# Filesystem walking
# ---------------------------------------------------------------------------

def _iter_dir(dir_abs: Path) -> Iterable[Path]:
    """Yield every file under ``dir_abs``, applying the exclusion filters."""
    for root, dirs, files in os.walk(dir_abs):
        dirs[:] = [
            d for d in dirs
            if d not in EXCLUDED_DIR_NAMES
            and not any(d.endswith(suffix) for suffix in EXCLUDED_DIR_SUFFIXES)
        ]
        for name in files:
            if any(name.endswith(suffix) for suffix in EXCLUDED_FILE_SUFFIXES):
                continue
            yield Path(root) / name


def _gather_files(config_closure: list[Path]) -> list[tuple[Path, str]]:
    """Build the (absolute_source, arcname) list to feed to the zip writer."""
    entries: list[tuple[Path, str]] = []
    seen_arcnames: set[str] = set()

    def _add(src_abs: Path) -> None:
        # arcname is the path relative to the repo root, with forward slashes.
        try:
            rel = src_abs.relative_to(REPO_ROOT)
        except ValueError:
            logger.warning("File outside repo root, skipping: %s", src_abs)
            return
        arc = rel.as_posix()
        if arc in seen_arcnames:
            return
        seen_arcnames.add(arc)
        entries.append((src_abs, arc))

    for rel_dir in WHITELIST_DIRS:
        abs_dir = REPO_ROOT / rel_dir
        if not abs_dir.exists():
            logger.warning("Whitelisted dir missing: %s", abs_dir)
            continue
        for file in _iter_dir(abs_dir):
            _add(file)

    for rel_file in WHITELIST_FILES:
        abs_file = REPO_ROOT / rel_file
        if not abs_file.exists():
            logger.warning("Whitelisted file missing: %s", abs_file)
            continue
        _add(abs_file)

    for cfg_abs in config_closure:
        _add(cfg_abs)

    entries.sort(key=lambda e: e[1])
    return entries


# ---------------------------------------------------------------------------
# Git provenance
# ---------------------------------------------------------------------------

def _git(args: list[str]) -> str:
    """Run a git command at REPO_ROOT; return stdout stripped, or '' on failure."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("git %s failed: %s", " ".join(args), e)
        return ""


def _git_provenance() -> dict[str, object]:
    sha = _git(["rev-parse", "HEAD"])
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    porcelain = _git(["status", "--porcelain"])
    return {
        "git_sha": sha,
        "git_branch": branch,
        "git_dirty": bool(porcelain),
        "git_dirty_files": porcelain.splitlines() if porcelain else [],
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def _sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _sanitize_for_dirname(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)


def make_snapshot(
    trial_yaml_path: str | Path,
    *,
    out_root: str | Path = "snapshots",
    note: str | None = None,
) -> SnapshotResult:
    """Build a snapshot directory + zip for the given trial config.

    Raises if the trial fails to validate. Returns paths the orchestrator
    can hand off to SLURM.
    """
    trial_yaml_path = Path(trial_yaml_path)
    if not trial_yaml_path.is_absolute():
        trial_yaml_path = (REPO_ROOT / trial_yaml_path).resolve()
    if not trial_yaml_path.exists():
        raise FileNotFoundError(f"Trial config not found: {trial_yaml_path}")

    # Validate the trial config the same way training would: this catches
    # mutex violations, missing required keys, broken schedule paths, etc.
    # ``require_data_schedule=False`` so eval-only trials still snapshot.
    trial = TrialConfig.load(trial_yaml_path, require_data_schedule=False)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{timestamp}_{_sanitize_for_dirname(trial.trial_name)}"

    out_root_path = Path(out_root)
    if not out_root_path.is_absolute():
        out_root_path = REPO_ROOT / out_root_path
    snap_dir = out_root_path / name
    snap_dir.mkdir(parents=True, exist_ok=False)

    closure = _collect_referenced_configs(trial_yaml_path, trial.raw)
    entries = _gather_files(closure)

    zip_path = snap_dir / "snapshot.zip"
    uncompressed_total = 0
    included_files: list[tuple[str, int]] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src_abs, arcname in entries:
            zf.write(src_abs, arcname)
            size = src_abs.stat().st_size
            included_files.append((arcname, size))
            uncompressed_total += size

    zip_sha = _sha256_of_file(zip_path)
    zip_size = zip_path.stat().st_size

    manifest = {
        "snapshot_name": name,
        "trial_name": trial.trial_name,
        "source_trial_path": str(trial_yaml_path.relative_to(REPO_ROOT))
            if trial_yaml_path.is_relative_to(REPO_ROOT) else str(trial_yaml_path),
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "created_by": getpass.getuser(),
        "hostname": socket.gethostname(),
        "python_version": sys.version.split()[0],
        "repo_root": str(REPO_ROOT),
        "zip_sha256": zip_sha,
        "zip_size_bytes": zip_size,
        "uncompressed_total_bytes": uncompressed_total,
        "file_count": len(included_files),
        "note": note,
        **_git_provenance(),
        "included_files": [
            {"path": p, "size": s} for p, s in included_files
        ],
    }
    manifest_path = snap_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as manifest_file_writer:
        json.dump(manifest, manifest_file_writer, indent=2)
        manifest_file_writer.write("\n")

    return SnapshotResult(
        snapshot_dir=snap_dir,
        zip_path=zip_path,
        manifest_path=manifest_path,
        included_files=included_files,
        zip_sha256=zip_sha,
        zip_size_bytes=zip_size,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an immutable snapshot of the code + configs feeding a "
            "single trial config. Prints the snapshot directory path on stdout."
        ),
    )
    parser.add_argument(
        "--trial", required=True,
        help="Path to the trial config YAML (e.g. configs/trial_cfgs/trial_cfg_1.yaml).",
    )
    parser.add_argument(
        "--out", default="snapshots",
        help="Root directory where the snapshot dir is created (default: snapshots/ at repo root).",
    )
    parser.add_argument(
        "--note", default=None,
        help="Optional free-text note stored in manifest.json.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress info-level logging; only print the snapshot path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    try:
        result = make_snapshot(args.trial, out_root=args.out, note=args.note)
    except Exception as exc:
        logger.error("Snapshot failed: %s", exc)
        return 1

    if not args.quiet:
        logger.info(
            "Snapshot created: %s (%d files, zip=%d bytes, sha256=%s)",
            result.snapshot_dir,
            len(result.included_files),
            result.zip_size_bytes,
            result.zip_sha256[:12],
        )
    # The orchestrator parses the last line of stdout to capture the path.
    print(result.snapshot_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
