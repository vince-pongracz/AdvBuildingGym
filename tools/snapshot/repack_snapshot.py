"""Repack a snapshot's extracted ``code/`` tree back into ``snapshot.zip``.

A snapshot's zip is the immutable artifact; ``code/`` is only the lazily
extracted working copy (see ``slurm_scripts/util/snapshot_mode.sh``). When
``code/`` is edited by hand the two drift apart, and the drift is silent:
``submit_snapshot._load_snapshot_trial_text`` prefers ``code/`` while the
wrapper re-extracts from the zip whenever ``code/`` is missing — so the same
snapshot can train two different configs depending on whether ``code/``
happens to exist.

This tool makes the zip match ``code/`` again and refreshes the manifest's
integrity fields (sha256, sizes, file list). It does NOT re-run the trial
validation ``make_snapshot`` does; it is a mechanical repack.

CLI:
    python -m tools.snapshot.repack_snapshot snapshots/<snap> [snapshots/<snap2> ...]
    python -m tools.snapshot.repack_snapshot snapshots/<snap> --dry-run
    python -m tools.snapshot.repack_snapshot snapshots/*_sac --note "code/ edited to SAC"
"""

from __future__ import annotations

import argparse
import datetime
import getpass
import json
import logging
import os
import shutil
import socket
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.snapshot.make_snapshot import (  # noqa: E402  (path setup above)
    EXCLUDED_DIR_NAMES,
    EXCLUDED_DIR_SUFFIXES,
    EXCLUDED_FILE_SUFFIXES,
    _sha256_of_file,
)

logger = logging.getLogger("repack_snapshot")


@dataclass
class RepackResult:
    """What ``repack_snapshot()`` returns for one snapshot."""

    snapshot_dir: Path
    zip_path: Path
    backup_path: Path | None
    file_count: int
    zip_sha256: str
    zip_size_bytes: int
    uncompressed_total_bytes: int
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Walking code/
# ---------------------------------------------------------------------------

def _iter_code_files(code_dir: Path) -> Iterable[Path]:
    """Yield every regular file under ``code/``, applying make_snapshot's filters.

    Symlinks are skipped entirely: ``snapshot_mode.sh`` drops a ``code/data``
    link pointing at the live repo's CSV tree, which must never be bundled
    (data is deliberately outside the snapshot).
    """
    for root, dirs, files in os.walk(code_dir):  # followlinks=False → symlinked dirs are not descended
        dirs[:] = [
            dir for dir in dirs
            if dir not in EXCLUDED_DIR_NAMES
            and not any(dir.endswith(suffix) for suffix in EXCLUDED_DIR_SUFFIXES)
            and not (Path(root) / dir).is_symlink()
        ]
        for name in files:
            if any(name.endswith(suffix) for suffix in EXCLUDED_FILE_SUFFIXES):
                continue
            path = Path(root) / name
            if path.is_symlink():
                logger.debug("Skipping symlink: %s", path)
                continue
            yield path


def _gather_code_entries(code_dir: Path) -> list[tuple[Path, str]]:
    """Build the (absolute_source, arcname) list for the new zip.

    Arcnames are relative to ``code/`` with forward slashes — the same
    repo-relative form ``make_snapshot`` writes, which is what
    ``manifest["source_trial_path"]`` and the wrapper's ``unzip -d code/``
    both assume.
    """
    entries = [(path, path.relative_to(code_dir).as_posix()) for path in _iter_code_files(code_dir)]
    entries.sort(key=lambda entry: entry[1])
    return entries


def _diff_against_zip(zip_path: Path, entries: list[tuple[Path, str]]) -> tuple[list[str], list[str], list[str]]:
    """Return (added, removed, changed) arcnames of code/ versus the existing zip.

    Content comparison uses the CRC32 the zip already stores per member, so no
    decompression is needed.
    """
    if not zip_path.exists():
        return [arc for _, arc in entries], [], []

    import zlib  # local: only needed for the diff report

    with zipfile.ZipFile(zip_path) as zf:
        old_crcs = {info.filename: info.CRC for info in zf.infolist() if not info.is_dir()}

    new_arcnames = {arc for _, arc in entries}
    added = sorted(new_arcnames - old_crcs.keys())
    removed = sorted(old_crcs.keys() - new_arcnames)

    changed: list[str] = []
    for src_abs, arc in entries:
        if arc not in old_crcs:
            continue
        if zlib.crc32(src_abs.read_bytes()) & 0xFFFFFFFF != old_crcs[arc]:
            changed.append(arc)
    return added, removed, sorted(changed)


# ---------------------------------------------------------------------------
# Repack
# ---------------------------------------------------------------------------

def repack_snapshot(
    snapshot_dir: str | Path,
    *,
    note: str | None = None,
    backup: bool = True,
    dry_run: bool = False,
) -> RepackResult:
    """Rebuild ``<snapshot>/snapshot.zip`` from ``<snapshot>/code/`` and update the manifest.

    Raises if ``code/`` or ``manifest.json`` is missing, or if the manifest's
    ``source_trial_path`` no longer resolves inside ``code/`` (which would
    leave the snapshot unsubmittable).
    """
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.is_absolute():
        snapshot_dir = (REPO_ROOT / snapshot_dir).resolve()

    code_dir = snapshot_dir / "code"
    manifest_path = snapshot_dir / "manifest.json"
    zip_path = snapshot_dir / "snapshot.zip"
    if not code_dir.is_dir():
        raise FileNotFoundError(f"No extracted code/ tree to repack: {code_dir}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json missing from {snapshot_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trial_rel = manifest.get("source_trial_path")
    if not trial_rel:
        raise ValueError(f"manifest.json has no source_trial_path: {manifest_path}")
    if not (code_dir / trial_rel).exists():
        raise FileNotFoundError(f"source_trial_path not present in code/: {code_dir / trial_rel}")

    entries = _gather_code_entries(code_dir)
    if not entries:
        raise ValueError(f"code/ contains no packable files: {code_dir}")
    added, removed, changed = _diff_against_zip(zip_path, entries)
    uncompressed_total = sum(src.stat().st_size for src, _ in entries)

    if dry_run:
        logger.info(
            "[dry-run] %s: would pack %d files (%d bytes uncompressed); +%d / -%d / ~%d vs current zip",
            snapshot_dir.name, len(entries), uncompressed_total, len(added), len(removed), len(changed),
        )
        return RepackResult(
            snapshot_dir=snapshot_dir, zip_path=zip_path, backup_path=None,
            file_count=len(entries), zip_sha256="", zip_size_bytes=0,
            uncompressed_total_bytes=uncompressed_total,
            added=added, removed=removed, changed=changed,
        )

    # Back up the old zip before overwriting: it is the only remaining copy of
    # whatever code/ diverged from.
    backup_path: Path | None = None
    if backup and zip_path.exists():
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = zip_path.with_suffix(f".zip.bak-{stamp}")
        shutil.copy2(zip_path, backup_path)
        logger.info("Backed up old zip → %s", backup_path.name)

    # Write to a temp file first so an interrupted run cannot leave a truncated
    # snapshot.zip behind (the wrapper would then extract garbage).
    tmp_zip = zip_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src_abs, arcname in entries:
            zf.write(src_abs, arcname)
    tmp_zip.replace(zip_path)

    zip_sha = _sha256_of_file(zip_path)
    zip_size = zip_path.stat().st_size

    manifest["zip_sha256"] = zip_sha
    manifest["zip_size_bytes"] = zip_size
    manifest["uncompressed_total_bytes"] = uncompressed_total
    manifest["file_count"] = len(entries)
    manifest["included_files"] = [{"path": arc, "size": src.stat().st_size} for src, arc in entries]
    # Keep the original creation provenance (created_at / git_*) intact — it
    # still describes where the snapshot came from — and append the repack as
    # its own audit trail entry.
    manifest.setdefault("repacks", []).append({
        "repacked_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "repacked_by": getpass.getuser(),
        "hostname": socket.gethostname(),
        "source": "code/",
        "files_added": added,
        "files_removed": removed,
        "files_changed": changed,
        "backup_zip": backup_path.name if backup_path else None,
        "note": note,
    })
    with manifest_path.open("w", encoding="utf-8") as manifest_file_writer:
        json.dump(manifest, manifest_file_writer, indent=2)
        manifest_file_writer.write("\n")

    logger.info(
        "Repacked %s: %d files, zip=%d bytes, sha256=%s (+%d / -%d / ~%d)",
        snapshot_dir.name, len(entries), zip_size, zip_sha[:12], len(added), len(removed), len(changed),
    )
    return RepackResult(
        snapshot_dir=snapshot_dir, zip_path=zip_path, backup_path=backup_path,
        file_count=len(entries), zip_sha256=zip_sha, zip_size_bytes=zip_size,
        uncompressed_total_bytes=uncompressed_total,
        added=added, removed=removed, changed=changed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild snapshot.zip from a snapshot's extracted code/ tree so the "
            "immutable artifact matches what runs, and refresh manifest.json's "
            "integrity fields."
        ),
    )
    parser.add_argument(
        "snapshots", nargs="+",
        help="One or more snapshot directories (e.g. snapshots/20260718_143903_...).",
    )
    parser.add_argument(
        "--note", default=None,
        help="Free-text note recorded in the manifest's repacks[] entry.",
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="Overwrite snapshot.zip without keeping a .zip.bak-<timestamp> copy.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be packed and how it differs from the current zip.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="List every added / removed / changed arcname instead of just counts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    exit_code = 0
    for snapshot in args.snapshots:
        try:
            result = repack_snapshot(
                snapshot,
                note=args.note,
                backup=not args.no_backup,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            logger.error("Repack failed for %s: %s", snapshot, exc)
            exit_code = 1
            continue
        if args.verbose:
            for label, paths in (("added", result.added), ("removed", result.removed), ("changed", result.changed)):
                for arc in paths:
                    logger.info("  %-7s %s", label, arc)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
