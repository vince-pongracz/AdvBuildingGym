"""Provenance helpers: capture trial config + git state next to eval outputs."""

import hashlib
import logging
import shutil
from pathlib import Path

import yaml

from .git_info import git_commit, git_is_dirty

logger = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_trial_yaml(trial_yaml_path: str | Path | None, output_dir: Path) -> None:
    """Copy the trial YAML alongside the eval results, if it exists."""
    if trial_yaml_path is None:
        return
    src = Path(trial_yaml_path)
    if src.is_file():
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        logger.info("Saved trial config: %s", dst)
    else:
        logger.warning("trial_yaml_path %s is not a file — skipping copy.", src)


def write_provenance(
    output_dir: Path,
    trial_yaml_path: str | Path | None,
    repo_dir: Path,
) -> None:
    """Write a small ``provenance.yaml`` next to ``trajectories.hdf5``."""
    record: dict = {}
    if trial_yaml_path is not None:
        src = Path(trial_yaml_path)
        if src.is_file():
            record["trial_yaml"] = {
                "name": src.name,
                "sha256": sha256(src),
            }
    commit = git_commit(repo_dir)
    if commit is not None:
        record["git"] = {"commit": commit}
        dirty = git_is_dirty(repo_dir)
        if dirty is not None:
            record["git"]["dirty"] = dirty
    out_path = output_dir / "provenance.yaml"
    with out_path.open("w") as f:
        yaml.safe_dump(record, f, sort_keys=False)
    logger.info("Saved provenance: %s", out_path)
