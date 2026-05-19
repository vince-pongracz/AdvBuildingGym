"""Snapshot the RLModule subtree of a Ray checkpoint into the eval output dir."""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

RL_MODULE_SUBPATH = "learner_group/learner/rl_module/default_policy"


def copy_rl_module(checkpoint_path: Path, output_dir: Path) -> None:
    """Copy the RLModule subtree used for inference next to trajectories.hdf5.

    Mirrors the layout under ``output_dir/rl_module/`` so the snapshot can
    be reloaded with ``RLModule.from_checkpoint(output_dir/'rl_module')``.
    """
    src = checkpoint_path / RL_MODULE_SUBPATH
    if not src.is_dir():
        logger.warning(
            "RLModule path %s not found — skipping model snapshot.", src,
        )
        return
    dst = output_dir / "rl_module"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    logger.info("Saved RLModule snapshot: %s", dst)
