"""Checkpoint discovery utilities for Ray/RLlib trained models.

Provides functions to locate the latest checkpoint directory within the
models/ tree, plus a high-level resolver used by the evaluation CLI.

Checkpointing is handled by Ray Tune's CheckpointConfig (periodic +
best-model tracking via checkpoint_score_attribute).  These utilities
discover checkpoints by the ``rllib_checkpoint.json`` marker that Ray
writes at each checkpoint root.
"""

import logging
import os

logger = logging.getLogger(__name__)


def _find_latest_checkpoint(base_path: str = "models") -> str:
    """Find the most recent Ray checkpoint by modification time.

    Identifies checkpoint root directories by the presence of
    ``rllib_checkpoint.json`` (new API stack) or ``.is_checkpoint``
    (older Ray versions).

    Args:
        base_path: Root directory to search.

    Returns:
        Path to the most recent checkpoint directory.

    Raises:
        FileNotFoundError: If no checkpoints are found.
    """
    checkpoint_paths = []

    for root, _, files in os.walk(base_path):
        # rllib_checkpoint.json is the authoritative marker written at
        # the checkpoint root by Ray Tune.
        is_checkpoint_root = (
            "rllib_checkpoint.json" in files
            or ".is_checkpoint" in files
        )
        if is_checkpoint_root:
            mtime = os.path.getmtime(root)
            checkpoint_paths.append((mtime, root))

    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {base_path}")

    # Sort by modification time descending — most recent checkpoint first.
    checkpoint_paths.sort(key=lambda entry: entry[0], reverse=True)
    latest_checkpoint = checkpoint_paths[0][1]

    logger.info(
        "Found %d checkpoints, using latest: %s",
        len(checkpoint_paths),
        latest_checkpoint,
    )
    return latest_checkpoint


def resolve_checkpoint_path(
    checkpoint: str | None,
    config_name: str,
    algorithm: str,
    models_base: str = "models",
) -> str:
    # TODO VP 2026.04.29. : Extend it so, that latest checkpoint within a specific training trial can be found
    """Resolve a checkpoint path using a two-step fallback strategy.

    1. If *checkpoint* is provided explicitly, use it directly.
    2. Otherwise, search ``models_base/{config_name}/ray/{algorithm}`` for
       the latest checkpoint by mtime.  If the algorithm-specific directory
       doesn't exist, broaden the search to the entire *models_base* tree.

    The returned path is always absolute.

    Args:
        checkpoint: Explicit checkpoint path, ``"latest"``, or ``None``
            to auto-discover.
        config_name: Configuration name (used to build the search path).
        algorithm: Algorithm name (e.g., ``"ppo"``, ``"sac"``).
        models_base: Root models directory.

    Returns:
        Absolute path to the resolved checkpoint directory.

    Raises:
        FileNotFoundError: If no checkpoint can be found.
    """
    if checkpoint is not None and checkpoint != "latest":
        return os.path.abspath(checkpoint)

    search_base = os.path.join(models_base, config_name, "ray", algorithm)

    if os.path.exists(search_base):
        logger.info("Searching for latest checkpoint in: %s", search_base)
        return os.path.abspath(_find_latest_checkpoint(search_base))

    logger.warning("Algorithm directory not found: %s — broadening search", search_base)
    return os.path.abspath(_find_latest_checkpoint(models_base))
