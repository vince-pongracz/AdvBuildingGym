"""Checkpoint discovery utilities for Ray/RLlib trained models.

Provides functions to locate the best or latest checkpoint directory
within the models/ tree, plus a high-level resolver that encapsulates
the multi-step fallback logic used by the evaluation CLI.
"""

import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def find_best_checkpoint(base_path: str) -> str:
    """Find the best-performing Ray checkpoint via best_checkpoint_metadata.json.

    Searches for metadata files saved by BestModelCheckpointCallback during
    training, selects the one with the highest metric value.

    Args:
        base_path: Root directory to search (e.g., models/{config}/ray/{algo})

    Returns:
        Path to the best checkpoint directory.

    Raises:
        FileNotFoundError: If no checkpoint metadata is found.
    """
    candidates = []

    for root, dirs, files in os.walk(base_path):
        if "best_checkpoint_metadata.json" in files:
            metadata_path = os.path.join(root, "best_checkpoint_metadata.json")
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                metric_value = metadata.get("best_metric_value", -np.inf)
                checkpoint_path = metadata.get("checkpoint_path", "")
                if checkpoint_path and os.path.exists(checkpoint_path):
                    candidates.append((metric_value, checkpoint_path, metadata))
                    logger.info(
                        "  Found checkpoint: %s=%s, path=%s",
                        metadata.get("metric", "unknown"),
                        metric_value,
                        checkpoint_path,
                    )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read metadata %s: %s", metadata_path, e)

    if not candidates:
        raise FileNotFoundError(
            f"No best_checkpoint_metadata.json found in {base_path}"
        )

    # Sort by metric value descending, pick best
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_value, best_path, best_metadata = candidates[0]

    logger.info(
        "Selected best checkpoint: %s=%.4f, episode=%d, path=%s",
        best_metadata.get("metric", "unknown"),
        best_value,
        best_metadata.get("episode", -1),
        best_path,
    )
    return best_path


def find_latest_checkpoint(base_path: str = "models") -> str:
    """Fallback: find the most recent Ray checkpoint by modification time.

    Args:
        base_path: Root directory to search.

    Returns:
        Path to the most recent checkpoint directory.

    Raises:
        FileNotFoundError: If no checkpoints are found.
    """
    checkpoint_paths = []

    for root, _, files in os.walk(base_path):
        # Ray checkpoints contain either .pkl files (older) or
        # algorithm_state.pkl / .is_checkpoint marker files (newer)
        is_checkpoint = (
            "checkpoint_" in root
            and any(
                f.endswith(".pkl") or f == ".is_checkpoint"
                for f in files
            )
        ) or (
            # Callback-saved best_model checkpoints
            "best_model_" in os.path.basename(root)
            and any(f.endswith(".pkl") or f == ".is_checkpoint" for f in files)
        )

        if is_checkpoint:
            mtime = os.path.getmtime(root)
            checkpoint_paths.append((mtime, root))

    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {base_path}")

    # Sort by modification time and return most recent
    checkpoint_paths.sort(reverse=True)
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
    """Resolve a checkpoint path using a three-step fallback strategy.

    1. If *checkpoint* is provided explicitly, use it directly.
    2. Otherwise, search ``models_base/{config_name}/ray/{algorithm}`` for the
       best checkpoint (via metadata), falling back to the latest by mtime.
    3. If the algorithm-specific directory doesn't exist, broaden the search
       to the entire *models_base* tree.

    The returned path is always absolute.

    Args:
        checkpoint: Explicit checkpoint path, or ``None`` to auto-discover.
        config_name: Configuration name (used to build the search path).
        algorithm: Algorithm name (e.g., ``"ppo"``, ``"sac"``).
        models_base: Root models directory.

    Returns:
        Absolute path to the resolved checkpoint directory.

    Raises:
        FileNotFoundError: If no checkpoint can be found.
    """
    if checkpoint is not None:
        return os.path.abspath(checkpoint)

    search_base = os.path.join(models_base, config_name, "ray", algorithm)

    if os.path.exists(search_base):
        try:
            logger.info("Searching for best checkpoint in: %s", search_base)
            path = find_best_checkpoint(search_base)
        except FileNotFoundError:
            logger.info(
                "No best checkpoint metadata found, falling back to latest checkpoint"
            )
            path = find_latest_checkpoint(search_base)
    else:
        logger.warning("Algorithm directory not found: %s", search_base)
        logger.info("Searching in all models...")
        try:
            path = find_best_checkpoint(models_base)
        except FileNotFoundError:
            path = find_latest_checkpoint(models_base)

    return os.path.abspath(path)
