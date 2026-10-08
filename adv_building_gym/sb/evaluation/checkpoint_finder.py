"""Checkpoint discovery for Stable-Baselines3 models.

SB3 saves models as ``.zip`` files (via ``model.save``), unlike Ray's
checkpoint directories. The training driver writes them under
``models/<trial>/sb3/<algo>/<run>/`` as:

* ``best/best_model.zip``                       (best-by-eval-mean-reward)
* ``checkpoints/ckpt_*.zip`` + ``best_checkpoint_metadata.json``
  (best-by-metric, top-N; the JSON points at the current best)
* ``final_model.zip``                           (last policy)

The resolver mirrors ``adv_building_gym.ray.utils.checkpoint_finder`` but
prefers the *best* model by default (eval wants the best policy, not the
latest), falling back to the most recent ``.zip`` by mtime.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


def _latest_by_mtime(paths: list[str]) -> str:
    """Return the most recently modified path."""
    return max(paths, key=os.path.getmtime)


def _find_sb_model(base_path: str, *, prefer_best: bool) -> str:
    """Locate an SB3 ``.zip`` model under ``base_path``.

    When ``prefer_best`` is True, prefer (1) the most recent
    ``best/best_model.zip``, then (2) the ``best_checkpoint`` pointer in the
    newest ``best_checkpoint_metadata.json``; otherwise fall back to the most
    recent ``.zip`` overall.

    Raises:
        FileNotFoundError: If no ``.zip`` model is found.
    """
    best_models: list[str] = []
    meta_files: list[str] = []
    other_zips: list[str] = []

    for root, _, files in os.walk(base_path):
        for fname in files:
            full = os.path.join(root, fname)
            if fname == "best_model.zip":
                best_models.append(full)
            elif fname == "best_checkpoint_metadata.json":
                meta_files.append(full)
            elif fname.endswith(".zip"):
                other_zips.append(full)

    if prefer_best:
        if best_models:
            chosen = _latest_by_mtime(best_models)
            logger.info("Using best SB3 model: %s", chosen)
            return chosen
        # Fall back to the best-by-metric pointer written by SBBestCheckpointCallback.
        for meta in sorted(meta_files, key=os.path.getmtime, reverse=True):
            try:
                with open(meta, encoding="utf-8") as f:
                    best_ckpt = json.load(f).get("best_checkpoint")
            except (OSError, ValueError) as exc:
                logger.warning("Could not read %s: %s", meta, exc)
                continue
            if best_ckpt and os.path.isfile(best_ckpt):
                logger.info("Using best SB3 checkpoint (from metadata): %s", best_ckpt)
                return best_ckpt

    all_zips = other_zips + best_models
    if not all_zips:
        raise FileNotFoundError(f"No SB3 .zip model found in {base_path}")

    latest = _latest_by_mtime(all_zips)
    logger.info("Found %d SB3 model(s), using latest: %s", len(all_zips), latest)
    return latest


def resolve_sb_checkpoint_path(
    checkpoint: str | None,
    trial_name: str,
    algorithm: str,
    models_base: str = "models",
) -> str:
    """Resolve a Stable-Baselines3 model path (always absolute).

    An explicit *checkpoint* (a ``.zip`` path, with or without the extension)
    is used directly. The sentinels ``"best"`` / ``"latest"`` and ``None``
    trigger a search under ``models_base/<trial_name>/sb3/<algorithm>``. A
    missing directory raises rather than broadening the search across all of
    ``models_base`` — that would silently pick a model from an unrelated
    trial/algorithm whose spaces need not match the eval env. ``None`` and
    ``"best"`` prefer the best-by-eval model; ``"latest"`` takes the most
    recent ``.zip`` by mtime.

    Args:
        checkpoint: Explicit path, ``"best"``, ``"latest"``, or ``None``.
        trial_name: Trial identifier (used to build the search path).
        algorithm: Algorithm name (``"ppo"`` / ``"sac"``).
        models_base: Root models directory.

    Returns:
        Absolute path to the resolved ``.zip`` model.

    Raises:
        FileNotFoundError: If the trial/algorithm directory is absent or holds
            no ``.zip`` model.
    """
    if checkpoint is not None and checkpoint not in ("best", "latest"):
        return os.path.abspath(checkpoint)

    search_base = os.path.join(models_base, trial_name, "sb3", algorithm)
    prefer_best = checkpoint != "latest"  # None + "best" → prefer best

    if not os.path.isdir(search_base):
        raise FileNotFoundError(
            f"No SB3 model directory for this trial/algorithm: {search_base}. "
            "Train it first, or pass an explicit --checkpoint."
        )
    logger.info("Searching for %s SB3 model in: %s",
                "best" if prefer_best else "latest", search_base)
    return os.path.abspath(_find_sb_model(search_base, prefer_best=prefer_best))
