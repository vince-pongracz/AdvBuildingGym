"""
Warning filter setup for Ray workers.

This module provides a centralized function to configure warning filters
that should be applied across all Ray worker processes (env_runners, learners).
"""

import logging
import warnings
from ray.util.annotations import RayDeprecationWarning

# Silence h5py DEBUG chatter ("h5py._conv - Creating converter from N to M") emitted
# while h5py builds its type-conversion table at first import. Per-worker package
# __init__ files set the root logger to DEBUG, so without this the messages bleed
# into every Ray worker log. Done at module-import time so the level is set before
# trajectory_utils triggers the h5py import (see adv_building_gym/utils/__init__.py).
logging.getLogger("h5py").setLevel(logging.WARNING)


def setup_warning_filters():
    """
    Configure warning filters to suppress known, expected warnings.

    This function should be called early in the initialization of Ray workers
    (e.g., in the environment __init__) to ensure warnings are suppressed
    in all worker processes, not just the main process.

    Suppressed warnings:
    - Ray deprecation warnings (internal Ray issues)
    - RLlib RLModule deprecation warnings (internal RLlib issues)
    - Gymnasium Box precision warnings (float64 to float32 casting)
    - Gymnasium passive env checker warnings
    """
    # Suppress Ray deprecation warnings
    warnings.filterwarnings("ignore", category=RayDeprecationWarning)

    # Suppress RLlib internal RLModule deprecation warning
    warnings.filterwarnings(
        "ignore",
        message=r".*RLModule.*deprecated.*",
        category=DeprecationWarning
    )

    # Suppress gymnasium Box precision warnings (float64 to float32 casting)
    warnings.filterwarnings(
        "ignore",
        message=r".*Box.*precision lowered.*",
        category=UserWarning
    )

    # Suppress gymnasium passive env checker warnings
    warnings.filterwarnings(
        "ignore",
        message=r".*obs returned by the.*",
        category=UserWarning
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*is not within the observation space.*",
        category=UserWarning
    )

    # Suppress repeated "running SAC/PPO on the new API stack" notices from
    # RLlib algorithm_config.py — the project deliberately uses the new stack
    # and the message clutters the startup banner (emitted 3x+ per run).
    warnings.filterwarnings(
        "ignore",
        message=r".*running .* on the new API stack.*",
    )