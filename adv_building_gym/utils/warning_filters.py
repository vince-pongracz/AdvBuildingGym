"""
Warning filter setup for Ray workers.

This module provides a centralized function to configure warning filters
that should be applied across all Ray worker processes (env_runners, learners).
"""

import warnings
from ray.util.annotations import RayDeprecationWarning


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
