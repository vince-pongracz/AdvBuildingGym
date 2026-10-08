"""Cross-runtime warning filters (no Ray imports).

Suppresses Gymnasium / h5py noise that affects every driver, regardless of
which RL framework is in use. The Ray-specific filters live in
``adv_building_gym/ray/utils/warning_filters.py`` and are layered on top.
"""

import logging
import warnings

# Silence h5py DEBUG chatter from its type-conversion table at import; workers set
# the root logger to DEBUG, so it would otherwise bleed into worker logs.
logging.getLogger("h5py").setLevel(logging.WARNING)


def setup_warning_filters() -> None:
    """Configure cross-runtime warning filters (gymnasium + h5py); call at any driver's startup.
    Ray drivers also layer the Ray-specific filters on top.
    """
    # Gymnasium Box precision warnings (float64 to float32 casting)
    warnings.filterwarnings(
        "ignore",
        message=r".*Box.*precision lowered.*",
        category=UserWarning,
    )

    # Gymnasium passive env checker warnings
    warnings.filterwarnings(
        "ignore",
        message=r".*obs returned by the.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*is not within the observation space.*",
        category=UserWarning,
    )
