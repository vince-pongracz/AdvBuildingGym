"""Ray-specific warning filter setup.

Layered on top of :mod:`adv_building_gym._common.warning_filters`: calling
``setup_warning_filters`` here first applies the common (gymnasium / h5py)
filters, then adds the Ray + RLlib-specific ones that need Ray imported.
"""

import warnings

from ray.util.annotations import RayDeprecationWarning

from adv_building_gym._common.warning_filters import (
    setup_warning_filters as _setup_common_warning_filters,
)


def setup_warning_filters() -> None:
    """Configure all warning filters used by Ray drivers (env_runners, learners).

    Suppresses:
    - Common (gymnasium Box precision, passive env checker — see _common module)
    - Ray deprecation warnings (internal Ray issues)
    - RLlib RLModule deprecation warnings (internal RLlib issues)
    - The repeated "running SAC/PPO on the new API stack" notices from
      RLlib algorithm_config.py
    """
    _setup_common_warning_filters()

    # Ray deprecation warnings
    warnings.filterwarnings("ignore", category=RayDeprecationWarning)

    # RLlib internal RLModule deprecation warning
    warnings.filterwarnings(
        "ignore",
        message=r".*RLModule.*deprecated.*",
        category=DeprecationWarning,
    )

    # Repeated "running SAC/PPO on the new API stack" notices from
    # RLlib algorithm_config.py — the project deliberately uses the new stack
    # and the message clutters the startup banner (emitted 3x+ per run).
    warnings.filterwarnings(
        "ignore",
        message=r".*running .* on the new API stack.*",
    )
