"""Stable-Baselines3 training utilities for AdvBuildingGym.

Parallel to ``adv_building_gym.ray.training`` and ``adv_building_gym.ray.callbacks``,
this package hosts the SB3-specific pieces (env factory, model selection,
common setup, callbacks) that consume the same trial-config layout as the
Ray driver but stay framework-agnostic to RLlib.
"""

from .env_creator import make_sb_env_factory, build_vec_env
from .training import sb_select_model, sb_common_model_setup

__all__ = [
    "make_sb_env_factory",
    "build_vec_env",
    "sb_select_model",
    "sb_common_model_setup",
]
