"""
Environment creator factory function for Ray RLlib/Tune.

This module provides the factory function used by Ray Tune to create
AdvBuildingGym environment instances with the configured settings.
"""

from .building_adv import AdvBuildingGym


def adv_building_env_creator(config):
    """
    Factory function for Ray Tune to create AdvBuildingGym instances.

    This function is registered with Ray Tune and called whenever a new
    environment instance is needed (e.g., for env runners, evaluation).

    Args:
        config: Configuration dict passed by Ray Tune (currently unused,
                environment config is loaded from env_config module)

    Returns:
        AdvBuildingGym instance configured with the settings from env_config
    """
    # Import env_config here to avoid circular imports
    from ..env_config import config as env_config

    # NOTE VP 2026.01.13. : Warnings about config fields being None are supressed, 
    # because Config initionalises them as None by default, 
    # but fills up later in a post_init step
    # --> that is why the type:ignore comments below
    return AdvBuildingGym(
        infras=env_config.infras, # type: ignore
        datasources=env_config.datasources, # type: ignore
        rewards=env_config.rewards,
        building_props=env_config.building_props,
    )
