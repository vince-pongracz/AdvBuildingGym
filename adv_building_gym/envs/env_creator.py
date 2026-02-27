"""
Environment creator factory function for Ray RLlib/Tune.

This module provides the factory function used by Ray Tune to create
AdvBuildingGym environment instances with the configured settings.
"""

from .building_adv import AdvBuildingGym


def adv_building_env_creator(config: dict) -> AdvBuildingGym:
    """
    Factory function for Ray Tune to create AdvBuildingGym instances.

    This function is registered with Ray Tune and called whenever a new
    environment instance is needed (e.g., for env runners, evaluation).

    IMPORTANT: Uses factory methods to create FRESH component instances for each
    environment. This ensures parallel env_runners don't share mutable state
    (iteration counters, internal buffers) which would cause state corruption.

    Args:
        config: Configuration dict passed by Ray Tune (currently unused,
                environment config is loaded from config module)

    Returns:
        AdvBuildingGym instance with independent component instances
    """
    # Import config here to avoid circular imports
    from ..config import config as env_config

    # Create fresh instances for this environment using factory methods.
    # Each env gets its own infras/statesources/rewards with independent state.
    infras = env_config.create_infras()
    statesources = env_config.create_statesources()
    rewards = env_config.create_rewards(infras)

    return AdvBuildingGym(
        infras=infras,
        statesources=statesources,
        rewards=rewards,
        building_props=env_config.building_props,
        data_combinator=env_config.data_combinator,
        # Pass log_full_info from config if available
        # TODO VP 2026.02.24. : Clean up log_full info and trajectory logging, this is a bit hacky
        log_full_info=config.get("log_full_info", False)
    )
