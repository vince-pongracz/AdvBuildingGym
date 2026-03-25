"""
Environment creator factory function for Ray RLlib/Tune.

This module provides the factory function used by Ray Tune to create
AdvBuildingGym environment instances with the configured settings.
"""

import gymnasium

from gymnasium.wrappers import RescaleAction

from .building_adv import AdvBuildingGym
from .wrappers import FlattenAction


def wrap_action_space(env: gymnasium.Env) -> gymnasium.Env:
    """Apply FlattenAction + RescaleAction wrapper chain.

    The resulting env exposes a flat Box(-1, 1) action space to the policy
    while the inner AdvBuildingGym receives named Dict actions with real
    component bounds.
    """
    env = FlattenAction(env)
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    return env


def adv_building_env_creator(config: dict) -> gymnasium.Env:
    """
    Factory function for Ray Tune to create AdvBuildingGym instances.

    This function is registered with Ray Tune and called whenever a new
    environment instance is needed (e.g., for env runners, evaluation).

    IMPORTANT: Uses factory methods to create FRESH component instances for each
    environment. This ensures parallel env_runners don't share mutable state
    (iteration counters, internal buffers) which would cause state corruption.

    Args:
        config: Configuration dict passed by Ray Tune. Supports:
            - ``data_combinator``: Pre-built DataCombinator instance to share
            across environments (loaded from YAML by the training entrypoint).
            - ``log_full_info``: When True, enables deep-copy of named state
            into info["state"] each step. Intended for evaluation EnvRunners
            only (set via ``config.evaluation(env_config=...)``).

    Returns:
        Wrapped AdvBuildingGym with flat Box(-1, 1) action space
    """
    # Import config here to avoid circular imports
    from ..config import config as env_config

    # Create fresh instances for this environment using factory methods.
    # Each env gets its own infras/statesources/rewards with independent state.
    infras = env_config.create_infras()
    statesources = env_config.create_statesources()
    rewards = env_config.create_rewards()

    env = AdvBuildingGym(
        infras=infras,
        statesources=statesources,
        rewards=rewards,
        building_props=env_config.building_props,
        data_combinator=config.get("data_combinator"),
    )

    # Set by Ray's evaluation env_config — only eval EnvRunners pass this.
    if config.get("log_full_info", False):
        env.log_full_info = True

    return wrap_action_space(env)
