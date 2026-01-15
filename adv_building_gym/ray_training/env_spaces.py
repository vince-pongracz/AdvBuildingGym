"""
Environment space utilities for Ray RLlib training.

This module provides utilities for extracting observation and action spaces
from environment instances.
"""

from typing import Callable, Any


def get_env_spaces(env_creator: Callable[[dict], Any]) -> tuple:
    """
    Instantiate env temporarily to extract observation and action spaces.

    This avoids RLlib inferring spaces from remote workers, making the
    configuration explicit and reducing startup overhead.

    Args:
        env_creator: Factory function that creates environment instances

    Returns:
        Tuple of (observation_space, action_space)
    """
    temp_env = env_creator({})
    obs_space = temp_env.observation_space
    action_space = temp_env.action_space
    temp_env.close()
    return obs_space, action_space
