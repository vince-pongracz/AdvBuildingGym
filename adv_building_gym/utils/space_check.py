"""Space compatibility checks between RL modules and environments.

Validates that observation and action space dimensions match before
running inference, providing diagnostic messages on mismatch.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def check_space_compatibility(rl_module, env) -> None:
    """Verify that the model's input/output spaces match the environment.

    Args:
        rl_module: Loaded RLModule with ``observation_space`` and
            ``action_space`` attributes.
        env: Gymnasium environment (possibly wrapped).

    Raises:
        ValueError: If observation or action dimensions differ.
    """
    # Model's expected observation size (flat)
    model_obs_size = np.prod(rl_module.observation_space.shape)
    # Env's actual observation size (flattened Dict -> flat Box)
    env_obs_size = sum(
        np.prod(space.shape)
        for space in env.observation_space.spaces.values()
    ) if hasattr(env.observation_space, "spaces") else np.prod(env.observation_space.shape)

    if model_obs_size != env_obs_size:
        raise ValueError(
            f"Observation space mismatch: model expects {model_obs_size} features "
            f"but the env produces {env_obs_size} features "
            f"(difference: {env_obs_size - model_obs_size}). "
            f"Check whether the config has changed since training "
            f"(e.g. ACTION_HISTORY_LENGTH, added/removed statesources)."
        )

    model_act_size = np.prod(rl_module.action_space.shape)
    env_act_size = np.prod(env.action_space.shape)
    if model_act_size != env_act_size:
        raise ValueError(
            f"Action space mismatch: model expects {model_act_size} action dims "
            f"but the env has {env_act_size} "
            f"(difference: {env_act_size - model_act_size}). "
            f"Check whether infrastructure components changed since training."
        )

    # TODO VP 2026.03.23. : check reward sizes/dimensions as well

    logger.info(
        "Space check OK: obs=%d, act=%d", model_obs_size, model_act_size,
    )
