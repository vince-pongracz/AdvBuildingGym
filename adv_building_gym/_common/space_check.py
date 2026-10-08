"""Space compatibility checks between RL modules and environments.

Validates that observation and action space dimensions match before
running inference, providing diagnostic messages on mismatch.
"""

import logging
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


def _pipeline_output_obs_size(pipeline, input_obs_space, input_action_space) -> int | None:
    """Propagate the env obs space through the connector pipeline → flat output size.

    A connector returning ``None`` from ``recompute_output_observation_space`` is a
    pass-through (keep the current space). Returns None only for an empty pipeline,
    so the caller can fall back to the raw env obs dim.
    """
    if not pipeline:
        return None
    obs_sp = input_obs_space
    for connector in pipeline:
        out_sp = connector.recompute_output_observation_space(obs_sp, input_action_space)
        if out_sp is not None:
            obs_sp = out_sp

    if hasattr(obs_sp, "spaces"):
        return int(sum(np.prod(s.shape) for s in obs_sp.spaces.values()))
    return int(np.prod(obs_sp.shape))


def check_space_compatibility(rl_module, env, pipeline: Iterable | None = None) -> None:
    """Verify the model's obs/action spaces match the env; raises ValueError on mismatch.

    With a ``pipeline``, env obs is propagated through it so the check compares against the
    post-connector flat size (else it would falsely fail for non-trivial pipelines).
    """
    # Model's expected (flat) observation size
    model_obs_size = int(np.prod(rl_module.observation_space.shape))

    env_obs_size: int | None = None
    if pipeline is not None:
        env_obs_size = _pipeline_output_obs_size(pipeline, env.observation_space, env.action_space)
    if env_obs_size is None:
        # Fallback: raw env observation size (flattened Dict -> flat Box)
        env_obs_size = int(sum(
            np.prod(space.shape)
            for space in env.observation_space.spaces.values()
        )) if hasattr(env.observation_space, "spaces") else int(np.prod(env.observation_space.shape))

    if model_obs_size != env_obs_size:
        raise ValueError(
            f"Observation space mismatch: model expects {model_obs_size} features "
            f"but the env produces {env_obs_size} features "
            f"(difference: {env_obs_size - model_obs_size}). "
            f"Check whether the config has changed since training "
            f"(e.g. ACTION_HISTORY_LENGTH, hst.tracked_keys/offsets, "
            f"added/removed statesources)."
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

    logger.info("Space check OK: obs=%d, act=%d", model_obs_size, model_act_size)
