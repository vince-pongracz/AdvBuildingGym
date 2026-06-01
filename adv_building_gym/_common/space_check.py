"""Space compatibility checks between RL modules and environments.

Validates that observation and action space dimensions match before
running inference, providing diagnostic messages on mismatch.
"""

import logging
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


def _pipeline_output_obs_size(pipeline, input_obs_space, input_action_space) -> int | None:
    """Propagate the env obs space through the connector pipeline and return the flat output size.

    A connector that does not override ``recompute_output_observation_space``
    inherits the base implementation which returns ``self.input_observation_space``.
    For pipeline-internal pieces (e.g. ``AddObservationsFromEpisodesToBatch``)
    that attribute is never set, so the call returns ``None``. That is a
    pass-through, not a failure — we keep the current ``obs_sp`` and move on.

    Returns None only if the pipeline is empty (no connectors) so the caller
    can fall back to the raw env obs dimension.
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
    """Verify that the model's input/output spaces match the environment.

    Args:
        rl_module: Loaded RLModule with ``observation_space`` and
            ``action_space`` attributes.
        env: Gymnasium environment (possibly wrapped).
        pipeline: Optional env-to-module ConnectorV2 pipeline. When given, the
            env obs is propagated through the pipeline so the check compares
            the model's input against the post-connector flat size. Without
            this the check would compare against the raw env obs and falsely
            fail on any policy trained with a non-trivial connector pipeline.

    Raises:
        ValueError: If observation or action dimensions differ.
    """
    # Model's expected observation size (flat)
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
            f"(e.g. ACTION_HISTORY_LENGTH, hst_env_wrapper_tracked_keys/offsets, "
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
