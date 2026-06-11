"""Inference utilities for trained RLModules: load from a checkpoint and run forward
inference, deterministic (``tanh(mean)``) or stochastic (squashed-Gaussian sample).
"""

import logging

import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule

logger = logging.getLogger(__name__)


def load_rl_module(checkpoint_path: str) -> RLModule:
    """Load an RLModule from a Ray checkpoint dir (appends the default-policy sub-path)."""
    full_path = (
        f"{checkpoint_path}/learner_group/learner/rl_module/default_policy"
    )
    logger.info("Loading RLModule from: %s", full_path)
    return RLModule.from_checkpoint(full_path)


def infer_action(
    rl_module: RLModule,
    flat_obs: np.ndarray,
    stochastic: bool = False,
    generator: torch.Generator | None = None,
    state_in: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Forward inference on a single (flat) observation.

    Handles ``action_dist_inputs`` (SAC/PPO squashed-Gaussian), direct ``actions`` (DreamerV3),
    and recurrent state threading. ``stochastic`` samples ``tanh(mean + exp(log_std)*eps)`` instead
    of ``tanh(mean)``; ``generator`` makes ``eps`` reproducible. ``state_in``: batched recurrent
    state ({} for stateless). Returns (action, state_out).
    Link: https://docs.ray.io/en/latest/rllib/package_ref/rl_modules.html

    Args:
        rl_module: A loaded ``RLModule``.
        flat_obs: 1-D float32 observation array (already flattened).
        stochastic: If True, sample from the squashed Gaussian instead of
            taking ``tanh(mean)``. The policy head outputs
            ``[mean, log_std]`` and we draw
            ``a = tanh(mean + exp(log_std) * eps)``, ``eps ~ N(0, 1)``.
        generator: Optional ``torch.Generator`` controlling ``eps`` so
            stochastic eval is reproducible. Ignored when
            ``stochastic=False`` or when the module returns ``actions`` directly.
        state_in: Recurrent state dict produced by a previous call's
            ``state_out`` (or by ``rl_module.get_initial_state()`` at the
            start of an episode). Pass ``{}`` for stateless modules.

    Returns:
        Tuple of (action, state_out). ``action`` is a numpy array (or
        Python scalar for 0-d outputs). ``state_out`` is ``{}`` for
        stateless modules.
    """
    batch: dict = {
        Columns.OBS: torch.as_tensor(flat_obs, dtype=torch.float32).unsqueeze(0),
    }
    if state_in:
        # RLlib convention: STATE_IN is a (nested) dict of tensors already batched on axis 0;
        # callers add the batch dim (get_initial_state() returns unbatched).
        batch[Columns.STATE_IN] = state_in

    with torch.no_grad():
        output = rl_module._forward_inference(batch)  # type: ignore[attr-defined]

        # SAC / PPO continuous: action_dist_inputs = [mean, log_std].
        if Columns.ACTION_DIST_INPUTS in output:
            dist_inputs = output[Columns.ACTION_DIST_INPUTS].squeeze(0)
            action_dim = dist_inputs.shape[-1] // 2
            action_mean = dist_inputs[:action_dim]
            if stochastic:
                action_log_std = dist_inputs[action_dim:]
                eps = torch.randn(
                    action_dim,
                    generator=generator,
                    dtype=action_mean.dtype,
                )
                raw_action = torch.tanh(
                    action_mean + torch.exp(action_log_std) * eps,
                ).numpy()
            else:
                raw_action = torch.tanh(action_mean).numpy()
        else:
            # DreamerV3 returns sampled actions directly under Columns.ACTIONS.
            raw_action = output[Columns.ACTIONS].squeeze(0).numpy()

        if isinstance(raw_action, np.ndarray) and raw_action.ndim == 0:
            raw_action = raw_action.item()

        state_out = output.get(Columns.STATE_OUT, {}) or {}

    return raw_action, state_out
