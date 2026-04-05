"""Inference utilities for Ray/RLlib trained RL modules.

Provides functions to load an RLModule from a checkpoint directory,
flatten dictionary observations (mirroring the FlattenObservations
connector used during training), and run deterministic forward inference.
"""

import logging

import numpy as np
import torch
from ray.rllib.core.rl_module.rl_module import RLModule

logger = logging.getLogger(__name__)


def load_rl_module(checkpoint_path: str) -> RLModule:
    """Load an RLModule from a Ray/RLlib checkpoint.

    Appends the standard sub-path to reach the default single-agent policy
    within the checkpoint directory tree.

    Args:
        checkpoint_path: Path to the top-level Ray checkpoint directory
            (e.g., ``checkpoint_000123``).

    Returns:
        The restored ``RLModule`` ready for inference.
    """
    full_path = (
        f"{checkpoint_path}/learner_group/learner/rl_module/default_policy"
    )
    logger.info("Loading RLModule from: %s", full_path)
    return RLModule.from_checkpoint(full_path)


def infer_action(rl_module: RLModule, flat_obs: np.ndarray) -> np.ndarray:
    """Run deterministic forward inference on a single observation.

    Builds a batched tensor, calls ``_forward_inference``, and handles
    both ``action_dist_inputs`` (SAC/PPO squashed-Gaussian) and direct
    ``actions`` outputs.
    Link: https://docs.ray.io/en/latest/rllib/package_ref/rl_modules.html

    Args:
        rl_module: A loaded ``RLModule``.
        flat_obs: 1-D float32 observation array (already flattened).

    Returns:
        Action as a numpy array (or scalar wrapped in 0-d array).
    """
    batch = {
        "obs": torch.as_tensor(flat_obs, dtype=torch.float32).unsqueeze(0),
    }
    with torch.no_grad():
        output = rl_module._forward_inference(batch)  # type: ignore[attr-defined]

        # SAC / PPO return "action_dist_inputs" (mean + log_std concatenated).
        # Extract deterministic action by taking the mean (first half) and
        # applying tanh squash.
        if "action_dist_inputs" in output:
            dist_inputs = output["action_dist_inputs"].squeeze(0)
            # For continuous: dist_inputs = [mean, log_std] concatenated
            action_dim = dist_inputs.shape[-1] // 2
            action_mean = dist_inputs[:action_dim]
            # Deterministic inference: use tanh(mean) for squashed Gaussian
            raw_action = torch.tanh(action_mean).numpy()
        else:
            raw_action = output["actions"].squeeze(0).numpy()

        if isinstance(raw_action, np.ndarray) and raw_action.ndim == 0:
            raw_action = raw_action.item()

    return raw_action
