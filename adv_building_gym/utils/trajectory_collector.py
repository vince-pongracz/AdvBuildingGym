"""TrajectoryCollector for standalone evaluation scripts.

Accumulates per-step info dicts during an env.step() loop and produces
structured columnar trajectory JSON via extract_trajectory_from_infos().
"""

import json
import logging
import os

import numpy as np

from .json_encoder import CustomJSONEncoder
from .trajectory_utils import extract_trajectory_from_infos, write_episode_to_hdf5

logger = logging.getLogger(__name__)


class TrajectoryCollector:
    """Collect per-step trajectory data during a manual eval loop.

    Usage::

        collector = TrajectoryCollector(env)
        obs, info = env.reset(seed=42)
        collector.on_reset(info)
        step = 0
        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            collector.on_step(step, obs, action, reward, info, raw_policy_action=action)
            step += 1
            done = terminated or truncated
        collector.on_episode_end(episode_id=0, seed=42)
        collector.save_json("trajectory.json")
    """

    def __init__(self, env) -> None:
        """Extract keys from the environment for trajectory extraction.

        Args:
            env: An AdvBuildingGym instance 
            (or compatible env with reward_funcs and Dict action_space).
        """

        # Extract state and action key names from env spaces
        self.state_keys: list[str] | None = None
        if hasattr(env, "observation_space") and hasattr(env.observation_space, "spaces"):
            self.state_keys = list(env.observation_space.spaces.keys())

        self.action_keys: list[str] | None = None
        # The native Dict action space lives on the unwrapped AdvBuildingGym;
        # wrappers (FlattenAction, RescaleAction) may hide it.
        action_space = getattr(env, "action_space", None)
        if hasattr(action_space, "spaces"):
            self.action_keys = list(action_space.spaces.keys())

        # Reward function names for summary
        self.reward_names: list[str] = []
        if hasattr(env, "reward_funcs"):
            self.reward_names = [r.name for r in env.reward_funcs]
        self.max_reward_per_step: float = 0.0
        if hasattr(env, "reward_funcs"):
            self.max_reward_per_step = sum(r.weight * r.max_reward for r in env.reward_funcs)

        self._initial_info: dict | None = None
        self._step_infos: list[dict] = []
        self._raw_policy_actions: list[np.ndarray] = []
        self._episode_id: int | str | None = None
        self._seed: int | None = None
        self._metadata: dict | None = None

    def on_reset(self, info: dict) -> None:
        """Store the reset info dict (initial conditions)."""
        self._initial_info = info

    def on_step(
        self,
        step: int,
        obs,
        action,
        reward: float,
        info: dict,
        raw_policy_action=None,
    ) -> None:
        """Accumulate one timestep's info dict and optional raw policy action."""
        self._step_infos.append(info)
        if raw_policy_action is not None:
            self._raw_policy_actions.append(np.asarray(raw_policy_action, dtype=np.float32).flatten())

    def on_episode_end(
        self,
        episode_id: int | str,
        seed: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Finalize episode, store metadata."""
        self._episode_id = episode_id
        self._seed = seed
        self._metadata = metadata

    def to_dict(self) -> dict:
        """Build the full trajectory dict ready for JSON serialization.

        Calls extract_trajectory_from_infos() on accumulated infos,
        adds raw policy actions, metadata, and summary statistics.
        """
        trajectory = extract_trajectory_from_infos(
            self._step_infos,
            initial_info=self._initial_info,
            state_keys=self.state_keys,
            action_keys=self.action_keys,
        )

        # Add raw policy actions (flattened) as columnar data
        if self._raw_policy_actions:
            ndim = self._raw_policy_actions[0].size
            # Prepend zeros for the initial-conditions row (if initial_info was provided)
            has_initial = self._initial_info is not None
            for d in range(ndim):
                # TODO VP 2026.02.25. : Adjust this output format, similar to the callback version, to support vector actions without flattening (e.g., "raw_policy_action": list of lists)
                col = f"raw_policy_action_{d}"
                values: list[float] = []
                if has_initial:
                    values.append(0.0)
                for act in self._raw_policy_actions:
                    values.append(float(act.flat[d]))
                trajectory[col] = values

        # Compute summary
        rewards = trajectory.get("reward", [])
        achieved_reward = sum(rewards)
        ep_length = len(self._step_infos)
        max_achievable = ep_length * self.max_reward_per_step
        reward_rate = achieved_reward / max_achievable if max_achievable > 0 else 0.0
        cum_values = trajectory.get("cum_E_kWh", [])
        final_cum_E = cum_values[-1] if cum_values else 0.0

        result = {
            "version": 1,
            "episode_id": self._episode_id,
            "seed": self._seed,
            "length": ep_length,
            "metadata": self._metadata or {},
            "summary": {
                "achieved_reward": float(achieved_reward),
                "max_achievable_reward": float(max_achievable),
                "reward_rate": float(reward_rate),
                "cum_E_kWh": float(final_cum_E),
            },
            "trajectory": trajectory,
        }
        return result

    def save_json(self, filepath: str) -> None:
        """Write the trajectory to a JSON file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data = self.to_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=CustomJSONEncoder, indent=4)
        logger.info("Trajectory saved to %s", filepath)

    def save_hdf5(self, hdf5_path: str, episode_id: str | None = None) -> None:
        """Append this episode's trajectory to an HDF5 file.

        Args:
            hdf5_path: Path to the HDF5 file (created if absent).
            episode_id: Group name in the HDF5 file. Defaults to
                ``self._episode_id`` set via ``on_episode_end()``.
        """
        ep_id = str(episode_id if episode_id is not None else self._episode_id)
        os.makedirs(os.path.dirname(hdf5_path) or ".", exist_ok=True)
        data = self.to_dict()
        write_episode_to_hdf5(hdf5_path, ep_id, data)
        logger.info("Trajectory appended to %s (episode %s)", hdf5_path, ep_id)

    def reset(self) -> None:
        """Clear accumulated data for the next episode."""
        self._initial_info = None
        self._step_infos = []
        self._raw_policy_actions = []
        self._episode_id = None
        self._seed = None
        self._metadata = None
