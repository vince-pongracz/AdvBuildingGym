"""TrajectoryCollector for standalone evaluation scripts.

Accumulates per-step info dicts during an env.step() loop and produces
structured columnar trajectory JSON via extract_trajectory_from_infos().
"""

import json
import logging
import os

import numpy as np

from adv_building_gym._common.json_encoder import CustomJSONEncoder
from adv_building_gym._common.trajectory_utils import extract_trajectory_from_infos, write_episode_to_hdf5

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
        """Extract state/action/reward keys from ``env`` (AdvBuildingGym or compatible)."""
        # state/action key names from env spaces
        self.state_keys: list[str] | None = None
        if hasattr(env, "observation_space") and hasattr(env.observation_space, "spaces"):
            self.state_keys = list(env.observation_space.spaces.keys())

        self.action_keys: list[str] | None = None
        # native Dict action space is on the unwrapped env (wrappers may hide it)
        action_space = getattr(env, "action_space", None)
        if hasattr(action_space, "spaces"):
            self.action_keys = list(action_space.spaces.keys())

        # Reward function names for summary
        self.reward_names: list[str] = []
        if hasattr(env, "reward_funcs"):
            self.reward_names = [r.name for r in env.reward_funcs]

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
        """Build the full trajectory dict (extract_trajectory_from_infos + raw actions, metadata, summary)."""
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

        # Compute summary from the per-step trajectory
        rewards = trajectory.get("reward", [])
        achieved_reward = sum(rewards)
        ep_length = len(self._step_infos)
        cum_values = trajectory.get("cum_E_kWh", [])
        final_cum_E = cum_values[-1] if cum_values else 0.0
        cum_price_values = trajectory.get("cum_price_EUR", [])
        final_cum_price = cum_price_values[-1] if cum_price_values else 0.0

        episode_date = self._initial_info.get("episode_date") if self._initial_info else None

        result = {
            "version": 1,
            "episode_id": self._episode_id,
            "seed": self._seed,
            "length": ep_length,
            "episode_date": episode_date,
            "metadata": self._metadata or {},
            "summary": {
                "achieved_reward": float(achieved_reward),
                "cum_E_kWh": float(final_cum_E),
                "cum_price_EUR": float(final_cum_price),
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
        """Append this episode's trajectory to an HDF5 file (``episode_id`` defaults to ``self._episode_id``)."""
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
