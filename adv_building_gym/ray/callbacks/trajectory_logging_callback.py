"""Trajectory logging callback (factory): saves full per-step eval trajectories as
per-episode JSON + a shared HDF5. Active only when ``in_evaluation``.
See docs/about_traj_hdf5_export.md; Link: https://docs.ray.io/en/latest/rllib/rllib-callback.html
"""

import os
import json
import logging
import datetime
from typing import Optional, Type

import numpy as np

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.single_agent_episode import SingleAgentEpisode

from adv_building_gym._common.json_encoder import CustomJSONEncoder
from ..utils.trajectory_utils import extract_trajectory_from_infos, write_episode_to_hdf5

logger = logging.getLogger(__name__)


def make_trajectory_logging_cb_class(
    metrics_base_dir: str = "ep_metrics",
    exec_date: Optional[datetime.datetime] = None,
) -> Type["TrajectoryLoggingCallback"]:
    """Factory → configured TrajectoryLoggingCallback class. Saves per-episode eval JSON +
    appends to trajectories.hdf5; gated by ``in_evaluation`` (no I/O during training)."""
    if exec_date is None:
        exec_date = datetime.datetime.now()

    _metrics_base_dir = os.path.abspath(metrics_base_dir)
    _exec_date = exec_date

    class TrajectoryLoggingCallback(RLlibCallback):
        """Save per-step trajectory (JSON + HDF5) during eval episodes.

        Active only when ``in_evaluation``; requires ``log_full_info=True`` (info["state"]).
        """

        def on_episode_end(
            self,
            *,
            episode: SingleAgentEpisode,
            env_runner,
            metrics_logger,
            env,
            **kwargs,
        ):
            # Only log trajectories during evaluation episodes
            if env_runner is None or not env_runner.config.in_evaluation:
                return

            episode_id: str = episode.id_[:6]

            try:
                infos = episode.get_infos()
                raw_actions = (
                    episode.get_actions() if hasattr(episode, "get_actions") else []
                )

                # infos[0] is from reset() (initial conditions), infos[1:] from step()
                initial_info = infos[0] if infos else None
                step_infos = infos[1:] if len(infos) > 1 else infos

                trajectory = extract_trajectory_from_infos(step_infos, initial_info=initial_info)

                # Add raw policy actions as columnar data
                if raw_actions is not None and len(raw_actions) > 0:
                    first_action = np.atleast_1d(raw_actions[0])
                    ndim = first_action.size
                    has_initial = initial_info is not None
                    for d in range(ndim):
                        col = f"raw_policy_action_{d}"
                        values: list[float] = []
                        if has_initial:
                            values.append(0.0)
                        for act in raw_actions:
                            values.append(float(np.atleast_1d(act).flat[d]))
                        trajectory[col] = values

                # Compute summary statistics
                ep_length = len(episode)
                ep_achieved_reward = float(np.sum(episode.get_rewards()))
                cum_E_kWh = None
                if infos and isinstance(infos[-1], dict):
                    cum_E_kWh = infos[-1].get("cum_E_kWh")

                traj_dump = {
                    "version": 1,
                    "episode_id": episode.id_,
                    "seed": initial_info.get("seed") if initial_info else None,
                    "length": ep_length,
                    "eval": env_runner.config.in_evaluation,
                    "episode_date": initial_info.get("episode_date") if initial_info else None,
                    "summary": {
                        "achieved_reward": ep_achieved_reward,
                        "episode_return": ep_achieved_reward,
                        "cum_E_kWh": float(cum_E_kWh) if cum_E_kWh is not None else None,
                    },
                    "trajectory": trajectory,
                }

                ep_metrics_dir = (f"{_metrics_base_dir}/{_exec_date.strftime('%Y%m%d_%H%M%S')}")
                os.makedirs(ep_metrics_dir, exist_ok=True)
                jsons_dir = f"{ep_metrics_dir}/jsons"
                os.makedirs(jsons_dir, exist_ok=True)
                traj_file = f"{jsons_dir}/{episode_id}_trajectory.json"
                with open(traj_file, "w", encoding="utf-8") as f:
                    json.dump(traj_dump, f, cls=CustomJSONEncoder, indent=4)
                logger.info("Trajectory saved to %s", traj_file)

                hdf5_path = f"{ep_metrics_dir}/trajectories.hdf5"
                write_episode_to_hdf5(hdf5_path, episode_id, traj_dump)
                logger.info("Trajectory appended to %s", hdf5_path)

            except Exception:
                logger.exception(
                    "Failed to save trajectory for episode %s", episode_id
                )

    return TrajectoryLoggingCallback
