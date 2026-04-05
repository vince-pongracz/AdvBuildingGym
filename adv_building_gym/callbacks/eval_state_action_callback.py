"""
Evaluation trajectory logging callback for Ray RLlib training.

Writes per-step eval trajectories to TensorBoard as separate sub-runs
so they overlay in a single chart:
- x-axis = episode timestep (0 … EPISODE_LENGTH-1)
- y-axis = value (raw physical, normalised state, action, power)
- one coloured line per eval round

Each eval round averages its N episodes (``evaluation_duration``) into a
single trajectory before writing.  Sub-runs are named by training
iteration for easy identification.

Point TensorBoard at ``<metrics_base_dir>/eval_trajectories/<exec_date>/``
to visualise:
    tensorboard --logdir ep_metrics/eval_trajectories/

Link: https://docs.ray.io/en/latest/rllib/rllib-callback.html
"""

import datetime
import logging
import os
from collections import defaultdict
from typing import Type

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.single_agent_episode import SingleAgentEpisode

logger = logging.getLogger(__name__)


def make_eval_state_action_cb_class(
    metrics_base_dir: str = "ep_metrics",
    exec_date: datetime.datetime | None = None,
) -> Type["EvalStateActionCallback"]:
    """Factory that returns a configured EvalStateActionCallback class.

    Each eval round (N episodes averaged) is written as a separate
    TensorBoard sub-run.  All sub-runs share the same tag names, so
    TensorBoard overlays them in a single chart.

    Args:
        metrics_base_dir: Base directory for output files.
        exec_date: Execution timestamp for directory naming.

    Returns:
        A configured RLlibCallback subclass (not an instance).
    """
    if exec_date is None:
        exec_date = datetime.datetime.now()

    # Resolve to absolute path at factory time so that file writes land in
    # the correct location regardless of process cwd (Ray Tune changes the
    # Trainable actor's cwd to the trial log directory).
    tb_log_dir = os.path.join(
        os.path.abspath(metrics_base_dir), "eval_trajectories",
        exec_date.strftime("%Y%m%d_%H%M%S"),
    )

    # Closure state shared across all callback instances on this worker.
    # Safe because eval runs on a single EnvRunner sequentially
    # (evaluation_parallel_to_training=False).
    _episode_buffer: list[dict[str, list[float]]] = []
    _eval_round: list[int] = [0]  # mutable int via list

    class EvalStateActionCallback(RLlibCallback):
        """Write per-step eval trajectories to TensorBoard sub-runs."""

        def on_episode_end(
            self,
            *,
            episode: SingleAgentEpisode,
            env_runner,
            metrics_logger,
            env,
            **kwargs,
        ):
            if not getattr(env_runner.config, "in_evaluation", False):
                return

            infos = episode.get_infos() if hasattr(episode, "get_infos") else []
            if not infos:
                return

            # Collect per-step data for this episode
            ep_data: dict[str, list[float]] = defaultdict(list)

            for info in infos:
                if not isinstance(info, dict):
                    continue

                # Raw (denormalised) physical values: temp_in_raw, temp_out_raw, …
                for key, val in info.get("raw", {}).items():
                    ep_data[f"raw/{key}"].append(float(val))

                # Normalised state variables (requires log_full_info=True on eval env)
                for key, val in info.get("state", {}).items():
                    arr = np.atleast_1d(val)
                    if arr.size == 1:
                        ep_data[f"state/{key}"].append(float(arr[0]))
                    else:
                        for dim, scalar in enumerate(arr.flat):
                            ep_data[f"state/{key}_{dim}"].append(float(scalar))

                # Clipped dict actions (multi-dim split into scalars)
                for act_key, act_val in info.get("action", {}).items():
                    arr = np.atleast_1d(act_val)
                    for dim, scalar in enumerate(arr.flat):
                        ep_data[f"action/{act_key}_{dim}"].append(float(scalar))

                # Instantaneous net power (kW)
                if "step_power_kW" in info:
                    ep_data["power/net_kW"].append(float(info["step_power_kW"]))

            _episode_buffer.append(dict(ep_data))

            # Wait until all episodes in this eval round are collected
            eval_duration = getattr(env_runner.config, "evaluation_duration", 2)
            if len(_episode_buffer) < eval_duration:
                return

            # --- Eval round complete: average and write ---
            _eval_round[0] += 1
            eval_interval = getattr(env_runner.config, "evaluation_interval", 5)
            training_iter = _eval_round[0] * eval_interval

            # Collect all keys across episodes
            all_keys: set[str] = set()
            for ep in _episode_buffer:
                all_keys.update(ep.keys())

            # Average trajectories per step across episodes
            averaged: dict[str, list[float]] = {}
            for key in sorted(all_keys):
                arrays = [ep[key] for ep in _episode_buffer if key in ep]
                if not arrays:
                    continue
                min_len = min(len(a) for a in arrays)
                stacked = np.array([a[:min_len] for a in arrays])
                averaged[key] = np.mean(stacked, axis=0).tolist()

            # Write to a TensorBoard sub-run named by training iteration.
            # TensorBoard overlays sub-runs with the same tag in one chart.
            run_name = f"iter_{training_iter:06d}"
            run_dir = os.path.join(tb_log_dir, run_name)
            writer = SummaryWriter(log_dir=run_dir)

            for key, vals in averaged.items():
                for step_idx, val in enumerate(vals):
                    writer.add_scalar(key, val, global_step=step_idx)

            writer.close()

            num_steps = max((len(v) for v in averaged.values()), default=0)
            logger.info(
                "Eval trajectory round %d (training iter %d): "
                "%d tags × %d steps → %s",
                _eval_round[0], training_iter, len(averaged), num_steps, run_dir,
            )

            _episode_buffer.clear()

    return EvalStateActionCallback
