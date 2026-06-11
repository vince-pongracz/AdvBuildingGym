"""Evaluation trajectory logging callback.

Writes per-step eval trajectories to TensorBoard as separate sub-runs
so they overlay in a single chart:
- x-axis = episode timestep (0 … EPISODE_LENGTH-1)
- y-axis = value (raw physical, normalised state, action, power)
- one coloured line per eval episode

Each eval round (``evaluation_duration`` episodes) is written as one
TensorBoard sub-run *per episode* under a shared ``iter_<N>`` group, so
the round's episodes overlay in a single chart (one line each). Per-step
averaging across the round is deliberately NOT done: with ``day="random"``
(and the training combinator's variant cadence) the episodes land on
different days / CSV bundles, so a per-step mean would smear unrelated
trajectories toward zero behind a meaningless min/max band. Filter the
TensorBoard run list by ``iter_<N>`` to inspect one round's spread.

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
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

logger = logging.getLogger(__name__)


def make_eval_state_action_cb_class(
    metrics_base_dir: str = "ep_metrics",
    exec_date: datetime.datetime | None = None,
    trial_name: str | None = None,
) -> Type["EvalStateActionCallback"]:
    """Factory → configured EvalStateActionCallback class.
    Each eval round writes one TB
    sub-run per episode under a shared ``iter_<N>`` group (shared tags → episodes overlay).

    Args:
        metrics_base_dir: Base directory for output files.
        exec_date: Execution timestamp for directory naming.

    Returns:
        A configured RLlibCallback subclass (not an instance).
    """
    if exec_date is None:
        exec_date = datetime.datetime.now()

    # suffix run dir with SLURM_JOB_ID/PID so same-second sbatches don't clobber tfevents
    job_suffix = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"
    run_dir_name = f"{exec_date.strftime('%Y%m%d_%H%M%S')}_{job_suffix}"

    # resolve absolute at factory time so writes land correctly regardless of cwd (Ray Tune)
    tb_log_dir = os.path.join(os.path.abspath(metrics_base_dir), "eval_trajectories", run_dir_name)

    # sanitised trial-name suffix so the TB run list shows which trial produced each curve
    if trial_name:
        import re as _re
        _trial_suffix = (_re.sub(r"[^A-Za-z0-9._-]+", "_", trial_name).strip("_") or "trial")
    else:
        _trial_suffix = ""

    # closure state shared across instances on this worker; safe — eval is single-runner sequential
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
            algo_config: AlgorithmConfig = env_runner.config
            if not algo_config.in_evaluation:
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

                # Instantaneous net power (kW) + per-component breakdown
                if "net_power_kW" in info:
                    ep_data["power/net_kW"].append(float(info["net_power_kW"]))
                for comp_name, comp_power in info.get("power_breakdown", {}).items():
                    # power_breakdown values are (production_kW, consumption_kW) tuples;
                    # net = production - consumption (see envs/_energy_tracker.py).
                    if isinstance(comp_power, tuple):
                        net_power = float(comp_power[0]) - float(comp_power[1])
                    else:
                        net_power = float(comp_power)
                    ep_data[f"power/{comp_name}_kW"].append(net_power)

                # Total step reward + per-component breakdown
                if "reward" in info:
                    ep_data["reward/total"].append(float(info["reward"]))
                for rew_name, rew_val in info.get("reward_breakdown", {}).items():
                    ep_data[f"reward/{rew_name}"].append(float(rew_val))

                # per-step reward diagnostics (0/1 flags); accumulated into a cumulative
                # sum below so the trajectory grows by 1 each step a flag fires
                # TODO VP 2026.06.08.: How does this work..?
                for diag_name, diag_val in info.get("reward_diagnostics", {}).items():
                    ep_data[f"reward_diag/{diag_name}"].append(float(diag_val))

            # reward_diag 0/1 series → per-step cumulative sums (count rises step-by-step)
            for key, series in ep_data.items():
                if key.startswith("reward_diag/"):
                    ep_data[key] = np.cumsum(series).tolist()

            _episode_buffer.append(dict(ep_data))

            # Wait until all episodes in this eval round are collected
            if len(_episode_buffer) < algo_config.evaluation_duration:
                return

            # --- Eval round complete: one sub-run per episode ---
            # No cross-episode averaging (different days/variants): each episode is its own
            # sub-run ``iter_<N>[_<trial>]/ep_<i>`` with shared tags, so they overlay per chart.
            _eval_round[0] += 1
            training_iter = _eval_round[0] * algo_config.evaluation_interval

            iter_group = (
                f"iter_{training_iter:06d}_{_trial_suffix}" if _trial_suffix
                else f"iter_{training_iter:06d}"
            )

            num_steps = 0
            all_tags: set[str] = set()
            for ep_idx, ep in enumerate(_episode_buffer):
                run_dir = os.path.join(tb_log_dir, iter_group, f"ep_{ep_idx:02d}")
                with SummaryWriter(log_dir=run_dir) as writer:
                    for key, series in ep.items():
                        all_tags.add(key)
                        num_steps = max(num_steps, len(series))
                        for step_idx, val in enumerate(series):
                            writer.add_scalar(key, val, global_step=step_idx)

            logger.info(
                "Eval trajectory round %d (training iter %d): %d episodes x %d tags x %d steps -> %s",
                _eval_round[0], training_iter, len(_episode_buffer), len(all_tags),
                num_steps, os.path.join(tb_log_dir, iter_group),
            )

            _episode_buffer.clear()

    return EvalStateActionCallback
