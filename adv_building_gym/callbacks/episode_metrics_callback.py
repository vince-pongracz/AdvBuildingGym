"""
Episode metrics callback for Ray RLlib training.

Provides EpisodeMetricsCallback via a factory function that logs scalar
episode metrics (achieved_reward, reward_rate, cum_E_kWh) and saves
per-episode JSON dumps with observations, actions, and rewards.

Link: https://docs.ray.io/en/latest/rllib/rllib-callback.html
"""

import os
import json
import logging
import datetime
from typing import Optional, Type

import numpy as np

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.single_agent_episode import SingleAgentEpisode

from ..utils import CustomJSONEncoder

logger = logging.getLogger(__name__)


def _extract_clipped_actions(episode: "SingleAgentEpisode") -> list | None:
    """Extract clipped actions from episode info dicts.

    Each info dict may contain an "action" key with the post-clip action
    (dict or ndarray). Falls back to raw policy outputs if extraction fails.

    Args:
        episode: The completed episode.

    Returns:
        List of flattened action arrays, or None if unavailable.
    """
    clipped_actions = []
    if hasattr(episode, "get_infos"):
        infos = episode.get_infos()
        if infos and len(infos) > 0:
            for info in infos:
                if isinstance(info, dict) and "action" in info:
                    action_dict = info["action"]
                    if isinstance(action_dict, dict):
                        flat_action = []
                        for key in sorted(action_dict.keys()):
                            act_val = action_dict[key]
                            if isinstance(act_val, np.ndarray):
                                flat_action.extend(act_val.flatten().tolist())
                            else:
                                flat_action.append(float(act_val))
                        clipped_actions.append(flat_action)
                    elif isinstance(action_dict, np.ndarray):
                        clipped_actions.append(action_dict.flatten().tolist())

    if not clipped_actions:
        episode_id = episode.id_[:6]
        logger.warning(
            "Episode %s: Failed to extract clipped actions from infos, "
            "using raw policy outputs. Actions may exceed [-1, 1] range.",
            episode_id,
        )
        clipped_actions = (
            episode.get_actions() if hasattr(episode, "get_actions") else None
        )

    return clipped_actions


def _save_episode_metrics_json(
    episode: SingleAgentEpisode,
    ep_metrics_file: str,
    ep_length: int,
    ep_achieved_reward: float,
    max_achievable_reward: float,
    reward_rate: float,
    cum_E_kWh: float | None,
    reward_component_totals: dict[str, float] | None = None,
) -> None:
    """Save per-episode metrics as a JSON file.

    Args:
        episode: The completed episode.
        ep_metrics_file: Output file path.
        ep_length: Number of steps in the episode.
        ep_achieved_reward: Total reward achieved.
        max_achievable_reward: Theoretical maximum reward.
        reward_rate: achieved / max ratio.
        cum_E_kWh: Cumulative energy consumption, or None.
        reward_component_totals: Per-component episode reward sums, or None.
    """
    clipped_actions = _extract_clipped_actions(episode)

    dump = {
        "id": episode.id_[:6],
        "length": ep_length,
        # NOTE VP 2026.01.12. : episode_return_mean is not available here, only in result dict.
        "achieved_reward": float(ep_achieved_reward),
        "total_reward": float(max_achievable_reward),
        "reward_rate": float(reward_rate),
        "cum_E_kWh": float(cum_E_kWh) if cum_E_kWh is not None else None,
        "reward_breakdown": reward_component_totals if reward_component_totals else None,
        "rewards": episode.get_rewards() if hasattr(episode, "get_rewards") else None,
        "observations": (
            episode.get_observations() if hasattr(episode, "get_observations") else None
        ),
        "actions": clipped_actions,
        "raw_policy_actions": (
            episode.get_actions() if hasattr(episode, "get_actions") else None
        ),
    }

    with open(ep_metrics_file, "w", encoding="utf-8") as f:
        json.dump(dump, f, cls=CustomJSONEncoder, indent=4)


def make_episode_metrics_cb_class(
    metrics_base_dir: str,
    exec_date: Optional[datetime.datetime],
    dump_metrics_json: bool,
) -> Type["EpisodeMetricsCallback"]:
    """Factory that returns a configured EpisodeMetricsCallback class.

    The returned class logs scalar episode metrics (achieved_reward,
    reward_rate, cum_E_kWh) via metrics_logger and saves per-episode
    JSON dumps with observations, actions, and rewards.

    Args:
        metrics_base_dir: Base directory for saving episode metrics JSON.
        exec_date: Execution datetime for directory naming. Defaults to now.
        dump_metrics_json: When True, save per-episode JSON files.

    Returns:
        A configured RLlibCallback subclass (not an instance).
    """
    if exec_date is None:
        exec_date = datetime.datetime.now()

    # Capture parameters in closure so each class definition is self-contained.
    # Resolve metrics_base_dir to absolute path at factory time so that file
    # writes land in the correct location regardless of process cwd (Ray Tune
    # changes the Trainable actor's cwd to the trial log directory).
    _metrics_base_dir = os.path.abspath(metrics_base_dir)
    _exec_date = exec_date

    class EpisodeMetricsCallback(RLlibCallback):
        """Log scalar episode metrics and save per-episode JSON dumps.

        Runs on every episode end (both training and evaluation EnvRunners).
        Registered as part of callbacks_class list in config.callbacks().
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
            # Calculate episode metrics
            ep_length = len(episode)
            ep_achieved_reward = np.sum(episode.get_rewards())

            # Sum step-wise max achievable rewards from info dicts.
            # Each step's info contains "max_reward_step" — the sum of
            # per-reward max values returned by get_reward() — so the
            # total adapts to state-dependent maxima (e.g. EV rewards
            # are 0 when the EV is disconnected).
            max_achievable_reward = 0.0
            cum_E_kWh = None
            reward_component_totals: dict[str, float] = {}
            if hasattr(episode, "get_infos"):
                infos = episode.get_infos()
                for info in infos:
                    if isinstance(info, dict):
                        max_achievable_reward += info.get("max_reward_step", 0.0)
                        breakdown = info.get("reward_breakdown")
                        if breakdown:
                            for comp_name, comp_val in breakdown.items():
                                reward_component_totals[comp_name] = (
                                    reward_component_totals.get(comp_name, 0.0) + comp_val
                                )
                if infos and len(infos) > 0 and isinstance(infos[-1], dict):
                    cum_E_kWh = infos[-1].get("cum_E_kWh")

            reward_rate = (
                ep_achieved_reward / max_achievable_reward
                if max_achievable_reward > 0
                else 0.0
            )

            # Register custom metrics with RLlib's metrics system
            # These appear in results under "env_runners/achieved_reward_mean" etc.
            metrics_logger.log_value("achieved_reward", ep_achieved_reward, reduce="mean")
            metrics_logger.log_value("reward_rate", reward_rate, reduce="mean")
            # Also log min/max for analysis
            metrics_logger.log_value("achieved_reward_min", ep_achieved_reward, reduce="min")
            metrics_logger.log_value("achieved_reward_max", ep_achieved_reward, reduce="max")
            metrics_logger.log_value("reward_rate_min", reward_rate, reduce="min")
            metrics_logger.log_value("reward_rate_max", reward_rate, reduce="max")
            # Log cumulative energy consumption
            if cum_E_kWh is not None:
                metrics_logger.log_value("cum_E_kWh", cum_E_kWh, reduce="mean")

            # Log per-component reward breakdown for TensorBoard.
            # Appears under env_runners/reward/<name> (training) and
            # evaluation/env_runners/reward/<name> (eval).
            for comp_name, comp_total in reward_component_totals.items():
                metrics_logger.log_value(f"reward/{comp_name}", comp_total, reduce="mean")

            episode_id: str = episode.id_[:6]
            logger.info(
                "Episode %s ended. Length: %s, Achieved Reward: %.2f, Reward Rate: %.4f",
                episode_id, ep_length, ep_achieved_reward, reward_rate,
            )
            
            if dump_metrics_json:
                # Save per-episode metrics JSON
                ep_metrics_dir = f"{_metrics_base_dir}/{_exec_date.strftime('%Y%m%d_%H%M')}00"
                os.makedirs(ep_metrics_dir, exist_ok=True)
                ep_metrics_file = f"{ep_metrics_dir}/episode_{episode_id}_metrics.json"

                _save_episode_metrics_json(
                    episode=episode,
                    ep_metrics_file=ep_metrics_file,
                    ep_length=ep_length,
                    ep_achieved_reward=float(ep_achieved_reward),
                    max_achievable_reward=float(max_achievable_reward),
                    reward_rate=float(reward_rate),
                    cum_E_kWh=cum_E_kWh,
                    reward_component_totals=reward_component_totals,
                )

    return EpisodeMetricsCallback
