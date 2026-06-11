"""Episode metrics callback (factory): logs scalar episode metrics (achieved_reward,
cum_E_kWh) and optional per-episode JSON dumps.
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

from adv_building_gym._common.json_encoder import CustomJSONEncoder

logger = logging.getLogger(__name__)

# Within-iter aggregation window; with clear_on_reduce=True each TB scalar is the
# per-iter arithmetic mean/min/max (no cross-iter blending). See _swap_trigger.py.
_WITHIN_ITER_WINDOW = 10_000


def _extract_clipped_actions(episode: SingleAgentEpisode) -> list | None:
    """Flattened post-clip actions from episode info ("action" key); falls back to raw policy
    outputs, or None if unavailable."""
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
    cum_E_kWh: float | None,
    cum_price_EUR: float | None,
    reward_component_totals: dict[str, float] | None = None,
) -> None:
    """Save per-episode metrics as JSON.

    Args:
        episode: The completed episode.
        ep_metrics_file: Output file path.
        ep_length: Number of steps in the episode.
        ep_achieved_reward: Total reward achieved.
        cum_E_kWh: Cumulative energy consumption, or None.
        cum_price_EUR: Cumulative electricity cost (EUR), or None.
        reward_component_totals: Per-component episode reward sums, or None.
    """
    clipped_actions = _extract_clipped_actions(episode)

    dump = {
        "id": episode.id_[:6],
        "length": ep_length,
        "achieved_reward": float(ep_achieved_reward),
        "cum_E_kWh": float(cum_E_kWh) if cum_E_kWh is not None else None,
        "cum_price_EUR": float(cum_price_EUR) if cum_price_EUR is not None else None,
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
    """Factory → configured EpisodeMetricsCallback class. 
    Logs scalar episode metrics and
    (if ``dump_metrics_json``) logs per-episode JSON.
    Args:
        metrics_base_dir: Base directory for saving episode metrics JSON.
        exec_date: Execution datetime for directory naming. Defaults to now.
        dump_metrics_json: When True, save per-episode JSON files.

    Returns:
        A configured RLlibCallback subclass (not an instance).
    """
    if exec_date is None:
        exec_date = datetime.datetime.now()

    # capture params in closure; resolve metrics_base_dir absolute at factory time so writes
    # land correctly regardless of cwd (Ray Tune changes the actor's cwd)
    _metrics_base_dir = os.path.abspath(metrics_base_dir)
    _exec_date = exec_date

    class EpisodeMetricsCallback(RLlibCallback):
        """Log scalar episode metrics and optional per-episode JSON, on every episode end."""

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
            episode_return = np.sum(episode.get_rewards())

            cum_E_kWh = None
            cum_price_EUR = None
            episode_count: int | None = None
            reward_component_totals: dict[str, float] = {}
            if hasattr(episode, "get_infos"):
                infos = episode.get_infos()
                for info in infos:
                    if isinstance(info, dict):
                        reward_breakdown = info.get("reward_breakdown")
                        if reward_breakdown:
                            for reward_key, reward_value in reward_breakdown.items():
                                reward_component_totals[reward_key] = reward_component_totals.get(reward_key, 0.0) + reward_value
                if infos and len(infos) > 0 and isinstance(infos[-1], dict):
                    cum_E_kWh = infos[-1].get("cum_E_kWh")
                    cum_price_EUR = infos[-1].get("cum_price_EUR")
                # episode_count is published by AdvBuildingGym in the reset info (infos[0]).
                if infos and isinstance(infos[0], dict):
                    episode_count = infos[0].get("episode_count")

            # window=_WITHIN_ITER_WINDOW + clear_on_reduce=True → per-iter mean/min/max
            # (not RLlib's default EMA / lifetime extreme), aligned with iter-aligned swaps
            metrics_logger.log_value("achieved_reward", episode_return, reduce="mean",
                                     window=_WITHIN_ITER_WINDOW, clear_on_reduce=True)
            metrics_logger.log_value("achieved_reward_min", episode_return, reduce="min",
                                     window=_WITHIN_ITER_WINDOW, clear_on_reduce=True)
            metrics_logger.log_value("achieved_reward_max", episode_return, reduce="max",
                                     window=_WITHIN_ITER_WINDOW, clear_on_reduce=True)

            # Log cumulative energy consumption
            if cum_E_kWh is not None:
                metrics_logger.log_value("cum_E_kWh", cum_E_kWh, reduce="mean",
                                        window=_WITHIN_ITER_WINDOW, clear_on_reduce=True)
                metrics_logger.log_value("cum_E_kWh_min", cum_E_kWh, reduce="min",
                                        window=_WITHIN_ITER_WINDOW, clear_on_reduce=True)
                metrics_logger.log_value("cum_E_kWh_max", cum_E_kWh, reduce="max",
                                        window=_WITHIN_ITER_WINDOW, clear_on_reduce=True)

            # Log cumulative electricity cost (positive = money spent)
            if cum_price_EUR is not None:
                metrics_logger.log_value("cum_price_EUR", cum_price_EUR, reduce="mean",
                                        window=_WITHIN_ITER_WINDOW, clear_on_reduce=True)
                metrics_logger.log_value("cum_price_EUR_min", cum_price_EUR, reduce="min",
                                        window=_WITHIN_ITER_WINDOW, clear_on_reduce=True)
                metrics_logger.log_value("cum_price_EUR_max", cum_price_EUR, reduce="max",
                                        window=_WITHIN_ITER_WINDOW, clear_on_reduce=True)

            # per-component reward breakdown → (evaluation/)env_runners/reward/<name>
            for reward_key, comp_total in reward_component_totals.items():
                metrics_logger.log_value(f"reward/{reward_key}", comp_total, reduce="mean",
                                        window=_WITHIN_ITER_WINDOW, clear_on_reduce=True)

            episode_id: str = episode.id_[:6]
            episode_num_str = str(episode_count) if episode_count is not None else "?"
            logger.info(
                "Episode %s (ID: %s) ended. Length: %s, Episode return: %.2f",
                episode_num_str, episode_id, ep_length, episode_return,
            )

            if dump_metrics_json:
                # Save per-episode metrics JSON
                ep_metrics_dir = f"{_metrics_base_dir}/{_exec_date.strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(ep_metrics_dir, exist_ok=True)
                ep_metrics_file = f"{ep_metrics_dir}/episode_{episode_id}_metrics.json"

                _save_episode_metrics_json(
                    episode=episode,
                    ep_metrics_file=ep_metrics_file,
                    ep_length=ep_length,
                    ep_achieved_reward=float(episode_return),
                    cum_E_kWh=cum_E_kWh,
                    cum_price_EUR=cum_price_EUR,
                    reward_component_totals=reward_component_totals,
                )

    return EpisodeMetricsCallback
