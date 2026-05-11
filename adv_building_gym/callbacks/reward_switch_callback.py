"""Reward schedule switching via RLlib on_train_result callback.

Pushes a new reward function subset to all env_runners (training AND
evaluation) at training iteration boundaries.  Optionally fires the
event-driven exploration bump (shared util) when the active set / weights
change.
"""

import logging

from adv_building_gym.callbacks._exploration_reset_util import make_decay_loop
from adv_building_gym.config.exploration_reset import ExplorationResetConfig
from adv_building_gym.config.reward_schedule_manager import RewardScheduleManager

logger = logging.getLogger(__name__)


def create_reward_switch_on_train_result_cb(
    reward_manager: RewardScheduleManager,
    exploration_reset: ExplorationResetConfig | None = None,
):
    """Factory returning an on_train_result callable.

    Args:
        reward_manager: Stateful manager driving the active subset.
        exploration_reset: Standalone exploration-reset config; the bump
            fires only when ``trigger`` includes ``on_reward_swap``.
    """
    expl_cfg = exploration_reset or ExplorationResetConfig()
    _state, maybe_decay, fire_bump = make_decay_loop(expl_cfg, "on_reward_swap")

    def on_train_result(*, algorithm, result: dict, **kwargs) -> None:
        iteration: int = result.get("training_iteration", 0)
        maybe_decay(algorithm, iteration)

        if iteration % reward_manager.swap_every_n_episodes != 0:
            return
        changed = reward_manager.advance()
        if not changed:
            return
        _push_rewards_to_runners(algorithm, reward_manager)
        fire_bump(algorithm, iteration)

    return on_train_result


def _push_rewards_to_runners(algorithm, reward_manager: RewardScheduleManager) -> None:
    def apply(env_runner) -> None:
        vec_env = getattr(env_runner, "env", None)
        if vec_env is None:
            return
        sync_vec = getattr(vec_env, "env", vec_env)
        for sub_env in getattr(sync_vec, "envs", []):
            unwrapped = sub_env.unwrapped
            if hasattr(unwrapped, "set_reward_funcs"):
                rewards = reward_manager.create_active_rewards()
                unwrapped.set_reward_funcs(rewards)

    algorithm.env_runner_group.foreach_env_runner(
        apply, local_env_runner=True, timeout_seconds=None,
    )
    if algorithm.eval_env_runner_group is not None:
        algorithm.eval_env_runner_group.foreach_env_runner(
            apply, local_env_runner=True, timeout_seconds=None,
        )
    logger.info(
        "Reward schedule: pushed active rewards [ %s ] to all env_runners",
        reward_manager.get_active_reward_names(),
    )
