"""Reward schedule switching via RLlib on_train_result callback.

Pushes a new reward function subset to all env_runners (training AND
evaluation) at training iteration boundaries.  The active subset is
determined by the RewardScheduleManager's mode and swap index.

Pattern mirrors ``data_schedule_callback.py`` (Approach D1).

``create_reward_switch_on_train_result(...)`` returns an
``on_train_result`` function that can be passed as a keyword argument
to ``config.callbacks(ExistingClass, on_train_result=func)``.
"""

import logging

from adv_building_gym.config.reward_schedule_manager import RewardScheduleManager

logger = logging.getLogger(__name__)


def create_reward_switch_on_train_result_cb(
    reward_manager: RewardScheduleManager,
):
    """Factory that returns an ``on_train_result`` callable.

    Usage::

        config.callbacks(
            SomeCallbackClass,
            on_train_result=create_reward_switch_on_train_result(manager),
        )

    Args:
        reward_manager: Stateful RewardScheduleManager that tracks the
            schedule position and creates reward subsets.
    """

    def on_train_result(*, algorithm, result: dict, **kwargs) -> None:
        iteration: int = result.get("training_iteration", 0)
        logger.info("RewardScheduleManager iteration: %d", iteration)
        if iteration % reward_manager.swap_every_n_iterations != 0:
            return

        changed = reward_manager.advance()
        if not changed:
            return

        _push_rewards_to_runners(algorithm, reward_manager)

    return on_train_result


def _push_rewards_to_runners(
    algorithm,
    reward_manager: RewardScheduleManager,
) -> None:
    """Create fresh reward instances and set them on all live envs.

    Both training and evaluation env_runners receive the same active
    reward set so that eval metrics reflect the current training objective.
    """

    def apply(env_runner) -> None:
        # env_runner.env is wrapped:
        #   DictInfoToList -> SyncVectorEnv -> [TimeLimit -> ... -> AdvBuildingGym]
        # The local (driver) env_runner may have env=None in the new API stack.
        vec_env = getattr(env_runner, "env", None)
        if vec_env is None:
            return
        # Unwrap DictInfoToList to reach SyncVectorEnv
        sync_vec = getattr(vec_env, "env", vec_env)
        for sub_env in getattr(sync_vec, "envs", []):
            unwrapped = sub_env.unwrapped
            if hasattr(unwrapped, "set_reward_funcs"):
                # Each sub-env gets its own fresh reward instances
                rewards = reward_manager.create_active_rewards()
                unwrapped.set_reward_funcs(rewards)

    algorithm.env_runner_group.foreach_env_runner(
        apply, local_env_runner=True, timeout_seconds=None,
    )
    # Eval runners get the SAME reward set as training runners.
    # Eval does not influence the policy gradient — it is purely for
    # monitoring — but must evaluate the same objective for metrics to
    # be meaningful.
    if algorithm.eval_env_runner_group is not None:
        algorithm.eval_env_runner_group.foreach_env_runner(
            apply, local_env_runner=True, timeout_seconds=None,
        )

    logger.info(
        "Reward schedule: pushed active rewards [ %s ] to all env_runners",
        reward_manager.get_active_reward_names(),
    )
