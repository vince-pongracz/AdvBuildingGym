"""Iteration-aligned infrastructure config scheduling via RLlib callback.

Pushes fresh Infrastructure instances to all env_runners at training
iteration boundaries, ensuring all workers change configuration
simultaneously.

Pattern mirrors ``data_schedule_callback.py`` (Approach D1).

``create_infra_schedule_on_train_result_cb(...)`` returns an
``on_train_result`` function that can be passed as a keyword argument
to ``config.callbacks(ExistingClass, on_train_result=func)``.
"""

import logging

from adv_building_gym.infra_combinator import InfraCombinator

logger = logging.getLogger(__name__)


def create_infra_schedule_on_train_result_cb(infra_combinator: InfraCombinator):
    """Factory that returns an ``on_train_result`` callable.

    Args:
        infra_combinator: InfraCombinator instance that defines the
            config pool and swap schedule.
    """

    def on_train_result(*, algorithm, result: dict, **kwargs) -> None:
        iteration: int = result.get("training_iteration", 0)
        if iteration % infra_combinator.swap_every_n_iterations != 0:
            return

        changed = infra_combinator.advance()
        if not changed:
            return

        _push_infras_to_runners(algorithm, infra_combinator)

    return on_train_result


def _push_infras_to_runners(
    algorithm,
    infra_combinator: InfraCombinator,
) -> None:
    """Create fresh infra instances and push to all env_runners."""

    swap_index = infra_combinator._swap_index

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
            if hasattr(unwrapped, "set_infras"):
                # Each sub-env gets its own fresh infra instances
                new_infras = infra_combinator.create_infras(swap_index)
                unwrapped.set_infras(new_infras)

    algorithm.env_runner_group.foreach_env_runner(
        apply, local_env_runner=True, timeout_seconds=None,
    )
    if algorithm.eval_env_runner_group is not None:
        algorithm.eval_env_runner_group.foreach_env_runner(
            apply, local_env_runner=True, timeout_seconds=None,
        )

    logger.info(
        "Iteration %d: all env_runners switched to infra config '%s' "
        "(swap_index=%d)",
        algorithm.training_iteration,
        infra_combinator.get_active_config_name(),
        swap_index,
    )
