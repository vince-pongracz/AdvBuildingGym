"""Iteration-aligned datasource variant scheduling via RLlib callback (Approach D1).

Pushes a new DataCombinator variant to all env_runners at training iteration
boundaries, ensuring all workers change dataset simultaneously.

Compatible with Approach A: if the environment also has its own episode counter
the two swap schedules are independent and additive.  Set data_combinator=None
on the Config (Approach A disabled) and use only this callback if
iteration-aligned swapping is desired.

``create_data_schedule_on_train_result(...)`` returns an ``on_train_result``
function that can be passed directly as a keyword argument to
``config.callbacks(ExistingClass, on_train_result=func)``.
"""

import logging

from adv_building_gym.config.data_combinator import DataCombinator
from adv_building_gym.envs.data_variant import DataVariantProvider

logger = logging.getLogger(__name__)


def create_data_schedule_on_train_result(
    combinator: DataCombinator,
    swap_every_n_iterations: int = 10,
):
    """Factory that returns an ``on_train_result`` function for ``config.callbacks()``.

    Usage::

        config.callbacks(
            CheckpointCallbackClass,
            on_episode_end=episode_end_fn,
            on_train_result=create_data_schedule_on_train_result(combinator, 10),
        )

    Args:
        combinator: DataCombinator instance that defines the variant pool.
        swap_every_n_iterations: Push a new variant every N training iterations.
    """

    def on_train_result(*, algorithm, result: dict, **kwargs) -> None:
        iteration: int = result.get("training_iteration", 0)
        if iteration % swap_every_n_iterations != 0:
            return

        variant = combinator.get_variant(iteration // swap_every_n_iterations)
        if not variant:
            return

        _push_variant_to_runners(algorithm, variant, iteration)

    return on_train_result


def _push_variant_to_runners(algorithm, variant: dict[str, str], iteration: int) -> None:
    """Apply a datasource variant to all env_runners (training + evaluation)."""

    def apply(env_runner) -> None:
        # env_runner.env is wrapped: DictInfoToList -> SyncVectorEnv -> [TimeLimit -> ... -> AdvBuildingGym]
        # The local (driver) env_runner may have env=None in the new API stack.
        vec_env = getattr(env_runner, "env", None)
        if vec_env is None:
            return
        # Unwrap DictInfoToList to reach SyncVectorEnv, then iterate sub-envs.
        # SyncVectorEnv.envs is the list of (possibly wrapped) Gymnasium envs.
        sync_vec = getattr(vec_env, "env", vec_env)
        for sub_env in getattr(sync_vec, "envs", []):
            unwrapped = sub_env.unwrapped
            if isinstance(unwrapped, DataVariantProvider):
                unwrapped.apply_data_variant(variant)

    algorithm.env_runner_group.foreach_env_runner(
        apply, local_env_runner=True, timeout_seconds=None
    )
    if algorithm.eval_env_runner_group is not None:
        algorithm.eval_env_runner_group.foreach_env_runner(
            apply, local_env_runner=True, timeout_seconds=None
        )
    
    logger.info("Iteration %d: all env_runners switched to variant %s", iteration, variant)
