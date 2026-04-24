"""Iteration-aligned data variant scheduling via RLlib callback.

Pushes a new variant coming from the DataCombinator to all env_runners at training iteration
boundaries, ensuring all workers holding an env change dataset simultaneously -- so they all see the same dataset at the same time.

Compatible with Approach A: if the environment also has its own episode counter
the two swap schedules are independent and additive.  Use an empty
DataCombinator() on the Config (Approach A disabled) and use only this
callback if iteration-aligned swapping is desired.

``create_data_schedule_on_train_result(...)`` returns an ``on_train_result``
function that can be passed directly as a keyword argument to
``config.callbacks(ExistingClass, on_train_result=func)``.
"""
# TODO VP 2026.04.23. : How is the random day swap within a variant?
# TODO VP 2026.04.23. : Is it still important, is there any constraint, what enforces that "all workers holding an env change dataset simultaneously"?
# It should not be because of the normalisation of the data as the max values of a dataset, a time series should be provided as context to the policy network, so it can take that factor into account.
# Is it only because of the during training eval calls, that should eval the same env as the training workers were working on? What if that picks a random year with a random day?

# TODO VP 2026.04.23. : Rewrite the about_data_mgmt.md. Adjust the docsstring to that.

import logging

from adv_building_gym.data_combinator import DataCombinator
from adv_building_gym.envs.data_variant import DataVariantProvider

logger = logging.getLogger(__name__)


def create_data_schedule_on_train_result_cb(
    combinator: DataCombinator,
    swap_every_n_iterations: int,
):
    """Factory returning an ``on_train_result`` function for ``config.callbacks()``.

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

        _publish_variant_to_runners(algorithm, variant, iteration)

    return on_train_result


def _publish_variant_to_runners(algorithm, variant: dict[str, str], iteration: int) -> None:
    """Apply a data variant to all env_runners (training + evaluation)."""

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

    algorithm.env_runner_group.foreach_env_runner(apply, local_env_runner=True, timeout_seconds=None)

    if algorithm.eval_env_runner_group is not None:
        algorithm.eval_env_runner_group.foreach_env_runner(apply, local_env_runner=True, timeout_seconds=None)
    
    logger.info("Iteration %d: all env_runners switched to variant %s", iteration, variant)
