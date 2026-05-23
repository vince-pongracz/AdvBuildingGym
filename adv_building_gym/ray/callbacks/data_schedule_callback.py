"""Episode-budget-aligned data variant scheduling via RLlib callback.

Pushes a variant from the DataCombinator to every env_runner — both
``algorithm.env_runner_group`` and ``algorithm.eval_env_runner_group`` — once
``num_episodes_lifetime`` (summed across all env_runners) has advanced by at
least ``max(swap_every_n_episodes, num_env_runners)`` since the previous
swap, so all workers consume the same year-long CSV bundle within a swap
window. This is one of three independent and additive swap surfaces (the
other two being the episode-boundary swap inside ``AdvBuildingGym.reset()``
and the ``reset(options={"data_variant": ...})`` external override). See
``docs/about_data_mgmt.md`` for the full picture and the wiring entry point
in ``ray_training/common_model_config.py``.

``create_data_schedule_on_train_result_cb(...)`` returns an ``on_train_result``
function that can be passed directly to
``config.callbacks(ExistingClass, on_train_result=func)`` or composed with the
reward / infra curriculum callbacks via ``register_callbacks(...)``.

Notes on synchronisation scope:

- Only the year-long variant (CSV bundle) is synchronised across runners. The
  per-episode day offset is still drawn independently by each env's
  ``reset()`` via ``DataCombinator.get_day_offset()`` against its local RNG,
  so with ``day="random"`` training and eval workers land on different days
  of the same variant.
- In-training eval is pushed the TRAINING combinator's variant. A held-out
  eval data config is only honoured by ``run_eval_ray.py``.
- Synchronised swap is required for interpretability, not correctness:
  PPO/SAC do not need synchronous variants across runners — the policy is
  conditioned on per-variant scale factors via ``ctxt_*`` observation keys
  (e.g. ``ctxt_temp_abs_max``), so mixed-variant batches are fine. The
  synchronous swap buys clean per-swap-window semantics: in-training eval
  reports on the same variant the training batch was collected on, reward /
  infra curriculum callbacks (also fired on ``on_train_result``) stay aligned
  with the data, and TensorBoard curves read as "window N had variant Z"
  instead of a moving cocktail across runners.
- Episodes longer than one day read consecutively past the day boundary. If
  ``EPISODE_LENGTH`` exceeds one calendar day, ``steps_per_day`` in
  ``building_adv.reset()`` should be decoupled from ``EPISODE_LENGTH`` so day
  indexing remains calendar-aligned.
"""

import logging

from adv_building_gym.ray.callbacks._swap_trigger import make_swap_gate
from adv_building_gym.config.data.data_combinator import DataCombinator
from adv_building_gym.core.data_variant import DataVariantProvider

logger = logging.getLogger(__name__)


def create_data_schedule_on_train_result_cb(
    combinator: DataCombinator,
    num_env_runners: int,
):
    """Factory returning an ``on_train_result`` function for ``config.callbacks()``.

    Args:
        combinator: DataCombinator instance that defines the variant pool.
        num_env_runners: Active env_runner count; used as the floor on the
            configured ``swap_every_n_episodes`` so each swap window spans
            at least one episode per runner.
    """
    gate = make_swap_gate(
        "DataSchedule", combinator.swap_every_n_episodes, num_env_runners,
    )
    swap_index = {"i": 0}

    def on_train_result(*, algorithm, result: dict, **kwargs) -> None:
        iteration: int = result.get("training_iteration", 0)
        decision = gate(iteration, result)
        if not decision.should_fire:
            return

        pool = combinator.variants
        if not pool:
            return
        variant = pool[swap_index["i"] % len(pool)]
        swap_index["i"] += 1

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
