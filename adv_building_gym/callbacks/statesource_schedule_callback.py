"""Iteration-aligned statesource scheduling via RLlib on_train_result callback.

Mirror of ``infra_schedule_callback.py`` for statesource bundles.  Pushes
fresh StateSource instances to all env_runners at training iteration
boundaries.  Optionally fires the shared exploration-reset bump.
"""

import logging

from adv_building_gym.callbacks._exploration_reset_util import make_decay_loop
from adv_building_gym.config.exploration_reset import ExplorationResetConfig
from adv_building_gym.statesource_combinator import StatesourceCombinator

logger = logging.getLogger(__name__)


def create_statesource_schedule_on_train_result_cb(
    statesource_combinator: StatesourceCombinator,
    exploration_reset: ExplorationResetConfig | None = None,
):
    expl_cfg = exploration_reset or ExplorationResetConfig()
    _state, maybe_decay, fire_bump = make_decay_loop(expl_cfg, "on_statesource_swap")

    def on_train_result(*, algorithm, result: dict, **kwargs) -> None:
        iteration: int = result.get("training_iteration", 0)
        maybe_decay(algorithm, iteration)

        if iteration % statesource_combinator.swap_every_n_iterations != 0:
            return

        changed = statesource_combinator.advance()
        if not changed:
            return
        _push_statesources_to_runners(algorithm, statesource_combinator)
        fire_bump(algorithm, iteration)

    return on_train_result


def _push_statesources_to_runners(algorithm, sc: StatesourceCombinator) -> None:
    swap_index = sc._swap_index

    def apply(env_runner) -> None:
        vec_env = getattr(env_runner, "env", None)
        if vec_env is None:
            return
        sync_vec = getattr(vec_env, "env", vec_env)
        for sub_env in getattr(sync_vec, "envs", []):
            unwrapped = sub_env.unwrapped
            if hasattr(unwrapped, "set_statesources"):
                new_ss = sc.create_statesources(swap_index)
                unwrapped.set_statesources(new_ss)

    algorithm.env_runner_group.foreach_env_runner(
        apply, local_env_runner=True, timeout_seconds=None,
    )

    if algorithm.eval_env_runner_group is not None:
        algorithm.eval_env_runner_group.foreach_env_runner(
            apply, local_env_runner=True, timeout_seconds=None,
        )
    logger.info(
        "Iteration %d: all env_runners switched to statesource config '%s' (swap_index=%d)",
        algorithm.training_iteration, sc.get_active_config_name(), swap_index,
    )
