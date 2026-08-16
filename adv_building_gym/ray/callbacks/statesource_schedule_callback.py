"""Episode-budget-aligned statesource scheduling via on_train_result.

Mirror of ``infra_schedule_callback.py``: pushes fresh StateSource instances to all
env_runners once the swap gate fires; optionally fires the exploration bump.
"""

import logging

from adv_building_gym.ray.callbacks._exploration_reset_util import make_decay_loop
from adv_building_gym.ray.callbacks._swap_trigger import make_swap_gate
from adv_building_gym.config.training.exploration_reset import ExplorationResetConfig
from adv_building_gym.config.env.statesource_combinator import StatesourceCombinator

logger = logging.getLogger(__name__)


def create_statesource_schedule_on_train_result_cb(
    statesource_combinator: StatesourceCombinator,
    num_env_runners: int,
    exploration_reset: ExplorationResetConfig | None = None,
):
    expl_cfg = exploration_reset or ExplorationResetConfig()
    _state, maybe_decay, fire_bump = make_decay_loop(expl_cfg, "on_statesource_swap")
    gate = make_swap_gate(
        "StatesourceSchedule", statesource_combinator.swap_every_n_episodes, num_env_runners,
    )

    def on_train_result(*, algorithm, result: dict, **kwargs) -> None:
        iteration: int = result.get("training_iteration", 0)
        maybe_decay(algorithm, iteration)

        decision = gate(iteration, result)
        if not decision.should_fire:
            return
        if decision.is_first_fire:
            _push_statesources_to_runners(algorithm, statesource_combinator)
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
                new_state_sources = sc.create_statesources(swap_index)
                unwrapped.set_statesources(new_state_sources)

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
