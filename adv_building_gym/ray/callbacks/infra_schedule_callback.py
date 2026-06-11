"""Episode-budget-aligned infrastructure scheduling via on_train_result.

Pushes fresh Infrastructure instances to all env_runners simultaneously once the swap gate
fires. ``create_infra_schedule_on_train_result_cb`` returns the callable.
"""

import logging

from adv_building_gym.ray.callbacks._exploration_reset_util import make_decay_loop
from adv_building_gym.ray.callbacks._swap_trigger import make_swap_gate
from adv_building_gym.config.training.exploration_reset import ExplorationResetConfig
from adv_building_gym.config.env.infra_combinator import InfraCombinator

logger = logging.getLogger(__name__)


def create_infra_schedule_on_train_result_cb(
    infra_combinator: InfraCombinator,
    num_env_runners: int,
    exploration_reset: ExplorationResetConfig | None = None,
):
    """Factory that returns an ``on_train_result`` callable."""
    expl_cfg = exploration_reset or ExplorationResetConfig()
    _state, maybe_decay, fire_bump = make_decay_loop(expl_cfg, "on_infra_swap")
    gate = make_swap_gate(
        "InfraSchedule", infra_combinator.swap_every_n_episodes, num_env_runners,
    )

    def on_train_result(*, algorithm, result: dict, **kwargs) -> None:
        iteration: int = result.get("training_iteration", 0)
        maybe_decay(algorithm, iteration)
        decision = gate(iteration, result)
        if not decision.should_fire:
            return
        if decision.is_first_fire:
            # Push the initial infra config without advancing the cycle.
            _push_infras_to_runners(algorithm, infra_combinator)
            return
        changed = infra_combinator.advance()
        if not changed:
            return
        _push_infras_to_runners(algorithm, infra_combinator)
        fire_bump(algorithm, iteration)

    return on_train_result


def _push_infras_to_runners(
    algorithm,
    infra_combinator: InfraCombinator,
) -> None:
    """Create fresh infra instances and push to all env_runners."""

    swap_index = infra_combinator._swap_index

    def apply(env_runner) -> None:
        # env wrapping: DictInfoToList → SyncVectorEnv → [... → AdvBuildingGym].
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

    algorithm.env_runner_group.foreach_env_runner(apply, local_env_runner=True, timeout_seconds=None)
    if algorithm.eval_env_runner_group is not None:
        algorithm.eval_env_runner_group.foreach_env_runner(apply, local_env_runner=True, timeout_seconds=None)

    logger.info(
        "Iteration %d: all env_runners switched to infra config '%s' "
        "(swap_index=%d)",
        algorithm.training_iteration,
        infra_combinator.get_active_config_name(),
        swap_index,
    )
