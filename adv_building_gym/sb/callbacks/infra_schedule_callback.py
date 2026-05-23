"""Infra schedule switching for SB3.

Episode-counting twin of ``infra_schedule_callback.py``. Each sub-env
(train + eval) gets its own fresh infra instances via
``InfraCombinator.create_infras`` so internal state (iteration counters,
action history references) stays independent. Eval follows training so
the eval signal describes the configuration currently being trained.
"""

from __future__ import annotations

import logging

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from adv_building_gym.config.training.exploration_reset import ExplorationResetConfig
from adv_building_gym.config.env.infra_combinator import InfraCombinator

from ._exploration_reset_util import make_decay_loop
from ._swap_trigger import make_swap_gate

logger = logging.getLogger(__name__)


class _InfraScheduleCallback(BaseCallback):
    def __init__(
        self,
        infra_combinator: InfraCombinator,
        num_env_runners: int,
        exploration_reset: ExplorationResetConfig | None,
        eval_env: VecEnv | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self._combinator = infra_combinator
        self._eval_env = eval_env
        self._gate = make_swap_gate(
            "InfraSchedule",
            infra_combinator.swap_every_n_episodes,
            num_env_runners,
        )
        self._maybe_decay, self._fire_bump = make_decay_loop(
            exploration_reset or ExplorationResetConfig(),
            event="on_infra_swap",
        )

    def _on_step(self) -> bool:
        self._maybe_decay(self.model)

        dones = self.locals.get("dones")
        self._gate.update(dones)
        decision = self._gate.check()
        if not decision.should_fire:
            return True

        if decision.is_first_fire:
            self._push_current()
            return True

        changed = self._combinator.advance()
        if not changed:
            return True
        self._push_current()
        self._fire_bump(self.model)
        return True

    def _push_current(self) -> None:
        swap_index = self._combinator._swap_index
        for vec_env, split in (
            (self.training_env, "train"),
            (self._eval_env, "eval"),
        ):
            if vec_env is None:
                continue
            for env_idx in range(vec_env.num_envs):
                new_infras = self._combinator.create_infras(swap_index, split=split)
                vec_env.env_method("set_infras", new_infras, indices=[env_idx])
        logger.info(
            "InfraSchedule: pushed config '%s' (swap_index=%d) to %d train + %d eval env(s).",
            self._combinator.get_active_config_name(), swap_index,
            self.training_env.num_envs,
            self._eval_env.num_envs if self._eval_env is not None else 0,
        )


def make_infra_schedule_callback(
    infra_combinator: InfraCombinator,
    *,
    num_env_runners: int,
    exploration_reset: ExplorationResetConfig | None = None,
    eval_env: VecEnv | None = None,
) -> BaseCallback:
    return _InfraScheduleCallback(
        infra_combinator,
        num_env_runners=num_env_runners,
        exploration_reset=exploration_reset,
        eval_env=eval_env,
    )
