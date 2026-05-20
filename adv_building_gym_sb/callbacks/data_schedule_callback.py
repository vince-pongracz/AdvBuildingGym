"""Data variant scheduling for SB3.

Episode-counting twin of ``adv_building_gym/callbacks/data_schedule_callback.py``.
On each swap window boundary, picks the next variant from the
``DataCombinator`` pool and broadcasts it to every sub-env in BOTH the
training and the eval VecEnv via ``vec_env.env_method("apply_data_variant", variant)``.

Eval mirrors training (same as the Ray callback) so the eval signal
describes the regime the policy is actually being trained on. The
per-episode day-of-year still varies per env via ``DataCombinator.day``,
so eval rollouts aren't degenerate — only the long-horizon variant
tracks training.
"""

from __future__ import annotations

import logging

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from adv_building_gym.data_combinator import DataCombinator

from ._swap_trigger import make_swap_gate

logger = logging.getLogger(__name__)


class _DataScheduleCallback(BaseCallback):
    def __init__(
        self,
        combinator: DataCombinator,
        num_env_runners: int,
        eval_env: VecEnv | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self._combinator = combinator
        self._eval_env = eval_env
        self._gate = make_swap_gate(
            "DataSchedule", combinator.swap_every_n_episodes, num_env_runners,
        )
        self._swap_index = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        self._gate.update(dones)
        decision = self._gate.check()
        if not decision.should_fire:
            return True

        pool = self._combinator.variants
        if not pool:
            return True

        variant = pool[self._swap_index % len(pool)]
        self._swap_index += 1

        # Push to both training and eval envs. env_method uses gymnasium's
        # get_wrapper_attr chain so the call reaches the unwrapped
        # AdvBuildingGym through HistoryWrapper / Forecast / FlattenAction
        # / RescaleAction without manual unwrapping.
        self.training_env.env_method("apply_data_variant", variant)
        if self._eval_env is not None:
            self._eval_env.env_method("apply_data_variant", variant)
        logger.info(
            "DataSchedule: pushed variant %s to %d train + %d eval env(s) "
            "(swap_index=%d).",
            variant, self.training_env.num_envs,
            self._eval_env.num_envs if self._eval_env is not None else 0,
            self._swap_index,
        )
        return True


def make_data_schedule_callback(
    combinator: DataCombinator,
    *,
    num_env_runners: int,
    eval_env: VecEnv | None = None,
) -> BaseCallback:
    return _DataScheduleCallback(combinator, num_env_runners=num_env_runners, eval_env=eval_env)
