"""Reward schedule switching for SB3 (episode-counting twin of the Ray callback).

On each swap boundary advances the ``RewardScheduleManager`` and pushes fresh active rewards
to every sub-env in both train and eval VecEnvs (``env_method("set_reward_funcs", ...)``); each
sub-env gets its own instances. On change, fires the exploration kick (SAC: raise the
temperature, auto-relaxed; PPO: raise entropy then decay over ``decay_iterations`` ticks).
SAC caveat: keep ``swap_every_n_episodes`` large vs buffer turnover.
"""

from __future__ import annotations

import logging

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from adv_building_gym.config.training.exploration_reset import ExplorationResetConfig
from adv_building_gym.config.rewards.reward_schedule_manager import RewardScheduleManager

from ._exploration_reset_util import make_decay_loop
from ._swap_trigger import make_swap_gate

logger = logging.getLogger(__name__)


class _RewardSwitchCallback(BaseCallback):
    def __init__(
        self,
        reward_manager: RewardScheduleManager,
        num_env_runners: int,
        exploration_reset: ExplorationResetConfig | None,
        eval_env: VecEnv | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self._reward_manager = reward_manager
        self._eval_env = eval_env
        self._gate = make_swap_gate(
            "RewardSchedule",
            reward_manager.swap_every_n_episodes,
            num_env_runners,
        )
        # exploration-reset hooks (see _exploration_reset_util.py): 
        # decay loop every _on_step, fire_bump on swap
        self._maybe_decay, self._fire_bump = make_decay_loop(
            exploration_reset or ExplorationResetConfig(),
            event="on_reward_swap",
        )

    def _on_step(self) -> bool:
        # decay any in-flight bump (step-aligned; SB3 has no iteration concept)
        self._maybe_decay(self.model)

        dones = self.locals.get("dones")
        self._gate.update(dones)
        decision = self._gate.check()
        if not decision.should_fire:
            return True

        if decision.is_first_fire:
            self._push_current()
            return True

        changed = self._reward_manager.advance()
        if not changed:
            return True
        self._push_current()
        self._fire_bump(self.model)
        return True

    def _push_current(self) -> None:
        """Send one freshly-instantiated reward list per sub-env (train + eval)."""
        for vec_env in (self.training_env, self._eval_env):
            if vec_env is None:
                continue
            for env_idx in range(vec_env.num_envs):
                rewards = self._reward_manager.create_active_rewards()
                vec_env.env_method("set_reward_funcs", rewards, indices=[env_idx])
        logger.info(
            "RewardSchedule: pushed active rewards %s to %d train + %d eval env(s).",
            self._reward_manager.get_active_reward_names(),
            self.training_env.num_envs,
            self._eval_env.num_envs if self._eval_env is not None else 0,
        )


def make_reward_switch_callback(
    reward_manager: RewardScheduleManager,
    *,
    num_env_runners: int,
    exploration_reset: ExplorationResetConfig | None = None,
    eval_env: VecEnv | None = None,
) -> BaseCallback:
    return _RewardSwitchCallback(
        reward_manager,
        num_env_runners=num_env_runners,
        exploration_reset=exploration_reset,
        eval_env=eval_env,
    )
