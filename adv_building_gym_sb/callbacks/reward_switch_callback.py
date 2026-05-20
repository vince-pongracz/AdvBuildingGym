"""Reward schedule switching for SB3.

Episode-counting twin of ``reward_switch_callback.py``. On each swap
window boundary advances the ``RewardScheduleManager`` and pushes the
new active reward set to every sub-env in BOTH train and eval VecEnvs
via ``env_method("set_reward_funcs", rewards)``.

Each sub-env gets its own *fresh* reward instances (the manager's
``create_active_rewards()`` is called per sub-env) so per-env reward
state (e.g. action history references) stays independent.

When ``exploration_reset.fires_on("on_reward_swap")`` and the active
set actually changes, the SB3 exploration reset utility bumps SAC's
``log_ent_coef`` (or PPO's ``ent_coef``) and any per-optimiser LR
multipliers, then decays them back over ``decay_iterations`` env-step
ticks of this callback.

Note on SAC: when the reward set changes, transitions already in the
replay buffer carry rewards computed under the previous set. Keep
``swap_every_n_episodes`` large compared to buffer turnover to avoid
mixing reward signals. Same caveat as the Ray version.
"""

from __future__ import annotations

import logging

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from adv_building_gym.config.exploration_reset import ExplorationResetConfig
from adv_building_gym.config.reward_schedule_manager import RewardScheduleManager

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
        # Exploration-reset hooks (see _exploration_reset_util.py). The
        # decay loop runs every _on_step; fire_bump triggers on swap.
        self._maybe_decay, self._fire_bump = make_decay_loop(
            exploration_reset or ExplorationResetConfig(),
            event="on_reward_swap",
        )

    def _on_step(self) -> bool:
        # Decay any in-flight exploration bump from a previous swap.
        # Step-aligned (not iter-aligned like RLlib) — fine since SB3
        # has no iteration concept; we just spread the decay over
        # decay_iterations env-step callbacks.
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
