"""Multi-agent wrapper around :class:`AdvBuildingGym` (Flavour A).

Each actuator is an agent with its own policy: the inner Dict action is partitioned by
key (``a_<name>``), observations are shared, rewards routed per-agent via a configurable
partition. No ``FlattenAction + RescaleAction`` chain — each agent emits ``Box(-1, 1)``
rescaled to the infra's native bounds in :meth:`step`.

Links: docs/about_multi_agent.md (Flavour A) ;
https://docs.ray.io/en/latest/rllib/multi-agent-envs.html
"""

# TODO VP 2026.09.30.: not used

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from gymnasium.spaces import Box, Dict as SDict
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID

from adv_building_gym.config.env.env_config import EnvConfig
from adv_building_gym.config.data.data_combinator import DataCombinator
from adv_building_gym.components.infrastructure import Infrastructure
from adv_building_gym.components.statesources import StateSource
from adv_building_gym.core.env import AdvBuildingGym
from adv_building_gym.components.rewards import RewardFunction
from adv_building_gym.components.rewards.aggregator import SumRewardAggregator

logger = logging.getLogger(__name__)


def _action_key_to_agent_id(action_key: str) -> str:
    """``a_hp`` → ``hp``; preserves keys without the ``a_`` prefix as-is."""
    return action_key[2:] if action_key.startswith("a_") else action_key


def _rescale(unit_action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Map ``Box(-1, 1)`` → ``Box(low, high)`` element-wise."""
    return low + (np.clip(unit_action, -1.0, 1.0) + 1.0) * 0.5 * (high - low)


class MultiAgentAdvBuildingGym(MultiAgentEnv):
    """Per-actuator multi-agent view of :class:`AdvBuildingGym`.

    Agent id = action key minus the ``a_`` prefix (``a_hp`` → ``hp``). Each agent has a unit
    Box action (rescaled in :meth:`step`); the full Dict obs is broadcast to all agents.
    ``reward_partition`` maps agent_id → reward names; ``None`` gives every agent the global
    reward (cooperative MARL).
    """

    def __init__(
        self,
        infras: list[Infrastructure],
        statesources: list[StateSource],
        rewards: list[RewardFunction],
        env_config: EnvConfig,
        *,
        data_combinator: DataCombinator,
        instance_id: str | None = None,
        reward_partition: dict[str, list[str]] | None = None,
        **kwargs,
    ):
        super().__init__()

        self._inner = AdvBuildingGym(
            env_config=env_config,
            reward_aggregator=SumRewardAggregator(),
            infras=infras,
            statesources=statesources,
            rewards=rewards,
            data_combinator=data_combinator,
            instance_id=instance_id,
        )

        # agent registry from inner action keys; order matters for policy registration
        self._action_keys: list[str] = list(self._inner.action_space_keys)
        self._agent_to_action_key: dict[str, str] = {
            _action_key_to_agent_id(k): k for k in self._action_keys
        }
        # Cache native action bounds for rescaling.
        self._agent_bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for agent_id, akey in self._agent_to_action_key.items():
            box = self._inner.action_space.spaces[akey]
            self._agent_bounds[agent_id] = (
                np.asarray(box.low, dtype=np.float32),
                np.asarray(box.high, dtype=np.float32),
            )

        self.possible_agents = list(self._agent_to_action_key.keys())
        self.agents = list(self.possible_agents)

        # per-agent spaces: shared full Dict obs, unit-Box actions of native shape
        self.observation_spaces = {
            agent_id: self._inner.observation_space for agent_id in self.possible_agents
        }
        self.action_spaces = {
            agent_id: Box(
                low=-1.0, high=1.0,
                shape=self._inner.action_space.spaces[akey].shape,
                dtype=np.float32,
            )
            for agent_id, akey in self._agent_to_action_key.items()
        }

        self._reward_partition = self._validate_partition(reward_partition)

        logger.info(
            "MultiAgentAdvBuildingGym created with agents=%s, partition=%s",
            self.possible_agents,
            "cooperative" if reward_partition is None else "custom",
        )

    # ------------------------------------------------------------------
    # Pass-through APIs (kept for callbacks that look these up)
    # ------------------------------------------------------------------

    @property
    def inner(self) -> AdvBuildingGym:
        return self._inner

    @property
    def log_full_info(self) -> bool:
        return self._inner.log_full_info

    @log_full_info.setter
    def log_full_info(self, value: bool) -> None:
        self._inner.log_full_info = bool(value)

    def apply_data_variant(self, variant: dict[str, str]) -> None:
        self._inner.apply_data_variant(variant)

    def set_reward_funcs(self, rewards: list[RewardFunction]) -> None:
        self._inner.set_reward_funcs(rewards)
        # Drop any partition entries that reference rewards no longer present.
        if self._reward_partition is not None:
            present = {r.name for r in rewards}
            self._reward_partition = {
                agent_id: [n for n in names if n in present]
                for agent_id, names in self._reward_partition.items()
            }

    def set_infras(self, infras: list[Infrastructure]) -> None:
        # inner env enforces name-equality on swap → action keys (and agent ids) stay stable
        self._inner.set_infras(infras)
        
        # TODO VP 2026.05.04.: What if the infras should not match anymore with the previous
        # setup and adding a new infra means instantiating and new agent? -- so we dynamically can instantiate
        # actuators with policies. If an actuator was already active, then it is not active, 
        # then it's active again -- bring back the old actuator and train that instead of initiating a new one.
        # So each infra element/actuator/agent has an is_active property -- if agent is active, it learns -- 
        # otherwise there isn't any update/training on its policy

    # ------------------------------------------------------------------
    # MultiAgentEnv API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self._inner.reset(seed=seed, options=options)
        self.agents = list(self.possible_agents)
        obs_dict = {agent_id: self._copy_obs(obs) for agent_id in self.agents}
        info_dict = {agent_id: info for agent_id in self.agents}
        return obs_dict, info_dict

    def step(self, action_dict: dict[str, np.ndarray]):
        inner_action = self._assemble_inner_action(action_dict)
        obs, _global_reward, terminated, truncated, info = self._inner.step(inner_action)

        rewards = self._route_rewards(info.get("reward_breakdown", {}), _global_reward)

        obs_out = {agent_id: self._copy_obs(obs) for agent_id in self.agents}
        info_out = {agent_id: info for agent_id in self.agents}

        terminated_out = {agent_id: bool(terminated) for agent_id in self.agents}
        truncated_out = {agent_id: bool(truncated) for agent_id in self.agents}
        terminated_out["__all__"] = bool(terminated)
        truncated_out["__all__"] = bool(truncated)

        return obs_out, rewards, terminated_out, truncated_out, info_out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assemble_inner_action(self, action_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Convert per-agent unit actions to the inner Dict action; missing agents default to
        the box centre (most neutral)."""
        inner: dict[str, np.ndarray] = {}
        for agent_id, akey in self._agent_to_action_key.items():
            low, high = self._agent_bounds[agent_id]
            if agent_id in action_dict:
                inner[akey] = _rescale(np.asarray(action_dict[agent_id], dtype=np.float32), low, high)
            else:
                inner[akey] = ((low + high) * 0.5).astype(np.float32)
        return inner

    def _route_rewards(
        self,
        breakdown: dict[str, float],
        global_reward: float,
    ) -> dict[AgentID | str, float]:
        """Distribute step rewards. Cooperative default: each agent gets ``global_reward``;
        with a partition, the sum of its assigned ``breakdown[name]`` (absent → 0)."""
        if self._reward_partition is None:
            return {
                agent_id: float(global_reward) 
                for agent_id in self.agents
            }
        return {
            agent_id: float(sum(breakdown.get(name, 0.0) for name in self._reward_partition.get(agent_id, [])))
            for agent_id in self.agents
        }

    def _validate_partition(
        self,
        reward_partitioning: dict[str, list[str]] | None,
    ) -> dict[AgentID | str, list[str]] | None:
        if reward_partitioning is None:
            return None
        unknown = set(reward_partitioning) - set(self.possible_agents)
        if unknown:
            raise ValueError(
                f"reward_partition references unknown agent(s) {sorted(unknown)}; "
                f"expected subset of {self.possible_agents}"
            )

        return {
            agent_id: list(reward_partitioning.get(agent_id, []))
            for agent_id in self.possible_agents
        }

    @staticmethod
    def _copy_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: np.array(v, copy=True) for k, v in obs.items()}
