"""Reward aggregation strategy used by AdvBuildingGym.

Keeps the env open to alternative aggregation policies (weighted blend,
curriculum gating, ...) without forcing edits to ``step()``. The default
strategy is a plain sum, matching the previous inlined behaviour.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np

from .base import RewardFunction


class RewardAggregator(ABC):
    """Strategy interface for combining per-component rewards into one scalar."""

    @abstractmethod
    def aggregate(
        self,
        reward_funcs: List[RewardFunction],
        actions: Dict,
        state: Dict,
        next_state: Dict,
        info: Dict,
    ) -> Tuple[float, Dict[str, float]]:
        """Combine per-component rewards over the (s, a, s') transition.

        ``state`` is the observed state ``s``; ``next_state`` is the
        resulting state ``s'``. Returns ``(total_reward, breakdown)``.
        """


class SumRewardAggregator(RewardAggregator):
    """Default aggregator — sum of weighted component rewards."""

    def aggregate(self, reward_funcs, actions, state, next_state, info):
        total_reward = 0.0
        breakdown: Dict[str, float] = {}
        for rf in reward_funcs:
            reward = float(np.asarray(rf.get_reward(actions, state, next_state, info=info)).item())
            breakdown[rf.name] = reward
            total_reward += reward
        return total_reward, breakdown

class NashRewardAggregator(RewardAggregator):
    """Nash equilibrium-based aggregator — product of weighted component rewards."""

    def aggregate(self, reward_funcs, actions, state, next_state, info):
        total_reward = 1.0
        breakdown: Dict[str, float] = {}
        for rf in reward_funcs:
            reward = float(np.asarray(rf.get_01_reward(actions, state, next_state, info=info)).item())
            breakdown[rf.name] = reward
            total_reward *= reward
        return total_reward, breakdown
