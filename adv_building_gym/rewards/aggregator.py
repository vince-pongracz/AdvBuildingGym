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
        states: Dict,
        info: Dict,
    ) -> Tuple[float, Dict[str, float], float]:
        """Return ``(total_reward, breakdown, total_max_step)``."""


class SumRewardAggregator(RewardAggregator):
    """Default aggregator — sum of weighted component rewards."""

    def aggregate(self, reward_funcs, actions, states, info):
        total_reward = 0.0
        total_max_step = 0.0
        breakdown: Dict[str, float] = {}
        for rf in reward_funcs:
            r, r_max = rf.get_reward(actions, states, info=info)
            r_val = float(np.asarray(r).item())
            breakdown[rf.name] = r_val
            total_reward += r_val
            total_max_step += r_max
        return total_reward, breakdown, total_max_step
