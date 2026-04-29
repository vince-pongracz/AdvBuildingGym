"""Rolling action history buffer extracted from AdvBuildingGym.

Owns both the deque consumed by reward functions (e.g.
ActionSmoothnessReward) and the policy-visible ``<key>_prev``
observation entries that mirror the just-applied action.
"""
from collections import deque
from typing import Dict, List

import numpy as np


class ActionHistoryBuffer:
    """Deque of action snapshots + ``<key>_prev`` observation mirror."""

    def __init__(self, action_keys: List[str], max_length: int) -> None:
        self._action_keys = action_keys
        self._deque: deque[Dict[str, np.ndarray]] = deque(maxlen=max_length)

    @property
    def history(self) -> deque:
        return self._deque

    def clear(self) -> None:
        self._deque.clear()

    def append_and_mirror(
        self,
        action: Dict[str, np.ndarray],
        state: Dict[str, np.ndarray],
    ) -> None:
        """Snapshot *action*, push to buffer, write to ``<key>_prev`` obs."""
        snapshot = {
            key: np.asarray(action.get(key, 0.0), dtype=np.float32).copy()
            for key in self._action_keys
        }
        self._deque.append(snapshot)
        for key in self._action_keys:
            dest = state[f"{key}_prev"]
            dest[...] = snapshot[key].astype(dest.dtype).reshape(dest.shape)
