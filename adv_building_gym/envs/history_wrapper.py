"""HistoryWrapper — per-key rolling history for the Dict observation space.

When enabled via env_meta.hst_env_wrapper in the trial config, this wrapper
replaces each time-varying Dict obs entry (``s_*``, ``a_*_prev``,
``raw_sim_hour``) with an ``(hst_len, *original_shape)`` rolling buffer.
Order: oldest -> newest; pre-padded with zeros until the buffer is full.
``ctxt_*`` keys are episode-static and pass through unchanged.

Alternative to the post-collection StridedHistoryConnector — do not enable
both for the same keys.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict

import gymnasium
import numpy as np
from gymnasium import spaces


def default_key_predicate(key: str) -> bool:
    """Stack keys that vary step-to-step: state signals, action mirrors, sim hour."""
    return key.startswith("s_") or key.endswith("_prev") or key == "raw_sim_hour"


class HistoryWrapper(gymnasium.ObservationWrapper):
    """Rolling per-key observation history (oldest -> newest, zero-padded).

    Args:
        env: A Gymnasium env whose observation space is a ``spaces.Dict``.
        hst_len: Buffer length (number of past steps kept per key, inclusive
            of the current step).
        key_predicate: Callable ``str -> bool``; returns True for keys to stack.
            Defaults to :func:`default_key_predicate`.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        hst_len: int,
        key_predicate: Callable[[str], bool] = default_key_predicate,
    ) -> None:
        super().__init__(env)
        if hst_len <= 0:
            raise ValueError(f"HistoryWrapper: hst_len must be > 0, got {hst_len}")
        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError(
                "HistoryWrapper requires a Dict observation space; got "
                f"{type(env.observation_space).__name__}"
            )

        self.hst_len = int(hst_len)
        self._tracked_keys: list[str] = []
        self._buffers: Dict[str, np.ndarray] = {}

        new_spaces: "OrderedDict[str, spaces.Space]" = OrderedDict()
        for key, sub in env.observation_space.spaces.items():
            if key_predicate(key) and isinstance(sub, spaces.Box):
                new_shape = (self.hst_len, *sub.shape)
                low = np.broadcast_to(sub.low, new_shape).astype(sub.dtype, copy=True)
                high = np.broadcast_to(sub.high, new_shape).astype(sub.dtype, copy=True)
                new_spaces[key] = spaces.Box(low=low, high=high, shape=new_shape, dtype=sub.dtype)
                self._tracked_keys.append(key)
                self._buffers[key] = np.zeros(new_shape, dtype=sub.dtype)
            else:
                new_spaces[key] = sub

        self.observation_space = spaces.Dict(new_spaces)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for key in self._tracked_keys:
            buf = self._buffers[key]
            buf.fill(0)
            buf[-1] = np.asarray(obs[key], dtype=buf.dtype)
        return self._build_obs(obs), info

    def observation(self, obs):
        for key in self._tracked_keys:
            buf = self._buffers[key]
            buf[:-1] = buf[1:]
            buf[-1] = np.asarray(obs[key], dtype=buf.dtype)
        return self._build_obs(obs)

    def _build_obs(self, obs: dict) -> dict:
        out: Dict[str, np.ndarray] = {}
        for key, val in obs.items():
            if key in self._buffers:
                out[key] = self._buffers[key].copy()
            else:
                out[key] = val
        return out
