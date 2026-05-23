"""HistoryWrapper — per-key strided observation history as an env wrapper.

For each user-listed Box obs key ``X``, this wrapper adds a new Dict obs
entry ``s_hst_X`` of shape ``(len(offsets), *X.shape)`` containing ``X`` at
each requested offset. The original ``X`` entry passes through unchanged
(mirrors ``ForecastWrapper``'s additive ``s_fc_<var>`` convention).

Pre-episode slots — for which no historical observation exists yet —
remain zero rather than replicating the current frame. Replicating present
values into "past" slots would make early-episode inputs indistinguishable
from a stationary signal; zeros at least encode "no information"
unambiguously.

Listing ``a_<x>_prev`` keys in ``tracked_keys`` just works: the env
publishes them as plain obs entries each step.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Dict, Iterable, List

import gymnasium
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)


class HistoryWrapper(gymnasium.ObservationWrapper):
    """Adds ``s_hst_<key>`` Box entries with strided history to a Dict obs space."""

    def __init__(
        self,
        env: gymnasium.Env,
        tracked_keys: Iterable[str],
        offsets: Iterable[int],
    ) -> None:
        super().__init__(env)

        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError(
                "HistoryWrapper requires a Dict observation space; got "
                f"{type(env.observation_space).__name__}"
            )

        # Use exactly the offsets the user supplies — the current frame
        # (offset 0) is included only if listed explicitly, since the
        # unwrapped obs already exposes the current value under the
        # original key. Preserve user order; dedup keeps first.
        raw_offsets = [int(ofs) for ofs in offsets]
        if any(ofs > 0 for ofs in raw_offsets):
            raise ValueError(f"HistoryWrapper: offsets must be <= 0 (0 = current step). Got: {raw_offsets}")
        seen: set[int] = set()
        ordered_offsets: List[int] = []
        for ofs in raw_offsets:
            if ofs not in seen:
                seen.add(ofs)
                ordered_offsets.append(ofs)
        if not ordered_offsets:
            raise ValueError("HistoryWrapper: offsets must be a non-empty list of non-positive ints.")
        self.offsets: tuple[int, ...] = tuple(ordered_offsets)
        self._max_lookback: int = -min(self.offsets)  # >= 0; 0 means current step only

        live_spaces = env.observation_space.spaces
        requested = list(tracked_keys)
        retained: list[str] = []
        for key in requested:
            sub = live_spaces.get(key)
            if sub is None:
                logger.warning(
                    "HistoryWrapper: tracked key '%s' not found in observation "
                    "space; skipping. Available keys: %s",
                    key, sorted(live_spaces),
                )
                continue
            if not isinstance(sub, spaces.Box):
                logger.warning(
                    "HistoryWrapper: tracked key '%s' is %s, not Box; skipping.",
                    key, type(sub).__name__,
                )
                continue
            hst_key = f"s_hst_{key}"
            if hst_key in live_spaces:
                raise ValueError(
                    f"HistoryWrapper: target key '{hst_key}' already present in "
                    "inner observation space — would clash with the wrapper output."
                )
            retained.append(key)
        self._tracked_keys: tuple[str, ...] = tuple(retained)

        # Pre-allocate one rolling buffer per tracked key. Length is
        # max_lookback + 1 so index -1 is always the current step and
        # index -1 + offset (for offset in self.offsets) addresses each
        # requested lag without bounds checks.
        self._buffers: Dict[str, np.ndarray] = {}
        new_spaces: "OrderedDict[str, spaces.Space]" = OrderedDict(live_spaces)
        n_off = len(self.offsets)
        for key in self._tracked_keys:
            sub: spaces.Box = live_spaces[key]  # type: ignore[assignment]
            stacked_shape = (n_off, *sub.shape)
            low = np.broadcast_to(sub.low, stacked_shape).astype(sub.dtype, copy=True)
            high = np.broadcast_to(sub.high, stacked_shape).astype(sub.dtype, copy=True)
            new_spaces[f"s_hst_{key}"] = spaces.Box(
                low=low, high=high, shape=stacked_shape, dtype=sub.dtype,
            )
            buf_shape = (self._max_lookback + 1, *sub.shape)
            self._buffers[key] = np.zeros(buf_shape, dtype=sub.dtype)

        self.observation_space = spaces.Dict(new_spaces)

        if not self._tracked_keys:
            logger.warning(
                "HistoryWrapper: no valid tracked keys remain after filtering; "
                "wrapper is a no-op."
            )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for key in self._tracked_keys:
            buf = self._buffers[key]
            buf.fill(0)
            buf[-1] = np.asarray(obs[key], dtype=buf.dtype)
        return self._build_obs(obs), info

    def observation(self, obs):
        # Step path: roll buffers (oldest dropped), write the new sample
        # into the last slot, then assemble the stacked s_hst_<key> entries.
        for key in self._tracked_keys:
            buf = self._buffers[key]
            buf[:-1] = buf[1:]
            buf[-1] = np.asarray(obs[key], dtype=buf.dtype)
        return self._build_obs(obs)

    def _build_obs(self, obs: dict) -> dict:
        out = dict(obs)
        for key in self._tracked_keys:
            buf = self._buffers[key]
            # buf[-1] = current; for negative offset o, buf[-1 + o] = lag |o|.
            frames = [buf[-1 + ofs] for ofs in self.offsets]
            out[f"s_hst_{key}"] = np.stack(frames, axis=0)
        return out
