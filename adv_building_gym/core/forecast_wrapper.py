"""ForecastWrapper — augment the Dict observation space with future-step values.

Queries each StateSource's ``forecast(selected_future_steps)`` and adds the returned
``s_fc_<var>`` arrays to the obs Dict each step. ``forecast_steps`` are positive
control-step offsets (sorted, deduped). 
Applied AFTER HistoryWrapper so forecasts stay untouched.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import gymnasium
import numpy as np
from gymnasium import spaces

from adv_building_gym.components.statesources.forecastable import Forecastable


class ForecastWrapper(gymnasium.ObservationWrapper):
    """Adds ``s_fc_<var>`` Box entries of shape ``(len(forecast_steps),)`` to the Dict obs.

    Only ``Forecastable`` sources contribute their declared ``s_fc_*`` keys (same
    normalisation as the live ``s_<var>``); others are ignored.
    """

    def __init__(self, env: gymnasium.Env, forecast_steps: Iterable[int]) -> None:
        super().__init__(env)

        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError(
                "ForecastWrapper requires a Dict observation space; got "
                f"{type(env.observation_space).__name__}"
            )

        # Defense-in-depth: positive, unique, ascending.
        steps = sorted({int(s) for s in forecast_steps})
        if not steps or steps[0] < 1:
            raise ValueError(f"ForecastWrapper: forecast_steps must be non-empty positive ints, got {forecast_steps!r}")
        
        self._forecast_steps_list: list[int] = list(steps)
        self._n: int = len(self._forecast_steps_list)

        # only the Forecastable subset of the inner env's statesources
        self._forecasters: tuple[Forecastable, ...] = tuple(
            ds for ds in env.unwrapped.statesources if isinstance(ds, Forecastable)
        )

        # discover s_fc_* keys via the static contract (well-defined before any CSV load)
        fc_keys: list[str] = []
        seen: set[str] = set()
        for ds in self._forecasters:
            for k in ds.forecast_keys():
                if not k.startswith("s_fc_"):
                    raise ValueError(
                        f"StateSource '{ds.name}' declared forecast key '{k}' that does not start with 's_fc_'."  # type: ignore[attr-defined]
                    )
                if k in seen:
                    raise ValueError(
                        f"Duplicate forecast key '{k}' declared by StateSource '{ds.name}'."  # type: ignore[attr-defined]
                    )
                seen.add(k)
                fc_keys.append(k)
        self._fc_keys: tuple[str, ...] = tuple(fc_keys)

        # mirror the live key's Box bounds if present, else (-inf, inf); dtype float32
        live_spaces = env.observation_space.spaces
        new_spaces: "OrderedDict[str, spaces.Space]" = OrderedDict(live_spaces)
        for key in self._fc_keys:
            live_key = "s_" + key[len("s_fc_"):]
            shape = (self._n,)
            if live_key in live_spaces and isinstance(live_spaces[live_key], spaces.Box):
                live_box = live_spaces[live_key]
                low_scalar = float(np.min(live_box.low))
                high_scalar = float(np.max(live_box.high))
                low = np.full(shape, low_scalar, dtype=np.float32)
                high = np.full(shape, high_scalar, dtype=np.float32)
            else:
                low = np.full(shape, -np.inf, dtype=np.float32)
                high = np.full(shape, np.inf, dtype=np.float32)
            new_spaces[key] = spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

        self.observation_space = spaces.Dict(new_spaces)

    def observation(self, obs):
        out = dict(obs)
        for ds in self._forecasters:
            for k, v in ds.forecast(self._forecast_steps_list).items():
                out[k] = np.asarray(v, dtype=np.float32)
        return out
