"""ForecastWrapper — augment the Dict observation space with future-step values.

When enabled via ``env_meta.forecast_env_wrapper`` in the trial config, this
wrapper queries every StateSource's ``forecast(selected_future_steps)`` method
and adds the returned ``s_fc_<var>`` arrays to the observation Dict each step.

The forecast offsets (``forecast_steps``) are positive integers in control-step
units, sorted ascending and deduplicated. Forecast values stay untouched by
HistoryWrapper when this wrapper is applied AFTER it (see env_creator.py
wrapper composition).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import gymnasium
import numpy as np
from gymnasium import spaces

from adv_building_gym.devices.statesources.forecastable import Forecastable


class ForecastWrapper(gymnasium.ObservationWrapper):
    """Adds ``s_fc_<var>`` Box entries with shape ``(len(forecast_steps),)`` to
    the wrapped env's Dict observation space.

    StateSources that implement the ``Forecastable`` interface contribute their
    declared ``s_fc_*`` keys; non-Forecastable sources are ignored. Values are
    returned in the same normalisation as the live ``s_<var>`` key.
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

        # AdvBuildingGym exposes its statesource list on the inner env. We
        # only care about the subset that opts into the Forecastable interface
        # — non-forecasting sources have no s_fc_* contract to honour.
        self._forecasters: tuple[Forecastable, ...] = tuple(
            ds for ds in env.unwrapped.statesources if isinstance(ds, Forecastable)
        )

        # Discover s_fc_* keys via the static Forecastable contract so the
        # space is well-defined even when CSVs have not been loaded yet.
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

        # Mirror the live key's Box bounds when one exists; otherwise use a
        # generous (-inf, inf) Box. The dtype is always float32 to match the
        # rest of the observation buffer.
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
