"""ForecastWrapper — augment the Dict observation space with future-step values.

Queries each ``Forecastable`` component's ``forecast(selected_future_steps)`` and adds
the returned ``<frame>_fc_<var>`` arrays to the obs Dict each step.
Both statesources (e.g. weather/price look-ahead) and infrastructure (e.g. PV / wind power, derived from the
future weather they read off the info channel) may be Forecastable.
``forecast_steps`` are positive control-step offsets (sorted, deduped).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Optional

import gymnasium
import numpy as np
from gymnasium import spaces

from adv_building_gym._common.forecasting import Forecastable

# Forecast keys mirror a live observation key with an ``fc_`` segment inserted: 
# ``s_fc_<var>`` ↔ ``s_<var>`` (normalised states),
# ``ctxt_fc_<var>`` ↔ ``ctxt_<var>`` (slow-changing context scalars).
_FORECAST_PREFIXES: dict[str, str] = {
    "s_fc_": "s_", 
    "ctxt_fc_": "ctxt_"
}


def live_key_for(fc_key: str) -> Optional[str]:
    """The live observation key a forecast key mirrors, or None if its prefix is unknown."""
    for fc_prefix, live_prefix in _FORECAST_PREFIXES.items():
        if fc_key.startswith(fc_prefix):
            return live_prefix + fc_key[len(fc_prefix):]
    return None


def canonise_forecast_steps(forecast_steps: Iterable[int]) -> list[int]:
    """Canonical look-ahead offsets: positive, unique, ascending."""
    steps = sorted({int(s) for s in forecast_steps})
    if not steps or steps[0] < 1:
        raise ValueError(f"forecast_steps must be non-empty positive ints, got {forecast_steps!r}")
    return steps


class ForecastWrapper(gymnasium.ObservationWrapper):
    """Adds ``<frame>_fc_<var>`` Box entries of shape ``(len(forecast_steps),)`` to the Dict obs.

    Only ``Forecastable`` sources contribute their declared forecast keys, each mirroring a
    live observation key in the same frame (``s_fc_*`` ↔ ``s_*``, ``ctxt_fc_*`` ↔ ``ctxt_*``);
    others are ignored.
    """

    def __init__(self, env: gymnasium.Env, forecast_steps: Iterable[int]) -> None:
        super().__init__(env)

        if not isinstance(env.observation_space, spaces.Dict):
            raise TypeError(
                "ForecastWrapper requires a Dict observation space; got "
                f"{type(env.observation_space).__name__}"
            )

        # Canonical step set (shared with the info-channel producers via canonise_forecast_steps).
        self._forecast_steps_list: list[int] = canonise_forecast_steps(forecast_steps)
        self._n: int = len(self._forecast_steps_list)

        # the Forecastable subset of the inner env's components (statesources first so
        # weather/price look-ahead is registered before infra-derived forecasts like
        # PV/wind power, which read the same future weather)
        inner = env.unwrapped
        self._forecasters: tuple[Forecastable, ...] = tuple(
            c for c in (*inner.statesources, *inner.infras) if isinstance(c, Forecastable)
        )

        # discover s_fc_* keys via the static contract (well-defined before any CSV load)
        fc_keys: list[str] = []
        seen: set[str] = set()
        for ds in self._forecasters:
            for k in ds.forecast_keys():
                if live_key_for(k) is None:
                    raise ValueError(f"StateSource '{ds.name}' declared forecast key '{k}' that does not start with one of {tuple(_FORECAST_PREFIXES)}.") # type: ignore[attr-defined]
                if k in seen:
                    raise ValueError(f"Duplicate forecast key '{k}' declared by StateSource '{ds.name}'.") # type: ignore[attr-defined]
                seen.add(k)
                fc_keys.append(k)
        self._fc_keys: tuple[str, ...] = tuple(fc_keys)

        # mirror the live key's Box bounds if present, else (-inf, inf); dtype float32
        # TODO VP 2026.06.24.: Why this specific guard?
        live_spaces = env.observation_space.spaces
        new_spaces: "OrderedDict[str, spaces.Space]" = OrderedDict(live_spaces)
        for key in self._fc_keys:
            live_key = live_key_for(key)  # non-None: validated above
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
        out_obs = dict(obs)
        for ds in self._forecasters:
            for k, v in ds.forecast(self._forecast_steps_list).items():
                out_obs[k] = np.asarray(v, dtype=np.float32)
        return out_obs
