"""Synthetic energy-price source: a date/season-aware time-of-use tariff (no CSV)."""

import logging
from dataclasses import dataclass
from typing import ClassVar, Sequence, Set

import numpy as np
from gymnasium.spaces import Box

from ..base import StateSource
from ..forecastable import Forecastable
from ..lookahead import Lookahead
from adv_building_gym.components.registry import ComponentRegistry
from adv_building_gym._common.constants import SECONDS_PER_HOUR
from adv_building_gym._common.season import season_for_month

logger = logging.getLogger(__name__)


def _validate_window(window: Sequence[int], field: str) -> list[int]:
    """Validate a ``[start, end)`` hour window (each 0-24); return it as an int list
    (a list, not a tuple, so ``Serializable.to_dict`` round-trips it)."""
    seq = [int(h) for h in window]
    if len(seq) != 2:
        raise ValueError(f"{field} must be a [start, end) pair; got {window!r}.")
    start, end = seq
    if not (0 <= start <= 24 and 0 <= end <= 24):
        raise ValueError(f"{field} hours must be within 0-24; got {window!r}.")
    return seq


def _in_window(hour: float, window: list[int]) -> bool:
    """Whether ``hour`` is in ``[start, end)``, wrapping midnight when start > end."""
    start, end = window
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


@dataclass(frozen=True)
class _TouSchedule:
    """3-level time-of-use tariff: peak wins over off-peak on overlap, else mid."""

    off_peak_price: float
    mid_price: float
    peak_price: float
    off_peak_hours: list[int]
    peak_hours: list[int]

    def level_for_hour(self, hour: float) -> float:
        if _in_window(hour, self.peak_hours):
            return self.peak_price
        if _in_window(hour, self.off_peak_hours):
            return self.off_peak_price
        return self.mid_price

    @property
    def level_magnitudes(self) -> tuple[float, ...]:
        return (abs(self.off_peak_price), abs(self.mid_price), abs(self.peak_price))


class EnergyPriceFixDataSource(StateSource, Forecastable, Lookahead):
    """Synthetic 3-level time-of-use tariff selected by the episode's season (no CSV).

    ``schedules`` maps a season (winter/spring/summer/autumn) or the catch-all ``"all"`` to a
    ``{off_peak_price, mid_price, peak_price, off_peak_hours, peak_hours}`` block (each window is
    ``[start, end)`` hours, may wrap midnight). The episode season comes from
    ``info["episode_date"]`` (published by DateSource); a missing season falls back to ``"all"``
    then ``default_season``.

    ``s_E_price = baseprice / divisor`` (not clipped); the divisor is the largest level magnitude
    across all schedules — a stable per-run scale, published as ``ctxt_E_price_max`` only when
    ``emit_ctxt`` is set. Hour of day = ``(effective_index * timestep / 3600) mod 24``.
    """

    _context_params: ClassVar[Set[str]] = {"timestep"}

    _DEFAULT_SCHEDULE: ClassVar[dict] = {
        "off_peak_price": 18.0, "mid_price": 28.0, "peak_price": 40.0,
        "off_peak_hours": [0, 6], "peak_hours": [17, 21],
    }

    def __init__(self, name: str,
                schedules: dict | None = None,
                default_season: str = "winter",
                emit_ctxt: bool = False,
                timestep: float = 300.0) -> None:
        super().__init__(name=name)

        self.schedules = schedules if schedules else {"all": dict(self._DEFAULT_SCHEDULE)}
        self.default_season = str(default_season)
        self.emit_ctxt = bool(emit_ctxt)
        self.timestep = float(timestep)

        self._schedules = {season: self._build(block) for season, block in self.schedules.items()}

        # Fixed divisor: largest level magnitude across all schedules (stable per-run scale).
        self.price_divisor: float = max(
            1e-6, max(mag for s in self._schedules.values() for mag in s.level_magnitudes)
        )

        self._active = self._select_schedule(self.default_season)
        # Raw baseprice (ct/kWh) of the current step — read for billing via get_raw_values.
        self.baseprice_raw: float = 0.0

    @staticmethod
    def _build(block: dict) -> _TouSchedule:
        return _TouSchedule(
            off_peak_price=float(block["off_peak_price"]),
            mid_price=float(block["mid_price"]),
            peak_price=float(block["peak_price"]),
            off_peak_hours=_validate_window(block["off_peak_hours"], "off_peak_hours"),
            peak_hours=_validate_window(block["peak_hours"], "peak_hours"),
        )

    def _select_schedule(self, season: str) -> _TouSchedule:
        for key in (season, "all", self.default_season):
            if key in self._schedules:
                return self._schedules[key]
        return next(iter(self._schedules.values()))

    def _episode_season(self, info: dict | None) -> str:
        """Season from ``info['episode_date']`` (ISO ``YYYY-MM-DD``), else ``default_season``."""
        date = (info or {}).get("episode_date")
        parts = str(date).split("-") if date else []
        if len(parts) == 3 and parts[0].isdigit() and len(parts[0]) == 4:
            return season_for_month(int(parts[1]))
        return self.default_season

    def _hour_of_day(self, index: int) -> float:
        return (index * self.timestep / SECONDS_PER_HOUR) % 24.0

    def _normalise(self, raw: float) -> float:
        return float(raw / self.price_divisor)

    def setup_spaces(self, state_spaces, action_spaces) -> tuple:
        if "s_E_price" not in state_spaces:
            state_spaces["s_E_price"] = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self._publish_ctxt(state_spaces, "ctxt_E_price_max", Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32))
        return state_spaces, action_spaces

    def reset(self, states, info=None) -> None:
        """Pick this episode's schedule from its season, then publish the first observation."""
        self._active = self._select_schedule(self._episode_season(info))
        self.update_state(states, info)

    def update_state(self, states, info=None) -> None:
        self.baseprice_raw = self._active.level_for_hour(self._hour_of_day(self.effective_index))
        states["s_E_price"][0] = np.float32(self._normalise(self.baseprice_raw))
        self._write_ctxt(states, "ctxt_E_price_max", np.float32(self.price_divisor))

    # ----- Lookahead (analytic; no CSV) -----
    def lookahead_keys(self) -> tuple[str, ...]:
        return ("baseprice",)

    def lookahead(self, steps: list[int]) -> dict[str, list[float]]:
        return {"baseprice": [
            self._active.level_for_hour(self._hour_of_day(self.effective_index + s)) for s in steps
        ]}

    # ----- Forecastable -----
    def forecast_keys(self) -> tuple[str, ...]:
        return ("s_fc_E_price",)

    def forecast(self, selected_future_steps: list[int]) -> dict[str, list[float]]:
        raw = self.lookahead(selected_future_steps)["baseprice"]
        return {"s_fc_E_price": [self._normalise(v) for v in raw]}

    def get_raw_values(self) -> dict[str, float]:
        return {"raw_E_price": self.baseprice_raw}


# register with ComponentRegistry
ComponentRegistry.register('statesource', EnergyPriceFixDataSource)
