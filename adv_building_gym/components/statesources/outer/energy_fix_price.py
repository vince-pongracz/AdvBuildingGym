"""Synthetic energy-price source: a date/season-aware time-of-use tariff (no CSV)."""

import logging
from dataclasses import dataclass
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from ..base import StateSource
from ..forecastable import Forecastable
from ..lookahead import Lookahead
from adv_building_gym.components.registry import ComponentRegistry, Serializable
from adv_building_gym._common.constants import SECONDS_PER_HOUR
from adv_building_gym._common.season import season_for_month

logger = logging.getLogger(__name__)


def _hhmm_to_hours(value: str) -> float:
    """Fractional hour-of-day for an ``"HH:MM"`` (or bare ``"HH"``) string, validated to [0, 24]."""
    hh, _, mm = str(value).partition(":")
    hours = int(hh) + (int(mm) / 60.0 if mm else 0.0)
    if not 0.0 <= hours <= 24.0:
        raise ValueError(f"PriceBand bound must be within 00:00-24:00; got {value!r}.")
    return hours


@dataclass(frozen=True)
class PriceBand(Serializable):
    """A time-of-use price band: ``price`` (ct/kWh) applies while the time-of-day is in
    ``[start, end)``, where ``start``/``end`` are ``"HH:MM"`` strings (minute resolution,
    wrapping past midnight when ``start > end``).

    Subclasses ``Serializable`` so a list of bands round-trips through ``Serializable.to_dict``
    (plain dicts inside a list are dropped by the serialiser)."""

    start: str
    end: str
    price: float

    def __post_init__(self) -> None:
        _hhmm_to_hours(self.start)  # raises on a malformed / out-of-range bound
        _hhmm_to_hours(self.end)

    @classmethod
    def from_mapping(cls, mapping: "PriceBand | dict") -> "PriceBand":
        """Build from a band instance or a ``{start, end, price}`` mapping; any ``class`` key
        left over from a serialised round-trip is ignored."""
        if isinstance(mapping, PriceBand):
            return mapping
        return cls(start=str(mapping["start"]), end=str(mapping["end"]), price=float(mapping["price"]))

    def contains(self, hour: float) -> bool:
        """Whether ``hour`` (fractional hour-of-day) falls in ``[start, end)``, wrapping midnight."""
        start, end = _hhmm_to_hours(self.start), _hhmm_to_hours(self.end)
        if start <= end:
            return start <= hour < end
        return hour >= start or hour < end


# A schedule is a plain ``{"default_price": float, "bands": list[PriceBand]}`` block — also the
# config / serialised shape, so it is the single source of truth (no parallel representation).
Schedule = dict


class EnergyPriceFixDataSource(StateSource, Forecastable, Lookahead):
    """Synthetic multi-level time-of-use tariff selected by the episode's season (no CSV).

    ``schedules`` maps a season (winter/spring/summer/autumn) or the catch-all ``"all"`` to a
    ``{default_price, bands}`` block, where ``bands`` is a list of ``{start, end, price}`` entries
    (``start``/``end`` are ``"HH:MM"`` times in 00:00-24:00, may wrap midnight). The first band
    covering the time-of-day wins; hours outside every band fall back to ``default_price`` — so
    off-peak / mid / peak is just two bands plus a default, and richer schedules add bands. The
    episode season comes from ``info["episode_date"]`` (published by DateSource); a missing season
    falls back to ``"all"`` then ``default_season``.

    ``s_E_price = baseprice / divisor`` (not clipped); the divisor is the largest band/default price
    magnitude across all schedules — a stable per-run scale, published as ``ctxt_E_price_max`` only
    when listed in ``ctxt_keys``. Hour of day = ``(effective_index * control_step / 3600) mod 24``.
    """

    _context_params: ClassVar[Set[str]] = {"control_step"}

    # Built-in default reproduces the legacy 3-level tariff: peak 40 ct/kWh (17:00-21:00), off-peak
    # 18 ct/kWh (00:00-06:00), and a 28 ct/kWh default for every other hour. Peak is listed first so
    # it wins on overlap (matching the previous "peak beats off-peak" rule).
    _DEFAULT_SCHEDULE: ClassVar[dict] = {
        "default_price": 28.0,
        "bands": (
            PriceBand("17:00", "21:00", 40.0),
            PriceBand("00:00", "06:00", 18.0),
        ),
    }

    def __init__(self, name: str,
                schedules: dict | None = None,
                default_season: str = "summer",
                ctxt_keys: list[str] | None = None,
                control_step: float = 300.0) -> None:
        super().__init__(name=name, control_step=control_step)

        raw = schedules or {"all": dict(self._DEFAULT_SCHEDULE)}
        # Canonical, round-trip-safe form (band dicts -> PriceBand); read directly by Serializable.to_dict.
        self.schedules = {season: self._normalise_schedule(block) for season, block in raw.items()}
        self.default_season = str(default_season)
        self.ctxt_keys = list(ctxt_keys) if ctxt_keys is not None else None

        # Fixed divisor: largest price magnitude across all schedules (stable per-run scale).
        self.price_divisor: float = max(1e-6, max(
            abs(price)
            for sch in self.schedules.values()
            for price in (sch["default_price"], *(band.price for band in sch["bands"]))
        ))

        self._active = self._select_schedule(self.default_season)
        # Raw baseprice (ct/kWh) of the current step — read for billing via get_raw_values.
        self.baseprice_raw: float = 0.0

    @staticmethod
    def _normalise_schedule(block: dict) -> Schedule:
        return {
            "default_price": float(block["default_price"]),
            "bands": [PriceBand.from_mapping(band) for band in block.get("bands", [])],
        }

    @staticmethod
    def _price_for_hour(schedule: Schedule, hour: float) -> float:
        """First band covering ``hour`` wins; hours outside every band fall back to default_price."""
        for band in schedule["bands"]:
            if band.contains(hour):
                return band.price
        return schedule["default_price"]

    def _select_schedule(self, season: str) -> Schedule:
        for key in (season, "all", self.default_season):
            if key in self.schedules:
                return self.schedules[key]
        return next(iter(self.schedules.values()))

    def _episode_season(self, info: dict) -> str:
        """Season from ``info['episode_date']`` (ISO ``YYYY-MM-DD``), else ``default_season``."""
        date = info.get("episode_date")
        parts = str(date).split("-") if date else []
        if len(parts) == 3 and parts[0].isdigit() and len(parts[0]) == 4:
            return season_for_month(int(parts[1]))
        return self.default_season

    def _hour_of_day(self, index: int) -> float:
        return (index * self.control_step / SECONDS_PER_HOUR) % 24.0

    def _normalise(self, raw: float) -> float:
        return float(raw / self.price_divisor)

    def setup_spaces(self, state_spaces, action_spaces) -> tuple:
        if "s_E_price" not in state_spaces:
            state_spaces["s_E_price"] = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self._publish_ctxt(state_spaces, "ctxt_E_price_max", Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32))
        return state_spaces, action_spaces

    def reset(self, states, info: dict) -> None:
        """Pick this episode's schedule from its season, then publish the first observation."""
        self._active = self._select_schedule(self._episode_season(info))
        self.update_state(states, info)

    def update_state(self, states, info: dict) -> None:
        self.baseprice_raw = self._price_for_hour(self._active, self._hour_of_day(self.effective_index))
        states["s_E_price"][0] = np.float32(self._normalise(self.baseprice_raw))
        self._write_ctxt(states, "ctxt_E_price_max", np.float32(self.price_divisor))

    # ----- Lookahead (analytic; no CSV) -----
    def lookahead_keys(self) -> tuple[str, ...]:
        return ("baseprice",)

    def lookahead(self, steps: list[int]) -> dict[str, list[float]]:
        return {"baseprice": [
            self._price_for_hour(self._active, self._hour_of_day(self.effective_index + s)) for s in steps
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
