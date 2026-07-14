import logging
from typing import Dict, Sequence

import numpy as np

from .battery_linear import BatteryLinear
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)

# Defensive cap for the rejection-sampling loop; unreachable in practice because
# __init__ validates that the exclusions never cover the whole 1-decimal grid.
_MAX_SAMPLE_ATTEMPTS = 1000


def _as_tenths(value: float) -> int:
    """Integer tenths of a 1-decimal value (17.3 → 173) — float-equality-safe comparisons."""
    return int(round(value * 10))


def _validate_range(param_name: str, value_range: Sequence[float]) -> list[float]:
    """Validate a [low, high] sampling range and return it as a float list."""
    if not isinstance(value_range, (list, tuple)) or len(value_range) != 2:
        raise ValueError(f"{param_name} must be a 2-item list [low, high], got {value_range!r}.")
    low, high = float(value_range[0]), float(value_range[1])
    if low < 0.0:
        raise ValueError(f"{param_name} bounds must be non-negative, got {value_range!r}.")
    if low > high:
        raise ValueError(f"{param_name} must satisfy low <= high, got {value_range!r}.")
    return [low, high]


def _validate_exclusions(param_name: str, value_range: list[float],
                        excluded_values: Sequence[float] | None) -> frozenset[int]:
    """Excluded values as a tenth-integer set; reject exclusions covering the whole range grid.

    Rounded draws live on the 1-decimal grid between round(low, 1) and
    round(high, 1) — if every grid point is excluded, sampling could never
    terminate, so fail fast here instead of looping at reset time.
    """
    excluded_tenths = frozenset(_as_tenths(float(v)) for v in (excluded_values or []))
    grid = set(range(_as_tenths(round(value_range[0], 1)), _as_tenths(round(value_range[1], 1)) + 1))
    if grid <= excluded_tenths:
        raise ValueError(
            f"{param_name} excludes every 1-decimal value in range {value_range} — nothing left to sample."
        )
    return excluded_tenths


def _draw_rounded_uniform_excluding(rng: np.random.Generator, low: float, high: float,
                                    excluded_tenths: frozenset[int]) -> float:
    """Uniform draw from [low, high] rounded to 1 decimal, redrawn while excluded."""
    for _ in range(_MAX_SAMPLE_ATTEMPTS):
        value = round(float(rng.uniform(low, high)), 1)
        if _as_tenths(value) not in excluded_tenths:
            return value
    raise RuntimeError(
        f"No acceptable sample from [{low}, {high}] after {_MAX_SAMPLE_ATTEMPTS} attempts "
        f"(excluded tenths: {sorted(excluded_tenths)})."
    )


class BatteryLinearWrapper(BatteryLinear):
    """BatteryLinear whose max power/capacity are re-sampled every episode.

    At each ``reset()``, ``max_power_kW`` / ``max_cap_kWh`` are drawn uniformly
    from ``max_power_range_kW`` / ``max_cap_range_kWh`` (env rng via
    ``info["_rng"]``, deterministic per worker), rounded to 1 decimal, and
    redrawn while the value appears in ``exc_max_power_kW`` /
    ``exc_max_cap_kWh`` (held-out sizes for eval generalisation).

    Obs/action spaces are identical to ``BatteryLinear`` (spaces never depend
    on the max values), so both classes are config-interchangeable — e.g.
    train with the wrapper, evaluate with fixed-size ``BatteryLinear``.
    """

    def __init__(self, name: str,
                max_power_range_kW: list[float],
                max_cap_range_kWh: list[float],
                control_step: int,
                start_soc_percentage: float,
                soc_min: float,
                soc_max: float,
                start_soc_jitter: float = 0.0,
                exc_max_power_kW: list[float] | None = None,
                exc_max_cap_kWh: list[float] | None = None,
                ctxt_keys: list[str] | None = None,
                ) -> None:
        """Initialize the sampling battery model.

        Args:
            name: Component identifier
            max_power_range_kW: [low, high] uniform sampling range for the
                per-episode max charge/discharge power in kW (low == high pins it)
            max_cap_range_kWh: [low, high] uniform sampling range for the
                per-episode battery capacity in kWh
            control_step: Timestep duration in seconds
            start_soc_percentage: Initial state of charge [0, 1]
            soc_min: Hardware minimum SoC (clipping floor)
            soc_max: Hardware maximum SoC (clipping ceiling)
            start_soc_jitter: Half-width of the uniform per-episode offset
                applied to the initial SoC (see BatteryLinear)
            exc_max_power_kW: 1-decimal power values never sampled (redrawn)
            exc_max_cap_kWh: 1-decimal capacity values never sampled (redrawn)
        """
        power_range = _validate_range("max_power_range_kW", max_power_range_kW)
        cap_range = _validate_range("max_cap_range_kWh", max_cap_range_kWh)

        # Placeholder mid-range values, alive only until the first reset():
        # every episode starts with reset(), which resamples before any step.
        super().__init__(
            name,
            max_power_kW=round((power_range[0] + power_range[1]) / 2.0, 1),
            max_cap_kWh=round((cap_range[0] + cap_range[1]) / 2.0, 1),
            control_step=control_step,
            start_soc_percentage=start_soc_percentage,
            soc_min=soc_min,
            soc_max=soc_max,
            start_soc_jitter=start_soc_jitter,
            ctxt_keys=ctxt_keys,
        )

        # Attribute names match the ctor params so Serializable.to_dict round-trips them.
        self.max_power_range_kW = power_range
        self.max_cap_range_kWh = cap_range
        self.exc_max_power_kW = [float(v) for v in (exc_max_power_kW or [])]
        self.exc_max_cap_kWh = [float(v) for v in (exc_max_cap_kWh or [])]
        self._excluded_power_tenths = _validate_exclusions("exc_max_power_kW", power_range, self.exc_max_power_kW)
        self._excluded_cap_tenths = _validate_exclusions("exc_max_cap_kWh", cap_range, self.exc_max_cap_kWh)

    def reset(self, states: Dict, info: dict) -> None:
        """Resample max power/capacity for the episode, then run the parent reset.

        Fixed draw order (power, capacity, then the parent's SoC jitter) keeps
        the rng stream deterministic per seed. The parent reset publishes the
        new values via update_state (ctxt keys + info power bounds).
        """
        rng = info.get("_rng") or np.random.default_rng()
        self.max_power_kW = _draw_rounded_uniform_excluding(
            rng, self.max_power_range_kW[0], self.max_power_range_kW[1], self._excluded_power_tenths)
        self.max_cap_kWh = _draw_rounded_uniform_excluding(
            rng, self.max_cap_range_kWh[0], self.max_cap_range_kWh[1], self._excluded_cap_tenths)
        logger.debug("%s: sampled max_power_kW=%.1f, max_cap_kWh=%.1f",
                    self.name, self.max_power_kW, self.max_cap_kWh)
        super().reset(states, info)


ComponentRegistry.register('infrastructure', BatteryLinearWrapper)
