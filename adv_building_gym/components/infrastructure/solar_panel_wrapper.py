"""SolarPanelWrapper — SolarPanel whose peak power is re-sampled every episode."""

import logging
from typing import Dict

import numpy as np

from .solar_panel import SolarPanel
from ._sampling import _draw_rounded_uniform_excluding, _validate_exclusions, _validate_range
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class SolarPanelWrapper(SolarPanel):
    """SolarPanel whose ``max_power_kW`` is re-sampled every episode.

    At each ``reset()``, ``max_power_kW`` is drawn uniformly from
    ``max_power_range_kW`` (env rng via ``info["_rng"]``, deterministic per
    worker), rounded to 1 decimal, and redrawn while the value appears in
    ``exc_max_power_kW`` (held-out sizes for eval generalisation).

    ``panel_area_m2`` is never a config input here: it is re-derived from each
    sampled rating via ``SolarPanel._derive_area_m2`` (headroom-factor rule), so
    the array stays STC-consistent — the formula peak ``A * η`` sits just below
    the rating and the clip in ``_power_from_irradiance`` only engages
    above-STC irradiance.

    Obs/action spaces are identical to ``SolarPanel`` (spaces never depend on
    the max values), so both classes are config-interchangeable — e.g. train
    with the wrapper, evaluate with fixed-size ``SolarPanel``.
    """

    def __init__(self, name: str,
                max_power_range_kW: list[float],
                control_step: int,
                pv_efficiency: float,
                exc_max_power_kW: list[float] | None = None,
                area_headroom_factor: float = 0.975,
                ctxt_keys: list[str] | None = None,
                ) -> None:
        """Initialize the sampling solar panel model.

        Args:
            name: Component identifier
            max_power_range_kW: [low, high] uniform sampling range for the
                per-episode peak power (STC rating) in kW (low == high pins it)
            control_step: Timestep duration in seconds
            pv_efficiency: Module efficiency η used by the irradiance→power formula
            exc_max_power_kW: 1-decimal rating values never sampled (redrawn)
            area_headroom_factor: Formula-peak fraction of the rating the derived
                panel area targets (default 0.975 → ≈ −2.5 % headroom below the clip)
        """
        power_range = _validate_range("max_power_range_kW", max_power_range_kW)

        # Placeholder mid-range rating, alive only until the first reset():
        # every episode starts with reset(), which resamples before any step.
        # panel_area_m2 stays omitted, so the parent derives it (and validates
        # pv_efficiency / area_headroom_factor).
        super().__init__(
            name,
            max_power_kW=round((power_range[0] + power_range[1]) / 2.0, 1),
            control_step=control_step,
            pv_efficiency=pv_efficiency,
            area_headroom_factor=area_headroom_factor,
            ctxt_keys=ctxt_keys,
        )

        # Attribute names match the ctor params so Serializable.to_dict round-trips them.
        self.max_power_range_kW = power_range
        self.exc_max_power_kW = [float(v) for v in (exc_max_power_kW or [])]
        self._excluded_powers = _validate_exclusions("exc_max_power_kW", power_range, self.exc_max_power_kW)

    def reset(self, states: Dict, info: dict) -> None:
        """Resample the rating for the episode, re-derive the area, then run the parent reset.

        The parent reset publishes the new values via update_state
        (``ctxt_solar_max_power_kW`` / ``ctxt_pv_area_m2`` + info power bounds).
        """
        rng = info.get("_rng") or np.random.default_rng()
        self.max_power_kW = _draw_rounded_uniform_excluding(
            rng, self.max_power_range_kW[0], self.max_power_range_kW[1], self._excluded_powers)
        self.panel_area_m2 = self._derive_area_m2(self.max_power_kW)
        logger.debug("%s: sampled max_power_kW=%.1f (panel_area_m2=%.2f)",
                    self.name, self.max_power_kW, self.panel_area_m2)
        super().reset(states, info)


ComponentRegistry.register('infrastructure', SolarPanelWrapper)
