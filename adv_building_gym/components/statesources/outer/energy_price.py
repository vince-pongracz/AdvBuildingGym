import logging
from typing import ClassVar, Set

import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from ..base import StateSource
from ..csv_loader import CsvLoader
from ..forecastable import Forecastable
from ..csv_lookahead import CsvLookahead
from ..reloadable import CsvReloadable
from adv_building_gym.components.registry import ComponentRegistry

logger = logging.getLogger(__name__)


class EnergyPriceDataSource(StateSource, Forecastable, CsvLookahead, CsvReloadable):
    """Energy-price data source with a year-based price divisor.

    The CSV is never rescaled and s_E_price is not clipped. Each step exposes
    ``s_E_price = raw_baseprice / divisor``. The divisor is a blend of the ``max_calc_mode``
    statistic over the whole series (weight 0.3) and over the per-episode window (0.7), refreshed
    each ``reset()`` so the scale tracks the episode's local price level while staying anchored to
    the full-series level.

    For the 1-day windowed (next-24h max) normalisation use ``EnergyPriceDynDataSource`` instead.

    The divisor (raw ct/kWh) is published as ``ctxt_E_price_max`` only when ``emit_ctxt`` is set,
    so the policy can condition on the price scale (``raw = s_E_price * ctxt_E_price_max``).
    """

    # episode_length comes from env context (EnvConfig.EPISODE_LENGTH); it sizes the per-episode
    # window used by the blend.
    _context_params: ClassVar[Set[str]] = {"episode_length"}

    _VALID_MAX_CALC_MODES: ClassVar[Set[str]] = {"mean", "median", "percentile", "mean_above_median"}

    # Blend weights (full series vs. per-episode window) restoring the earlier
    # _dynamic_price_divisor behaviour so the divisor tracks the episode's local price level.
    _SERIES_BLEND_WEIGHT: ClassVar[float] = 0.3
    _EPISODE_BLEND_WEIGHT: ClassVar[float] = 0.7

    # Lookahead channel -> CSV column (drives Lookahead.lookahead and forecast()).
    _lookahead_columns: ClassVar[dict[str, str]] = {"baseprice": "baseprice"}

    def __init__(self, name: str, ds_path: str | None = None,
                max_calc_mode: str = "mean",
                percentile: float = 0.75,
                emit_ctxt: bool = False,
                episode_length: int = 288) -> None:
        """Args:
            name: Source identifier.
            ds_path: Optional CSV path; pushed later via DataCombinator.reload.
            max_calc_mode: Statistic blended into the price divisor (magnitude taken, so it stays
                positive even for negative-price data). One of:
                  mean / median     — over the whole series and the per-episode window
                  percentile        — `percentile` quantile
                  mean_above_median — mean of values above the median
            percentile: Quantile in (0, 1); used only when max_calc_mode == "percentile".
            emit_ctxt: Also publish ctxt_E_price_max (the active divisor) for policy conditioning.
            episode_length: Episode length in steps (context-injected); sizes the blend window.
        """
        super().__init__(name=name)

        if max_calc_mode not in self._VALID_MAX_CALC_MODES:
            raise ValueError(f"max_calc_mode must be one of {sorted(self._VALID_MAX_CALC_MODES)}; got {max_calc_mode!r}.")
        if max_calc_mode == "percentile" and not (0.0 < percentile < 1.0):
            raise ValueError(f"percentile must be in (0, 1) when max_calc_mode='percentile'; got {percentile}.")

        self.max_calc_mode = max_calc_mode
        self.emit_ctxt = bool(emit_ctxt)
        self.percentile = float(percentile)
        self.episode_length: int = episode_length

        # Raw-unit divisors for s_E_price. `series_divisor` is the full-series statistic (set at
        # load); `price_divisor` is the active blended value (set per episode at reset).
        self.series_divisor: float = 1.0
        self.price_divisor: float = 1.0

        # Raw baseprice (ct/kWh) for the current step — updated by update_state.
        self.baseprice_raw: float = 0.0

        self.loader = CsvLoader(ds_path, on_reload=self._run_post_load)
        if ds_path is not None:
            logger.info("Use data file: %s", ds_path)

    def _post_load_data_processing(self) -> None:
        """Validate the baseprice column and cache the full-series divisor.

        reset() blends this with the per-episode-window statistic; it is also the pre-reset baseline.
        """
        assert self.ts is not None, "self.ts must be set by CsvLoader before _post_load_data_processing."

        if "baseprice" not in self.ts.columns:
            raise ValueError(f"EnergyPriceDataSource '{self.name}': CSV '{self.ds_path}' has no 'baseprice' column.")

        self.series_divisor = self._compute_divisor(self.ts["baseprice"])
        self.price_divisor = self.series_divisor

        # update_state / reset / forecast only ever read 'baseprice'; drop the rest.
        self._keep_ts_columns({"baseprice"})

    def _compute_divisor(self, baseprice: pd.Series) -> float:
        """Positive divisor for the baseprice slice (1.0 when empty).

        Takes the magnitude of the max_calc_mode statistic, so it stays positive for
        negative-price data.
        """
        if baseprice.empty:
            return 1.0
        match self.max_calc_mode:
            case "mean":
                chosen = float(baseprice.mean())
            case "median":
                chosen = float(baseprice.median())
            case "percentile":
                chosen = float(baseprice.quantile(self.percentile))
            case "mean_above_median":
                med = float(baseprice.median())
                above = baseprice[baseprice > med]
                chosen = float(above.mean()) if not above.empty else med
            case _:
                # Already validated in __init__; defensive fallback.
                return 1.0
        return max(abs(chosen), 1e-6)

    def _normalise(self, raw: float) -> float:
        """Scale a raw baseprice by the active divisor (not clipped)."""
        return float(raw / self.price_divisor)

    def setup_spaces(self, state_spaces, action_spaces) -> tuple:
        if "s_E_price" not in state_spaces:
            state_spaces["s_E_price"] = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        # Price divisor (raw ct/kWh) — policy-only conditioning, gated by emit_ctxt. raw = s_E_price * this.
        self._publish_ctxt(state_spaces, "ctxt_E_price_max", Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32))

        return state_spaces, action_spaces

    def update_state(self, states, info=None) -> None:
        if self.ts is None:
            raise RuntimeError(
                f"EnergyPriceDataSource '{self.name}': no CSV loaded. The DataCombinator "
                "must push an E_price variant before update_state is called."
            )
        idx = min(self.effective_index, len(self.ts) - 1)
        self.baseprice_raw = float(self.ts["baseprice"].iloc[idx])

        states["s_E_price"][0] = np.float32(self._normalise(self.baseprice_raw))
        # No-op unless ctxt_E_price_max was published (emit_ctxt).
        self._write_ctxt(states, "ctxt_E_price_max", np.float32(self.price_divisor))

    def reset(self, states, info=None) -> None:
        if self.ts is not None:
            # Blend the full-series statistic with the per-episode-window one
            # (restores the earlier _dynamic_price_divisor behaviour).
            start = self.row_offset
            end = min(start + self.episode_length, len(self.ts))
            episode_divisor = self._compute_divisor(self.ts["baseprice"].iloc[start:end])
            self.price_divisor = max(
                self._SERIES_BLEND_WEIGHT * self.series_divisor + self._EPISODE_BLEND_WEIGHT * episode_divisor,
                1e-6,
            )
        self.update_state(states, info)

    def forecast_keys(self) -> tuple[str, ...]:
        return ("s_fc_E_price",)

    def forecast(self, selected_future_steps: list[int]) -> dict[str, list[float]]:
        # s_fc_E_price = normalised baseprice lookahead (same divisor as the live s_E_price).
        raw_fc = self.lookahead(selected_future_steps)["baseprice"]
        return {"s_fc_E_price": [self._normalise(v) for v in raw_fc]}

    def get_raw_values(self) -> dict[str, float]:
        return {
            "raw_E_price": self.baseprice_raw
        }

# register with ComponentRegistry
ComponentRegistry.register('statesource', EnergyPriceDataSource)
