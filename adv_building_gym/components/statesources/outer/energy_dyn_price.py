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
from adv_building_gym._common.constants import SECONDS_PER_DAY

logger = logging.getLogger(__name__)


class EnergyPriceDynDataSource(StateSource, Forecastable, CsvLookahead, CsvReloadable):
    """Energy-price data source with a dynamic 1-day (next-24h) price divisor.

    The CSV is never rescaled and s_E_price is not clipped. Each step exposes
    ``s_E_price = raw_baseprice / divisor``. The divisor is the max of |raw baseprice| over the
    NEXT 24h window, recomputed each ``reset()`` (per episode / per day). One day =
    ``SECONDS_PER_DAY / control_step`` steps, capped at the episode length so the window never
    extends past the episode.

    For the year-based (whole-series blend) normalisation use ``EnergyPriceDataSource`` instead.

    The divisor (raw ct/kWh) is always published as ``ctxt_E_price_max`` because the price scale
    changes every episode, so the policy must condition on it (``raw = s_E_price * ctxt_E_price_max``).
    """

    # episode_length and timestep come from env context (EnvConfig.EPISODE_LENGTH / CONTROL_STEP).
    # The control step is injected as `timestep` (EnvConfig._statesource_context), so use that name.
    _context_params: ClassVar[Set[str]] = {"episode_length", "timestep"}

    # Lookahead channel -> CSV column (drives Lookahead.lookahead and forecast()).
    _lookahead_columns: ClassVar[dict[str, str]] = {"baseprice": "baseprice"}

    def __init__(self, name: str, ds_path: str | None = None,
                episode_length: int = 288,
                timestep: float = 300.0) -> None:
        """Args:
            name: Source identifier.
            ds_path: Optional CSV path; pushed later via DataCombinator.reload.
            episode_length: Episode length in steps (context-injected); caps the 24h window.
            timestep: Control step in seconds (context-injected); window = SECONDS_PER_DAY / timestep steps.
        """
        super().__init__(name=name)

        # The per-episode price scale changes every reset(), so the policy must always see it.
        self.emit_ctxt = True
        self.episode_length = int(episode_length)
        # One day in control steps; bounds the per-episode lookahead window.
        self.steps_per_day = max(1, round(SECONDS_PER_DAY / timestep))

        # Raw-unit divisors for s_E_price. `series_divisor` is the full-series abs max (set at
        # load, pre-reset baseline); `price_divisor` is the active next-24h value (set at reset).
        self.series_divisor: float = 1.0
        self.price_divisor: float = 1.0

        # Raw baseprice (ct/kWh) for the current step — updated by update_state.
        self.baseprice_raw: float = 0.0

        self.loader = CsvLoader(ds_path, on_reload=self._run_post_load)
        if ds_path is not None:
            logger.info("Use data file: %s", ds_path)

    # TODO noprio VP 2026.06.22.: rolling window for the max and for the normalisation?
    def _post_load_data_processing(self) -> None:
        """Validate the baseprice column and cache the full-series abs max.

        reset() overrides the active divisor with the next-24h value; this full-series value is
        only the pre-reset baseline.
        """
        assert self.ts is not None, "self.ts must be set by CsvLoader before _post_load_data_processing."

        if "baseprice" not in self.ts.columns:
            raise ValueError(f"EnergyPriceDynDataSource '{self.name}': CSV '{self.ds_path}' has no 'baseprice' column.")

        self.series_divisor = self._compute_divisor(self.ts["baseprice"])
        self.price_divisor = self.series_divisor

        # update_state / reset / forecast only ever read 'baseprice'; drop the rest.
        self._keep_ts_columns({"baseprice"})

    def _compute_divisor(self, baseprice: pd.Series) -> float:
        """Positive divisor = magnitude of the max baseprice over the slice (1.0 when empty).

        The magnitude keeps the divisor positive even for negative-price data.
        """
        if baseprice.empty:
            return 1.0
        return max(abs(float(baseprice.abs().max())), 1e-6)

    def _normalise(self, raw: float) -> float:
        """Scale a raw baseprice by the active divisor (not clipped)."""
        return float(raw / self.price_divisor)

    def setup_spaces(self, state_spaces, action_spaces) -> tuple:
        if "s_E_price" not in state_spaces:
            state_spaces["s_E_price"] = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        # Price divisor (raw ct/kWh) — always published (per-episode scale). raw = s_E_price * this.
        self._publish_ctxt(state_spaces, "ctxt_E_price_max", Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32))

        return state_spaces, action_spaces

    def update_state(self, states, info: dict) -> None:
        if self.ts is None:
            raise RuntimeError(
                f"EnergyPriceDynDataSource '{self.name}': no CSV loaded. The DataCombinator "
                "must push an E_price variant before update_state is called."
            )
        idx = min(self.effective_index, len(self.ts) - 1)
        self.baseprice_raw = float(self.ts["baseprice"].iloc[idx])

        states["s_E_price"][0] = np.float32(self._normalise(self.baseprice_raw))
        self._write_ctxt(states, "ctxt_E_price_max", np.float32(self.price_divisor))

    def reset(self, states, info: dict) -> None:
        if self.ts is not None:
            # Windowed: |baseprice| max over the next 24h, capped at the episode length so we
            # never normalise against prices outside the episode.
            start = self.row_offset
            end = min(start + min(self.steps_per_day, self.episode_length), len(self.ts))
            self.price_divisor = self._compute_divisor(self.ts["baseprice"].iloc[start:end])
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
ComponentRegistry.register('statesource', EnergyPriceDynDataSource)
