import logging
from typing import ClassVar, Set

import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from ..base import StateSource
from ..csv_loader import CsvLoader
from ..forecastable import Forecastable
from adv_building_gym.components.registry import ComponentRegistry
from adv_building_gym._common.normalisation import Normalisation, normalise_with_scale_factor

logger = logging.getLogger(__name__)


class EnergyPriceDataSource(StateSource, Forecastable):
    """Data source for energy pricing information."""

    # price_max and dynamic_max_norm{,_ep} are derived from data, don't serialize
    _exclude_params: ClassVar[Set[str]] = {'price_max', 'baseprice_raw', 'dynamic_max_norm', 'dynamic_max_norm_ep'}

    # episode_length is supplied from env context (EnvConfig.EPISODE_LENGTH).
    _context_params: ClassVar[Set[str]] = {"episode_length"}

    _VALID_MAX_CALC_MODES: ClassVar[Set[str]] = {"mean", "median", "percentile", "mean_above_median"}

    def __init__(self, name: str, ds_path: str | None = None,
                normalise: Normalisation | str | None = Normalisation.ABS_MIN_MAX_SCALING,
                dynamic_max_price_calc: bool = False,
                max_calc_mode: str | None = None,
                percentile: float = 0.75,
                episode_length: int = 288) -> None:
        """Data source for energy pricing information.

        Args:
            name: Source identifier.
            ds_path: Optional CSV path; pushed later via DataCombinator.reload.
            normalise: Normalisation method for s_E_price (default ABS_MIN_MAX
                against the raw baseprice peak).
            dynamic_max_price_calc: If True, populate ctxt_E_price_dynamic_max
                with a data-driven price denominator in s_E_price's normalised
                frame. If False (default), the ctxt is fixed at 1.0 so any
                consumer dividing by it is a no-op.
            max_calc_mode: Selects the statistic used when
                dynamic_max_price_calc is True. One of:
                  mean              — mean of raw baseprice over the loaded series
                  median            — median of raw baseprice
                  percentile        — quantile of raw baseprice at `percentile`
                  mean_above_median — mean of raw baseprice values strictly above the median
            percentile: Quantile in (0, 1); consulted only when
                max_calc_mode == "percentile".
        """
        super().__init__(name=name)

        self.normalise = Normalisation.init(normalise)

        if dynamic_max_price_calc:
            if max_calc_mode not in self._VALID_MAX_CALC_MODES:
                raise ValueError(
                    f"max_calc_mode must be one of {sorted(self._VALID_MAX_CALC_MODES)} "
                    f"when dynamic_max_price_calc=True; got {max_calc_mode!r}."
                )
            if max_calc_mode == "percentile" and not (0.0 < percentile < 1.0):
                raise ValueError(
                    f"percentile must be in (0, 1) when max_calc_mode='percentile'; got {percentile}."
                )

        self.dynamic_max_price_calc = bool(dynamic_max_price_calc)
        self.max_calc_mode = max_calc_mode
        self.percentile = float(percentile)
        self.episode_length = int(episode_length)

        self.price_max: float = 1.0
        # Denominator (in s_E_price's normalised frame) used by downstream
        # rewards / policy ctxt. 1.0 means "no rescaling".
        # `dynamic_max_norm` is computed once per CSV (full series);
        # `dynamic_max_norm_ep` is recomputed each episode on the
        # [row_offset, row_offset + episode_length] window.
        self.dynamic_max_norm: float = 1.0
        self.dynamic_max_norm_ep: float = 1.0

        # Raw baseprice (ct/kWh) for the current step — updated by update_state.
        self.baseprice_raw: float = 0.0

        self.loader = CsvLoader(ds_path, on_reload=self._run_post_load)
        if ds_path is not None:
            logger.info("Use data file: %s", ds_path)

    def _post_load_data_processing(self) -> None:
        """Normalise the baseprice column and cache the raw maximum."""
        assert self.ts is not None, "self.ts was None. self.ts should be set by CsvLoader before _post_load_data_processing is called."

        if "baseprice" not in self.ts.columns:
            raise ValueError(
                f"EnergyPriceDataSource '{self.name}': CSV '{self.ds_path}' has no "
                "'baseprice' column."
            )

        self.ts["E_price_norm"], self.price_max = normalise_with_scale_factor(self.ts["baseprice"], self.normalise)
        self.dynamic_max_norm = self._compute_dynamic_max_norm(self.ts["baseprice"])

    def _compute_dynamic_max_norm(self, baseprice: pd.Series) -> float:
        """Return the price denominator in s_E_price's normalised frame.

        Returns 1.0 when the feature is disabled or the input window is
        empty, so the consuming reward's division is a no-op. When enabled,
        picks a statistic over the supplied raw baseprice slice and rescales
        by price_max so the result lives in the same frame as s_E_price.
        """
        if not self.dynamic_max_price_calc or baseprice.empty:
            return 1.0
        match self.max_calc_mode:
            case "mean":
                chosen_raw = float(baseprice.mean())
            case "median":
                chosen_raw = float(baseprice.median())
            case "percentile":
                chosen_raw = float(baseprice.quantile(self.percentile))
            case "mean_above_median":
                med = float(baseprice.median())
                above = baseprice[baseprice > med]
                chosen_raw = float(above.mean()) if not above.empty else med
            case _:
                # Already validated in __init__; defensive fallback.
                return 1.0
        if not self.price_max:
            return 1.0
        return max(chosen_raw / self.price_max, 1e-6)

    def setup_spaces(self,
                    state_spaces,
                    action_spaces) -> tuple:

        if "s_E_price" not in state_spaces.keys():
            state_spaces["s_E_price"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Running normalised min/max of s_E_price seen so far this episode.
        # Seeded at reset to the first step's price; expanded by update_state.
        if "s_E_price_min_norm" not in state_spaces.keys():
            state_spaces["s_E_price_min_norm"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        if "s_E_price_max_norm" not in state_spaces.keys():
            state_spaces["s_E_price_max_norm"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Raw maximum energy price (ct/kWh) — changes only when a new data
        # variant is loaded.  Allows the policy to reconstruct physical
        # price from the normalised E_price observation.
        if "ctxt_E_price_max" not in state_spaces.keys():
            state_spaces["ctxt_E_price_max"] = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        if "ctxt_E_price_dynamic_max" not in state_spaces.keys():
            state_spaces["ctxt_E_price_dynamic_max"] = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        # Same statistic as ctxt_E_price_dynamic_max, but computed over the
        # current episode's window [row_offset, row_offset + episode_length].
        # Constant within an episode, refreshed in reset().
        if "ctxt_E_price_dynamic_max_ep" not in state_spaces.keys():
            state_spaces["ctxt_E_price_dynamic_max_ep"] = Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        return state_spaces, action_spaces

    def update_state(self, states, info=None) -> None:
        if self.ts is None:
            raise RuntimeError(
                f"EnergyPriceDataSource '{self.name}': no CSV loaded. The DataCombinator "
                "must push an E_price variant before update_state is called."
            )
        row = self.ts.iloc[min(self.effective_index, len(self.ts) - 1)]
        energy_price = float(row["E_price_norm"])
        self.baseprice_raw = float(row["baseprice"])

        states["s_E_price"][0] = np.float32(energy_price)
        # Raw maximum price (ct/kWh) — constant within an episode, changes
        # only when a new data variant is loaded.
        states["ctxt_E_price_max"][0] = np.float32(self.price_max)
        # Data-driven price denominator in s_E_price's normalised frame
        # (1.0 when dynamic_max_price_calc is disabled).
        states["ctxt_E_price_dynamic_max"][0] = np.float32(self.dynamic_max_norm)
        # Same statistic, but restricted to the current episode's window;
        # refreshed in reset() since row_offset changes per episode.
        states["ctxt_E_price_dynamic_max_ep"][0] = np.float32(self.dynamic_max_norm_ep)

        prev_min = float(states["s_E_price_min_norm"][0])
        prev_max = float(states["s_E_price_max_norm"][0])
        states["s_E_price_min_norm"][0] = np.float32(min(prev_min, energy_price))
        states["s_E_price_max_norm"][0] = np.float32(max(prev_max, energy_price))

    def reset(self, states, info=None) -> None:
        # Seed running min/max to the first step's normalised price so that
        # update_state's min/max accumulation starts from a real value rather
        # than the zero-initialised state buffer.
        if self.ts is not None:
            row = self.ts.iloc[min(self.effective_index, len(self.ts) - 1)]
            first = np.float32(row["E_price_norm"])
            states["s_E_price_min_norm"][0] = first
            states["s_E_price_max_norm"][0] = first
            # Recompute the within-episode dynamic max for the active window.
            # row_offset is already set on self.sync by the env at this point.
            start = self.row_offset
            end = min(start + self.episode_length, len(self.ts))
            window = self.ts["baseprice"].iloc[start:end]
            self.dynamic_max_norm_ep = self._compute_dynamic_max_norm(window)
        self.update_state(states, info)

    def forecast_keys(self) -> tuple[str, ...]:
        return ("s_fc_E_price",)

    def forecast(self, selected_future_steps: list[int]) -> dict[str, list[float]]:
        if self.ts is None:
            return {"s_fc_E_price": [0.0] * len(selected_future_steps)}
        return {
            "s_fc_E_price": self._csv_forecast(self.ts, self.effective_index, "E_price_norm", selected_future_steps),
        }

    @property
    def E_price_max_raw(self) -> float:
        """Raw (unnormalised) maximum energy price."""
        return float(self.price_max)

    def get_raw_values(self) -> dict[str, float]:
        return {
            "raw_E_price": self.baseprice_raw
        }

    def _get_serialize_value(self, param_name: str, value):
        """Handle enum serialization for normalise parameter."""
        if param_name == 'normalise' and isinstance(value, Normalisation):
            return value.value
        return super()._get_serialize_value(param_name, value)


# Register EnergyPriceDataSource with the component registry
ComponentRegistry.register('statesource', EnergyPriceDataSource)
