import logging
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from ..base import StateSource
from adv_building_gym.utils.serializable import ComponentRegistry
from adv_building_gym.utils.normalisation import Normalisation, normalise_with_scale_factor

logger = logging.getLogger(__name__)


class EnergyPriceDataSource(StateSource):
    """Data source for energy pricing information."""

    # price_max is derived from data, don't serialize
    _exclude_params: ClassVar[Set[str]] = {'iteration', 'ts', 'price_max', 'baseprice_raw'}

    def __init__(self, name: str, ds_path: str | None = None,
                normalise: Normalisation | str | None = Normalisation.ABS_MIN_MAX_SCALING) -> None:
        super().__init__(name, ds_path)

        self.normalise = Normalisation.init(normalise)

        self.price_max: float = 1.0
        # Raw baseprice (ct/kWh) for the current step — updated by update_state.
        self.baseprice_raw: float = 0.0
        if self.ts is not None:
            logger.info("Use data file: %s", ds_path)
            self._run_post_load()

    def _post_load_data_processing(self) -> None:
        """Normalise the baseprice column and cache the raw maximum."""
        if "baseprice" not in self.ts.columns:
            raise ValueError(
                f"EnergyPriceDataSource '{self.name}': CSV '{self.ds_path}' has no "
                "'baseprice' column."
            )

        self.ts["E_price_norm"], self.price_max = normalise_with_scale_factor(self.ts["baseprice"], self.normalise)
        
    def setup_spaces(self,
                    state_spaces,
                    action_spaces) -> tuple:

        if "s_E_price" not in state_spaces.keys():
            state_spaces["s_E_price"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Raw maximum energy price (ct/kWh) — changes only when a new data
        # variant is loaded.  Allows the policy to reconstruct physical
        # price from the normalised E_price observation.
        if "ctxt_E_price_max" not in state_spaces.keys():
            state_spaces["ctxt_E_price_max"] = Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        if "raw_sim_hour" not in state_spaces.keys():
            state_spaces["raw_sim_hour"] = Box(low=np.full((1,), 0, dtype=np.float32),
                                            high=np.full((1,), np.inf, dtype=np.float32),
                                            shape=(1,), dtype=np.float32)
            
        # TODO VP 2026.05.02. : Idea -- within a day, store the normalised daily min and max prices.
        # so it's within the state space how high is the current price -- compared to the known min and max.

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

    @property
    def E_price_max_raw(self) -> float:
        """Raw (unnormalised) maximum energy price."""
        return float(self.price_max)

    def get_raw_values(self) -> dict[str, float]:
        return {"raw_E_price": self.baseprice_raw}

    def _get_serialize_value(self, param_name: str, value):
        """Handle enum serialization for normalise parameter."""
        if param_name == 'normalise' and isinstance(value, Normalisation):
            return value.value
        return super()._get_serialize_value(param_name, value)


# Register EnergyPriceDataSource with the component registry
ComponentRegistry.register('statesource', EnergyPriceDataSource)
