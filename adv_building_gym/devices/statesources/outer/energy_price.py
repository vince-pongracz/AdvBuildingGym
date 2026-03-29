import logging
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from ..base import StateSource
from adv_building_gym.utils.serializable import ComponentRegistry
from adv_building_gym.utils.normalisation import Normalisation, normalise_series

logger = logging.getLogger(__name__)


class EnergyPriceDataSource(StateSource):
    """Data source for energy pricing information."""

    # price_max is derived from data, don't serialize
    _exclude_params: ClassVar[Set[str]] = {'iteration', 'ts', 'price_max'}

    def __init__(self, name: str, ds_path: str | None = None,
                normalise: Normalisation | str | None = Normalisation.ABS_MIN_MAX_SCALING) -> None:
        super().__init__(name, ds_path)

        normalise = Normalisation.init(normalise)
        self.normalise = normalise

        if self.ts is not None:
            logger.info("Use data file: %s", ds_path)
            self._post_load_data_processing()
        else:
            self.price_max = 1.0

    def _post_load_data_processing(self) -> None:
        """Normalise the baseprice column and cache the raw maximum."""
        if self.ts is not None:
            self.price_max = float(self.ts["baseprice"].abs().max())
            self.ts["E_price_norm"] = normalise_series(self.ts["baseprice"], self.normalise)
        
    def setup_spaces(self,
                    state_spaces,
                    action_spaces) -> tuple:

        if "E_price" not in state_spaces.keys():
            state_spaces["E_price"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        if "E_price_max" not in state_spaces.keys():
            state_spaces["E_price_max"] = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        if "sim_hour" not in state_spaces.keys():
            state_spaces["sim_hour"] = Box(low=np.full((1,), 0, dtype=np.float32),
                                            high=np.full((1,), np.inf, dtype=np.float32),
                                            shape=(1,), dtype=np.float32)

        return state_spaces, action_spaces

    def update_state(self, states, info=None) -> None:
        if self.ts is not None:
            row = self.ts.iloc[min(self.effective_index, len(self.ts) - 1)]
            energy_price = float(row["E_price_norm"])
        else:
            current_sim_hour = states.get("sim_hour", np.zeros(shape=(1,), dtype=np.float32))[0]
            current_sim_hour = current_sim_hour % 24
            # Apply a simple time-of-use tariff if no CSV data is provided
            if current_sim_hour < 4:
                energy_price = 0.25
            elif current_sim_hour < 8:
                energy_price = 0.50
            else:
                energy_price = 0.75

        states["E_price"][0] = np.float32(energy_price)
        # E_price is already normalised, so the normalised max is 1.0
        states["E_price_max"][0] = np.float32(1.0)

    @property
    def E_price_max_raw(self) -> float:
        """Raw (unnormalised) maximum energy price."""
        return float(self.price_max)

    def _get_serialize_value(self, param_name: str, value):
        """Handle enum serialization for normalise parameter."""
        if param_name == 'normalise' and isinstance(value, Normalisation):
            return value.value
        return super()._get_serialize_value(param_name, value)


# Register EnergyPriceDataSource with the component registry
ComponentRegistry.register('statesource', EnergyPriceDataSource)
