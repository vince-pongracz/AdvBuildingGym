import logging
from collections import OrderedDict
from typing import ClassVar, Set

import numpy as np
from gymnasium.spaces import Box

from ..base import StateSource
from adv_building_gym.utils.serializable import ComponentRegistry
from adv_building_gym.utils.normalisation import Normalisation, normalise_series

logger = logging.getLogger(__name__)

# CSV column produced by preproc/hh_consumption/extract_hh_consumption.py
SOURCE_COLUMN: str = "hh_consumption_kW"

# Normalised column name added by _post_load_data_processing
NORM_COLUMN: str = "desired_energy_need_norm"


class DesiredUserEnergyNeed(StateSource):
    """Data source for desired user energy need information.

    When a CSV is provided (via ``ds_path``), reads the ``hh_consumption_kW``
    column produced by the WPuQ household consumption preprocessing pipeline,
    normalises it to [0, 1] using min-max scaling, and exposes it as the
    ``desired_energy_need`` observation.

    state key: desired_energy_need
    Definition desired_energy_need:
    - positive: user consumes E.
    - negative: no meaning
    - zero: user does not need E.
    """

    # consumption_max is derived from data, don't serialize
    _exclude_params: ClassVar[Set[str]] = {'iteration', 'ts', 'consumption_max'}

    def __init__(self, name: str, ds_path: str | None = None,
                 normalise: Normalisation | str | None = Normalisation.MIN_MAX_SCALING) -> None:
        super().__init__(name, ds_path)

        normalise = Normalisation.init(normalise)
        self.normalise = normalise

        if self.ts is not None:
            logger.info("Use data file: %s", ds_path)
            self._run_post_load()
        else:
            self.consumption_max = 1.0
            logger.debug("No initial data file for '%s', using synthetic energy need profile", name)

    def _post_load_data_processing(self) -> None:
        """Normalise the hh_consumption_kW column and cache the raw maximum."""
        if self.ts is not None:
            if SOURCE_COLUMN not in self.ts.columns:
                raise ValueError(
                    f"CSV must contain a '{SOURCE_COLUMN}' column. "
                    f"Found: {list(self.ts.columns)}"
                )
            self.consumption_max = float(self.ts[SOURCE_COLUMN].max())
            self.ts[NORM_COLUMN] = normalise_series(
                self.ts[SOURCE_COLUMN], self.normalise
            )

    def setup_spaces(self,
                    state_spaces: OrderedDict,
                    action_spaces: OrderedDict) -> tuple[OrderedDict, OrderedDict]:
        """Setup observation spaces for desired user energy need."""

        if "desired_energy_need" not in state_spaces.keys():
            state_spaces["desired_energy_need"] = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Raw maximum household consumption (kW) — changes only when a new
        # data variant is loaded.
        if "hh_consumption_max" not in state_spaces.keys():
            state_spaces["hh_consumption_max"] = Box(
                low=0, high=np.inf, shape=(1,), dtype=np.float32
            )

        if "sim_hour" not in state_spaces.keys():
            state_spaces["sim_hour"] = Box(low=np.full((1,), 0, dtype=np.float32),
                                            high=np.full((1,), np.inf, dtype=np.float32),
                                            shape=(1,),
                                            dtype=np.float32)

        return state_spaces, action_spaces

    def update_state(self, states, info=None) -> None:
        """Update desired energy need state based on current iteration."""
        if self.ts is not None:
            row = self.ts.iloc[min(self.effective_index, len(self.ts) - 1)]
            desired_energy = float(row[NORM_COLUMN])
        else:
            current_sim_hour = states.get("sim_hour", np.zeros(shape=(1,), dtype=np.float32))[0]
            current_sim_hour = current_sim_hour % 24
            # Synthetic time-based energy need profile when no CSV data is provided
            if current_sim_hour < 6:
                desired_energy = 0.2  # Low demand during night
            elif current_sim_hour < 9:
                desired_energy = 0.6  # Morning peak
            elif current_sim_hour < 17:
                desired_energy = 0.4  # Daytime moderate
            elif current_sim_hour < 21:
                desired_energy = 0.8  # Evening peak
            else:
                desired_energy = 0.3  # Evening low

        states["desired_energy_need"][0] = np.float32(desired_energy)
        # Raw maximum consumption (kW) — constant within an episode.
        states["hh_consumption_max"][0] = np.float32(self.consumption_max)

    @property
    def consumption_max_raw(self) -> float:
        """Raw (unnormalised) maximum consumption in kW."""
        return float(self.consumption_max)

    def _get_serialize_value(self, param_name: str, value):
        """Handle enum serialization for normalise parameter."""
        if param_name == 'normalise' and isinstance(value, Normalisation):
            return value.value
        return super()._get_serialize_value(param_name, value)


# Register DesiredUserEnergyNeed with the component registry
ComponentRegistry.register('statesource', DesiredUserEnergyNeed)
