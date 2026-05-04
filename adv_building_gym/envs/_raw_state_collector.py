"""Raw (unnormalised) physical-value collection extracted from AdvBuildingGym."""
from typing import Dict, Iterable

import numpy as np

from adv_building_gym.devices.infrastructure import Infrastructure
from adv_building_gym.devices.statesources import StateSource


class RawStateCollector:
    """Aggregates per-component raw values into the diagnostic ``info["raw"]`` dict.

    Most components publish their raw values via ``get_raw_values()``. The indoor
    temperature is the exception: ``s_temp_in_norm`` is mutated dynamically by HP
    and BuildingHeatLoss and is not exposed by any component, so this collector
    synthesises ``raw_temp_in`` by denormalising with ``ctxt_temp_abs_max``.
    """

    def collect(
        self,
        statesources: Iterable[StateSource],
        infras: Iterable[Infrastructure],
        state: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        raw: Dict[str, float] = {}
        for src in list(statesources) + list(infras):
            raw.update(src.get_raw_values())

        # ctxt_temp_abs_max is published into ``state`` by WeatherDataSource each
        # step — read from state rather than caching so the scale factor stays
        # correct across statesource hot-swaps.
        temp_abs_max = float(state.get("ctxt_temp_abs_max", np.ones(1, dtype=np.float32))[0])
        temp_in_norm = float(state.get("s_temp_in_norm", np.zeros(1, dtype=np.float32))[0])
        raw["raw_temp_in"] = temp_in_norm * temp_abs_max
        return raw