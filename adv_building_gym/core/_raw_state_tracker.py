"""Raw (unnormalised) physical-value collection extracted from AdvBuildingGym."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable

if TYPE_CHECKING:
    # Annotation-only: core must not import components at runtime.
    from adv_building_gym.components.infrastructure import Infrastructure
    from adv_building_gym.components.statesources import StateSource


class RawStateTracker:
    """Aggregates per-component ``get_raw_values()`` into the diagnostic ``info["raw"]`` dict.

    Infras collected before statesources, so on key collisions the statesource wins
    (mirrors update_state order). E.g. HP and BuildingHeatLoss both publish ``raw_temp_in``;
    BuildingHeatLoss runs last → final post-heat-loss temperature.
    """

    def collect(
        self,
        statesources: Iterable[StateSource],
        infras: Iterable[Infrastructure],
    ) -> Dict[str, float]:
        raw_values: Dict[str, float] = {}
        for ifs in infras:
            raw_values.update(ifs.get_raw_values())
        for src in statesources:
            raw_values.update(src.get_raw_values())
        return raw_values
