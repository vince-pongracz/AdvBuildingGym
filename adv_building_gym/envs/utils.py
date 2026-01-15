"""Utility classes for AdvBuildingGym environments."""


class BuildingProps:
    """Building thermal properties for simulation.

    Args:
        mC: Thermal mass times heat capacity [J/K]. Default: 300.
        K: Thermal conductance [W/K]. Default: 20.
    """

    def __init__(self, mC: float = 300, K: float = 20):
        self.mC = mC
        self.K = K
