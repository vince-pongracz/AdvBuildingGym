"""Utility classes for AdvBuildingGym environments."""


# NOTE VP 2026.01.20. : These params are not constant for real for the building's lifetime...
# Link: https://www.researchgate.net/publication/301310479_Particle_Swarm_Optimization_and_Kalman_Filtering_for_Demand_Prediction_of_Commercial_Buildings

class BuildingProps:
    """Building thermal properties for simulation.

    Args:
        mC: Thermal mass times heat capacity [J/K]. Default: 300.
        K: Thermal conductance [W/K]. Default: 20.
    """

    def __init__(self, mC: float = 300, K: float = 20):
        self.mC = mC
        self.K = K
