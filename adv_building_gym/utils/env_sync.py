

# TODO use of PyDispatcher?

from abc import ABC

class EnvSyncInterface(ABC):
    """Interface for synchronizing devices in the environment."""
    def __init__(self):
        self.iteration = 0

    def synchronize(self, iteration: int) -> None:
        self.iteration = iteration
