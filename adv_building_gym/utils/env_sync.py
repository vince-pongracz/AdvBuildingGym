

# NOTE VP 2026.01.14. : use of PyDispatcher? -- 
# only when more components needs to be synced or iteration and 
# .syncronise calls are not liked anymore.

from abc import ABC

class EnvSyncInterface(ABC):
    """Interface for synchronizing devices in the environment."""
    def __init__(self):
        self.iteration = 0

    def synchronise(self, iteration: int) -> None:
        self.iteration = iteration
