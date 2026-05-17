

# NOTE VP 2026.01.14. : use of PyDispatcher? -- 
# only when more components needs to be synced or iteration and 
# .syncronise calls are not liked anymore.

from abc import ABC

class EnvSync(ABC):
    """Interface for synchronizing devices in the environment."""
    def __init__(self):
        super().__init__()
        self.iteration = 0
        self.row_offset = 0

    def synchronise(self, iteration: int, row_offset: int | None = None) -> None:
        self.iteration = iteration
        
        if row_offset is not None:
            self.row_offset = row_offset

    @property
    def effective_index(self) -> int:
        """Row index into time-series data: row_offset + iteration."""
        return self.row_offset + self.iteration
