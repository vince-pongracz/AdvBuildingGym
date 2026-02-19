"""Training hyperparameter configuration.

Provides a dataclass that holds algorithm-agnostic training hyperparameters
(learning rate, batch size, etc.) and can be loaded from a JSON file.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    """Algorithm-agnostic training hyperparameters.

    Attributes:
        learning_rate: Learning rate for optimiser(s).
        batch_size: Mini-batch size for gradient updates.
    """

    learning_rate: float = 3e-4
    batch_size: int = 64

    @staticmethod
    def from_json(path: str | Path) -> "TrainingConfig":
        """Load training config from a JSON file.

        Args:
            path: Path to the JSON config file.

        Returns:
            TrainingConfig populated from the file.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return TrainingConfig(**data)
