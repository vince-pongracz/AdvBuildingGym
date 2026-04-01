"""Centralized RNG service for reproducible experiments.

RngService is a singleton that provides per-caller deterministic
random numbers derived from a single base seed.  Any component that
needs randomness calls ``RngService.get().random(caller_id)`` to
obtain the next number in its stream — no need to thread seed
parameters through constructors.
"""

import logging
from typing import ClassVar

import numpy as np

logger = logging.getLogger(__name__)


class RngService:
    """Singleton RNG registry.

    Each unique ``caller_id`` gets its own deterministic seed chain
    derived from the global ``base_seed``.

    Usage::

        # At script start (optional — uses OS entropy if omitted):
        RngService.initialize(training_config.seed)

        # In any component that needs a random number:
        value = RngService.get().random(self.name)
    """

    _instance: ClassVar["RngService | None"] = None

    def __init__(self, base_seed: int | None = None) -> None:
        if base_seed is None:
            base_seed = int(np.random.SeedSequence().entropy)
        self.base_seed = base_seed
        self._seed_seq = np.random.SeedSequence(base_seed)
        self._registry: dict[str, int] = {}
        logger.info("RngService initialized with base_seed=%d", base_seed)

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def initialize(cls, base_seed: int | None = None) -> "RngService":
        """Create the singleton with *base_seed*. Can only be called once."""
        if cls._instance is not None:
            raise RuntimeError("RngService is already initialized")
        cls._instance = cls(base_seed)
        return cls._instance

    @classmethod
    def get(cls) -> "RngService":
        """Return the singleton, auto-initializing with DEFAULT_SEED if needed."""
        if cls._instance is None:
            cls.initialize()
        return cls._instance  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Random number generation
    # ------------------------------------------------------------------

    def get_random(self, caller_id: str) -> int:
        """Return the next deterministic random int for *caller_id*.

        First call for a given *caller_id* uses ``base_seed`` as the
        initial seed.  Each call creates an RNG from the stored seed,
        generates a number, stores it as the seed for the next call,
        and returns it.
        """
        if caller_id not in self._registry:
            logger.info("Registered caller: %s", caller_id)
            self._registry[caller_id] = self.base_seed

        current_seed = self._registry[caller_id]
        rng = np.random.default_rng(current_seed)
        next_seed = int(rng.integers(0, 2**31))
        self._registry[caller_id] = next_seed
        return next_seed
