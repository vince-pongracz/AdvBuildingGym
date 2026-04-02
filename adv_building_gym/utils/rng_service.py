"""Centralized RNG service for reproducible experiments.

RngService provides per-caller deterministic random numbers derived from
a single base seed. Any component that needs randomness, calls
``RngService.get().get_random(caller_id)`` to obtain the next number in
its stream — no need to thread seed parameters through constructors.

When Ray is running, the RNG state lives in a **Ray Named Actor** on
the head node so that all workers (EnvRunners, Learners) share one true
instance.  When Ray is not available (evaluation, tests) it falls back
to a local in-process singleton.
"""

import logging
from typing import ClassVar

import numpy as np

logger = logging.getLogger(__name__)

_RAY_ACTOR_NAME = "rng_service"


# ======================================================================
# Core RNG logic (single source of truth for the algorithm)
# ======================================================================

class _RngEngine:
    """Deterministic per-caller RNG registry.

    Each unique ``caller_id`` gets its own seed chain derived from
    ``base_seed``.  Used by both the Ray Actor and the local fallback.
    """

    def __init__(self, base_seed: int) -> None:
        self.base_seed = base_seed
        self._registry: dict[str, int] = {}

    def get_random(self, caller_id: str) -> int:
        if caller_id not in self._registry:
            logger.info("Registered caller: %s", caller_id)
            self._registry[caller_id] = self.base_seed

        current_seed = self._registry[caller_id]
        rng = np.random.default_rng(current_seed)
        next_seed = int(rng.integers(0, 2**31))
        self._registry[caller_id] = next_seed
        return next_seed


# ======================================================================
# Ray Actor — deployed on the head node
# ======================================================================

def _make_actor_cls():
    """Lazily build the ``@ray.remote`` actor class.

    Importing ``ray`` at module level would force every process that
    imports ``rng_service`` to depend on Ray.  Building the class inside
    a function defers the import to the moment it's actually needed.
    """
    import ray  # noqa: E402 — deferred import

    @ray.remote(num_cpus=0)
    class RngServiceActor:
        """Ray actor that owns the RNG registry.

        Deployed once on the head node via
        ``RngService.initialize(seed)``.  All workers discover it by
        name and call ``get_random`` remotely.
        """

        def __init__(self, base_seed: int) -> None:
            self._engine = _RngEngine(base_seed)
            logger.info("RngServiceActor initialized with base_seed=%d", base_seed)

        def get_random(self, caller_id: str) -> int:
            return self._engine.get_random(caller_id)

    return RngServiceActor


# ======================================================================
# Local fallback — used when Ray is not running
# ======================================================================

class _LocalRngService:
    """In-process RNG service (no Ray dependency)."""

    def __init__(self, base_seed: int) -> None:
        self._engine = _RngEngine(base_seed)
        logger.info("RngService (local) initialized with base_seed=%d", base_seed)

    def get_random(self, caller_id: str) -> int:
        return self._engine.get_random(caller_id)


# ======================================================================
# Proxy — transparently delegates to actor or local instance
# ======================================================================

class _ActorProxy:
    """Thin wrapper that makes remote actor calls look synchronous."""

    def __init__(self, handle) -> None:
        self._handle = handle

    def get_random(self, caller_id: str) -> int:
        import ray  # noqa: E402
        return ray.get(self._handle.get_random.remote(caller_id))


# ======================================================================
# Public API
# ======================================================================

class RngService:
    """Singleton RNG registry — Ray Actor when distributed, local otherwise.

    Usage::

        # In the driver, **after** ``ray.init()``::
        RngService.initialize(training_config.seed)

        # In any component (driver or worker)::
        value = RngService.get().get_random(self.name)

    When Ray is running, ``initialize()`` creates a Named Actor on the
    head node.  ``get()`` discovers it by name.  When Ray is *not*
    running, both methods fall back to a plain in-process singleton.
    """

    _instance: ClassVar["_ActorProxy | _LocalRngService | None"] = None

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def initialize(cls, base_seed: int | None = None) -> "_ActorProxy | _LocalRngService":
        """Create the singleton with *base_seed*.

        * If Ray is running → deploys a Named Actor on the head node.
        * Otherwise → creates a local ``_LocalRngService``.

        Must be called exactly once (raises on repeat calls).
        """
        if cls._instance is not None:
            raise RuntimeError("RngService is already initialized")

        if base_seed is None:
            base_seed = int(np.random.SeedSequence().entropy)

        if _ray_is_running():
            ActorCls = _make_actor_cls()
            # Named Actor lives as long as the driver job (default lifetime).
            # Link: https://docs.ray.io/en/latest/ray-core/actors/named-actors.html
            handle = ActorCls.options(name=_RAY_ACTOR_NAME).remote(base_seed)
            cls._instance = _ActorProxy(handle)
            logger.info("RngService: created Ray Named Actor '%s'", _RAY_ACTOR_NAME)
        else:
            cls._instance = _LocalRngService(base_seed)

        return cls._instance

    @classmethod
    def get(cls) -> "_ActorProxy | _LocalRngService":
        """Return the singleton proxy / local instance.

        On workers the Named Actor is discovered automatically the
        first time ``get()`` is called (no manual ``initialize()``
        needed).  Falls back to a local instance when Ray is not running.
        """
        if cls._instance is not None:
            return cls._instance

        # On a Ray worker: look up the existing Named Actor.
        if _ray_is_running():
            import ray  # noqa: E402
            try:
                handle = ray.get_actor(_RAY_ACTOR_NAME)
                cls._instance = _ActorProxy(handle)
                logger.debug(
                    "RngService: discovered Ray Named Actor '%s' on worker",
                    _RAY_ACTOR_NAME,
                )
                return cls._instance
            except ValueError:
                # Actor doesn't exist yet — fall through to local init.
                logger.warning(
                    "RngService: Named Actor '%s' not found; "
                    "falling back to local instance (this breaks the "
                    "single-instance guarantee)",
                    _RAY_ACTOR_NAME,
                )

        # Fallback: local instance with OS entropy.
        cls._instance = _LocalRngService(int(np.random.SeedSequence().entropy))
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Tear down the singleton (useful in tests)."""
        cls._instance = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _ray_is_running() -> bool:
    """Return True if Ray has been initialized in this process."""
    try:
        import ray  # noqa: E402
        return ray.is_initialized()
    except ImportError:
        return False
