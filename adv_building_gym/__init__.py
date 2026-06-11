"""AdvBuildingGym — Gymnasium-compatible building energy control environment.

Top-level package marker; subpackages are imported explicitly (no re-exports).

Layering: 
- ``core/`` (env, wrappers)
- ``components/`` (infra/statesources/rewards + registry)
- ``config/`` (YAML loaders + dataclasses)
- ``ray/`` (RLlib adapter)
- ``sb/`` (Stable-Baselines3)
- ``controllers/`` (Pyomo/scipy baselines)
- ``_common/`` (cross-layer utils)
"""

__version__ = "0.1.0"
