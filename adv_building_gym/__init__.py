"""AdvBuildingGym — Gymnasium-compatible building energy control environment.

Top-level package marker. Subpackages must be imported explicitly:

    from adv_building_gym.config.trial_config import TrialConfig
    from adv_building_gym.core.env import AdvBuildingGym
    from adv_building_gym.ray.evaluation import evaluate_model

Layering (each layer's external deps obvious from its name):
- ``core/``        — gymnasium + numpy + pandas (env, wrappers)
- ``components/``  — env plugins (infrastructure, statesources, rewards) + registry
- ``config/``      — YAML loaders + dataclasses + combinators
- ``ray/``         — Ray RLlib adapter (env_creator, ma_env, training, callbacks, evaluation, utils)
- ``sb/``          — Stable-Baselines3 adapter
- ``controllers/`` — Pyomo / scipy baseline controllers
- ``_common/``     — cross-layer utilities (rng_service, normalisation, …)
"""

__version__ = "0.1.0"
