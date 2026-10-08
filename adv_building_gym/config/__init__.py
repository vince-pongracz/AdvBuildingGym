"""Config subpackage marker.

Submodules are imported explicitly (no eager re-exports). Layout:
- ``config.trial_config`` — orchestrator (TrialConfig)
- ``config.env/``         — env topology + infra/statesource schedules
- ``config.data/``         — data scenario selection + DataCombinator
- ``config.rewards/``     — reward composition + curriculum
- ``config.training/``    — training hyperparameters + exploration reset
- ``config.utils/``       — cross-axis dataclass logging mixin
"""
