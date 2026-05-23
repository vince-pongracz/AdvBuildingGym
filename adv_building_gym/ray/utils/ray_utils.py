"""Ray Tune utilities for custom trial naming and configuration."""

import re


def _slugify(name: str) -> str:
    """Filesystem-safe slug for trial-name suffixes.

    Strips characters that confuse path tooling / TB tag parsing — keeps
    [A-Za-z0-9._-] and collapses everything else to '_'.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "trial"


def make_trial_dirname_creator(trial_name: str | None = None):
    """Factory: return a Ray Tune ``trial_dirname_creator`` that appends
    ``_<trial_name>`` to the short trial id so the on-disk dir reflects which
    trial config produced it (useful when multiple trials share an algo dir).

    Args:
        trial_name: TrialConfig.trial_name. ``None`` falls back to the legacy
            behaviour (short trial_id only).
    """
    suffix = f"{_slugify(trial_name)}" if trial_name else ""

    def trial_dirname_creator(trial) -> str:
        trial_id_short = trial.trial_id[:6] if trial.trial_id else "unknown"
        trial_id_short = trial_id_short.replace("_", "")
        return f"{trial_id_short}_{suffix}"

    return trial_dirname_creator
