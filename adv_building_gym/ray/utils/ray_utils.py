"""Ray Tune utilities for custom trial naming and configuration."""

import re


def _slugify(name: str) -> str:
    """Filesystem-safe slug for trial-name suffixes.

    Strips characters that confuse path tooling / TB tag parsing — keeps
    [A-Za-z0-9._-] and collapses everything else to '_'.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "trial"


def make_trial_dirname_creator(trial_name: str | None = None):
    """Factory → Ray Tune ``trial_dirname_creator`` appending ``_<trial_name>`` to the short
    trial id (so the dir reflects the trial config). ``None`` → short trial_id only."""
    suffix = f"{_slugify(trial_name)}" if trial_name else ""

    def trial_dirname_creator(trial) -> str:
        trial_id_short = trial.trial_id[:6] if trial.trial_id else "unknown"
        trial_id_short = trial_id_short.replace("_", "")
        return f"{trial_id_short}_{suffix}"

    return trial_dirname_creator
