"""Ray Tune utilities for custom trial naming and configuration."""


def trial_dirname_creator(trial) -> str:
    """
    Create a custom dir name for Ray Tune trials.

    Args:
        trial: Ray Tune Trial object with properties:
            - trial_id: unique ID like "d1e85_00000"
            - config: the param_space dict
            - experiment_tag: experiment identifier

    Returns:
        Custom directory name string (uses only trial_id, no seed).
    """
    trial_id_short = trial.trial_id[:6] if trial.trial_id else "unknown"
    return f"{trial_id_short}"
