"""Utility for converting per-step info dicts into columnar trajectory data."""

import numpy as np


def extract_trajectory_from_infos(
    infos: list[dict],
    initial_info: dict | None = None,
    state_keys: list[str] | None = None,
    action_keys: list[str] | None = None,
) -> dict:
    """Convert a list of per-step info dicts into columnar trajectory data.

    Args:
        infos: List of info dicts from AdvBuildingGym.step() calls (length T).
            Each dict is expected to contain:
              - info["state"]: dict[str, np.ndarray] (when log_full_info=True)
              - info["action"]: dict[str, np.ndarray] (clipped, dict format)
              - info["reward"]: float
              - info["reward_breakdown"]: dict[str, float]
              - info["cum_E_kWh"]: float
        initial_info: Optional info dict from reset() (contains initial state).
            When provided, the initial/reset state is prepended as step 0
            (with zero actions and zero reward).
        state_keys: State keys to extract. Auto-discovered from first info if None.
        action_keys: Action keys to extract. Auto-discovered from first info if None.

    Returns:
        dict with columnar trajectory data, ready for JSON serialization.
        Keys include "step", "state" (nested dict keyed by original state
        names — scalars as flat lists, vectors preserved as lists of lists),
        "action" (nested dict, same convention), "reward",
        "reward_breakdown" (nested dict), "cum_E_kWh", "step_power_kW",
        and optionally "power_breakdown" (nested dict keyed by infra name).
    """
    if not infos:
        return {}

    # Auto-discover keys from the first step info
    first = infos[0]
    if state_keys is None and "state" in first:
        state_keys = list(first["state"].keys())
    if action_keys is None and "action" in first:
        action_keys = list(first["action"].keys())

    # Discover reward breakdown keys
    reward_names: list[str] = []
    if "reward_breakdown" in first:
        reward_names = list(first["reward_breakdown"].keys())

    # Discover per-infrastructure power breakdown keys
    power_names: list[str] = []
    if "power_breakdown" in first:
        power_names = list(first["power_breakdown"].keys())

    # Build initial-conditions row (step 0) from reset info
    initial_row: dict | None = None
    if initial_info is not None:
        initial_row = {
            "reward": 0.0,
            "cum_E_kWh": 0.0,
        }
        # Initial state from reset
        if "state" in initial_info and state_keys:
            initial_row["state"] = initial_info["state"]
        # Zero actions
        if action_keys and "action" in first:
            initial_row["action"] = {
                k: np.zeros_like(first["action"][k]) for k in action_keys
            }
        # Zero reward breakdown
        if reward_names:
            initial_row["reward_breakdown"] = {name: 0.0 for name in reward_names}
        # Zero power breakdown
        if power_names:
            initial_row["power_breakdown"] = {name: 0.0 for name in power_names}

    # Combine initial row + step infos
    all_rows = ([initial_row] if initial_row else []) + list(infos)
    num_steps = len(all_rows)

    # Initialize columnar output
    trajectory: dict[str, list | dict] = {"step": list(range(num_steps))}

    # State columns (nested under "state" dict).
    # Scalars → flat list of floats; vectors → list of lists.
    if state_keys:
        state_dict: dict[str, list] = {}
        for key in state_keys:
            sample = None
            for row in all_rows:
                if "state" in row:
                    sample = np.atleast_1d(row["state"].get(key))
                    break
            if sample is None:
                continue
            is_scalar = sample.size == 1
            col: list = []
            for row in all_rows:
                if "state" in row and key in row["state"]:
                    val = np.atleast_1d(row["state"][key])
                    col.append(float(val.flat[0]) if is_scalar else val.tolist())
                else:
                    col.append(None)
            state_dict[key] = col
        if state_dict:
            trajectory["state"] = state_dict

    # Action columns (nested under "action" dict).
    # Scalars → flat list of floats; vectors → list of lists.
    if action_keys:
        action_dict: dict[str, list] = {}
        for key in action_keys:
            sample = None
            for row in all_rows:
                if "action" in row:
                    sample = np.atleast_1d(row["action"].get(key))
                    break
            if sample is None:
                continue
            is_scalar = sample.size == 1
            col: list = []
            for row in all_rows:
                if "action" in row and key in row["action"]:
                    val = np.atleast_1d(row["action"][key])
                    col.append(float(val.flat[0]) if is_scalar else val.tolist())
                else:
                    col.append(None)
            action_dict[key] = col
        if action_dict:
            trajectory["action"] = action_dict

    # Reward column
    trajectory["reward"] = [
        float(row.get("reward", 0.0)) for row in all_rows
    ]

    # Per-reward breakdown columns (nested under "reward_breakdown" dict)
    if reward_names:
        reward_bd: dict[str, list] = {}
        for name in reward_names:
            reward_bd[name] = []
            for row in all_rows:
                bd = row.get("reward_breakdown", {})
                reward_bd[name].append(float(bd.get(name, 0.0)))
        trajectory["reward_breakdown"] = reward_bd

    # Cumulative energy
    cum_values = [float(row.get("cum_E_kWh", 0.0)) for row in all_rows]
    trajectory["cum_E_kWh"] = cum_values

    # Instantaneous power (kW) in a step = ΔEnergy (kWh) / ΔTime (h)
    trajectory["step_power_kW"] = [float(row.get("step_power_kW", 0.0)) for row in all_rows]

    # Per-infrastructure power breakdown (nested under "power_breakdown" dict)
    if power_names:
        power_bd: dict[str, list] = {}
        for name in power_names:
            power_bd[name] = []
            for row in all_rows:
                bd = row.get("power_breakdown", {})
                power_bd[name].append(float(bd.get(name, 0.0)))
        trajectory["power_breakdown"] = power_bd

    return trajectory
