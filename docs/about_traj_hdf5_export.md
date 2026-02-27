# Trajectory HDF5 Export

The `TrajectoryLoggingCallback` writes evaluation episode trajectories to a
single HDF5 file (`trajectories.hdf5`) alongside the per-episode JSON files.
One file collects all episodes from a run, making bulk analysis across
episodes efficient without loading individual JSONs.

Source: `adv_building_gym/callbacks/trajectory_logging_callback.py`

## Output location

```
ep_metrics/
  YYYYMMDD_HHMM00/
    trajectories.hdf5           <-- single HDF5 for the run
    jsons/
      abc123_trajectory.json  <-- per-episode JSONs (unchanged)
      def456_trajectory.json
```

The file is opened in append mode (`h5py.File(..., "a")`) on each episode end,
so episodes accumulate incrementally as the evaluation progresses.

## HDF5 structure

```
trajectories.hdf5
|
+-- <episode_id>/                          GROUP  (one per episode, e.g. "abc123")
|   |
|   |   attrs:  version       (int)        schema version, currently 1
|   |           episode_id    (str)        full episode UUID
|   |           seed          (int)        RNG seed used for reset
|   |           length        (int)        number of env steps
|   |           eval          (bool)       True (always, callback is eval-only)
|   |
|   +-- summary/                           GROUP
|   |       attrs:  achieved_reward         (float)  total reward collected
|   |               max_achievable_reward   (float)  theoretical max reward
|   |               reward_rate             (float)  achieved / max, in [0, 1]
|   |               cum_E_kWh              (float)  cumulative energy consumption
|   |
|   +-- trajectory/                        GROUP
|       |
|       |-- step                           DATASET  (N,)       int step index
|       |-- reward                         DATASET  (N,)       per-step reward
|       |-- cum_E_kWh                      DATASET  (N,)       cumulative energy
|       |-- step_power_kW                  DATASET  (N,)       instantaneous power
|       |-- raw_policy_action_0            DATASET  (N,)       raw action dim 0
|       |-- raw_policy_action_1            DATASET  (N,)       raw action dim 1
|       |-- ...                            DATASET  (N,)       one per action dim
|       |
|       +-- state/                         GROUP
|       |   |-- temp_in_norm               DATASET  (N,)       scalar state
|       |   |-- temp_out_norm              DATASET  (N,)       scalar state
|       |   |-- battery_pct               DATASET  (N,)       scalar state
|       |   |-- battery_pct_hist           DATASET  (N, H)     history buffer
|       |   |-- prev_action               DATASET  (N, A)     previous action
|       |   +-- ...
|       |
|       +-- action/                        GROUP
|       |   |-- HP_action                  DATASET  (N, 2)     [energy, mode]
|       |   |-- battery_action             DATASET  (N,)       charge/discharge
|       |   +-- ...
|       |
|       +-- reward_breakdown/              GROUP
|           |-- temp_reward                DATASET  (N,)       per-step component
|           |-- economic_reward            DATASET  (N,)       per-step component
|           +-- ...
|
+-- <episode_id>/                          (next episode, same structure)
+-- ...
```

N = episode length (including the prepended initial/reset step).

### Mapping rules

| JSON type                     | HDF5 representation          | Example                         |
|-------------------------------|------------------------------|---------------------------------|
| Scalar metadata               | Group attribute              | `version`, `seed`, `length`     |
| Summary scalar                | `summary/` group attribute   | `achieved_reward`, `reward_rate`|
| Flat list of numbers          | 1-D dataset `(N,)`           | `step`, `reward`, `cum_E_kWh`  |
| Dict of lists (state, action) | Subgroup with datasets       | `state/temp_in_norm`            |
| List of lists (history)       | 2-D dataset `(N, M)`         | `state/battery_pct_hist`        |

All numeric datasets are stored as `float32`.

## Reading the file

```python
import h5py

with h5py.File("ep_metrics/20260224_143000/trajectories.hdf5", "r") as f:
    # List all episodes
    print("Episodes:", list(f.keys()))

    # Read one episode
    ep = f["abc123"]
    print("Seed:", ep.attrs["seed"])
    print("Reward rate:", ep["summary"].attrs["reward_rate"])

    # Load trajectory arrays
    rewards = ep["trajectory/reward"][:]          # shape (N,)
    temp    = ep["trajectory/state/temp_in_norm"][:]  # shape (N,)
    hp_act  = ep["trajectory/action/HP_action"][:]    # shape (N, 2)

    # Iterate all episodes into a dict
    data = {}
    for eid in f:
        data[eid] = {
            "reward_rate": f[eid]["summary"].attrs["reward_rate"],
            "rewards": f[f"{eid}/trajectory/reward"][:],
        }
```

```python
# Quick inspection: print full tree
with h5py.File("trajectories.hdf5", "r") as f:
    f.visit(print)
```
