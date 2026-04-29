# Callbacks

RLlib callbacks that hook into the training loop to log metrics, save checkpoints,
record trajectories, and schedule data variants.

Link: https://docs.ray.io/en/latest/rllib/rllib-callback.html#rllib-callback-docs

## Overview

| Callback | Hook | Runs on | Purpose |
|---|---|---|---|
| `EpisodeMetricsCallback` | `on_episode_end` | EnvRunner (worker) | Log scalar episode metrics (reward, reward_rate, energy) |
| `TrajectoryLoggingCallback` | `on_episode_end` | EnvRunner (eval only) | Save full per-step trajectory as JSON + HDF5 |
| `BestModelCheckpointCallback` | `on_train_result` | Algorithm (main process) | Save best-model checkpoints every N episodes |
| `DataScheduleCallback` | `on_train_result` | Algorithm (main process) | Push new data variant to all env_runners every N iterations |

## Architecture

All callbacks use the **factory pattern**: a `make_*` function captures configuration
in a closure and returns a configured class (or callable). RLlib requires class types
for `callbacks_class` and plain functions for keyword hooks like `on_train_result`.

### Class-based vs function-based

- **Class-based** (`RLlibCallback` / `DefaultCallbacks` subclass): Used when the
  callback needs internal state across calls (e.g. `BestModelCheckpointCallback`
  tracks `best_metric_value`). Returned by `make_checkpoint_callback_class()`,
  `make_episode_metrics_callback_class()`, and `make_trajectory_logging_callback_class()`.

- **Function-based** (plain callable): Used for stateless per-iteration logic.
  `create_data_schedule_on_train_result()` returns an `on_train_result` function.

### Registration

All callbacks are assembled and registered in `common_model_config.py`:

```python
callback_classes = [checkpoint_class, episode_metrics_class]
if log_trajectories:
    callback_classes.append(trajectory_class)

config.callbacks(
    callbacks_class=callback_classes,
    on_train_result=create_data_schedule_on_train_result(combinator, n),
)
```

RLlib executes subclass callbacks in list order, then callables.

## Callback details

### EpisodeMetricsCallback (`episode_metrics_callback.py`)

Factory: `make_episode_metrics_callback_class(env_id, rewards, metrics_base_dir, exec_date, dump_metrics_json)`

Fires on every `on_episode_end` (training and evaluation). Logs:
- `achieved_reward` (mean/min/max) — episode total reward
- `reward_rate` (mean/min/max) — achieved / max_achievable (normalised [0, 1])
- `cum_E_kWh` — cumulative energy consumption

When `dump_metrics_json=True`, also writes a per-episode JSON file with observations,
actions, and rewards to `{metrics_base_dir}/{date}/`.

### TrajectoryLoggingCallback (`trajectory_logging_callback.py`)

Factory: `make_trajectory_logging_callback_class(rewards, metrics_base_dir, exec_date)`

Fires on `on_episode_end` but **only during evaluation** (`env_runner.config.in_evaluation`).
Requires `env.log_full_info = True` on evaluation EnvRunners so `info["state"]` contains
named state variables.

Outputs:
- Per-episode JSON in `{metrics_base_dir}/{date}/jsons/`
- Shared HDF5 file (`trajectories.hdf5`) with one group per episode

### DataScheduleCallback (`data_schedule_callback.py`)

Factory: `create_data_schedule_on_train_result(combinator, swap_every_n_iterations)`

Fires on `on_train_result`. Every N training iterations, pushes a new `DataCombinator`
variant to all env_runners (training + evaluation) via `foreach_env_runner`. An empty
combinator (no variants) is a safe no-op.

## Important: `on_episode_end` vs `on_train_result`

These hooks run on **different processes**:

- `on_episode_end` runs on **EnvRunner actors** (workers). No access to the Algorithm
  or its state. Use `metrics_logger.log_value()` to report metrics.
- `on_train_result` runs on the **Algorithm actor** (main process). Has access to
  `algorithm.save()`, the full `result` dict, and can call `foreach_env_runner`.

State is **not shared** between these instances.
