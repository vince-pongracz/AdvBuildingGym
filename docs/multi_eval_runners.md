# Parallel eval workers for `EvalStateActionCallback`

## Status

Not implemented. This document is the plan for **option B** discussed alongside
the `eval_trajectories/<exec_date>__<job>` directory-collision fix (option A,
already applied — see [eval_state_action_callback.py](../adv_building_gym/callbacks/eval_state_action_callback.py)).

Option A made parallel **sbatch jobs** safe by suffixing the run directory
with `SLURM_JOB_ID`. This document is about parallelising **evaluation inside a
single trial** (`evaluation_num_env_runners > 0`) so that an eval round of
`evaluation_duration=N` episodes can be split across multiple EnvRunners and
finish in `~N/K` wall-clock time.

## Motivation

Today, `evaluation_num_env_runners=0` (default, see
[common_model_config.py:352](../adv_building_gym/ray_training/common_model_config.py#L352))
means an eval round runs sequentially on the eval group's in-process local
EnvRunner. An eval round of 2 episodes × 288 steps takes roughly
`2 * 288 * step_time` seconds on the algo actor. With 4 training EnvRunners
already provisioned, increasing eval throughput is purely a callback-design
problem, not a resource problem.

## Why the current callback breaks under multi-worker eval

[`make_eval_state_action_cb_class`](../adv_building_gym/callbacks/eval_state_action_callback.py)
keeps its state in two module-level closure cells:

```python
_episode_buffer: list[dict[str, list[float]]] = []
_eval_round: list[int] = [0]
```

When `evaluation_num_env_runners=K>0`, RLlib serialises the AlgorithmConfig
(including the callback class) to each of the K eval EnvRunner actors. Each
worker process gets **its own copy** of these closure cells — they are not
shared, not aggregated, not visible to the driver.

Concretely, with `evaluation_duration=N` episodes distributed across K eval
EnvRunners (each receiving roughly `N/K` episodes per round):

1. Each worker's `_episode_buffer` only ever holds the episodes that worker
   sampled. The mean/min/max written to TB is over that worker's slice, not
   over all N episodes.
2. The flush condition `len(_episode_buffer) < evaluation_duration` never
   triggers correctly: a worker with `N/K < N` episodes per round never
   reaches `N` until it has accumulated buffers across `K` rounds. By that
   point its `_eval_round[0]` counter is `K`× off from the algorithm's
   actual `training_iteration`, so the `iter_NNNN` directory label is wrong.
3. Each worker opens its own `SummaryWriter(log_dir=…/iter_NNN)` to the
   *same* directory. K writers, same path, same global_step values, same
   tag names → TensorBoard merges them with last-write-wins on overlapping
   `(tag, step)` pairs. Output is garbled.

## Design

Move state off the closure and onto the EnvRunner instance, do all file I/O
on the driver in `on_evaluate_end`, gather per-worker buffers with
`foreach_env_runner`.

### Where state lives

- **Per-worker buffer**: attach to the EnvRunner instance as an ad-hoc
  attribute, e.g. `env_runner._eval_traj_buf`. This is reachable from the
  driver via `foreach_env_runner(lambda er: er._eval_traj_buf)`.
- **No driver-side `_eval_round` counter**: read `algorithm.iteration`
  directly inside `on_evaluate_end`. The previous per-worker counter was
  a workaround for not knowing the training iter from inside `on_episode_end`.
- **No closure cells**: factory still parameterises `tb_log_dir` and the
  run-dir suffix, but the mutable lists go away.

### Hook responsibilities

| Hook | Process | Responsibility |
|---|---|---|
| `on_episode_end` | per-worker | Skip if `env_runner.config.in_evaluation` is False. Extract per-step `raw/*`, `state/*`, `action/*`, `power/*`, `reward/*` from `episode.get_infos()` into a single `dict[str, list[float]]`. Append to `env_runner._eval_traj_buf`. **No file I/O.** |
| `on_evaluate_end` | driver | Gather buffers from every healthy eval EnvRunner via `foreach_env_runner`. Clear them. Flatten across workers. Compute per-step mean/min/max across all episodes. Open exactly one `SummaryWriter(log_dir=…/iter_{algorithm.iteration:06d})` and write. |

### Sketch

```python
def make_eval_state_action_cb_class(metrics_base_dir, exec_date=None):
    # ... compute tb_log_dir with job_suffix as today ...

    class EvalStateActionCallback(RLlibCallback):

        def on_episode_end(self, *, episode, env_runner, metrics_logger, env, **kwargs):
            if not env_runner.config.in_evaluation:
                return
            infos = episode.get_infos()
            if not infos:
                return
            ep_data = _extract_ep_data(infos)  # same logic as today
            buf = getattr(env_runner, "_eval_traj_buf", None)
            if buf is None:
                buf = env_runner._eval_traj_buf = []
            buf.append(ep_data)

        def on_evaluate_end(self, *, algorithm, metrics_logger, evaluation_metrics, **kwargs):
            eval_group = algorithm.eval_env_runner_group or algorithm.env_runner_group

            # Pull buffers from every eval EnvRunner (local + remote).
            worker_buffers: list[list[dict]] = eval_group.foreach_env_runner(
                lambda er: getattr(er, "_eval_traj_buf", []),
                local_env_runner=True,
            )
            # Reset buffers for the next round.
            eval_group.foreach_env_runner(
                lambda er: setattr(er, "_eval_traj_buf", []),
                local_env_runner=True,
            )

            episodes = [ep for wb in worker_buffers for ep in wb]
            if not episodes:
                return

            summarised = _aggregate_mean_min_max(episodes)  # same math as today
            run_dir = os.path.join(tb_log_dir, f"iter_{algorithm.iteration:06d}")
            with SummaryWriter(log_dir=run_dir) as writer:
                for key, vals in summarised.items():
                    for step_idx, val in enumerate(vals):
                        writer.add_scalar(key, val, global_step=step_idx)

    return EvalStateActionCallback
```

`_extract_ep_data` and `_aggregate_mean_min_max` are the same per-step extract
and per-step reduce that exist inline in the current callback today; the
rewrite should just lift them into module-level helpers.

## Concrete edits

1. [`adv_building_gym/callbacks/eval_state_action_callback.py`](../adv_building_gym/callbacks/eval_state_action_callback.py):
   - Delete `_episode_buffer` and `_eval_round` closure cells.
   - Move per-step extraction (currently inside `on_episode_end`) into a
     module-level `_extract_ep_data(infos)` helper.
   - Move the aggregation block (currently inside the same hook) into a
     module-level `_aggregate(episodes)` helper.
   - Reduce `on_episode_end` to: guard + extract + append to
     `env_runner._eval_traj_buf`.
   - Add `on_evaluate_end` with the gather → aggregate → write logic above.
2. [`adv_building_gym/ray_training/common_model_config.py:352`](../adv_building_gym/ray_training/common_model_config.py#L352):
   - Uncomment / set `evaluation_num_env_runners=K`. K=2 is a safe starting
     point; K should not exceed `evaluation_duration` (a worker can't sample
     a fractional episode).
   - Provision for the extra CPU(s): subtract K from `num_env_runners`
     derivation in `common_model_setup` if you want training EnvRunners to
     stay at the current count. Otherwise SLURM CPU pressure is unchanged
     because eval workers are idle during training.

## Things to be careful about

- **`evaluation_duration` vs `K`**: the eval `episodes` budget is divided
  across K workers. `evaluation_duration=2, K=4` means 2 workers run 1
  episode and 2 sit idle that round. Keep `evaluation_duration ≥ K`.
- **`foreach_env_runner(healthy_only=True)`**: a worker that is mid-restart
  during the round-trip will be skipped silently. That worker's episodes are
  lost from the TB trace for that iter. The current single-worker callback
  has the same risk on the local runner; with K workers the surface is K×
  larger but per-worker probability is per-RLlib-defaults small.
- **Side-channel attribute on EnvRunner**: `env_runner._eval_traj_buf` is not
  part of any RLlib state-dict. If Ray serialises an EnvRunner for migration
  (autoscaling, restart), the attribute is not preserved — the new actor
  starts with no buffer. Consequence: at most one eval round of missing data
  after the migration. Acceptable.
- **Eval `in_evaluation` guard**: must remain. With `evaluation_num_env_runners=0`
  the **training** EnvRunner local also gets the callback, so we must still
  skip when `env_runner.config.in_evaluation` is False to avoid buffering
  training episodes.

## Out of scope for this plan

- Aggregation across **slurm jobs / trials** (option A handles that by
  directory naming).
- Aggregation across **seeds** for figure-quality summaries — that belongs in
  the offline plotting layer (`plotting/`), not in the live TB callback.
- Migrating `EpisodeMetricsCallback` or `TrajectoryLoggingCallback` to the
  same pattern. They already work per-worker via `metrics_logger.log_value`
  and per-episode JSON / HDF5 files; they don't share files between workers
  and don't need this fix.

## Validation checklist before merging the rewrite

- Run with `evaluation_num_env_runners=0` — expect one tfevents file per
  `iter_NNN/`, identical content to pre-rewrite.
- Run with `evaluation_num_env_runners=2, evaluation_duration=2` — expect
  one tfevents file per `iter_NNN/`, mean curves averaged over all 2
  episodes regardless of which worker sampled which.
- Run with `evaluation_num_env_runners=4, evaluation_duration=4` — same;
  wall-clock per eval round should drop to ~1/4× the K=0 baseline.
- Diff TB output between K=0 and K=2: traces should look essentially the
  same (sample noise aside) for the same seed.
