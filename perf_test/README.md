# perf_test — observation flattening benchmark

Measures the per-step cost of the two observation-flattening mechanisms that can turn
AdvBuildingGym's `Dict` observation space into the flat `Box` a policy consumes:

| | mechanism | where it runs | key order |
|---|---|---|---|
| **A** | `gymnasium.wrappers.FlattenObservation` | env-side, outermost wrapper (`ray/env_creator.py`) | Dict insertion order (`spaces.items()`) |
| **B** | `ray.rllib.connectors.env_to_module.FlattenObservations` | env-to-module + learner connector pipeline | dm-tree, keys sorted alphabetically |

**A is the current production path** — `adv_building_env_creator` applies it for all
single-agent algorithms, and `common_model_setup(flatten_observations_env_side=True)`
skips the connector flatteners accordingly. B is the retired path, kept here as the
comparison baseline (still used by the multi-agent driver).

The two produce **identical values in a different index order** — a fixed permutation
(verified separately over 20 episodes / 5760 steps: 0/5760 index-identical,
5760/5760 identical after permuting). A checkpoint is therefore only valid with the
mechanism it was trained under; mixing them silently feeds the policy permuted features
because the dimension check (87 == 87) still passes.

## Running it

```bash
sbatch perf_test/slurm_bench_flatten.sh      # 2 CPUs, CPU-only, ~20 s
```

Logs land in `perf_test/logs/slurm-bench-flatten-<jobid>.{out,err}`.

The benchmark steps the Dict-observation env directly and, **at every step**, flattens
that same real observation with both mechanisms: the production `FlattenObservation`
instance, and the real `FlattenObservations` connector driven on a growing
`SingleAgentEpisode` exactly as RLlib's sampling loop drives it. `env.step()` is timed
alongside as the baseline.

## Results

Two independent SLURM runs: **1660610** (below) and **1660621** (repeatability check).

Environment: node `haicn1708`, 2 CPUs (`OMP/MKL/OPENBLAS_NUM_THREADS=2`), Python 3.12.9,
trial `configs/trial_cfgs/v0/STA/lin_battery_only_price.yaml`, observation space
`Dict` with 21 keys → flat `(87,)`, 3 episodes × 288 steps = 864 steps.

### Per-step timings (mean µs/step)

| Episode | steps | `env.step()` | wrapper (A) | connector (B) |
|--------:|------:|-------------:|------------:|--------------:|
| 1 | 288 | 476.8 | 40.9 | 58.2 |
| 2 | 288 | 473.7 | 40.7 | 57.7 |
| 3 | 288 | 471.3 | 40.5 | 57.9 |
| **ALL** | **864** | **473.9** | **40.7** | **57.9** |

### Derived

| metric | value |
|---|---|
| connector / wrapper ratio | **1.42×** |
| saving by using the wrapper | 17.2 µs/step (5.0 ms/episode) |
| wrapper overhead on top of a step | 7.91 % |
| connector overhead on top of a step | 10.89 % |
| flattening cost per 288-step episode | 11.7 ms (A) vs 16.7 ms (B) |

### Repeatability — job 1660621 (independent rerun, same config)

| Episode | steps | `env.step()` | wrapper (A) | connector (B) |
|--------:|------:|-------------:|------------:|--------------:|
| 1 | 288 | 475.8 | 40.2 | 57.5 |
| 2 | 288 | 486.8 | 40.5 | 58.2 |
| 3 | 288 | 473.5 | 39.8 | 57.3 |
| **ALL** | **864** | **478.7** | **40.2** | **57.7** |

Ratio **1.44×**, wrapper 7.74 % / connector 10.75 % overhead, saving 17.5 µs/step.
Run-to-run spread across the two jobs is ~1 % on the flatteners (40.7 vs 40.2 µs;
57.9 vs 57.7 µs) and ~1 % on the ratio (1.42× vs 1.44×).

### Batch vs interactive (login node)

The same benchmark run interactively on the shared login node is uniformly ~1.6× slower
in absolute terms, but the ratio is unchanged — the relative result is robust to
hardware and contention:

| | login node | compute node (2 CPUs) |
|---|---:|---:|
| `env.step()` | 789.4 | 473.9 |
| wrapper (A) | 65.4 | 40.7 |
| connector (B) | 93.6 | 57.9 |
| **ratio** | **1.43×** | **1.42×** |

## Interpretation

The wrapper is consistently ~1.4× cheaper per call. Two causes: the gymnasium kernel is
faster than RLlib's (a plain loop over `spaces.items()` vs dm-tree structure traversal —
45.0 vs 55.9 µs measured in isolation), and the connector adds episode-iterator dispatch
plus an in-place rewrite of the episode's stored observation. The connector path also
pays a second pass on the learner side, re-flattening every observation in each train
batch.

In absolute terms the difference is small: ~17 µs/step against a ~474 µs env step
dominated by physics and CSV lookups, i.e. ~3 % of sampling wall-clock. Speed was
therefore *not* the reason to move to the wrapper — DreamerV3 architecturally requires
it (`training_step` reads `env.single_observation_space`, and a `Dict` space has
`shape=None`, crashing `do_symlog_obs`). The speed win is a side benefit.

Trade-off retained by the connector path: any connector placed before
`FlattenObservations` still sees named Dict keys (per-key transforms, `keys_to_remove`,
masking). Env-side flattening erases names at the env boundary, and the
`FlattenObservation` wrapper must stay outermost forever — a future obs-adding wrapper
placed outside it would corrupt the layout.

## Files

| file | purpose |
|---|---|
| `bench_flatten_episodes.py` | the benchmark (3 episodes, per-step dual timing) |
| `slurm_bench_flatten.sh` | sbatch wrapper, 2 CPUs, CPU-only partition |
| `logs/` | stdout/stderr of recorded runs (jobs 1660610, 1660621) |
