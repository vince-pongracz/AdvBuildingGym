# Dynamic Data Source Management

> **Status**:
> - Approaches A, C, and D1: **implemented**.
> - Approach B and D2: not realised.

## Problem

The environment previously trained every episode on the same fixed set of CSV time-series
files: one day of weather measurements, one energy-price curve, and one EV usage profile.
Every `reset()` replayed the identical 288-step window, which meant the agent saw the same
conditions on every episode.

This causes:
- **Overfitting to a single day** — the policy learns to exploit the specific patterns of
  that day rather than a general control strategy.
- **Poor generalisation** — performance on different days (different seasons, EV user
  behaviours, price volatility) is unknown and likely worse.
- **No curriculum learning** — difficulty/complexity of the scenario cannot be increased
  over the course of training.

---

## Relevant Architecture

### How statesources consume data

Each `StateSource` loads its CSV at construction (`pd.read_csv(ds_path)` in
`adv_building_gym/devices/statesources/base.py`). It then reads rows by index in
`update_state()`:
```python
value = float(self.ts.iloc[int(self.iteration)]["column_name"])
```

`self.iteration` is reset to 0 on every `env.reset()` and incremented each `env.step()`.

Statesources that need post-processing after CSV load override `_post_load_data_processing()`:

| StateSource | `_post_load_data_processing()` implementation |
|---|---|
| `WeatherDataSource` | Normalises temperature column via `self.normalise` strategy |
| `EnergyPriceDataSource` | Caches `self.price_max` scalar |
| `EVState` | Resets runtime state (`_ev_connected`, `_current_spec`, etc.) and calls `_parse_events()` |
| `InsideTemperature` | Detects column name; normalises to `[-1, 1]` |

The base class provides `reload(ds_path)` which re-reads the CSV, calls
`_post_load_data_processing()`, and logs the change. Relative paths are resolved against
`_PROJECT_ROOT` so Ray worker processes (whose CWD may differ) can still find the files.

### Where data paths are wired

`EnvConfig.create_statesources()` in `adv_building_gym/config/env_config.py` creates statesources
**without** a `ds_path` — they start with `self.ts = None`. The `DataCombinator`
provides the actual file paths and pushes them to statesources via `reload()` at episode
boundaries (Approach A) or training iteration boundaries (Approach D1).

### Available data variants

```
data/
├── eval1/
│   ├── LLEC_outdoor_temperature_5min_data.csv     # weather (single day, legacy)
│   └── price_data_2025_1.csv                      # energy price (single day, legacy)
├── e_price/
│   ├── awattar/
│   │   ├── <YEAR>_prices.csv                      # raw hourly aWATTar fetch
│   │   └── price_data_<YEAR>.csv                  # preprocessed 5-min (2017–2026)
│   └── e_charts/
│       ├── <YEAR>_15m_prices.csv                  # raw 15-min e-charts fetch
│       └── price_data_<YEAR>.csv                  # preprocessed 5-min (2018–2026)
├── weather/
│   ├── LLEC/                                      # LLEC sensor data (legacy)
│   ├── dwd/
│   │   ├── downloaded/                            # raw DWD station files
│   │   └── preprocessed/
│   │       ├── <YEAR>_merged_04177.csv            # merged 5-min weather (2008–2026)
│   │       └── <YEAR>_missing_entries.txt          # gap reports per year
│   └── zenodo/
│       ├── csvs_weather/
│       │   └── <YEAR>_weather.csv                 # multi-year weather (2018–2020)
│       ├── csvs_2018_data_1min/SFH10.csv …        # 1-min building data
│       └── csvs_2020_data_1min/SFH10.csv …        # 1-min building data
├── ev_usage_profiles/
│   ├── ev_0.csv                                   # empty profile (no EV events)
│   └── ev_1.csv  …  ev_5.csv                     # 5 distinct EV user profiles
├── inside_temp/
│   └── inside_temp_0.csv  …  inside_temp_2.csv   # desired indoor temperature profiles
```

---

## Proposed Approaches

### Approach A — DataCombinator with Episode-Count Swap *(Implemented)*

The environment maintains a **DataCombinator** object and an **episode counter**. Every N
completed episodes the combinator selects the next variant and reloads only the affected
statesources in-place.

#### DataCombinator data model

`DataCombinator` is a dataclass (`adv_building_gym/data_combinator/data_combinator.py`) that
distinguishes between **correlated** and **independent** data sources:

- **`scenarios`** — a list of explicit variant dicts. Each scenario is a bundle of sources
  that must always change together (e.g. weather + electricity price, which correlate
  seasonally). The Cartesian product is never applied within a scenario.
- **`variable`** — a dict mapping `source_name → list[paths]` for sources that are
  independent of everything else (e.g. EV profiles, which reflect individual user behaviour
  unrelated to season or price). The Cartesian product is applied across all `variable` axes.

The final variant pool is `scenarios × variable_combinations`. If only one field is provided
the other collapses to a single neutral element, reproducing the simpler cases.

```python
# Simple: five EV profiles, weather unchanged — 5 variants
DataCombinator(
    variable={"ev_schedule": [f"data/ev_usage_profiles/ev_{i}.csv" for i in range(1, 6)]},
    swap_every_n_episodes=1,
    mode="cycle",
)

# Correlated weather+price, independent EV — 2×5 = 10 variants
# Weather and price always change together; EV cycles independently of them.
DataCombinator(
    scenarios=[
        {"weather": "data/weather_summer.csv", "E_price": "data/price_summer.csv"},
        {"weather": "data/weather_winter.csv", "E_price": "data/price_winter.csv"},
    ],
    variable={
        "ev_schedule": [f"data/ev_usage_profiles/ev_{i}.csv" for i in range(1, 6)],
    },
    swap_every_n_episodes=20,
)
# Generated variants (10 total):
#   {weather: summer, E_price: summer, ev_schedule: ev_1}
#   {weather: summer, E_price: summer, ev_schedule: ev_2}
#   ...
#   {weather: winter, E_price: winter, ev_schedule: ev_5}
```

A *variant* is a plain `dict[str, str]` mapping `source_name → path`. Only the statesources
whose names appear in the dict are reloaded; others are untouched.

The DataCombinator also supports **day selection** via the `day` parameter:
- `"random"` — sample a uniformly random day each episode (default for training).
- `"each"` — walk through days sequentially (default for evaluation).
- A date string (e.g. `"2025-03-15"`) — pin every episode to that calendar day.

Training and evaluation each have their own DataCombinator config:
- `configs/train_data_combinator_config.yaml` — shuffled, random days, swap every 2 episodes.
- `configs/eval_data_combinator_config.yaml` — deterministic, sequential days, swap every episode.

These YAML configs define `scenario_sources` with `{year}` placeholders and a `years` list;
the DataCombinator expands these into concrete file paths at construction time. Augmented
data files (e.g. from data synthesize pipelines) can be auto-discovered via `synthesized_paths`.

#### Code changes (implemented)

**1. `adv_building_gym/data_combinator/data_combinator.py`** — DataCombinator dataclass with
`scenarios`/`variable` axes, `variants` property (Cartesian product), `get_variant()`,
day selection, year-based scenario expansion, and augmented data discovery.

**2. `adv_building_gym/devices/statesources/base.py`** — `reload(ds_path)` method and
`_post_load_data_processing()` template-method hook. Relative paths resolved against
`_PROJECT_ROOT` (four levels up from `base.py`). Both `__init__` CSV loading and `reload()`
call the hook so subclasses define post-processing once.

**3. Subclass `_post_load_data_processing()` overrides (four files)**

| File | What moves into `_post_load_data_processing()` |
|---|---|
| `outer/weather.py` | Column normalisation via `self.normalise` strategy → `self.ts["temp_out_norm"]` |
| `outer/energy_price.py` | `normalise_series(self.ts["baseprice"], self.normalise)` → `self.ts["E_price_norm"]` |
| `outer/ev_state.py` | Runtime state reset (`_ev_connected`, `_current_spec`, `_events`, `_event_lookup`) + `_parse_events()` |
| `outer/inside_temperature.py` | Column detection + min-max normalisation to `[-1, 1]` |

**4. `adv_building_gym/envs/building_adv.py`** — `data_combinator` constructor parameter,
`episode_count`, `_rng` attributes, public `apply_data_variant(variant)` method,
Approach A swap in `reset()`, Approach C override via `reset(options=...)`.

**5. `adv_building_gym/config/env_config.py`** — `EnvConfig` dataclass (renamed from `Config`).
`create_statesources()` creates sources **without** `ds_path` — the combinator provides
paths at runtime.

**6. `adv_building_gym/config/env_config_manager.py`** — `EnvConfigManager` (renamed from
`ConfigManager`). Serialises/deserialises `EnvConfig` to/from **YAML** (previously JSON).

**7. `adv_building_gym/envs/env_creator.py`** — passes `data_combinator` to `AdvBuildingGym`.

**8. `adv_building_gym/data_combinator/__init__.py`** — exports `DataCombinator`.

#### Multi-worker behaviour

Each Ray env-runner creates its own `AdvBuildingGym` instance with its own episode counter.
Workers advance through variants independently, which increases experience diversity in the
replay buffer or rollout batches. This is intentional.

#### Pros / Cons

| | |
|---|---|
| ✅ | Self-contained — no Ray callbacks needed |
| ✅ | Zero overhead per step (reload only at episode boundaries) |
| ✅ | Works with SB3, standalone eval, and Ray RLlib |
| ✅ | Serialisable — `DataCombinator` round-trips through YAML |
| ✅ | Partial swaps — only sources listed in `scenarios`/`variable` are ever reloaded |
| ✅ | Combinatorial — specifying N axes generates N₁×N₂×… variants automatically |
| ✅ | Correlation-safe — correlated sources (weather + price) bundled in `scenarios`; independent sources in `variable` |
| ✅ | `data_combinator=None` → zero behaviour change (fully backwards-compatible) |
| ⚠️ | Workers have independent counters (desired diversity, but not coordinated) |
| ⚠️ | File I/O at episode boundary (negligible for CSV; could be pre-loaded for HDF5) |

---

### Approach B — Consecutive-Day Offset *(Alternative)*

Instead of swapping files, each statesource receives a `row_offset` and reads
`ts.iloc[self.iteration + row_offset]`. A single long CSV spanning many days is loaded once.
The offset increments by `EPISODE_LENGTH` (288) on every `reset()`, so each episode
automatically reads the next day's data.

```
Day 0:  rows   0 – 287
Day 1:  rows 288 – 575
Day 2:  rows 576 – 863
...
```

When the offset exceeds the file length the env wraps around or raises an exception.

**Required changes:**
- `EnvSyncInterface.synchronise()` extended to carry `row_offset`
- All `update_state()` implementations updated to add the offset
- A single multi-day CSV prepared for each statesource

**Pros**: No file I/O during training; smooth, sequential coverage of all data.
**Cons**: Requires new multi-day CSV preparation; higher refactor cost in all `update_state()`
implementations; aligns with the second TODO (`VP 2026.02.11`) but is a larger change.
Best suited as a follow-up once the CSV preprocessing pipeline is in place.

---

### Approach C — External Control via `reset(options=...)` *(Implemented, extension on top of A)*

Standard Gymnasium allows `env.reset(options={"data_variant": {...}})`. An external
scheduler (Ray callback, curriculum object, or test harness) can inject a specific variant
before each episode rather than relying on the built-in counter.

```python
# In AdvBuildingGym.reset() — evaluated after Approach A swap
if options and "data_variant" in options:
    self.apply_data_variant(options["data_variant"])
```

Useful for evaluation (always force a fixed test variant) or curriculum learning (external
scheduler decides difficulty). Since `apply_data_variant` is already defined by
Approach A, no additional code is needed beyond that one `if` block.

---

### Approach D — EnvRunner-Level Reconfiguration via RLlib Callbacks *(Advanced)*

Rather than each environment managing its own swap schedule internally (Approach A), this
approach delegates variant switching to the RLlib training loop. A callback fires between
training iterations and pushes a new variant to every env_runner simultaneously, guaranteeing
that all workers change dataset at the same training iteration boundary.

Two sub-variants exist depending on how deeply the reconfiguration must go.

---

#### D1 — In-place env mutation via `foreach_env_runner` *(Implemented)*

The environment objects already exist inside each Ray actor. The callback reaches into each
actor through `EnvRunnerGroup.foreach_env_runner()` and calls
`env.apply_data_variant(variant)` directly. No Ray actors are stopped or restarted.

**When to prefer D1 over Approach A:**

| Concern | Approach A | D1 |
|---|---|---|
| Swap granularity | Per-episode (per-worker, unsynchronised) | Per-iteration (all workers at once) |
| Metric alignment | Approximate — different workers are on different episodes | Exact — iteration boundary = swap boundary |
| Curriculum feedback | Episode counter only | Can read `result` dict (reward_rate, loss, …) |
| RLlib dependency | None | Requires RLlib callback |
| `data_combinator=None` regression | Zero | Callback simply not registered |

**Implementation — `adv_building_gym/callbacks/data_schedule_callback.py`:**

The D1 callback uses a **function-based factory** (`create_data_schedule_on_train_result`)
that returns an `on_train_result` function. This integrates cleanly with the existing
class-based checkpoint callback via `config.callbacks(CheckpointClass, on_train_result=fn)`.

```python
def create_data_schedule_on_train_result(
    combinator: DataCombinator,
    swap_every_n_iterations: int = 10,
):
    def on_train_result(*, algorithm, result: dict, **kwargs) -> None:
        iteration: int = result.get("training_iteration", 0)
        if iteration % swap_every_n_iterations != 0:
            return
        variant = combinator.get_variant(iteration // swap_every_n_iterations)
        if not variant:
            return
        _push_variant_to_runners(algorithm, variant, iteration)
    return on_train_result
```

The `_push_variant_to_runners` helper handles both training and evaluation env_runner groups,
and guards against `env_runner.env is None` (the local driver env_runner may have no env in
the new API stack).

**Registration — wired through `common_model_config.py`:**

The D1 callback is conditionally registered in `common_model_config()` when
`data_combinator` is not None. `run_train_ray.py` passes
`active_config.data_combinator` through.

```python
# In common_model_config():
callback_kwargs = {"on_episode_end": on_episode_end_callback}
if data_combinator is not None:
    callback_kwargs["on_train_result"] = create_data_schedule_on_train_result(
        data_combinator, data_swap_every_n_iterations,
    )
config.callbacks(checkpoint_callback_class, **callback_kwargs)
```

**Env access path — wrapper chain (verified at runtime):**

RLlib wraps the user environment in multiple layers. The full chain is:

```
env_runner.env → DictInfoToList → SyncVectorEnv → .envs[i] → TimeLimit → OrderEnforcing → PassiveEnvChecker → AdvBuildingGym
```

- `env_runner.env` is a `DictInfoToList` (Gymnasium vector wrapper), **not** `AdvBuildingGym`.
- `.env` on `DictInfoToList` gives the `SyncVectorEnv`.
- `.envs` on `SyncVectorEnv` is a list of individually-wrapped sub-environments.
- `.unwrapped` on each sub-env traverses `TimeLimit → OrderEnforcing → PassiveEnvChecker` to reach `AdvBuildingGym`.

The D1 callback navigates this chain:
```python
vec_env = getattr(env_runner, "env", None)
if vec_env is None:
    return
sync_vec = getattr(vec_env, "env", vec_env)  # unwrap DictInfoToList
for sub_env in getattr(sync_vec, "envs", []):
    sub_env.unwrapped.apply_data_variant(variant)
```

**Lesson learned:** The original plan assumed `env_runner.env` would be the raw
`AdvBuildingGym` — two successive SLURM runs (`1616771`, `1616939`) revealed
`DictInfoToList` and then `SyncVectorEnv` wrappers. Using `.unwrapped` on the outermost
wrapper is insufficient because `SyncVectorEnv` is not a Gymnasium `Wrapper` subclass.
The correct approach is to explicitly traverse `DictInfoToList → SyncVectorEnv → .envs[i] → .unwrapped`.

---

#### D2 — Full EnvRunner recreation *(for structural env changes)*

Stops all remote Ray actors and starts entirely new ones from scratch via
`EnvRunnerGroup.reset(new_remote_workers)`. Required only if the change involves
constructor-level parameters that cannot be mutated after construction — e.g. different
observation/action space dimensionality, different episode length, or adding/removing
statesource components entirely. For file-path swaps alone (the current use case), D2 is
**not needed**.

**APIs involved:**

```python
# 1. Build new env config (modify env_config.py fields)
new_env_config = copy_and_modify(algorithm.config)

# 2. Create replacement Ray actor handles (RLlib internal helper)
new_handles = [
    algorithm._make_env_runner(
        env_context=new_env_config.env_config,
        worker_index=i + 1,
        ...
    )
    for i in range(algorithm.config.num_env_runners)
]

# 3. Hard-swap all remote workers
algorithm.env_runner_group.reset(new_remote_workers=new_handles)

# 4. Re-sync model weights and connector states
algorithm.env_runner_group.sync_weights()
algorithm.env_runner_group.sync_env_runner_states(
    config=algorithm.config,
    from_worker=algorithm.env_runner_group.local_env_runner,
)
```

**Cost / risk table for D2:**

| Factor | Impact | Notes |
|---|---|---|
| In-flight samples | Lost | Accept; use at iteration boundaries |
| Episode continuity | Broken | All episodes reset on new actors |
| Replay buffer (SAC) | **Not affected** — lives in learner, not env_runners | No data loss for off-policy |
| Model weights | Must resync explicitly (`sync_weights()`) | Straightforward |
| Connector states | Must resync (`sync_env_runner_states()`) | Observation preprocessing consistency |
| Ray actor creation overhead | ~1–2 s per worker | Batch; do every 50–100 iterations |

**Recommendation:** implement D2 only when structural env changes are required. For the
dataset-scheduling use case D1 (or Approach A) is sufficient and far cheaper.

---

#### Approach D — Pros / Cons

| | |
|---|---|
| ✅ | All workers switch variant at the **same training iteration** — precise metric tracking |
| ✅ | Callback can read `result` dict and react to training progress (curriculum) |
| ✅ | Evaluation env_runners can be kept on a fixed variant independently |
| ✅ | D1 has near-zero overhead (no actor restart) |
| ✅ | Fully composable with Approach A (both can run simultaneously or independently) |
| ⚠️ | Requires RLlib callback infrastructure (not usable with SB3 or standalone) |
| ⚠️ | `env_runner.env` is wrapped (`DictInfoToList → SyncVectorEnv → sub-envs`) — must traverse chain to reach `AdvBuildingGym` |
| ⚠️ | D2 requires internal RLlib APIs that may change between RLlib versions |

---

## Implementation Summary

**Approaches A, C, and D1** are implemented. Empty `variable`/`scenarios` → no reloads →
existing behaviour preserved. `data_combinator=None` on `EnvConfig` disables all swapping.

**Approach B** is deferred until multi-day CSVs are prepared from the zenodo/HDF5 sources.

**Approach D2** is deferred until structural environment changes (observation space, episode
length) are required.

### Files touched

| File | Action | Status |
|---|---|---|
| `adv_building_gym/data_combinator/data_combinator.py` | **Created** (moved from `config/`) | Done |
| `adv_building_gym/callbacks/data_schedule_callback.py` | **Created** (D1 callback) | Done |
| `adv_building_gym/devices/statesources/base.py` | Added `reload()` + `_post_load_data_processing()` | Done |
| `adv_building_gym/devices/statesources/outer/weather.py` | Extract → `_post_load_data_processing()` | Done |
| `adv_building_gym/devices/statesources/outer/energy_price.py` | Extract → `_post_load_data_processing()` | Done |
| `adv_building_gym/devices/statesources/outer/ev_state.py` | Extract → `_post_load_data_processing()` | Done |
| `adv_building_gym/devices/statesources/outer/inside_temperature.py` | Extract → `_post_load_data_processing()` | Done |
| `adv_building_gym/envs/building_adv.py` | Episode counter + combinator + Approach A/C | Done |
| `adv_building_gym/config/env_config.py` | `EnvConfig` dataclass (renamed from `Config`) | Done |
| `adv_building_gym/config/env_config_manager.py` | `EnvConfigManager` — YAML serialisation (renamed from `ConfigManager`, migrated from JSON) | Done |
| `adv_building_gym/envs/env_creator.py` | Pass `data_combinator` | Done |
| `adv_building_gym/data_combinator/__init__.py` | Export `DataCombinator` | Done |
| `adv_building_gym/callbacks/__init__.py` | Export `create_data_schedule_on_train_result` | Done |
| `adv_building_gym/ray_training/common_model_config.py` | D1 callback wiring | Done |
| `run_train_ray.py` | Pass `data_combinator` to `common_model_config()` | Done |
| `configs/env/env_test1_small.yaml` (+ `configs/infras/test1_small.yaml`) | **Created** — small env config (YAML, replaces `test1.json`); originally a single `configs/env_cfg/env_test1_small.yaml`, later split into wrapper + infras + statesources + env_meta | Done |
| `configs/env/env_test1_mid.yaml` (+ `configs/infras/test1_mid.yaml`) | **Created** — mid env config (YAML); later split as above | Done |
| `configs/env/env_test1_large.yaml` (+ `configs/infras/test1_large.yaml`) | **Created** — large env config (YAML); later split as above | Done |
| `configs/train_data_combinator_config.yaml` | **Created** — training DataCombinator config | Done |
| `configs/eval_data_combinator_config.yaml` | **Created** — evaluation DataCombinator config | Done |
| `configs/training_param_config.yaml` | **Created** — algorithm hyperparameters (YAML) | Done |
| `data/ev_usage_profiles/ev_0.csv` | **Created** — empty EV profile (no events) | Done |
| `data/inside_temp/inside_temp_{0..2}.csv` | **Created** — desired indoor temperature profiles | Done |

---

## Verification

1. **Smoke test — cycling**: create env with a 2-entry EV combinator
   (`swap_every_n_episodes=2`), run 6 resets, assert `ev_state.ds_path` cycles
   `ev_1 → ev_1 → ev_2 → ev_2 → ev_1 → ev_1`. *(Passed during implementation)*

2. **Cartesian product (variable only)**: `DataCombinator(variable={"ev_schedule": [ev1, ev2], "weather": [w1, w2]}).variants`
   returns exactly 4 dicts covering all combinations. *(Passed during implementation)*

3. **Correlation-safe product**: `DataCombinator(scenarios=[...], variable={"ev_schedule": [ev1, ev2, ev3]}).variants`
   returns 6 dicts (2 scenarios × 3 EV profiles). No variant mixes `w1` with `p2` or `w2` with `p1`. *(Passed during implementation)*

4. **No regression**: `data_combinator=None` → identical behaviour to previous code. *(Passed during implementation)*

5. **Approach C override**: `env.reset(options={"data_variant": {"ev_schedule": ev3}})`;
   assert `ev_state.ds_path == ev3` regardless of episode counter.

6. **Serialisation roundtrip**: `EnvConfigManager.save(config, path)` / `EnvConfigManager.load(path)` (YAML);
   assert `config.data_combinator.scenarios`, `.variable`, and `swap_every_n_episodes` are preserved. *(Passed during implementation)*

7. **Training integration (Approach A)**: `python run_train_ray.py --algorithm ppo --episodes 100` with
   EV combinator enabled; inspect `ep_metrics/` — EV-related reward contributions should
   vary across episodes matching the profile swap schedule.

8. **Approach D1 callback**: run a short training with `data_swap_every_n_iterations=1`;
   after iteration 1 assert all env_runners report the expected `ev_state.ds_path`.
   Note: to inspect sub-envs from outside, traverse the wrapper chain:
   `env_runner.env.env.envs[0].unwrapped.statesources[-1].ds_path`.
