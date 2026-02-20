# Dynamic Data Source Management

## Problem

The environment currently trains every episode on the same fixed set of CSV time-series
files: one day of weather measurements, one energy-price curve, and one EV usage profile.
Every `reset()` replays the identical 288-step window, which means the agent sees the same
conditions on every episode.

This causes:
- **Overfitting to a single day** — the policy learns to exploit the specific patterns of
  that day rather than a general control strategy.
- **Poor generalisation** — performance on different days (different seasons, EV user
  behaviours, price volatility) is unknown and likely worse.
- **No curriculum learning** — difficulty/complexity of the scenario cannot be increased
  over the course of training.

This is the gap already flagged in `adv_building_gym/config/env_config.py` (line 22):
> `# TODO VP 2026.01.13. : How to learn more days during training? -- solve consecutive days from data sources`

---

## Relevant Architecture

### How statesources consume data today

Each `StateSource` loads its CSV once at construction (`pd.read_csv(ds_path)` in
`adv_building_gym/devices/statesources/base.py:34`). It then reads rows by index in
`update_state()`:
```python
value = float(self.ts.iloc[int(self.iteration)]["column_name"])
```

`self.iteration` is reset to 0 on every `env.reset()` and incremented each `env.step()` —
so every episode reads exactly the same timestep rows from the different data files.

Several statesources also perform post-processing after loading the CSV, which must be
repeated when the file is swapped:

| StateSource | Post-processing after CSV load |
|---|---|
| `WeatherDataSource` | Normalises temperature column to `[-1, 1]` |
| `EnergyPriceDataSource` | Caches `price_max` scalar |
| `EVState` | Parses timestamp events into an `_event_lookup` dict |
| `InsideTemperature` | Detects column name; normalises to `[-1, 1]` |

### Where data paths are wired

`Config.create_statesources()` in `adv_building_gym/config/env_config.py:53` hard-codes the
paths. `adv_building_env_creator` in `adv_building_gym/envs/env_creator.py:11` calls this
factory; the resulting statesources live for the lifetime of that env-runner process.

### Available data variants today

```
data/
├── LLEC_outdoor_temperature_5min_data.csv       # weather (single day)
├── LLEC_outdoor_temperature_5min_data_cleaned.csv
├── price_data_2025.csv                          # energy price (single day)
├── ev_usage_profiles/
│   ├── ev_1.csv  …  ev_5.csv                    # 5 distinct EV user profiles
└── zenodo/
    ├── 2018_weather.hdf5  …  2020_weather.hdf5  # multi-year weather (HDF5)
    └── csvs_2018_data_1min/SFH10.csv …          # 1-min building data
```

The EV profiles are the most immediately usable set of variants because they already exist
in the correct CSV format with matching column names.

---

## Proposed Approaches

### Approach A — DataCombinator with Episode-Count Swap *(Recommended)*

The environment maintains a **DataCombinator** object and an **episode counter**. Every N
completed episodes the combinator selects the next variant and reloads only the affected
statesources in-place.

#### DataCombinator data model

`DataCombinator` is a small dataclass (`adv_building_gym/config/data_combinator.py`) that
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

#### Required code changes

**1. NEW `adv_building_gym/config/data_combinator.py`**

```python
import itertools
from dataclasses import dataclass, field
from typing import Literal
import numpy as np

@dataclass
class DataCombinator:
    """Schedules CSV data source variants across training episodes.

    Separates correlated sources (scenarios) from independent ones (variable):

    Args:
        scenarios: Explicit variant bundles for correlated sources (e.g. weather + price).
                   Each entry is a dict[source_name, path]. These are never cross-producted
                   with each other — they advance as a unit.
        variable:  Maps source_name → list[paths] for sources independent of everything else
                   (e.g. EV profiles). Cartesian product is applied across variable axes.
        swap_every_n_episodes: Advance to the next variant every N episodes.
        mode: "cycle" (round-robin) or "random".

    Final pool = scenarios × variable_combinations.
    If scenarios is empty, only variable combinations are used (and vice versa).
    """
    scenarios: list[dict[str, str]] = field(default_factory=list)
    variable: dict[str, list[str]] = field(default_factory=dict)
    swap_every_n_episodes: int = 1
    mode: Literal["cycle", "random"] = "cycle"

    @property
    def variants(self) -> list[dict[str, str]]:
        # Build variable combinations (Cartesian product of independent axes)
        if self.variable:
            keys = list(self.variable.keys())
            variable_combos: list[dict[str, str]] = [
                dict(zip(keys, combo))
                for combo in itertools.product(*(self.variable[k] for k in keys))
            ]
        else:
            variable_combos = [{}]  # neutral element — no independent sources

        # Cross-product: each scenario × each variable combination
        if self.scenarios:
            return [{**scenario, **var_combo}
                    for scenario in self.scenarios
                    for var_combo in variable_combos]
        # No scenarios — return variable combinations only (omit the empty-dict case)
        return variable_combos if self.variable else []

    def get_variant(self, episode_count: int, rng: np.random.Generator | None = None) -> dict[str, str]:
        pool = self.variants
        if not pool:
            return {}
        if self.mode == "random" and rng is not None:
            return pool[int(rng.integers(0, len(pool)))]
        return pool[(episode_count // self.swap_every_n_episodes) % len(pool)]

    def to_dict(self) -> dict:
        return {"scenarios": self.scenarios,
                "variable": self.variable,
                "swap_every_n_episodes": self.swap_every_n_episodes,
                "mode": self.mode}

    @classmethod
    def from_dict(cls, d: dict) -> "DataCombinator":
        return cls(scenarios=d.get("scenarios", []),
                   variable=d.get("variable", {}),
                   swap_every_n_episodes=d.get("swap_every_n_episodes", 1),
                   mode=d.get("mode", "cycle"))
```

**2. `adv_building_gym/devices/statesources/base.py` — `reload()` + `_post_load()` hook**

The naive approach (`self.ts = pd.read_csv(ds_path)`) is insufficient because subclasses
perform post-processing (normalization, event parsing, column detection) right after loading.
Use a template-method hook so each subclass defines processing once:

```python
def _post_load(self) -> None:
    """Override to re-run post-processing after a new CSV is loaded."""
    pass

def reload(self, ds_path: str) -> None:
    """Load a new time-series file without recreating this StateSource instance."""
    self.ds_path = ds_path
    self.ts = pd.read_csv(ds_path)
    self._post_load()
    logger.info("StateSource '%s' reloaded from %s", self.name, ds_path)
```

Also call `self._post_load()` at the end of the existing CSV-loading branch in `__init__`,
so subclasses only need to override `_post_load()` once.

**3. Subclass `_post_load()` overrides (four files)**

In each statesource extract the post-CSV logic currently in `__init__` into `_post_load()`,
then replace the original block with `self._post_load()`:

| File | What moves into `_post_load()` |
|---|---|
| `outer/weather.py` | Column normalisation → `self.ts["temp_out_norm"]` |
| `outer/energy_price.py` | `self.price_max = float(self.ts["price_normalized"].max())` |
| `outer/ev_state.py` | `self._parse_events()` call |
| `outer/inside_temperature.py` | Column detection + min-max normalisation |

**4. `adv_building_gym/envs/building_adv.py` — episode counter + combinator**

New constructor parameter:
```python
data_combinator: DataCombinator | None = None,
```

New attributes:
```python
self.episode_count: int = 0
self.data_combinator = data_combinator
self._rng: np.random.Generator | None = None   # seeded in reset()
```

New private method:
```python
def _apply_datasource_variant(self, variant: dict[str, str]) -> None:
    for ss in self.statesources:
        if ss.name in variant:
            ss.reload(variant[ss.name])
```

Logic inserted at the top of `reset()`, before the iteration/synchronise block:
```python
self.episode_count += 1
if self.data_combinator is not None:
    variant = self.data_combinator.get_variant(self.episode_count, self._rng)
    if variant:
        self._apply_datasource_variant(variant)
        logger.info("Episode %d: datasource variant %s", self.episode_count, variant)
```

Seed the RNG when a seed is provided:
```python
if seed is not None:
    self._rng = np.random.default_rng(seed)
```

**5. `adv_building_gym/config/env_config.py` — `data_combinator` field on `Config`**

```python
from adv_building_gym.config.data_combinator import DataCombinator

@dataclass
class Config:
    ...
    data_combinator: DataCombinator | None = None
```

Remove the `# TODO VP 2026.01.13.` comment (addressed by this feature).

**6. `adv_building_gym/config/config_manager.py` — serialise `DataCombinator`**

In `to_dict()`:
```python
if config.data_combinator is not None:
    d["data_combinator"] = config.data_combinator.to_dict()
```

In `from_dict()`:
```python
if "data_combinator" in d:
    config.data_combinator = DataCombinator.from_dict(d["data_combinator"])
```

**7. `adv_building_gym/envs/env_creator.py` — pass combinator**

```python
return AdvBuildingGym(
    infras=infras,
    statesources=statesources,
    rewards=rewards,
    building_props=env_config.building_props,
    data_combinator=env_config.data_combinator,
)
```

**8. `adv_building_gym/config/__init__.py` — export**

```python
from .data_combinator import DataCombinator
```

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
| ✅ | Serialisable — `DataCombinator` round-trips through JSON |
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

### Approach C — External Control via `reset(options=...)` *(Extension on top of A)*

Standard Gymnasium allows `env.reset(options={"datasource_variant": {...}})`. An external
scheduler (Ray callback, curriculum object, or test harness) can inject a specific variant
before each episode rather than relying on the built-in counter.

```python
def reset(self, *, seed=None, options=None):
    ...
    # Approach C: external override (additive, evaluated after Approach A swap)
    if options and "datasource_variant" in options:
        self._apply_datasource_variant(options["datasource_variant"])
```

This is a one-liner added to `reset()` on top of Approach A. Useful for evaluation
(always force a fixed test variant) or curriculum learning (external scheduler decides
difficulty). Since `_apply_datasource_variant` is already defined by Approach A, no
additional code is needed beyond that one `if` block.

---

### Approach D — EnvRunner-Level Reconfiguration via RLlib Callbacks *(Advanced)*

Rather than each environment managing its own swap schedule internally (Approach A), this
approach delegates variant switching to the RLlib training loop. A callback fires between
training iterations and pushes a new variant to every env_runner simultaneously, guaranteeing
that all workers change dataset at the same training iteration boundary.

Two sub-variants exist depending on how deeply the reconfiguration must go.

---

#### D1 — In-place env mutation via `foreach_env_runner` *(recommended sub-variant)*

The environment objects already exist inside each Ray actor. The callback reaches into each
actor through `EnvRunnerGroup.foreach_env_runner()` and calls
`env._apply_datasource_variant(variant)` directly. No Ray actors are stopped or restarted.

**When to prefer D1 over Approach A:**

| Concern | Approach A | D1 |
|---|---|---|
| Swap granularity | Per-episode (per-worker, unsynchronised) | Per-iteration (all workers at once) |
| Metric alignment | Approximate — different workers are on different episodes | Exact — iteration boundary = swap boundary |
| Curriculum feedback | Episode counter only | Can read `result` dict (reward_rate, loss, …) |
| RLlib dependency | None | Requires `RLlibCallback` |
| `data_combinator=None` regression | Zero | Callback simply not registered |

**Key RLlib APIs (confirmed present in installed RLlib version):**

- `algorithm.env_runner_group.foreach_env_runner(func, local_env_runner=True, timeout_seconds=None)` — synchronously calls `func` on every healthy env_runner (local + remote); blocks until all return.
- `algorithm.eval_env_runner_group` — same API for evaluation workers; may be `None` if evaluation is disabled.
- Callback hook: `on_train_result(*, algorithm, result, **kwargs)` — fires after every `algorithm.train()` call.

**Implementation — `adv_building_gym/callbacks/data_schedule_callback.py`:**

```python
import logging
from ray.rllib.callbacks.callbacks import RLlibCallback
from adv_building_gym.config.data_combinator import DataCombinator

logger = logging.getLogger(__name__)

class DataScheduleCallback(RLlibCallback):
    """Pushes a new DataCombinator variant to all env_runners every N training iterations.

    Compatible with Approach A: if the environment also has its own episode counter the two
    swap schedules are independent and additive.  Set data_combinator=None on the Config
    (Approach A disabled) and use only this callback if iteration-aligned swapping is desired.
    """

    def __init__(self, combinator: DataCombinator, swap_every_n_iterations: int = 10) -> None:
        super().__init__()
        self.combinator = combinator
        self.swap_every_n_iterations = swap_every_n_iterations

    def on_train_result(self, *, algorithm, result: dict, **kwargs) -> None:
        iteration: int = result.get("training_iteration", 0)
        if iteration % self.swap_every_n_iterations != 0:
            return

        variant = self.combinator.get_variant(iteration // self.swap_every_n_iterations)
        if not variant:
            return

        def apply(env_runner) -> None:
            # env_runner.env is the AdvBuildingGym instance (SingleAgentEnvRunner)
            env_runner.env._apply_datasource_variant(variant)

        algorithm.env_runner_group.foreach_env_runner(
            apply, local_env_runner=True, timeout_seconds=None
        )
        if algorithm.eval_env_runner_group is not None:
            algorithm.eval_env_runner_group.foreach_env_runner(
                apply, local_env_runner=True, timeout_seconds=None
            )
        logger.info("Iteration %d: all env_runners switched to variant %s", iteration, variant)
```

**Registration in `run_train_ray.py`:**

```python
from adv_building_gym.callbacks.data_schedule_callback import DataScheduleCallback
from adv_building_gym.config.data_combinator import DataCombinator

combinator = DataCombinator(
    scenarios=[
        {"weather": "data/weather_summer.csv", "E_price": "data/price_summer.csv"},
        {"weather": "data/weather_winter.csv", "E_price": "data/price_winter.csv"},
    ],
    variable={"ev_schedule": [f"data/ev_usage_profiles/ev_{i}.csv" for i in range(1, 6)]},
    swap_every_n_iterations=10,
)
config.callbacks(DataScheduleCallback, combinator=combinator, swap_every_n_iterations=10)
```

**Note on env access path:** `env_runner.env` is the `AdvBuildingGym` instance when using
`SingleAgentEnvRunner` (new API stack). If RLlib wraps it in a `VectorEnv` the path would be
`env_runner.env.envs[0]`; verify with `type(env_runner.env)` at runtime and adjust if needed.

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
| ⚠️ | `env_runner.env` access path must be verified for the installed RLlib version |
| ⚠️ | D2 requires internal RLlib APIs that may change between RLlib versions |

---

## Recommended Approach

Implement **Approach A** (`DataCombinator` + `reload()/_post_load()` + episode counter +
`data_combinator` field on `Config` + wired through env creator). Empty `variable`/`scenarios`
→ no reloads → existing behaviour preserved.

Add **Approach C** as a one-liner in `reset()` at the same time (trivial cost).

Add **Approach D1** (`DataScheduleCallback`) as an optional callback that can be registered
in `run_train_ray.py` when iteration-aligned, coordinated swapping is needed. It reuses
`DataCombinator` and `_apply_datasource_variant()` from Approach A — no new core logic.

Defer **Approach B** until multi-day CSVs are prepared from the zenodo/HDF5 sources.

Defer **Approach D2** until structural environment changes (observation space, episode
length) are required.

### Files touched

| File | Action |
|---|---|
| `adv_building_gym/config/data_combinator.py` | **Create** |
| `adv_building_gym/devices/statesources/base.py` | Add `reload()` + `_post_load()` |
| `adv_building_gym/devices/statesources/outer/weather.py` | Extract → `_post_load()` |
| `adv_building_gym/devices/statesources/outer/energy_price.py` | Extract → `_post_load()` |
| `adv_building_gym/devices/statesources/outer/ev_state.py` | Extract → `_post_load()` |
| `adv_building_gym/devices/statesources/outer/inside_temperature.py` | Extract → `_post_load()` |
| `adv_building_gym/envs/building_adv.py` | Episode counter + combinator + Approach C |
| `adv_building_gym/config/env_config.py` | Add `data_combinator` field |
| `adv_building_gym/config/config_manager.py` | Serialise `DataCombinator` |
| `adv_building_gym/envs/env_creator.py` | Pass `data_combinator` |
| `adv_building_gym/config/__init__.py` | Export `DataCombinator` |
| `adv_building_gym/callbacks/data_schedule_callback.py` | **Create** (Approach D1, optional) |

---

## Verification

1. **Smoke test — cycling**: create env with a 2-entry EV combinator
   (`swap_every_n_episodes=2`), run 6 resets, assert `ev_state.ds_path` cycles
   `ev_1 → ev_1 → ev_2 → ev_2 → ev_1 → ev_1`.

2. **Cartesian product (variable only)**: `DataCombinator(variable={"ev_schedule": [ev1, ev2], "weather": [w1, w2]}).variants`
   returns exactly 4 dicts covering all combinations.

3. **Correlation-safe product**: `DataCombinator(scenarios=[{"weather": w1, "E_price": p1}, {"weather": w2, "E_price": p2}], variable={"ev_schedule": [ev1, ev2, ev3]}).variants`
   returns 6 dicts (2 scenarios × 3 EV profiles). No variant mixes `w1` with `p2` or `w2` with `p1`.

4. **No regression**: `data_combinator=None` → identical behaviour to current code.

5. **Approach C override**: `env.reset(options={"datasource_variant": {"ev_schedule": ev3}})`;
   assert `ev_state.ds_path == ev3` regardless of episode counter.

6. **Serialisation roundtrip**: `ConfigManager.save(config)` / `ConfigManager.load(path)`;
   assert `config.data_combinator.scenarios`, `.variable`, and `swap_every_n_episodes` are preserved.

7. **Training integration (Approach A)**: `python run_train_ray.py --algorithm ppo --timesteps 5e4` with
   EV combinator enabled; inspect `ep_metrics/` — EV-related reward contributions should
   vary across episodes matching the profile swap schedule.

8. **Approach D1 callback**: register `DataScheduleCallback` with `swap_every_n_iterations=1`
   on a short run; after iteration 1 assert all env_runners report the expected `ev_state.ds_path`
   via `algorithm.env_runner_group.foreach_env_runner(lambda r: r.env.statesources[-1].ds_path)`.
