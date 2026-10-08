# Data Source Management

Reference for how `AdvBuildingGym` swaps the CSV time-series each `StateSource` consumes,
across episodes and training iterations.

## Why swap data at all

Without rotation each `reset()` replays the same 288-step (one-day) window, which leads
to overfitting to that day, no estimate of generalisation across seasons / users / price
regimes, and no curriculum control. Rotating CSVs and starting offsets per episode gives
the policy a much wider distribution at zero per-step cost.

## Core building block — `DataCombinator`

`adv_building_gym/data_combinator/data_combinator.py` defines the variant pool. It
distinguishes correlated bundles from independent axes:

- **`scenarios`** — list of dicts; each dict is a bundle of sources that must change
  together (weather + price correlate seasonally, so they live in the same scenario).
  No Cartesian product is applied within a scenario.
- **`variable`** — `dict[source_name, list[paths]]` for sources independent of everything
  else (e.g. EV usage profile, desired-temperature profile). The Cartesian product
  is applied across these axes.

Effective pool: `len(scenarios) × Π len(variable[axis])`. A *variant* is a flat
`dict[source_name, path]`; only the statesources whose names appear in the dict are
reloaded — the rest are untouched.

Other knobs:

- `swap_every_n_episodes` — episode-boundary cadence used by the env itself.
- `mode` — `"cycle"` (deterministic walk over the pool) or `"random"` (uniform pick).
- `day` — `"random"` (sample a day each episode), `"each"` (walk days sequentially),
  or an ISO date string (pin every episode to that day). Day offset is computed by
  `DataCombinator.get_day_offset(...)` and is independent of variant selection.

YAML loading is done via `load_data_combinator_config(...)` in
`adv_building_gym/config/data_config.py`. Configs live in `configs/data_scheduler/`:

- `train_data_combinator_config.yaml` — shuffled, random days, swap every N episodes.
- `eval_data_combinator_config.yaml` — deterministic, sequential days.

`scenario_sources` may use a `{year}` placeholder against a `years:` list to expand
into concrete file paths; with `include_synthesized: true` the loader picks up
synthesis-pipeline outputs via `config/utils/discover_scenarios.discover_synthetic_scenarios()`.

## How a swap reaches a `StateSource`

`StateSource.reload(ds_path)` (in `adv_building_gym/devices/statesources/base.py`)
re-reads the CSV and calls the template hook `_post_load_data_processing()`. Subclasses
override the hook so post-processing (normalisation, event parsing, scale-factor caching)
runs both at construction and on every reload:

| StateSource | `_post_load_data_processing()` |
|---|---|
| `WeatherDataSource` | normalise temperature column via `self.normalise` strategy |
| `EnergyPriceYearDynDataSource` | cache `price_max`; normalise `baseprice` → `E_price_norm` |
| `EVState` | reset runtime state and re-parse connect/disconnect events |
| `InsideTemperature` | detect column name and normalise to `[-1, 1]` |

Relative paths in the variant dict are resolved against the project root (four levels
up from `base.py`) so worker CWDs do not matter.

`EnvConfig.create_statesources()` builds statesources **without** a `ds_path` — they
start with `self.ts = None` and only become usable after the first variant push.

## Three swap surfaces

These mechanisms are independent and additive. In a normal Ray run all three coexist.

### 1. Episode-boundary swap (`AdvBuildingGym.reset()`)

The env owns an `episode_count` and a fallback `DataCombinator`. On every `reset()`:

```python
if options and "data_variant" in options:           # external override (3.)
    self.apply_data_variant(options["data_variant"])
elif self.data_combinator.variants:                  # local episode-boundary swap
    variant = self.data_combinator.get_variant(self.episode_count, self._rng)
    self.apply_data_variant(variant)
# Day offset always picked from the combinator, regardless of variant origin
row_offset, day_mode = self.data_combinator.get_day_offset(...)
```

This is the only swap path for evaluation, controllers, and any non-Ray driver. In Ray
training it provides per-worker variance: each env_runner advances its own counter, so
workers explore different variants between episode-budget pushes.

### 2. Episode-budget push (`data_schedule_callback.py`)

`create_data_schedule_on_train_result_cb(combinator, num_env_runners)` returns
an `on_train_result` function. After every
`max(combinator.swap_every_n_episodes, num_env_runners)` episodes accumulated across
all env_runners (read from `result["env_runners"]["num_episodes_lifetime"]`) it pulls
the next variant from `combinator.variants` and pushes it to **every** env_runner —
both `algorithm.env_runner_group` and `algorithm.eval_env_runner_group` — via
`foreach_env_runner`. The first invocation always fires (initial CSV push). Each
runner traverses its wrapper chain to reach `AdvBuildingGym` and calls
`apply_data_variant(variant)`.

What this buys:

- All training workers see the same year-long CSV bundle within an iteration window,
  so reward / loss curves and per-iteration TensorBoard tags are interpretable as
  "iteration N had variant Z" instead of a moving cocktail.
- In-training eval is run on the same variant the training batch was collected on
  (eval against a held-out config requires `run_eval_ray.py` with its own data YAML).
- Reward and infrastructure curriculum callbacks (`reward_switch_callback`,
  `infra_schedule_callback`) also fire on `on_train_result`, so all three curricula
  share the same iteration boundaries.

What it does **not** synchronise: the per-episode day offset within a variant — each
env's `reset()` still calls `DataCombinator.get_day_offset()` against its local RNG, so
with `day="random"` workers land on different days of the same year.

The callback is composed (with optional reward_switch / infra_schedule callbacks) into
a single `on_train_result` chain by `register_callbacks(...)` in
`adv_building_gym/ray_training/common_model_config.py`, which `common_model_setup(...)`
wires in when `data_combinator is not None`.

### 3. External override via `reset(options=...)`

Standard Gymnasium escape hatch:

```python
env.reset(options={"data_variant": {"ev_schedule": "data/ev_usage_profiles/ev_3.csv"},
                   "row_offset": 0})
```

Used by tests, controller benchmarks, or any caller that needs to pin a specific
variant or day. The override beats the combinator-driven episode swap.

## Wiring at run time

`run_train_ray.py`:

1. `data_combinator = load_data_combinator_config(args.data_config or default, seed)`
2. Registers `"AdvBuilding"` env via `adv_building_env_creator`, passing `data_combinator`
   in the env config dict.
3. Calls `common_model_setup(..., data_combinator=data_combinator, ...)`. That, in turn,
   conditionally registers the iteration-boundary callback inside `register_callbacks(...)`.

`adv_building_env_creator` (in `adv_building_gym/envs/env_creator.py`) reads
`config.get("data_combinator")` and passes it to `AdvBuildingGym(...)` as the per-worker
fallback combinator.

`run_eval_ray.py` builds a separate combinator (typically `eval_data_combinator_config.yaml`)
and uses only the episode-boundary path; there is no iteration callback during eval.

## Multi-worker semantics summary

| Aspect | Coordinated across runners? |
|---|---|
| Year-long CSV bundle (variant) | Yes, when the iteration callback is registered |
| Day offset within the variant | No — independent per env's local RNG |
| Episode counter | No — each runner owns its own |

If you need fully synchronised days as well, set `day` to a fixed ISO string in the
training YAML (loses day diversity) or extend the iteration callback to push a shared
day offset.

## Verification quick-checks

- **Cycling**: 2-entry EV combinator with `swap_every_n_episodes=2`; 6 resets cycle
  `ev_1, ev_1, ev_2, ev_2, ev_1, ev_1`.
- **Cartesian (variable only)**: `variable={"ev": [a, b], "weather": [w1, w2]}` →
  `len(combinator.variants) == 4`.
- **Correlation-safe**: `scenarios=[{w1,p1},{w2,p2}], variable={"ev":[a,b,c]}` →
  6 variants, no `w1`+`p2` mix.
- **No regression**: `DataCombinator()` (empty) → no reloads, identical behaviour to
  pre-combinator runs.
- **Override**: `env.reset(options={"data_variant": {"ev_schedule": p}})` sets
  `ev_state.ds_path == p` regardless of episode counter.
- **Roundtrip**: `EnvConfigManager.save(...)` / `load(...)` preserves the
  data-combinator YAML reference (combinator state itself lives in the data YAML).
- **Episode-budget push**: with `swap_every_n_episodes=1`, after the first swap
  every env_runner reports the expected `ds_path`. Inspecting from outside requires walking
  the wrapper chain:
  `env_runner.env.env.envs[0].unwrapped.statesources[-1].ds_path`.

## Wrapper chain (important when reaching the env from a callback)

RLlib wraps the user environment in several layers:

```
env_runner.env → DictInfoToList → SyncVectorEnv → .envs[i] → TimeLimit
              → OrderEnforcing → PassiveEnvChecker → AdvBuildingGym
```

`SyncVectorEnv` is **not** a Gymnasium `Wrapper` subclass, so calling `.unwrapped` on
the outermost layer does not reach `AdvBuildingGym`. The data-schedule callback walks
the chain explicitly:

```python
vec_env = getattr(env_runner, "env", None)            # may be None on local driver
sync_vec = getattr(vec_env, "env", vec_env)           # unwrap DictInfoToList
for sub_env in getattr(sync_vec, "envs", []):
    sub_env.unwrapped.apply_data_variant(variant)
```

## Related files

- `adv_building_gym/data_combinator/data_combinator.py` — combinator dataclass.
- `adv_building_gym/config/data_config.py` — YAML loader.
- `adv_building_gym/config/utils/discover_scenarios.py` — auto-discovery of
  synthesis-pipeline outputs.
- `adv_building_gym/devices/statesources/base.py` — `reload()` and the
  `_post_load_data_processing()` template-method hook.
- `adv_building_gym/envs/building_adv.py` — episode-boundary swap and `reset(options=...)`
  override.
- `adv_building_gym/callbacks/data_schedule_callback.py` — iteration-boundary callback.
- `adv_building_gym/ray_training/common_model_config.py` — composes the callback into
  the shared `on_train_result` chain alongside reward / infra curricula.
- `configs/data_scheduler/{train,eval}_data_combinator_config.yaml` — runtime configs.
