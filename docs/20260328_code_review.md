# Code Review: AdvBuildingGym

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Critical Issues](#2-critical-issues)
3. [SOLID Principle Violations](#3-solid-principle-violations)
4. [Dead Code and Unnecessary Code](#4-dead-code-and-unnecessary-code)
5. [Clean Code Issues](#5-clean-code-issues)
6. [Architectural Weaknesses](#6-architectural-weaknesses)
7. [Code Duplication](#7-code-duplication)
8. [Meaningless or Unused Parameters](#8-meaningless-or-unused-parameters)
9. [Recommendations by Priority](#9-recommendations-by-priority)

---

## 1. Executive Summary

The codebase implements a well-structured plugin architecture for RL-based building energy control. The component model (Infrastructure, StateSource, RewardFunction) is sound in concept. However, the implementation has accumulated significant technical debt across several dimensions:

- **27+ TODO/FIXME comments** indicating known but unaddressed issues
- **Duplicated patterns** across reward functions, state sources, and training scripts

---

## 2. Critical Issues

### 2.2 Global Seed Mutation

`building_adv.py:370-371` sets `np.random.seed(seed)` and `random.seed(seed)` on every `reset()` call. This mutates the global RNG state for all threads/processes sharing the Python interpreter.

`base_building_gym.py:421` sets `np.random.seed(42)` during data splitting, which resets the global seed at an unpredictable point.

**Impact:** In Ray's multi-worker setup, workers calling `reset()` concurrently will interfere with each other's random state, causing non-reproducible results. The environment already has `self._rng` from `gymnasium.Env.reset(seed=seed)` -- the global seed calls are redundant and harmful.

### 2.4 Module-Level Config Singleton

`env_config.py:165` creates `config = EnvConfig()` at module scope and exports it in `__all__`. While factory methods exist (`create_infras()`, `create_statesources()`), the singleton is still importable and used directly. In Ray, each worker gets its own copy via `fork()`, but any code that mutates this shared instance before forking would propagate that state to all workers.

TODO VP: I think this one is not relevant anymore, that config is never used -- verify this and remove if not needed anymore.

---

## 3. SOLID Principle Violations

### 3.1 Single Responsibility Principle

**`envs/building_adv.py` -- AdvBuildingGym class** handles too many concerns:
- Environment dynamics (step/reset lifecycle)
- Data variant management (`apply_data_variant`, `_resolve_episode_date`)
- Raw state collection (`_get_raw_state_values`) -- TODO VP: not anymore, but verify
- Component orchestration (infras + statesources + rewards)
- Action history tracking -- TODO VP: not anymore
- Energy accounting

The `step()` method (lines 473-587, ~115 lines) performs action execution, time advancement, state updates, energy tracking, reward computation, action history management, NaN/Inf guards, and info dict assembly. The `reset()` method (lines 361-455, ~95 lines) performs seed management, data variant selection (3 different code paths), day offset computation, state initialization, and component synchronization.

**Recommendation:** Extract `DataVariantManager`, `EnergyTracker`, and `ActionHistoryBuffer` as separate classes. Break `step()` into `_execute_actions()`, `_advance_time()`, `_compute_rewards()`, `_update_action_history()`.

**`run_train_ray.py` -- `main()` function** is 420+ lines (lines 89-532) in a single function handling argument parsing, config loading, Ray initialization, algorithm selection, callback composition, training loop, and checkpoint management.

**`ray_training/common_model_config.py`** -- single function handles resource allocation, API stack configuration, environment setup, debugging config, callback class creation, callback composition, and resource validation.

### 3.2 Open/Closed Principle

**Reward aggregation is hardcoded** in `building_adv.py:539-543` as a simple sum. Supporting weighted, conditional, or curriculum-based reward strategies requires modifying `step()`.

**Power breakdown** in `building_adv.py:518-532` is hardcoded to specific infrastructure types. Adding a new infrastructure type that contributes to energy tracking requires modifying the environment core.

**Controller interface** -- `controllers/` has no shared abstraction. `PIDController`, `PIController`, `MPCController`, and `FuzzyController` all implement `predict()` independently with slightly different signatures. Adding a new controller requires understanding each existing one's API.

### 3.3 Liskov Substitution Principle

**Inconsistent `update_state()` signatures** across component types:
- `StateSource.update_state(states=..., info=...)`
- `Infrastructure.update_state(self.state, info=...)`

Code in `building_adv.py:418` does `for sync in self.infras + self.statesources:` assuming both have `synchronise()`, but there is no shared interface or type annotation guaranteeing this.

**Controller `predict()` signatures** are inconsistent:
- `predict(obs, deterministic=True)` vs `predict(obs, deterministic=True, **kwargs)`
- All return `(action, None)` but this contract is implicit

### 3.4 Interface Segregation Principle

**Infrastructure base class** (`devices/infrastructure/base.py`) defines methods like `set_target()`, `exec_action()`, `update_state()` all with `pass` stubs. Some are meant to be overridden, others are true no-ops. The base class does not distinguish between "must override" (abstract) and "optional override" (hook).

The `info: dict | None = None` parameter appears in many base methods but is only used by some subclasses (e.g., `LinearEVCharger`), forcing all implementations to accept a parameter they may never use.

### 3.5 Dependency Inversion Principle

**`building_adv.py:22`** directly imports the global `config` object. The environment depends on a concrete global instance rather than receiving configuration through injection.

**`building_adv.py:217`** hardcodes `DataCombinator` instantiation instead of accepting it as a dependency.

**Training scripts** (`run_train_ray.py`, `run_eval_ray.py`) directly import and instantiate concrete classes (`EnvConfigManager`, `RewardConfigManager`, `DataCombinator`, `TrainingParamConfig`) with duplicated initialization logic.

---

## 4. Dead Code and Unnecessary Code

### 4.1 Unused Imports

| File | Import | Status |
|------|--------|--------|
| `envs/building_adv.py:1` | `import sys` | Never used in this file |
| `envs/building_adv.py:7` | `from gymnasium import Space` | Never used; only `spaces` is used |
| `envs/base_building_gym.py:42-47` | Commented controller imports | Dead code block |

### 4.2 TODO/FIXME Comments (27+ instances)

These indicate known technical debt that has not been converted to trackable issues:

| File | Line(s) | Content |
|------|---------|---------|
| `config/__init__.py` | 32 | Circular dependency resolution |
| `config/env_config.py` | 6 | Simplify env config |
| `config/training_param_config.py` | 13 | Separation of concerns |
| `envs/building_adv.py` | 351, 589, 601-612 | Refactoring, normalization, entropy logging |
| `rewards/energy_consumption_reward.py` | 7, 27 | Battery reward, design review |
| `rewards/temp_reward.py` | 7 | Temperature comfort reward |
| `rewards/economic_reward.py` | 12 | Review economic reward |
| `run_train_ray.py` | 324, 401, 489 | Parameter refactoring |
| `callbacks/__init__.py` | 7 | Rename callbacks |
| `data_combinator/data_combinator.py` | 51 | Use generator function |

### 4.3 Commented-Out Code

- `envs/base_building_gym.py:42-47` -- commented controller import block
- `controllers/mpc_controller.py:101` -- commented assignment for `price_list`
- `run_train_rl.py:16` -- commented stable_baselines3 import

### 4.4 Debug Print Statements

`controllers/mpc_controller.py` lines 110, 116, 121-123, 130 contain `print(f"[DEBUG] ...")` statements that should use the logging module or be removed.

### 4.5 Dead Methods and Variables

| File | Item | Issue |
|------|------|-------|
| `battery_tremblay.py:370-372` | `_get_actual_battery_charge_kW()` | Defined but never called |
| `envs/base_building_gym.py:892-898` | `render()`, `close()` | Empty `pass` stubs |
| `controllers/pi_controller.py:30` | `self.horizon = 1` | Assigned, never read |
| `controllers/mpc_controller.py:100` | `T_out_list` initial assignment | Immediately overwritten |

---

## 5. Clean Code Issues

### 5.1 God Methods

| File | Method | Lines | Issues |
|------|--------|-------|--------|
| `envs/building_adv.py` | `step()` | ~115 | 8 distinct responsibilities |
| `envs/building_adv.py` | `reset()` | ~95 | 5 distinct responsibilities |
| `envs/base_building_gym.py` | `_create_schedule()` | ~119 | 3+ levels deep nesting |
| `envs/base_building_gym.py` | `__init__()` | ~61 | 20+ attributes managed |
| `run_train_ray.py` | `main()` | ~420 | Config, init, training, checkpointing |
| `controllers/mpc_controller.py` | `predict()` | ~130 | Model construction + solving + extraction |
| `infrastructure/hp.py` | `exec_action()` | ~90 | 4-level nesting, thermal model + clipping + back-calculation |
| `ev_charger/linear_ev_charger.py` | `exec_action()` | ~70 | 4-level nesting |
| `battery_tremblay.py` | `__init__()` | ~76 | 36 parameters |
| `utils/trajectory_utils.py` | `extract_trajectory_from_infos()` | ~180 | Auto-discovery + building + flattening |

### 5.2 Magic Numbers

**Physics and device models:**

| File | Line(s) | Value | Context |
|------|---------|-------|---------|
| `hp.py` | 131 | `0.001 * control_step` | Undocumented coefficient |
| `building_heat_loss.py` | 84 | `0.001 * timestep` | Same undocumented coefficient |
| `battery_tremblay.py` | 96-100 | `3.2`, `0.009`, `0.468`, `3.529` | Tremblay model constants without source reference |
| `battery_tremblay.py` | 220 | `0.001 * Q` | Arbitrary epsilon |
| `ev_charger/linear_ev_charger.py` | 74 | `0.1` | V2G playroom threshold |
| `weather.py` | 56, 65 | `-999`, `-1998` | Sentinel values without named constants |
| `hh_consumers.py` | 114-123 | `6, 9, 17, 21` | Time-of-day profile hour thresholds |
| `solar_panel.py` | 124-125 | `6, 18` | Sunrise/sunset hour thresholds |

**Training and evaluation:**

| File | Line(s) | Value | Context |
|------|---------|-------|---------|
| `building_adv.py` | 182 | `8 * 12` | Prediction horizon, assumes 5-min steps without validation |
| `building_adv.py` | 358 | `86400` | Seconds per day, should be named constant |
| `building_adv.py` | 504, 523 | `3600.0` | kJ to kWh conversion factor, repeated |
| `common_model_config.py` | 129 | `25` | Metrics smoothing window |
| `mpc_controller.py` | 142, 166, 170 | `0.001`, `0.05`, `0.01/H` | Scale, smoothness penalty, regularization |
| `fuzzy_controller.py` | 40-41, 112-113 | `1.5`, `0.75`, `15`, `-5` | Tuning constants |

**Controllers:**

| File | Line(s) | Value | Context |
|------|---------|-------|---------|
| `mpc_controller.py` | 178 | `/home/iai/ii6824/.local/bin/ipopt` | Hardcoded absolute path to solver binary |
| `base_building_gym.py` | 308-310 | `28`, `32` | Temperature bounds for simple schedule |
| `base_building_gym.py` | 454-459 | `0.25`, `0.50`, `0.75` | Time-of-use tariff prices |

### 5.3 Error Suppression

`utils/json_encoder.py` (lines 18, 35, 72, 77, 87) uses bare `except Exception: pass` and `except Exception: return {}` blocks. These silently swallow all errors, making debugging impossible when serialization fails.

### 5.4 Fragile State Discovery

`building_adv.py:303-312` uses `dir()` and `endswith("_raw")` to discover raw attributes on components at init time. This is magic-string coupling to implementation details -- adding or renaming a `_raw` attribute in any component silently changes environment behavior.

---

## 6. Architectural Weaknesses

### 6.1 Fragile Component Orchestration

The `step()`/`reset()` lifecycle depends on precise call ordering:
1. `synchronise()` called AFTER iteration increment but BEFORE `update_state()`
2. Comments in the code acknowledge a previous bug related to iteration lag

There is no mechanism to enforce or validate this ordering. A new component that depends on a different call order will silently produce incorrect results.

### 6.2 Shared State via Mutable Dict

All components read from and write to a shared `self.state` OrderedDict. There is no ownership model -- any infrastructure or state source can overwrite any key. This makes it difficult to reason about state changes and creates implicit coupling between components that share state keys (e.g., both `HP` and `BuildingHeatLoss` write to `temp_in_norm`).

### 6.3 RngService Singleton

`utils/seed_provider.py:33` implements a singleton pattern with `ClassVar`. The `get()` method auto-initializes if the instance is `None`, which could cause different seeds across Ray workers if they auto-initialize independently.

### 6.4 Config/Component Coupling

`env_config.py` imports all concrete infrastructure, statesource, and reward classes at module level (lines 11-21). This means importing the config module forces loading the entire device tree, even if only the dataclass fields are needed. This also causes the circular dependency documented in section 2.3.

### 6.5 No Controller Interface

The four controllers (`PIDController`, `PIController`, `MPCController`, `FuzzyController`) have no shared abstract base class. Each independently implements `predict()` with slightly different signatures and return conventions. Adding a new controller or swapping controllers requires knowledge of each implementation's API quirks.

---

## 7. Code Duplication

### 7.2 State Space Registration

The `sim_hour` observation space is registered identically in 4 different state sources:

- `statesources/outer/weather.py:116-118`
- `statesources/outer/energy_price.py:49-51`
- `statesources/outer/desired_user_energy_need.py:47-49`
- `statesources/outer/inside_temperature.py:58-60`

All use: `Box(low=np.full((1,), 0, dtype=np.float32), high=np.full((1,), 24, dtype=np.float32))`

Temperature state spaces (`temp_in_norm`, `temp_out_norm`) are registered identically in `hp.py:71-73` and `building_heat_loss.py:60-62`.

### 7.3 Synthetic Profile Generation

`solar_panel.py:_synthetic_irradiance()` and `hh_consumers.py:_synthetic_consumption()` implement nearly identical hour-based piecewise profile logic with the same structure (hour thresholds, sin interpolation).

### 7.4 Training Script Initialization

`run_train_ray.py` and `run_eval_ray.py` duplicate:
- Config loading logic
- Data combinator initialization
- Warning filter setup
- Seed/RNG initialization

No shared utility extracts these common patterns.

### 7.5 Env Runner Traversal in Callbacks

`callbacks/data_schedule_callback.py` and `callbacks/reward_switch_callback.py` both implement the same pattern to traverse Ray env runners and unwrap environments:

```python
def apply(env_runner):
    vec_env = getattr(env_runner, "env", None)
    sync_vec = getattr(vec_env, "env", vec_env)
    for sub_env in getattr(sync_vec, "envs", []):
        unwrapped = sub_env.unwrapped
        # ...
```

This fragile unwrapping logic should be extracted to a shared utility.

---

## 8. Meaningless or Unused Parameters

### 8.1 Consistently Unused Across Interface

| Parameter | Where | Issue |
|-----------|-------|-------|
| `deterministic` | All 4 controllers' `predict()` | Accepted but never read. Documented as "Unused" or "Ignored". |
| `info: dict \| None` | `Infrastructure.exec_action()`, `update_state()` | Part of base interface but unused by most subclasses (only `LinearEVCharger` uses it). |
| `render_mode` | `AdvBuildingGym.__init__()` | Stored but never used. Exists for Gymnasium compatibility only. |
| `ds_path` | `BuildingHeatLoss.__init__()` | Accepted but never used (documented as "not used for this datasource"). |

### 8.2 Ambiguous or Redundant Parameters

| Parameter | Where | Issue |
|-----------|-------|-------|
| `training` | `AdvBuildingGym.__init__()` | Passed to infras/statesources but never inspected by the environment itself. |
| `action_history_length` | `AdvBuildingGym.__init__()` | Nullable with fallback to `env_config.ACTION_HISTORY_LENGTH` -- inconsistent source of truth. |
| `step` | `TrajectoryCollector.on_step()` | Parameter accepted but never used; step count maintained by list length. |
| `control_step` | `SolarPanel.__init__()` | Stored as instance variable but never referenced. |
| `history_length` | `BatteryLinear.__init__()` | Only affects state space shape, not core model logic -- could be extracted to env config. |
| `max_charge_time_hrs` | `LinearEVCharger.__init__()` | Only used for normalization, not core charging simulation. |

---

## 9. Recommendations by Priority

### Priority 1 -- Safety and Correctness

2. **Remove global seed mutations.** Delete `np.random.seed()` and `random.seed()` calls in `building_adv.py:370-371` and `base_building_gym.py:421`. Use `self.np_random` (Gymnasium's per-instance RNG) exclusively.

### Priority 2 -- Architecture

4. **Resolve circular dependency.** Move component imports in `env_config.py` to factory methods (lazy) rather than module level. Remove the `__getattr__` workaround in `config/__init__.py`.

5. **Extract shared initialization logic** from `run_train_ray.py` and `run_eval_ray.py` into a `setup_session()` utility (config loading, data combinator init, warning filters, seed setup).

7. **Extract env runner traversal** from callbacks into a shared utility function to eliminate the duplicated unwrapping pattern.

### Priority 3 -- Clean Code

8. **Break up god methods:**
   - `building_adv.py:step()` -> `_execute_actions()`, `_advance_time()`, `_compute_rewards()`, `_update_action_history()`
   - `building_adv.py:reset()` -> `_select_data_variant()`, `_compute_day_offset()`, `_init_state()`
   - `run_train_ray.py:main()` -> extract config loading, algorithm setup, and training loop into separate functions

9. **Replace magic numbers with named constants.** At minimum:
   - `SECONDS_PER_HOUR = 3600`, `SECONDS_PER_DAY = 86400`
   - `WEATHER_SENTINEL = -999`
   - Tremblay model constants should reference their source paper

11. **Convert TODO comments to issues.** Remove inline TODOs that have been present for multiple commits. Track them as GitHub issues with proper context.

### Priority 4 -- Cleanup

12. **Remove dead code:** unused imports (`sys`, `Space` in `building_adv.py`), commented code blocks, empty method stubs (`render()`, `close()` in `base_building_gym.py`), unused methods (`_get_actual_battery_charge_kW` in `battery_tremblay.py`).

13. **Replace debug prints** in `mpc_controller.py` with `logger.debug()` calls.

14. **Fix error suppression** in `json_encoder.py` -- replace bare `except Exception: pass` with specific exception types and logging.

15. **Remove unused parameters** where safe (`ds_path` on `BuildingHeatLoss`, `control_step` on `SolarPanel`, `step` on `TrajectoryCollector.on_step()`).
