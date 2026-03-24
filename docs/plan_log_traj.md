# Trajectory Logging & Metrics Tracking — Implementation Plan

TODO VP: check this document, adjust its state to the current repo state


## Motivation

Currently, evaluation-time data collection is limited:
- **`episode_callbacks.py`** (training): Saves flat observations, clipped actions, and raw policy actions per episode as JSON. However, observations are opaque flat arrays — there's no mapping back to named state variables (e.g., `temp_in_norm`, `E_price`, `solar_irradiance`).
- **`run_eval_ray.py`** (evaluation): Collects only per-episode aggregates (total reward, reward rate, length). Step-level trajectories are discarded (`step_info` is not preserved — there's even a TODO for this).
- **`run_evaluation.py`** (SB3 eval): Records per-step data into a DataFrame, but this is for `BaseBuildingGym` only, not `AdvBuildingGym`. Out of scope for this plan.

**What's missing:** A structured, per-step trajectory log during evaluation that tracks every state variable by name, every action component by name, per-reward-function breakdowns, and energy metrics — all in a format ready for analysis and plotting. 
Store the data variant along the trajectory log as well.

---

## RLlib Built-in Options Assessment

Before designing a custom solution, here is what RLlib (Ray 2.x) provides natively and why it is insufficient for our needs:

### SingleAgentEpisode API (new API stack)

The `SingleAgentEpisode` object stores full trajectory data internally via `InfiniteLookbackBuffer` and exposes:
- `episode.get_observations(indices)` — all observations (one more than actions due to reset obs)
- `episode.get_actions(indices)` — raw policy outputs
- `episode.get_rewards(indices)` — scalar rewards per step
- `episode.get_infos(indices)` — environment info dicts per step
- `episode.get_extra_model_outputs(key, indices)` — e.g., `"action_logp"`

All getters support single int, list, slice, or `None` (all items). Negative indices work. Data is available in both `on_episode_step` and `on_episode_end` callbacks.

**Key insight:** `episode.get_infos()` returns the full list of `info` dicts from every step. When `log_full_info=True` (evaluation mode), `AdvBuildingGym` puts a deep copy of the named state dict into `info["state"]`, so we can reconstruct full named trajectories from the episode object at `on_episode_end` — no per-step callback needed. During training, this deep copy is skipped to save memory.

### Episode data types and serialization (detailed)

The return types of `episode.get_*()` depend on whether `to_numpy()` / `finalize()` has been called. **Crucially, `on_episode_end` fires BEFORE numpy'ization**, so we get the easier-to-handle list-based format:

| Method (with `indices=None`) | Return type in `on_episode_end` | Length |
|---|---|---|
| `get_observations()` | `list[dict[str, np.ndarray]]` — list of observation dicts | T+1 (includes reset obs) |
| `get_actions()` | `list[np.ndarray]` — list of flat action arrays | T |
| `get_rewards()` | `list[float]` (may be `np.float32`) | T |
| `get_infos()` | `list[dict]` — raw info dicts from env | T+1 (infos are **never** numpy'ized) |

With Dict observation space (our case), each observation is `{"temp_in_norm": np.array([0.5]), "E_price": np.array([0.12]), ...}`.

**Can these be directly JSON-serialized?** Almost — the only blocker is `np.ndarray` and `np.float32` leaf values. Our existing `CustomJSONEncoder` already handles these (`np.ndarray` → `.tolist()`, `np.generic` → `.item()`). So `json.dump(episode.get_observations(), f, cls=CustomJSONEncoder)` works.

**Why we still need a transformation step:** While raw serialization works, it produces a **list-of-dicts** format:
```json
[{"temp_in_norm": [0.5], "E_price": [0.12]}, {"temp_in_norm": [0.51], "E_price": [0.11]}, ...]
```
This is verbose and inconvenient for analysis. What we want is the **columnar** (dict-of-lists) format:
```json
{"state": {"temp_in_norm": [0.5, 0.51, ...], "E_price": [0.12, 0.11, ...]}, "action": {...}, ...}
```
The transformation from list-of-dicts → nested columnar dict is straightforward (transpose + flatten multi-dim arrays into named columns, grouped under `"state"`, `"action"`, `"reward_breakdown"` sub-dicts). This is the core of the `extract_trajectory_from_infos()` utility.

**After `to_numpy()` (NOT our case, but for reference):** Observations become a single dict-of-arrays `{"temp_in_norm": np.array([[0.5], [0.51], ...]), ...}` which is closer to columnar format but still needs numpy→list conversion and key renaming. Infos are never affected.

**`episode.get_state()` / `from_state()`:** These exist for msgpack-based binary serialization (used by RLlib internals for checkpoint/offline data). Not JSON-compatible, not useful for our analysis needs.

**Implication for the plan:** The callback-based approach (Option B) is simpler than initially thought. At `on_episode_end`, we can extract all trajectory data directly from `episode.get_infos()` (which contains `info["state"]`, `info["action"]`, `info["reward_breakdown"]`, `info["cum_E_kWh"]` at every step — assuming `log_full_info=True` on evaluation EnvRunners) and transform it to columnar format. No per-step accumulation needed — the episode already stores everything.

### Callback Execution Model (new API stack)

**Episode-level callbacks run exclusively on EnvRunner workers.** They do NOT run on Learner workers or on the Algorithm (driver) process. RLlib currently has no callback hooks on Learner actors at all.

The execution split across actor types:

| Actor | Callbacks |
|-------|-----------|
| **Algorithm (driver)** | `on_algorithm_init`, `on_train_result`, `on_evaluate_start`, `on_evaluate_end`, `on_checkpoint_loaded`, `on_env_runners_recreated` |
| **EnvRunner workers** (training AND evaluation) | `on_environment_created`, `on_episode_created`, `on_episode_start`, `on_episode_step`, `on_episode_end`, `on_sample_end` |
| **Learner workers** | **None** — no callback hooks exist on Learners as of Ray 2.53 |

**Full lifecycle of a single training iteration:**
```
Algorithm.train()                              [Algorithm/driver]
  |
  |-- Training sampling:
  |     EnvRunner[0..N].sample()               [training EnvRunners, parallel]
  |       on_episode_created                   [EnvRunner]  after Episode() created
  |       on_episode_start                     [EnvRunner]  after env.reset()
  |       on_episode_step                      [EnvRunner]  after each env.step() (repeats)
  |       on_episode_end                       [EnvRunner]  when terminated/truncated
  |       (loop if sample budget not exhausted)
  |       on_sample_end                        [EnvRunner]  after sample() returns batch
  |
  |-- Learning/updating:
  |     Learner[0..K].update()                 [Learner workers]
  |       (NO callbacks)
  |
  |-- on_train_result                          [Algorithm/driver]
  |
  |-- Evaluation (if evaluation_interval triggers):
        on_evaluate_start                      [Algorithm/driver]
        EvalEnvRunner[0..M].sample()           [eval EnvRunners, parallel]
          on_episode_created                   [eval EnvRunner]
          on_episode_start                     [eval EnvRunner]
          on_episode_step                      [eval EnvRunner]  (repeats)
          on_episode_end                       [eval EnvRunner]
          on_sample_end                        [eval EnvRunner]
        on_evaluate_end                        [Algorithm/driver]
```

**Key takeaways:**
- **`on_episode_end` fires on BOTH training and evaluation EnvRunners.** Evaluation EnvRunners run the same `SingleAgentEnvRunner._sample()` loop, which triggers the same callbacks.
- **Each EnvRunner gets its own callback instance** (`cls()` called in `SingleAgentEnvRunner.__init__`). With 4 training + 2 eval EnvRunners, there are 6 separate callback instances with no shared state.
- **`on_episode_step` fires on every step** across all EnvRunners — avoid heavy I/O here. Collect everything at `on_episode_end` instead, since the full trajectory is available via `episode.get_*()`.

### Training vs Evaluation Detection

RLlib provides `env_runner.config.in_evaluation` (boolean) inside callbacks:

```python
def on_episode_end(self, *, episode, env_runner, metrics_logger, env, **kwargs):
    if env_runner.config.in_evaluation:
        # Evaluation episode — do expensive trajectory logging
        self._save_trajectory(episode, env)
    else:
        # Training episode — log only scalar aggregates
        metrics_logger.log_value("reward_rate", reward_rate, reduce="mean")
```

**`in_evaluation` is reliable:** It is a config property set at EnvRunner construction time (not a transient flag). On the new API stack, `env_runner` is always passed as `self` from the EnvRunner — it is never `None` in practice (despite the `Optional` type hint). A defensive `if env_runner is not None and env_runner.config.in_evaluation` guard is safe but technically unnecessary.

**There is no way to register separate callback classes for training vs evaluation.** The same `callbacks_class` is used for all EnvRunners. Branching inside the callback via `in_evaluation` is the official pattern.

### RLlib Offline Data / Output Writers

RLlib has a built-in offline data recording system (`config.offline_data(output=...)`) that writes Parquet files with serialised `SingleAgentEpisode` objects. This is designed for **offline RL** (recording data for later training with CQL/MARWIL), not for structured trajectory analysis. The legacy `JsonWriter` exists but writes raw episode chunks, not per-step named DataFrames. Neither is suitable for our use case.

### MetricsLogger

`metrics_logger.log_value()` supports reductions: `mean`, `min`, `max`, `sum`, `ema`, `item`, `item_series`, `None`. While `reduce=None` could accumulate per-step values, it is designed for **scalar aggregates** across episodes, not for storing entire trajectory arrays. Direct file I/O is the right approach for full trajectories.

### Conclusion

**RLlib does not have a built-in "export structured trajectory to named-column JSON" feature.** The episode data *can* be serialized with our existing `CustomJSONEncoder`, but it needs a list-of-dicts → columnar transformation for usable output.

**Since `on_episode_end` already gives access to the full trajectory via `episode.get_infos()`,** the callback-based approach is the natural primary solution for both training-time eval and standalone eval. The same `extract_trajectory_from_infos()` utility serves both:

1. **Callback-based logging** (Option B — primary for training-time eval): Extract trajectory from `episode.get_infos()` at `on_episode_end`, gated by `env_runner.config.in_evaluation` + configurable flag.
2. **Standalone eval script** (Option A): Use the same extraction utility, but fed from step-by-step `info` dicts collected in the eval loop. Needed because `run_eval_ray.py` does not use RLlib callbacks — it runs its own `env.step()` loop.

---

## Current Architecture Summary

### Data flow in `AdvBuildingGym.step()`
1. Flat action → clipped → converted to `Dict[str, np.ndarray]` via `_flat_action_to_dict()`
2. Each infrastructure: `exec_action(action_dict, state)` then `update_state(state)`
3. Each statesource: `update_state(state)`
4. `state["prev_action"]` updated
5. Energy accumulated: `cum_E_kWh`
6. Reward computed: sum over `reward_funcs`
7. `info` dict returned with: `action` (dict format), `reward`, `reward_breakdown`, `cum_E_kWh`, and conditionally `state` (deep copy, only when `log_full_info=True`)

### Key observation
The `info["state"]` contains a full deep copy of all named state variables at each step — but only when `log_full_info=True` (evaluation mode). During training, this deep copy is skipped to save memory. For trajectory logging, the flag must be enabled on evaluation EnvRunners.

---

## Decisions (Resolved)

1. **Save format:** JSON — one JSON file per episode and hdf.
2. **Training-time logging:** Configurable on/off. When on, trajectory logging runs during training evaluation episodes (gated by `env_runner.config.in_evaluation` + a config flag). Off by default.
3. **Plotting utilities:** Separate scope — not part of this plan.
4. **Storage structure:** One file per episode. All episodes from a single eval run go under a dedicated directory: `<output_base>/<run_id>/episode_<N>_trajectory.json`.

---

## Plan

### Core utility: `extract_trajectory_from_infos()`

The central piece shared by both options. Converts a list of `info` dicts (from `episode.get_infos()` or collected during an eval loop) into the columnar JSON format.

**Where:** `adv_building_gym/utils/trajectory_utils.py`

```python
def extract_trajectory_from_infos(
    infos: list[dict],
    initial_info: dict | None = None,
    control_step: int = 300,
    state_keys: list[str] | None = None,
    action_keys: list[str] | None = None,
) -> dict:
    """Convert a list of per-step info dicts into columnar trajectory data.

    Args:
        infos: List of info dicts from AdvBuildingGym.step() calls (length T).
            Each dict is expected to contain:
              - info["state"]: dict[str, np.ndarray] — named state variables
              - info["action"]: dict[str, np.ndarray] — named action components (clipped). This is the only action representation from info (no separate clipped_action).
              - info["reward"]: float — total reward
              - info["reward_breakdown"]: dict[str, float] — per-reward-function values
              - info["cum_E_kWh"]: float — cumulative energy
        initial_info: Optional info dict from reset() (contains initial state).
            When provided, the initial/reset state is prepended as step 0
            (with zero actions and zero reward), so the trajectory includes
            the starting conditions. Callers must pass this explicitly —
            the function never auto-strips a T+1 list.
        state_keys: State keys to extract. Auto-discovered from first info if None.
        action_keys: Action keys to extract. Auto-discovered from first info if None.

    Transforms list-of-dicts:
      [{"state": {"temp_in": [0.5], "hist": [0.1, 0.2]}, ...}, ...]
    Into nested columnar dict:
      {"state": {"temp_in": [0.5, 0.51, ...], "hist": [[0.1, 0.2], [0.15, 0.25], ...]}, ...}

    States and actions use the original key names from the env (no _0/_1
    suffix splitting). Scalar values (size 1) become flat lists of floats;
    vector values (size > 1) are preserved as lists of lists.
    Per-reward breakdowns are grouped under "reward_breakdown".
    Top-level scalar columns ("step", "reward", "cum_E_kWh", "step_power_kW")
    remain at the top level.

    Also computes derived columns:
      - step_power_kW: per-step power derived as cum_E_kWh[t] - cum_E_kWh[t-1],
        converted from kWh back to kW via (delta_kWh / control_step_hours).
        For step 0 (or when initial_info is absent), uses cum_E_kWh[0] directly.

    Returns:
        dict with columnar trajectory data, ready for JSON serialization.
        Keys include "step", "state" (nested dict keyed by original state
        names — scalars as flat lists, vectors preserved as lists of lists),
        "action" (nested dict, same convention), "reward",
        "reward_breakdown" (nested dict), "cum_E_kWh", "step_power_kW".
    """
```

**What gets logged per step:**
| Category | Keys | Source |
|----------|------|--------|
| **States** | Nested under `"state"` dict, keyed by original state name (e.g., `state.temp_in_norm`). Scalars → flat list of floats; vectors → list of lists preserving dimensionality. | `info["state"]` |
| **Actions (clipped, dict)** | Nested under `"action"` dict, keyed by original action name (e.g., `action.HP_action`). Same scalar/vector convention as states. | `info["action"]` (dict format). This is the only action representation logged from `info`. |
| **Actions (raw policy)** | Flat raw action from policy output | `episode.get_actions()` or eval loop. Added by caller, not by `extract_trajectory_from_infos()`. |
| **Reward** | Total reward | `info["reward"]` |
| **Per-reward breakdown** | Nested under `"reward_breakdown"` dict, one entry per reward function (e.g., `reward_breakdown.temp_reward`) | `info["reward_breakdown"]` |
| **Energy** | `cum_E_kWh`, `step_power_kW` | `info["cum_E_kWh"]` (raw cumulative values stored as-is). `step_power_kW` derived inside `extract_trajectory_from_infos()` as `(cum_E_kWh[t] - cum_E_kWh[t-1]) / control_step_hours`. For step 0 (or when no `initial_info`), uses `cum_E_kWh[0] / control_step_hours`. |
| **Metadata** | `step`, `episode_id`, `seed` | From caller |

### Option A: TrajectoryCollector (standalone, for eval scripts)

Needed because `run_eval_ray.py` runs its own `env.step()` loop without RLlib callbacks. Wraps `extract_trajectory_from_infos()` with step-by-step accumulation and JSON output.

**Where:** `adv_building_gym/utils/trajectory_collector.py`

**Interface:**
```python
class TrajectoryCollector:
    def __init__(self, env: AdvBuildingGym):
        """Extract state keys, action keys, reward function names from env."""

    def on_reset(self, info: dict):
        """Store the reset info dict (initial conditions). Called after env.reset()."""

    def on_step(self, step: int, obs, action, reward, info, raw_policy_action=None):
        """Accumulate one timestep's info dict + raw policy action."""

    def on_episode_end(self, episode_id, seed, metadata=None):
        """Finalize episode, store metadata."""

    def to_dict(self) -> dict:
        """Calls extract_trajectory_from_infos() on accumulated infos
        (passing initial_info from on_reset()),
        adds raw policy actions, metadata, summary. Returns JSON-serializable dict."""

    def save_json(self, filepath: str):
        """Save to_dict() output to JSON file."""

    def reset(self):
        """Clear accumulated data (including initial_info) for next episode."""
```

**Integration in `run_eval_ray.py`:**
```python
from adv_building_gym.utils import TrajectoryCollector

collector = TrajectoryCollector(env)

# Create run directory
run_id = f"eval_{timestamp}"
run_dir = os.path.join(output_dir, run_id)
os.makedirs(run_dir, exist_ok=True)

for ep in range(num_episodes):
    obs, reset_info = env.reset(seed=seed + ep)
    collector.reset()
    collector.on_reset(reset_info)
    done = False
    step = 0

    while not done:
        raw_action = algo.compute_single_action(obs, explore=False)
        next_obs, reward, terminated, truncated, step_info = env.step(raw_action)
        collector.on_step(step, obs, raw_action, reward, step_info)
        obs = next_obs
        step += 1
        done = terminated or truncated

    collector.on_episode_end(episode_id=ep, seed=seed + ep)
    collector.save_json(f"{run_dir}/episode_{ep}_trajectory.json")
```

### Option B: Callback-based logging (training-time, configurable)

Enhance `episode_callbacks.py` to optionally save full named trajectories during training evaluation episodes. This is simpler than initially expected because `episode.get_infos()` already gives us all the info dicts at once — no per-step accumulation needed.

**Why `episode.get_infos()` works directly:**
- `on_episode_end` fires **before** `to_numpy()` / `finalize()`, so `get_infos()` returns a plain `list[dict]`
- Infos are **never numpy'ized** even after finalization (by design — they're heterogeneous dicts)
- Each info dict already contains `info["state"]` (deep copy of named state, when `log_full_info=True`), `info["action"]` (dict format), `info["reward_breakdown"]`, `info["cum_E_kWh"]`
- We also get raw policy actions from `episode.get_actions()` (returns `list[np.ndarray]`)

**So the callback trajectory logging is simply:**
```python
def on_episode_end(self, *, episode, env_runner, metrics_logger, env, **kwargs):
    # ... existing scalar metrics logic (always runs) ...

    if log_trajectories and env_runner.config.in_evaluation:
        infos = episode.get_infos()        # list[dict], length T+1
        raw_actions = episode.get_actions() # list[np.ndarray], length T

        # infos[0] is from reset() (initial conditions, no action/reward).
        # infos[1:] are from step() calls.
        initial_info = infos[0]   # reset info — contains seed, initial state
        step_infos = infos[1:]    # length T

        trajectory = extract_trajectory_from_infos(step_infos, initial_info=initial_info)
        # Add raw policy actions from episode object
        # initial_info row has no raw action, so prepend None/zeros
        trajectory["raw_policy_actions"] = [[0.0] * len(raw_actions[0])] + [a.tolist() for a in raw_actions]

        dump = {
            "episode_id": episode.id_,
            "seed": initial_info.get("seed"),
            "length": len(episode),
            "summary": { ... },
            "trajectory": trajectory,
        }
        os.makedirs(ep_metrics_dir, exist_ok=True)
        save_path = f"{ep_metrics_dir}/episode_{episode.id_}_trajectory.json"
        with open(save_path, "w") as f:
            json.dump(dump, f, cls=CustomJSONEncoder, indent=4)
```

**Configuration:** A `log_trajectories` boolean flag passed to the callback factory:

```python
def create_on_episode_end_callback(
    env_id: str,
    rewards: List,
    metrics_base_dir: str = "ep_metrics",
    exec_date: Optional[datetime.datetime] = None,
    log_trajectories: bool = False,  # NEW: configurable trajectory logging
):
```

**Behaviour:**
- `log_trajectories=False` (default): No change from current behaviour — only scalar metrics
- `log_trajectories=True`:
  - If `env_runner.config.in_evaluation`: Extract full trajectory from `episode.get_infos()` + `episode.get_actions()`, save as JSON alongside the existing metrics JSON
  - If training episode: Skip trajectory logging (only scalar metrics) — avoids I/O overhead during training

### Per-reward breakdown and info cleanup

To log individual reward components, extend `AdvBuildingGym.step()` to include per-reward values in `info`:

```python
# In step(), replace the existing reward loop:
reward: float = 0
reward_breakdown = {}
for rew_f in self.reward_funcs:
    rew_val = float(np.asarray(rew_f.get_reward(action, self.state)).item())
    reward_breakdown[rew_f.name] = rew_val
    reward += rew_val

# Add to info:
info["reward_breakdown"] = reward_breakdown
```

Also remove `info["clipped_action"]` (the flat clipped action array) from `step()`. It is redundant with `info["action"]` which provides the same clipped values in the more useful dict format. Only `info["action"]` is logged in trajectories.

This is a small, low-overhead change that makes reward debugging much easier.

### Conditional state in info (training memory optimisation)

During training, RLlib's `InfiniteLookbackBuffer` stores every `info` dict from every step. The deep copy of the full state dict (`info["state"]`) is expensive in memory when accumulated across thousands of training episodes. To avoid this:

```python
# In AdvBuildingGym:
self.log_full_info: bool = False  # Set to True for evaluation

# In step():
info = {
    "action": action,
    "reward": reward,
    "reward_breakdown": reward_breakdown,
    "cum_E_kWh": self.cum_E_kWh,
}
if self.log_full_info:
    info["state"] = {k: np.array(v, copy=True) for k, v in self.state.items()}
```

The flag is set to `True` by the eval script (before the eval loop) and by the callback setup (on evaluation EnvRunners via `on_environment_created`). Training EnvRunners leave it `False`, avoiding the per-step deep copy overhead.

---

## Implementation Steps

### Step 1: Extend `info` dicts in `building_adv.py`
- **File:** `adv_building_gym/envs/building_adv.py`
- **Changes:**
  1. In `step()`, compute rewards individually into a `reward_breakdown` dict and add `info["reward_breakdown"]`. Remove `info["clipped_action"]` (redundant — same data as `info["action"]` in dict format).
  2. In `reset()`, add `info["state"]` with a deep copy of the initial state (same pattern as `step()`), so the reset info captures starting conditions. The seed is already present (`info = {"seed": seed}`).
  3. Make `info["state"]` in `step()` conditional on evaluation mode: add a `self.log_full_info: bool` flag (default `False`). When `False`, `step()` omits the expensive deep copy of state into info. When `True` (set during evaluation), the full `info["state"]` is included. This avoids memory pressure from RLlib's `InfiniteLookbackBuffer` storing deep-copied state dicts for every training step across all episodes.
- **Impact:** Minimal — restructuring existing computation. The `log_full_info` flag is toggled by the eval script and the callback setup.

### Step 2: Create trajectory extraction utility
- **File:** `adv_building_gym/utils/trajectory_utils.py`
- **Function:** `extract_trajectory_from_infos(infos, initial_info, state_keys, action_keys) -> dict`
- **Purpose:** Shared logic for converting a list of step `info` dicts into a structured columnar dict with named keys. When `initial_info` (from `reset()`) is provided, the initial state is prepended as step 0 with zero actions/reward — preserving starting conditions. Also computes `step_power_kW` from consecutive `cum_E_kWh` differences. Used by both `TrajectoryCollector` (Option A) and the callback (Option B).
- **Exports:** Add to `adv_building_gym/utils/__init__.py`

### Step 3: Extend episode callbacks with configurable trajectory logging (Option B)
- **File:** `adv_building_gym/callbacks/episode_callbacks.py`
- **Changes:**
  - Add `log_trajectories: bool = False` parameter to `create_on_episode_end_callback()`
  - Inside `on_episode_end`, when `log_trajectories=True` and `env_runner.config.in_evaluation`:
    - Split `episode.get_infos()` into `initial_info = infos[0]` (reset) and `step_infos = infos[1:]` (steps)
    - Call `extract_trajectory_from_infos(step_infos, initial_info=initial_info)`
    - Add raw policy actions from `episode.get_actions()` (with a zero-padding entry for the initial-conditions row)
    - Ensure output directory exists with `os.makedirs(ep_metrics_dir, exist_ok=True)` before writing
    - Save JSON to the episode metrics directory
  - Training episodes: unchanged (only scalar metrics)
- **Training script integration:** Pass `log_trajectories=True/False` from training config or CLI flag. When `log_trajectories=True`, the training script also sets `env.log_full_info = True` on evaluation EnvRunners (via `on_environment_created` callback or env config) so that `step()` includes `info["state"]`.

### Step 4: Create `TrajectoryCollector` (Option A)
- **File:** `adv_building_gym/utils/trajectory_collector.py`
- **Class:** `TrajectoryCollector` — wraps the extraction utility, manages per-episode accumulation, JSON serialisation
- **Purpose:** For `run_eval_ray.py` which uses its own `env.step()` loop without RLlib callbacks
- **Exports:** Add to `adv_building_gym/utils/__init__.py`
- **Dependencies:** `numpy`, `json` (no new deps)

### Step 5: Integrate into `run_eval_ray.py`
- **File:** `run_eval_ray.py`
- **Changes:**
  - Set `env.log_full_info = True` before the eval loop (so `step()` includes `info["state"]`)
  - Instantiate `TrajectoryCollector`
  - Call `collector.on_reset(reset_info)` after `env.reset()`
  - Call `collector.on_step()` in eval loop
  - Save JSON per episode under `<output_dir>/<run_id>/episode_<N>_trajectory.json`
  - Save existing summary JSON/CSV alongside in the same run directory
- **New CLI flag:** `--log-trajectories` (default: True for eval)

### ~~Step 6: Integrate into `run_evaluation.py`~~ — Removed
`BaseBuildingGym` integration is out of scope for this plan.

---

## Output Format

### Directory structure (eval run)
```
eval_results/
  eval_PPO_20260217_143000/          # One directory per eval run
    summary.json                      # Per-episode aggregates (existing format)
    episodes.csv                      # Per-episode stats table (existing format)
    episode_0_trajectory.json         # Full trajectory for episode 0
    episode_1_trajectory.json         # Full trajectory for episode 1
    ...
```

### Directory structure (training callback, when enabled)
```
ep_metrics/
  20260217_143000/                    # Existing metrics directory
    episode_abc123_metrics.json       # Existing scalar metrics (always saved)
    episode_abc123_trajectory.json    # Full trajectory (only when log_trajectories=True + eval)
    ...
```

### JSON trajectory file structure
```json
{
  "version": 1,
  "episode_id": 0,
  "seed": 42,
  "length": 288,
  "metadata": {
    "env_id": "AdvBuildingGym_config1",
    "env_config_name": "test1",
    "checkpoint_path": "models/test1/ray/ppo/best_model_ep500_..."
  },
  "summary": {
    "achieved_reward": 245.6,
    "max_achievable_reward": 288.0,
    "reward_rate": 0.8528,
    "cum_E_kWh": 12.34
  },
  "trajectory": {
    "step": [0, 1, 2, ...],
    "state": {
      "temp_in_norm": [0.5, 0.51, 0.52, ...],
      "temp_out_norm": [0.3, 0.3, 0.31, ...],
      "E_price": [0.12, 0.12, 0.11, ...],
      "solar_irradiance": [0.0, 0.0, 0.05, ...],
      "prev_action": [[0.0, 0.0], [0.3, 0.7], [0.4, 0.8], ...]
    },
    "action": {
      "HP_action": [[0.3, 0.7], [0.4, 0.8], [0.5, 0.9], ...],
      "solar_action": [-0.0, -0.0, -0.05, ...]
    },
    "raw_policy_action_0": [0.32, 0.41, 0.53, ...],
    "raw_policy_action_1": [0.68, 0.79, 0.88, ...],
    "reward": [0.9, 0.85, 0.88, ...],
    "reward_breakdown": {
      "temp_reward": [0.8, 0.75, 0.78, ...],
      "economic_reward": [0.1, 0.1, 0.1, ...]
    },
    "cum_E_kWh": [0.04, 0.08, 0.12, ...],
    "step_power_kW": [0.5, 0.5, 0.6, ...]
  }
}
```

**Row 0 contains initial conditions** from `reset()`: state values reflect the starting state, actions and rewards are zero, `cum_E_kWh` is 0.0. Subsequent rows (1..T) correspond to `step()` calls.

The columnar format (arrays per key) is compact, JSON-native, and convertible to a DataFrame for analysis. Nested dicts (`state`, `action`, `reward_breakdown`) are flattened with their group as prefix:
```python
import pandas as pd
traj = trajectory_data["trajectory"]
# Flatten nested dicts into prefixed columns
flat = {}
for key, val in traj.items():
    if isinstance(val, dict):
        for sub_key, sub_val in val.items():
            flat[f"{key}/{sub_key}"] = sub_val
    else:
        flat[key] = val
df = pd.DataFrame(flat)
```

---

## References

### RLlib Documentation (Ray 2.x, new API stack)
- [SingleAgentEpisode API Reference](https://docs.ray.io/en/latest/rllib/package_ref/env/env/ray.rllib.env.single_agent_episode.SingleAgentEpisode.html)
- [SingleAgentEpisode Guide (Episodes concept)](https://docs.ray.io/en/latest/rllib/single-agent-episode.html)
- [SingleAgentEpisode Source Code](https://docs.ray.io/en/latest/_modules/ray/rllib/env/single_agent_episode.html)
- [get_observations()](https://docs.ray.io/en/latest/rllib/package_ref/env/env/ray.rllib.env.single_agent_episode.SingleAgentEpisode.get_observations.html)
- [get_infos()](https://docs.ray.io/en/master/rllib/package_ref/env/env/ray.rllib.env.single_agent_episode.SingleAgentEpisode.get_infos.html)
- [RLlib Callbacks Guide](https://docs.ray.io/en/latest/rllib/rllib-callback.html)
- [RLlibCallback API Reference](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.callbacks.callbacks.RLlibCallback.html)
- [RLlibCallback Source Code](https://docs.ray.io/en/latest/_modules/ray/rllib/callbacks/callbacks.html)
- [on_episode_end callback](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.callbacks.callbacks.RLlibCallback.on_episode_end.html)
- [on_episode_step callback](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.callbacks.callbacks.RLlibCallback.on_episode_step.html)
- [MetricsLogger API](https://docs.ray.io/en/latest/rllib/metrics-logger.html)
- [MetricsLogger Reference](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.utils.metrics.metrics_logger.MetricsLogger.html)
- [Working with Offline Data](https://docs.ray.io/en/latest/rllib/rllib-offline.html)
- [AlgorithmConfig.evaluation](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.evaluation.html)

### RLlib Source Code
- [SingleAgentEnvRunner Source](https://docs.ray.io/en/latest/_modules/ray/rllib/env/single_agent_env_runner.html) — shows where callbacks are triggered (`_make_on_episode_callback`, `_sample`)
- [SingleAgentEnvRunner on GitHub](https://github.com/ray-project/ray/blob/master/rllib/env/single_agent_env_runner.py)
- [Algorithm.evaluate() on GitHub](https://github.com/ray-project/ray/blob/master/rllib/algorithms/algorithm.py) — evaluation EnvRunners use same callback class

### RLlib PRs
- [PR #47294: Store episodes in state form (get_state/from_state)](https://github.com/ray-project/ray/pull/47294)
- [PR #57017: Add support for complex observations in SingleAgentEpisode](https://github.com/ray-project/ray/pull/57017)

### Community Discussions
- [Ray Discuss: Getting custom metrics from SingleAgentEpisode](https://discuss.ray.io/t/rllib-callbacks-to-get-custom-metrics-such-as-observation-reward-etc-in-each-episode-from-singleagentepisode-and-access-it-in-the-trainer/16022)
