# Monte Carlo Rollout Script

## Context

The current eval flow conflates single-episode evaluation with stochastic rollouts. A stochastic eval (`run_eval_ray.py --stochastic`) produced very jittery actions (`20260518_145654_eval_stoch/plots/ep_4/ep_4_actions.html`) because `infer_action()` resamples i.i.d. Gaussian noise every step (`adv_building_gym/ray_training/rl_module_inference.py:90-99`: `a = tanh(μ + σ·ε)` with fresh ε per step). `ActionSmoothnessReward` only penalises this retroactively via FFT on the action history, so it never suppresses sampling jitter at the source.

Goal: a standalone Monte Carlo rollout tool that, from a fixed calendar date, runs K reproducible stochastic rollouts of N steps with the same policy and overlays them (per-episode traces + aggregate stats) on one set of plots — answering "what is the spread of behaviour my stochastic policy produces?". The script also exposes a `--temperature` knob that scales `log_std` at inference time so the user can dial sampling variance without retraining; this is the agreed lever for the oscillation issue (root-cause fix is out of scope here).

## Approach

### 1. New rollout module — `adv_building_gym/evaluation/montecarlo_runner.py`

Sibling to [eval_runner.py](adv_building_gym/evaluation/eval_runner.py). Reuses heavily; does not subclass.

Public entry: `run_montecarlo(trial_config, checkpoint, *, date, num_ro_episodes, num_ro_steps, temperature, base_seed, out_dir, plot)` returning a list of in-memory episode dicts (one per `TrajectoryCollector.to_dict()`).

Implementation notes:
- **Date pinning.** Take `--date YYYY-MM-DD`, set `trial_config.data_combinator.day = date` *before* `init_singletons()`. `DataCombinator.get_day_offset()` already routes string-day to `_date_to_day_index()` → fixed `row_offset` (data_combinator.py:151-179). All K episodes will index from the same row.
- **Env build.** Reuse the env-construction block from [eval_runner.py:168-175](adv_building_gym/evaluation/eval_runner.py#L168-L175) (manual env + `FlattenAction + RescaleAction`, same connector pipeline). Override `EPISODE_LENGTH = num_ro_steps` on `env_config` for the duration of the run so episodes auto-terminate at N.
- **Policy load.** Reuse `load_rl_module()` + connector pipeline construction — identical to eval (eval_runner.py around the RLModule load).
- **Per-episode loop.** K iterations, each with `episode_seed = base_seed + ep` + a dedicated `torch.Generator` (mirrors eval_runner.py:250-254). One `TrajectoryCollector` per episode, `.to_dict()` appended to a list. Optionally also call `save_hdf5()` per episode to a shared `trajectories.hdf5` so existing single-episode plots still work if needed.
- **Output dir.** `eval_results/<timestamp>_mc/` with `<timestamp>_mc.json` (summary across K), per-episode `ep_<i>.json`, and `trajectories.hdf5` — symmetric with eval.

### 2. Temperature flag — minimal change to `infer_action`

Edit [adv_building_gym/ray_training/rl_module_inference.py:39-111](adv_building_gym/ray_training/rl_module_inference.py#L39-L111):

- Add optional `temperature: float = 1.0` kwarg.
- In the stochastic branch (lines 90-99), replace `exp(log_std)` with `exp(log_std) * temperature` (i.e. scale σ, not log σ — clearer semantics for the user). `temperature=0.0` should fall through to the deterministic path so `--temperature 0` is a clean determinism switch.
- Deterministic path unchanged.
- Eval runner unaffected (defaults to 1.0).

### 3. Thin CLI shell — `run_montecarlo_ray.py` at repo root

Mirrors [run_eval_ray.py](run_eval_ray.py) layout. Flags:

- `--trial <path>` — required, loaded via `TrialConfig.load()` (trial_config.py:108-125). Same plumbing as eval.
- `--checkpoint <path>` — required, resolved via existing `resolve_checkpoint_path()` (utils/checkpoint_finder.py). Kept separate per established convention.
- `--date YYYY-MM-DD` — required, pins all rollouts to one calendar day.
- `--num-ro-episodes K` (default 10).
- `--num-ro-steps N` (default = `env_config.EPISODE_LENGTH`).
- `--temperature` (default 1.0). Always stochastic in MC mode (no separate `--stochastic` flag needed — the script's purpose is stochastic rollouts).
- `--seed` (base seed; per-episode seeds derive from it).
- `--plot` / `--no-plot`.

CPU-only `ray.init` like eval (no GPU needed for inference). Validate that `--date` falls inside the loaded scenario CSV range and that `num_ro_steps ≤ steps_per_day - offset_within_day` (or relax for multi-day).

### 4. Overlay plotting — `plotting/mc_plotting/`

A new subpackage parallel to `plotting/traj_plotting/`, sharing utilities. The user picked the dedicated-package layout *(in the answers)*; this keeps single-episode plotter untouched.

Modules (mirroring `traj_plotting`):
- `mc_plot.py` — entry point `generate_mc_plots(episodes: list[dict], out_dir, config)`; produces one HTML/PNG per variable family (states, actions, rewards, energy, raw).
- Shared aggregator `_aggregate(values: np.ndarray[K, T]) -> {mean, std, min, max}` (np.nan-safe).
- Per-family modules can largely be thin wrappers that:
  1. Build a `pd.DataFrame` indexed by step, columns per episode, for each tracked key.
  2. Add K traces (low opacity, shared color per episode index).
  3. Add `mean` trace (bold).
  4. Add `mean ± std` as a filled band (two traces with `fill='tonexty'`).
  5. Add `min` and `max` as dashed envelope traces.
- Reuse `EpisodeData` from [plotting/utils.py](plotting/utils.py) but introduce `MultiEpisodeData = list[EpisodeData]` plus a `to_episodedata_list(in_memory_dicts)` helper so we don't have to round-trip through HDF5.
- Config: new `plotting/config/mc_plot_config.yaml` (copy of traj config with sections for `band_alpha`, `episode_alpha`, `show_min_max`, etc.).

Plot config (chosen): K thin per-episode traces + mean + ±std band + min/max envelope, all on the same subplot per variable.

### 5. Files to modify / create

**Create:**
- `run_montecarlo_ray.py`
- `adv_building_gym/evaluation/montecarlo_runner.py`
- `plotting/mc_plotting/__init__.py`
- `plotting/mc_plotting/mc_plot.py`
- `plotting/mc_plotting/aggregators.py`
- `plotting/config/mc_plot_config.yaml`

**Modify (small):**
- `adv_building_gym/ray_training/rl_module_inference.py` — add `temperature` kwarg.
- `adv_building_gym/__init__.py` — optionally re-export `run_montecarlo`.

**Do not touch:**
- `run_eval_ray.py`, `eval_runner.py`, `plotting/traj_plotting/*`, callbacks, env code.

### 6. Reuse map (no duplication)

| Need | Reuse |
|---|---|
| Load trial | `TrialConfig.load()` (config/trial_config.py:108) |
| Resolve checkpoint | `resolve_checkpoint_path()` (utils/checkpoint_finder.py) |
| Build env | inline block from eval_runner.py:168-175 |
| Load policy | `load_rl_module()` (ray_training/rl_module_inference.py) |
| Sample actions | `infer_action()` (same file) with new `temperature` kwarg |
| Collect trajectory | `TrajectoryCollector` (utils/trajectory_collector.py) |
| Persist HDF5 | `write_episode_to_hdf5()` (utils/trajectory_utils.py) |
| Date → row offset | `DataCombinator.day = "YYYY-MM-DD"` (data_combinator/data_combinator.py:151-179) |
| Plot scaffolding | copy structure from `plotting/traj_plotting/`; aggregators new |

### 7. Note on oscillation (recorded, not fixed here)

`--temperature 0.3` should visibly reduce action jitter while preserving spread in slow-changing observables (battery SoC, indoor temp). Document in the script's `--help` that temperature 0 = deterministic, 1 = trained sampling, <1 = sharpened. A real fix (OU noise / sample-once-per-episode / fixed σ schedule) is a separate task.

## Verification

1. **Smoke test.** From repo root:
   ```
   source /home/iai/dj0397/adv_env/bin/activate
   python run_montecarlo_ray.py --trial configs/trial_cfgs/trial_cfg_1_dreamerv3.yaml \
       --checkpoint <known_good_ckpt> --date 2022-07-15 \
       --num-ro-episodes 10 --num-ro-steps 288 --temperature 1.0 --plot
   ```
   Expect: `eval_results/<ts>_mc/` populated; `trajectories.hdf5` has 10 episodes; one HTML per variable family in `plots/`.

2. **Date pinning.** Run twice with the same `--date` and `--seed`; per-episode trajectories should match (deterministic given seed + date), even if `temperature > 0`, because `torch.Generator` is reseeded per episode (`base_seed + ep`).

3. **Temperature sweep.** Run with `--temperature 0.0`, `0.3`, `1.0`. Expect: 0.0 → all K episodes identical (deterministic); 0.3 → small spread, smooth actions; 1.0 → wider spread, oscillating actions (reproduces the current behaviour).

4. **Plot sanity.** Open `plots/actions.html`: K per-episode traces visible, bold mean overlaid, std band visible, min/max dashed envelope. Number of traces per subplot = K + 4 (mean + 2 band edges + min + max — verify legend).

5. **Independence from eval.** Run `python run_eval_ray.py --trial ... --checkpoint ... --episodes 2` after MC changes — must still work unchanged (temperature defaults to 1.0 inside infer_action; eval path remains deterministic by default via `stochastic=False`).
