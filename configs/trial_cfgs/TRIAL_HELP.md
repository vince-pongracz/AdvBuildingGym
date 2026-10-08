# Trial Configs — Overview

Each trial YAML in `configs/trial_cfgs/` is the single source of truth for one
training/eval run. Launch it through SLURM with:

```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial <path/to/trial.yaml>
```

All run-level knobs (algorithm, seed, episode count, metric, checkpoint cadence,
schedules, exploration reset) live inside the YAML — the SLURM script only
forwards `--trial`.

---

## Top-level (baseline / sweep) trials

### `trial_cfg_1_ppo.yaml`
```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/trial_cfg_1_ppo.yaml
```
Baseline PPO reference run with the full default infra stack (HP + Tremblay battery + LinearEV + SolarPanel + hh_consumers), 1-day episodes, all rewards active, no schedules. Used as the PPO sanity/baseline against which all SAC variants are compared.

### `trial_cfg_1_sac.yaml`
```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/trial_cfg_1_sac.yaml
```
Baseline SAC run on the standard 1-day episode with WindTurbine added and a smaller (5 kW / 10 m²) PV + 7 kW EV charger; uses tuned SAC defaults (`n_step_return=6`, replay = 500 days). Reference point for the SAC ablations below.

### `trial_cfg_1_sac_default_tr_params.yaml`
```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/trial_cfg_1_sac_default_tr_params.yaml
```
SAC with the **default RLlib training params** (no custom `replay_batch_size` / `training_intensity`) and `exploration_reset.enabled=true` triggered on reward swaps. Tests how much of the SAC gain comes from explicit UTD tuning vs. exploration-reset bumps.

### `trial_cfg_1_sac_3day_ep.yaml`
```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/trial_cfg_1_sac_3day_ep.yaml
```
SAC on **3-day episodes** (`EPISODE_LENGTH=864`) — probes whether longer horizons help battery/EV scheduling that spans multiple days. Same infra/reward setup as the SAC baseline.

### `trial_cfg_1_sac_7day_ep.yaml`
```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/trial_cfg_1_sac_7day_ep.yaml
```
SAC on **7-day episodes** (`EPISODE_LENGTH=2016`, halved `max_episodes_to_run=1500`). Long-horizon credit-assignment stress test; same infra and rewards as the SAC baseline.

### `trial_cfg_1_sac_lb_variant.yaml`
```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/trial_cfg_1_sac_lb_variant.yaml
```
SAC **large-batch variant**: `replay_batch_size=1024`, `training_intensity=4` (UTD ≈ 0.004). Tests the trade-off between bigger, more stable gradients and reduced update-to-data ratio.

---

## `STA/` — Static, single-focus baselines (no schedules)

Each STA trial isolates one device + its matching reward to measure the
RL-controllable headroom of that subsystem alone.

### `STA/lin_battery_only.yaml`
```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/STA/lin_battery_only.yaml
```
Linear battery model only (HP, EV, Tremblay battery removed). Rewards reduced to `BatteryTargetReward + BatteryMgmtReward + LongTermEconomicReward`. Static baseline for linear-battery management.

### `STA/lin_battery_only_lb.yaml`
```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/STA/lin_battery_only_lb.yaml
```
Same `lin_battery_only` topology but with the **large-batch SAC** params (1024 batch, intensity=4). Used to pair the LB ablation with the linear-battery isolation.

### `STA/lin_battery_only_eval.yaml`
```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/STA/lin_battery_only_eval.yaml
```
Eval-twin of `lin_battery_only`: uses the **eval data schedule for both train and eval** so that policies can be replayed on held-out data without retraining-time data drift.

### `STA/tre_battery_only.yaml`
```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/STA/tre_battery_only.yaml
```
Tremblay (non-linear, cell-level) battery only — direct comparison vs. the linear variant under identical rewards and data.

### `STA/temp_only.yaml`
```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/STA/temp_only.yaml
```
HP + building only (no battery, no EV). Rewards: temperature + economic + minimise-energy + operator-limit. Static baseline for indoor-comfort control.

### `STA/EV_only.yaml`
```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/STA/EV_only.yaml
```
EV charger only (HP, battery removed). Rewards: `EVChargingReward + EVChargingOnTimeReward + OperatorEnergyControlReward`. Isolates EV-scheduling policy quality.

---

## `GEN/` — Generalisation across infra/statesource variants (curricula)

GEN trials cycle a list of YAMLs through `infra_schedule` or
`statesource_schedule` so the agent sees a population of variants during a
single training run.

`infra_schedule.configs` and `statesource_schedule.configs` are dicts with
`train` and `eval` lists (both required):

```yaml
infra_schedule:
  mode: "cycle"
  swap_every_n_episodes: 100
  configs:
    train:
      - configs/infra_cfgs/.../foo_1.yaml
      - configs/infra_cfgs/.../foo_2.yaml
    eval:
      - configs/infra_cfgs/.../foo_1.yaml
      - configs/infra_cfgs/.../foo_2.yaml
```

Training cycles `configs.train` via the swap callback. `run_eval_ray.py`
iterates each entry of `configs.eval` for `--episodes` episodes, writing
results into per-config subdirectories under
`eval_results/<timestamp>_eval/<config_stem>/`. Only one of
`infra_schedule.configs.eval` and `statesource_schedule.configs.eval` may
have more than one entry; the other axis must stay single-entry.

### `GEN/battery_lin_capacity.yaml`
```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/GEN/battery_lin_capacity.yaml
```
Cycles 5 **linear-battery capacity** variants (`infra_cfgs/GEN/battery_lin_capacity/battery_lin_cap_{1..5}.yaml`) every 100 iterations. Generalisation focus: robust battery control across capacity ranges.

### `GEN/battery_tre_capacity.yaml`
```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/GEN/battery_tre_capacity.yaml
```
Same idea on the **Tremblay battery** — cycles 5 capacity variants every 300 iterations. Tests whether the non-linear cell model transfers across pack sizes.

### `GEN/solar_power.yaml`
```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/GEN/solar_power.yaml
```
Cycles 5 **PV/solar configurations** (panel area / max power) every 300 iterations. Forces the policy to adapt to a wide range of on-site generation.

### `GEN/building_changes.yaml`
```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/GEN/building_changes.yaml
```
Uses **`statesource_schedule`** (not infra) to swap 5 building-property bundles (`statesource_cfgs/GEN/building/building_{1..5}.yaml`) every 300 iterations — varies `BuildingHeatLoss.K/mC` and related building params. Generalisation focus: thermal-envelope robustness.

---

## `TL/` — Transfer-learning / reward-curriculum trials

All TL trials enable `grad_train: true` so `reward_schedule.mode` is honoured.
They use the full infra stack and exercise different reward-curriculum policies.

### `TL/gradual_all.yaml`
```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/TL/gradual_all.yaml
```
`mode: gradual_add` — starts with `TempReward` only and adds one reward every 100 episodes following the explicit `reward_order`. Curriculum-learning scenario: progressively expanding the multi-objective.

### `TL/random_all_1_3.yaml`
```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/TL/random_all_1_3.yaml
```
`mode: random` with `random_active_count: 3, random_swap_count: 1` — 3 rewards active at any time, **1 swapped** every 100 episodes. Smooth random-curriculum transfer.

### `TL/random_all_2_3.yaml`
```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/TL/random_all_2_3.yaml
```
`mode: random`, `active=3, swap=2` per 100 episodes — more aggressive swapping under the same active-set size; tests policy resilience to faster reward shifts.

### `TL/random_all_2_4.yaml`
```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial configs/trial_cfgs/TL/random_all_2_4.yaml
```
`mode: random`, `active=4, swap=2` per 100 episodes — bigger active reward set (richer objective at any moment) with the same swap rate as `2_3`.

---

## `env_meta.hst_env_wrapper` — per-key strided observation history

Optional sub-dict under `env_meta`:

```yaml
env_meta:
  EPISODE_LENGTH: 288
  control_step: 300
  allow_early_termination: false
  hst_env_wrapper:
    enabled: true                          # if false or absent, env is not wrapped
    tracked_keys: [s_temp_in_norm, a_hp_prev]
    offsets: [-1, -2, -4, -12]             # negative = past env steps
```

When enabled, `HistoryWrapper` (`adv_building_gym/envs/history_wrapper.py`) adds
one new `s_hst_<key>` Dict entry per tracked key, with shape
`(len(offsets), *original_shape)`. Offset `0` (current step) is always prepended;
the remaining entries must be `< 0` (lag in env steps). Originals pass through
unchanged. Pre-episode slots stay zero rather than replicating the current
frame.

Listing `a_<x>_prev` keys works: the env publishes them as plain obs entries
each step.
