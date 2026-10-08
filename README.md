<p float="left">
    <img src="data/img/icon_kit.png" width="10%" hspace="20"/>
</p>

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green?logo=opensource)](./LICENSE)

<h1 align="center">AdvBuildingGym</h1>

<p align="center"><em>Repository to inspect deep reinforcement learning methods for generic/varying home energy management environments. 
A modular Gymnasium environment with training and evaluation pipelines. Thesis title: Generalization in Model-Free and Model-Based Reinforcement Learning for Home Energy Management Systems</em></p>


AdvBuildingGym is a framework for reinforcement learning (RL) in home energy management systems (HEMS). 
It simulates a residential building with a battery (BES), PV, a wind turbine, household load and dynamic electricity prices and optionally a heat pump, the thermal model of the building and an EV charger. 
Training and evaluation can run on Ray RLlib or Stable-Baselines3, however RLlib is used and tested.


The research question is **generalisation**: 
can one policy, trained across many building configurations and data scenarios, control buildings and days it has never seen? 
The main tool is **context variables**: static building parameters, such as battery power or PV rating, that the environment adds to the observation so the policy can condition on them.


> **Status:** volatile / active. 
Research code, configurations and interfaces can still change. 


## Contents

- [Research focus](#research-focus)
- [Environment](#environment)
- [Prepared extensions](#prepared-extensions)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data](#data)
- [Running experiments](#running-experiments)
- [Trial configuration](#trial-configuration)
- [Outputs](#outputs)
- [Tests](#tests)
- [Origin and citation](#origin-and-citation)
- [License](#license)
<!-- - [Further documentation](#further-documentation) -->

## Research focus

### Generalisation across buildings and data

A policy is trained on a set or a range of building configurations and evaluated on configurations held out from training. 
The generalisation trials live in [configs/trial_cfgs/v0/GEN/](configs/trial_cfgs/v0/GEN/). 
The groups `gs_lin_bat_power`, `gs_lin_bat_pv` and `gs_lin_bat_power_pv` each compare three training regimes on the same held-out evaluation configurations (`infra_schedule.configs.eval`).
 `gs_lin_bat_power_pv` runs every regime with PPO, SAC and DreamerV3.


| Regime | Configurations seen in training | Example from `gs_lin_bat_power_pv/` (battery power × PV rating -- generic environment) |
|---|---|---|
| Specialist | one fixed configuration | `lin_bat_power_pv_spec_ppo.yaml`: 9 kW × 10 kW |
| Grid | a discrete set, cycled every few episodes (`infra_schedule.mode: cycle`) | `lin_bat_power_pv_ppo.yaml`: {3, 9, 15} kW × {5, 10, 15} kW |
| Sampled | sizes redrawn every episode from a range, with the evaluation sizes excluded (`BatteryLinearWrapper`, `SolarPanelWrapper`) | `lin_bat_power_pv_sampled_ppo.yaml`: 3–15 kW × 5–15 kW |


All three are evaluated on held-out sizes in between: {3.7, 8.0, 11.0} kW × {6, 12} kW. 
Battery capacity is fixed at 10 kWh in this group.

Training and evaluation also use different data. 
The default data schedules ([configs/schedules/data/](configs/schedules/data/)) train on 2024 weather and prices and evaluate on 2025–2026, with separate EV schedules, indoor set-point profiles and household load profiles.


### Context variables for policy conditioning

Components can publish `ctxt_*` observation keys: these are parameters that remain constant within an episode. 
A key enters the observation space only if it is listed in that component's `ctxt_keys` in the YAML. 
An unknown key name fails at start-up. ([adv_building_gym/components/context_emitter.py](adv_building_gym/components/context_emitter.py))

```yaml
infras:
- class: BatteryLinearWrapper
  name: battery
  ctxt_keys: [ctxt_battery_power_kW]   # the policy observes this episode's max power
  max_power_range_kW: [3.0, 15.0]      # redrawn at every reset
  max_cap_range_kWh: [10.0, 10.0]
  exc_max_power_kW: [3.7, 8.0, 11.0]   # never drawn in training; used for evaluation
  start_soc_percentage: 0.3
  soc_min: 0.1
  soc_max: 0.95
```

| Component | Opt-in context keys |
|---|---|
| `BatteryLinear`, `BatteryLinearWrapper` | `ctxt_battery_capacity_kWh`, `ctxt_battery_power_kW` |
| `SolarPanel`, `SolarPanelWrapper` | `ctxt_solar_max_power_kW`, `ctxt_pv_area_m2` |
| `WindTurbine` | `ctxt_wind_rated_power_kW` |
| `HP`, `HPRbc` | `ctxt_hp_max_power_kW` |
| `LinearEVCharger` | `ctxt_evc_v2g_effective` |
| `BuildingHeatLoss` | `ctxt_building_K`, `ctxt_building_mC` |
| `DesiredUserEnergyNeed` | `ctxt_hh_consumption_max` |
| Energy price sources | `ctxt_E_price_max` |
| `BatteryTremblay` | `ctxt_battery_capacity_kWh`, `ctxt_battery_max_power_kW` |


A few context keys that rewards depend on are always published, for example the EV charger's `ctxt_evc_max_charging_kW` and the grid operator's `ctxt_operator_max_power_kW`.


### Generalisation experiments

Three experiments vary one or two building parameters each, their evaluation notebooks are in [thesis_eval/](thesis_eval/). 
Each one compares the specialist, grid and sampled regimes on the same held-out sizes and runs them with PPO, SAC and DreamerV3.

| Experiment (`thesis_eval/`) | Trials ([configs/trial_cfgs/v0/GEN/](configs/trial_cfgs/v0/GEN/)) | Varied in training | Held-out evaluation | Fixed |
|---|---|---|---|---|
| `3_gen_bat_cap_v1`: battery capacity | `gs_test_scenario_so/`, `battery_lin_capacity_sampled.yaml` | specialist 15 kWh or 17 kWh; grid {10, 13, 17, 20, 25} kWh; sampled 10–25 kWh | 12, 16, 22 kWh | 10 kW battery power, 5 kW PV |
| `5_gen_bat_power`: battery power | `gs_lin_bat_power/` | specialist 9 kW; grid {3, 6, 9, 12, 15} kW; sampled 3–15 kW | 3.7, 8.0, 11.0 kW | 10 kWh capacity, 5 kW PV |
| `6_gen_bat_cap_pv_power`: battery capacity × PV rating | `gs_lin_bat_pv/` | specialist 15 kWh × 10 kW; grid {10, 15, 20} kWh × {5, 10, 15} kW; sampled 10–20 kWh × 5–15 kW | {12, 16} kWh × {6, 12} kW | 10 kW battery power |

In all three the battery is the only controlled device, the reward is `EconomicRewardV0` alone, and training runs for 7000 episodes. 
The grid regime swaps the configuration every 12 episodes (every 4 in `5_gen_bat_power`). 
The sampled regime never draws the evaluation sizes.

The sizes above are the ones in the snapshots used by the notebooks. 
For `3_gen_bat_cap_v1`, the grid files in [configs/infra_cfgs/GEN/battery_lin_capacity_so_v2/](configs/infra_cfgs/GEN/battery_lin_capacity_so_v2/) have changed since then, so reproduce it from its snapshots.

`9_cmpl_gen` evaluates `gs_lin_bat_power_pv`, the combined setting described in [Generalisation across buildings and data](#generalisation-across-buildings-and-data). 
It varies battery power × PV rating, includes a rule-based heat pump and EV charger, pays exports at 0.7 × price (`EconomicSellFactorRewardV0`), and trains for 28000 episodes.


### Battery control with model-free and model-based RL

In the generalisation trials the battery (`a_battery`) is the only device the policy controls. 
PV, wind and household load are not controllable. 
Where a heat pump and an EV charger are present (`gs_lin_bat_power_pv`), they are rule-based (`HPRbc`, `EvcsRbc`): they draw power but take no policy action, and their rule-based actions (`ar_hp`, `ar_evcs`) are part of the observation. 
The battery-and-EV trial below is the exception: there the policy also controls the EV charger.


| Algorithm | Type | Ray RLlib | Stable-Baselines3 |
|---|---|---|---|
| PPO | model-free, on-policy | ✓ | ✓ |
| SAC | model-free, off-policy | ✓ | ✓ |
| DreamerV3 | model-based (learned world model) | ✓ | — |


Rule-based battery strategies in [adv_building_gym/rbc_strats/](adv_building_gym/rbc_strats/) serve as baselines: `do_nothing`, `pv_surplus_charge`, `self_coverage`, `deficit_discharge`, and price-threshold variants (`price_median`, `price_mean`, scaled and `*_autarky` versions).

### Joint battery and EV charger control

One trial controls two devices at the same time: [configs/trial_cfgs/v0/STA/battery_lin_EV.yaml](configs/trial_cfgs/v0/STA/battery_lin_EV.yaml) (`sta_lin_battery_EV`). 
The policy outputs `a_battery` (`BatteryLinear`, 10 kW, 16 kWh) and `a_lin_ev_charger` (`LinearEVCharger`, 11 kW, with V2G) for one fixed building with PV, wind and household load. 
EV charging sessions come from the usage profiles in [data/ev_usage_profiles/](data/ev_usage_profiles/): arrival time, vehicle, start and target state of charge, and the time allowed to reach the target. 
The reward is multi-objective: energy cost (`EconomicRewardV0`), a 20 kW grid-operator limit (`OperatorEnergyControlRewardV0`) and EV charging sessions (`EVChargingSessionReward`). 
A configuration also adds `BESRegulatorReward` and `EVRegulatorReward`, which pay a flat penalty whenever the battery or the charger has to override the requested action, but their effect does not match their intentions completely.
It is only trained with PPO and SAC.


## Environment

`AdvBuildingGym` ([adv_building_gym/core/env.py](adv_building_gym/core/env.py)) is a Gymnasium environment with Dict observations and Dict actions. 
By default one step is 5 minutes (`control_step: 300` s) and one episode is one day long (`EPISODE_LENGTH: 288`).
The final environment is assembled from three types of components.


**Infrastructures**: devices that consume or generate power, some of which take an action.


| Class | Description | Action |
|---|---|---|
| `BatteryLinear` / `BatteryLinearWrapper` | linear battery model; the wrapper redraws power and capacity every episode | `a_battery` |
| `BatteryTremblay` | Tremblay battery model with a series-parallel cell pack | `a_battery` |
| `SolarPanel` / `SolarPanelWrapper` | PV output from irradiance; the wrapper redraws the rating every episode | — |
| `WindTurbine` | power curve with cut-in, rated and cut-out wind speeds | — |
| `HouseholdEnergyConsumers` | household load driven by the load profile | — |
| `HP` | heat pump for heating and cooling | `a_hp` |
| `HPRbc` | rule-based heat pump (deadband around the set-point) | — |
| `LinearEVCharger` | EV charger, optionally with vehicle-to-grid (V2G) | `a_lin_ev_charger` |
| `EvcsRbc` | rule-based EV charger (charges to the target state of charge by the deadline) | — |


**State sources**: time series and physics that drive the observation.

| Class | Description |
|---|---|
| `WeatherDataSource` | outdoor temperature, solar irradiance, wind speed (DWD or WPuQ CSVs) |
| `EnergyPriceDayDynDataSource`, `EnergyPriceYearDynDataSource` | day-ahead spot prices; they differ in how the price is normalised (max over the next 24 h, or a blend of year and episode statistics) |
| `EnergyPriceFixDataSource` | fixed or time-of-use tariff defined by price bands |
| `DesiredUserEnergyNeed` | household load profile |
| `EVState` | EV arrival and departure schedule with per-session targets |
| `OperatorEnergyControl` | grid-operator power limit, constant or from a CSV |
| `DateSource` | day of the year |
| `InsideTemperature` | indoor set-point profile; publishes the temperature error |
| `BuildingHeatLoss` | 1R1C thermal model of the building (`K`, `mC`) |


**Rewards** are summed into one scalar. 
They exist in two families, [rewards/v0/](adv_building_gym/components/rewards/v0/) and [rewards/v1/](adv_building_gym/components/rewards/v1/), used by the `configs/trial_cfgs/v0/` and `v1/` trials respectively. 
They cover economic cost (per step, long-term, asymmetric sell price), energy use, thermal comfort, battery targets and management, EV charging, grid-operator limits and action smoothness.

**Step order.** Actions are applied, then devices and the building physics update, then the power balance and cost are computed. 
Time then advances, the time series move to the next row, and termination and rewards are evaluated over the full (s, a, s′) transition ([`AdvBuildingGym.step`](adv_building_gym/core/env.py)).


**Wrappers.** The Ray environment creator ([ray/env_creator.py](adv_building_gym/ray/env_creator.py)) applies these in order:

1. `HistoryWrapper` (optional): past values of selected keys.
2. `ForecastWrapper` (optional): `s_fc_*` look-ahead values from forecastable sources.
3. `FlattenAction` + `RescaleAction`: the policy acts in `Box(-1, 1)`.
4. `FlattenObservation`: flat observation vector.

The Stable-Baselines3 adapter keeps the Dict observation and uses `MultiInputPolicy`.


## Prepared extensions

The environment supports more than the current experiments use:

- **Fixed and time-of-use tariffs** through `EnergyPriceFixDataSource`.
- **Asymmetric buy and sell prices.** `EconomicSellFactorRewardV0` pays exports at `sell_price_factor` × price, and the env's cost tracker bills `cum_price_EUR` with the same factor.
- **Grid-operator power limits** (`OperatorEnergyControl` + `OperatorEnergyControlReward`).
- **Non-controllable devices.** `HPRbc` and `EvcsRbc` replace `HP` and `LinearEVCharger` with the same observation keys and no action. 
Configure one or the other, never both.
- **Heat pump control on the 1R1C building model**, plus building envelope variants in [configs/statesource_cfgs/GEN/building/](configs/statesource_cfgs/GEN/building/).
- **Curricula.** Reward schedules (modes `off`, `fix`, `gradual_add`, `random`, `dirichlet`), infrastructure and state-source schedules that hot-swap components during training, and an exploration reset on swaps.
Only infrastructure schedules are used currently in the experiments.
- **[BETA] Multi-agent training** with one agent per actuator ([rl_ma_train.py](rl_ma_train.py), [ray/ma_env.py](adv_building_gym/ray/ma_env.py)). Experimental and not used in the current experiments.

<!-- - **Classic controllers** carried over from LLECBuildingGym (PI, PID, fuzzy, MPC with Pyomo) in [adv_building_gym/controllers/](adv_building_gym/controllers/). No current training or evaluation script uses them. -->


## Repository layout

```text
AdvBuildingGym/
├── adv_building_gym/
│   ├── core/             # AdvBuildingGym env + wrappers (history, forecast, action flattening)
│   ├── components/       # infrastructure/, statesources/, rewards/, registry, context emitter
│   ├── config/           # trial YAML loading: env/, data/, rewards/, training/
│   ├── ray/              # Ray RLlib adapter: training, callbacks, evaluation
│   ├── sb/               # Stable-Baselines3 adapter: training, callbacks, evaluation
│   ├── rbc_strats/       # rule-based battery strategies (baselines)
│   ├── controllers/      # PI / PID / fuzzy / MPC (legacy, not wired in)
│   └── _common/          # shared utilities (normalisation, eval results, early stopping, ...)
├── configs/
│   ├── trial_cfgs/       # trial YAMLs: v0/, v1/ → GEN/ (generalisation), STA/ (fixed setups), TL/ (reward curricula, v0 only)
│   ├── infra_cfgs/       # infrastructure sets referenced by infra schedules
│   ├── statesource_cfgs/ # building variants referenced by state-source schedules
│   ├── schedules/        # data and reward schedules
│   └── eval_sweeps/      # seed sweeps for snapshot evaluation
├── preprocessing/        # data download, preprocessing and synthesis pipelines
├── data/                 # raw and preprocessed CSVs; EV, set-point and operator profiles
├── plotting/             # trajectory plots, dataset plots, evaluation dashboard
├── tools/snapshot/       # immutable code + config snapshots and SLURM submission
├── slurm_scripts/        # sbatch wrappers
├── thesis_eval/          # notebooks, figures and tables for the thesis experiments
├── tests/                # pytest suite
├── docs/                 # design notes, analyses
├── run_train_ray.py      # training (Ray RLlib: PPO, SAC, DreamerV3)
├── run_train_sb.py       # training (Stable-Baselines3: PPO, SAC)
├── run_eval_ray.py       # evaluation of Ray checkpoints
├── run_eval_sb.py        # evaluation of SB3 models
├── run_eval_rule_based.py# evaluation of rule-based strategies
└── rl_ma_train.py        # multi-agent training ([BETA] experimental)
```

## Installation

Requires Python 3.12.

```bash
git clone https://github.com/vince-pongracz/AdvBuildingGym
python3.12 -m venv adv_env
source adv_env/bin/activate
cd AdvBuildingGym

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```


**Evaluation dashboard assets.** Trajectory plots and the evaluation dashboard inline two JavaScript libraries, managed with npm. Install them once (Node.js and npm required):

```bash
npm install --prefix plotting/dashboard
```

The libraries are only needed to *build* dashboards, not to view them. Versions are pinned in [plotting/dashboard/package.json](plotting/dashboard/package.json).


**Jupyter kernel** (for the notebooks in `thesis_eval/`):

```bash
python -m ipykernel install --user --name=adv_env --display-name "Python (adv_env)"
```


## Data

All data download and preprocessing runs from one entry point, it is worth running preprocessing using the SLURM script as it fetches a lot of data:

```bash
python preprocessing/data_setup.py                         # all pipelines
python preprocessing/data_setup.py --years 2024 2025 2026  # restrict to some years
python preprocessing/data_setup.py --skip-weather          # prices only
python preprocessing/data_setup.py --synthesize            # also generate synthetic variants
sbatch slurm_scripts/slurm_data_setup.sh                   # on SLURM
```


| Data | Source | Output |
|---|---|---|
| Day-ahead electricity prices | [aWATTar](https://www.awattar.de/) and [Energy-Charts](https://www.energy-charts.info/) APIs | `data/e_price/` |
| Weather | [DWD Climate Data Center](https://opendata.dwd.de/climate_environment/CDC/), station 04177 (Rheinstetten) by default | `data/weather/dwd/` |
| Weather, household load | WPuQ dataset ([paper](https://www.nature.com/articles/s41597-022-01156-1), [Zenodo](https://zenodo.org/records/5642902)) | `data/weather/zenodo/`, `data/hh_consumption/wpuq/` |
| EV usage, indoor set-points, grid-operator signals | CSV profiles in the repository | `data/ev_usage_profiles/`, `data/inside_temp/`, `data/operator_signals/` |

Preprocessed CSVs keep physical units; the state sources normalise at runtime. Details: [data/DATA_README.md](data/DATA_README.md), [preprocessing/SYNTHESIZE_README.md](preprocessing/SYNTHESIZE_README.md).

## Running experiments

Every driver takes a single trial YAML (`--trial`). The trial bundles the algorithm, environment, rewards, schedules and hyperparameters.

### Training

The followings are just example scripts, it is not recommended to run them without SLURM and HPC.

```bash
python run_train_ray.py --trial configs/trial_cfgs/v0/GEN/gs_lin_bat_power_pv/lin_bat_power_pv_sampled_ppo.yaml
python run_train_sb.py  --trial <trial.yaml>          # PPO or SAC only
python run_train_ray.py --trial <trial.yaml> --cpu    # CPU-only smoke test
```

Training expects a GPU allocated by SLURM (`CUDA_VISIBLE_DEVICES`) and exits without one; `--cpu` bypasses that check. The algorithm comes from the trial's `algorithm:` key.

### Evaluation

```bash
python run_eval_ray.py --trial <trial.yaml> --checkpoint <path> --episodes 10
python run_eval_sb.py  --trial <trial.yaml> --checkpoint <path> --episodes 10
python run_eval_rule_based.py --trial <trial.yaml> --strategy all --episodes 10
```

- Without `--checkpoint`, `run_eval_ray.py` uses the latest checkpoint under `models/<trial_name>/ray/<algorithm>/`.
- If the trial lists several evaluation configurations (`infra_schedule.configs.eval` or `statesource_schedule.configs.eval`), each one is evaluated separately.
- `--data-mode` and `--data-day` override the data selection.
- Trajectory dashboards are written by default.


### Reproducible runs: snapshots

The snapshot tool freezes code and configs into `snapshots/<id>/` and submits a SLURM job against that copy:

```bash
python -m tools.snapshot.submit_snapshot --trial <trial.yaml> --kind train --sbatch="--time=24:00:00"
python -m tools.snapshot.submit_snapshot --snapshot snapshots/<id> --kind eval -- --episodes 20
```

Kinds are `train`, `train-cpu`, `train-sb`, `train-sb-cpu`, `train-ma`, `eval`, `eval-sb` and `eval-rbc`. 
For `train`, the SLURM wrapper is chosen from the trial's algorithm; DreamerV3 gets a wrapper with fewer CPUs. 
NOTE: this should be changed, as the framework updates, later RLlib versions can handle multiple EnvRunners for DreamerV3.
Outputs land in `snapshots/<id>/runs/<kind>_<timestamp>/`. To evaluate many snapshots across many seeds, use [submit_snapshot_eval_seeds.sh](submit_snapshot_eval_seeds.sh) with a sweep file from [configs/eval_sweeps/](configs/eval_sweeps/).


### SLURM

```bash
sbatch slurm_scripts/slurm_train_ray.sh --trial <trial.yaml>   # submit directly, without a snapshot
squeue -u $USER                                                # your jobs
squeue --start -u $USER                                        # estimated start times
scontrol show job <job_id>                                     # job details
scancel <job_id>                                               # cancel a job
```


Logs go to `slurm_logs/<task>/`; the `.err` file is the main debugging source. See [slurm_scripts/SLURM_README.md](slurm_scripts/SLURM_README.md) and the [HAICORE batch documentation](https://www.nhr.kit.edu/userdocs/haicore/batch/).


### Plots and monitoring

```bash
# Plot trajectories
python -m plotting.traj_plotting --hdf5 eval_results/<run>/trajectories.hdf5 --all-episodes

# See and inspect tensorboard logs of a training (return, in-training evaluation curves, exploration, etc...)
./start_tensorboard.sh models/<trial_name>/ray/<algorithm>/
```

Dataset-level plots: `sbatch slurm_scripts/slurm_data_vis.sh`. See [plotting/PLOTTING_README.md](plotting/PLOTTING_README.md).


## Trial configuration

A trial YAML has these top-level sections:

| Key | Purpose |
|---|---|
| `trial_name` | run name; also the output folder under `models/` |
| `algorithm` | `ppo`, `sac` or `dreamerv3` |
| `metric` | metric Ray Tune ranks results by (for example `episode_return_mean`) |
| `num_envs` | number of parallel environments for the Stable-Baselines3 driver, default is 1. |
| `seed`, `env_seed`, `eval_seed` | learner seed; environment seed (defaults to `seed`); in-training eval env seed (defaults to `env_seed`) |
| `env_meta` | episode length, control step, `hst_env_wrapper`, `forecast_env_wrapper` configurations |
| `training_params` | `common` (evaluation interval, early stopping, episode budget) plus `ppo` / `sac` / `dreamerv3` hyperparameters |
| `infras`, `statesources` | inline component lists (or `null` when a schedule provides them) |
| `rewards` | reward classes, weights and parameters |
| `infra_schedule`, `statesource_schedule` | train and eval component sets, swap mode and cadence |
| `reward_schedule`, `grad_train` | [BETA] reward curriculum (off unless `grad_train: true`) |
| `data_schedule` | train and eval data schedule YAMLs |
<!-- | `exploration_reset` | raise exploration again after schedule swaps | -->


`seed` drives network initialisation and exploration. 
`env_seed` drives everything random inside the environment: data variant, episode day and per-episode size draws. 
Sweeping `seed` with `env_seed` fixed varies only the learner over an identical data sequence.


## Outputs

| Path | Content |
|---|---|
| `models/<trial_name>/ray/<algorithm>/` | Ray checkpoints and Tune results |
| `models/<trial_name>/sb3/<algorithm>/` | SB3 models |
| `ep_metrics/` | training-time episode metrics and eval trajectories (TensorBoard) |
| `eval_results/<YYYYmmdd_HHMMSS>_eval/` | evaluation summary (JSON/CSV), `trajectories.hdf5`, plots |
| `snapshots/<id>/runs/` | outputs of snapshot runs |
| `slurm_logs/` | SLURM logs by task |

These also apply under/within a snapshot directory.

In evaluation results, `cum_price_EUR` is the net electricity cost of an episode: lower is better, and negative values mean net earnings.

## Tests

```bash
python -m pip install pytest
python -m pytest tests
```

<!-- 
## Further documentation

- [docs/workflow.md](docs/workflow.md): end-to-end flow (data → train → evaluate → plot)
- [configs/trial_cfgs/TRIAL_HELP.md](configs/trial_cfgs/TRIAL_HELP.md): notes on individual trials
- [configs/schedules/reward/README.md](configs/schedules/reward/README.md): reward schedules
- [docs/about_generalisation_for_rl.md](docs/about_generalisation_for_rl.md): generalisation in RL, reading notes
- [docs/hst_mgmt.md](docs/hst_mgmt.md): observation history design
- [docs/eval_data_combinator.md](docs/eval_data_combinator.md): data variants in evaluation
- [docs/README_archive.md](docs/README_archive.md): earlier goals, research ideas, literature notes and superseded README sections 

-->


## Origin and citation

AdvBuildingGym has been inspired by the [LLECBuildingGym](https://github.com/KIT-IAI/LLECBuildingGym), a heat-pump control environment, related to the Heat Pump House at the [Living Lab Energy Campus (LLEC)](https://www.iai.kit.edu/english/RPE-LLEC.php), KIT. 
This repository has been evolved further away from that.


Reference to the LLECBuildingGym:

```bibtex
@inproceedings{demirel2025_LLECBuildingGym,
      title={Advanced Deep Reinforcement Learning for Heat Pump Control in Residential Buildings},
      author={Gökhan Demirel and Ömer Ekin and Jianlei Liu and Luigi Spatafora and Kevin Förderer and Veit Hagenmeyer},
      year={2025},
      booktitle={Proceedings of the IEEE ISGT Europe 2025 (accepted)},
      address = {Malta},
      url = {https://github.com/KIT-IAI/LLECBuildingGym},
      pages={1--5}
}
```

## License

This code is licensed under the [MIT License](LICENSE).
For questions, contact the owner of the repository.