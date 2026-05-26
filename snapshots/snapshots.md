# Snapshots

This directory holds **snapshot bundles** created by `tools/snapshot/`.
Each bundle freezes the exact code + configs that feed a trial, so a
SLURM job — or a later eval re-run — produces results that don't drift
when the live repo changes.

Apart from this file, everything under `snapshots/` is gitignored.

## What's in a snapshot

```
snapshots/<YYYYMMDD_HHMMSS>_<trial_name>/
├── snapshot.zip          # immutable archive — code + the trial YAML + every
│                         #   YAML it transitively references (no CSV data,
│                         #   no plotting/)
├── manifest.json         # provenance: git SHA, dirty flag, sha256(zip),
│                         #   file list, creator, hostname, optional note
├── code/                 # extracted from snapshot.zip on first run; reused
│                         #   on later runs. Never edited.
└── runs/
    └── <kind>_<YYYYMMDD_HHMMSS>/
        ├── slurm_<jobid>.out
        ├── slurm_<jobid>.err
        ├── models/       # Ray checkpoints (train)
        ├── ep_metrics/   # per-episode metrics + tensorboard
        ├── eval_results/ # eval JSON/CSV/HDF5/plots
        └── result_*.json
```

A typical snapshot is ~270 KB on disk; the `runs/` outputs are the
variable part (hundreds of MB per training run).

## Usage

All flows are driven by `tools/snapshot/submit_snapshot.py`:

```bash
# 1. Create a fresh snapshot and submit training against it
python -m tools.snapshot.submit_snapshot \
    --trial configs/trial_cfgs/trial_cfg_1_sac.yaml \
    --kind train \
    --sbatch="--time=00:30:00"

# 2. Re-run eval against an existing snapshot. The script auto-discovers
#    the latest train checkpoint inside the same snapshot.
python -m tools.snapshot.submit_snapshot \
    --snapshot snapshots/<existing>/ \
    --kind eval \
    -- --episodes 20 --plot-all

# 3. Dry-run: print the sbatch command without creating a snapshot or job
python -m tools.snapshot.submit_snapshot \
    --trial configs/trial_cfgs/trial_cfg_1_sac.yaml --kind train --dry-run

# 4. Snapshot only (no submission). Prints the new snapshot dir on stdout.
python -m tools.snapshot.make_snapshot \
    --trial configs/trial_cfgs/trial_cfg_1_sac.yaml --note "baseline"

# 5. Fan out multiple seeds from a single snapshot. Each run gets its own
#    isolated dir; the original snapshot and live-repo YAML stay untouched.
python -m tools.snapshot.submit_snapshot --snapshot snapshots/<existing>/ --kind train --seed 123
python -m tools.snapshot.submit_snapshot --snapshot snapshots/<existing>/ --kind train --seed 456
python -m tools.snapshot.submit_snapshot --snapshot snapshots/<existing>/ --kind eval  --seed 999 -- --episodes 10
```

`--kind` ∈ `{train, eval, train-ma, train-sb}` selects the SLURM wrapper.
Anything after `--` is forwarded verbatim to the entry script
(`run_train_ray.py`, `run_eval_ray.py`, `rl_ma_train.py`, `run_train_sb.py`).

### `--seed N` — per-run seed override

`trial.seed` is the only knob that varies which days/variants the
`DataCombinator` samples and the stochastic streams used during training
(PyTorch, Ray init, `RngService`, replay shuffle). Without this flag, two
runs from the same snapshot evaluate on the exact same episode dates and
train along identical trajectories.

`--seed N` lets you fan out multiple runs from a single snapshot without
re-snapshotting and without mutating the live-repo trial YAML:

1. The run id is suffixed with `_seed{N}` (e.g. `train_20260524_185655_seed123`)
   so per-seed runs never collide.
2. The snapshot's `configs/` tree is **copied** (not symlinked) into the
   run dir.
3. The top-level `seed:` line in the copied trial YAML is rewritten to `N`
   via a line-level regex — trailing comments are preserved verbatim, and
   nested seeds (e.g. `data_combinator.seed`) are untouched.
4. `--trial` is pointed at the per-run copy, so the entry script loads the
   rewritten YAML.

The original `<snapshot>/code/configs/<trial>.yaml` and the live-repo
`configs/<trial>.yaml` are untouched. `snapshot_mode.sh` detects the
pre-existing real `configs/` directory and skips the symlink it would
normally create.

**Conflict guard:** `--seed N` cannot be combined with a pass-through
`--trial` (the override would silently bypass the rewritten copy). The
submitter rejects that combination up front.

## How it works

`tools/snapshot/submit_snapshot.py` invokes `sbatch` with three env vars:

| Variable | Purpose |
|----------|---------|
| `SNAPSHOT_DIR`    | absolute path of the snapshot dir |
| `SNAPSHOT_RUN_ID` | unique per-submission id (e.g. `train_20260521_232358`) |
| `LIVE_REPO_ROOT`  | absolute path of the live repo (so `plotting/` resolves) |

The SLURM wrappers source `slurm_scripts/util/snapshot_mode.sh`. When
`SNAPSHOT_DIR` is set it:

1. Extracts `snapshot.zip` into `<SNAPSHOT_DIR>/code/` on first run.
2. `cd`s into `<SNAPSHOT_DIR>/runs/<SNAPSHOT_RUN_ID>/`, so every relative
   output path (`models/...`, `ep_metrics/...`, `eval_results/...`,
   `result_*.json`) lands inside the snapshot.
3. Creates three symlinks so CSV paths and config references resolve
   correctly from both the driver and Ray's env-runner actors:
   ```
   <run_dir>/configs       -> <SNAPSHOT_DIR>/code/configs
   <run_dir>/data          -> <LIVE_REPO_ROOT>/data
   <SNAPSHOT_DIR>/code/data -> <LIVE_REPO_ROOT>/data
   ```
   The first two cover driver-side CWD-relative path checks (e.g.
   `Path("data/...").exists()` in the data combinator). The third is for
   `adv_building_gym/devices/statesources/csv_loader.py`, which resolves
   relative CSV paths against `Path(__file__).parents[3]` so Ray workers
   load CSVs regardless of their CWD — that anchor lands inside
   `<SNAPSHOT_DIR>/code/`, so the live data must be reachable from there
   too. Re-applied each run, so a snapshot moved to a different host
   picks up the new `LIVE_REPO_ROOT`.
4. Sets `ENTRY_SCRIPT` to the absolute path of the snapshotted python file.
   Because `sys.path[0]` is the script's directory, `import adv_building_gym`
   always resolves to the snapshot's frozen copy and never to the live repo's
   editable install — that guarantees experiment determinism.
5. Prepends `LIVE_REPO_ROOT` to `PYTHONPATH` so non-snapshotted packages
   like `plotting/` still resolve from the live tree.
6. The orchestrator passes `--trial` as an absolute path into the snapshot's
   `code/` dir, so the entry script always reads the frozen trial YAML.

When `SNAPSHOT_DIR` is unset (the legacy `sbatch slurm_scripts/...` flow)
the wrappers behave exactly as before — fully backward compatible.

## What's deliberately excluded

* `data/` — CSV data is large and external; configs reference it by
  templates that resolve at run time against whatever is on disk.
* `plotting/` — plotting only renders results; the live repo's copy is
  used at run time via `LIVE_REPO_ROOT` on `PYTHONPATH`.
* `preprocessing/`, `tests/`, `docs/`, `models/`, `eval_results/`,
  `ep_metrics/`, `slurm_logs/`, `__pycache__/`, `*.egg-info/`.

See `tools/snapshot/make_snapshot.py` for the exact whitelist.
