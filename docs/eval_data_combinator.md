# Integrate DataCombinator into Evaluation Scripts

## Context

The `DataCombinator` manages CSV data variant selection across episodes (weather, energy price, EV profiles) and is fully integrated into training. However, evaluation scripts (`run_eval_ray.py`, `eval_runner.py`) previously created environments **without** a `DataCombinator`, meaning eval always ran on the default data files. This limited the ability to evaluate models across diverse scenarios (different years, weather sources, price providers).

The environment (`AdvBuildingGym`) already accepts `data_combinator` and handles variant switching in `reset()` — the eval scripts simply needed to wire it up.

## CLI Arguments

Three new arguments on `run_eval_ray.py`:

| Argument | Description | Default |
|---|---|---|
| `--data-config` | Path to YAML (same format as training) | `None` (no combinator) |
| `--data-mode` | Override variant mode: `cycle` or `random` | YAML default |
| `--data-day` | Override day mode: `each`, `random`, or date string like `2022-07-15` | YAML default |

## Usage Examples

```bash
# Unchanged behavior (no data combinator)
python run_eval_ray.py --algorithm ppo --episodes 10

# Eval with data variants, using YAML defaults
python run_eval_ray.py --algorithm ppo --episodes 50 \
    --data-config configs/eval_data_combinator_config.yaml

# Eval-specific config (cycle all variants, sequential days)
python run_eval_ray.py --algorithm ppo --episodes 50 \
    --data-config configs/eval_data_combinator_config.yaml

# Override mode and day on the fly
python run_eval_ray.py --algorithm ppo --episodes 50 \
    --data-config configs/eval_data_combinator_config.yaml \
    --data-mode cycle --data-day each

# Pin specific date for controlled A/B comparison
python run_eval_ray.py --algorithm ppo --episodes 10 \
    --data-config configs/eval_data_combinator_config.yaml \
    --data-day 2022-07-15
```

## Design Decisions

- **Reuse training YAML** — same scenario pool, different traversal via `--data-mode`/`--data-day` overrides
- **Default = no change** — without `--data-config`, behavior is identical to before
- **Variant info in reset_info** — avoids calling `get_variant()` twice (which would break random mode); cleanly reads what `reset()` actually applied
- **Eval-specific config** — `configs/eval_data_combinator_config.yaml` uses `shuffle: false`, `mode: cycle`, `day: each`, `swap_every_n_episodes: 1` for deterministic reproducible evaluation

## Output

Per-episode results (JSON and CSV) now include optional fields when a DataCombinator is active:
- `data_variant`: dict mapping statesource name to CSV path used
- `episode_date`: date string (e.g. `"2022-07-15"`) for the episode
