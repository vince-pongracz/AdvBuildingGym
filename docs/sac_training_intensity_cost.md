# SAC `training_intensity` — Wall-clock · UTD · Sample-reuse

How SAC's `training_intensity` dial trades iteration wall-clock against
sample-efficiency, derived from the RLlib new-API-stack training loop and
validated against a real run.

**Reference run:** snapshot `20260604_143721_sta_lin_battery_only`, SAC, SLURM
job `1649357`, iteration 1750 (`models/.../sac_seed42_20260605_024556/.../result.json`).
The trial is [configs/trial_cfgs/v0/STA/lin_battery_only.yaml](../configs/trial_cfgs/v0/STA/lin_battery_only.yaml)
with `sac.training_intensity: 128`.

> The absolute numbers below are specific to this run's shape (4 env runners,
> episode length 288, replay batch 256). The **formulas** are general — re-derive
> the constants for a different runner count / batch size.

---

## 1. Where the time goes

`training_step` (≈ the whole iteration) is dominated by the SAC update loop in
`DQN._training_step_new_api_stack` (SAC inherits DQN), which runs
`sample_and_train_weight` times — one `local_replay_buffer.sample()` +
one `learner_group.update()` per pass
(`ray/rllib/algorithms/dqn/dqn.py:684,688,716`).

The per-pass loop count comes from `calculate_rr_weights`
(`ray/rllib/algorithms/dqn/dqn.py:578-592`):

```
native_ratio            = total_train_batch_size / (rollout_fragment_length × num_envs_per_runner × (num_env_runners + 1))
                        = 256 / (288 × 1 × (4 + 1)) = 0.17778
sample_and_train_weight = round(training_intensity / native_ratio) = round(TI × 5.625)   # grad steps / iter
```

### Verified constants (this run)

| Symbol | Value | Source |
|---|---|---|
| `total_train_batch_size` | 256 | `sac_replay_batch_size` default; `num_items=` at `dqn.py:689` |
| `rollout_fragment_length` | 288 | [common_model_config.py:280](../adv_building_gym/ray/training/common_model_config.py#L280) (= `EPISODE_LENGTH`) |
| `num_envs_per_env_runner` | 1 | RLlib default; corroborated `1152 = 4×288×1` |
| `num_env_runners` | 4 | SLURM `.err` resource line |
| env steps / iter (`I`) | 1152 | `result.json` `num_env_steps_sampled` |
| replay `sample()` per call | 66.7 ms | `result.json` `timers.replay_buffer_sampling_timer` |
| `learner.update()` per call | 32.0 ms | `result.json` `timers.learner_update_timer` |

Timers are **exact within-iter arithmetic means** (not RLlib's default EMA): the
project forces `reduce="mean", clear_on_reduce=True` Stats in
[iter_timing_callback.py:104](../adv_building_gym/ray/callbacks/iter_timing_callback.py#L104),
so `per-call × grad/iter` is exact. Cost model:

```
train_step(TI) ≈ fixed + grad_steps × (66.7 + 32.0) ms ,   fixed ≈ 2.1 s/iter (env sampling + buffer add + synch)
```

---

## 2. Merged table

Identities: `grad/iter = round(TI × 5.625)` · `UTD = grad/1152` ·
`sample-seen = grad × 256 / 1152 = TI × 1.25 = UTD × 256` ·
`train_step = 2.1 s + grad × 98.7 ms`.

| TI | grad steps/iter | UTD (updates/env-step) | sample seen (×) | train_step | env-steps/s | speedup vs 128 | →7000 ep |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16  | 90   | 0.078 | 20×  | 11.0 s  | 104.9 | 6.66× | 5.3 h |
| 32  | 180  | 0.156 | 40×  | 19.9 s  | 58.0  | 3.68× | 9.7 h |
| 48  | 270  | 0.234 | 60×  | 28.7 s  | 40.1  | 2.55× | 14.0 h |
| 64  | 360  | 0.312 | 80×  | 37.6 s  | 30.6  | 1.94× | 18.3 h |
| 96  | 540  | 0.469 | 120× | 55.4 s  | 20.8  | 1.32× | 26.9 h |
| **128** (current) | **720** | **0.625** | **160×** | **73.2 s** | **15.7** | **1.00×** | **35.6 h** |
| 192 | 1080 | 0.938 | 240× | 108.7 s | 10.6  | 0.67× | 52.8 h |
| 256 | 1440 | 1.250 | 320× | 144.2 s | 8.0   | 0.51× | 70.1 h |

- **`→7000 ep`** is `1750 iters × train_step` (eps/iter = 4, `result.json`),
  counting `training_step` only. Real wall-clock adds the ~20 s/iter eval +
  checkpoint overhead on eval iterations (`evaluation_interval=4`), roughly
  TI-independent, so it shifts every row by a near-constant amount.
- Everything is **linear in TI** — no operating point buys cheap wall-clock
  without proportionally cutting updates and reuse.

---

## 3. Independent cross-checks (model-free, from `result.json`)

| Quantity | From disk | Model | Match |
|---|---|---|---|
| grad steps/iter | `num_module_steps_trained 184320 / 256` = 720 | `round(128 × 5.625)` = 720 | ✓ |
| sample reuse | `num_module_steps_trained_lifetime 320,348,160 / num_env_steps_sampled_lifetime 2,016,000` = 158.9 | steady-state 160 (gap = warmup) | ✓ |
| train_step wall | measured 73.16 s | bottom-up `720 × 98.7 ms + 1.88 s` = 72.94 s | ✓ (0.3%) |

---

## 4. "Sample seen" is independent of buffer capacity

`sample-seen` = average number of gradient batches a stored transition is drawn
into over its whole buffer life. In steady state (full buffer, FIFO eviction,
uniform `EpisodeReplayBuffer`):

```
residence R = C / I  iterations         # bigger buffer ⇒ longer life
draws while resident = D · R            # D = grad × batch = draws/iter
P(a draw hits this transition) = 1 / C  # bigger buffer ⇒ less likely per draw
reuse = D · R · (1/C) = D · (C/I) · (1/C) = D / I     ← C cancels
```

So `reuse = D/I = grad × batch / env_steps = TI × 1.25`, with **no buffer
capacity term**. Equivalently (conservation): in steady state transitions leave
at the inflow rate `I`, and the `D` uniform draws/iter spread evenly over them,
so each catches `D/I` on average. Verified by Monte-Carlo (FIFO + uniform draw):
capacities 250 / 500 / 1000 / 2000 episodes all yield mean reuse **160.0**.

Buffer capacity controls **staleness/diversity** and the temporal spread of reuse
— not the average reuse count. The only knobs that move `sample-seen` are
`training_intensity` (via `D`) and the env-step inflow `I` (runners × fragment).

---

## 5. Takeaway

TI=128 sits at 160× reuse / 0.625 UTD. Standard SAC targets UTD ≈ 1.0, so this is
already on the conservative side, and the bottleneck is the Python per-transition
replay sampling, not learning. **TI=64** (80× reuse, 0.31 UTD) ≈ halves wall-clock
(35.6 h → 18.3 h to 7000 episodes) while staying in normal SAC territory. Lower
reuse also reduces overfitting-to-buffer risk — relevant when the goal is
generalisation. This is a per-run dial: `sac.training_intensity` in the trial YAML,
no code change.
