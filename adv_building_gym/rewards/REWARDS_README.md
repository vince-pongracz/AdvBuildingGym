# Reward Functions

Reward functions inherit from `RewardFunction` (defined in [base.py](base.py)).
Each `get_reward(actions, states, info)` returns a tuple
`(weighted_reward, weighted_max_step)` — both already multiplied by
`self.weight`.

The base class exposes a `max_reward_in_step` class attribute (default `1.0`)
that declares the maximum **raw** (unweighted) value a reward class can return
per step. It is consumed by callbacks / eval to compute
`reward_rate = sum(reward) / sum(max_step)`:

```python
max_reward_per_step = sum(r.weight * r.max_reward_in_step for r in rewards)
```

Subclasses whose raw output can exceed `1.0` (or whose best-case is below it)
must override `max_reward_in_step` so `reward_rate` is meaningful. Pure-penalty
rewards set it to `0.0` so they don't inflate the denominator.

Aggregation is performed by [`SumRewardAggregator`](aggregator.py) which sums
the per-reward `(reward, max_step)` pairs across the active list.


## Reward ranges at a glance

All ranges below are **raw** (before multiplication by `weight`). Defaults
are the constructor defaults. The `dense/sparse` column distinguishes
per-step shaping from one-shot episode-end / terminal events — the
bound that's "right" for each is different (see the design-principle
section below). The `bound OK?` column flags whether the current range
fits the design rule for its bucket (`[-1, 1]` per step for dense,
`O(N)` per fire for sparse).

| Reward                              | File                                                                   | raw min                                                            | raw max                  | `max_reward_in_step` | type    | bound OK? |
|-------------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------|----------------------|---------|-----------|
| `TempReward`                        | [temp_reward.py](temp_reward.py)                                       | non-terminal ≈ −58 (dense); `terminate_penalty` (-100) sparse      | 1.0                      | 1.0                  | mixed   | ❌ dense; ✅ sparse |
| `EconomicReward`                    | [economic_reward.py](economic_reward.py)                               | -1.0 (clipped)                                                     | +1.0 (clipped)           | 1.0                  | dense   | ✅        |
| `LongTermEconomicReward`            | [long_term_economic_reward.py](long_term_economic_reward.py)           | 0 most steps; on flush ≈ `−sum(net_kW · E_price)` ≈ ±10³–10⁴       | 0 / +unbounded on flush  | 1.0 (per step; flush still pays 0 max) | sparse | ❌ unbounded |
| `MinimiseEnergyConsumptionReward`   | [energy_consumption_reward.py](energy_consumption_reward.py)           | -1.0 (clipped)                                                     | 0.0                      | 1.0                  | dense   | ✅        |
| `OperatorEnergyControlReward`       | [operator_energy_control_reward.py](operator_energy_control_reward.py) | dense `harsh_penalty` (-4.0); sparse `terminate_penalty` (-100)    | 0.0                      | 0.0                  | mixed   | ❌ dense; ✅ sparse |
| `BatteryTargetReward`               | [battery_target_reward.py](battery_target_reward.py)                   | -1.0                                                               | 0.0                      | 0.0                  | dense   | ✅        |
| `BatteryMgmtReward`                 | [battery_mgmt_reward.py](battery_mgmt_reward.py)                       | `-1/scale` on terminal (-2.0 with `scale=0.5`); else 0             | 0.0                      | 1.0                  | sparse  | ❌ too small for N=288 |
| `EVChargingReward`                  | [ev_charging_reward.py](ev_charging_reward.py)                         | dense connected ≈ 0; sparse failure / min-curve (-100)             | dense 1.0; sparse success (+10) | 1.0 ⚠️       | mixed   | ✅ dense; ✅ sparse (but `max_reward_in_step` mismatched) |
| `EVChargingOnTimeReward`            | [ev_charging_ontime_reward.py](ev_charging_ontime_reward.py)           | `harsh_penalty` (-5.0) — fires every step time-up                  | 1.0                      | 1.0                  | dense   | ❌        |
| `ActionSmoothnessReward`            | [action_smoothness_reward.py](action_smoothness_reward.py)             | `-n_action_keys`                                                   | 0.0                      | `n_action_keys` (auto, set on first call) | dense | ❌ (grows with action-space size) |

After weighting: `[weight * raw_min, weight * raw_max]`.

⚠️ **`EVChargingReward.max_reward_in_step` is not aligned with `success_reward`/
`failure_penalty`.** The class still uses the base default `1.0` while emitting
up to `±100` on disconnect, so `reward_rate` under-counts terminal contributions.

EV rewards return `(0.0, 0.0)` while the EV is disconnected so disconnected
periods do not inflate the `reward_rate` denominator.


## Per-reward details

### TempReward
- **Goal**: Maintain indoor temperature close to desired setpoint.
- **Shape**: piecewise, continuous at `d = zero_reward_diff_celsius` (`d0`).
  - `d < d0`: concave parabola `1 − (d / d0)²` — peaks at `+1.0` at `d = 0`,
    crosses zero at `d = d0`.
  - `d ≥ d0`: linear tail `−(d − d0)` (slope −1 per °C). The °C error is
    pre-clamped at `floor_diff_celsius` (default 30 °C) so the tail can't
    grow without bound when early termination is disabled.
- **Wrong-direction penalty**: when the HP heats while too hot (or cools while
  too cold), adds `wrong_direction_penalty * |a_hp| * min(|°C error|, floor_diff_celsius)`.
- **Terminal**: votes terminate via `should_terminate` when
  `|T_in − T_set| > terminate_diff_celsius`; on the terminating step
  `get_reward` emits `terminate_penalty` (e.g. `−100`).
- **Cap**: `min(reward, 1.0)` (no lower cap apart from the floor on `d`).
- **Range**: `[terminate_penalty, 1.0]`. Outside the terminal branch the
  tail + wrong-direction term can reach ≈ `−58` with defaults
  (`d0 = 2`, `floor = 30`, `wrong_direction_penalty = −1`).

### EconomicReward
- **Goal**: Minimise energy cost; reward grid export (income).
- **Formula**: `−net_power_kW · E_price / reference_power_kW`.
  `reference_power_kW` resolves from `ctxt_operator_max_power_kW` when
  present, else the constructor fallback.
- **Cap**: `np.clip(raw, -1.0, 1.0)`.
- **Range**: `[-1, 1]`.

### LongTermEconomicReward
- **Goal**: Sparse companion to `EconomicReward` — accumulates per-step
  `net_power_kW · E_price` over the episode and emits a single bonus at
  window end OR on early termination (`info["terminated"]`).
- **Per step**: returns `(0.0, 0.0)`.
- **On flush**: returns `(weight * sum, 0.0)` — no clipping, no division by
  `reference_power_kW`. The sum has units of `kW · normalised price`.
- **Reset**: `on_reset` clears the accumulator at episode start.
- **Range (instantaneous)**: `0` most steps; on flush ≈ `±(net_kW · E_price · steps)`,
  which with `net_kW ~ ±30`, `E_price ∈ [-1, 1]`, and `EPISODE_LENGTH = 288`
  reaches `±10³`–`±10⁴`. **Way out of `[-1, 1]` and unbounded.**

### MinimiseEnergyConsumptionReward
- **Goal**: Penalise penalisable consumption above a `threshold_kWh` dead-zone.
- **Formula**: `−max(0, penalisable_kW − threshold_kW) / max_consumption_kW`
  where `threshold_kW = threshold_kWh · 3600 / control_step_s`.
- **Cap**: `np.clip(..., -1.0, 0.0)`.
- **Range**: `[-1, 0]`.

### OperatorEnergyControlReward
- **Goal**: Keep grid power below operator-specified limit.
- **Zones** by `ratio = net_power_kW / ctxt_operator_max_power_kW`:
  - `ratio ≤ soft_threshold_pct` (default 0.9) → `0.0`
  - `soft_threshold_pct < ratio ≤ 1.0` → `exp(−5 · t) − 1` (with
    `t = (ratio − soft) / (1 − soft)`), decaying smoothly from `0` to ≈ `−1`
    at the limit
  - `1.0 < ratio ≤ terminate_threshold_pct` → flat `harsh_penalty`
    (default `−4.0`), then `recovery_steps` steps where the warning-zone
    reward is overridden by `harsh_penalty · exp(−rate · k)` decaying
    from `harsh_penalty` toward 0
- **Terminal**: votes terminate when `ratio > terminate_threshold_pct`
  (default `1.1`); emits `terminate_penalty` (default `−100`) on that step
  when the env honours early termination.
- **`max_reward_in_step`**: `0.0` (pure penalty).
- **Range**: `[terminate_penalty, 0.0]` = `[-100, 0]` by default; without
  termination the floor is `harsh_penalty = -4.0`.

### BatteryTargetReward
- **Goal**: Dead-zone SoC guardrail — punish any excursion outside
  `[min_pct, max_pct]`.
- **Shape**:
  - `min_pct ≤ SoC ≤ max_pct` → `0.0`
  - else → `-1.0`
- **`max_reward_in_step`**: `0.0` (pure penalty).
- **Range**: `[-1, 0]`.

### BatteryMgmtReward
- **Goal**: Terminal-only penalty for ending the episode below the starting
  SoC. Asymmetric: ending equal or above yields 0, so the agent is free to
  cycle the battery mid-episode for arbitrage as long as it replenishes by
  the end.
- **Per step**: returns `(0.0, 0.0)` while `info["terminated"]` is `False`.
- **On terminal step**:
  `deficit = max(0, episode_start_soc − episode_end_soc)`
  `reward  = −deficit / scale`
- **`max_reward_in_step`**: `1.0` (base default, but raw best-case is `0`).
- **Range**: `[-1/scale, 0]` = `[-2, 0]` with the default `scale = 0.5`.

### EVChargingReward
- **Goal**: Track target SoC while connected; judge sessions on
  disconnect; enforce a lazy back-from-target min curve.
- **Disconnected**: `(0.0, 0.0)`.
- **Connected, normal step**:
  - `|SoC − target| < diff_threshold` (default 0.02) → `1.0`
  - else → `exp(−soc_diff_multiplier · |SoC − target|)` (default rate 5),
    so reward stays in `(0, 1]`.
- **Min-curve violation** (only when `info["ev_session_active"]` and
  `s_ev_soc < s_ev_soc_min`): emits `min_curve_violation_penalty`
  (default `−100`); also votes terminate.
- **Disconnect step** (`info["ev_just_disconnected"]`):
  - within `disconnect_soc_tolerance` → `success_reward` (default `+10.0`)
  - else → `failure_penalty` (default `−100`); votes terminate
- **Range**: `[failure_penalty, success_reward]` (default `[-100, 10]`).
- ⚠️ `max_reward_in_step` defaults to `1.0` — not aligned with `success_reward`.

### EVChargingOnTimeReward
- **Goal**: Charge to target SoC before departure; reward only when the
  agent is actively charging (gates `a_lin_ev_charger > 0.0`).
- **Disconnected**: `(0.0, 0.0)`.
- **Connected**:
  - target met (`SoC ≥ target`) → `1.0`
  - no time left, target unmet → `harsh_penalty` (default `−5.0`)
  - else → `max(0, 1 − energy_needed / energy_achievable)`; zeroed when
    not actively charging.
- **Range**: `[harsh_penalty, 1.0]` (default `[-5, 1]`).

### ActionSmoothnessReward
- **Goal**: Penalise oscillation in actions via spectral analysis on a
  rolling FFT window (see class docstring + `docs/action_smoothness_spectral.md`).
- **Per-key penalty**: in `[0, 1]` (fraction of analytic worst-case
  square-wave energy at Nyquist).
- **Aggregation**: summed across action keys, negated → raw reward in
  `[-n_action_keys, 0]`.
- **`max_reward_in_step`**: auto-set to `n_action_keys` on first call
  (overridable via `max_reward_override`).
- **Range**: `[-n_action_keys, 0]`. With several actuators this is well
  outside `[-1, 1]` at the lower end (e.g. `−5` for a 5-actuator config).


## Design principle: bound dense rewards, scale sparse rewards

Two separate rules apply depending on how often a reward fires.

**Dense rewards** (`TempReward`, `EconomicReward`, `Minimise…`,
`OperatorEnergyControl…`, `BatteryTarget…`, `ActionSmoothness`,
connected-step `EVCharging…`) should have a **per-step** raw range
inside `[-1, 1]`. They fire every step, so their cumulative episode
contribution is already `O(N)` with `N = EPISODE_LENGTH = 288`.
Per-step bounds keep:
- weights linearly interpretable across rewards,
- SAC critic targets bounded (no spike-induced gradient explosions),
- `reward_rate = sum(reward) / sum(max_step)` in `[-1, 1]`.

**Sparse rewards** (`LongTermEconomicReward`, `BatteryMgmtReward`, the
terminal branches of `EVChargingReward` and `TempReward`/`OperatorEnergyControl…`)
fire **once per episode**. To stay commensurate with dense rewards in
multi-objective sums, their per-fire magnitude needs to scale with
`N_active`, not be clipped to `±1`:

```
|sparse_one_shot|  ~  |dense_per_step|  ·  N_active
```

Otherwise the sparse signal becomes `1/N_active`-th of the dense
contribution to the return and disappears under the noise. The
constants `±100` / `±10` in the current code reflect this — they aren't
arbitrary, they're "roughly comparable to a full episode of dense
shaping". A blanket clip to `[-1, 1]` would break that balance.

Standalone use (a single sparse reward as the *only* objective) is hard
regardless of the bound — sparse-only learning relies on the algorithm's
return propagation. The standalone benefit of sparse rewards isn't
gradient strength; it's removing reward-shaping bias. The magnitude
mostly matters when mixed with dense rewards.

### What to set `max_reward_in_step` to

- **Dense, positive-and-negative reward** (e.g. `EconomicReward`,
  `TempReward`): `1.0`.
- **Dense, pure-penalty** (e.g. `BatteryTargetReward`,
  `OperatorEnergyControl` non-terminal): `0.0` (no upside to lose).
- **Sparse one-shot**: `N_active` if the reward can also be positive
  (so the maximum-achievable episode return through this reward equals
  one dense step's worth times the episode length, matching the dense
  rewards). `0.0` if it's a pure terminal penalty.


## Out-of-range rewards & redesign proposals

The rewards flagged ❌ in the table fall into two buckets:

1. **Dense rewards leaking out of `[-1, 1]` per step** — these are
   genuine bugs. Tail growth, sums across action keys, or harsh-penalty
   constants set without considering the rest of the curve.
   *Fix*: bound the per-step output to `[-1, 1]` or `[-1, 0]`.

2. **Sparse one-shots emitting `O(10)`–`O(100)`** — these are
   *intentional* and roughly correct in magnitude for an episode of
   length ~288. They are out of `[-1, 1]` because they have to be in
   order to remain commensurate with the cumulative dense signal.
   *Fix*: replace the hard-coded constants with explicit
   episode-length-aware defaults, and align `max_reward_in_step` so
   `reward_rate` works.

The proposals below are split accordingly.


### TempReward — split dense tail from sparse terminal
The current dense tail `−(d − d0)` (slope −1 per °C, floored at
`floor_diff_celsius = 30`) plus additive wrong-direction term reaches
`≈ −58` per step. That's dense-bucket damage.

The `terminate_penalty = -100` is fired once on the terminating step —
sparse-bucket, intentional.

**Proposal (dense part)**: bound the shaped curve to `[-1, 1]` per step
while keeping a non-vanishing gradient over the whole error range.

```
d_eff   = min(|T_in - T_set|, floor_diff_celsius)
d_norm  = d_eff / d0
if d_eff < d0:
    base = 1.0 - d_norm**2                          # in (0, 1]
else:
    base = -(d_eff - d0) / (floor_diff_celsius - d0)  # linear, in [-1, 0]
wrong  = (sign(a_hp) == sign(T_in - T_set) and |a_hp| > 0)
extra  = wrong * wrong_direction_weight * |a_hp| * (d_eff / floor_diff_celsius)
                                                     # in [-wrong_direction_weight, 0]
reward = clip(base + extra, -1.0, 1.0)
```

- **Benefits**: per-step bound preserved; gradient is non-zero across
  the whole tail (linear, not asymptotic — this is *deliberately better
  than `(1-x²)/(1+x²)`* which decays as `1/x²` and loses learning signal
  on large errors); weight tuning trivial.
- **Drawbacks**: piecewise — the seam at `d = d0` is not C¹. (`d`-derivative
  jumps from `-2/d0` to `-1/(floor-d0)`.) RL doesn't actually need C¹,
  but it shows up as a small kink in plots.

**Proposal (sparse terminal)**: keep `terminate_penalty` configurable
with a default scaled to episode length: `default = -EPISODE_LENGTH` (or
`-1.0 * weight_of_dense_companion * EPISODE_LENGTH`). The current `-100`
already lives in that ballpark for `N = 288`; replacing it with `-1`
would shrink the failure signal to one dense step's worth and remove
the reason to avoid comfort violations.

`max_reward_in_step = 1.0` (from the dense part). The terminal does not
contribute a positive max, so it doesn't affect `max_reward_in_step`.


### LongTermEconomicReward — keep the sum, normalise the bound
This is a **sparse one-shot**. The intent is for the cumulative
episode-level cost to be a *single* visible signal at flush. Today the
sum is genuinely unbounded (`O(net_kW · steps)`), which is the actual
bug — but reducing it to `±1` makes it ≪ the cumulative dense
`EconomicReward` and the sparse companion becomes invisible.

**Proposal**:

```
per_step    = clip(- net_power_kW * E_price / reference_power_kW, -1.0, 1.0)
accumulator += per_step
on flush:
    reward = clip(accumulator, -episode_length, +episode_length)
    return weight * reward, weight * episode_length   # max_reward_in_step = episode_length
```

- Per-step contribution is bounded to `[-1, 1]` — matches `EconomicReward`.
- On flush, the sum lives in `[-N, +N]` (where `N = steps_seen`),
  matching the order of magnitude of `sum(EconomicReward)` over the
  episode. This is the sparse/dense parity we want.
- `max_reward_in_step = steps_seen` (set at flush time) makes
  `reward_rate = accumulator / steps_seen ∈ [-1, 1]`.
- **Benefit (multi-objective)**: a single end-of-episode signal whose
  magnitude is on par with the cumulative dense per-step rewards, so
  weights stay interpretable.
- **Benefit (standalone)**: with no other rewards, this provides a
  return-level signal of total energy bill. Hard to learn from alone
  (sparse), but at least the return scales meaningfully with policy
  quality — a uniformly-good policy returns `+N`, a uniformly-bad one
  returns `-N`.
- **Drawback**: per-step clipping discards extreme price spikes.
  Acceptable trade-off — extreme spikes already saturate
  `EconomicReward` the same way.


### OperatorEnergyControlReward — bound the dense branches, keep terminal sparse
- Warning zone: dense, already in `[-1, 0]`. **No change.**
- Over-limit (`ratio > 1`, non-terminal): dense, currently `harsh_penalty = -4`.
  Fix to `-1.0`. Keeps the smooth boundary with the warning zone
  (which approaches `-1` at `ratio = 1`). Recovery curve becomes
  `-1.0 · exp(-rate · k)` ∈ `[-1, 0]`.
- Terminal (`ratio > terminate_threshold_pct`): sparse one-shot.
  Default `terminate_penalty = -100` is already roughly right for
  `N = 288`. Document this as "scales with episode length" and consider
  defaulting to `-EPISODE_LENGTH`. Do **not** reduce to `-1` — that
  would make the terminal indistinguishable from one over-limit step.

`max_reward_in_step = 0.0` (pure-penalty, unchanged).

- **Benefits**: dense branches now respect `[-1, 0]` per step, so SAC
  Q-targets stay bounded between dense violations and dense safe steps.
  The boundary at `ratio = 1` is now smooth: `-0.99 → -1.0` instead of
  `-0.99 → -4`.
- **Drawback**: with termination disabled (`allow_early_termination=False`),
  the only over-limit signal is `-1` per step — identical magnitude to a
  warning step at `ratio = 1`. The agent can no longer distinguish
  "barely over" from "way over" without a steeper continuation. If this
  matters, optionally extend with `-1 · min(1, 1 + over_severity·(ratio - 1))`
  capped at `-1` — but that's just bounded scaling within `[-1, 0]`.


### BatteryMgmtReward — leave as a sparse one-shot, fix the constants
This reward fires only on terminal. It is sparse by design.

**Proposal**:

```
deficit = max(0.0, episode_start_soc - episode_end_soc)   # in [0, 1]
# Per-fire range scaled with episode length so it stays commensurate
# with the cumulative dense rewards. Concretely:
reward  = -clip(deficit / scale, 0.0, 1.0) * episode_length
max_reward_in_step = episode_length   # at the flush step; 0 otherwise
```

Where `scale ∈ (0, 1]` controls how big a SoC deficit saturates the
penalty (default `scale = 1.0` → linear in `deficit`).

- **Benefits**: per-fire range is `[-EPISODE_LENGTH, 0]`, on par with
  what dense rewards accumulate over an episode. SoC management has
  authority in the return.
- **Drawback (standalone)**: still sparse → still hard to learn from
  alone. No way around that without converting it to a per-step shaping
  (which would lose the "free to cycle mid-episode" property the asymmetry
  was designed for).
- **Drawback (multi-objective)**: with the magnitude on par with dense
  returns, picking the right `weight` matters. The previous `-2` was
  effectively muted; `-N` makes it loud. Users will need to tune.


### EVChargingReward — separate connected-step from disconnect terminal
The connected-step branch is **dense** (`[0, 1]`).
The disconnect-step and min-curve-violation branches are **sparse**.

**Proposal**:
- Connected-step shaping: **no change**, already in `[0, 1]`.
- Disconnect / min-curve constants: keep them large and explicit.
  Default them to:
  - `success_reward = +session_length_steps` (or `+EPISODE_LENGTH` as a
    proxy; whichever is closer to the achievable cumulative connected
    reward).
  - `failure_penalty = -session_length_steps` / `-EPISODE_LENGTH`.
  - `min_curve_violation_penalty = -session_length_steps` /
    `-EPISODE_LENGTH`.

  Equivalently: replace the hard-coded `±10` / `±100` with values
  derived at runtime from `info["episode_length"]`. The session-length
  variant is more honest but requires knowing the session length at
  emit time; episode-length is a safe upper bound.
- `max_reward_in_step = episode_length` on disconnect step (only),
  `1.0` while connected, `0.0` while disconnected. This fixes the
  `⚠️ max_reward_in_step ≠ success_reward` mismatch noted in the table.

- **Benefits**: the disconnect verdict carries weight commensurate with
  a full episode's worth of dense shaping — a *single* successful
  charging session contributes ~`+N` to the return, on par with
  perfect dense shaping. Failure cancels that contribution out.
- **Benefits (standalone)**: even alone, the sparse terminal magnitude
  is large enough that PPO/SAC value estimates capture it cleanly.
- **Drawback**: harder to weight by intuition. Old defaults `±10` /
  `±100` had ad-hoc justification; new defaults need to be explained
  in code comments (which is what `info["episode_length"]` and the
  comment in the source can carry).


### EVChargingOnTimeReward — bound the dense penalty
This reward is **dense** while connected. The `harsh_penalty = -5` fires
on every step where time is up and target unmet — not a one-shot.

**Proposal**: bound to `[-1, 0]`. Use `-1.0` as the default
`harsh_penalty`.

- **Benefits**: per-step dense bound restored. The relative importance
  vs. other dense rewards is set by `RewardFunction.weight`, not by an
  out-of-band constant.
- **Drawback**: the "time-up" signal is now indistinguishable in
  magnitude from "regular slow charge". If discriminating those matters,
  set the `weight` higher rather than scaling the constant.


### ActionSmoothnessReward — average across action keys
Dense per-step reward currently in `[-n_keys, 0]`. The
`max_reward_in_step = n_keys` auto-scaling fixes `reward_rate`, but the
raw per-step value still grows linearly with action-space size, so this
reward dominates the per-step sum in configs with many actuators.

**Proposal**: average instead of sum across keys.

```
osc_total = mean(per_key_penalty for ...)    # in [0, 1]
raw       = -osc_total                       # in [-1, 0]
max_reward_in_step = 1.0
```

- **Benefits**: per-step bound `[-1, 0]` regardless of action-space
  size; `weight:` has the same meaning across env configs.
- **Drawback**: an oscillation localised to a single actuator only
  contributes `1/n_keys` of the penalty it did before. If single-key
  oscillation must saturate the penalty, use
  `osc_total = max(per_key_penalty for ...)` instead — still in `[0, 1]`
  with stronger gradient signal at the cost of being max-pooling rather
  than mean-pooling. (My recommendation: `mean` for shaping consistency,
  `max` if oscillation in any actuator must be visible.)


## How to read this in a config

A practical heuristic for the YAML weight column:
- Dense rewards: set `weight` to the relative importance you want
  among per-step shaping (sum of dense weights ~ 1–5 is a good
  target; the per-step max reward is then `sum(weight_i * 1.0)`).
- Sparse rewards: set `weight` to ~1.0; the magnitude is already
  baked into the sparse value (`±N` after these proposals). If you
  scale the sparse reward weight, you're telling the agent that this
  *single event* matters more than `N` perfect dense steps — usually
  too aggressive.


## Adding a new reward

1. Create a subclass of `RewardFunction` in a new file under `rewards/`.
2. Implement `get_reward(actions, states, info=None) -> tuple[float, float]`
   returning `(weight * raw, weight * max_step)`.
3. If the raw best-case differs from `1.0`, set `max_reward_in_step` on
   the class (or assign in `__init__`, as `BatteryTargetReward` does).
4. Keep the raw per-step reward inside `[-1, 1]`. Bounded rewards make
   weight tuning predictable and keep SAC critic targets stable —
   terminal one-shots should not be order(s) of magnitude larger than
   per-step shaping.
5. Optionally implement `should_terminate(actions, states, info)` — the
   env runs all `should_terminate` votes in a Phase-1 pass before any
   `get_reward`, so terminal verdicts are independent of YAML reward
   ordering.
6. Register with `ComponentRegistry.register('reward', MyReward)`.
7. Export from [`rewards/__init__.py`](__init__.py) and add to `__all__`.
8. Add a YAML entry under [`configs/schedules/reward/`](../../configs/schedules/reward/).
9. Update this file with the new reward's range.
