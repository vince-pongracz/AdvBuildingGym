# Reward Functions

Reward functions inherit from `RewardFunction` (defined in [base.py](base.py)).
Each `get_reward(actions, states, info)` returns a tuple
`(weighted_reward, weighted_max_step)` — both already multiplied by
`self.weight`.

The base class exposes a `max_reward` class attribute (default `1.0`) that
declares the maximum **raw** (unweighted) value a reward class can return per
step. It is consumed by callbacks / eval to compute
`reward_rate = sum(reward) / sum(max_step)`:

```python
max_reward_per_step = sum(r.weight * r.max_reward for r in rewards)
```

Subclasses whose raw output can exceed `1.0` (or whose best-case is below it)
must override `max_reward` so `reward_rate` is meaningful.

Aggregation is performed by [`SumRewardAggregator`](aggregator.py) which sums
the per-reward `(reward, max_step)` pairs across the active list.


## Reward ranges at a glance

All ranges below are **raw** (before multiplication by `weight`). Defaults
are the constructor defaults; ranges with terminal/harsh penalties are noted.

| Reward                          | File                              | raw min                         | raw max          | `max_reward` |
|---------------------------------|-----------------------------------|---------------------------------|------------------|--------------|
| `TempReward`                    | [temp_reward.py](temp_reward.py)                                       | `terminate_penalty` (e.g. -100) | 1.0              | 1.0          |
| `EconomicReward`                | [economic_reward.py](economic_reward.py)                               | -1.0 (clipped)                  | +1.0 (clipped)   | 1.0          |
| `LongTermEconomicReward`        | [long_term_economic_reward.py](long_term_economic_reward.py)           | -`steps` on flush; 0 otherwise  | +`steps`; 0 otherwise | 1.0 (per-step base; flush emits `weight*steps` max) |
| `MinimiseEnergyConsumptionReward` | [energy_consumption_reward.py](energy_consumption_reward.py)         | -1.0 (clipped)                  | 0.0              | 1.0          |
| `UserEnergyNeedReward`          | [user_energy_need_reward.py](user_energy_need_reward.py)               | 0.0 (`exp(-shortfall)→0`)       | 1.0              | 1.0          |
| `OperatorEnergyControlReward`   | [operator_energy_control_reward.py](operator_energy_control_reward.py) | `terminate_penalty` (-100)      | 1.0              | 1.0          |
| `BatteryTargetReward`           | [battery_target_reward.py](battery_target_reward.py)                   | `max_penalty` (-3.0)            | `in_band_reward` (1.0) | 1.0 (= `in_band_reward`) |
| `EVChargingReward`              | [ev_charging_reward.py](ev_charging_reward.py)                         | `failure_penalty` / `min_curve_violation_penalty` (-100) | `success_reward` (10.0) on disconnect; 1.0 while connected | 1.0 ⚠️ |
| `EVChargingOnTimeReward`        | [ev_charging_ontime_reward.py](ev_charging_ontime_reward.py)           | `harsh_penalty` (-5.0)          | 1.0              | 1.0          |
| `ActionSmoothnessReward`        | [action_smoothness_reward.py](action_smoothness_reward.py)             | `max_reward - n_action_keys` (-n+0.3) | `max_reward` (0.3) | 0.3          |

After weighting: `[weight * raw_min, weight * raw_max]`.

⚠️ **`EVChargingReward.max_reward` is not aligned with `success_reward`/`failure_penalty`.**
The class still uses the base default `1.0` while emitting up to `±100` on
disconnect, so `reward_rate` underestimates the achieved share whenever a
session terminates. Override `max_reward` (or rescale the terminal values) if
the rate metric matters for this experiment.

EV rewards return `(0.0, 0.0)` while the EV is disconnected so disconnected
periods do not inflate the `reward_rate` denominator.


## Per-reward details

### TempReward
- **Goal**: Maintain indoor temperature close to desired setpoint.
- **Shape**: `-d² + e^(-d)` where `d` is normalised temperature error
  scaled so the curve crosses zero at `zero_reward_diff_celsius`. Smooth
  everywhere → always provides a gradient.
- **Wrong-direction penalty**: heating when too hot (or cooling when too
  cold) subtracts `wrong_direction_penalty * |energy| * |temp_error|`.
- **Terminal**: votes terminate via `should_terminate` when
  `|T_in - T_set| > terminate_diff_celsius`; on the terminating step
  `get_reward` emits `terminate_penalty` (e.g. `-100`).
- **Cap**: `min(reward, 1.0)` (no lower clip apart from the terminal branch).
- **Range**: `[terminate_penalty, 1.0]`.

### EconomicReward
- **Goal**: Minimise energy cost; reward grid export (income).
- **Formula**: `-net_power_kW * E_price / reference_power_kW`.
  `reference_power_kW` resolves from `ctxt_operator_max_power_kW` when
  present, else the constructor fallback.
- **Cap**: `np.clip(raw, -1.0, 1.0)`.
- **Range**: `[-1, 1]`.

### LongTermEconomicReward
- **Goal**: Sparse companion to `EconomicReward` — accumulates the same
  per-step quantity over a window (`info["episode_length"]`) and emits a
  single bonus at window end OR on early termination
  (`info["terminated"]`).
- **Per step**: returns `(0.0, 0.0)`.
- **On flush**: returns
  `(weight * clip(sum, -steps, +steps), weight * steps)`
  where `steps` is the actually accumulated count (so `reward_rate` stays
  consistent under early termination).
- **Reset detection**: a `_lt_econ_active_<id>` sentinel in `info` —
  missing ⇒ fresh episode (the env clears `_component_info` on `reset`).
- **Range (instantaneous)**: `0` most steps; `[-steps, +steps]` on flush.

### MinimiseEnergyConsumptionReward
- **Goal**: Penalise penalisable consumption.
- **Formula**: `-info["penalisable_power_kW"] / info["max_consumption_kW"]`.
- **Cap**: `np.clip(..., -1.0, 0.0)`. Returns `0` if either denominator
  or `info` keys are missing/zero.
- **Range**: `[-1, 0]`.

### UserEnergyNeedReward
- **Goal**: Match actual energy to `s_desired_energy_need`.
- **Formula**: `1.0` when meeting/exceeding demand; `exp(-shortfall)`
  otherwise — penalises only underproduction.
- **Range**: `[0, 1]`.

### OperatorEnergyControlReward
- **Goal**: Keep grid power below operator-specified limit.
- **Three-zone reward** by `ratio = net_power_kW / operator_limit_kW`:
  - `ratio ≤ soft_threshold_pct` (default 0.9) → `1.0`
  - `soft_threshold_pct < ratio ≤ 1.0` → `exp(-5 · (ratio - 0.9) / 0.1)`,
    decaying from 1.0 to ~0.007 at the limit
  - `ratio > 1.0` → flat `harsh_penalty` (default `-4.0`), then a
    multi-step exponential recovery (`recovery_steps` long) where the
    ceiling is `harsh_penalty * exp(-rate * k)` rising back towards 0
- **Terminal**: votes terminate when `ratio > terminate_threshold_pct`
  (default `1.1`); emits `terminate_penalty` (default `-100`) on that step.
- **Range**: `[terminate_penalty, 1.0]` (default `[-100, 1]`).

### BatteryTargetReward
- **Goal**: Keep SoC inside a healthy band; punish hard limits.
- **Shape** (continuous across the band edges):
  - `min_pct ≤ SoC ≤ max_pct` → `in_band_reward`
  - `SoC < min_pct` → `in_band_reward + (max_penalty - in_band_reward) ·
    ((min_pct - SoC) / min_pct)²`
  - `SoC > max_pct` → `in_band_reward + (max_penalty - in_band_reward) ·
    ((SoC - max_pct) / (1 - max_pct))²`
- Floors at `max_penalty`. `max_reward` is set to `in_band_reward`.
- **Range** (defaults `min_pct=0.15`, `max_pct=0.85`,
  `in_band_reward=1.0`, `max_penalty=-3.0`): `[-3.0, 1.0]`.

### EVChargingReward
- **Goal**: Track target SoC while connected; judge sessions on
  disconnect; enforce a lazy back-from-target min curve.
- **Connected, normal step**:
  - `|SoC - target| < diff_threshold` (default 0.02) → `1.0`
  - else → `exp(-soc_diff_multiplier · |SoC - target|)` (default rate 5)
- **Min-curve violation** (only when `info["ev_session_active"]` and
  `s_ev_soc < s_ev_soc_min`): emits `min_curve_violation_penalty`
  (default `-100`); also votes terminate.
- **Disconnect step** (`info["ev_just_disconnected"]`):
  - within `disconnect_soc_tolerance` → `success_reward` (default `10.0`)
  - else → `failure_penalty` (default `-100`); votes terminate
- **Disconnected**: returns `(0.0, 0.0)`.
- **Range**: `[failure_penalty, success_reward]` (default `[-100, 10]`).
- ⚠️ `max_reward` defaults to `1.0` — not aligned with `success_reward`.

### EVChargingOnTimeReward
- **Goal**: Charge to target SoC before departure; reward only when the
  agent is actually charging (gates `a_lin_ev_charger ≥ 0.01`).
- **Disconnected**: `(0.0, 0.0)`.
- **Connected**:
  - target met (`SoC ≥ target`) → `1.0`
  - no time left, target unmet → `harsh_penalty` (default `-5.0`)
  - else → `max(0, 1 - energy_needed / energy_achievable)`
    (zeroed if not actively charging)
- **Range**: `[harsh_penalty, 1.0]` (default `[-5, 1]`).

### ActionSmoothnessReward
- **Goal**: Penalise oscillation (sign-reversal of first differences in
  the action history) weighted by squared acceleration. Reads the rolling
  history from `info["action_history"]` (env deque, separate from the
  policy-visible `<action_key>_prev` channels).
- **Per-key penalty**: in `[-1, 0]`, the fraction of worst-case
  oscillation. Sum across keys, then add `max_reward = 0.3` to shift the
  best case up.
- **Properties**: smooth ramps → 0; single justified step from steady
  state → 0; persistent oscillation → heavy penalty.
- **Range**: `[max_reward - n_action_keys, max_reward]`
  = `[0.3 - n_keys, 0.3]`.


## Adding a new reward

1. Create a subclass of `RewardFunction` in a new file under `rewards/`.
2. Implement `get_reward(actions, states, info=None) -> tuple[float, float]`
   returning `(weight * raw, weight * max_step)`.
3. If the raw best-case differs from `1.0`, set `max_reward` on the class
   (or assign in `__init__`, as `BatteryTargetReward` does).
4. Optionally implement `should_terminate(actions, states, info)` — the
   env runs all `should_terminate` votes in a Phase-1 pass before any
   `get_reward`, so terminal verdicts are independent of YAML reward
   ordering.
5. Register with `ComponentRegistry.register('reward', MyReward)`.
6. Export from [`rewards/__init__.py`](__init__.py) and add to `__all__`.
7. Add a YAML entry under [`configs/reward_cfg/`](../../configs/reward_cfg/).
8. Update this file with the new reward's range.
