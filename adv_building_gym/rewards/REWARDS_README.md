# Reward Functions

Reward functions inherit from the class `RewardFunction` (defined in `base.py`).
Each function returns `weight * raw_reward` from `get_reward(actions, states)`.

The base class exposes a `max_reward` class attribute (default `1.0`) that
declares the maximum **raw** (unweighted) value a reward class can return per step.
`max_reward` is used for computing `reward_rate = achieved / max_achievable` in
callbacks and evaluation scripts:

```python
max_reward_per_step = sum(r.weight * r.max_reward for r in rewards)
```

Subclasses whose raw output can exceed 1.0 must override `max_reward`.


## Reward ranges

All ranges below are **raw** (before multiplication by `weight`).

```
                         raw min          raw max     max_reward
                        ─────────       ──────────   ──────────
TempReward                  0               1            1.0
EconomicReward             ~-1             ~+1           1.0
MinimiseEnergy             -1              ~+1           1.0
EVChargingOnTime            0               1            1.0
OperatorEnergyControl     -10               1            1.0
UserEnergyNeed              0               1            1.0
```

After weighting: `[weight * raw_min, weight * raw_max]`.


## Per-reward details

### TempReward
- **File**: `temp_reward.py`
- **Goal**: Maintain indoor temperature close to desired setpoint.
- **Formula**:
  - `1.0` if `|actual - desired| < diff_threshold`
  - `exp(-|actual - desired|)` otherwise
- **Range: [0, 1]** -- always non-negative. Perfect comfort gives 1.0,
  large temperature deviation approaches 0 asymptotically.

### EconomicReward
- **File**: `economic_reward.py`
- **Goal**: Minimise energy cost; reward income from grid export.
- **Formula**: `-net_power_kW * price / (max_power_kW * price_max)`
  - `net_power_kW` = sum of `get_electric_consumption()` across all infras
  - Positive net_power (grid import) produces negative reward (cost).
  - Negative net_power (grid export) produces positive reward (income).
- **Range: ~[-1, +1]** -- not explicitly clipped. Bounds depend on whether
  net power can exceed the sum of rated capacities (`max_power_kW`).

### MinimiseEnergyConsumption_Reward
- **File**: `energy_consumption_reward.py`
- **Goal**: Penalise total energy consumption across all actions.
- **Formula**: `-e_consumption / num_action_keys`
  - HP: only the energy dimension (index 0, range [-1, 0]) is summed.
  - Other actions: full value summed.
- **Range: [-1, ~+1]** -- negative when consuming, positive possible when
  actions are negative (e.g. battery discharge).

### EVChargingOnTimeReward
- **File**: `ev_charging_ontime_reward.py`
- **Goal**: Charge EV to target SoC before departure.
- **Formula**:
  - `1.0` if EV not connected (objective fulfilled)
  - `1.0` if `current_soc >= target_soc`
  - `max(0, 1 - energy_needed / energy_achievable)` otherwise
- **Range: [0, 1]** -- always non-negative. On track or target met gives
  1.0, falling behind schedule approaches 0.

### OperatorEnergyControlReward
- **File**: `operator_energy_control_reward.py`
- **Goal**: Keep grid power below operator-specified limit.
- **Three-zone reward** based on `ratio = grid_power_kW / operator_limit_kW`:
  - `ratio <= soft_threshold_pct` (default 0.9): reward = **1.0**
  - `soft_threshold_pct < ratio <= 1.0`: `exp(-5 * (ratio - 0.9) / 0.1)`,
    exponential decay from 1.0 towards ~0.007 at the limit
  - `ratio > 1.0`: **harsh_penalty** (default -2.0)
- **Range: [harsh_penalty, 1]** (default [-2, 1]).
- **Parameters**: `soft_threshold_pct` (default 0.9), `harsh_penalty` (default -2.0).

### UserEnergyNeedReward
- **File**: `user_energy_need_reward.py`
- **Goal**: Match energy production to user demand.
- **Formula**:
  - `1.0` if `actual_energy >= desired_energy`
  - `exp(-shortfall)` otherwise
- **Range: [0, 1]** -- always non-negative. Meeting or exceeding demand
  gives 1.0, shortfall decays exponentially towards 0.


## Adding a new reward

1. Create a subclass of `RewardFunction` in a new file under `rewards/`.
2. Implement `get_reward(actions, states) -> float`.
3. If the raw max differs from 1.0, set `max_reward` on the class.
4. Register with `ComponentRegistry.register('reward', MyReward)`.
5. Add the class to `rewards/__init__.py` and `__all__`.
6. Update this file with the new reward's range.
