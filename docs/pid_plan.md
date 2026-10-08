# PID Controller as an RLlib Algorithm — Implementation Plan

## Motivation

Implement a deterministic PID/rule-based controller as a first-class RLlib Algorithm using the **new API stack** (RLModule + Learner, EnvRunner + ConnectorV2). This allows:

- Running PID through RLlib's rollout and evaluation machinery
- Fair comparison against RL algorithms (PPO, SAC) using identical metrics and callbacks
- Hyperparameter tuning of PID gains via Ray Tune
- Consistent checkpointing, logging, and episode metrics

## Architecture Overview

```
PIDRLModule (RLModule)          — deterministic obs → action mapping
    ├── HP channel              — PID on temperature error
    ├── Battery channel         — rule-based on price + SoC
    ├── EV Charger channel      — rule-based on SoC + connection status
    └── Solar channel           — pass-through (no control needed)

PIDAlgorithm (Algorithm)        — evaluation-only training_step()
PIDConfig (AlgorithmConfig)     — holds PID gains and channel specs
```

## Improvements Over Original ChatGPT Plan

| # | Issue in original plan | Improvement |
|---|------------------------|-------------|
| 1 | `dt = 0.02` | Fixed to `dt = 300` (control_step in seconds) |
| 2 | Assumes Dict obs reach RLModule | Skip `FlattenObservations` connector for PID — receive Dict obs with semantic keys |
| 3 | Outputs Dict actions | Output flat `Box` actions matching env's flattened action space |
| 4 | No anti-windup | Add integral clamping to prevent windup |
| 5 | Derivative spike on first step | Set derivative to 0 when `prev_error is None` (matches existing controller) |
| 6 | Generic obs keys (`v_ref`, `yaw`) | Use actual env keys: `temp_in_norm`, `desired_temp_in_norm`, `E_price`, etc. |
| 7 | No integration with project | Integrate with `select_model.py`, `common_model_config.py`, `run_train_ray.py` |
| 8 | Ignores optional infrastructure | Guard each channel with obs key existence checks (e.g., no solar in some configs) |
| 9 | No Ray Tune config | Concrete `tune.loguniform` ranges for gain search |
| 10 | Single-channel PID only | Multi-channel: PID for HP, rule-based for Battery/EV/Solar |

---

## 1. PIDRLModule — New API Stack

The RLModule receives Dict observations directly (FlattenObservations connector is skipped for PID). It outputs a flat action vector matching the env's `Box` action space.

**Key design decisions:**
- PID state (integral, prev_error) stored via `Columns.STATE_IN` / `STATE_OUT` for proper per-episode reset
- Each infrastructure channel checks whether its obs keys exist before computing
- Action output order must match `env.action_space_keys` (HP, Solar, Battery, EV in registration order)

### File: `adv_building_gym/ray_training/pid_module.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.columns import Columns


@dataclass
class PIDChannelSpec:
    """Specification for a single PID control channel."""
    action_key: str                 # Key in env's _dict_action_space
    action_dim: int                 # Dimension of this action component
    setpoint_key: str               # Obs key for setpoint
    measurement_key: str            # Obs key for measurement
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    integral_limit: float = 10.0   # Anti-windup clamp
    clip_low: float = -1.0
    clip_high: float = 1.0


class PIDRLModule(RLModule, nn.Module):
    """
    New-API-stack RLModule implementing a multi-channel PID/rule-based controller.

    Receives Dict observations directly (FlattenObservations connector is skipped).
    Outputs a flat action tensor matching the env's Box action space.

    Channels:
      - HP: PID on temperature error (desired_temp_in_norm - temp_in_norm)
      - Battery: rule-based on E_price and battery_pct
      - EV Charger: rule-based on ev_soc, ev_target_soc, ev_connected
      - Solar: pass-through (solar_action = -solar_irradiance)
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config: Dict[str, Any] | None = None,
        *,
        pid_channels: List[Dict[str, Any]],
        dt: float = 300.0,
        action_keys_order: List[str],   # Must match env.action_space_keys
        device: str | torch.device = "cpu",
    ):
        nn.Module.__init__(self)
        RLModule.__init__(
            self, observation_space, action_space, model_config=model_config
        )

        self._device = torch.device(device)
        self.dt = dt
        self.action_keys_order = action_keys_order

        # Build channel specs
        self.channels: List[PIDChannelSpec] = [
            PIDChannelSpec(**ch) for ch in pid_channels
        ]
        self._channel_by_key = {ch.action_key: ch for ch in self.channels}

        # Total flat action dim
        self.total_action_dim = sum(ch.action_dim for ch in self.channels)

        self.to(self._device)

    # ---- State handling (per-episode PID memory) ----
    def get_initial_state(self) -> Dict[str, torch.Tensor]:
        state = {}
        for ch in self.channels:
            state[f"integral::{ch.action_key}"] = torch.zeros(
                ch.action_dim, dtype=torch.float32, device=self._device
            )
            state[f"prev_error::{ch.action_key}"] = torch.zeros(
                ch.action_dim, dtype=torch.float32, device=self._device
            )
            # Flag: 0 = first step (no derivative), 1 = has prev_error
            state[f"has_prev::{ch.action_key}"] = torch.zeros(
                1, dtype=torch.float32, device=self._device
            )
        return state

    # ---- Core PID computation ----
    def _compute_actions(
        self,
        obs: Dict[str, torch.Tensor],
        state_in: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute flat action vector from Dict observations.
        obs: dict of [B, *shape] tensors
        state_in: dict of [B, *shape] tensors (batched by RLlib)
        Returns: (actions [B, total_action_dim], state_out)
        """
        state_out = dict(state_in)
        action_parts = []

        for act_key in self.action_keys_order:
            ch = self._channel_by_key.get(act_key)
            if ch is None:
                # Unknown action key — output zeros
                # Determine dim from observation (fallback)
                action_parts.append(torch.zeros(1))
                continue

            # Check if required obs keys exist
            sp_key = ch.setpoint_key
            meas_key = ch.measurement_key

            if sp_key not in obs or meas_key not in obs:
                # Infrastructure not in this config — output neutral action
                batch_size = next(iter(obs.values())).shape[0]
                neutral = torch.zeros(
                    batch_size, ch.action_dim,
                    dtype=torch.float32, device=self._device
                )
                action_parts.append(neutral)
                continue

            sp = obs[sp_key].to(self._device).float()
            meas = obs[meas_key].to(self._device).float()
            error = sp - meas

            integ_key = f"integral::{ch.action_key}"
            prev_key = f"prev_error::{ch.action_key}"
            has_prev_key = f"has_prev::{ch.action_key}"

            integral = state_in[integ_key]
            prev_error = state_in[prev_key]
            has_prev = state_in[has_prev_key]

            # Update integral with anti-windup
            integral_new = integral + error * self.dt
            integral_new = torch.clamp(
                integral_new, -ch.integral_limit, ch.integral_limit
            )

            # Derivative (zero on first step to avoid spike)
            derivative = torch.where(
                has_prev > 0.5,
                (error - prev_error) / self.dt,
                torch.zeros_like(error),
            )

            u = ch.kp * error + ch.ki * integral_new + ch.kd * derivative
            u = torch.clamp(u, ch.clip_low, ch.clip_high)

            action_parts.append(u)

            state_out[integ_key] = integral_new
            state_out[prev_key] = error
            state_out[has_prev_key] = torch.ones_like(has_prev)

        # Concatenate all action components into flat vector [B, total_dim]
        actions = torch.cat(action_parts, dim=-1)
        return actions, state_out

    def forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        obs = batch[Columns.OBS]
        state_in = batch.get(Columns.STATE_IN, self.get_initial_state())
        actions, state_out = self._compute_actions(obs, state_in)
        return {
            Columns.ACTIONS: actions,
            Columns.STATE_OUT: state_out,
        }

    def forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self.forward_inference(batch, **kwargs)

    def forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self.forward_inference(batch, **kwargs)
```

---

## 2. PID Channel Specifications

Based on the actual AdvBuildingGym observation and action spaces:

```python
# Default PID channel configurations
# Action key order must match env.action_space_keys

HP_CHANNEL = {
    "action_key": "HP_action",
    "action_dim": 2,                    # [energy_level, mode]
    "setpoint_key": "desired_temp_in_norm",
    "measurement_key": "temp_in_norm",
    "kp": 0.15,                         # From existing PIDController
    "ki": 0.0002,
    "kd": 0.01,
    "integral_limit": 10.0,
    "clip_low": 0.0,                    # HP action range [0, 1]
    "clip_high": 1.0,
}

BATTERY_CHANNEL = {
    "action_key": "battery_action",
    "action_dim": 1,
    # The battery component is hardware-only and no longer publishes a
    # target SoC into the obs space.  A PID-style controller must supply
    # its own setpoint.
    "setpoint_key": "battery_target_pct",   # Caller-supplied; not in obs
    "measurement_key": "battery_pct",       # Current SoC
    "kp": 1.0,                              # Tune via Ray Tune
    "ki": 0.0,
    "kd": 0.0,
    "integral_limit": 5.0,
    "clip_low": -1.0,                       # [-1, 1]: discharge ↔ charge
    "clip_high": 1.0,
}

EV_CHARGER_CHANNEL = {
    "action_key": "lin_ev_charger_action",
    "action_dim": 1,
    "setpoint_key": "ev_target_soc",
    "measurement_key": "ev_soc",
    "kp": 2.0,                             # Aggressive charging when needed
    "ki": 0.0,
    "kd": 0.0,
    "integral_limit": 5.0,
    "clip_low": -1.0,                       # [-1, 1] if V2G, [0, 1] if not
    "clip_high": 1.0,
}

SOLAR_CHANNEL = {
    "action_key": "solar_action",
    "action_dim": 1,
    "setpoint_key": "solar_irradiance",     # "setpoint" = irradiance
    "measurement_key": "solar_irradiance",  # Same key → error = 0
    # Solar pass-through: action = -irradiance (produce all available)
    # NOTE: This is a simplification. The actual solar action should be
    # computed differently — see note below.
    "kp": 0.0,
    "ki": 0.0,
    "kd": 0.0,
    "integral_limit": 1.0,
    "clip_low": -1.0,
    "clip_high": 0.0,
}
```

> **Note on Solar:** Solar panels are passive producers. The PID framework doesn't naturally fit here. A better approach is to override the solar action directly in `_compute_actions()` as `action = -solar_irradiance` (produce all available solar energy). This should be handled as a special case in the RLModule, not as a PID channel.

> **Note on HP action (2D):** The HP has a 2D action `[energy_level, mode]`. The PID error maps naturally to `energy_level`, but `mode` (heat/cool/idle) should be determined by the sign of the temperature error: `error > 0 → mode > 0.6 (heat)`, `error < 0 → mode < 0.4 (cool)`, `|error| small → mode ≈ 0.5 (idle)`. This requires special handling in `_compute_actions()`.

---

## 3. PIDAlgorithm — Evaluation-Only Algorithm

### File: `adv_building_gym/ray_training/pid_algorithm.py`

```python
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig


class PIDConfig(AlgorithmConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or PIDAlgorithm)
        self.pid_channels = []
        self.pid_dt = 300.0

    def pid(self, *, pid_channels: list, dt: float = 300.0):
        """Configure PID channel specifications."""
        self.pid_channels = pid_channels
        self.pid_dt = dt
        return self


class PIDAlgorithm(Algorithm):
    @classmethod
    def get_default_config(cls):
        return PIDConfig()

    def training_step(self):
        """No learning — just run evaluation/rollouts and report metrics."""
        results = self.evaluate()
        eval_results = results.get("evaluation", {})
        return {
            "episode_reward_mean": eval_results.get("episode_reward_mean", 0.0),
            "evaluation": eval_results,
        }
```

---

## 4. Integration with Existing Project

### 4.1 `select_model.py` — Add PID case

```python
# In select_model(), add:
elif algorithm == "pid":
    from adv_building_gym.ray_training.pid_algorithm import PIDConfig
    config = PIDConfig()
    config.pid(
        pid_channels=[HP_CHANNEL, BATTERY_CHANNEL, EV_CHARGER_CHANNEL, SOLAR_CHANNEL],
        dt=300.0,
    )
    # No rl_module model_config needed — PIDRLModule ignores it
    # Provide RLModuleSpec instead:
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    from adv_building_gym.ray_training.pid_module import PIDRLModule
    config.rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=PIDRLModule,
            module_kwargs={
                "pid_channels": [HP_CHANNEL, BATTERY_CHANNEL, EV_CHARGER_CHANNEL, SOLAR_CHANNEL],
                "dt": 300.0,
                "action_keys_order": ["HP_action", "battery_action", "lin_ev_charger_action", "solar_action"],
            },
        )
    )
    return config
```

### 4.2 `common_model_config.py` — Skip FlattenObservations for PID

```python
# In common_model_config(), conditionally set env_to_module_connector:
from adv_building_gym.ray_training.pid_algorithm import PIDConfig

if isinstance(config, PIDConfig):
    # PID needs Dict obs with semantic keys — do NOT flatten
    config.env_runners(
        num_env_runners=num_env_runners,
        num_envs_per_env_runner=1,
        num_cpus_per_env_runner=num_cpus_per_env_runner,
        num_gpus_per_env_runner=0,
        # No FlattenObservations — PID accesses obs by key
    )
else:
    config.env_runners(
        ...
        env_to_module_connector=lambda env, spaces, device: FlattenObservations(),
    )
```

### 4.3 `run_train_ray.py` — Accept `--algorithm pid`

Add `"pid"` to the `choices` list in the argparse `--algorithm` argument.

### 4.4 `run_eval_ray.py` — Evaluate PID checkpoints

The existing eval script uses `algo.compute_single_action(obs, explore=False)` which should work with the PID algorithm since PIDRLModule implements `forward_inference()`. No changes needed if checkpoint loading works correctly.

---

## 5. Ray Tune Hyperparameter Search

```python
from ray import tune

# Tune PID gains as hyperparameters
tune.Tuner(
    PIDAlgorithm,
    param_space=config.to_dict() | {
        # HP gains
        "pid_channels": [
            {
                **HP_CHANNEL,
                "kp": tune.loguniform(1e-3, 1e1),
                "ki": tune.loguniform(1e-6, 1e-1),
                "kd": tune.loguniform(1e-4, 1e0),
            },
            {
                **BATTERY_CHANNEL,
                "kp": tune.loguniform(1e-2, 1e2),
            },
            {
                **EV_CHARGER_CHANNEL,
                "kp": tune.loguniform(1e-1, 1e2),
            },
            SOLAR_CHANNEL,  # No tuning needed
        ],
    },
    tune_config=tune.TuneConfig(
        metric="episode_reward_mean",
        mode="max",
        num_samples=50,
    ),
).fit()
```

---

## 6. Files to Create / Modify

| File | Action | Description |
|------|--------|-------------|
| `adv_building_gym/ray_training/pid_module.py` | **Create** | PIDRLModule implementation |
| `adv_building_gym/ray_training/pid_algorithm.py` | **Create** | PIDAlgorithm + PIDConfig |
| `adv_building_gym/ray_training/select_model.py` | **Modify** | Add `"pid"` algorithm case |
| `adv_building_gym/ray_training/common_model_config.py` | **Modify** | Skip FlattenObservations for PID, provide obs_space |
| `run_train_ray.py` | **Modify** | Add `"pid"` to `--algorithm` choices |

---

## 7. Open Questions / Considerations

1. **HP 2D action:** The PID error gives a single scalar, but HP needs `[energy_level, mode]`. Mode should be derived from the sign of the error (heat vs cool). This needs custom logic in `_compute_actions()`.

2. **Solar pass-through:** Solar doesn't fit PID. It should be a direct `action = -irradiance` mapping, handled as a special case.

3. **Battery/EV strategy:** Simple P-control on SoC error is a starting point, but a price-aware strategy (charge when cheap, discharge when expensive) would be more effective. This could be a second iteration.

4. **Integral reset on episode boundary:** RLlib should reset `STATE_IN` at episode start via `get_initial_state()`. Verify this works correctly — if integral carries across episodes, it causes windup.

5. **action_keys_order:** This must exactly match `env.action_space_keys` from `building_adv.py`. The order is determined by infrastructure registration order in `Config.__post_init__()`. Currently: `["HP_action", "solar_action", "battery_action", "lin_ev_charger_action"]` — verify this.

---

## 8. Verification Plan

1. Run `python run_train_ray.py --algorithm pid --episodes 35` and verify:
   - No crashes during rollout
   - Episode metrics logged (achieved_reward, cum_E_kWh)
   - Checkpoint saved successfully

2. Run `python run_eval_ray.py --checkpoint <pid_checkpoint_path>` and verify:
   - Evaluation completes
   - Metrics are reasonable (temperature stays near setpoint)

3. Test with different configs (with/without solar, with/without EV) and verify:
   - Missing infrastructure channels output neutral actions
   - No key errors from missing obs keys

4. Run Ray Tune gain search and verify:
   - Different gain combinations are sampled
   - Best gains improve episode reward over defaults
