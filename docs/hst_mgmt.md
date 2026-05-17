# History Management for the Policy — Design Notes

Summary of a design discussion about exposing temporal history to PPO/SAC
policies in AdvBuildingGym (Ray RLlib new API stack).

**Status — SUPERSEDED.** The original design (per-key strided
`StridedHistoryConnector` + `build_env_to_module_connectors` in
`adv_building_gym/ray_training/history_connector.py`) has been replaced
by an in-env Gymnasium wrapper, `adv_building_gym/envs/history_wrapper.py`,
which adds `s_hst_<key>` Dict entries with the same strided semantics.
Configuration moved from `training_params.hst.*` to
`env_meta.hst_env_wrapper.{tracked_keys, offsets}` in trial YAMLs (see
`configs/trial_cfgs/TRIAL_HELP.md`). The discussion below is preserved
for historical context.

---

## 1. Current state of history in the codebase

History is baked **inside the environment**, not handled by RLlib connectors.
Three mechanisms coexist:

1. **Action history** — `adv_building_gym/envs/building_adv.py:235-247`
   - For every action key, an `hst_{key}` Box of shape
     `(ACTION_HISTORY_LENGTH, *action_shape)` is added to the observation dict.
   - Updated each `step()` via shift-and-insert (`building_adv.py:561-570`).
   - Consumed by `ActionSmoothnessReward` (`action_smoothness_reward.py:59`)
     and exposed to the policy through `FlattenObservations`.
   - Window length: `EnvConfig.ACTION_HISTORY_LENGTH = 4`
     (`config/env_config.py:44`).

2. **Per-component state history** — e.g. `hst_s_battery_pct`, `hst_s_ev_soc`
   declared and maintained by the respective infrastructure components
   (`battery_linear.py`, `battery_tremblay.py`, `linear_ev_charger.py`).

3. **Dead code** — `utils/temporal_features.py` (`TemporalFeatureBuffer`) is
   not wired into `building_adv.py`.

All three rely on `FlattenObservations`
(`common_model_config.py:259`) to flatten the Dict observation — including the
stacked history entries — into the flat vector the MLP RLModule sees.

---

## 2. RLlib-native alternatives (new API stack)

Two orthogonal mechanisms to give the policy history:

- **Frame stacking via env-to-module connector**
  (`ray.rllib.connectors.env_to_module.FrameStackingEnvToModule`) paired with
  `FrameStackingLearner` on the learner side.
  - Stateful, episode-aware ring buffer in the connector.
  - Replay buffer (SAC) stores only single frames; the learner connector
    reconstructs the stack at train time → big memory win for off-policy.
  - PPO still needs the learner connector so training/rollout spaces agree.
- **Recurrent RLModule (LSTM)** via
  `DefaultModelConfig(use_lstm=True, max_seq_len=..., lstm_cell_size=...)`.
  - No explicit stacking; RLlib feeds sequences, BPTT maintains latent memory.

### Data flow with a connector

```
env.step() → obs_t
         → FrameStackingEnvToModule (ring buffer of last N)
         → FlattenObservations
         → RLModule.forward_inference  # input is (B, N*obs_dim)
```

Frame buffer lives on the env runner (rollout) and is reconstructed on the
learner via the paired learner connector. Pre-padding with the first obs at
reset handles the cold-start steps.

### MLP vs LSTM

- **Stacking + MLP**: first layer becomes `Linear(N*obs_dim, 32)`; the network
  learns which slice is which lag. No architectural change.
- **LSTM only**: drops stacking; policy sees one obs per step but carries
  hidden state.
- **Stacking + LSTM**: each LSTM step sees a stacked window; BPTT over
  `max_seq_len` policy steps. Effective receptive field = stacks × seq_len.

### Tradeoffs vs. current in-env approach

| Aspect | In-env (current) | Connector-based |
|---|---|---|
| Reward funcs can read history | Yes (via `self.state["hst_*"]`) | No — connector output only reaches the RLModule |
| Dict obs of heterogeneous keys | Selective per key | Uniform stacking; selective stacking needs custom connector |
| Memory on learner (SAC replay) | Stored N× | Reconstructed → much smaller buffer |
| Eval path | Env owns state | Must add same connector in `eval_runner.py:132` |

`ActionSmoothnessReward` depends on `hst_*` being in `self.state`, so a
wholesale switch to connectors isn't free. A **hybrid** is the realistic path:
keep `hst_{action}` in-env for the reward, move observation stacking
(weather, price, temperature) to a connector.

---

## 3. Short + strided history (the main use case discussed)

Goal: short dense window (last 5 steps) *plus* dilated/strided lags
(−5, −10, −15, −20, …) — i.e. a dilation-style observation. Build a custom
`ConnectorV2` that extracts exactly those offsets from the episode and
concatenates them; optionally add an LSTM on top for latent long-term
context.

### Correct but subtle points

- The strided-stacking idea is sound and is more sample-efficient than
  stacking 20 consecutive frames when the env has periodic or delayed
  dynamics.
- Stacking and LSTM solve different problems (explicit velocity/delay
  features vs. latent context memory) and can be combined.
- `episode.get_observations(-1)` returns the most recent obs.

### Mistakes to avoid in an implementation

1. **`env_to_module_connector` lambda takes three args**, not one:
   `lambda env, spaces, device: ...` (see
   `common_model_config.py:259`). The one-arg form `lambda env: ...` will
   `TypeError`.
2. **New API stack uses `model_config=DefaultModelConfig(...)`**, not
   `model_config_dict={...}`. Mixing old and new API stack patterns is
   explicitly prohibited by `CLAUDE.md`.
3. **ConnectorV2 `__call__` kwarg is `batch`**, not `data`, and obs lives
   under a module-id key (`batch[DEFAULT_MODULE_ID][Columns.OBS]`). Writing
   `data["obs"][i] = …` does not match the actual batch layout.
4. **Env-to-module connectors run only on the EnvRunner.** For train/eval
   parity you need either a matching `learner_connector` (pattern:
   `FrameStackingEnvToModule` ↔ `FrameStackingLearner`) or to write the
   stacked obs back into the episode.
5. **This project's raw obs is a Dict** (`building_adv.py:250`). A custom
   stacking connector must either run *after* `FlattenObservations` (input is
   then a flat Box) or iterate Dict keys explicitly. `np.repeat` on
   `Box.low`/`.high` won't work for Dict.
6. **`get_observations(0)` is the first obs of the episode**, not the current
   one. Misleading indexing comments will bite future editors.
7. **SAC + LSTM is not purely automatic.** New stack wires sequence batching,
   but `max_seq_len`, burn-in, and the `EpisodeReplayBuffer` config still
   need attention.
8. **Padding.** `get_observations` on a missing index may return an empty
   list rather than raising; prefer an explicit length check over
   `try/except IndexError`.
9. **Eval parity.** `eval_runner.py:132` instantiates connectors manually.
   Any training-side connector change must be mirrored there or eval silently
   uses a different obs layout.

### Corrected skeleton

```python
# Runs AFTER FlattenObservations so input space is a flat Box.
import numpy as np
import gymnasium as gym
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig


class StridedHistoryConnector(ConnectorV2):
    OFFSETS = (0, -1, -2, -3, -4, -6, -12, -18, -24, -36, -48, -60, -72, -84, -96, -108, -120, -132, -144)

    def recompute_output_observation_space(self, input_observation_space):
        low = np.tile(input_observation_space.low, len(self.OFFSETS))
        high = np.tile(input_observation_space.high, len(self.OFFSETS))
        return gym.spaces.Box(low, high, dtype=input_observation_space.dtype)

    def __call__(self, *, rl_module, batch, episodes, **kwargs):
        zero = np.zeros_like(self.input_observation_space.low)
        for mid, mbatch in batch.items():
            new_obs = []
            for ep in episodes:
                frames = []
                for off in self.OFFSETS:
                    idx = -1 + off
                    frames.append(
                        ep.get_observations(idx) if len(ep) >= -idx else zero
                    )
                new_obs.append(np.concatenate(frames, axis=-1))
            mbatch[Columns.OBS] = np.asarray(new_obs, dtype=np.float32)
        return batch


config.env_runners(
    env_to_module_connector=lambda env, spaces, device: [
        FlattenObservations(),
        StridedHistoryConnector(),
    ],
)
config.rl_module(model_config=DefaultModelConfig(
    fcnet_hiddens=[32, 32, 32],
    use_lstm=True,
    max_seq_len=20,
    lstm_cell_size=256,
))
```

This is a rollout-side sketch only. A matching learner connector and a shared
pipeline used by `eval_runner.py` are still needed for end-to-end correctness.

---

## 4. Recommendations for AdvBuildingGym

- **Keep `hst_{action}` in-env** while `ActionSmoothnessReward` reads it from
  `self.state`. Either leave as-is or refactor the reward to read the
  previous action from a dedicated env attribute before migrating.
- **Migrate observation stacking to a connector** for signals that only the
  policy needs (weather, price, indoor/outdoor temperature). Biggest win is
  on SAC replay memory.
- **Start from the built-ins** (`FrameStackingEnvToModule` +
  `FrameStackingLearner`) and only drop to a custom `StridedHistoryConnector`
  once uniform stacking is working end-to-end.
- **Add LSTM only after** stacking alone is evaluated — they are alternative
  or complementary, not free.
- **Share one connector pipeline definition** between
  `common_model_config.py` and `eval_runner.py` to prevent train/eval skew.
- **Observation space size**: vector obs, so 9× blow-up from strided
  stacking is cheap; revisit if obs dimension grows.

---

## 5. Open questions / next steps

- Prototype the hybrid: connector stacking for obs + in-env action history,
  measure replay-buffer footprint on SAC.
- Decide whether `TemporalFeatureBuffer` in `utils/temporal_features.py`
  should be deleted or wired in.
- Design a shared connector-pipeline factory used by both training and eval
  entry points.
