# Handling Evolving State Spaces in RL

When the observation layout changes during development, earlier checkpoints
become unloadable. The goal is **explicit mapping** — the data describes itself
and old checkpoints either keep working or fail loudly.

Targets this project's stack: Gymnasium 1.0+ `gym.spaces.Dict` observations,
Ray RLlib new API stack, `FlattenObservations` env-to-module connector.

## Current state

- Top-level obs is a `gym.spaces.Dict` — `adv_building_gym/envs/building_adv.py:250`.
- Flattened at the RLModule boundary by `FlattenObservations()` —
  `adv_building_gym/ray_training/common_model_config.py:11, 259`.
- `"version": 1` tag exists in trajectory dumps
  (`callbacks/trajectory_logging_callback.py`,
  `utils/trajectory_collector.py`), not in checkpoints, not in `info`.
- Ray checkpoints store RLModule weights only, no obs-space metadata.
  `rl_module_inference.py` assumes static spaces.

---

## 1. Adapter pattern

Map each "raw state version" to a **canonical state** via a version-specific
encoder; keep the policy head stable. On the new API stack, implement the
adapter as a **ConnectorV2** —
[`SingleAgentObservationPreprocessor`](https://docs.ray.io/en/master/_modules/ray/rllib/connectors/env_to_module/observation_preprocessor.html)
with `preprocess(obs, episode)` and, when the space changes,
`recompute_output_observation_space()`. The legacy `Preprocessor` class does
not work here.

## 2. Dict spaces (with caveats)

Use `gym.spaces.Dict` at the env layer; avoid top-level `Box`. Caveats on this
stack:

- New API stack lacks native Dict-space encoder routing
  ([ray#59537](https://github.com/ray-project/ray/issues/59537),
  [ray#46631](https://github.com/ray-project/ray/issues/46631));
  `FlattenObservations` concatenates leaves into a single `Box`. Adding a Dict
  key therefore changes the flat vector length and breaks old checkpoints.
  Dict-ness alone does not buy compatibility — it only makes the semantics
  addressable.
- Do **not** use `spaces.Text(...)` for version metadata — it does not flow
  through `FlattenObservations`. Put schema metadata in the `info` dict.
- `FlattenObservations` follows Dict key order. Reordering
  `create_infras()` / `create_statesources()` silently changes the flat layout
  with no `setup_spaces()` diff.

### Additive vs. breaking changes

| Change                                         | Compatible? | Strategy |
| :--------------------------------------------- | :---------- | :------- |
| Add a new Dict key, existing keys unchanged    | Additive    | Adapter drops new key (or zero-pads for old→new) |
| Rename a key, same shape                       | Additive    | Adapter renames |
| Change shape / dtype / bounds of existing key  | Breaking    | Retrain; bump `SCHEMA_VERSION` |
| Same shape, different meaning                  | Breaking    | Retrain; bump `SCHEMA_VERSION` |

## 3. Key-value / set embedding

For rapidly changing sensor counts/types, tag each value with a Type ID and
sum `Embedding(ID_i) * value_i` into a fixed-size latent. Requires a custom
`RLModule`; incompatible with `FlattenObservations`. Overkill for the current
fixed device set (HP, battery, EV, weather, price).

## 4. Hypernetworks / context (brief)

For static episode context (season, building props), prefer concatenation or
a FiLM layer. Hypernetworks are correct in theory but rarely worth the
machinery here.

---

## Concrete solutions

### 5.1 Schema fingerprint + version gate

Single source of truth for the schema version and its fingerprint.

```python
# adv_building_gym/config/utils/schema_fingerprint.py
import hashlib, json, numpy as np
from gymnasium import spaces

SCHEMA_VERSION = 1  # bump on every breaking setup_spaces() change

def fingerprint(space: spaces.Dict) -> dict:
    leaves = []
    for k in sorted(space.spaces.keys()):            # canonical, order-independent
        s = space[k]
        leaves.append({
            "key": k,
            "shape": list(s.shape),
            "dtype": str(s.dtype),
            "low":  np.asarray(s.low).reshape(-1)[:8].tolist(),
            "high": np.asarray(s.high).reshape(-1)[:8].tolist(),
        })
    blob = json.dumps({"schema_version": SCHEMA_VERSION, "leaves": leaves},
                      sort_keys=True).encode()
    return {"schema_version": SCHEMA_VERSION,
            "hash": hashlib.sha256(blob).hexdigest()[:16],
            "leaves": leaves}
```

Route `SCHEMA_VERSION` through `info` in `building_adv.py`'s `reset()`/`step()`
and through the trajectory files (replace the hard-coded `"version": 1`).

**CI gate:** a unit test that asserts the current hash equals a pinned fixture
(`tests/fixtures/obs_schema.<version>.json`). Any schema change without a
matching `SCHEMA_VERSION` bump and fixture update breaks CI. This turns
"developer remembers to bump" into a mechanical check.

### 5.2 Canonical flatten order

`FlattenObservations` is sensitive to Dict insertion order. Wrap it with a
`CanonicalKeyOrder` connector that re-keys the Dict in sorted order before
flattening, so the fingerprint hash is authoritative over the layout.

### 5.3 Checkpoint sidecar + load-time check

On checkpoint save (`callbacks/checkpoint_callbacks.py`) write
`obs_schema.json` (the fingerprint) and the `EnvConfig` YAML next to the Ray
checkpoint. On load (`run_eval_ray.py`, `rl_module_inference.py`):

```python
ckpt_fp = json.load(open(ckpt_dir / "obs_schema.json"))
env_fp  = fingerprint(env.observation_space)
ckpt_keys = {l["key"] for l in ckpt_fp["leaves"]}
env_keys  = {l["key"] for l in env_fp["leaves"]}

if ckpt_fp["hash"] == env_fp["hash"]:
    adapter = None                                    # exact match
elif ckpt_keys.issubset(env_keys):                    # env is a superset
    adapter = SchemaAdapter(target=rebuild_space(ckpt_fp),
                            drop_keys=tuple(env_keys - ckpt_keys))
else:
    raise SchemaMismatch(diff(ckpt_fp, env_fp))       # fail loudly
```

### 5.4 `SchemaAdapter` ConnectorV2

Inserted ahead of `FlattenObservations` in `common_model_config.py:259` when
adapter is not `None`.

```python
# adv_building_gym/ray_training/connectors/schema_adapter.py
from ray.rllib.connectors.env_to_module import SingleAgentObservationPreprocessor
from gymnasium import spaces
import numpy as np

class SchemaAdapter(SingleAgentObservationPreprocessor):
    """Project current env Dict obs onto a target (older) schema."""
    def __init__(self, target: spaces.Dict,
                 drop_keys: tuple[str, ...] = (),
                 zero_pad_keys: tuple[str, ...] = ()):
        self._target, self._drop, self._pad = target, set(drop_keys), set(zero_pad_keys)
        super().__init__()

    def preprocess(self, observation, episode):
        out = {k: v for k, v in observation.items() if k not in self._drop}
        for k in self._pad:
            out.setdefault(k, np.zeros(self._target[k].shape,
                                       dtype=self._target[k].dtype))
        return out

    def recompute_output_observation_space(self, input_observation_space, input_action_space):
        return self._target
```

### 5.5 Structural vs. learned adapters — and how to retrain

The adapter absorbs schema variance so the downstream trunk + head see a
fixed-dim **canonical input**:

```
obs_vX ──> Adapter_vX ──> canonical z ──> shared trunk ──> policy/value head
                                           (frozen)         (frozen / fine-tuned)
```

Two variants with different trade-offs:

| Kind           | Where it lives                             | Trainable? |
| :------------- | :----------------------------------------- | :--------- |
| **Structural** | env-to-module ConnectorV2 (§5.4)           | No parameters |
| **Learned**    | Inside a custom `RLModule`, before the trunk | Yes |

ConnectorV2 pieces run in env-runner processes and are not in the learner's
compute graph — so anything that needs gradient flow **must** live inside the
`RLModule`, not in the connector.

**Retraining a learned adapter**, cheapest first:

1. **Partial fine-tune.** Freeze trunk + head; continue RL training with only
   the adapter unfrozen. Handles most additive changes.
2. **Feature-alignment distillation.** Supervised: minimise
   `‖f_vX(obs_vX) − f_{vX-1}(obs_{vX-1})‖²` on overlapping keys, where `f` is
   the adapter's canonical output. No RL needed; cheapest when overlap is
   large.
3. **Action distillation.** Keep the vX-1 policy as a teacher on shared
   states; train the new adapter (+ head) to match teacher action
   distributions.
4. **Joint fine-tune.** Unfreeze everything; short RL run from vX-1 weights.
   Fallback when semantics of an existing key changed.

**No retraining is needed** when the change is purely additive and the policy
can ignore the new information — the structural adapter in §5.4 drops the new
keys and zero-training loading works. Retraining enters the moment you want
the policy to actually *use* the new signal.

---

## Recommended sequencing

1. `schema_fingerprint.py` + CI fixture gate (§5.1).
2. Route `SCHEMA_VERSION` through `info` and trajectory dumps (§5.1).
3. Checkpoint sidecar + hard-fail load-time check, no adapter yet (§5.3).
4. `CanonicalKeyOrder` connector (§5.2).
5. `SchemaAdapter` (structural, zero-training) for additive cases (§5.4).
6. Custom `RLModule` with a learned adapter prefix when new keys must carry
   signal (§5.5); retrain with partial fine-tune or feature-alignment
   distillation.
7. Revisit native Dict routing when ray#59537 lands.

Steps 1–3 alone stop silent breakage; 4–5 enable backward-compatible loading
across additive changes.

---

## References

- ConnectorV2 observation preprocessor: <https://docs.ray.io/en/master/_modules/ray/rllib/connectors/env_to_module/observation_preprocessor.html>
- New API stack migration guide: <https://docs.ray.io/en/latest/rllib/new-api-stack-migration-guide.html>
- ConnectorV2 API reference: <https://docs.ray.io/en/latest/rllib/package_ref/connector-v2.html>
- Native Dict obs support: <https://github.com/ray-project/ray/issues/59537>
- Dict-space issues on the new API stack: <https://github.com/ray-project/ray/issues/46631>
