# RLlib Iteration, Timestep, and Episode Semantics

This document clarifies the meaning of **timestep**, **episode**, and **training iteration** in the context of Ray RLlib and how these concepts map to this project's configuration. It is motivated by the log line:

```
Ray Tune-level: every 1 iterations (288.0 timesteps/episode, 4000 timesteps/iteration), num_to_keep=5
```

---

## 1. Timestep

A **timestep** is one call to `env.step(action)` — the atomic unit of environment interaction.

In this project (`adv_building_gym/config/env_config.py:37`):
```python
control_step: int = 300  # seconds (5 minutes)
```

Each timestep corresponds to a **5-minute control interval** in the simulated building.

TODO VP: remove timestep based stopping, use rather episode based criterias

RLlib counts timesteps via `num_env_steps_sampled_lifetime`, which is also used as the primary stopping criterion (`run_train_ray.py:279`):
```python
stop_criteria = {
    "num_env_steps_sampled_lifetime": args.timesteps,
    ...
}
```

**Reference:** [RLlib Environment Steps](https://docs.ray.io/en/latest/rllib/rllib-training.html)

---

## 2. Episode

An **episode** is a complete trajectory from `env.reset()` to termination — a contiguous sequence of timesteps within a single simulated day.

In this project (`adv_building_gym/config/env_config.py:36`):
```python
EPISODE_LENGTH: int = 288
```

```
1 episode = 288 timesteps × 300 s/timestep = 86 400 s = 24 hours
```

The env runners collect full episodes before returning data to the learner. This is enforced via `rollout_fragment_length` (`common_model_config.py:142`):
```python
rollout_fragment_length=env_config.EPISODE_LENGTH,
```

Without this, off-policy algorithms (SAC) default to `rollout_fragment_length=1`, causing episodes to appear as length 1 in callbacks.

Episodes are from within the same day.

TODO VP: When and how to switch between multiple days?

**Reference:** [RLlib Environments](https://docs.ray.io/en/latest/rllib/package_ref/env.html)

NOTE VP: It is not mandatory to use up the whole day as a single episode. 
It is maybe more effective, if we split up the day into several shorter trajectories.

---

## 3. Training Iteration

A **training iteration** is one call to `algorithm.train()`, orchestrated by Ray Tune's internal loop.

Each iteration:
1. **Env runners** collect `train_batch_size_per_learner` timesteps from the environment.

TODO VP: More steps should be collected as an episode length... -- because the learner needs more

2. **Learner** performs gradient updates on the collected batch.
3. Metrics are aggregated and reported.

In this project (`select_model.py:64`, `training_config.json`):
```python
train_batch_size_per_learner=4000  # timesteps collected per iteration per learner
```

Episodes per iteration (approximate):
```
4000 timesteps/iteration ÷ 288 timesteps/episode ≈ 13.9 episodes/iteration
```

Because episodes are not split across iterations (due to `rollout_fragment_length=288`), the actual number is either 13 or 14 episodes per iteration. RLlib pads or truncates to align with complete episodes.

`training_iteration` is the second stopping criterion and acts as a safety cap (`run_train_ray.py:281`):
```python
"training_iteration": 250,
```

**Reference:** [RLlib Algorithm API](https://docs.ray.io/en/latest/rllib/rllib-training.html#using-the-python-api)

---

## 4. How the Three Concepts Relate

```
Timestep  →  Episode  →  Iteration
   ×288           ×~13.9
```

| Concept   | Unit                    | Value in this project |
|-----------|-------------------------|-----------------------|
| Timestep  | 1 `env.step()` call     | 5 min of simulation   |
| Episode   | 288 timesteps           | 24 h of simulation    |
| Iteration | ~4000 timesteps         | ~13.9 episodes        |

TODO VP: Fix iterations, calculate it based on timesteps...

For 1 million total timesteps:
```
1 000 000 ts ÷ 4000 ts/iter  = 250 iterations
1 000 000 ts ÷ 288 ts/ep     ≈ 3 472 episodes
```

---

## 5. Interpreting the Log Line

The log line is emitted by `run_train_ray.py:270-274`:

```python
logger.info(
    "  Ray Tune-level: every %d iterations (%.1f timesteps/episode, %d timesteps/iteration), num_to_keep=5",
    checkpoint_freq_iterations,
    timesteps_per_episode,
    timesteps_per_iteration
)
```

### Fields

| Field                           | Value | Source                                          |
|---------------------------------|-------|-------------------------------------------------|
| `checkpoint_freq_iterations`    | 1     | Computed from episode frequency (see below)     |
| `timesteps/episode`             | 288.0 | `config.EPISODE_LENGTH`                         |
| `timesteps/iteration`           | 4000  | `train_batch_size_per_learner`                  |
| `num_to_keep`                   | 5     | `CheckpointConfig(num_to_keep=5)`               |

### Checkpoint Frequency Calculation

`run_train_ray.py:261-262`:
```python
timesteps_per_episode  = active_config.EPISODE_LENGTH        # 288
timesteps_per_iteration = param_space.get("train_batch_size_per_learner", 4000)  # 4000

checkpoint_freq_iterations = max(1, int(
    (args.checkpoint_frequency_episodes * timesteps_per_episode) / timesteps_per_iteration
))
```

With default `--checkpoint-frequency-episodes 20`:
```
checkpoint_freq_iterations = max(1, int((20 × 288) / 4000))
                           = max(1, int(5760 / 4000))
                           = max(1, int(1.44))
                           = max(1, 1)
                           = 1
```

The result `1` means a Ray Tune checkpoint is saved **every iteration**. This happens because 20 episodes (5760 ts) is less than 2 iterations (8000 ts) — the integer division floors to 1, and the `max(1, ...)` guard prevents zero. To checkpoint less frequently at the Ray Tune level, use a larger `--checkpoint-frequency-episodes` value (e.g., `≥ 28` episodes yields `≥ 2` iterations per checkpoint).

### Two-Level Checkpointing

There are two independent checkpointing mechanisms:

| Level              | Mechanism                        | Controlled by                         |
|--------------------|----------------------------------|---------------------------------------|
| **Ray Tune**       | `CheckpointConfig(checkpoint_frequency=N)` | `run_train_ray.py:317`   |
| **Callback**       | `BestModelCheckpointCallback`    | `--checkpoint-frequency-episodes` CLI |

The callback-level checkpoint tracks the **best** model by metric. The Ray Tune-level checkpoint saves full algorithm state (weights + optimizer state) for recovery. `num_to_keep=5` at the Ray Tune level retains the last 5 such snapshots; older ones are deleted automatically.

**Reference:** [Ray Tune CheckpointConfig](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CheckpointConfig.html)

---

## 6. References

- [RLlib Key Concepts](https://docs.ray.io/en/latest/rllib/key-concepts.html)
- [RLlib Training Guide](https://docs.ray.io/en/latest/rllib/rllib-training.html)
- [Ray Tune CheckpointConfig API](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CheckpointConfig.html)
- [RLlib Environment Runners](https://docs.ray.io/en/latest/rllib/package_ref/env_runners.html)
- Project files:
  - `adv_building_gym/config/env_config.py` — `EPISODE_LENGTH`, `control_step`
  - `adv_building_gym/ray_training/select_model.py` — `train_batch_size_per_learner`
  - `adv_building_gym/ray_training/common_model_config.py` — `rollout_fragment_length`, `evaluation_interval`
  - `run_train_ray.py` — checkpoint frequency calculation, stop criteria
