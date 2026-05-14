# RLlib Actor Architecture: Algorithm, EnvRunners, and Learner

How Ray RLlib distributes work across processes in the new API stack.

## The Three Actor Types

### 1. Algorithm Driver (Coordinator)

The orchestration process. It runs the training loop:

- Tells EnvRunners to collect episodes
- Manages the replay buffer
- Decides when to trigger learning
- Runs evaluation episodes (when `evaluation_parallel_to_training=False`, evaluation runs in this same process)

It does **not** do gradient updates and does **not** run environments for training. It delegates work but doesn't do the heavy lifting itself.

### 2. EnvRunners (Environment Samplers)

The number of these processes is controlled by `num_env_runners`. Their only job is to run the environment and collect trajectories:

- Hold an env instance
- Query the policy (inference)
- Step the env
- Ship episodes back to the algorithm driver

They use **CPU only**.

### 3. Learner (GPU Worker)

The gradient computation process:

- Takes batches of experience from the replay buffer
- Performs SGD updates on the neural network weights
- Uses **GPU**
- Never touches the environment

## Why the Separation?

```
EnvRunners (CPU)          Algorithm Driver (CPU)       Learner (GPU)
──────────────────         ─────────────────────        ──────────────
run env -> episodes  -->   buffer + orchestration  -->  gradient updates
                     <--   updated weights          <--
```

Sampling and learning have different resource requirements:

- **Sampling** is CPU-bound (env physics, Python logic)
- **Learning** is GPU-bound (matrix ops on the network)

Decoupling them means the GPU is never idle waiting for env steps, and env runners don't block on gradient computation.

## Process Summary

| Process        | Count | Role                         |
|----------------|-------|------------------------------|
| Algorithm driver | 1   | Orchestration + evaluation   |
| EnvRunners     | 2     | Sampling (training episodes) |
| Learner        | 1     | Gradient updates (GPU)       |

> **Naming note:** The algorithm label on the driver process (e.g. `SAC`) refers to the algorithm *object*, not to "the thing doing SAC gradient updates" -- that's the Learner. The naming can be confusing because in older RLlib, one process did both.
