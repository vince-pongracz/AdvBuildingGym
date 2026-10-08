# SAC & PPO Training Parameters — Analysis and Exploration

> RLlib 2.52.1, new API stack. Parameters discovered from `PPOConfig.training()`,
> `SACConfig.training()`, `AlgorithmConfig.training()`, and `DefaultModelConfig`.

---

## Table of Contents

1. [Shared Base Parameters (AlgorithmConfig)](#1-shared-base-parameters-algorithmconfig)
2. [PPO-Specific Parameters (PPOConfig)](#2-ppo-specific-parameters-ppoconfig)
3. [SAC-Specific Parameters (SACConfig)](#3-sac-specific-parameters-sacconfig)
4. [Neural Network Configuration (DefaultModelConfig)](#4-neural-network-configuration-defaultmodelconfig)
5. [How Our Project Uses These Parameters](#5-how-our-project-uses-these-parameters)
6. [Parameter Interactions and Tuning Notes](#6-parameter-interactions-and-tuning-notes)

---

## 1. Shared Base Parameters (AlgorithmConfig)

These parameters are available on **all** algorithm configs (PPO, SAC, etc.) via
`config.training(...)`.

> **Column legend**
> - **Factory default** — RLlib's out-of-the-box value (from `PPOConfig()` / `SACConfig()`)
> - **User default** — Value in `training_param_config.yaml` (our YAML config)
> - **Actual** — Value that reaches RLlib after `select_model.py` transforms the YAML (may differ from user default due to unit conversion, formulas, etc.)

| Parameter | Type | Factory Default | User Default | Actual | Description |
|-----------|------|-----------------|--------------|--------|-------------|
| `gamma` | float | 0.99 | *not set* | 0.99 | **Discount factor.** Controls how much the agent values future rewards vs immediate ones. `gamma=0` → fully myopic (only cares about the next reward). `gamma=1` → treats all future rewards equally (no discounting). For our 288-step episodes (one day at 5 min resolution), gamma=0.99 means the agent still gives ~5.7% weight to a reward 288 steps ahead (`0.99^288 ≈ 0.057`). Lower values (e.g. 0.95) make it focus on shorter-term comfort/cost, higher values encourage planning ahead (e.g. pre-charging battery at night for daytime solar gaps). |
| `lr` | float \| schedule | 1e-3 (base) / 5e-5 (PPO) | 3e-4 | 3e-4 (PPO) | **Global learning rate.** Used by PPO's single optimizer. `AlgorithmConfig` base default is 1e-3; `PPOConfig` overrides to 5e-5. SAC ignores `lr` entirely — it uses separate `actor_lr`, `critic_lr`, `alpha_lr`. Can also be a schedule `[[timestep, lr], ...]` for annealing. We set 6× the PPO factory default. |
| `grad_clip` | float \| None | None | *not set* | None (PPO) / 1.0 (SAC) | **Gradient clipping threshold.** Prevents exploding gradients during training. The clipping method depends on `grad_clip_by`. Without clipping, a single bad batch (e.g. extreme reward spike) can destabilize the entire network — which is why we set `grad_clip=1.0` for SAC after observing NaN crashes. |
| `grad_clip_by` | str | "global_norm" | *not set* | "global_norm" | **Gradient clipping method.** Three modes: (1) `"value"` — clips each gradient element independently to [-grad_clip, +grad_clip]. (2) `"norm"` — clips L2-norm of each parameter tensor independently. (3) `"global_norm"` — computes a single L2-norm across *all* parameter gradients and scales them down uniformly if it exceeds the threshold. Global norm is the standard choice because it preserves the relative direction of gradients across layers. |
| `train_batch_size_per_learner` | int | 32 (base) / 4000 (PPO) / 256 (SAC) | 10 episodes (PPO) / 128 (SAC) | 2880 (PPO) / 128 (SAC) | **Transitions per learner per training iteration.** This is the fundamental batch size control on the new API stack. `AlgorithmConfig` base default is 32; PPO and SAC override it. For PPO it means "timesteps collected before each policy update" — we express it as episodes in the YAML (`10 × 288 = 2880`). For SAC it means "transitions sampled from replay buffer per gradient step". The total effective batch across all learners is `num_learners × train_batch_size_per_learner`. |

Note: `num_epochs`, `minibatch_size`, and `shuffle_batch_per_epoch` are technically
defined on `AlgorithmConfig` (the base class), so both PPO and SAC *inherit* them.
However, `PPOConfig.__init__` overrides the base defaults to `num_epochs=30`,
`minibatch_size=128`, and `shuffle_batch_per_epoch=True`, while `SACConfig` keeps the
base defaults (`num_epochs=1`, `minibatch_size=None`, `shuffle_batch_per_epoch=False`).
SAC's training loop ignores these — it does one pass per sampled batch, controlled by
`training_intensity` instead.  Because these parameters are only meaningful for PPO,
they are documented in [Section 2 (PPO)](#2-ppo-specific-parameters-ppoconfig).

### Parameters We Don't Currently Set (But Could)

- **`gamma`** — We use the factory default 0.99. For a building environment with daily episodes and 5 min steps, this is reasonable. Lowering to 0.95–0.98 would make the agent more short-sighted (less battery planning), raising to 0.995+ would encourage more forward-looking behavior.
- **`grad_clip_by`** — We don't set this (factory default `"global_norm"`). This is fine for most cases.

---

## 2. PPO-Specific Parameters (PPOConfig)

PPO (Proximal Policy Optimisation) is an **on-policy** algorithm. It collects a batch
of experience using the current policy, updates the policy using that batch (multiple
epochs), then *discards* the data. This makes it sample-inefficient but stable.

Link: https://arxiv.org/abs/1707.06347

> Column legend: see [Section 1](#1-shared-base-parameters-algorithmconfig).

| Parameter | Type | Factory Default | User Default | Actual | Description |
|-----------|------|-----------------|--------------|--------|-------------|
| `num_epochs` | int | 30 (base: 1) | 20 | 20 | **SGD passes over the training batch.** After collecting the batch, PPO runs this many full passes over it, each split into mini-batches. More epochs = more gradient updates per batch of experience, but risks overfitting to stale data and policy divergence (mitigated by KL penalty and clipping). `AlgorithmConfig` base default is 1; `PPOConfig` overrides to 30. SAC ignores this (keeps base default 1). |
| `minibatch_size` | int | 128 (base: None) | 64 | 64 | **SGD mini-batch size within each epoch.** Splits the training batch into chunks of this size. Smaller mini-batches add noise (can help generalization) but increase compute time per epoch. `AlgorithmConfig` base default is None (whole batch as one mini-batch); `PPOConfig` overrides to 128. SAC ignores this. |
| `shuffle_batch_per_epoch` | bool | True (PPO; base: False) | *not set* | True | **Whether to shuffle the batch before each epoch.** `PPOConfig` overrides the base default of False to True. Shuffling breaks temporal correlations. If the batch has a time axis, shuffling only happens along the batch dimension (preserving episode sequences). |
| `use_critic` | bool | True | *not set* | True | **Enable value function baseline.** The critic V(s) estimates the expected return from state s. Subtracting it from actual returns gives the *advantage* A(s,a), which has lower variance than raw returns. Required for GAE. Disabling it reverts to REINFORCE-style high-variance updates. Always keep True. |
| `use_gae` | bool | True | *not set* | True | **Generalised Advantage Estimation.** Computes advantage using a weighted blend of n-step TD errors, controlled by `lambda_`. Without GAE, advantage is just `R - V(s)` (high variance). With GAE, it's an exponentially-weighted average of multi-step advantages, providing a smooth bias-variance tradeoff. Link: https://arxiv.org/abs/1506.02438 |
| `lambda_` | float | 1.0 | *not set* | 1.0 | **GAE lambda.** Controls bias-variance tradeoff in advantage estimation. `lambda_=0` → uses only 1-step TD error (low variance, high bias — heavily relies on critic accuracy). `lambda_=1` → uses full Monte Carlo returns up to episode boundary (high variance, low bias). Typical values: 0.95–0.99. For our building env, the critic may be inaccurate early in training, so a higher lambda (0.95+) is safer until the value function converges. |
| `use_kl_loss` | bool | True | *not set* | True | **KL divergence penalty in the loss.** Adds a soft penalty term `kl_coeff × KL(old_policy ‖ new_policy)` to the loss function. This is PPO's *adaptive KL penalty* variant (PPO1), used alongside the clipping mechanism (PPO2). When both are active, they provide dual protection against destructive policy updates. |
| `kl_coeff` | float | 0.2 | *not set* | 0.2 | **Initial KL penalty coefficient.** Automatically adapted during training: if mean KL exceeds `2 × kl_target`, `kl_coeff` is multiplied by 1.5; if mean KL drops below `0.5 × kl_target`, `kl_coeff` is halved. This dead-zone mechanism avoids oscillation — the coefficient only changes when KL is far from target. Higher initial values = more conservative updates. |
| `kl_target` | float | 0.01 | *not set* | 0.01 | **Target KL divergence.** The desired mean KL between old and new policy per update. 0.01 is a common choice. The `kl_coeff` adaptation uses a dead-zone: only when mean KL exceeds `2 × kl_target` does the coefficient increase (by 1.5x), and only when mean KL drops below `0.5 × kl_target` does it decrease (by 0.5x). This prevents catastrophic policy updates while avoiding excessive conservatism. |
| `vf_loss_coeff` | float | 1.0 | *not set* | 1.0 | **Value function loss coefficient.** Scales the value function loss relative to the policy loss. If value and policy share layers (`vf_share_layers=True`), this must be tuned carefully — too high and policy gradients are drowned out, too low and the value function trains slowly. Since we use separate encoder layers (the RLlib default for new API stack), this is less critical. |
| `entropy_coeff` | float | 0.0 | *not set* | 0.0 | **Entropy bonus coefficient.** Adds `entropy_coeff × H(pi)` to the objective, encouraging exploration by penalizing deterministic policies. PPO already has clipping and KL-penalty to prevent collapse, so this is often 0. Non-zero values (0.001–0.01) can help in environments with many local optima. Can also be a schedule. |
| `entropy_coeff_schedule` | list | None | *not set* | None | **Entropy coefficient schedule.** Format: `[[timestep, value], ...]`. Allows annealing entropy bonus from a high initial value (explore) to 0 (exploit). Useful for environments where early exploration is critical but later precision matters. |
| `clip_param` | float | 0.3 | *not set* | 0.3 | **PPO clipping parameter (epsilon).** The core mechanism of PPO2. Clips the probability ratio `r(theta) = pi_new(a|s) / pi_old(a|s)` to `[1-epsilon, 1+epsilon]`. This prevents the policy from changing too drastically in a single update. Smaller values (0.1–0.2) = more conservative updates. The original paper uses 0.2; RLlib defaults to 0.3 which is slightly more aggressive. For our building environment, 0.2 might be safer as the reward landscape has sharp cliffs (temperature penalties). |
| `vf_clip_param` | float | 10.0 | *not set* | 10.0 | **Value function loss clipping.** Clamps the per-sample squared VF error `(V_pred - V_target)²` to `[0, vf_clip_param]`. This prevents outlier samples (where the value prediction is far off) from dominating the VF gradient. With `vf_clip_param=10.0`, any sample whose VF error exceeds `sqrt(10) ≈ 3.16` gets capped. Sensitive to reward scale — with our [-1, 1] per-step rewards and 288-step episodes, episode returns can reach ~288, so value targets may be large. 10.0 could be too restrictive if the critic needs to learn large values early on. Consider raising to 50–100. |
| `lr_schedule` | list | None | *not set* | None | **Learning rate schedule.** Format: `[[timestep, lr], ...]`. Linear interpolation between points. Alternative to the base `lr` schedule. Annealing LR is common practice: start high for fast initial progress, reduce for fine-tuning. |

### PPO Data Flow (How Parameters Interact)

```
Iteration:
1. EnvRunners collect episodes using current policy
   → Total timesteps = ppo_episodes_per_iteration × episode_length
   → This becomes train_batch_size_per_learner

2. Learner connector pipeline processes the batch:
   → V(s) forward pass on all observations
   → Compute GAE advantages + value targets for entire batch (using lambda_, gamma)
   (advantages are computed ONCE, before the epoch loop)

3. SGD epoch loop:
   → For epoch in range(num_epochs):          # 20 in our config
       → Shuffle batch (if shuffle_batch_per_epoch)
       → Split into chunks of minibatch_size   # 64 in our config
       → For each minibatch:
           → Compute clipped surrogate loss (clip_param) using pre-computed advantages
           → Add KL penalty (if use_kl_loss, weighted by kl_coeff)
           → Add entropy bonus (entropy_coeff × H(pi))
           → Add value loss (vf_loss_coeff × MSE)
           → Backprop with grad_clip

4. Adapt kl_coeff: if mean_kl > 2×kl_target → ×1.5; if mean_kl < 0.5×kl_target → ×0.5
5. Discard the batch (on-policy — cannot reuse)
```

**With our config:** 10 episodes × 288 steps = 2880 timesteps per iteration.
Each epoch splits this into 2880/64 = 45 mini-batches.
20 epochs × 45 mini-batches = 900 gradient updates per iteration.

---

## 3. SAC-Specific Parameters (SACConfig)

SAC (Soft Actor-Critic) is an **off-policy** algorithm with **entropy regularisation**.
It maintains a replay buffer of past experience and learns from random samples, making
it far more sample-efficient than PPO. The entropy term encourages exploration and
prevents premature convergence.

Link: https://arxiv.org/abs/1802.09477

> Column legend: see [Section 1](#1-shared-base-parameters-algorithmconfig).

| Parameter | Type | Factory Default | User Default | Actual | Description |
|-----------|------|-----------------|--------------|--------|-------------|
| `actor_lr` | float \| schedule | 3e-5 | 3e-4 | 3e-4 | **Policy network learning rate.** Controls how fast the policy (actor) updates. RLlib's factory default is 3e-5 (one decade lower than critic), following the *two-timescale* principle: the critic should converge faster than the actor so the actor receives stable gradient signals. We currently use the same LR for both (10x the factory default) — this works but may cause instability if the critic hasn't converged. Consider actor_lr = 1e-4 with critic_lr = 3e-4. |
| `critic_lr` | float \| schedule | 3e-4 | 3e-4 | 3e-4 | **Q-network learning rate.** The critic estimates Q(s,a) — the expected return for taking action a in state s. It should learn faster than the actor so the policy receives accurate gradient information. 3e-4 is standard. Matches factory default. |
| `alpha_lr` | float \| schedule | 3e-4 | 3e-4 | 3e-4 | **Entropy coefficient (alpha) learning rate.** Alpha weights the entropy bonus in SAC's objective: `J = E[Q(s,a) - alpha × log(pi(a|s))]`. It is *auto-tuned*: if the policy's entropy drops below `target_entropy`, alpha increases (encouraging more exploration). If entropy is too high, alpha decreases. The LR controls how fast this adaptation happens. Matches factory default. |
| `initial_alpha` | float | 1.0 | *not set* | 1.0 | **Starting value for entropy coefficient.** Higher values = more exploration at the start. Since alpha is auto-tuned, the initial value matters most for early training. 1.0 is standard — the entropy bonus starts with equal weight to the Q-value. If training is unstable early on, try lowering to 0.1–0.5. |
| `target_entropy` | str \| float | "auto" | *not set* | "auto" | **Target entropy for alpha auto-tuning.** When "auto", computed as `-np.prod(action_space.shape)`. For our flat action space of shape `(3,)` (HP energy + HP mode + battery), target = -3.0. Alpha is increased when policy entropy drops below this target (encouraging more exploration) and decreased when above (allowing more exploitation). Lower target = more deterministic final policy. Can be manually set for finer control. |
| `twin_q` | bool | True | *not set* | True | **Twin Q-networks (clipped double-Q).** Uses two independent Q-networks and takes the minimum of their predictions for the target value. This addresses the *overestimation bias* in Q-learning (where the max operator systematically overestimates Q-values). Always keep True — disabling it causes divergence in most continuous-action tasks. Link: https://arxiv.org/abs/1802.09477 |
| `tau` | float | 0.005 | *not set* | 0.005 | **Soft (Polyak) update coefficient for target networks.** Target networks are updated as `theta_target = tau × theta + (1-tau) × theta_target`. Small tau = slow, stable target updates. Large tau = fast but potentially unstable. 0.005 is standard. In our building env, 0.001–0.01 is a reasonable range. |
| `target_network_update_freq` | int | 0 | *not set* | 1 | **Environment steps between target network updates.** Measured in `NUM_ENV_STEPS_SAMPLED_LIFETIME` (not gradient steps). The target networks are soft-updated when `current_timestep - last_update_timestep >= target_network_update_freq`. Factory default 0 means update after every training call. We hardcode 1 in `select_model.py`, meaning target networks are soft-updated after every environment step — essentially equivalent to the factory default behaviour. Combined with tau=0.005, each update blends 0.5% of the online weights into the target. |
| `n_step` | int \| (int, int) | 1 | *not set* | 6 | **N-step returns.** Transforms (s, a, r, s') tuples into (s, a, sum_of_discounted_r, s_{t+n}) tuples. n=1 is standard TD learning. Higher n propagates reward signals faster through the critic, but introduces more bias from the value estimate at step n. Can also be a tuple (min, max) for random n-step per sample. We hardcode n=6 in `select_model.py` (30 min lookahead at 5 min resolution), which helps the agent learn cause-effect relationships in thermal dynamics — e.g. a heat pump action now affects temperature 30 min later. |
| `training_intensity` | float \| None | None | 32 | 32 | **Ratio of replayed steps to sampled steps.** Controls how many gradient updates SAC performs relative to new experience collected. UTD (update-to-data ratio) ≈ training_intensity / train_batch_size. With our values: 32/128 = UTD 0.25, meaning one gradient step per four new environment steps. Standard SAC uses UTD ≈ 1.0; our lower value trades sample efficiency for training speed and stability. Higher values (128, 256) train more aggressively on existing data (REDQ-style). None defaults to `train_batch_size / (rollout_fragment_length × num_workers)`. |
| `replay_buffer_config` | dict | PrioritizedEpisodeReplayBuffer, 1M | 1500 days | EpisodeReplayBuffer, 432k | **Replay buffer configuration.** Defines buffer type and capacity. RLlib 2.52 factory default: `PrioritizedEpisodeReplayBuffer` with capacity=1,000,000, alpha=0.6, beta=0.4. We override to `EpisodeReplayBuffer` with `capacity = episode_length × sac_episodes_to_keep_in_replay_buffer = 288 × 1500 = 432,000 timesteps`. This holds ~1500 episodes of experience. **Note:** the code currently passes `alpha=0.6` and `beta=0.4` alongside `type: "EpisodeReplayBuffer"` — these prioritisation params are silently ignored by the non-prioritised buffer. Remove them for clarity, or switch to `PrioritizedEpisodeReplayBuffer` to actually use them. See the dedicated section below for the prioritised replay parameters. |
| `num_steps_sampled_before_learning_starts` | int | 1500 | *not set* | 1500 (factory) | **Warm-up period.** Number of timesteps to collect before the first gradient update. Fills the replay buffer with diverse experience first. Previously hardcoded in `select_model.py` as `10 × episode_length = 2880` (10 episodes), but currently **commented out** — so the factory default of 1500 applies. This means SAC starts learning after ~5 episodes (1500/288 ≈ 5.2). Too low → early gradient updates are on near-identical data. Too high → wasted time. Consider uncommenting and tuning. |
| `grad_clip` | float \| None | None | *not set* | 1.0 | **Gradient clipping.** Hardcoded in `select_model.py`. Same as base, but especially important for SAC where large Q-values can cause exploding gradients. We added this after observing NaN crashes (SLURM job 1624328). |
| `store_buffer_in_checkpoints` | bool | False | *not set* | False | **Save replay buffer in checkpoints.** When True, checkpoints include the full buffer contents — makes checkpoints much larger but allows training to resume without losing experience. Useful for long training runs that may be interrupted. |
| `q_model_config` | dict | `{fcnet_hiddens: [256, 256], fcnet_activation: "relu", ...}` | *not set* | factory default | **Q-network architecture override.** RLlib's SAC factory default provides a full architecture spec (256×256 ReLU). When `config.rl_module(model_config=...)` is set, the `DefaultModelConfig` takes precedence for the shared encoder. This config allows overriding the Q-network architecture independently if the Q-function needs more capacity than the policy. |
| `policy_model_config` | dict | `{fcnet_hiddens: [256, 256], fcnet_activation: "relu", ...}` | *not set* | factory default | **Policy network architecture override.** Same structure and defaults as `q_model_config`, but for the actor network. |
| `_deterministic_loss` | bool | False | *not set* | False | **Debug: deterministic loss computation.** Removes the stochastic sampling step from the loss. Only for debugging — breaks the entropy regularisation that makes SAC work. |
| `_use_beta_distribution` | bool | False | *not set* | False | **Debug: Beta distribution for bounded actions.** Alternative to the default SquashedGaussian (tanh-squashed Normal) for bounded action spaces. Not recommended for production use. |

### Default vs Our Replay Buffer Configuration

RLlib 2.52 defaults to a **PrioritizedEpisodeReplayBuffer**:

```python
# RLlib default
{
    "type": "PrioritizedEpisodeReplayBuffer",
    "capacity": 1_000_000,
    "alpha": 0.6,   # How much prioritisation (0 = uniform, 1 = full priority)
    "beta": 0.4,    # Importance-sampling correction (0 = none, 1 = full correction)
}
```

We override to **EpisodeReplayBuffer** (uniform sampling):

```python
# Our config (select_model.py)
{
    "type": "EpisodeReplayBuffer",
    "capacity": 288 * 1500,  # 432,000 timesteps
    "alpha": 0.6,   # ⚠ Ignored — only used by PrioritizedEpisodeReplayBuffer
    "beta": 0.4,    # ⚠ Ignored — only used by PrioritizedEpisodeReplayBuffer
}
```

**Note:** `alpha` and `beta` are leftover from a previous configuration. They have no
effect on `EpisodeReplayBuffer` (uniform sampling). Either remove them for clarity, or
switch the type to `PrioritizedEpisodeReplayBuffer` to actually enable prioritisation.

**Prioritised replay** samples transitions with higher TD-error more frequently, which
can accelerate learning. The trade-off: it adds overhead (TD-error bookkeeping) and
can introduce bias corrected by importance-sampling weights (controlled by `beta`).
Worth experimenting with — switch to `PrioritizedEpisodeReplayBuffer` and see if
SAC converges faster on our building environment.

| Prioritised Replay Parameter | Default | Description |
|------------------------------|---------|-------------|
| `alpha` | 0.6 | Degree of prioritisation. 0.0 = uniform sampling (no priority). 1.0 = full priority (highest-TD-error transitions dominate sampling). 0.6 is a moderate default. |
| `beta` | 0.4 | Importance-sampling exponent. Corrects the bias introduced by non-uniform sampling. 0.0 = no correction. 1.0 = full correction. Often annealed from 0.4 to 1.0 during training. |
| `epsilon` (eps) | 1e-6 | Small constant added to TD-errors to ensure all transitions have non-zero sampling probability. Prevents starvation of zero-error transitions. |

### SAC Data Flow (How Parameters Interact)

```
Iteration:
1. EnvRunners collect experience using current stochastic policy
   (SAC's policy outputs a distribution — exploration comes from sampling it,
    not from explicit noise like TD3. Entropy regularisation keeps it stochastic.)
   → rollout_fragment_length = 64 per worker
   → Total new timesteps = 64 × num_env_runners

2. New experience is inserted into EpisodeReplayBuffer
   → Buffer capacity = 432,000 timesteps (1500 episodes)

3. If total_steps > num_steps_sampled_before_learning_starts (1500, factory default):
   → Determine number of gradient steps from training_intensity:
     num_updates = training_intensity × (new_steps / train_batch_size)
     = 32 × (64 × num_workers / 128)
     = 16 × num_workers gradient updates per iteration

   → For each gradient update:
       → Sample train_batch_size (128) transitions from buffer
       → Compute n-step (n=6) target Q-values using target networks (min of twin Q)
       → Update critic (Q-networks) via MSE loss with critic_lr
       → Update actor (policy) via SAC policy loss with actor_lr
       → Update alpha based on entropy vs target_entropy with alpha_lr
   → After each training call, if env_steps_lifetime - last_update >= target_network_update_freq (1):
       → Soft-update target networks: theta_t = tau×theta + (1-tau)×theta_t
```

---

## 4. Neural Network Configuration (DefaultModelConfig)

Shared by both PPO and SAC via `config.rl_module(model_config=...)`. Defines the
network architecture for default RLlib RLModules.

### Fully Connected Network (FCNet) — Used in Our Project

> Column legend: see [Section 1](#1-shared-base-parameters-algorithmconfig).

| Parameter | Type | Factory Default | User Default | Actual | Description |
|-----------|------|-----------------|--------------|--------|-------------|
| `fcnet_hiddens` | list[int] | [256, 256] | *not set* | [32, 32] | **Hidden layer sizes.** Each entry is a hidden layer with that many neurons. Factory default `[256, 256]` = two hidden layers of 256 units each. We override to `[32, 32]` in `select_model.py` — two small layers with 32 units each. This is intentionally compact for our relatively low-dimensional observation space (~10-20 features). If the agent struggles to represent complex policies, consider scaling up to [128, 128] or [256, 256]. |
| `fcnet_activation` | str | "tanh" | *not set* | "relu" | **Activation function.** Applied after each hidden layer. Options: "tanh", "relu", "swish"/"silu", "elu", "linear". ReLU is standard for most RL tasks — fast to compute, avoids vanishing gradients. Tanh squashes to [-1,1] which can be useful for normalised observations. Swish (x × sigmoid(x)) is smoother than ReLU and sometimes trains better. We override the factory default "tanh" to "relu" in `select_model.py`. |
| `fcnet_kernel_initializer` | str \| None | None | *not set* | None | **Weight initialisation scheme.** None uses PyTorch/TF defaults (Kaiming uniform for linear layers in PyTorch). Options include "xavier_uniform", "orthogonal", etc. Orthogonal initialisation is sometimes recommended for RL — it preserves gradient norms across layers. |
| `fcnet_bias_initializer` | str \| None | None | *not set* | None | **Bias initialisation.** None uses framework defaults (zeros in PyTorch). |

### Head Network (Policy/Value Heads)

| Parameter | Type | Factory Default | User Default | Actual | Description |
|-----------|------|-----------------|--------------|--------|-------------|
| `head_fcnet_hiddens` | list[int] | [] | *not set* | [] | **Additional layers after the shared encoder, before the output.** Empty = output head is a single linear layer on top of the encoder. Adding layers here gives the policy and value heads independent capacity beyond the shared layers. |
| `head_fcnet_activation` | str | "relu" | *not set* | "relu" | **Activation for head layers.** |
| `vf_share_layers` | bool | True | *not set* | True | **Whether policy and value function share the encoder.** True = shared representation (fewer parameters, may have gradient conflicts). False = separate encoders (more parameters, independent learning). For SAC, this doesn't apply (actor and critic are always separate). For PPO, sharing can cause the value function gradients to interfere with policy gradients — tune `vf_loss_coeff` carefully if sharing. |

### Stochastic Policy Parameters

| Parameter | Type | Factory Default | User Default | Actual | Description |
|-----------|------|-----------------|--------------|--------|-------------|
| `free_log_std` | bool | False | *not set* | False | **State-independent log-std.** When True, the log standard deviation of the policy is a free parameter (not a function of state). This simplifies the policy but removes the ability to be more uncertain in some states than others. False = std is state-dependent (output of the network). |
| `log_std_clip_param` | float | 20.0 | *not set* | 20.0 | **Clipping range for log-std.** Clips log_std to [-log_std_clip_param, +log_std_clip_param] to prevent numerical issues. 20.0 is very permissive. For SAC's SquashedGaussian, values of 2.0–5.0 are sometimes used to prevent extreme exploration/exploitation. |

### LSTM / Recurrent (Not Currently Used)

| Parameter | Type | Factory Default | User Default | Actual | Description |
|-----------|------|-----------------|--------------|--------|-------------|
| `use_lstm` | bool | False | *not set* | False | **Enable LSTM layer after the encoder.** Adds temporal memory — the agent can condition on the *sequence* of observations, not just the current one. Useful when the environment is partially observable (missing state information). Our building env might benefit from this since thermal dynamics have inertia. |
| `lstm_cell_size` | int | 256 | *not set* | 256 | **LSTM hidden state size.** Larger = more memory capacity but more parameters. For our use case, 32–128 would likely suffice. |
| `max_seq_len` | int | 20 | *not set* | 20 | **Maximum sequence length for LSTM training.** Episodes are chunked into sequences of this length for backpropagation through time (BPTT). Longer sequences capture more temporal context but use more memory and are slower to train. |
| `lstm_use_prev_action` | bool | False | *not set* | False | **Feed previous action to LSTM.** Helps the agent learn action-dependent dynamics (e.g. "I turned the heat pump on 2 steps ago, so temperature should be rising now"). Recommended for our use case. |
| `lstm_use_prev_reward` | bool | False | *not set* | False | **Feed previous reward to LSTM.** Helps the agent learn reward-predictive features. Less commonly useful than prev_action. |

### Convolutional Network (Not Applicable)

Parameters like `conv_filters`, `conv_activation`, etc. are for image/grid observations and not relevant to our vector-based observation space.

---

## 5. How Our Project Uses These Parameters

### Current Configuration (`training_param_config.yaml`)

```yaml
common:
  seed: 42
  episode_lookback_horizon_steps: 120  # 5 min steps, 120 steps = 10 hours; auto-raised to max(|hst.offsets|) if smaller
  max_episodes_to_run: 7000

ppo:
  episodes_per_iteration: 10    # → 10 × 288 = 2880 timesteps/iteration
  minibatch_size: 64
  num_epochs: 20

sac:
  replay_batch_size: 128
  days_to_keep_in_replay_buffer: 1500   # → 432,000 timesteps
  training_intensity: 32                # → UTD ≈ 0.25 (32/128)
  rollout_fragment_length: 64           # timesteps collected per env runner per round
```

### What We Set vs What We Leave as Default

> **Legend:** Factory = RLlib out-of-the-box. User = `training_param_config.yaml`. Actual = value reaching RLlib.

| Parameter | Factory Default | User Default | Actual (PPO) | Actual (SAC) | Notes |
|-----------|-----------------|--------------|--------------|--------------|-------|
| Learning rate | 5e-5 (PPO) / 3e-5 (SAC actor) / 3e-4 (SAC critic/alpha) | 3e-4 | `lr=3e-4` | `actor/critic/alpha_lr=3e-4` | We set 6× PPO factory, 10× SAC actor factory. SAC typically uses lower actor LR. |
| Batch size | 4000 (PPO) / 256 (SAC) | 10 episodes (PPO) / 128 (SAC) | 2880 ts/iter | 128 from replay | Fundamentally different semantics. PPO batch expressed as episodes in YAML, converted to timesteps. |
| Mini-batch / epochs | 128 / 30 | 64 / 20 | 64 / 20 | N/A (off-policy) | PPO-only. We use fewer epochs and smaller mini-batches than factory default. |
| Grad clip | None | *not set* | None | 1.0 | Only SAC — hardcoded in `select_model.py` after NaN crash. |
| Gamma | 0.99 | *not set* | 0.99 | 0.99 | Factory default used for both. |
| GAE lambda | 1.0 | *not set* | 1.0 | N/A | PPO-only, factory default (full MC returns). |
| KL loss | True | *not set* | True | N/A | Factory default active with default coeff/target. |
| Clip param | 0.3 | *not set* | 0.3 | N/A | PPO-only, factory default. |
| Entropy coeff | 0.0 (PPO) | *not set* | 0.0 | *auto-tuned alpha* | Different mechanisms per algorithm. |
| Twin Q | True | *not set* | N/A | True | Factory default. |
| Tau | 0.005 | *not set* | N/A | 0.005 | Factory default. |
| Target update freq | 0 (every call) | *not set* | N/A | 1 | Hardcoded in `select_model.py`. Effectively same as factory default (update every step). |
| N-step | 1 | *not set* | N/A | 6 | Hardcoded in `select_model.py`. 30 min lookahead for thermal dynamics. |
| Training intensity | None | 32 | N/A | 32 | UTD ≈ 0.25 (32/128). Lower than standard SAC UTD of 1.0. |
| Rollout fragment length | 1 (SAC default) | 64 | N/A | 64 | Explicit setting; SAC default of 1 causes broken episode callbacks. |
| Replay buffer | PrioritizedEpisodeReplayBuffer, 1M | EpisodeReplayBuffer, 1500 days | N/A | EpisodeReplayBuffer, 432k | Uniform replay with smaller capacity. Has stale alpha/beta params (ignored). |
| Network | [256, 256] tanh | *not set* | [32, 32] relu | [32, 32] relu | We override both size and activation in `select_model.py`. |

### Parameters Worth Experimenting With

**High priority (likely impact on performance):**

1. **`gamma`** (both) — Currently 0.99. Try 0.95 and 0.995 to see how discount rate affects battery management strategy. Lower gamma → agent focuses on immediate comfort/cost. Higher gamma → better long-term planning.

2. **`lambda_`** (PPO) — Currently 1.0 (full Monte Carlo). Try 0.95 or 0.97 — this should reduce variance in advantage estimates once the critic is reasonably accurate.

3. **`clip_param`** (PPO) — Currently 0.3. The original paper uses 0.2. Lower values make PPO more conservative (slower but stabler). Try 0.1–0.2 if training is unstable.

4. **`actor_lr` vs `critic_lr`** (SAC) — Currently identical (3e-4). Standard practice uses actor_lr = 3e-5 (or 1e-4) with critic_lr = 3e-4. This lets the critic converge first, giving the actor better gradient signals.

5. **`n_step`** (SAC) — Currently 6 (30 min lookahead). Already non-default. If training is unstable, try lowering to n=3 (15 min). If stable, n=8-10 (40-50 min) could propagate thermal dynamics signals further, but increases bias.

6. **`entropy_coeff`** (PPO) — Currently 0.0. Adding a small entropy bonus (0.001–0.01) may help PPO explore more action combinations (e.g. different heat pump modes).

**Medium priority:**

7. **`num_epochs`** (PPO) — Currently 20 (RLlib default is 30). The original PPO paper uses 3–10. Both our value and the default are high. More epochs risk overfitting to the current batch. Try 4–10.

8. **`vf_clip_param`** (PPO) — Currently 10.0. With 288-step episodes and [-1,1] rewards, episode returns can reach ~288. The value function may need more room — try 50 or 100.

9. **`training_intensity`** (SAC) — Currently 32 (UTD ≈ 0.25). Below standard SAC's UTD of 1.0. Increasing to 128 (UTD ≈ 1.0) would match the standard setting. Higher values (256, 512) train more aggressively on existing data (REDQ-style) — higher sample efficiency but diminishing returns and overfitting risk.

10. **LSTM** (both) — Enabling `use_lstm=True` with `lstm_use_prev_action=True` would let the agent learn temporal patterns in thermal dynamics. Adds complexity and training time, but our building physics has clear temporal dependencies.

**Lower priority:**

11. **`fcnet_hiddens`** — Currently [32, 32]. Try [64, 64] or [128, 128] (more capacity) if the agent underfits. Try [32, 32, 32] (deeper but narrow) for more representational depth.
12. **`initial_alpha`** (SAC) — Try 0.1 or 0.5 for less aggressive initial exploration.
13. **`target_entropy`** (SAC) — Manual tuning instead of "auto" for finer exploration control.

---

## 6. Parameter Interactions and Tuning Notes

### PPO: The Three-Way Balance

PPO has a three-way interaction between **`clip_param`**, **`kl_coeff`/`kl_target`**, and **`num_epochs`**:

- More epochs → more gradient updates → higher KL divergence per iteration
- Higher KL → if mean KL > 2× `kl_target`, `kl_coeff` is multiplied by 1.5 → effectively reduces learning rate
- Smaller `clip_param` → tighter constraint → limits how much each epoch can change the policy

If you increase `num_epochs`, you may need to lower `clip_param` or `kl_target` to maintain stability. Conversely, if training is too slow (conservative), reduce epochs or increase `clip_param`.

### PPO: Batch Size vs Mini-batch Size

`train_batch_size_per_learner / minibatch_size` determines the number of mini-batches per epoch. With 2880/64 = 45 mini-batches and 20 epochs, we do **900 gradient updates per iteration**. For comparison, the RLlib default is 4000/128 = 31 mini-batches × 30 epochs = 930 updates, and typical PPO setups in literature use 2048 batch, 64 minibatch, 4 epochs = 128 updates. Our setup is comparable to RLlib defaults (~900 vs 930) but still ~7× more than literature baselines. This could lead to overfitting on the current batch — consider reducing `num_epochs` to 4–10.

### SAC: UTD and Training Intensity

The update-to-data (UTD) ratio is the key SAC tuning knob:

```
UTD = training_intensity / train_batch_size_per_learner
    = 32 / 128 = 0.25
```

This means one gradient step per four new environment steps — below the standard SAC UTD of 1.0. This conservative setting prioritises stability and training speed over sample efficiency. Increasing `training_intensity` to 128 (UTD ≈ 1.0) would match the standard setting. Recent work (REDQ, DroQ) shows that even higher UTD (10-20) can dramatically improve sample efficiency by reusing replay data more aggressively, though this requires additional regularisation (dropout, ensemble Q-networks) to prevent overfitting.

### SAC: Replay Buffer Sizing

Buffer capacity determines how old the oldest experience can be:

```
buffer_capacity = 432,000 timesteps = 1,500 episodes
timesteps_per_iteration = rollout_fragment_length × num_env_runners = 64 × num_env_runners
```

With e.g. 8 env runners, each iteration adds 64 × 8 = 512 timesteps (~1.8 episodes). The buffer holds 432,000 / 512 ≈ 844 iterations of experience. This is generous — the oldest data is from ~844 iterations ago, when the policy was significantly different. Very old data may have low relevance but still contributes to critic stability. If the policy changes rapidly, a smaller buffer (500–1000 episodes) could help by keeping data more on-policy.

### Network Size and Training Stability

Our current network is small ([32, 32]). If scaling up (e.g. to [128, 128] or [256, 256]), be aware of these interactions:
- **Learning rate**: Larger networks may need lower LR to avoid overshooting
- **Grad clip**: More parameters → higher global gradient norm → more likely to trigger clipping
- **Training intensity (SAC)**: Larger networks need more gradient steps to converge
- **Num epochs (PPO)**: Larger networks can absorb more epochs without overfitting (more parameters = more capacity)

### RLlib-Specific Gotchas

1. **`train_batch_size` vs `train_batch_size_per_learner`**: On the new API stack, always use `train_batch_size_per_learner`. The old `train_batch_size` is the *total* across all learners and is deprecated.

2. **`rollout_fragment_length`**: For SAC, we explicitly set this to 64 (via `sac_rollout_fragment_length` in YAML). Without it, SAC defaults to 1, which causes episode callbacks to report length=1 (a known RLlib quirk). The value does not need to match `episode_length` — it just controls how many timesteps each env runner collects per round before returning data to the learner.

3. **`EpisodeReplayBuffer` vs `ReplayBuffer`**: The new API stack requires an Episode-based buffer. RLlib 2.52 defaults to `PrioritizedEpisodeReplayBuffer`; we override to plain `EpisodeReplayBuffer`. Using the old `ReplayBuffer` type (non-episode) will crash.

4. **`vf_share_layers`**: Defaults to True in `DefaultModelConfig`, but on the new API stack, PPO's RLModule typically creates separate encoder instances for policy and value function regardless (check implementation). SAC always has separate actor/critic networks.

5. **Observation space with `FlattenObservations` connector**: Do not pass `observation_space` to `config.environment()` — the connector transforms it, and RLlib infers the flattened space automatically. Passing the original Dict space causes shape mismatches.
