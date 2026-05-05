# MORL with Dirichlet-Sampled Reward Weights

**Context:** RL, multi-objective RL, PPO and SAC, RLlib new API stack, multiple reward functions.

**Question:** What are the benefits and drawbacks of learning the Pareto front by using different reward weights for each episode? Weights are sampled from a Dirichlet distribution — how do we ensure that the reward weighting and the optimal actions/optimal policy are both learnt?

---

Your approach — using Dirichlet-distributed reward weights to sample different reward weightings per episode in multi-objective RL (MORL) — is a classic way to approximate the Pareto front (the set of non-dominated policies). This is often called **weighted-sum scalarization** or **linear scalarization**. Below are the benefits, drawbacks, and how to ensure joint learning of reward weights and optimal policies.

## Benefits of Sampling Reward Weights per Episode

### 1. Approximates the Pareto Front
- By sampling weights from a Dirichlet distribution (which generates compositions over the simplex), you explore a diverse set of trade-offs between objectives.
- Each weight vector corresponds to a different scalarized reward function, and the optimal policy for that weighting is a point on the Pareto front.
- Over many episodes, you can approximate the entire front by collecting policies for different weightings.

### 2. Simple and Scalable
- No need for complex multi-objective algorithms (e.g., NSGA-II, MOEA/D).
- Works with any single-objective RL algorithm (PPO, SAC, etc.) by just modifying the reward function.
- Easy to implement in frameworks like RLlib (just pass the sampled weights as part of the episode config).

### 3. Theoretical Guarantees
- If the true Pareto front is convex, the weighted-sum method is guaranteed to find all Pareto-optimal solutions (under ideal conditions).
- Dirichlet sampling ensures uniform coverage of the weight simplex, which helps in exploring the front thoroughly.

### 4. Adaptability
- Can dynamically adjust the Dirichlet concentration parameter (α) to focus on certain regions of the front (e.g., higher α for more uniform weights, lower α for extreme trade-offs).

### 5. Compatibility with Existing RL Methods
- Works seamlessly with PPO, SAC, or any policy-gradient method since the reward is just a weighted sum.
- No need to modify the RL algorithm itself — just the reward function.

## Drawbacks and Challenges

### 1. Non-Convex Pareto Fronts
- If the Pareto front is non-convex, the weighted-sum method cannot represent all Pareto-optimal solutions.
- Some parts of the front may be missed entirely, leading to an incomplete approximation.

### 2. Inefficient Exploration
Random sampling may waste episodes on:
- Redundant weightings (multiple episodes with similar weights).
- Uninformative regions (e.g., weights that lead to degenerate policies).
- No adaptive focusing on uncertain or interesting parts of the front.

### 3. Credit Assignment Problem
The agent must simultaneously learn:
- How to weight the rewards (which weights lead to good policies).
- The optimal policy for each weighting.

If the weights are purely random, the agent may struggle to correlate weights with good policies.

### 4. Instability in Learning
Changing reward weights every episode can lead to:
- Non-stationary learning dynamics (the "optimal" action changes frequently).
- Slow convergence (the agent never fully optimizes for any single weighting).

This is especially problematic for on-policy methods like PPO, which assume a fixed reward function.

### 5. Dirichlet Sampling Limitations
- `Dirichlet(α=1)` is uniform, but it favors extreme weights (e.g., `[0.9, 0.05, 0.05]`) more than balanced ones (e.g., `[0.33, 0.33, 0.33]`).
- May not cover the entire simplex uniformly in high dimensions.
- Sparse rewards: if one objective dominates, the agent may ignore others.

### 6. Computational Cost
- Requires many episodes to approximate the front well.
- Each episode may require full retraining (or at least fine-tuning) for the new weights.

## How to Ensure Joint Learning of Weights and Policies

The core challenge: *How to learn both the reward weights and the optimal policies simultaneously?*

### 1. Two-Timescale Learning (Meta-Learning)
- **Fast timescale:** update the policy for a fixed set of weights (inner loop).
- **Slow timescale:** update the weight sampling distribution based on performance (outer loop).

Examples:
- Use Dirichlet as a prior, but adapt its parameters (α) over time to focus on promising regions.
- Use gradient-based optimization (e.g., RL², MAML) to learn weights that lead to high-performing policies.

```python
# Pseudocode
for episode in range(total_episodes):
    # Sample weights from Dirichlet (adaptive α)
    weights = dirichlet_sample(alpha=learned_alpha)

    # Run RL with these weights
    policy = train_ppo_or_sac(env, weights)

    # Update α to favor weights that led to high returns
    if episode % meta_update_freq == 0:
        learned_alpha = update_alpha(policy_performance, learned_alpha)
```

### 2. Conditional Policy Networks
Train a single policy that takes reward weights as input (conditioning). The policy learns to adapt its behavior based on the weights.

**Advantage:** no need to retrain for each weight — just condition the policy.

**Implementation:** modify the policy network to accept weights as an additional input (e.g., concatenate with observations). Use PPO/SAC with a weighted reward and train the policy to generalize across weights.

```python
# Policy network takes (obs, weights) as input
def policy_network(obs, weights):
    x = concatenate([obs, weights])
    return actor(x), critic(x)

# During training, sample weights per episode and pass to policy
for episode in range(total_episodes):
    weights = dirichlet_sample(alpha)
    rewards = weighted_sum_reward(weights)
    train_ppo(policy_network, env, rewards, weights)
```

### 3. Multi-Objective RL Algorithms

Instead of random sampling, use dedicated MORL algorithms that explicitly learn the Pareto front:

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| NSGA-II | Evolutionary algorithm for multi-objective optimization. | Finds diverse front, no weight tuning. | Slow, not sample-efficient. |
| MOEA/D | Decomposition-based MORL. | Good for high-dimensional objectives. | Complex to implement. |
| Pareto RL | Maintains a set of policies approximating the front. | Explicitly tracks Pareto-optimal policies. | Computationally expensive. |
| Constrained RL | Optimize one objective subject to constraints on others. | Works well for prioritized objectives. | Requires constraint tuning. |

**Recommendation:** if you want to stick with weight sampling, combine it with conditional policies or two-timescale learning.

### 4. Adaptive Weight Sampling
Instead of uniform Dirichlet sampling, use:
- **Thompson Sampling:** model weights as a distribution and sample from it (e.g., Bayesian optimization).
- **Upper Confidence Bound (UCB):** prefer weights with high uncertainty or potential.
- **Curriculum Learning:** start with extreme weights (e.g., `[1,0,0]`, `[0,1,0]`) and gradually move toward balanced weights.

```python
# Initialize a distribution over weights (e.g., Dirichlet)
weight_distribution = Dirichlet(alpha=[1.0, 1.0, 1.0])

for episode in range(total_episodes):
    # Sample weights from current belief
    weights = weight_distribution.sample()

    # Train policy with these weights
    policy = train_ppo(env, weights)
    returns = evaluate(policy)

    # Update weight_distribution based on returns
    weight_distribution.update(weights, returns)
```

### 5. Regularization and Diversity Promotion
- **Reward bonus for diversity:** add a term to the reward that penalizes similar policies (e.g., encourage policies to cover different parts of the front).
- **Entropy regularization:** ensure the weight distribution remains diverse (e.g., maximize entropy of sampled weights).
- **Pareto archive:** maintain a set of non-dominated policies and bias sampling toward under-explored regions.

### 6. Hybrid Approach: Weighted Sum + Pareto Filtering
1. Sample weights from Dirichlet and train policies.
2. Filter the policies to keep only the Pareto-non-dominated ones.
3. Use the Pareto archive to guide future weight sampling (e.g., focus on regions where the front is sparse).

## Practical Recommendations for RLlib

Since you're using RLlib's new API stack, here's how to implement this:

### 1. Custom Episode Config
Use `register_env` with a custom config that samples weights per episode.

```python
def env_creator(config):
    env = YourMultiObjectiveEnv(config)
    # Sample weights per episode
    weights = np.random.dirichlet(config["dirichlet_alpha"])
    env.set_reward_weights(weights)
    return env

register_env("mo_env", env_creator)
```

### 2. Conditional Policy (Recommended)
Modify the policy to accept weights as input:

```python
class WeightConditionedPolicy:
    def __init__(self, obs_space, action_space, weight_dim):
        self.actor = nn.Sequential(
            nn.Linear(obs_space + weight_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_space)
        )

    def forward(self, obs, weights):
        x = torch.cat([obs, weights], dim=-1)
        return self.actor(x)
```

In RLlib, pass weights as part of the `info` dict or observation.

### 3. Two-Timescale Learning
Use RLlib's `Trainer` API to alternate between:
- Policy updates (inner loop, fast).
- Weight distribution updates (outer loop, slow).

```python
trainer = PPOTrainer(config)
for i in range(num_meta_iterations):
    # Sample weights and train policy
    weights = dirichlet_sample(alpha)
    result = trainer.train()
    # Update alpha based on results
    alpha = update_alpha(result, alpha)
    trainer.config["dirichlet_alpha"] = alpha
```

## Key Takeaways

| Goal | Recommended Approach | RLlib Compatibility |
|------|----------------------|---------------------|
| Approximate Pareto front | Dirichlet sampling + PPO/SAC | ✅ Easy |
| Learn weights and policies jointly | Conditional policy + two-timescale learning | ✅ Moderate |
| Handle non-convex fronts | Pareto RL or MOEA/D | ❌ Harder |
| Improve sample efficiency | Adaptive weight sampling (Thompson/UCB) | ✅ Moderate |
| Ensure diversity | Pareto filtering + entropy regularization | ✅ Easy |

## Final Answer

> *"How to ensure that the reward weighting and the optimal actions/optimal policy are both learnt?"*

Use a combination of:
1. **Conditional policies** (weights as input to the policy network) to generalize across weightings.
2. **Adaptive weight sampling** (e.g., Thompson Sampling, UCB) to focus on promising regions of the weight space.
3. **Two-timescale learning** to update weights slowly while the policy adapts quickly.
4. **Pareto filtering** to maintain a diverse set of non-dominated policies.

This ensures that:
- The policy learns to adapt to different weightings (instead of overfitting to one).
- The weight sampling improves over time (instead of being purely random).
- The Pareto front is approximated efficiently.
