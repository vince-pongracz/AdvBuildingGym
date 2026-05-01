# Changing Reward Weights — Idea & Analysis

## Q1: Per-step random reward weights, conditioned policy

> Look at the following scenario: RL algorithms, PPO and SAC. In each step I have a random weight set, weighting the rewards. PPO and SAC must predict the actions and the expected reward. The PPO and SAC receive the casual env state in a step and the actual sampled reward weighting. Analyse this scenario, benefits, drawbacks, etc.

What you're describing is essentially training PPO and SAC under a **non-stationary reward function**, where the reward is dynamically reweighted at every step. It overlaps with ideas from multi-objective RL, meta-RL, and preference-conditioned policies.

### What the setup really is

At each timestep:

- The environment state $s_t$ is observed
- A random reward weight vector $w_t$ is sampled
- The agent receives both $(s_t, w_t)$
- The reward is computed as a weighted combination: $r_t = w_t \cdot r_t^{\text{components}}$

So effectively, the agent is learning a **policy, conditioned on reward preferences**. This is similar to:

- Universal Value Function Approximators (UVFA)
- Preference-conditioned RL
- Multi-objective RL with scalarization

### Potential benefits

1. **Generalization across reward functions** — instead of learning a policy for one objective, the agent learns "given this weighting, what should I do?". This produces a single policy that adapts to many objectives, with post-training flexibility (change weights without retraining).
2. **Better coverage of the Pareto front** — sampling many weightings explores many trade-offs and encourages policies across the Pareto frontier.
3. **Implicit robustness** — because the reward changes constantly, the agent can't overfit to a fixed reward signal and may learn more stable, general strategies.
4. **Meta-learning flavor** — "learning how to optimize any reward weighting". For SAC (off-policy), this could reuse experience across different weightings and improve sample efficiency in theory.

### Major drawbacks / challenges

1. **Non-stationarity (biggest issue)** — from the agent's perspective, the reward function changes every step, violating the core RL assumption that the MDP is stationary. Consequences: value function becomes hard to learn, targets become noisy and inconsistent, slower or unstable convergence.
2. **Credit assignment becomes messy** — if weights change every timestep, long-term return is inconsistent and the same action in the same state can lead to very different returns. This hurts PPO's advantage estimation and SAC's Q-function learning.
3. **PPO instability (on-policy sensitivity)** — PPO is particularly sensitive because it relies on consistent advantage estimates; rapidly changing rewards lead to high variance gradients, noisy updates, possible collapse or slow learning.
4. **SAC complications (off-policy mismatch)** — SAC stores transitions in a replay buffer, but those transitions were generated under different weights. Q-learning assumes a consistent reward distribution; mixing different reward weightings introduces off-policy bias — unless you include $w_t$ in the state (which you do) AND the Q-function conditions on it.
5. **Exploration confusion** — the agent is optimizing "all reward functions at once", which can lead to a diluted learning signal and no strong gradient toward any particular behavior.
6. **Policy may average instead of specialize** — instead of learning sharp strategies, the agent may learn compromise behaviors, especially if weights are uniformly random.

--> 1. Mitigate non-stat with episode-wise weight change
--> 2.3.It's mitigated if weights stay consistent longer
--> 4. ensure strong conditioning -- 
--> 5. and 6.: how to mitigate these?

### PPO vs SAC in your setup

**PPO**

- Pros: simpler, stable in stationary settings.
- Cons: high variance from changing rewards, advantage estimation becomes unreliable, likely struggles more than SAC.

**SAC**

- Pros: off-policy → reuses data; entropy regularization helps exploration; can handle conditioning better.
- Cons: replay buffer contains mixed reward regimes; Q-function must learn a much harder mapping $Q(s, a, w)$.

Overall: **SAC is likely better suited than PPO**, but still non-trivial.

### Key design choices

1. **Include weights in the state** (already done) — essential: $\pi(a \mid s, w),\; Q(s, a, w)$. Without this, the problem is unsolvable.
2. **Control how weights are sampled** — uniform random may not be ideal. Better: structured sampling (e.g. Dirichlet), curriculum over weights, focus on meaningful trade-offs.
3. **Keep weights fixed per episode** (important improvement) — sample $w$ once per episode. This restores stationarity within a trajectory and dramatically stabilizes learning.
4. **Normalize rewards** — different weights produce different reward scales; normalize reward components or total reward.
5. **Consider multi-head critics or architectures** — shared encoder + weight-conditioned head, or separate critics per objective.

--> Why don't I use multi head critics? Uniform head networks, as many as reward functions, each of them sitting on top of the base critics network -- that's the SOTA.

### When this approach works well

- You want one model for many objectives.
- Preferences change at runtime.
- You care about trade-offs, not a single optimum.

Examples: robotics (speed vs safety), recommendation systems (engagement vs diversity), resource allocation.

### When it's a bad idea

- You only need a single fixed reward.
- Environment is already noisy.
- Sample efficiency is critical.
- You need stable, fast convergence.

### Bottom line

Powerful but tricky.

- **Strengths**: generalization across objectives, Pareto-efficient behavior, flexible deployment.
- **Weaknesses**: breaks stationarity, harder optimization, PPO likely struggles, SAC requires careful design.

### Practical recommendation

1. Condition policy and critic on weights.
2. **Sample weights per episode, not per step** (biggest fix).
3. Use SAC over PPO.
4. Normalize reward components.
5. Consider structured weight sampling.

---

## Q2: Per-episode Gaussian resampling with moving-average mean

> What would happen if only per episode the weights are resampled with some constraints — Gaussian resampling of weights with a predefined variance and mean is a moving avg of the last N weights and then they are normalised, so that the weight sum is 1.
>
> To "credit assignment becomes messy": the sampled weights are part of the state space, so "the same action same state can lead to very different returns" doesn't seem correct.
>
> To "SAC complications (off-policy mismatch)": the Q function should condition on it since the weights are part of the state space — the Q function gets those states.
>
> "Policy may average instead of specialize": how to mitigate? I assume longer episodes with same weights already mitigate this a bit. But how to prevent e.g. a scenario where only 1 weight is non-zero — all others zeroed out, so the multi-objectivity collapses.
>
> "Control how weights are sampled": what do you mean by curriculum? What is Dirichlet sampling?

Switching to per-episode weight sampling is a big improvement. It changes the problem from "chaotic non-stationary RL" into something much closer to a **contextual MDP** or **preference-conditioned RL**, which is far more tractable.

### 1. Per-episode resampling with Gaussian + moving average

What you're doing:

- At episode start, sample weights $w \sim \mathcal{N}(\mu_t, \Sigma)$.
- Mean evolves as moving average of last $N$ weights.
- Normalize so $\sum_i w_i = 1$.

What this changes:

- **Stationarity within episodes** — each trajectory has a fixed reward function. PPO advantage estimation becomes valid again, SAC targets become consistent.
- **Smoother exploration of weight space** — moving average prevents wild jumps; Gaussian variance controls diversity.
- **But introduces bias** — you're no longer sampling the full weight space uniformly; you bias learning toward recent preferences. This can be good (stabilizes learning) or bad (may collapse exploration of the Pareto front).

### 2. "Credit assignment becomes messy" — your objection

You're right to push back, but there's nuance.

Your argument: weights are part of the state, so the same $(s, w, a)$ always leads to consistent returns. **Correct** if $w$ is part of the state and fixed during the episode. Then the problem becomes $Q(s, a, w)$ — no ambiguity, credit assignment is well-defined.

Where the original concern still partially applies: even with conditioning, the agent must learn many reward functions simultaneously; the same $(s, a)$ pair appears with many different $w$. This leads to higher variance targets and harder function approximation. So the issue isn't *incorrectness* but **learning complexity**.

### 3. SAC off-policy issue — your correction

Again, you're mostly right. If $Q = Q(s, a, w)$, then replay-buffer transitions are valid because each transition includes its $w$ and Bellman targets are consistent for that $w$. This fixes the theoretical mismatch.

Remaining practical issue: the replay buffer now contains a **mixture of tasks**. SAC must learn $Q(s, a, w)$ across many $w$, which can cause slower convergence and underfitting if network capacity is too small. Not a correctness issue — a **capacity / generalization issue**.

### 4. Policy "averaging instead of specializing"

Your intuition is right — longer episodes already help, because the policy experiences coherent reward signals per trajectory and is encouraged toward consistent behavior per $w$.

How to further mitigate averaging:

1. **Strong conditioning on $w$** — make sure $w$ is deeply integrated into the network (not just concatenated at input). Better: FiLM layers (feature-wise modulation), attention over reward weights, separate embedding for $w$.
2. **Increase weight diversity carefully** — if weights are too similar, the policy learns "average behavior". Ensure sufficient variance.
3. **Contrastive learning signal (advanced)** — encourage different weights → different behaviors; e.g. penalize similar actions for very different $w$.
4. **Multi-head or modular policies** — shared backbone with weight-conditioned heads instead of one monolithic policy.
5. **Train with "extreme" weights explicitly** — include cases like one objective dominant and balanced objectives to improve specialization.

### 5. Degenerate case: one weight = 1, others = 0

This is **not necessarily bad** — it corresponds to optimizing a single objective, useful for covering the Pareto front. Problems arise only if it happens too often or dominates training.

How to prevent collapse:

- **Lower-bound weights**: enforce $w_i \geq \epsilon$.
- **Entropy regularization on weights**: encourage spread, penalize spiky distributions.
- **Controlled sampling distribution**: avoid extremes.

### 6. Dirichlet sampling

The Dirichlet distribution is perfect for this case. It generates vectors $w$ such that $w_i \geq 0$ and $\sum_i w_i = 1$. You don't need the Gaussian + normalization hack — it naturally produces valid weights.

Behavior controlled by $\alpha$:

- $\alpha < 1$: sparse, extreme weights (one dominates).
- $\alpha = 1$: uniform over the simplex.
- $\alpha > 1$: smooth, balanced weights.

Practical use:

- Want diversity? $\alpha < 1$.
- Want stability? $\alpha > 1$.
- Want both? Mix distributions.

### 7. What "curriculum" means here

A curriculum = gradually changing training difficulty. In your context: gradually changing how weights are sampled.

| Phase | Sampling | Purpose |
|-------|----------|---------|
| 1 — Simple | Fixed or near-uniform weights | Easy learning signal |
| 2 — Moderate | Slight variation (Gaussian / Dirichlet $\alpha > 1$) | Introduce diversity |
| 3 — Hard | Full diversity (including extreme weights) | Generalization |

Why this helps: stable gradients early, generalization later. Without a curriculum, the agent may struggle early due to too much variability.

### Updated overall assessment

With your modifications:

- Stationarity restored (per episode).
- SAC becomes fully valid.
- PPO becomes viable again.
- Credit assignment is correct.

Remaining challenges:

- Function approximation complexity: learning $Q(s, a, w)$ is harder than $Q(s, a)$.
- Coverage of weight space: moving average may reduce diversity.
- Policy collapse to "average behavior".

### Final practical recommendations

1. Prefer Dirichlet sampling over Gaussian normalization.
2. Keep weights fixed per episode.
3. Use SAC with sufficiently large networks.
4. Ensure strong conditioning on $w$.
5. Maintain diversity of weights (avoid too small variance).
6. Consider a curriculum over weight distributions.
7. Optionally enforce a minimum weight floor.

---

## Q3: Policy predicting expected reward + ensuring strong conditioning on w

> Is it possible that the policy NN predicts the actions and the expected reward as well — then that reward is compared with the true reward, which could be used for some learning, for some loss function? Further follow-up: how to ensure strong conditioning on $w$?

Yes — you can have the policy network also predict expected return and train it with a loss, but be careful about how that signal is used. Done right, it helps representation learning; done naively, it destabilizes RL.

### 1. Policy predicting actions and expected reward

What you're proposing is essentially a **shared network with two heads**:

- Actor head: $\pi(a \mid s, w)$
- Value head: $V(s, w)$ or $Q(s, a, w)$

This already exists in standard actor–critic methods: PPO uses a value head; SAC uses Q-functions (usually separate networks, but can share encoders). So you're rediscovering a common pattern.

#### Your twist: comparing predicted reward vs actual reward

This can mean a few things:

**Case A — Predict immediate reward $\hat r(s, a, w)$.** Loss: $L_{\text{reward}} = (\hat r - r)^2$.

- Pros: helps learn better reward-structure representations, useful for complex / multi-objective rewards, can stabilize early learning.
- Cons: doesn't directly optimize long-term return; can distract from policy learning if over-weighted.
- Verdict: **safe and often useful as an auxiliary loss**.

**Case B — Predict expected return $\hat V(s, w) \approx G_t$.** Loss: $(\hat V - G_t)^2$. This is just standard value-function learning — not adding anything new, just rephrasing actor–critic.

**Case C (dangerous) — Use reward prediction error to train the policy directly.** "If predicted reward is wrong → update policy" breaks RL principles: the policy should be optimized via advantage / Q-values, not via raw reward prediction error. Leads to bias, instability, wrong gradients.

#### Best practical version

Use it as **auxiliary learning, not primary signal**:

$$L = L_{\text{RL}} + \lambda_r L_{\text{reward}}$$

where $L_{\text{RL}}$ is the PPO or SAC loss and $L_{\text{reward}}$ is the reward prediction MSE. This often improves sample efficiency and helps representation learning for $w$.

### 2. How to ensure strong conditioning on w

Crucial — if conditioning is weak, everything collapses to "average policy". Concatenation alone is often not enough.

**Method 1 — Feature-wise modulation (FiLM, very effective).** Instead of $[s, w] \to \text{network}$, encode state into features $h(s)$ and use $w$ to modulate them: $h' = \gamma(w) \odot h + \beta(w)$. Forces the network to use $w$; different $w$ produce different internal activations.

**Method 2 — Separate encoders.** $h_s = f(s)$, $h_w = g(w)$, then combine via concatenation at deeper layers or multiplicative interaction. Better than raw input concatenation.

**Method 3 — Attention over objectives.** Treat weights as importance signals; state features attend to objectives weighted by $w$. More complex but powerful.

**Method 4 — Conditioning at multiple layers.** Inject $w$ throughout the network: early layers for coarse behavior, late layers for fine control.

**Method 5 — Normalize and structure $w$.** Already normalized to sum 1. Also consider log-transform if very skewed; ensure consistent scale.

### How to verify conditioning is actually working

Often overlooked:

- **Test 1 — Fix state, vary $w$**: does the action change meaningfully? If not, conditioning is weak.
- **Test 2 — Interpolation**: interpolate between two weights; behavior should change smoothly.
- **Test 3 — Extreme weights**: with one objective dominant, does the policy specialize?

### Putting it together

Refined idea:

- Use SAC (preferred).
- Condition on $w$ strongly (FiLM or similar).
- Sample weights per episode (Dirichlet recommended).
- Add an auxiliary reward-prediction head.

What you gain: better representations of reward structure, stronger dependence on $w$, improved generalization across objectives.

What to avoid: training the policy directly from reward prediction error; relying on simple concatenation only; letting the weight distribution collapse.

### Intuition summary

You're moving toward: *"a single agent that understands what you care about (via $w$) and adapts behavior accordingly."*

To make that work:

- $w$ must deeply influence the network.
- Learning must remain stable (RL loss still primary).
- Auxiliary tasks (like reward prediction) should support, not dominate.
