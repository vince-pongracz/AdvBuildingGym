# Dirichlet Reward Weighting for Multi-Objective RL

## Question

> RL, multi-objective RL, PPO/SAC, reward weighting: I want to weight e.g. 3 reward functions as parts of a composite sum reward. I want to try several weightings — let's say each weighting has 20 episodes to perform. For this weighting I then get a mean reward calculated over the 20 episodes. Then I get another weighting and another mean reward for that new weight set.
>
> I want to use a Dirichlet distribution for weight selection — in the beginning with $\alpha = 1.0$. But after 500 episodes I get a score overview about certain weightings and their scores. How can I reparametrise the Dirichlet so it uses the scores (mean rewards) for the recently sampled weight combinations and takes them into account? Obviously, exploration should still remain, but the approach should yield better weights after getting some scores on uniformly sampled weights.

## Answer

This is a classic **meta-optimization** problem. You are essentially treating the weight selection as an upper-level optimization task (bandit or black-box optimization) while the RL agent handles the lower-level control task.

To reparameterize your Dirichlet distribution based on performance, you can use a **Weighted Method of Moments** or a **Cross-Entropy Method (CEM)** approach. Since the Dirichlet is defined by the vector $\alpha = [\alpha_1, \alpha_2, \alpha_3]$, your goal is to shift these values so the distribution's mean moves toward "successful" weight regions while maintaining enough variance for exploration.

Here is how you can systematically update your $\alpha$ parameters.

---

### 1. Data collection and normalization

After your 500 episodes (25 samples of $w$, assuming 20 episodes each), you have a dataset of:

$$
D = \{(w^{(i)}, \bar{R}^{(i)})\}_{i=1}^{25}
$$

Where $w^{(i)}$ is the $i$-th weight vector sampled from $\text{Dir}(\alpha)$ and $\bar{R}^{(i)}$ is the mean reward obtained.

**Crucial step:** you must normalize or filter these rewards. If you use raw rewards, a single high-scale reward component could hijack the distribution.

- **Option A — Elite sampling:** pick the top $K$ weight sets (e.g. the top 5 performers).
- **Option B — Softmax weighting:** transform rewards into "importance weights" $v_i$:

$$
v_i = \frac{\exp(\bar{R}^{(i)} / \tau)}{\sum_j \exp(\bar{R}^{(j)} / \tau)}
$$

where $\tau$ is a temperature parameter controlling how aggressively you favour high scores.

---

### 2. The reparameterization formula

To find the new $\alpha$, you want the new expected value $E[X]$ of the distribution to align with the successful weights.

The mean of a Dirichlet for the $k$-th component is:

$$
E[w_k] = \frac{\alpha_k}{\sum_{j=1}^{K} \alpha_j}
$$

To update $\alpha$ from your scores, use the **weighted mean** of your sampled weights.

**Calculate the target mean** $\mu^*$:

$$
\mu^* = \sum_{i=1}^{N} v_i \, w^{(i)}
$$

(using your normalized scores $v_i$ as weights).

**Update $\alpha$:** choose a concentration parameter $S$ (this controls how peaked your distribution is):

$$
\alpha_\text{new} = \mu^* \cdot S
$$

---

### 3. Managing the exploration–exploitation trade-off

The value of $S$ (the sum of all $\alpha_k$) is your "inverse exploration" dial:

- **Small $S$** (e.g. $S \approx 3$): high variance, spreads samples across the simplex → **exploration**.
- **Large $S$** (e.g. $S > 20$): low variance, samples stay very close to the mean $\mu^*$ → **exploitation**.

**The strategy:** start with $\alpha = [1, 1, 1]$, i.e. $S = 3$. After your 500-episode check-in, calculate $\mu^*$ from your best performers and set:

$$
\alpha_\text{next} = \mu^* \cdot (S + \Delta S)
$$

Incrementally increasing $S$ over time (e.g. adding 2 to $S$ every update) will naturally cause the agent to narrow its focus onto the most promising weight combinations while still occasionally testing the fringes.

---

### 4. Implementation example (the "elite" method)

If you want something robust and easy to code, use a CMA-ES-style update:

1. Sort your 25 weight samples by reward.
2. Take the top 5.
3. Calculate their average: $w_\text{avg} = \frac{1}{5} \sum_{i \in \text{top5}} w^{(i)}$.
4. Update: $\alpha = w_\text{avg} \cdot S$.
5. **Keep a floor:** ensure no $\alpha_k$ drops below a minimum value (e.g. 0.1) to prevent the distribution from collapsing and completely ignoring one of your reward objectives.

---

### Summary table for $\alpha$ tuning

| Parameter state | Effect on sampling | RL phase |
|---|---|---|
| $\alpha = [1, 1, 1]$ | Uniformly random over the simplex | Initial discovery |
| $\alpha = [10, 10, 10]$ | Heavily biased toward $[0.33, 0.33, 0.33]$ | Balanced preference |
| $\alpha = [15, 2, 3]$ | Focuses heavily on the first reward component | Exploitation |
| $\alpha = [0.1, 0.1, 0.1]$ | Samples mostly at the "corners" (single objectives) | Extreme boundary search |

### Sample Dirichlet sampling:

Extend it with a rejection sampling, so weight and thus reward won't be zeroed out.

```python
from scipy.stats import dirichlet

alphas = [1.0, 1.0, 1.0, 1.0]

# Create a frozen distribution object
dist = dirichlet(alphas)

# Sample observations
samples = dist.rvs(20)

print("SciPy Samples:\n")
for s in samples:
	print(s)
```

```python
from scipy.stats import dirichlet
import numpy as np

alphas = [1.0, 1.0, 1.0]

# Create a frozen distribution object
dist = dirichlet(alphas)

# Sample 3 observations
samples = dist.rvs(200)

print("SciPy Samples:\n", samples[:10])
print("AVG: ", np.mean(samples, axis=0))
```