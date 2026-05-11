# Theoretical and Algorithmic Frameworks for Multi-Objective Reinforcement Learning: Implementation Strategies within the Ray RLlib New API Stack

The paradigm shift from single-objective optimization to multi-objective reinforcement learning (MORL) represents one of the most significant theoretical expansions in the field of autonomous decision-making. Standard reinforcement learning (RL) assumes that a task can be encapsulated within a single scalar reward function, guiding the agent toward a unique optimal policy. However, real-world systems — ranging from autonomous vehicle control to financial portfolio management and industrial supply chain optimization — are inherently characterized by conflicting goals such as speed versus safety, return versus risk, or efficiency versus environmental impact. In these contexts, the search for a single "best" solution is replaced by the identification of a Pareto frontier: a set of non-dominated policies where any improvement in one objective necessitates a degradation in another. The recent architectural overhaul of the Ray RLlib library, specifically the transition to the new API stack, provides a robust, distributed framework for addressing the computational and representational challenges inherent in MORL. By separating neural network logic (`RLModule`s), gradient optimization (`Learner`s), and data transformation (`Connector`s), RLlib enables the development of sophisticated algorithms capable of maintaining policy swarms and navigating complex Pareto manifolds.

## Formalization of the Multi-Objective Markov Decision Process

The mathematical foundation for MORL is the Multi-Objective Markov Decision Process (MOMDP), which generalizes the traditional MDP by extending the reward function from a scalar to a vector space. An MOMDP is defined by the tuple $(\mathcal{S}, \mathcal{A}, P, \mathbf{r}, \gamma)$, where $\mathcal{S}$ and $\mathcal{A}$ represent the state and action spaces, $P$ denotes the transition probability distribution, and $\gamma$ is the discount factor. Crucially, the reward function $\mathbf{r}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to \mathbb{R}^m$ yields an $m$-dimensional reward vector, where each component $r_i$ corresponds to a distinct objective.

In this framework, the agent's performance is measured by the vector-valued return

$$\mathbf{J}(\pi) = \mathbb{E}_{\pi,P}\!\left[\sum_{t=0}^{\infty} \gamma^t\,\mathbf{r}_t\right] \in \mathbb{R}^m.$$

Because the return is vector-valued, the standard total ordering of real numbers no longer applies, requiring the introduction of Pareto dominance. A policy $\pi_a$ is said to strongly dominate $\pi_b$ ($\pi_a \succ \pi_b$) if and only if $J_i(\pi_a) \ge J_i(\pi_b)$ for all $i \in \{1, \dots, m\}$ and there exists at least one $j$ such that $J_j(\pi_a) > J_j(\pi_b)$. The Pareto set $\Pi^*$ consists of all policies that are not strictly dominated by any other policy in the feasible space, and its image in the objective space constitutes the Pareto frontier.

Finding these policies requires navigating the preference simplex $\Delta_{m-1}$, where a weight vector $w \in \mathbb{R}^m$ defines the relative importance of each objective. Traditional approaches often rely on scalarization, transforming the vector return into a scalar utility $U(\mathbf{J}(\pi), w)$. While linear scalarization ($w^\top \mathbf{r}$) is the most common, it is fundamentally limited by its inability to recover solutions in concave regions of the Pareto frontier. This limitation motivates the use of non-linear scalarizations, such as the Tchebycheff metric, which can identify any Pareto-optimal point regardless of frontier convexity.

## Architectural Paradigm of the RLlib New API Stack

The evolution of RLlib's new API stack is a response to the need for greater modularity and scalability in distributed RL workloads. The redesign reduces over a dozen critical legacy classes to a handful of core abstractions, strictly applying principles of separation of concerns and fine-grained modularity. This structure is particularly advantageous for MORL, where custom loss functions, multi-head architectures, and complex data pipelines are often required.

### Core Components and Responsibilities

The new stack is built around four primary pillars: `RLModule`, `Learner`, `ConnectorV2`, and `Episode`. The interplay between these components defines the modern RLlib training loop.

| Component | Responsibility in New API Stack | Significance for MORL |
|-----------|---------------------------------|-----------------------|
| `RLModule` | Replaces `ModelV2` and `PolicyMap`. Encapsulates NN architecture and forward logic for exploration, inference, and training. | Allows for multi-head value functions and preference-conditioned networks. |
| `Learner` | Replaces `RolloutWorker` (training aspects) and `Policy`. Manages gradient-based updates and optimizers. | Enables "gradient surgery" (PCGrad/MGDA) to handle conflicting objective gradients. |
| `ConnectorV2` | Replaces legacy `Connector`. Manages data transformation between env and module, and module and learner. | Optimal location for reward vectorization, preference injection, and advantage calculation. |
| `Episode` | `SingleAgentEpisode` and `MultiAgentEpisode` replace `SampleBatch`. Standardized trajectory storage. | Supports vector-valued rewards and multi-agent state tracking in policy swarms. |

The `RLModule` is the central model abstraction, exposing three distinct forward methods: `_forward_exploration` for data collection, `_forward_inference` for deployment, and `_forward_train` for loss computation. For MORL, this allows a single module to use a preference-conditioned head for inference while utilizing a multi-head value network for training. The `Learner` class, operating in a data-parallel fashion across multiple GPU workers, provides a unified interface for model updates. In a multi-objective context, the `Learner` can maintain multiple optimizers or adjust learning rates dynamically for different objective heads.

### The Role of ConnectorV2 in Multi-Objective Pipelines

Connectors in the new stack are organized into three distinct pipelines: env-to-module, module-to-env, and learner connectors. The learner connector pipeline is particularly crucial for MORL as it transforms a list of `Episode` objects into a tensor batch for the `forward_train` method. This is the natural architectural hook for reward shaping and preference integration. By subclassing `ConnectorV2`, researchers can implement logic that modifies the rewards column of a batch based on a preference vector $w$ sampled at runtime. This approach ensures that the "on-policy" nature of algorithms like PPO is maintained even when the preference weights are changing across iterations.

## Algorithmic Strategies for Finding Pareto Optimal Policies

Finding Pareto optimal policies involves solving the trade-off between exploring the preference space and optimizing performance within each sampled preference. Two primary methodologies have emerged: preference-conditioned single-policy models and multi-policy swarms.

### Preference-Conditioned Models and Hypernetworks

A preference-conditioned policy $\pi_\theta(a \mid s, w)$ accepts both the state $s$ and a preference weight $w$ as input. This allows a single network to represent the entire Pareto frontier. However, these models are prone to two major failure modes: destructive gradient interference and representational mode collapse. Gradient interference occurs when the gradients from conflicting objectives cancel each other out, stalling the learning process. Mode collapse occurs when the network ignores the preference input and converges to a single behavior regardless of the requested trade-off.

To address these issues, the Pareto Set Learning (PSL-MORL) framework utilizes hypernetworks to generate policy parameters for each decomposition weight. A hypernetwork $H_\phi(w)$ maps the preference $w$ to the parameters $\theta$ of a target policy $\pi_\theta$. This allows the model to produce a continuum of personalized policy networks. In RLlib, this is implemented by creating a custom `RLModule` where the `setup` method initializes the hypernetwork, and the forward passes dynamically compute $\theta$ based on $w$ before executing the policy.

### Gradient Surgery: PCGrad and MGDA++

When multiple objectives are optimized simultaneously, their gradients may point in opposing directions. Gradient surgery techniques attempt to resolve these conflicts before updating the model weights. Projecting Conflicting Gradients (PCGrad) is a widely used method that identifies conflicting gradients (where the cosine similarity is negative) and projects each onto the normal plane of the other. This ensures that an update intended to improve one objective does not catastrophically interfere with another.

An alternative approach is the Multiple Gradient Descent Algorithm (MGDA). MGDA seeks a common descent direction that is optimal for all objectives simultaneously by solving a minimum-norm problem in the convex hull of the objective gradients. While standard MGDA can be subject to "weak" Pareto convergence, MGDA++ provides theoretical guarantees for convergence to strong Pareto optimal solutions in convex bi-objective problems. Implementing these in RLlib requires overriding the `compute_gradients` method in the `Learner` class, allowing the user to manipulate the gradient tensors after loss computation but before the optimizer step.

### Decomposed, Diversity-Driven Policy Optimization (D3PO)

D3PO is a preference-conditioned framework that reorganizes the optimization pipeline to preserve per-objective learning signals. By introducing a multi-head critic that estimates values for all dimensions simultaneously and a dimension-wise surrogate objective, D3PO avoids premature scalarization. Furthermore, it employs a scaled diversity regularizer to enforce sensitivity to preference changes, preventing the representational collapse common in earlier single-policy methods.

| Feature | Standard PPO | MOPPO / D3PO |
|---------|--------------|--------------|
| Objective | Scalar $r$ | Vector $\mathbf{r} \in \mathbb{R}^m$ |
| Advantage | Scalar $\hat{A}$ | Vectorized GAE scalarized via $w$ |
| Surrogate Loss | Single clipped term | Dimension-wise clipping + late combining |
| Regularization | Entropy only | Entropy + Diversity Regularizer |
| Model Head | Single policy/value | Multi-head critic + Conditioned policy |

## Maintaining and Managing a Swarm of Policies

For complex or non-convex Pareto frontiers, maintaining a "swarm" of independent policies is often more reliable than using a single preference-conditioned model. RLlib provides several mechanisms for managing such populations.

### Multi-Agent Abstractions for Policy Populations

In RLlib, a "swarm" can be modeled using the multi-agent execution model. Each agent in a `MultiAgentEnv` can be assigned a different policy, even if they all operate within the same underlying environment dynamics. This allows for Level 2 multi-agent training: multiple agents and multiple policies. The `policy_mapping_fn` is the key tool here, as it can dynamically bind agents to specific policies representing different trade-offs. This decomposition not only allows for more scalable learning but also effectively increases the amount of training data generated per environment step.

### Population-Based Training (PBT) and Pareto Tracking

Population-Based Training (PBT) is a powerful technique for evolving policy populations. In a traditional PBT setup, multiple "trials" run in parallel, periodically replacing poorly performing configurations with perturbations of better ones. For MORL, the selection and mutation rules can be adapted to navigate the Pareto frontier. Instead of a single performance metric, the "exploit" step can use Hypervolume or non-dominance as the criteria for selecting which configurations to preserve.

Recent research has introduced "population-free" Pareto front tracking mechanisms, such as MPFT, which eliminates the need to maintain a massive static population. MPFT works in stages: identifying Pareto-vertex policies (the extreme points for each objective) and then tracking the frontier between them by initializing new policies in sparse regions. This can be implemented in RLlib by utilizing the dynamic `add_module` and `remove_module` methods of the `MultiRLModule` at runtime, allowing the population of policies to grow and shrink based on the discovered frontier geometry.

## Implementation of Multi-Objective PPO and SAC

The adaptation of PPO and SAC for MORL requires modifications to the model architecture and the loss calculation logic within the new API stack.

### Multi-Objective PPO (MOPPO) Mechanics

Proximal Policy Optimization (PPO) is the de facto standard for many RL applications due to its stable clipped surrogate objective. In a multi-objective context, the standard GAE (Generalized Advantage Estimation) must be computed for each objective. The vectorized advantage $\mathbf{A}_t$ is then scalarized using the preference $w$ to produce the advantage signal used in the PPO update.

To implement MOPPO in the new RLlib stack:

- **`RLModule`**: Subclass `DefaultPPOTorchRLModule` to include multiple value function heads. Implement the `ValueFunctionAPI` by overriding `compute_values` to return values for all $m$ objectives.
- **`Learner`**: Override `compute_loss_for_module`. Calculate the loss for each objective head. If using D3PO, apply the PPO clipping individually to each objective's surrogate before weighting them by $w$.
- **`ConnectorV2`**: Write a custom learner connector to handle the calculation of vectorized GAE. The $\lambda$ parameter in GAE balances short-term, low-variance estimates against long-term, high-variance returns, and may need to be tuned per objective.

### Soft Actor-Critic (SAC) for Multi-Objective Optimization

SAC is an off-policy, actor-critic algorithm based on the maximum entropy reinforcement learning framework. For MORL, SAC typically employs a multi-head Q-network or a preference-conditioned Q-network. The Q-function estimates the expected vector-valued return for each action, and the actor is updated to maximize the scalarized Q-value while maintaining high policy entropy.

A key advantage of SAC in MORL is its compatibility with distributional RL. Multi-Dimensional Distributional DQN (MD3QN) and similar SAC variants can capture not only the joint return distribution from multiple reward sources but also the correlations between objective randomness. In the new API stack, this involves implementing the `QNetAPI` within the `RLModule`, which requires overriding `compute_q_values` to return vector-valued tensors.

## Selecting Pareto Optimal Policies A Posteriori

Once a set of policies has been trained, the problem of selecting a single policy for deployment according to user-defined preferences remains. This is known as the policy selection problem.

### Scalarization Functions for Selection

Selecting a policy from the Pareto frontier is non-trivial, as the front may contain a large number of solutions. Several scalarization approaches are commonly used:

- **Linear Scalarization**: Simple but limited to convex regions of the front.
- **Tchebycheff Scalarization**: Minimizes the weighted distance to an ideal reference point, capable of finding solutions in concave regions.
- **Satisficing Trade-Off Method (STOM)**: Compares the current solution to a desired reference point representing the aspiration level of the decision-maker.
- **Local-Utopia Distance (LUSA)**: A versatile approach that selects a policy by exploiting a novel scalarization function and heuristics to ensure accuracy and even distribution across the frontier.

### Implementing Selection in RLlib Callbacks

RLlib's callback system is the ideal location for implementing selection and evaluation logic. `RLlibCallback` methods can be injected at various phases of training and evaluation. Specifically, `on_evaluate_end` can be used to gather metrics from all parallel `EnvRunner`s and calculate the Hypervolume of the current population.

| Callback Event | MORL Application |
|----------------|------------------|
| `on_algorithm_init` | Initialize Pareto archives or hypernetwork parameters. |
| `on_episode_end` | Extract vector-valued rewards from the `SingleAgentEpisode` for frontier tracking. |
| `on_train_result` | Perform non-dominated sorting and update the league of Pareto-optimal policies. |
| `on_evaluate_end` | Select the "best" policy according to a specific scalarization (e.g., LUSA) for final reporting. |

The `MetricsLogger` API, available inside these callbacks, allows for the creation of custom metrics that are reduced across parallel components. This is essential for calculating global metrics like Hypervolume or sparsity across a distributed population of agents.

## Practical Considerations and Case Studies

The implementation of MORL in production environments often faces challenges such as credit assignment, non-stationarity in swarms, and computational overhead.

### The Credit Assignment and Non-Stationarity Problems

In multi-agent policy swarms, credit assignment is particularly difficult. When a global task succeeds, it is non-trivial to determine which specific agent's decisions (or which objective's weight) caused the outcome. Naive approaches that assign the same reward to all agents dilute the learning signal. Furthermore, multi-agent systems are inherently non-stationary; as one policy learns, it changes the environment for all other policies in the swarm. This can lead to unstable training runs where policies fail to converge to the Pareto frontier.

### Domain Application: Supply Chain Optimization

In supply chain environments, agents must optimize purchase orders, product selection, and supplier choice while balancing stock levels, profit, and customer demand. A common approach is to use a multi-agent setup where each agent learns a policy for a specific product line. By using a shared policy with a multi-head architecture, agents can share knowledge across products while still optimizing for their individual (and collective) multi-objective rewards. PPO is frequently the algorithm of choice here due to its robustness, although it requires careful tuning of the observation and action spaces to ensure convergence in such high-dimensional scenarios.

### Domain Application: Network Security

Network defense games involve balancing confidentiality, integrity, and availability. Using Pareto optimization allows network administrators to choose between different defense strategies depending on the current threat level. The Pareto-Q-learning approach can be used to find optimal solutions, while scalarization methods like STOM help administrators select the most appropriate defense strategy during active attacks. RLlib's ability to handle custom environment logic via the `gymnasium.Env` subclassing makes it a natural fit for these complex security simulations.

## Conclusion: Future of MORL in RLlib

The transition to the new API stack has made RLlib a premier platform for multi-objective reinforcement learning research and deployment. The clean separation between `RLModule`, `Learner`, and `ConnectorV2` allows for the seamless integration of advanced MORL techniques such as hypernetworks, gradient surgery, and preference-conditioned models. While challenges such as non-stationarity and the computational cost of maintaining large swarms persist, the scalability of the Ray ecosystem ensures that even thousands of policies can be trained in parallel across massive clusters. Future improvements, including better support for multi-agent vectorization and model-parallel distribution for large-scale architectures, will likely continue to push the boundaries of what is possible in multi-criteria decision-making. By leveraging these tools, practitioners can move beyond scalar compromises toward a nuanced, Pareto-optimal understanding of complex real-world systems.

## Works Cited

1. Pareto Set Learning for Multi-Objective Reinforcement Learning — arXiv, https://arxiv.org/html/2501.06773v2
2. Pareto-Optimal Multi-Objective RL — Emergent Mind, https://www.emergentmind.com/topics/pareto-optimal-multi-objective-reinforcement-learning
3. PPO-Based Multi-Objective Reinforcement Learning — Emergent Mind, https://www.emergentmind.com/topics/proximal-policy-optimization-based-multi-objective-reinforcement-learning-framework
4. Local-utopia Policy Selection for Multi-objective Reinforcement Learning — Intelligent Autonomous Systems, https://www.ias.tu-darmstadt.de/uploads/Site/EditPublication/parisi2016local.pdf
5. Multi-objective Reinforcement Learning with Continuous Pareto Frontier Approximation — AAAI, https://cdn.aaai.org/ojs/9617/9617-13-13145-1-2-20201228.pdf
6. Learning the Pareto Front with Hypernetworks — Aviv Navon, https://avivnavon.github.io/ParetoHN/
7. New API stack migration guide — Ray 2.55.1, https://docs.ray.io/en/latest/rllib/new-api-stack-migration-guide.html
8. RLlib: Industry-Grade, Scalable Reinforcement Learning — Ray Docs, https://docs.ray.io/en/latest/rllib/index.html
9. Key concepts — Ray 2.55.1 — Ray Docs, https://docs.ray.io/en/latest/rllib/key-concepts.html
10. Preference Conditioned Multi-Objective Reinforcement Learning: Decomposed, Diversity-Driven Policy Optimization — arXiv, https://arxiv.org/html/2602.07764v1
11. "Pareto" in layman's terms? — r/reinforcementlearning (Reddit), https://www.reddit.com/r/reinforcementlearning/comments/1ek0rsq/pareto_in_laymans_terms/
12. Pareto Optimal Solutions for Network Defense Strategy Selection Simulator in Multi-Objective Reinforcement Learning — MDPI, https://www.mdpi.com/2076-3417/8/1/136
13. RL Modules — Ray 2.55.1, https://docs.ray.io/en/latest/rllib/rl-modules.html
14. `ray.rllib.core.rl_module.rl_module.RLModule` — Ray Docs, https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.core.rl_module.rl_module.RLModule.html
15. Learner (Alpha) — Ray 2.55.1, https://docs.ray.io/en/latest/rllib/rllib-learner.html
16. Algorithm Configuration API — RLlib — Ray Docs, https://docs.ray.io/en/latest/rllib/package_ref/algorithm-config.html
17. Examples — Ray 2.55.1 — Ray Docs, https://docs.ray.io/en/latest/rllib/rllib-examples.html
18. `ray/rllib/examples/connectors/prev_actions_prev_rewards.py` — GitHub, https://github.com/ray-project/ray/blob/master/rllib/examples/connectors/prev_actions_prev_rewards.py
19. ConnectorV2 API — Ray 2.55.0, https://docs.ray.io/en/latest/rllib/package_ref/connector-v2.html
20. Learner connector pipelines — RLlib — Ray Docs, https://docs.ray.io/en/latest/rllib/learner-connector.html
21. Preference Conditioned Multi-Objective Reinforcement Learning: Decomposed, Diversity-Driven Policy Optimization — OpenReview, https://openreview.net/forum?id=iH7mSOTR4q
22. [2602.07764] Preference Conditioned Multi-Objective Reinforcement Learning: Decomposed, Diversity-Driven Policy Optimization — arXiv, https://arxiv.org/abs/2602.07764
23. `ray.rllib.core.rl_module.multi_rl_module.MultiRLModule`, https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.core.rl_module.multi_rl_module.MultiRLModule.html
24. WeiChengTseng/Pytorch-PCGrad — Pytorch reimplementation for "Gradient Surgery for Multi-Task Learning" — GitHub, https://github.com/WeiChengTseng/Pytorch-PCGrad
25. [2410.19372] Toward Finding Strong Pareto Optimal Policies in Multi-Agent Reinforcement Learning — arXiv, https://arxiv.org/abs/2410.19372
26. Toward Finding Strong Pareto Optimal Policies in Multi-Agent Reinforcement Learning, https://arxiv.org/html/2410.19372v1
27. `ray.rllib.core.learner.learner` — Ray 2.55.0 — Ray Docs, https://docs.ray.io/en/latest/_modules/ray/rllib/core/learner/learner.html
28. How could I implement gradient accumulation? — RLlib — Ray, https://discuss.ray.io/t/how-could-i-implement-gradient-accumulation/22677
29. RLlib for Deep Hierarchical Multiagent Reinforcement Learning — Jonathan Mugan (Medium), https://medium.com/@jmugan/rllib-for-deep-hierarchical-multiagent-reinforcement-learning-6aa96cdee154
30. An Open Source Tool for Scaling Multi-Agent Reinforcement Learning — RISE Lab, https://rise.cs.berkeley.edu/blog/scaling-multi-agent-rl-with-rllib/
31. Multi-Agent Environments — Ray 2.55.1, https://docs.ray.io/en/latest/rllib/multi-agent-envs.html
32. Scaling Multi-Agent Reinforcement Learning — Berkeley AI Research Lab, https://bair.berkeley.edu/blog/2018/12/12/rllib/
33. Visualizing Population Based Training (PBT) Hyperparameter Optimization — Ray Docs, https://docs.ray.io/en/latest/tune/examples/pbt_visualization/pbt_visualization.html
34. A Guide to Population Based Training with Tune — Ray Docs, https://docs.ray.io/en/latest/tune/examples/pbt_guide.html
35. A Multi-Agent Reinforcement Learning-Evolutionary and Bayesian Optimization: An example — Medium, https://medium.com/@abatrek059/a-multi-agent-reinforcement-learning-evolutionary-and-bayesian-optimization-an-example-6e354fc4ab08
36. Hypervolume Optimization in Multi-Objective RL — Emergent Mind, https://www.emergentmind.com/topics/hypervolume-optimization-hvo
37. Multi-Policy Pareto Front Tracking Based Multi-Objective Reinforcement Learning — OpenReview, https://openreview.net/forum?id=K3E05Agd6W
38. RLModule APIs — Ray 2.55.1, https://docs.ray.io/en/latest/rllib/package_ref/rl_modules.html
39. Algorithms — Ray 2.55.1 — Ray Docs, https://docs.ray.io/en/latest/rllib/rllib-algorithms.html
40. `ray/rllib/examples/offline_rl/train_w_bc_finetune_w_ppo.py` — GitHub, https://github.com/ray-project/ray/blob/master/rllib/examples/offline_rl/train_w_bc_finetune_w_ppo.py
41. [2110.13578] Distributional Reinforcement Learning for Multi-Dimensional Reward Functions — arXiv, https://arxiv.org/abs/2110.13578
42. Callbacks — Ray 2.55.1 — Ray Docs, https://docs.ray.io/en/latest/rllib/rllib-callback.html
43. `ray.rllib.callbacks.callbacks` — Ray 2.55.0 — Ray Docs, https://docs.ray.io/en/latest/_modules/ray/rllib/callbacks/callbacks.html
44. `ray/rllib/examples/metrics/custom_metrics_in_env_runners.py` — GitHub, https://github.com/ray-project/ray/blob/master/rllib/examples/metrics/custom_metrics_in_env_runners.py
45. Your Multi-Agent Swarm Is Not Learning. Here Is the Architecture That Changes That. — Medium, https://theneildave.medium.com/your-multi-agent-swarm-is-not-learning-here-is-the-architecture-that-changes-that-93b422a08b68
46. Multi-agent Supply Chain Optimization with RLlib — r/reinforcementlearning (Reddit), https://www.reddit.com/r/reinforcementlearning/comments/1djltne/multiagent_supply_chain_optimization_with_rllib/
47. `ray/rllib/examples/envs/custom_gym_env.py` — GitHub, https://github.com/ray-project/ray/blob/master/rllib/examples/envs/custom_gym_env.py
48. Environments — Ray 2.55.1 — Ray Docs, https://docs.ray.io/en/latest/rllib/rllib-env.html
49. RLlib scaling guide — Ray 2.55.1, https://docs.ray.io/en/latest/rllib/scaling-guide.html
50. Ray Use Cases — Ray 2.55.1 — Ray Docs, https://docs.ray.io/en/latest/ray-overview/use-cases.html
