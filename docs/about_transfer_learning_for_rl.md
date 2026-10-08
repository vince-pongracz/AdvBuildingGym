# Transfer learning for RL

Cherry picking from papers

Definitions:

## Paper: Transfer Learning for Reinforcement Learning Domains: A Survey

Paper link: https://www.jmlr.org/papers/volume10/taylor09a/taylor09a.pdf

Paper reference: Taylor, M. E., & Stone, P. (2009). "Transfer learning for reinforcement learning domains: A survey." Journal of Machine Learning Research

- "The core idea of transfer is that experience gained in learning to perform one task can help improve learning performance in a related, but different, task"

--> In my case, the task is the same (and the spaces are the same as well)


- "The insight behind transfer learning (TL) is that generalization may occur not only within tasks,
but also across tasks"

Based on the survey paper, there should be a target and a source task -- however tasks are the same here, only the circumstances differ.

Transfer learning in my case can be if we gradually train the agent. 
Gradual training means here that in the early runs, only simpler goals should be achieved (a single infrastructure element, with a single reward). 
This would look like that the agent first learns how to control a heatpump based on a single reward, but the other infrastructure elements are there as well (so the environment [state and action space] is fixed). 
Then the agent learns to control heatpump and EV charger together.

Another approach can be, that an agent first trains how to control/manage the battery, then it learns to control the EV charge -- as these two infrastructures are modelled similarly.
Or: It is easier to learn to handle such an EV charger and EV spec, which can not do V2G, which can from the beginning. 
Gradual improvement possibility: learn to use non-V2G first, then learn how to use V2G -- difference in task.

It's worth thinking about whether the single or multi reward setups can be gradual as well, like in the 1st runs agent learns on a single reward, then it learns on multiple rewards.

It's a nice extension idea as well, how is it worth combining these 2 gradual training -- and achieve a transfer learning.

### TL metrics, which sounds useful for my case:

- "Total Reward: The total reward accumulated by an agent (i.e., the area under the learning curve) may be improved if it uses transfer, compared to learning without transfer."

Useful metrics (section is same as the quoting):
- Total Reward
- Transfer Ratio
- Time to Threshold

Dimensions of comparison:
- "Task difference assumptions", page 8 of the link

Asymptotic performance would be the interesting thing for us (Figure 3. in the paper)


### TL counterproductivity:

Link, source: https://milvus.io/ai-quick-reference/how-does-transfer-learning-work-in-rl

TL can be counterproductive as well -- in the case of unrelated tasks (no connection between source and target tasks). 
Task similarity --> greater chance of success



### Conclusions drawn

- For transfer learning in RL (TL in RL), source and target tasks needed -- implies different tasks to achieve by the agent(s).
- TL can be built in the process, selecting some options (plans) for transfer and comparing them
- Checkpointing for TL, a scenario for TL in RL:
    - R1 algo learns on the full setup.
    - R2 algo learns on a restricted setup.
    - both have the same amount of iterations to learn, which is predefined.
    - R1 continues to learn
    - R2 learns on the full setup (not on the restricted anymore)
    - both algos learning the same amount of iters (predefined)
    - Eval: run both on the same 10 episodes (on the same full setup), check achieved reward or other metric -- which one is better

### If TL in RL needed / desired:

- TODO VP: design TL mode of the environment, when some rewards, states and actions are part of the environment, thus the config is the same, however they do not act / their provided information is null/neutral to the system. 
In this case, reread the linked paper.

