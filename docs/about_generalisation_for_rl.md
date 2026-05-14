

# Generalisation for RL

General google search: 
- How well can the algorithm perform on a different configuration, but same task
- Ability to perform
- performance on unseen context?
- performance between training and varied environments


## Paper: On the Power of Pre-training for Generalization in RL: Provable Benefits and Hardness

Link to paper: https://proceedings.mlr.press/v202/ye23a/ye23a.pdf or https://proceedings.mlr.press/v202/ye23a.html

- "one possible way to formulate generalization is to allow further interaction with the target environment during the test stage"

--> Generalisation is different than in simple supervised learning.

More fine tuning on the final env means more accurate approx of the optimal policy.
Eventually, the original env (pre-training) fades away, if the fine-tuning lasts long enough, if the algorithm has enough interaction with the new environment.

## Blog:

Link to blog: https://robertkirk.github.io/2022/01/17/generalisation-in-reinforcement-learning-survey.html#:~:text=Reinforcement%20Learning%20(RL)%20could%20be,Free%20Lunch%20theorem%20may%20apply.

"Reality is varied, non-stationarity and open-ended, and to handle this algorithms need to be robust to variation in their environments, and be able to transfer and adapt to unseen (but similar) environments during their deployment."

## Conclusions, thoughts

- Different environment (config change, dynamic change, reward calculation [but not objective!] change) --> generalisation
- Different task (different objective -- means different reward, but same env.) --> transfer learning

Generalisation: env. where only HP is meaningfully present or env. where the dynamic is different (different configuration means different dynamics as well, since the states differ -- e.g. bigger battery capacity means different battery states, which implies different battery actions)

Transfer learning: env. stays the same, but tasks (goals and rewards) differ. 
TL can make sense in case of: battery goal -> EV goal
