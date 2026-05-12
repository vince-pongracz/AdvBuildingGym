<p float="left">
    <img src="data/img/icon_kit.png" width="10%" hspace="20"/>
</p>

[![Python](https://img.shields.io/badge/Python-3.12.1-blue?logo=python)](https://www.python.org/downloads/release/python-3121/)
[![License](https://img.shields.io/badge/License-MIT-green?logo=opensource)](./LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-black-000000.svg?logo=python)](https://github.com/psf/black)


<h1 align="center">Deep Reinforcement Learning for Smart Energy Management in Residential Buildings -- Framework</h1>

### Goals

Main objectives: Energy efficiency (kWh), costs (EUR)
Subobjectives: temp, battery SOC, EV SOC, grid limits

Outline:
- base: deterministic controllers, no optimisation for cost, for energy efficiency, no joint optimisation -- optimising only a single var, check on Energy efficiency and costs.
 - Energy efficiency and cost optimisation is not feasible, as we do not know which cost trajectory to follow (okay, system could be forced towards zero, however it's not optimal for sure and it's not feasible) can't really feasible as these controllers are for trajectory tracking, not really for optimisation.
 - a couple plots about control results, how well are trajectory or range tracking
- case1: deterministic controllers on the different actuators, working together, no joint optimisation
- case1.1: how to jointly optimise with deterministic controllers?
  - cascade controllers ???
  - 
- case2: RL-based controllers for the standalone actuators -- compare performance with deterministic controllers
- case3: RL-based controllers, SA, joint optimisation --> not optimal at all
- case4: Directions of improvement -- SA / MA:
 - SA:
  - non stationary reward weighting, weights change per episode -- reward weights are part of the observation space
   - sample weights based on some distribution
   - try pre-defined weight combinations
  - curriculum learning:
   - TL: gradual -- 1st temp control, then temp and battery control -- using reward schedules
   - TL: 5 rewards alltogether, always select 2 or 3 and change 1 between iterations -- using reward scheduling
 - MA:
  - vector rewards, standalone Q network for each reward, Pareto front search
  - each actuator is an agent: scalar rewards, curriculum learning: actuator items are selected or deselected per episode -- rewards always there, but action from infrastructure is not always present -- each actuator has its own policy NN.
- What about generalisation? How will any of these methods work on an unseen infrastructure?

- Real time eval, Monte Carlo simulations of ANY above mentioned solution -- eval scripts
- Question of reward formulation: everything depends on that...

NOTE VP: is there such a scenario, where during training env is allowed not to terminate, but in the eval env it must terminate?
--> yes, during eval termination is not allowed at all, so it's maybe worth switching off the termination during eval. 
Maybe termination is not a good idea at all... -- however it reduces wrong states in the trajectory buffer.


- Try to eliminate most of the bad states... -- we want to optimise

TODO VP: setting the gamma (discount factor) to 1.0 -- all reward from all future timesteps would have the same effect as only the next timestep... So it would not matter when the reward is received -- does this help?

TODO VP: Multi phase rewards -- reward can't decrease between phases... how to ensure this one?
Agents can't learn the phase change only on their own -- reward signal must be maintained, so if a goal is reached and the objective is shifting, the reward must keep up so the agent can still believe that it's on a good track and concentrate on the next goal, on the next objective... -- this one is important for curriculum learning and switching rewards on the fly
In the case of multi phase RL (non stationary rewards): encode the phase into the observation space -- so agent can observe it
Terminate in case of goal reached -- for single objective tasks that works, but how can I apply this to my task?

Difficulty: how to avoid reward tuning/reward engineering but make it work?

CES/CEM -- TODO VP read about them, how to apply them in training?
CEM for reward weight sampling distributions? -- but eval of the params/reward weights should happen on the same s,a pairs... -- a lot of computational effort.

Idea: 2 stage learning -- CEM: 
- 1st: Base Trajectories using human expert trajectories -- joint human optimisation --> learns reward function, but needs trajectory "drawer" for easy catch of expert trajectories
- 2nd: Run the RL with this reward functions -- try to find even more optimal solutions? If better trajectories found, add them to the expert trajectories
- iteration: find new reward functions based on the new expert trajectories


My problem -- energy management: Control problem, almost infinite horizon

- Framework for Residual building energy management, flexible environment: flexibility about rewards, infrastructure elements (actuators -- their physics) and state sources.

- Generalisation: train a single PPO/SAC on a set of slightly different building configs, evaluate how it performs generally on unseen, but similar buildings.
 - context aware observation space, context aware policies
 - During eval: unseen building, unseen time series data
 - threats: 
  - not enough building configurations seen during training
  - problem is too complex for policy NN
  - 

- Transfer learning (TL): curriculum learning, change the reward setup -- the goals -- of the optimisation slowly under the RL algorithm and inspect the performance of this approach against the straight away difficult scenario, where all the rewards are present.
 - Random selection: select always a subset of rewards, replace a subsubset of those, but always maintain a fix ratio, which does not change between 2 consecutive selections.

- Compare energy usage, energy efficiency and temperature comfort of rule based controllers (Fuzzy, PI, PID) to the trained TL solution.
  - comparison should happen on unseen and seen building configurations (building data hasn't seen, building data already seen -- variable during eval are the price, weather and other user behaviour)

- For each trained RL agent, RL policy -- Monte Carlo rollouts, this should be quite the same as the eval script.

- Multiple agents, cooperation, Mixture of Experts
 - env, shared state space -- multiple policies running on the same env, each of them with a different (partially overlapping, non-overlapping) set of rewards -- weighting of policies and their actions: can be user preference during eval -- but during training as well
 - extension opportunity: network to decide about the weighting.

- Extension idea: use expert action trajectories to train policies -- at the SAC it's straightforward (push episodes into the replay buffer, they will be sampled), but at the PPO it shouldn't be too complicated as well

- MORL, achieve Pareto front: reward functions have dynamic weighthing, which is always sampled and added to the state space. During training, the rewards are weighted with these weights. During eval, they are random from the same distribution. During user eval, user gives these weights and eval is based on this -- policy weighting is part of the state space.

### Issues

- Dynamic env assumed -- during development the env changes, the rewards change, their weights change

## References, data sources

### Data about residental homes and their heat pump energy need (WPuQ dataset)

Paper: Dataset on electrical single-family house and heat pump load profiles in Germany

Paper link: https://www.nature.com/articles/s41597-022-01156-1

Data link: https://zenodo.org/records/5642902



About reward shaping: https://link.springer.com/rwe/10.1007/978-0-387-30164-8_731

- Potential based shaping --> add per state key potential rewards -- good for target following rewards, not so good for range controller states.
- Realisation idea: automatic wrapper on the reward functions, which has the same api as reward functors, but store the previous value and acts as a proxy on reward functors -- substracts potential based reward based on prev state and adds potential based reward for the current state. (TODO VP: potential based reward shaping). Uses a standalone config yaml.

An old paper about reward shaping and construction: https://link.springer.com/article/10.1023/A:1018068507504


<!-- TODO VP: add it to the repo setup description... -->

TODO VP: Show expert trajectories to the policies, which work fine -- Programming using expert knowledge -- difficulty -- multi dim trajectories, hard to really give expert trajectories.

TODO VP: Idea 2. The "Mixture of Experts" or Hierarchical Approach
You can have a single agent that switches between different policies based on the state.

Define a multi-agent setup where one "Manager" policy selects which "Worker" policy to use. Even though it's technically a single entity in the game, RLlib treats it as a coordination task between multiple policies.

Diff between local and district heating networks: https://www.npro.energy/main/en/district-heating-cooling/local-district-heating

District heating grid ^^

### Power measure explanations

Active, reactive, apparent power

Link: https://eshop.se.com/in/blog/post/difference-between-active-power-reactive-power-and-apparent-power.html?srsltid=AfmBOoo_z3uMQTGngU470DqVz29bTNpcSOKL1ch39emWHsMA7PthqQVC

https://en.wikipedia.org/wiki/AC_power

### Data setup

All data fetching and preprocessing is handled by a single entry point:

```bash
python preprocessing/data_setup.py
sbatch slurm_scripts/slurm_data_setup.sh
sbatch slurm_scripts/slurm_data_setup.sh --synthesize --quality-report-dir
```

Runs three pipelines:

1. **Electricity prices** — fetch day-ahead EPEX Spot prices from aWATTar and Energy Charts APIs, converts units (Eur/MWh → ct/kWh), and resample to 5-minute resolution
2. **Zenodo weather** — download WPuQ dataset (residential heat pump load profiles), extract HDF5 archives, and produce per-house weather CSVs.
3. **DWD weather** — download 10-minute station data from the DWD Climate Data Center, merge parameters, and upsample to 5 minutes

Preprocessed files are written to `data/e_price/` and `data/weather/`.

**Common flags:**

```bash
# Run only the price pipeline for specific years
python preprocessing/data_setup.py --skip-weather --years 2024 2025

# Run only the DWD weather pipeline
python preprocessing/data_setup.py --skip-prices --skip-wpuq --dwd-station-id 04177

# Run only the WPuQ/Zenodo weather pipeline
python preprocessing/data_setup.py --skip-prices --skip-dwd

# Generate synthetic (noised + shifted) dataset variants after all pipelines
python preprocessing/data_setup.py --synthesize

# Preprocess existing raw price files without re-fetching
python preprocessing/data_setup.py --skip-weather --skip-price-fetch --raw-price-files data/e_price/2025_prices.csv
```

EV usage profiles (`data/ev_usage_profiles/ev_*.csv`) are manually authored and do not require fetching.

<!-- Add NOTEs:
TODO VP: SAC and PPO notes
SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html
- test alpha param, controlling exploitation, exploration tradeoff

PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

-->

### Slurm

A bit more detailed help here: https://www.nhr.kit.edu/userdocs/haicore/batch/

```bash
sbatch slurm_scripts/slurm_train_ray.sh # start a job
scancel jobID # cancel a job
scontrol show job [jobid] # see job state info
squeue #Displays information about active, eligible, blocked, and/or recently completed jobs
```

<div align="center">
    <img src="data/img/HeatPumpEnvironment.gif" style="width:44%;">
</div>


**⚠️ Note**: _Last update on 28.01.2026_

<div align="left"> 
This repository contains the official code of our paper <strong>"Advanced Deep Reinforcement Learning for Heat Pump Control in Residential Buildings"</strong>.
It features a custom <a href="https://github.com/Farama-Foundation/Gymnasium" target="_blank"><strong>Gymnasium</strong></a> environment for smart heat pump control in residential buildings, inspired by the Heat Pump House at the  
<a href="https://www.iai.kit.edu/english/RPE-LLEC.php" target="_blank"><strong>Living Lab Energy Campus (LLEC)</strong></a>, KIT.
</div>

## 1. Introduction LLECBuildingGym

<details>
  <summary>Click to expand/collapse</summary>

### 1.1 Description

The **[adv_building_gym.py](adv_building_gym/envs/adv_building_gym.py)** simulates thermal building dynamics with heat pump control in 5-minute intervals.
This framework leverages the **[Gymnasium](https://github.com/Farama-Foundation/Gymnasium)** and **[Pyomo](https://github.com/Pyomo/pyomo)** libraries, making it suitable for both reinforcement learning agents and advanced control strategies.

### Papers -- literature research

#### Learning to Optimize Multi-Objective Alignment Through Dynamic Reward Weighting

Link: https://arxiv.org/abs/2509.11452v1

About LLMs and multi objective RL setup, how to align LLMs to multiple goals -- topic: dynamic reward weighting

Dynamic reward weight generator -- based on Dirichlet distribution
lists rewards --> knows how many weights and for which rewards should it schedule. There is already a markdown about this. It's already implemented as a reward schdedule mode.

Hypervolume guided weight adaptation: by default fix weights, but somehow if hypervolume could be enlarged then the weighting is updated. The original, user-set weights are not overwritten, the original reward is multiplied with a meta-reward.

Gradient-based weight optimization: no pre-defined reward weights, but compute learning of how each objective contributes to the overall performance -- based on gradients and reallocates weights based on this. 

Defines Pareto front, Hypervolume indicator

TODO VP: "To the best of our knowledge" -- important phrase to use in thesis work

#### Comprehensive Overview of Reward Engineering and Shaping in Advancing Reinforcement Learning Applications

Link: https://ieeexplore.ieee.org/abstract/document/10763475

Reward engineering:
- R(s, a, s')
- guidance towards desired states and actions
- informative for learning, sparse to prevent trivial solutions
- How to design such rewards, which find the desired behaviour and not the unintended shortcuts.

Reward shaping:
- about fine-tuning the reward function
- improve to learning process without altering the policy -- only about faster learning
- Potential-based reward shaping -- reward based only on s, s' -- R'(s,a,s') = R(s,a,s') + γ R_p(s') − R_p(s)

2 approaches:
- "Reward is Enough" -- single scalar reward meaning progress, environment complexity not considered -- SO
- "Reward is Not Enough" -- can't rely on a single scalar value, vector rewards -- MO

Reward design pitfalls:
- Reward Sparsity
- Deceptive Rewards
- Reward Hacking
- Unintended Consequences
- Misaligned Reward with True Objective
- Reward Function Complexity
- Difficulty in Evaluating Reward Design



#### Reinforcement Learning-Based Energy Management of Smart Home with Rooftop Solar Photovoltaic System, Energy Storage System, and Home Appliances. 
Link: https://www.mdpi.com/1424-8220/19/18/3937
Uses RL, tabular Q learning (tables, discrete state-action pairs), PV, ESS, AC and washing machine.
Cost and comfort optimisation. Seems like each infrastructure has its own agent -- or at least own head in the policy network
Restricted weather data (only temp)

Previous works: almost everything done... multi agent, Q learning, manage HVAC, manage ESS
This paper: 
- ESS + consumer comfort -- but with TOU tariff, not with variable, real day ahead data
- only optimises for energy cost and thermal comfort -- 2 optimisation goals
- shiftable and non shiftable energy consumption -- washing machine can't be stopped at any time, charging can
- shiftable interruptable/non-interruptable, non shiftable interruptable/non-interruptable
- only PV, ESS, HVAC -- no EV and Wind turbine
- only binary (on/off) ESS control?
- schedule of the energy usage is learnt, not the actual energy allocation -- energy allocation is discrete in this paper
- scheduling resolution is 1h, not 5min -- update in each hour, not in every 5 mins.
- for indoor temp: they predict it with an NN -- no physical model, just NN behind actual temp prediction -- T_act in the current step is predicted by an NN -- I have physics here instead
- user desired temp is a range, not an exact value -- in my project it's a fix value with a threshold up and down
- they compare MILP and RL control of the same setup -- RL is better

New stuff can be in my thesis: 
- based on data and actions, forecast the passive states as well -- try to learn the passive states -- model based RL (?)
- resolution is more fine grained, I use wholesale price data
- Flexibility: not only PV, HVAC and ESS -- wind turbine, etc, config and flexibility
- more rewards, more reward aspects, flexible to config how many rewards. Optimise on achieved reward or to reward rate

TODO VP: How to solve that the same model used for different infra/state configs?
--> if it's multi agent, then it's easy -- each agent outputs an action, number of agents change, but not really their state
- What if the state sources config changes as well? -- I guess no need to overcome this

TODO noprio VP: tune discount factor of the Q values -- long term or short term optimisation

TODO VP: take out big oscillations from the battery charge discharge actions -- or at least inspect whether it happens or not -- refactor the ActionSmoothReward that it catches automatically the last 10 actions and computes the FT and detects high frequency oscillations -- if detected, punishes, if not detected, zero reward.
TODO VP: at ESS -- add lifetime decay/degradation in capacity or in discharge rate
TODO VP: use the WPuQ PV production data (actions..?) along with its weather data?

#### Enhanced Robust Index Model for Load Scheduling of a Home Energy Local Network With a Load Shifting Strategy

Link: https://ieeexplore.ieee.org/document/8600304

Paper:
- load scheduling
- robust index model: to opt home energy local network (HELN)
- rather deals with how to optimise so, that in the case of max uncertainty (worst case scenario) the system is still functional and does not violate hard constraints.
- no RL, it's not a really relevant paper


New idea for my thesis:
- predict actions and states for N steps (model based RL) -- MPC and Monte Carlo sims would be something like this
See into the future for statesources where it's possible.

TODO VP 2026.01.20. : Add forecasting window (and thus MPC) for the states and the
actions as well in the config, generally window size is 0.
Allow it only for the forecasted desired states -- not for the actual system states
Handle if no more forecasting is available (csv ended and similar scenarios)
Add this as a Wrapper on the env...
So the wrapper extends the observations with the forecasting data

#### Real building implementation of a deep reinforcement learning controller to enhance energy efficiency and indoor temperature control

Link: https://www.sciencedirect.com/science/article/pii/S0306261924008304?via%3Dihub

SAC on building, thermal and economic goals
Goal was: beat the RBCs -- rule based controllers
2 objectives: maintain temperature and optimise energy consumption
"Resistance-Capacitance (RC) model calibrated with real building data" -- exactly what Gökhan's paper was about.
In the papaer: comparison of RBCs, PI, MPC and DRL controllers



#### State of the Art of Machine Learning Models in Energy Systems, a Systematic Review

Link: https://www.mdpi.com/1996-1073/12/7/1301

Paper:
- Comprehensive review of ML and energy systems, 2019, ANN, but no RL
- single domain systems (only PV, only HP, etc..)
- likely not really relevant, as it is an older survey paper, a SOTA overview from 2019
- does not mention RL --> drop this

#### Optimal Energy System Scheduling Using A Constraint-Aware Reinforcement Learning Algorithm

Link: https://www.sciencedirect.com/science/article/pii/S0142061523002879

GitHub: https://github.com/EnergyQuantResearch/Optimal-Energy-System-Scheduling-Combining-Mixed-Integer-Programming-and-Deep-Reinforcement-Learning

Summary:
- goes with RL and MIP (mixed integer programming) -- MIP-DQP
- model free RL
- constraints are important, they consider them better (...)
- They deal with: "enforcing operational constraints during the online scheduling stage is a critical challenge for DRL algorithms and it must be addressed in order to enable their wide adoption in real system"
 - operational constraints of RL algorithms
- a lot of implementations are not freely accessible...
- strict enforcement of each operational condition in the action space (e.g. power balance constraint), even in
unseen test data
- uses day ahead wholesale prices
- needs full future information (consumption, dynamic prices, weather) -- to keep/ensure all the constraints

Conclusion:
- no generalisation for multiple env setups
- constrainsts: Env and the infra elements enforce them...
- quite similar to my project...
- they do not handle varying infrastructure or other user interventions -- like EV connect/disconnect
- they only deal with energy, only optimise for energetic balance -- no other rewards regarding temperature, EV, etc...


TODO VP: look up KIT EnergyLab 2.0 data sources for weather data -- is it existing, can I use it?

#### Reinforcement Learning-based Home Energy Management with Heterogeneous Batteries and Stochastic EV Behaviour

Link: https://www.researchgate.net/publication/400459460_Reinforcement_Learning-based_Home_Energy_Management_with_Heterogeneous_Batteries_and_Stochastic_EV_Behaviour

Arxiv: https://arxiv.org/abs/2602.04578

Summary:
- EV, battery, PV -- focus on the exact battery degradation and on its simulation
- DRL, constrained Markov decision process (CMDP) and Lagrangian SAC
- HVAC included
- has different battery degradation dynamics
- primary and secondary constraints... -- ESS, cost opt and comfort, EV constraints
- benchmarks against 2 rule based controllers...
- nice, but no code available publically

Conclusion:
- in my framework battery deg can be built in (however it's not) -- only using a different battery class is needed
- they covered almost everything...
- no wind energy
- no TL, no generalisation -- only a single building
--> My project: TL and generalisation across configurations -- can we find such representation of the states, which is infrastructure independent and general for a lot of building charachteristics?
Finding the: "General controller" -- is it possble?

#### A comparative analysis of PPO and SAC algorithms for energy optimization with country-level energy consumption insights

Paper: https://www.sciencedirect.com/science/article/pii/S2468601825000501

Summary:
- rather larger scale: national-scale energy optimization
- PPO vs. SAC evaluation
- multi-phase evaluation strategy -- TODO VP: what do they mean by that?

Conclusion:
- not really relevant
- controlled thing is not clear (share of renewables and fossiles in the energy mix -- renewables are not controllable for the most of the time...)

#### A deep reinforcement learning approach based energy management strategy for home energy system considering the time-of-use price and real-time control of energy storage system

Link: https://www.sciencedirect.com/science/article/pii/S2352484724001501

Summary:
- 

Conclusion:

#### A multi-objective optimisation approach applied to offshore wind farm location selection

Link: https://link.springer.com/article/10.1007/s40722-017-0092-8




#### Deep reinforcement learning for energy management in a microgrid with flexible demand

Link: https://www.sciencedirect.com/science/article/pii/S2352467720303441

Source code: https://zenodo.org/records/3598386

Summary:
- Energy management of a microgrid -- wind turbine, ESS, HVAC, grid
- flexible resources, schedule them (e.g.: directly controllable loads, thermostatically controlled loads, price responsive loads, EVs)
- Electricity prices considered
- "increase the flexibility in demand by combining groups of TCLs and price-responsive loads participating in a demand response (DR) program, alongside a shared ESS, a wind power resource"
- writes about model based and model free methods, MPC and RL
- 7 SOTA RL algo, like A3C, PPO -- they improve these 2 algorihtms as well
- E_price and renewable production data from Finland
- optimisation on "gross energy profit from operations, and optimal use of local resources and flexibility components"
- they use real multi agent setup (with 3 layer architecture: control, information and physical layers)
- They use a clean MDP -- means no history states or history tracking.
- They only use ON/OFF at the HP, no exact energy control -- RL only controls if T_in is in a range, otherwise deterministic safety controller against too extreme temperatures. 
--> TODO VP: build in this safety control mechanism to the HP -- add a positive and negative bound to the desired temp datasource, read them as well as observations and in the case of violation, do max heating/max cooling regardless of rewards to maintain temperature. Maybe a similar mechanism is useful for EV charger -- plan a trajectory in the 1st place, if it's not followed with a margin, charge and do not care about other reward violations.
- They use discrete observation and discrete action space, e.g. discrete price levels, discrete, level-based actions -- smaller state and action spaces 
- they can shift consumptions and power grid loads to a certain extent. In each timestep, they choose a set of shifted loads to actually enable/disable -- formula for the choice, at high prices less likely execution, at low prices more likely exec -- Model for price responsive loads.
- they use single agent setup
- simple MDP: how is the dependency on the actual state understood? What the environment exposes? Basically yes
- Rewards only the energy aspects, uses hard constraints -- backup manual control if e.g. it would underheat the building, etc.. 
--> comfort aspects are not enforced by rewards.
- They use DQN, SARSA, Double DQN, PPO, SAC, A3C, DDPG

- TODO VP: pre heating/pre-cooling reward/behaviour when energy is cheap -- how to motivate this with reward? PreTempMaintain. 


- TODO VP: Add FutureObservationCollectorConnector -- to see static future states, like weather, prices, etc... -- add it as optional connector or EnvWrapper.
- TODO VP: Prediction model about the future state -- model based RL

##### MLFlow and Tensorboard

TODO VP: maybe integrate it -- for model, experiment and state-observation space management

MLFlow:
- Lifecycle management
- Tracking: log parameter, code versions, metrics, and output files, hyperparameters
- Model Registry: model versioning, version mangement and collaboration tool
- Deployment: model packaging -- how to deploy model to real application
- Agnostic: works with many language and frameworks
--> it's for reproducibility -- that's needed for me, to have a registry about models and their env -- as during the development both can change (mostly state and action spaces change of course -- the env, not really the model)
- MLflowLoggerCallback in ray tune, can store the final model checkpoint -- however this one is solved by the current setup as well.
--> MLFlow does not seem to good to track Env changes, behavioural changes...

Tensorboard:
- For metrics -- during training and across training runs, track achieved_reward, reward_rate and episode_reward_mean
- track policy_entropy, episode_reward_mean


TODO VP: This is implemented already: no need for real reward scheduling, because it's enough to have a fix set of rewards and it's enough to schedule the weighting of the rewards -- switched off rewards get 0.0 as weight.
The open question is the scheduling -- how are then the reward weights scheduled among all the rewards? --> e.g. Dirichlet (more or less)

Conclusion:
- Idea: use trajectory tracking for the hard constraints -- for temperature, EV charging, etc.., and use RL based controllers for th ESS -- which can react to the changes, it can plan and follow strategy.
- Idea: create a plan for EV charge, then track the planned trajectory -- planning can be RL based, trajectory tracking can be rule based.
- Idea: buying and selling energy prices can differ -- data difference causes then different strategies -- one could show this as well
- Important finding: semi deterministic training of A3C -- Eps greedy action selection: for X % select action based on policy, for the remaining select action completely random -- keeps up the exploration in later phases as well.
- Important finding: experience replay -- replay buffer, off policy algorithms, prioritise newer trajectories from the replay buffer.

TODO VP: https://www.clear.kit.edu/sparkassenpreis.php -- Thesis einreichen.

#### SACn: Soft Actor-Critic with n-step Returns

Link: https://www.researchgate.net/publication/398720775_SACn_Soft_Actor-Critic_with_n-step_Returns

N step returns: convergence speedup
Some improvements with the basic n step returns... -- because the original N step return with SAC introduces a bias, because of the changes in action distribution...

Not an important paper !!!

#### Continual Multi-Objective Reinforcement Learning via Reward Model Rehearsal

Link ACM: https://dl.acm.org/doi/10.24963/ijcai.2024/490
Link paper: https://www.ijcai.org/proceedings/2024/0490.pdf

About MORL
user weights the preferences -- which reward signal will be stronger, which objective is prioritised
"evolution of objectives throughout the learning process" -- multi phase, multi objective RL setup

TODO VP: How to mitigate the issue, that the EV controller does not always need a control signal -- but if actions are supressed, the exploration in that dimension dies, so when an EV is connected, the SA policy does not do anything because it has not discovered anything yet -- it was supressed.
--> this one yields for a standalone policy NN and standalone action selection for the EV controller as it's a shorter term problem...

"agent encounters a sequence of MORL tasks with objectives altering continually" -- it's like my constrained random reward selection idea/solution

some technique to update the policy with outdated rewards to not forget the older learnt behaviour... -- prevent catastrophic forgetting

TODO VP: continue a bit before the "4 Method" section

TODO VP: Hypervolume metric -- famous MORL metric

#### A fast and elitist multiobjective genetic algorithm: NSGA-II

Link: https://ieeexplore.ieee.org/document/996017



#### A practical guide to multi-objective reinforcement learning and planning

Link: https://link.springer.com/article/10.1007/s10458-022-09552-y

TODO VP: read this

#### Meta-Learning for Multi-objective Reinforcement Learning

Link: https://kth.diva-portal.org/smash/record.jsf?pid=diva2%3A1445556&dswid=-9547 / https://arxiv.org/abs/1811.03376

#### Dynamic weights in multi-objective deep reinforcement learning

Link: https://biblio.vub.ac.be/vubirfiles/76358018/abels19a.pdf

#### Prediction-Guided Multi-Objective Reinforcement Learning for Continuous Robot Control

Link: https://people.csail.mit.edu/jiex/papers/PGMORL/paper.pdf

"updates a policy population using an evolutionary algorithm to approximate the Pareto front"



#### Deep Reinforcement Learning for Real-Time Energy Management in Smart Home

Link: https://ieeexplore.ieee.org/document/10066193




- TODO VP: Idea: potential based reward shaping -- to the actual reward: add estimation about next state, substract estimation about the actual state.
- TODO VP: How to use expert trajectories for reward shaping -- distill reward function from expert trajectories -- inverse reinforcement learning, etc...
- TODO VP: 4. Preference-Based Construction: Sometimes you don't have full trajectories, or the trajectories are "noisy." You can construct a reward function by having a human (expert) compare two trajectory segments and say which is better. Collect pairs of short clips of the agent's behavior.Expert Labels: The expert identifies which clip is "better."Reward Modeling: A neural network $r_\psi(s, a)$ is trained via cross-entropy loss to predict the expert's preference.RL Training: Use the learned $r_\psi$ as the reward signal for standard RL.

#### Sample-Efficient Multi-Objective Learning via Generalized Policy Improvement Prioritization

Link: https://arxiv.org/pdf/2301.07784

MORL: set of policies with different preferences over the reward functions




### Frameworks

#### CityLearn

Link: https://www.citylearn.net/
GitHub: https://github.com/citylearn-project/CityLearn

Summary:
- MARL for energy coordination among multiple buildings
- flatten the energy need of a neighbourhood -- control multiple households with cooperating agents
- several controller types: Rule based control (RBC), MPC, RL
- PV, EV with V2G
- multiple buildings controlled together, to simulate a district -- possible to simulate a single building as well
- a simulation environment -- maybe a decent starting point

My project:
- single household -- no grid, no cooperation with other buildings
- intervention at eval, not only static behaviour (still a TODO)
- more options to eval: generalisation and transfer -- feasible with this one as well, just the data and config management is missing I guess
- Monte Carlo rollouts

#### SinerGym

Link: https://www.sciencedirect.com/science/article/pii/S0378778824011915
GitHub: https://github.com/ugr-sail/sinergym
Docs: https://ugr-sail.github.io/sinergym/compilation/main/index.html

Summary:
- seems really similar to my Gym and repo...
- Building energy optimisation (BEO)
- 3 other frameworks: RL Testbed for EnergyPlus, BOPTEST-Gym, Energym -- they are still active
- not maintained anymore: Gym-Eplus [10], ModelicaGym, [41], Tropical Precooling Environment [42], COmprehensive Building, Simulator (COBS) [43], and RL-EmsPy
- GridLearn [45] and Grid2Op [46] -- rahter grid management and not BEO
- it seems like they do not use price/energy cost data
- it seems like they only use TMY (typical meterological year -- median weather data over multiyear period), not daily weather data
- RL: they trained 40 years simulation -- 365 * 40 = 14600 episodes in my framework...
- fixed reward: in my repo, it's easily replaceable, flexible
- for them PPO was strong -- but only temp control and min. energy usage were the objectives.
- has W&B integration, configurable -- maybe a TODO VP: W&B and tensorboard fire up
- hyperparameter optimization of a DRL algorithm... -- that's what I would do as well...

Conclusion:
- in framework, it's similar, in goals it's not -- SinerGym paper is only about the framework and it's basic usage.
- I need a bit more advanced data and env config management
- in my work it's RLLib
- the paper shows some good figures about the training process, worth using them for ideas

TODO VP: Idea -- maybe it is easier to have MA setup with distinct state spaces -- then each agent NN has its own input heads (general input heads and specific input heads)
The problem is the state space inputs -- that can't be changed easily.
What if each state source / input has a pre-net, which translates the actual state to an intermediate N long vector -- each state variable/state source (history with K steps) would be mapped to an N long vector (only an N long vector, so it's an encoder...) -- state-source encoder? Trained to have an intermediate representation about the specific statesource.

TODO VP: Google DeepMind -- they reduced their energy usage as well, take a look onto that

##### EnergyPlus -- simulator:

Link: https://energyplus.readthedocs.io/en/latest/api.html

GitHub: https://github.com/NatLabRockies/EnergyPlus

Summary:
- What is it? An energy analysis and thermal load simulation program for buildings
- a mighty simulator/simulation engine
- Deals with building geometry, materials, HVAC systems, thermal calculations and energy usage
- OpenSource
- processes building config and weather time series

Conclusion:
- just simulator, no RL, no config management
- no building wide generalisation targeted -- EnergyPlus simulates buildings, nothing more

#### NatLabRockies/dss-cosim

GitHub: https://github.com/NatLabRockies/dss-cosim

Summary:
- Simulation framework
- interaction between power distribution and distr. energy resource controllers
- bridge between control logic and OpenDSS power distr. system simulator

Conclusion:
- not really relevant, it's rather large scale and more physical
- it's a simulator bridge, no control defined -- that's another module
- it's for testing controllers


#### Curriculum learning

Link: https://docs.ray.io/en/latest/rllib/rllib-examples.html#curriculum-learning

Useful docs, link: https://docs.ray.io/en/latest/rllib/rllib-advanced-api.html#curriculum-learning


#### Explicable Reward Design for Reinforcement Learning Agents

Link to paper: https://machineteaching.mpi-sws.org/files/papers/explicable_reward_design.pdf

Summary:
- it's about reward design, how to design explainable rewards -- what are the mathematical criteria
- explainable: informativeness and spareseness -- tradeoff


#### Key Features

- Single-zone indoor thermal model with electric heat pump control and heat loss dynamics
- Dynamic energy pricing and weather inputs
- Configurable heat pump control every 5 minutes
- Exogenous variables outdoor temperature and dynamic energy prices
- Modular design supporting custom reward modes and controllers (RL, PI, PID, Fuzzy, MPC)

### 1.2 Project Structure

```bash
LLECBuildingGym/                              # Root directory of the project
├── data/                                     # Input data (e.g., weather, pricing)
├── adv_building_gym/                         # Main Python package: Gym environment and controllers
│   ├── controllers/                          # Other controllers; Fuzzy, MPC, PI, PID
│   │   ├── __init__.py                       # Exports controller classes
│   │   ├── fuzzy_controller.py               # Fuzzy controller
│   │   ├── mpc_controller.py                 # MPC controller
│   │   ├── pi_controller.py                  # PI controller
│   │   ├── pid_controller.py                 # PID controller
│   │   └── README_MPC.md                     # MPC documentation and usage instructions
│   ├── envs/                                 # Submodule with environment definitions
│   │   ├── __init__.py                       # Exports environments for external use
│   │   └── base_building_gym.py              # Main environment logic and control integration
│   └── __init__.py                           # Registers environments
├── models/                                   # Saved trained models (PPO, SAC, DDPG,TD3, A2C)
├── plot-paper/                               # Notebooks to generate figures and tables
│   ├── check_envs_registration.ipynb         # Verifies registered Gymnasium environments
│   ├── generate_table03_summary_stats.ipynb  # Generate Table 03
│   ├── plot_fig03_temperature_data.ipynb     # Plots indoor/outdoor temperature data for Figure 03
│   ├── plot_fig04_price_data.ipynb           # Plots dynamic energy prices for Figure 04
│   ├── plot_fig05_indoor_temp_setpoint.ipynb # Plots dynamic indoor temp setpoints for Figure 05
│   └── preprocess_outdoor_temperature.ipynb  # Prepares outdoor temperature time series
├── slurm_logs/
│   ├── eval/                                 # SLURM logs from evaluation jobs
│   └── train/                                # SLURM logs from training jobs
│   └── data_setup/                           # SLURM logs from data setup jobs
├── slurm_script/                             # SLURM job submission scripts
├── results/                                  # Evaluation logs and result CSVs
├── .gitignore                                # Ignore in version control
├── LICENSE                                   # Licensing
├── README.md                                 # Repo documentation and usage instructions
├── pyproject.toml                            # Build system configuration
├── requirements.txt                          # Python dependencies
├── run_train_ray.py                          # Train RL models on the Ray RLlib new API stack
├── run_eval_ray.py                           # Evaluate Ray-trained models
└── run_train_sb.py                           # (Secondary) Stable-Baselines3 training
```

</details>

## 2. Installation and Environment Setup

<details>
  <summary>Click to expand/collapse</summary>

### 2.1a Haicore (Linux):

Install / make sure you have Python 3.12.1 (`python --version` or `python3.12 --version`)

Install link: https://www.python.org/downloads/release/python-3121/

Clone the repository:
```bash
git clone https://github.com/vince-pongracz/AdvBuildingGym
python3.12 -m venv adv_env
source adv_env/bin/activate
cd AdvBuildingGym

alias pip='python -m pip'

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

The virtual environment and project directory should be organized as shown below:
```bash
adv_env/        # Python virtual environment
AdvBuildingGym/ # Root directory of the project
```

### 2.1b Local (Windows):

Install Python 3.12.1 from https://www.python.org/downloads/release/python-3918 (newer Python versions may work but are not tested).

```bash
git clone https://github.com/KIT-IAI/LLECBuildingGym
py -3.12 -m venv llec_env
.\llec_env\Scripts\activate
cd LLECBuildingGym

python -m pip install --upgrade --force-reinstall pip
pip install -r requirements_windows.txt
pip install -e .
```


### 2.2 Reinstallation (after code changes):

```bash
pip uninstall adv_building_gym -y
pip install -e .
```

### 2.3 Environment Check (verify that the environment is registered correctly):

```bash
python check_envs_registration.ipynb
```

### 2.4 For using Jupyter notebooks:

```bash
source llec_env/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=llec_env --display-name "Python (llec_env)"
jupyter kernelspec list
```

Always activate the virtual environment (`source llec_env/bin/activate`) before starting Jupyter to ensure correct dependencies.
After registering the kernel, restart Jupyter so the `Python (llec_env)` kernel becomes available.

</details>

## Data Preprocessing

Before training, set up data with the unified setup script.
By default it runs **both** pipelines: electricity prices and weather/Zenodo.

```bash
# Recommended: full setup (price + weather)
python preprocessing/data_setup.py

# Price-focused run only (disable weather pipeline)
python preprocessing/data_setup.py --skip-weather --years 2023 2024 2025 2026

# If raw prices already exist locally, skip API calls
python preprocessing/data_setup.py --skip-weather --years 2025 --skip-price-fetch --raw-price-files data/e_price/2025_prices.csv

# Weather-focused run only (disable price pipeline)
python preprocessing/data_setup.py --skip-prices --skip-zenodo-download
```

See [data/DATA_README.md](data/DATA_README.md) for all options and manual fallback commands.

## 3. Training and Evaluation

The current training stack is **Ray RLlib (new API stack)**. RL training and
evaluation each have their own driver scripts; baseline controllers (PI, PID,
Fuzzy, MPC) live in `adv_building_gym/controllers/` and are reachable from the
evaluation flow.

### 3.1 RL Training (Ray RLlib)

Driver: [run_train_ray.py](run_train_ray.py). Requires an env-topology YAML
via `--load-config`; SLURM-allocated GPU is required.

```bash
# PPO on the small env, 3500 episodes
python run_train_ray.py --algorithm ppo --load-config configs/env/env_test1_small.yaml --episodes 3500

# SAC, custom seed and best-checkpoint metric
python run_train_ray.py --algorithm sac --load-config configs/env/env_test1_mid.yaml --seed 18 --episodes 5000 --metric achieved_reward
```

Common flags: `--algorithm {ppo,sac}`, `--episodes`, `--seed`,
`--metric {reward_rate,achieved_reward,episode_return_mean}`,
`--checkpoint-frequency-episodes`, `--data-config`, `--reward-schedule`,
`--infra-schedule`, `--grad-train`, `--log-trajectories`.

Hyperparameters (algorithm-agnostic + per-algorithm) live in
[configs/training_param_config.yaml](configs/training_param_config.yaml). See
[docs/workflow.md](docs/workflow.md) for the end-to-end flow.

### 3.2 Evaluation (Ray RLlib)

Driver: [run_eval_ray.py](run_eval_ray.py). Runs CPU-only, loads an `RLModule`
checkpoint, and replays episodes through the same connector pipeline used at
training time.

```bash
python run_eval_ray.py --algorithm ppo --load-config configs/env/env_test1_small.yaml --checkpoint <path> --episodes 10
python run_eval_ray.py --algorithm sac --load-config configs/env/env_test1_mid.yaml --plot
```

Outputs (mean / std / min / max per metric, per-episode trajectory JSON, and
`trajectories.hdf5`) land in `eval_results/<YYYYmmdd_HHMM>_eval/`.

### 3.3 Baselines

`adv_building_gym/controllers/` contains PI, PID, Fuzzy and MPC (Pyomo)
implementations that share the env's Dict action space — useful as
non-learning baselines for evaluation comparisons.

### 3.4 Stable-Baselines3 (secondary)

[run_train_sb.py](run_train_sb.py) provides a Stable-Baselines3 path for the
same env. It is kept around for cross-checking but is not the primary
training stack.

TODO VP: Rework this part

<h2>4. Citation &#128221;</h2>
<p>
If you use this framework in your research, please consider citing our paper &#128221; and giving the repository a star &#11088;:
</p>

```bibTeX
@inproceedings{demirel2025_LLECBuildingGym,
      title={Advanced Deep Reinforcement Learning for Heat Pump Control in Residential Buildings},
      author={Gökhan Demirel and Ömer Ekin and Jianlei Liu and Luigi Spatafora and Kevin Förderer and Veit Hagenmeyer},
      year={2025},
      booktitle={Proceedings of the IEEE ISGT Europe 2025 (accepted)},
      address = {Malta},
      url = {https://github.com/KIT-IAI/LLECBuildingGym},
      pages={1--5}
}
```

## License

This code is licensed under the **[MIT License](LICENSE)**.
For any issues or any intention of cooperation, please feel free to contact me at **[pongrvin@gmail.com](pongrvin@gmail.com)**.
