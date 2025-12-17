import sys
from typing import Any, Dict
from collections import OrderedDict
import logging

import gymnasium as gym
from gymnasium import Space, spaces
from gymnasium.spaces import Dict as SDict
import numpy as np
from numpy.typing import NDArray
import pandas as pd

from llec_building_gym.devices.data_sources import DataSource
from llec_building_gym.rewards.rewards import RewardFunction
from llec_building_gym.devices.infrastructure import Infrastructure

from llec_building_gym.utils.temporal_features import TemporalFeatureBuffer
from llec_building_gym.utils.action_space_wrapper import dict_to_vec, vec_to_dict, dict_space_to_space

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.debug(f"sys.argv: {sys.argv}")


class BuildingProps():
    def __init__(self, mC: float = 300, K: float = 20):
        self.mC = mC
        self.K = K


class AdvBuildingGym(gym.Env):
    """
    AdvBuildingGym
    
    A Gymnasium-compatible environment for controlling building-level devices and
    energy systems (e.g., heat pumps, batteries, consumers) via a collection of
    Infrastructure modules, DataSource providers, and reward functions.
    This environment is designed to be modular: infrastructures define actuators
    and their action space and implement how actions affect the environment state;
    datasources provide observation signals and state update logic; reward
    functions compute objective values from the current action and state.
    Core behavior
    - The environment advances in fixed discrete control steps (control_step, seconds)
        until simulation_time is reached or the episode is otherwise terminated.
    - Observations are assembled from datasources (and any internal bookkeeping)
        into a dictionary-based observation (SDict).
    - Actions are provided as a dictionary mapping infrastructure identifiers to
        their respective action arrays; each infrastructure executes its own action
        via infr.exec_action(action, state).
    - Rewards are the sum of values returned by each configured RewardFunction.
    - The environment keeps a simple iteration counter and exposes cumulative
        energy and other datasource-provided signals in the state dict.
    Initialization (constructor arguments)
    - infras: list[Infrastructure]
            Infrastructure objects that define actions, their effects and setup logic.
            Each infrastructure is expected to implement setup_spaces(observation_space, action_space)
            and exec_action(action, state).
    - datasources: list[DataSource]
            Data sources that populate observations and update state over time. Each
            datasource must implement setup_spaces(observation_space, action_space)
            and update_state(state).
    - rewards: list[RewardFunction]
            List of reward function objects used to compute the environment reward at
            each step. Each reward function is queried via get_reward(action, state).
    - building_props: BuildingProps
            Static description of building parameters used by infras/datasources/rewards.
    - simulation_time: int (seconds, default 24*60*60)
            Total simulation duration.
    - control_step: int (seconds, default 300)
            Duration of a single control step / time advancement between calls to step.
    - schedule_type: optional
            Optional type identifier controlling e.g. outdoor temperature schedules.
    - render_mode: optional
            Reserved for Gym compatibility.
    - training: bool (default True)
            Training mode flag (user-defined semantics).
    - train_ratio: float (default 0.8)
            Ratio used by some Datasources to split training/validation sequences (if applicable).
    - prediction_horizon: int
            Number of discrete steps to keep in TemporalFeatureBuffer for forecasting signals.
    - **kwargs:
            Additional environment-specific parameters forwarded or ignored.
    Observation and action spaces
    - The environment builds an observation_space (SDict) and action_space (SDict)
        by aggregating spaces declared by every Infrastructure and DataSource. The
        env also supplies a "prev_action" entry (Box in [-1, 1]) and bookkeeping keys
        such as "cum_E_Wh" and "iteration".
    - Observations returned by reset() and step() are Python dicts matching the
        observation_space keys. The environment does not return a flattened vector
        by default.
    State
    - self.state is a dict that holds the latest observation values along with
        internal bookkeeping entries. Datasources and infrastructures update this
        dict in-place during reset and step. Typical keys include:
            - iteration: current step index (int)
            - cum_E_Wh: cumulative electrical energy consumed (or produced) in Wh
            - device-specific signals provided by datasources/infrastructures
    Reset semantics
    - reset(seed=None, options=None) -> (state: dict, info: dict)
            Resets internal state and datasources. If seed is None, a random integer
            seed is sampled. The returned state is a dict of observations; info
            contains the seed used.
    Step semantics
    - step(action: Dict[str, np.ndarray]) -> (state: dict, reward: float,
        terminated: bool, truncated: bool, info: dict)
            - Executes the provided actions by delegating to each infrastructure.
            - Calls datasources to update the state after actions are applied.
            - Aggregates rewards by summing get_reward(action, state) from each
                RewardFunction in self.reward_funcs.
            - Increments the internal "iteration" counter.
            - terminated is True when the maximum number of iterations (simulation_time /
                control_step) is reached; truncated is always False in the current
                implementation.
            - info contains keys: "action", "reward", "state", and "E_HP_el_Wh" (alias
                for cumulative energy key) among any additional diagnostic entries.
    Rewarding
    - Rewards are computed as the sum of all configured RewardFunction objects.
        Individual reward contributions are available in local reward_hist during
        step execution and may be exposed via debug logging or extended info dicts.
    Termination
    - By default, the episode terminates when iteration >= max_iteration
        (computed from simulation_time and control_step). Additional termination
        logic can be implemented inside infrastructures/datasources/reward functions
        by modifying state or setting flags.
    Extensibility notes
    - The environment is intentionally modular: add new actuators or sensors by
        implementing and registering Infrastructure and DataSource classes that
        follow the expected interfaces.
    - RewardFunction objects encapsulate objective logic and can be combined to
        form multi-objective rewards.
    - TemporalFeatureBuffer is used internally to store prediction windows for
        forecasted signals (prediction_horizon). Datasources may interact with it.
    Return types and compatibility
    - Conforms to the Gymnasium step/reset semantics returning Python objects:
        - reset -> (observation, info)
        - step  -> (observation, reward, terminated, truncated, info)
    - Observation and action structures are dictionary-like (SDict) rather than
        single numpy arrays; agents must map their policies to the composite action
        dictionary expected by the registered infrastructures.
    Example (high level)
            env = AdvBuildingGym(infras, datasources, rewards, building_props)
            state, info = env.reset()
            action = {infra.name: infra.default_action() for infra in infras}
            next_state, reward, done, truncated, info = env.step(action)
    """
    def __init__(
        self,
        infras: list[Infrastructure],
        datasources: list[DataSource],
        rewards: list[RewardFunction],
        building_props: BuildingProps,
        simulation_time=24 * 60 * 60,
        control_step=300,
        schedule_type=None,
        render_mode=None,
        training=True,
        train_ratio=0.8,
        # Number of steps to look ahead for forecasted values
        prediction_horizon=12 * 5 * 6,
        **kwargs,
    ):
        """
        Initialize the Gym environment with building parameters and reward structure.

        Args:
            mC (float, optional): Effective thermal mass (J/K). Defaults to 300.
            K (float, optional): Overall heat transfer coefficient (W/K). Defaults to 20.
            Q_HP_Max (float, optional): Max heat pump power in W. Defaults to 1500.
            simulation_time (int, optional): Total simulation duration in seconds.
                Defaults to one day (24*60*60).
            control_step (int, optional): Simulation time step in seconds.
                Defaults to 300.
            schedule_type (str, optional): Type of outdoor temperature schedule.
                Defaults to "24-hour".
            reward_mode (str, optional): One of {"temperature", "combined"}.
                Determines how rewards are calculated. Defaults to "temperature".
            temperature_weight (float, optional): Weight for the temperature objective
                in the combined mode. Defaults to 1.0.
            economic_weight (float, optional): Weight for the economic objective
                in the combined mode. Defaults to 1.0.
            render_mode (None or str, optional): For compatibility with Gym APIs.
                Not used in this environment. Defaults to None.
        """
        super(AdvBuildingGym, self).__init__()

        self.iteration = 0
        
        observation_space = OrderedDict()
        action_space = OrderedDict()

        # TODO VP 2025.12.03. : refactor it to a datasource...
        observation_space["cum_E_Wh"] = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        self.infras = infras
        self.action_space_keys = []
        for infr in self.infras:
            infr.setup_spaces(observation_space, action_space)
            self.action_space_keys.append(f"{infr.name}_action")

        self.datasources = datasources
        for ds in self.datasources:
            ds.setup_spaces(observation_space, action_space)

        self.reward_funcs = rewards

        observation_space["prev_action"] = spaces.Box(low=-1, high=1, shape=(len(action_space),), dtype=np.float32)

        # Assign spaces, but it only goes like this...
        self.observation_space = SDict(observation_space)
        self.state = OrderedDict()
        for state_name, state_space in observation_space.items():
            self.state[state_name] = np.zeros(shape=state_space.shape)
        
        # self.action_space_keys = list(action_space.keys())
        self.action_space = SDict(action_space)
        # self.action_space = dict_space_to_space(action_space)

        self.building_props = building_props
        self.simulation_time = simulation_time
        self.prediction_horizon = prediction_horizon
        self.control_step = control_step
        self.training = training
        self.train_ratio = train_ratio
        self.max_iteration = int(self.simulation_time / self.control_step)

        # TODO VP 2025.12.09. : inspect this
        self.temporal_features = TemporalFeatureBuffer(window_size=self.prediction_horizon)

        self.state, _ = self.reset()
        
        # logger.debug("After reset")
        # for k, v in self.state.items():
        #     logger.debug(f"{k}: shape: {v.shape}")

        logger.info("AdvBuildingGym created!")
        logger.info("  Objectives: %s", [rew.name for rew in rewards])
        logger.info("  Actions: %s", [infr.name for infr in infras])
        logger.info("  States: %s", [ds.name for ds in datasources])
        

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is None:
            seed = np.random.randint(0, 10000)  # global RNG
        super().reset(seed=seed, options=options)

        self.iteration = 0
        for sync in self.infras + self.datasources:
            sync.synchronize(self.iteration)

        logger.debug(self.state.keys())

        for k, v in self.state.items():
            if isinstance(v, np.ndarray):
                self.state[k] = np.zeros(self.state[k].shape, dtype=v.dtype)  # type: ignore -- supresses warning
            else:
                logger.debug("Unidentified type: %s", type(v))
                
        info = {"seed": seed}
        return self.state, info

    def _get_observation(self) -> dict:
        # Start with the current env state so datasources have access to
        # bookkeeping keys such as "iteration" and "sim_hour" during reset.
        state = OrderedDict(self.state) if isinstance(self.state, OrderedDict) else OrderedDict()
        for ds in self.datasources:
            # datasources accept a dict and update it in-place
            ds.update_state(states=state)
        return state

    def is_done(self):
        return bool(self.iteration >= self.max_iteration)

    def step(self, action):
        """
        Execute a single control step in the env by applying the selected
        action to the heat pump.

        Calculates:
          - new state
          - reward
          - Termination conditions

        Args:
            action (np.array): ... the control action in [-1, 1].

        Returns:
            observation (np.array): The next observation.
            reward (float): The computed reward signal.
            terminated (bool): True if the episode has ended, else False.
            truncated (bool): False in this environment.
            info (dict): Additional information data.
        """
        # d_action = vec_to_dict(action, self.action_space_keys)
        
        # Store the current action

        for infr in self.infras:
            infr.exec_action(action, self.state)
            infr.update_state(self.state)

        for ds in self.datasources:
            ds.update_state(states=self.state)

        # Track last executed action for observability (flattened order matches action_space_keys)
        prev_action_vec = []
        for key in self.action_space_keys:
            act_val = action.get(key)
            act_scalar = float(np.atleast_1d(act_val)[0]) if act_val is not None else 0.0
            prev_action_vec.append(act_scalar)
        self.state["prev_action"] = np.array(prev_action_vec, dtype=np.float32)
            
        # logger.debug("Updates")
        # for k, v in self.state.items():
        #     logger.debug(f"{k}: shape: {v.shape}")

        reward: float = 0
        reward_hist = dict()
        for rew_f in self.reward_funcs:
            rew_val = float(np.asarray(rew_f.get_reward(action, self.state)).item())
            reward += rew_val
            reward_hist[rew_f.name] = rew_val

        # logger.debug(
        #     "act %.2f | price %.3f | cost %.1f | econ_r(norm) %.3f",
        #     action[0],
        #     self.building.energy_price,
        #     -reward_economic,
        #     reward_economic_norm,
        # )
        # Check if episode should terminate
        terminated = self.is_done()
        truncated = False
        
        self.iteration += 1
        # Sync iterations
        for sync in self.infras + self.datasources:
            sync.synchronize(self.iteration)

        logger.debug(f"Iter: {self.iteration}\n")
        
        info = {
            "action": action,
            "reward": reward,
            "state": {k: np.array(v, copy=True) for k, v in self.state.items()},
            "E_HP_el_Wh": self.state["cum_E_Wh"],
        }

        return self.state, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        pass
