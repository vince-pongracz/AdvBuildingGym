import sys
from typing import Any, Dict
from collections import OrderedDict
import logging

import gymnasium as gym
from gymnasium import Space, spaces
from gymnasium.spaces import Dict as SDict
import numpy as np
import pandas as pd

from adv_building_gym.devices.statesources import StateSource
from adv_building_gym.rewards import RewardFunction
from adv_building_gym.devices.infrastructure import Infrastructure

from adv_building_gym.utils.temporal_features import TemporalFeatureBuffer
from adv_building_gym.utils.warning_filters import setup_warning_filters
from adv_building_gym.envs.utils import BuildingProps

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True  # Override any existing logging configuration (e.g., from Ray/RLlib)
)
logger = logging.getLogger(__name__)

# TODO VP 2026.01.12. : Use Env-to-module and module-to-Env pipelines and keep the dict state and action spaces in the env...
# So the mapping from action vector to dict and vice versa is done in the pipelines, not in the env directly... -- this mapping is rather the task of the Rllib, not the env's
# In this case, SB could not really work anymore, because of the dict spaces... but RLlib could work with custom pipelines...
# Link: https://docs.ray.io/en/latest/rllib/env-to-module-connector.html#env-to-module-pipeline-docs

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
        statesources: list[StateSource],
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
        """
        # TODO VP 2026.01.07. : add docsstring
        # Setup warning filters for Ray workers (must be called early)
        setup_warning_filters()

        super(AdvBuildingGym, self).__init__()


        self.iteration = 0

        observation_space = OrderedDict()
        action_space = OrderedDict()

        # TODO VP 2025.12.03. : refactor it to a datasource..? -- no, but keep track of it
        observation_space["cum_E_Wh"] = spaces.Box(
            low=np.zeros((1,), dtype=np.float32),
            high=np.full((1,), np.inf, dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        self.infras = infras
        for infr in self.infras:
            infr.setup_spaces(observation_space, action_space)
        self.action_space_keys = list(action_space.keys())

        self.statesources = statesources
        for ds in self.statesources:
            ds.setup_spaces(observation_space, action_space)

        self.reward_funcs = rewards

        # Calculate total action dimension (sum of all action space shapes)
        # This must match the flattened action vector built in step()
        total_action_dim = sum(int(np.prod(space.shape)) for space in action_space.values())

        observation_space["prev_action"] = spaces.Box(
            low=np.full((total_action_dim,), -1.0, dtype=np.float32),
            high=np.full((total_action_dim,), 1.0, dtype=np.float32),
            shape=(total_action_dim,),
            dtype=np.float32,
        )

        # Assign spaces, but it only goes like this...
        self.observation_space = SDict(observation_space)
        self.state = OrderedDict()
        for state_name, state_space in observation_space.items():
            # Initialize internal state arrays with the same dtype as the declared space
            self.state[state_name] = np.zeros(shape=state_space.shape, dtype=state_space.dtype)

        # Store the original Dict action space for internal use
        self._dict_action_space = SDict(action_space)

        # Flatten Dict action space to Box for compatibility with RLlib's
        # vectorized environments (SyncVectorEnv). Dict action spaces cause
        # IndexError in gymnasium's _iterate_dict when used with vectorization.
        # Note: total_action_dim already calculated above for prev_action space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(total_action_dim,),
            dtype=np.float32,
        )

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

        logger.debug("AdvBuildingGym created!")
        logger.debug("  Objectives: %s", [rew.name for rew in rewards])
        logger.debug("  Actions: %s", [infr.name for infr in infras])
        logger.debug("  States: %s", [ds.name for ds in statesources])

    def get_state_space(self):
        return self.observation_space
    
    def get_action_space(self):
        return self.action_space
    
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is None:
            seed = np.random.randint(0, 10000)  # global RNG
        super().reset(seed=seed, options=options)

        self.iteration = 0
        for sync in self.infras + self.statesources:
            sync.synchronize(self.iteration)

        for k, v in self.state.items():
            if isinstance(v, np.ndarray):
                # Initialize with zeros using the correct dtype (float32)
                self.state[k] = np.zeros(self.state[k].shape, dtype=np.float32)
            else:
                logger.debug("Unidentified type: %s", type(v))

        # Update state from statesources to populate initial observations
        # This ensures observations are within bounds after reset
        for ds in self.statesources:
            ds.update_state(states=self.state)

        # Update infrastructure states as well
        for infr in self.infras:
            infr.update_state(self.state)

        info = {"seed": seed}
        return self.state, info

    def _get_observation(self) -> dict:
        # Start with the current env state so statesources have access to
        # bookkeeping keys such as "iteration" and "sim_hour" during reset.
        state = OrderedDict(self.state) if isinstance(self.state, OrderedDict) else OrderedDict()
        for ds in self.statesources:
            # statesources accept a dict and update it in-place
            ds.update_state(states=state)
        return state

    def is_done(self) -> bool:
        return bool(self.iteration >= self.max_iteration)

    def _flat_action_to_dict(self, flat_action: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert flat action array to Dict format for infrastructure use.

        Handles various input formats from RLlib/SyncVectorEnv:
        - 1D array: [a1, a2] -> use directly
        - 2D array with batch dim: [[a1, a2]] -> flatten
        - Nested structures -> flatten
        """
        logger.debug(f"Flat action dim: {flat_action.shape}")

        # Convert to numpy array and flatten to 1D
        flat_action = np.asarray(flat_action, dtype=np.float32).flatten()

        # Calculate expected dimension
        expected_dim = sum(
            int(np.prod(space.shape)) for space in self._dict_action_space.spaces.values()
        )

        # Validate action size
        if flat_action.size != expected_dim:
            raise ValueError(
                f"Action size mismatch: expected {expected_dim}, got {flat_action.size}. "
                f"Action shape: {np.asarray(flat_action).shape}, Action: {flat_action}"
            )

        dict_action = {}
        idx = 0
        for key in self.action_space_keys:
            space = self._dict_action_space.spaces[key]
            action_dim = int(np.prod(space.shape))
            dict_action[key] = flat_action[idx:idx + action_dim].reshape(space.shape)
            idx += action_dim
        return dict_action

    def step(self, action):
        """
        Execute a single control step in the env by applying the selected action.

        Calculates:
          - new state
          - reward
          - termination conditions

        Args:
            action (np.array): Flat action array in [-1, 1] with shape (total_action_dim,).

        Returns:
            observation (np.array): The next observation.
            reward (float): The computed reward signal.
            terminated (bool): True if the episode has ended, else False.
            truncated (bool): False in this environment.
            info (dict): Additional information data.
        """
        # Clip actions and store for logging
        clipped_flat_action = np.clip(action, -1, 1)
        # Convert flat action to Dict format for internal infrastructure use
        action = self._flat_action_to_dict(clipped_flat_action)

        for infr in self.infras:
            infr.exec_action(action, self.state)
            infr.update_state(self.state)

        for ds in self.statesources:
            ds.update_state(states=self.state)

        # Track last executed action for observability (flattened order matches action_space_keys)
        prev_action_vec = []
        for key in self.action_space_keys:
            act_val = action.get(key)
            if act_val is not None:
                # Flatten multi-dimensional actions (e.g., HP_action is 2D)
                act_array = np.atleast_1d(act_val).flatten()
                prev_action_vec.extend(act_array)
            else:
                # Default to 0 if action not present
                space = self._dict_action_space.spaces[key]
                action_dim = int(np.prod(space.shape)) if space.shape else 1
                prev_action_vec.extend([0.0] * action_dim)
        self.state["prev_action"] = np.array(prev_action_vec, dtype=np.float32)

        # Calculate reward
        reward: float = 0
        for rew_f in self.reward_funcs:
            rew_val = float(np.asarray(rew_f.get_reward(action, self.state)).item())
            reward += rew_val

        # Sync iterations
        for sync in self.infras + self.statesources:
            sync.synchronize(self.iteration)

        # Check if episode should terminate
        terminated = self.is_done()
        truncated = False

        if not terminated:
            self.iteration += 1

        info = {
            "action": action,  # Dict format (for compatibility)
            "clipped_action": clipped_flat_action,  # Flat clipped action for logging
            "reward": reward,
            "state": {k: np.array(v, copy=True) for k, v in self.state.items()},
            "cum_E_Wh": self.state["cum_E_Wh"],
        }

        return self.state, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        pass
