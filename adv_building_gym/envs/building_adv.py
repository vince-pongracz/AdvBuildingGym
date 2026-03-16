import sys
from typing import Any, Dict
from collections import OrderedDict
import logging

import gymnasium as gym
from gymnasium import Space, spaces
from gymnasium.spaces import Dict as SDict
import numpy as np
import pandas as pd

from adv_building_gym.data_combinator import DataCombinator
from adv_building_gym.devices.statesources import StateSource
from adv_building_gym.rewards import RewardFunction
from adv_building_gym.devices.infrastructure import Infrastructure

from adv_building_gym.utils.temporal_features import TemporalFeatureBuffer
from adv_building_gym.utils.warning_filters import setup_warning_filters
from adv_building_gym.envs.data_variant import DataVariantProvider
from adv_building_gym.envs.utils import BuildingProps

from adv_building_gym.config.env_config import config as env_config

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True  # Override any existing logging configuration (e.g., from Ray/RLlib)
)
logger = logging.getLogger(__name__)


class AdvBuildingGym(gym.Env, DataVariantProvider):
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
        until max_iterations reached or the episode is otherwise terminated.
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
        env also supplies per-key action history windows (``prev_{key}_hist``).
    - Observations returned by reset() and step() are Python dicts matching the
        observation_space keys. The environment does not return a flattened vector
        by default.
    State
    - self.state is a dict that holds the latest observation values along with
        internal bookkeeping entries. Datasources and infrastructures update this
        dict in-place during reset and step. Typical keys include:
            - iteration: current step index (int)
            - device-specific signals provided by datasources/infrastructures
    Info
    - step() returns an info dict containing:
            - cum_E_kWh: cumulative net electrical energy in kWh (positive=consumption, negative=production)
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
            - terminated is True when the maximum number of iterations is reached; truncated is always False in the current
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
        control_step: int | None = None,
        schedule_type=None,
        render_mode=None,
        training=True,
        train_ratio=0.8,
        # Number of steps to look ahead for forecasted values
        prediction_horizon=8 * 12,  # 8 hours at 5-minute steps
        data_combinator: DataCombinator | None = None,
        log_full_info: bool = False,
        action_history_length: int | None = None,
        **kwargs,
    ):
        """Initialise the building-energy gym environment.

        Args:
            infras: Infrastructure components (HP, battery, EV charger, etc.)
                that define controllable action and observation sub-spaces.
            statesources: External state sources (weather, pricing, schedules)
                that inject uncontrollable observations into the state.
            rewards: Reward functions evaluated at each step to produce
                the scalar reward signal.
            building_props: Physical and thermal properties of the building.
            control_step: Time between control actions in seconds (default: 300 s).
            schedule_type: Optional schedule identifier for occupancy / usage patterns.
            render_mode: Gymnasium render mode (currently unused).
            training: If True, sample from the training data split; otherwise
                use the held-out evaluation split.
            train_ratio: Fraction of available data used for training (default: 0.8).
            prediction_horizon: Number of future time-steps included in
                forecast observations (default: 360, i.e. 30 h at 300 s steps).
        """

        # Setup warning filters for Ray workers (must be called early)
        setup_warning_filters()

        super(AdvBuildingGym, self).__init__()


        self.iteration = 0
        self.cum_E_kWh = 0.0  # Cumulative net energy in kWh (tracked in info, not observation)
        
        self.episode_count: int = 0
        self.data_combinator = data_combinator if data_combinator is not None else DataCombinator()
        self._rng: np.random.Generator | None = None
        self._episode_date: str = ""
        self._episode_day_mode: str = "none"

        observation_space = OrderedDict()
        action_space = OrderedDict()

        self.infras = infras
        for infr in self.infras:
            infr.setup_spaces(observation_space, action_space)
        self.action_space_keys = list(action_space.keys())

        self.statesources = statesources
        for ds in self.statesources:
            ds.setup_spaces(observation_space, action_space)

        self.reward_funcs = rewards

        # sim_hour: normalised simulation hour [0, 1] derived from the current
        # step within the episode.  0.0 = start of day, 1.0 = end of day.
        # Managed directly by the environment (not a StateSource).
        observation_space["sim_hour"] = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32,
        )

        # Add per-key action history windows to the observation space.
        # Each entry ``prev_{key}_hist`` has shape ``(action_history_length, *action_shape)``
        # and stores a rolling window of the N most recent executed actions
        # (oldest first, newest last).  The agent can use this to reason about
        # action smoothness and the ActionSmoothnessReward reads the latest
        # entry to compute the change penalty.
        # FlattenObservations (RLlib connector) handles flattening for the RL module.
        self.action_history_length = action_history_length if action_history_length is not None else env_config.ACTION_HISTORY_LENGTH
        for key, space in action_space.items():
            hist_shape = (self.action_history_length, *space.shape)
            obs_key = f"prev_{key}_hist"
            observation_space[obs_key] = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=hist_shape,
                dtype=np.float32,
            )

        # Assign spaces
        self.observation_space = SDict(observation_space)
        self.state = OrderedDict()
        for state_name, state_space in observation_space.items():
            # Initialize internal state arrays with the same dtype as the declared space
            self.state[state_name] = np.zeros(shape=state_space.shape, dtype=state_space.dtype)

        # Store the original Dict action space for internal use
        self._dict_action_space = SDict(action_space)

        # Flatten Dict action space to Box for compatibility with RLlib.
        # RLlib's SingleAgentEnvRunner.get_spaces() reads env.action_space
        # directly and passes it to the RLModule/Catalog, which only accepts
        # Box or Discrete (e.g. SAC rejects Dict).  The flat<->dict conversion
        # is done internally via _flat_action_to_dict().
        total_action_dim = sum(int(np.prod(space.shape)) for space in action_space.values())
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(total_action_dim,),
            dtype=np.float32,
        )

        self.building_props = building_props
        # NOTE VP 2026.02.28. : Simulation time is in seconds
        self.simulation_time = env_config.CONTROL_STEP * env_config.EPISODE_LENGTH
        self.prediction_horizon = prediction_horizon
        self.control_step = control_step if control_step is not None else env_config.CONTROL_STEP
        self.training = training
        self.train_ratio = train_ratio
        self.max_iteration = env_config.EPISODE_LENGTH

        # When True, step()/reset() include a deep copy of the full named state
        # dict in info["state"]. Expensive in memory — enable for evaluation only.
        self.log_full_info: bool = log_full_info

        # TODO VP 2025.12.09. : inspect this -- drop it, it is not useful for us for now
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
    
    def apply_data_variant(self, variant: dict[str, str]) -> None:
        """Reload statesources whose names appear in *variant*.

        Args:
            variant: Mapping of statesource name -> new CSV file path.
                     Only matching statesources are reloaded; others are untouched.
        """
        for state_src in self.statesources:
            if state_src.name in variant:
                state_src.reload(variant[state_src.name])

    # TODO VP 2026.03.10. : This does not belong strictly to the env...
    def _resolve_episode_date(self, row_offset: int) -> str:
        """Derive a date string from the row offset using the first statesource with a 'start' column."""
        for src in self.statesources:
            if src.ts is not None and "start" in src.ts.columns and row_offset < len(src.ts):
                return str(pd.to_datetime(src.ts.iloc[row_offset]["start"]).date())
        # Fallback: day-of-year index
        steps_per_day = int(86400 / self.control_step)
        return f"day-{row_offset // steps_per_day}"

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is None:
            seed = np.random.randint(0, 10000)  # global RNG
        super().reset(seed=seed, options=options)

        # Seed the deterministic RNG for DataCombinator random mode
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # ======== Data variant selection logic ========
        # Approach A: episode-count-based data variant swap
        self.episode_count += 1
        variant = self.data_combinator.get_variant(self.episode_count, self._rng)
        if variant:
            self.apply_data_variant(variant)

        # Approach C: external override via reset(options={"data_variant": {...}})
        if options and "data_variant" in options:
            variant = options["data_variant"]
            self.apply_data_variant(variant)

        # Compute day offset from DataCombinator (must run before logging so get_day_date() is set)
        steps_per_day = env_config.EPISODE_LENGTH  # Assuming 1 day per episode; adjust if multiple days per episode
        row_offset = 0
        # Determine max available days and data start year from the first statesource with data
        max_days = 1
        data_start_year = None
        for src in self.statesources:
            if src.ts is not None and len(src.ts) >= steps_per_day:
                max_days = len(src.ts) // steps_per_day
                # Extract year from the first date-like column
                for col in ("start", "start_timestamp", "date", "datetime"):
                    if col in src.ts.columns:
                        data_start_year = pd.Timestamp(src.ts[col].iloc[0]).year
                        break
                break
        row_offset, self._episode_day_mode = self.data_combinator.get_day_offset(
            self.episode_count, max_days, steps_per_day, data_start_year, self._rng,
        )
        self._episode_date = self._resolve_episode_date(row_offset)

        if variant:
            logger.info("Episode %d, date %s: data variant %s", self.episode_count, self.data_combinator.get_day_date(), variant)

        # Allow external override via reset options
        if options and "row_offset" in options:
            row_offset = int(options["row_offset"])
            self._episode_date = self._resolve_episode_date(row_offset)
            self._episode_day_mode = "manual"

        # ======== Reset state and synchronise datasources/infras ========
        self.iteration = 0
        self.cum_E_kWh = 0.0  # Reset cumulative energy on episode reset
        for sync in self.infras + self.statesources:
            sync.synchronise(self.iteration, row_offset)

        # ======== Initialise state dict with zeros (matching observation space dtypes) ========
        for k, v in self.state.items():
            if isinstance(v, np.ndarray):
                # Initialize with zeros using the correct dtype (float32)
                self.state[k] = np.zeros(self.state[k].shape, dtype=np.float32)
            else:
                logger.debug("Unidentified type: %s", type(v))

        # Set sim_hour for step 0 (start of episode)
        self.state["sim_hour"][0] = np.float32(0.0)

        # ======== Update state from statesources to populate initial observations ========
        # This ensures observations are within bounds after reset
        for ds in self.statesources:
            ds.update_state(states=self.state)

        # Update infrastructure states as well
        for infr in self.infras:
            infr.update_state(self.state)

        # ======= Pass initial state to info ========
        info = {
            "seed": seed,
            "episode_date": self._episode_date,
            "episode_day_mode": self._episode_day_mode,
            "data_variant": variant if variant else None,
        }
        if self.log_full_info:
            info["state"] = {k: np.array(v, copy=True) for k, v in self.state.items()}

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
        """        
        :return: True if episode (a day) elapsed
        :rtype: bool
        """
        return bool(self.iteration >= self.max_iteration)

    def _flat_action_to_dict(self, flat_action: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert flat action array to Dict format for infrastructure use.

        Handles various input formats from RLlib/SyncVectorEnv:
        - 1D array: [a1, a2] -> use directly
        - 2D array with batch dim: [[a1, a2]] -> flatten
        - Nested structures -> flatten
        """
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
            observation (dict): The next observation.
            reward (float): The computed reward signal.
            terminated (bool): True if the episode has ended, else False.
            truncated (bool): False in this environment.
            info (dict): Additional information data.
        """
        # Clip actions and convert flat action to Dict format
        clipped_flat_action = np.clip(action, -1, 1)
        action = self._flat_action_to_dict(clipped_flat_action)

        # 1. Execute all infrastructure actions
        for infr in self.infras:
            infr.exec_action(action, self.state)

        # 2. Advance time: increment iteration, then synchronise all components so
        #    update_state reads the correct (new) row from time-series data.
        #    Previously synchronise was called AFTER update_state, causing exogenous
        #    datasources (price, weather, EV schedule) to lag 2 iterations behind.
        self.iteration += 1
        # Update normalised simulation hour: fraction of episode elapsed
        self.state["sim_hour"][0] = np.float32(self.iteration / self.max_iteration)

        for sync in self.infras + self.statesources:
            sync.synchronise(self.iteration)

        # 3. Update observable states for the new iteration
        for infr in self.infras:
            infr.update_state(self.state)
        for ds in self.statesources:
            ds.update_state(states=self.state)

        # Accumulate net energy consumption from all infrastructures
        # Positive = consumption from grid, Negative = production to grid
        power_breakdown = {
            infra.name: infra.get_electric_consumption(action)
            for infra in self.infras
        }
        total_power_kW = sum(power_breakdown.values())
        energy_kWh = total_power_kW * (self.control_step / 3600)  # kW * hours = kWh
        self.cum_E_kWh += energy_kWh

        # Calculate reward with per-function breakdown
        reward: float = 0
        reward_breakdown = {}
        for rew_f in self.reward_funcs:
            rew_val = float(np.asarray(rew_f.get_reward(action, self.state)).item())
            reward_breakdown[rew_f.name] = rew_val
            reward += rew_val

        # Track executed actions in a rolling history window (oldest first, newest last).
        # Must happen AFTER reward computation so that ActionSmoothnessReward can
        # compare the current action against the previous one stored in history[-1].
        for key in self.action_space_keys:
            obs_key = f"prev_{key}_hist"
            history = self.state[obs_key]
            # Shift rows up (drop oldest) and insert latest action at the end
            history[:-1] = history[1:]
            act_val = action.get(key)
            if act_val is not None:
                history[-1] = np.asarray(act_val, dtype=np.float32)
            else:
                history[-1] = 0.0

        # Check if episode should terminate (iteration already incremented above)
        terminated = self.is_done()
        truncated = False

        info = {
            "action": action,  # Dict format (clipped)
            "reward": reward,
            "reward_breakdown": reward_breakdown,
            "cum_E_kWh": self.cum_E_kWh,  # Cumulative net energy (positive=consumption, negative=production)
            "step_power_kW": total_power_kW,  # Instantaneous net power at this step
            "power_breakdown": power_breakdown,  # Per-infrastructure power (kW)
            "episode_date": self._episode_date,
            **self._get_raw_state_values(),
        }
        if self.log_full_info:
            info["state"] = {k: np.array(v, copy=True) for k, v in self.state.items()}

        return self.state, reward, terminated, truncated, info

    # TODO VP 2026.03.10. : Rework environment that it accepts data series, in state sources things are normalised, but 
    # original values are stored as well in the info dict -- to show real data later in the plots

    def _get_raw_state_values(self) -> dict[str, float]:
        """Collect raw (unnormalised) values from all state sources.

        Each state source may expose ``*_raw`` attributes that hold the
        original physical values before normalisation.  This method
        iterates over all sources, picks up every attribute ending in
        ``_raw``, and also derives ``temp_in_raw`` from the normalised
        indoor temperature using the weather scale factor.
        """
        raw: dict[str, float] = {}
        temp_abs_max = 1.0

        for src in self.statesources + self.infras:
            # Collect any attribute ending in '_raw' exposed by a component
            for attr_name in dir(src):
                if attr_name.endswith("_raw") and not attr_name.startswith("_"):
                    raw[attr_name] = float(getattr(src, attr_name))
            # Cache weather scale factor for temp_in_raw derivation
            if hasattr(src, "temp_abs_max"):
                temp_abs_max = float(src.temp_abs_max)

        # temp_in_norm is a simulated value using the same scale as
        # temp_out_norm (MAX_ABS_SCALING with temp_abs_max)
        temp_in_norm = float(self.state.get("temp_in_norm", 
                                            np.zeros(1, dtype=np.float32),)[0])
        raw["temp_in_raw"] = temp_in_norm * temp_abs_max

        return raw

    def render(self):
        """Render the environment."""
        pass
