import sys
from typing import Any, Dict
from collections import OrderedDict
import logging

import gymnasium as gym
from gymnasium import Space, spaces
from gymnasium.spaces import Dict as SDict
import numpy as np
import random
import pandas as pd

from adv_building_gym.data_combinator import DataCombinator
from adv_building_gym.devices.statesources import StateSource
from adv_building_gym.rewards import RewardFunction
from adv_building_gym.devices.infrastructure import Infrastructure

from adv_building_gym.utils.episode_date import resolve_episode_date
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

    Data contract — observation vs. info
        The environment maintains three distinct data channels:

        1. **Observation space** (``self.state`` / ``observation_space``):
           Policy-relevant values only. Every key is registered via
           ``setup_spaces()`` with bounded ``Box`` limits and normalised to
           small ranges ([-1, 1], [0, 1], or [0, 24] for ``sim_hour``).
           The RL policy sees exactly these keys (after ``FlattenObservations``).

        2. **Component info** (``self._component_info``):
           Shared inter-component dict for raw physical values that other
           components need but the policy must not see (e.g. EV schedule
           kWh/kW capacities, ``_temp_abs_max`` scale factor). Passed as
           the ``info`` argument to ``update_state()`` and ``exec_action()``.

        3. **Step/reset info** (returned to the caller):
           Diagnostics and logging data: reward breakdown, energy totals
           (``cum_E_kWh``, ``step_power_kW``), episode metadata, and
           ``info["raw"]`` containing denormalised physical values collected
           from component attributes ending in ``_raw``.
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
    - render_mode: optional
            Reserved for Gym compatibility.
    - **kwargs:
            Additional environment-specific parameters forwarded or ignored.
    Observation and action spaces
    - The environment builds an observation_space (SDict) and action_space (SDict)
        by aggregating spaces declared by every Infrastructure and DataSource. The
        env also supplies per-key action history windows (``prev_{key}_hist``).
    - The native action_space is a Dict whose keys/bounds are defined by the
        Infrastructure components. External wrappers (FlattenAction + RescaleAction)
        convert the flat [-1, 1] interface expected by RL libraries.
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
        render_mode=None,
        data_combinator: DataCombinator | None = None,
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
            render_mode: Gymnasium render mode (currently unused).
        """

        # Setup warning filters for Ray workers (must be called early)
        setup_warning_filters()

        super(AdvBuildingGym, self).__init__()


        self.iteration = 0
        self.cum_E_kWh = 0.0  # Cumulative net energy in kWh (tracked in info, not observation)
        
        self.episode_count: int = 0
        self.data_combinator = data_combinator if data_combinator is not None else DataCombinator()
        self._rng: np.random.Generator = np.random.default_rng()
        self._episode_date: str = ""
        self._episode_day_mode: str = "none"

        # Build observation and action spaces from components.
        # Time-varying signals are normalised to small ranges; raw scale
        # factors (e.g. temp_abs_max, E_price_max) are included
        # unnormalised so the policy can reconstruct physical units.
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

        # sim_hour: hour of day (0–24) derived from the current step
        # within the episode.  Used by statesource synthetic profiles
        # and SolarPanel for time-of-day logic.
        # Managed directly by the environment (not a StateSource).
        observation_space["sim_hour"] = spaces.Box(
            low=0.0, high=24.0, shape=(1,), dtype=np.float32,
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
            # Tile per-action bounds across the history window
            low_tiled = np.tile(space.low, (self.action_history_length, 1)).reshape(hist_shape)
            high_tiled = np.tile(space.high, (self.action_history_length, 1)).reshape(hist_shape)
            observation_space[obs_key] = spaces.Box(
                low=low_tiled.astype(np.float32),
                high=high_tiled.astype(np.float32),
                shape=hist_shape,
                dtype=np.float32,
            )

        # Assign spaces
        self.observation_space = SDict(observation_space)
        self.state = OrderedDict()
        for state_name, state_space in observation_space.items():
            # Initialize observation state arrays with the same dtype as the declared space
            self.state[state_name] = np.zeros(shape=state_space.shape, dtype=state_space.dtype)

        # Shared dict for inter-component data that is NOT part of the
        # observation space (raw kWh/kW/°C values, EV schedule parameters).
        # Components write/read via the ``info`` argument of update_state()
        # and exec_action().  Persists across steps; included in step info.
        self._component_info: dict = {}

        # Native Dict action space — each key maps to the component's real bounds.
        # External wrappers (FlattenAction + RescaleAction) convert between the
        # flat [-1, 1] interface expected by RL libraries and this Dict space.
        self.action_space = SDict(action_space)

        self.building_props = building_props
        # NOTE VP 2026.02.28. : Simulation time is in seconds
        self.simulation_time = env_config.CONTROL_STEP * env_config.EPISODE_LENGTH
        self.control_step = control_step if control_step is not None else env_config.CONTROL_STEP
        self.max_iteration = env_config.EPISODE_LENGTH

        # When True, step()/reset() include a deep copy of the full named state
        # dict in info["state"]. Expensive in memory — enable for evaluation only.
        # Not a constructor param: set by the evaluation context (eval_runner,
        # Ray evaluation env_config) rather than at construction time.
        self.log_full_info: bool = False

        # Cache the WeatherDataSource for temp_in_raw denormalisation.
        self._weather_source = next(
            (src for src in self.statesources if hasattr(src, "temp_abs_max")),
            None,
        )

        logger.debug("AdvBuildingGym created!")
        logger.debug("  Objectives: %s", [rew.name for rew in rewards])
        logger.debug("  Actions: %s", [infr.name for infr in infras])
        logger.debug("  States: %s", [ds.name for ds in statesources])

    def set_reward_funcs(self, rewards: list[RewardFunction]) -> None:
        """Hot-swap the active reward functions.

        Called by the reward_switch callback to change which objectives
        the environment evaluates during ``step()``.

        Args:
            rewards: New list of RewardFunction instances.
        """
        self.reward_funcs = rewards
        logger.debug(
            "Reward functions updated: %s",
            [r.name for r in rewards],
        )

    def set_infras(
        self,
        infras: list[Infrastructure],
        building_props: BuildingProps | None = None,
    ) -> None:
        """Hot-swap infrastructure components.

        Called by the infra_schedule callback to change which
        infrastructure configuration the environment uses.  Only
        parameters differ -- names and types must match.  Spaces are
        invariant (all infra ``setup_spaces`` use normalised bounds).

        When *building_props* is provided, also updates
        ``self.building_props`` and propagates K/mC to statesources
        that depend on them (e.g. BuildingHeatLoss).

        Args:
            infras: New Infrastructure instances.  Must have the same
                names in the same order as the current infras.
            building_props: Updated building thermal properties.  When
                ``None`` the existing props are kept.
        """
        old_names = [i.name for i in self.infras]
        new_names = [i.name for i in infras]
        if old_names != new_names:
            raise ValueError(
                f"Infrastructure names must match. "
                f"Old: {old_names}, New: {new_names}"
            )
        self.infras = infras

        # Propagate building_props to dependent statesources
        if building_props is not None:
            self.building_props = building_props
            for src in self.statesources:
                if hasattr(src, "K") and hasattr(src, "mC"):
                    src.K = building_props.K
                    src.mC = building_props.mC

        logger.debug(
            "Infrastructure swapped: %s",
            [i.name for i in infras],
        )

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

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        # ======= Seed =======
        super().reset(seed=seed)

        # Only reseed when an explicit seed is provided (typically the first
        # reset per env runner).  Subsequent resets (seed=None) must NOT
        # recreate the RNG — doing so would destroy the deterministic
        # sequence and make episodes non-reproducible across runs.
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self._rng = np.random.default_rng(seed)

        # ======== Data variant selection logic ========
        # During distributed training the D1 callback pushes variants to all
        # runners.  For single-env usage (evaluation, local testing) the env
        # selects the variant itself from its DataCombinator.
        self.episode_count += 1
        variant = None

        # Approach C: external override via reset(options={"data_variant": {...}})
        if options and "data_variant" in options:
            variant = options["data_variant"]
            self.apply_data_variant(variant)
        elif self.data_combinator.variants:
            # No external push — select variant from the combinator
            variant = self.data_combinator.get_variant(self.episode_count, self._rng)
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
        self._episode_date = resolve_episode_date(self.statesources, row_offset, self.control_step)

        if variant:
            logger.info("Episode %d, date %s: data variant %s", self.episode_count, self.data_combinator.get_day_date(), variant)

        # Allow external override via reset options
        if options and "row_offset" in options:
            row_offset = int(options["row_offset"])
            self._episode_date = resolve_episode_date(self.statesources, row_offset, self.control_step)
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

        # Set sim_hour for step 0 (midnight start of day)
        self.state["sim_hour"][0] = np.float32(0.0)

        # ======== Update state from statesources to populate initial observations ========
        # reset_state() calls update_state() by default; subclasses (e.g.
        # InsideTemperature) override it for reset-specific initialisation.
        self._component_info.clear()
        for ds in self.statesources:
            ds.reset(states=self.state, info=self._component_info)

        # Update infrastructure states as well
        for infr in self.infras:
            infr.reset(self.state, info=self._component_info)

        # Reset info — episode metadata and raw values for logging.
        # None of these keys are part of the observation space.
        info = {
            "seed": seed,
            "episode_date": self._episode_date,
            "episode_day_mode": self._episode_day_mode,
            "data_variant": variant if variant else None,
            "raw": self._get_raw_state_values(),  # Denormalised physical values (°C, €, etc.)
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
            ds.update_state(states=state, info=self._component_info)
        return state

    def is_done(self) -> bool:
        """        
        :return: True if episode (a day) elapsed
        :rtype: bool
        """
        return bool(self.iteration >= self.max_iteration)

    def step(self, action):
        """Execute a single control step by applying *action*.

        Args:
            action (dict[str, np.ndarray]): Dict action mapping component
                keys (e.g. ``HP_action``, ``battery_action``) to arrays
                whose bounds match the Dict action space declared by each
                Infrastructure. 
                External wrappers (FlattenAction + RescaleAction) handle the 
                conversion from the flat [-1, 1] array produced by the RL policy.

        Returns:
            observation (dict): The next observation.
            reward (float): The computed reward signal.
            terminated (bool): True if the episode has ended, else False.
            truncated (bool): False in this environment.
            info (dict): Additional information data.
        """

        # 1. Execute all infrastructure actions
        for infr in self.infras:
            infr.exec_action(action, self.state, info=self._component_info)

        # 2. Advance time: increment iteration, then synchronise all components so
        #    update_state reads the correct (new) row from time-series data.
        #    Previously synchronise was called AFTER update_state, causing exogenous
        #    datasources (price, weather, EV schedule) to lag 2 iterations behind.
        self.iteration += 1
        # Update simulation hour: actual hour of day (0–24)
        # control_step is in seconds; convert elapsed time to hours
        self.state["sim_hour"][0] = np.float32(
            (self.iteration * self.control_step) / 3600.0
        )

        for sync in self.infras + self.statesources:
            sync.synchronise(self.iteration)

        # 3. Update observable states for the new iteration
        # Reset directional power bound accumulators before infras publish.
        self._component_info["max_consumption_kW"] = 0.0
        self._component_info["max_export_kW"] = 0.0
        for infr in self.infras:
            infr.update_state(self.state, info=self._component_info)
        for ds in self.statesources:
            ds.update_state(states=self.state, info=self._component_info)

        # Accumulate net energy consumption from all infrastructures
        # Positive = consumption from grid, Negative = production to grid
        power_breakdown = {
            infra.name: infra.get_electric_consumption(action)
            for infra in self.infras
        }
        total_power_kW = sum(power_breakdown.values())
        energy_kWh = total_power_kW * (self.control_step / 3600)  # kW * hours = kWh
        self.cum_E_kWh += energy_kWh

        # Publish power data into component info so reward functions can
        # access it without holding infrastructure references.
        self._component_info["power_breakdown"] = power_breakdown
        self._component_info["net_power_kW"] = total_power_kW
        self._component_info["penalisable_power_kW"] = sum(
            infra.get_penalisable_consumption(action, self.state)
            for infra in self.infras
        )

        # Calculate reward with per-function breakdown
        reward: float = 0
        reward_breakdown = {}
        max_reward_step: float = 0
        for rew_f in self.reward_funcs:
            rew_val, rew_max = rew_f.get_reward(action, self.state, info=self._component_info)
            rew_val = float(np.asarray(rew_val).item())
            reward_breakdown[rew_f.name] = rew_val
            reward += rew_val
            max_reward_step += rew_max

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

        # Guard against NaN/Inf in state — these propagate through the neural
        # network and crash the action distribution (std = NaN → RuntimeError).
        # Replace bad values with 0 so training can continue.
        for key, val in self.state.items():
            if isinstance(val, np.ndarray) and not np.all(np.isfinite(val)):
                logger.error("NaN/Inf in state['%s']: %s (episode %d, step %d) — replaced with 0",
                            key, val, self.episode_count, self.iteration)
                self.state[key] = np.where(np.isfinite(val), val, np.zeros_like(val))

        # Check if episode should terminate (iteration already incremented above)
        terminated = self.is_done()
        truncated = False

        # Step info — diagnostics and raw values for logging/evaluation.
        # None of these keys are part of the observation space.
        info = {
            "action": action,  # Dict format (clipped)
            "reward": reward,
            "reward_breakdown": reward_breakdown,
            "max_reward_step": max_reward_step,  # Step-wise max achievable reward
            "cum_E_kWh": self.cum_E_kWh,  # Cumulative net energy (positive=consumption, negative=production)
            "step_power_kW": total_power_kW,  # Instantaneous net power at this step
            "power_breakdown": power_breakdown,  # Per-infrastructure power (kW)
            "raw": self._get_raw_state_values(),  # Denormalised physical values (°C, €, etc.)
        }
        if self.log_full_info:
            info["state"] = {k: np.array(v, copy=True) for k, v in self.state.items()}

        return self.state, reward, terminated, truncated, info


# TODO VP 2026.03.23. : encourage exploration more
# TODO VP 2026.03.23. : Use more history as state input -- from the 6h , 5h, 4h, 3h, 2h and 1h ago -- and the last 30min: each step from here
# To this, implement a history collector -- collect specified timesteps from the past, according to the current simulation time: t-6h, t-4h, etc... -- can be generalised, it only needs the spec, the time series and the current simulation time.
# Maybe not only for states, but for trajectory as well -- so that complete (s, a, r, s') tuples caputured from the past...
# TODO VP 2026.03.23. : Eval script -- Plot all (reward, cum_E_usage) eval curves together -- with avg and variance
# TODO VP 2026.03.23. : Add standalone input and output heads for the policy NN, fix the core policy NN -- investigate this option

    def _get_raw_state_values(self) -> dict[str, float]:
        """Collect raw (unnormalised) physical values from all components.

        Each component reports its own raw values via ``get_raw_values()``.
        Additionally derives ``temp_in_raw`` by denormalising the simulated
        indoor temperature.
        """
        raw: dict[str, float] = {}
        for src in self.statesources + self.infras:
            raw.update(src.get_raw_values())

        # temp_in_raw: denormalise simulated indoor temperature
        temp_abs_max = (
            float(self._weather_source.temp_abs_max)
            if self._weather_source is not None
            else 1.0
        )
        temp_in_norm = float(
            self.state.get("temp_in_norm", np.zeros(1, dtype=np.float32))[0]
        )
        raw["temp_in_raw"] = temp_in_norm * temp_abs_max

        return raw

    def render(self):
        """Render the environment."""
        pass
