"""
Heat Pump Building Dynamics Environment
======================================
This simulation framework models the thermal behavior of a single-zone
building model with a reversible heat pump.
It uses a lumped-capacitance (1R1C) energy balance and includes a
dynamic electricity price signal.
The environment offers a Gymnasium-compatible interface for
reinforcement learning experiments.

Key Features
------------
* **Physics‑based building core** – explicit‑Euler integration of a first‑order
  resistance–capacitance network capturing transmission and ventilation losses.
* **Bidirectional heat pump** – user‑defined maximum thermal power for heating
  (positive action) or cooling (negative action).
* **Dynamic energy tariff** – plug‑and‑play support for aWATTar/EPEX Spot
  day‑ahead prices (CSV) or a fallback time‑of‑use profile.
* **Rich observation variants** – modular state vectors that combine thermal,
  temporal and economic information for ablation studies.
* **Baseline controllers** – PI/PID, fuzzy logic and MPC implementations for
  benchmarking against RL agents.

  Classes:
--------
- Building
- BaseBuildingGym
- FuzzyController
- MPCController
- PIController
- PIDController

Author: Gökhan Demirel <goekhan.demirel@kit.edu>
"""

import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

# from llec_building_gym.controllers import (
#    FuzzyController,
#    MPCController,
#    PIController,
#    PIDController
# )
from adv_building_gym.utils.temporal_features import TemporalFeatureBuffer
import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True
)
logger = logging.getLogger(__name__)
logger.debug(f"sys.argv: {sys.argv}")


# Building class
class Building:
    """Lumped‑capacitance single‑zone building model with a reversible
    heat‑pump and dynamic tariff.

    Parameters
    ----------
    mC : float, optional
        Effective thermal capacity of the zone in joules per kelvin (J/K).
        Defaults to ``300``.
    K : float, optional
        Overall heat‑transfer coefficient in watts per kelvin (W/K).  Higher
        values imply a leakier envelope.  Defaults to ``20``.
    Q_HP_Max : float, optional
        Maximum absolute thermal power that the heat pump can supply (W).
        Positive values correspond to heating.  Defaults to ``1500``.
    simulation_time : int, optional
        Episode length in seconds.  Defaults to 24 h (``24*60*60``).
    control_step : int, optional
        Discrete control/physics step in seconds.  Defaults to 300 s (5 min).
    schedule_type : str, optional
        Outdoor‑temperature profile.  One of ``{"path", "24-hour",
        "12-hour", "simple"}``.  Defaults to ``"24-hour"``.
    training : bool, optional
        Whether the environment is in training mode (random day sampling) or
        evaluation mode (deterministic).  Defaults to ``None`` which means the
        flag is ignored.
    train_ratio : float, optional
        Fraction of days to allocate to the training split when using real
        price data.  Defaults to ``0.8``.
    outdoor_temperature_path : str or None, optional
        CSV file containing an external temperature time series.  If ``None`` a
        synthetic sinusoid is used.  Defaults to ``None``.
    energy_price_path : str or None, optional
        CSV file with a column ``price_normalized`` providing pre‑scaled energy
        prices in the range [0, 1].  If ``None`` a time‑of‑use tariff is
        generated.  Defaults to ``None``.
    """

    def __init__(
        self,
        mC=300,
        K=20,
        Q_HP_Max=1500,
        simulation_time=24 * 60 * 60,
        control_step=300,
        schedule_type="24-hour",
        training=None,
        train_ratio=0.8,
        outdoor_temperature_path=None,
        energy_price_path=None,
    ):
        """
        Initialize the building model parameters and simulation constraints.
        Parameters:
            mC (float): Effective thermal mass (J/K). Defaults to 300.
            K (float): Heat transfer coefficient (W/K). Defaults to 20.
            Q_HP_Max (float): Max heat pump power (heating or cooling) in W.
                              Defaults to 1500.
            simulation_time (int): Total simulation time in seconds.
                                   Defaults to 24*60*60 (one day).
            control_step (int): Simulation/control time step in seconds.
                                Defaults to 300 (5 minutes).
            schedule_type (str): Schedule type for outdoor temperature profile.
                                 (e.g., "path","24-hour", "simple").
                                 Defaults to "24-hour".
        """
        self.simulation_time = simulation_time
        self.timestep = control_step
        self.max_iteration = int(self.simulation_time / self.timestep)
        self.Q_HP_Max = Q_HP_Max
        self.mC = mC
        self.K = K

        # States
        self.iteration = 0
        self.T_in = 20  # Indoor temperature (degC)
        self.T_set = 25  # Setpoint temperature (degC)
        self.T_out = None  # Deterministic outdoor temperature profile (degC)
        self.T_out_measurement = None  # Measured outdoor temperature profile
        self.wiener_noise = None

        # Energy tariff
        self.energy_price = 0.1  # Default low tariff at the start
        self.schedule_type = schedule_type
        self.energy_price_path = energy_price_path

        # Outdoor temperature path
        self.outdoor_temperature_path = outdoor_temperature_path

        self.training = training
        self.train_ratio = train_ratio

        self.train_days = None
        self.eval_days = None
        self.price_vector = None
        self.test_counter = 0  # für Eval-Mode

        # Energy price preparation, if path given
        if self.energy_price_path is not None:
            self._load_energy_price()

        # Create setpoint schedule (default: BA_RES)
        self.T_set_profile = self._generate_setpoint_profile(profile_type="BA_RES")
        self.class_params = {
            "Very cold": {"A": 4.932, "phi": 0.184, "C": -1.037},
            "Cold": {"A": 5.137, "phi": 0.137, "C": 3.876},
            "Normal": {"A": 5.590, "phi": 0.061, "C": 14.287},
            "Hot": {"A": 5.945, "phi": -0.094, "C": 25.412},
        }

    def reset(self, T_in=20, seed=None):
        """
        Reset the building simulation to initial conditions to
        start a new simulation episode.
        Parameters:
            T_in (float): Initial indoor temperature (degC).
                          Defaults to 20.
            seed (int or None): Random seed for reproducibility.
                                Defaults to 58.
        """
        self.iteration = 0
        self.T_in = T_in
        # Seed for reproducibility
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()
        # Generate schedule with noise
        self._create_schedule(add_noise=True, seed=seed)
        # Energy price preparation, if path given
        if self.energy_price_path is not None:
            self._load_energy_price()
        # Select start index depending on training or test mode
        steps_per_day = int(23 * 3600 // self.timestep)
        if self.energy_price_path is not None:
            if self.training:
                day = self._rng.choice(self.train_days)
                self.start = day * steps_per_day
            else:
                day = self.eval_days[self.test_counter]
                self.start = day * steps_per_day
                self.test_counter += 1
                self.test_counter = self.test_counter % len(self.eval_days)
        self._update_energy_price()

    def _generate_setpoint_profile(self, profile_type="BA_RES"):
        """
        Creates a daily setpoint temperature profile repeated to match
        the simulation duration.
        Args:
            profile_type (str): Profile type, one of
                                ["BA_RES", "HOME_OFFICE", "DOE_COMM"].

        Returns:
            np.ndarray: Array of setpoints per timestep.
        """
        setpoints = {
            "BA_RES": {
                **{h: 21.7 for h in range(6, 9)},
                **{h: 18.3 for h in range(9, 17)},
                **{h: 21.7 for h in range(17, 24)},
                **{h: 18.3 for h in range(0, 6)},
            },
            "HOME_OFFICE": {
                **{h: 21.7 for h in range(7, 22)},
                **{h: 18.3 for h in list(range(22, 24)) + list(range(0, 7))},
            },
            "DOE_COMM": {
                **{h: 21.0 for h in range(6, 18)},
                **{h: 15.6 for h in list(range(18, 24)) + list(range(0, 6))},
            },
        }
        hourly_profile = setpoints.get(profile_type, setpoints["BA_RES"])
        steps_per_hour = int(3600 / self.timestep)
        profile = []
        for h in range(24):
            profile.extend([hourly_profile[h]] * steps_per_hour)
        repeats = int(np.ceil(self.simulation_time / 3600 / 24))
        full_profile = np.tile(profile, repeats)[: self.max_iteration]
        return np.array(full_profile)

    def _generate_reflected_wiener_noise(
        self, t, noise_std, lower_bound=-5, upper_bound=5, seed=None
    ):
        """
        Generate a reflected Wiener process(Brownian motion with boundaries),
        ensuring the process stays within specified upper and lower bounds
        by reflecting any value that exceeds them.

        Parameters:
            t (ndarray): Discrete time vector. (s)
            noise_std (float): Scaling factor for the random increments.
            lower_bound (float): Lower boundary for allowable
                                 value of the noise.
            upper_bound (float): Upper boundary for allowable
                                 value of the noise.

        Returns:
            wiener_noise (ndarray): Reflected Wiener noise over time.
        """
        rng = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )

        noise_increments = noise_std * rng.normal(0, np.sqrt(self.timestep), size=t.shape)
        wiener_noise = np.zeros_like(t, dtype=float)

        for i in range(1, len(t)):
            new_val = wiener_noise[i - 1] + noise_increments[i]
            # Reflect if out of bounds
            while new_val > upper_bound or new_val < lower_bound:
                if new_val > upper_bound:
                    overshoot = new_val - upper_bound
                    new_val = upper_bound - overshoot
                elif new_val < lower_bound:
                    overshoot = lower_bound - new_val
                    new_val = lower_bound + overshoot
            wiener_noise[i] = new_val
        return wiener_noise

    def _create_schedule(self, noise_std=0.1, add_noise=True, seed=None):
        """
        Create a temperature schedule for T_out, optionally adding Wiener noise
        for a more realistic outdoor temperature measurement.
        Different schedule_types can be used: 'simple' or '24-hour'.
        Supported schedule_types:
            - 'simple': static two-step profile (28degC/32degC)
            - '24-hour': sinusoidal profile with full-day periodicity
            - 'None': CSV-based profile from outdoor_temperature_path
        Parameters:
            noise_std (float, optional): Std. deviation factor for the Wiener process.
                                         Default is 0.1.
            add_noise (bool, optional): If True, adds reflected noise.
                                        Default is True.
            seed (int, optional): Random seed for reproducibility.
        """
        # Generate simulation time steps
        t = np.arange(0, self.simulation_time, self.timestep)
        if self.schedule_type == "simple":
            T_deterministic = np.empty(len(t))
            half = len(t) // 2
            T_deterministic[:half] = 28
            T_deterministic[half:] = 32
            logger.info("Using 'simple' two-step temperature profile.")
        elif self.schedule_type == "24-hour":
            # A sinusoidal profile for a day (24-hour simulation) periodicity
            T_min, T_max = 14, 28
            T_mean = (T_min + T_max) / 2
            T_amp = (T_max - T_min) / 2
            seconds_per_day = 24 * 60 * 60
            # ≈ 14:00 peak temperature
            peak_at = 14 * 60 * 60
            phase_shift = 2 * np.pi * (peak_at / seconds_per_day) - np.pi / 2
            T_deterministic = T_mean + T_amp * np.sin(
                2 * np.pi * (t / seconds_per_day) - phase_shift
            )
            logger.info(
                f"[EnvID:{id(self)}] schedule=24-hour | "
                f"Tmin={T_min:.1f}degC  Tmax={T_max:.1f}degC  phase={phase_shift:.3f}  seed={seed}"
            )
        else:
            # Various deterministic temperature profiles
            if self.outdoor_temperature_path is not None:
                # Load outdoor temperature data from CSV file
                outdoor_df = pd.read_csv(self.outdoor_temperature_path)
                # Set up RNG with optional seed
                self._rng = (
                    np.random.default_rng(seed)
                    if seed is not None
                    else np.random.default_rng()
                )
                # Flatten temperature data
                outdoor_values = outdoor_df.values.flatten()
                # Ensure data covers the full simulation period
                if len(outdoor_values) < len(t):
                    raise ValueError(
                        "Outdoor temperature data is shorter than simulation time."
                    )
                start_idx = self._rng.integers(0, len(outdoor_values) - len(t) + 1)
                T_deterministic = outdoor_values[start_idx : start_idx + len(t)]
                # Check for NaNs and replace if necessary
                if np.isnan(T_deterministic).any():
                    nan_indices = np.where(np.isnan(T_deterministic))[0]
                    logger.warning("NaNs found in T_deterministic — replacing with 0.0")
                    logger.warning(
                        f"Start index: {start_idx}, NaN indices: {nan_indices.tolist()}"
                    )
                    for idx in nan_indices:
                        logger.debug(f"NaN at idx {idx}: value={T_deterministic[idx]}")
                    T_deterministic = np.nan_to_num(T_deterministic, nan=0.0)
                logger.info(
                    f"[EnvID: {id(self)}] Loading outdoor temperature profile from CSV file: '{self.outdoor_temperature_path}' (Seed: {seed})"
                )
                logger.debug(f"CSV preview:\n{outdoor_df.head()}")
            else:
                # Class-based synthetic profiles (random class per episode)
                # built-in defaults
                if self.class_params is None:
                    self.class_params = {
                        "Very cold": {"A": 4.932, "phi": 0.184, "C": -1.037},
                        "Cold": {"A": 5.137, "phi": 0.137, "C": 3.876},
                        "Normal": {"A": 5.590, "phi": 0.061, "C": 14.287},
                        "Hot": {"A": 5.945, "phi": -0.094, "C": 25.412},
                    }
                # make sure RNG exists
                if not hasattr(self, "_rng"):
                    self._rng = np.random.default_rng(seed)
                self.current_class = self._rng.choice(list(self.class_params))
                p = self.class_params[self.current_class]  # {"A":…, "phi":…, "C":…}
                # map to same variable names
                T_min, T_max = p["C"] - p["A"], p["C"] + p["A"]
                T_mean, T_amp = p["C"], p["A"]
                seconds_per_day = 24 * 60 * 60
                phase_shift = p["phi"]
                T_deterministic = T_mean + T_amp * np.sin(
                    2 * np.pi * (t / seconds_per_day) - phase_shift
                )
                logger.info(
                    f"[EnvID:{id(self)}] schedule=synthetic-temp-class "
                    f"class={self.current_class} | "
                    f"Tmin={T_min:.1f}degC  Tmax={T_max:.1f}degC  "
                    f"φ={phase_shift:.3f}  seed={seed}"
                )

        if add_noise:
            wiener_noise = self._generate_reflected_wiener_noise(
                t, noise_std, seed=seed
            )
            T_noisy = T_deterministic + wiener_noise
        else:
            wiener_noise = np.zeros_like(t)
            T_noisy = T_deterministic.copy()

        # Store as instance attributes
        self.wiener_noise = wiener_noise
        self.T_out = T_deterministic
        self.T_out_measurement = T_noisy
        logger.debug(f"First 5 values T_out (deterministic): {T_deterministic[:5]}")
        logger.debug(f"First 5 values Wiener noise: {wiener_noise[:5]}")
        logger.debug(f"First 5 values T_out_measurement (noisy): {T_noisy[:5]}")

    def _load_energy_price(self):
        """
        Load energy price data and split it into training and evaluation days.
        Uses a fixed seed to ensure reproducibility.
        """
        # Read the full energy price time series
        price_df = pd.read_csv(self.energy_price_path)
        # Keep entire DataFrame
        self.full_price_df = price_df

        steps_per_day = int(23 * 3600 // self.timestep)
        total_days = len(price_df) // steps_per_day

        # Generate a reproducible random split of days
        days = np.arange(total_days)

        # Reproducible random split of training/evaluation days
        # (sets the global NumPy random seed to ensure deterministic behavior)
        np.random.seed(42)
        np.random.shuffle(days)

        num_train_days = int(self.train_ratio * total_days)
        self.train_days = days[:num_train_days]
        self.eval_days = days[num_train_days:]
        # Debug logging
        logger.debug(f"Loaded {total_days} days from energy price data.")
        logger.debug(
            f"Training days: {len(self.train_days)}, Evaluation days: {len(self.eval_days)}"
        )

    def _update_energy_price(self):
        """
        Update the current energy price based on the simulation time or a fixed schedule.
        If external CSV data is available, the price is taken from the normalized time series.
        Otherwise, a default time-of-use tariff is applied.
        """
        # Calculate the current simulation time in seconds and convert to hours
        current_time = self.iteration * self.timestep
        current_hour = current_time / 3600.0

        if self.energy_price_path is not None:
            # Use value from normalized CSV time series
            if self.start + self.iteration < len(self.full_price_df):
                self.energy_price = self.full_price_df.iloc[
                    self.start + self.iteration
                ]["price_normalized"]
            else:
                # Fallback to the last available value
                self.energy_price = self.full_price_df.iloc[-1]["price_normalized"]
        else:
            # Apply a simple time-of-use tariff if no CSV data is provided
            if current_hour < 4:
                self.energy_price = 0.25
            elif current_hour < 8:
                self.energy_price = 0.50
            else:
                self.energy_price = 0.75

    def update_Tin(self, action):
        """
        Update the indoor temperature by one time step, accounting for
        heat transfer and heat pump action.

        Heat Pump Influence:
        Positive action corresponds to heating, negative to cooling. The maximum
        magnitude of thermal power is given by Q_HP_Max.
        The building's thermal evolution is given by:

            T_in(t+Δt) = T_in(t) + (Δt / mC) * [ -K * (T_in(t) - T_out(t)) + Q_HP(t) ]

        where Q_HP(t) = action * Q_HP_Max.

        Parameters:
            action (float): Control action in [-1, 1].
                            Positive => heating, Negative => cooling.
        """
        Q_HP = action * self.Q_HP_Max
        # Scale factor for unit consistency (0.001)
        dT = (
            0.001
            * (self.timestep / self.mC)
            * (self.K * (self.T_in - self.T_out[self.iteration]) + Q_HP)
        )
        self.T_in -= dT

        # Move the simulation forward by one timestep
        self.iteration += 1

    def is_done(self):
        """
        Check if the simulation episode has reached its maximum duration.

        Returns:
            bool: True if iteration >= max_iteration, else False.
        """
        return self.iteration >= self.max_iteration


# Gymnasium environment
class BaseBuildingGym(gym.Env):
    """
    A Gymnasium environment for controlling a building with a heat pump system.

    Observations:
    -------------
    Depending on the reward_mode, the observation space can include:
      - temperature_deviation (the difference between indoor temperature and setpoint),
      - energy_price (normalized to [0,1]).

    Action:
    -------
    A continuous value in [-1, 1], scaled internally by Q_HP_Max:
      - Negative action => cooling,
      - Positive action => heating.

    Reward:
    -------
    Several reward modes are supported:
      1. "temperature": A temperature-based exponential penalty, emphasizing comfort in terms of minimizing |T_in - T_set|.
      2. "combined": A weighted combination of temperature and economic objectives.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        mC=300,
        K=20,
        Q_HP_Max=1500,
        simulation_time=24 * 60 * 60,
        control_step=300,
        schedule_type=None,
        # or combined
        reward_mode="temperature",
        temperature_weight=1.0,
        economic_weight=1.0,
        render_mode=None,
        training=True,
        train_ratio=0.8,
        outdoor_temperature_path=None,
        energy_price_path=None,
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
        super(BaseBuildingGym, self).__init__()

        self.EPS = 1e-9  # small positive epsilon to avoid division-by-zero
        # Coefficients of Performance (COP)
        # Setting default_cop=1.0 preserves the original baseline behavior (thermal ~= electric).
        # Typical air-source HPs (mild European climate): COP_HEAT≈3.0, COP_COOL≈2.5.
        # Both parameters can be overridden via kwargs, e.g., cop_heat=3.0, cop_cool=2.5.
        default_cop = 1.0
        self.COP_HEAT = float(kwargs.get("cop_heat", default_cop))  # [-] heating COP
        self.COP_COOL = float(kwargs.get("cop_cool", default_cop))  # [-] cooling COP
        if self.COP_HEAT <= 0 or self.COP_COOL <= 0:
            raise ValueError("cop_heat and cop_cool must be positive.")

        self.obs_variant = kwargs.get("obs_variant", "T01")  # move up
        self.prediction_horizon = (
            12 * 5 * 6
        )  # Number of steps to look ahead for forecasted values

        # Instantiate the building model
        self.building = Building(
            mC=mC,
            K=K,
            Q_HP_Max=Q_HP_Max,
            simulation_time=simulation_time,
            control_step=control_step,
            schedule_type=schedule_type,
            outdoor_temperature_path=outdoor_temperature_path,
            energy_price_path=energy_price_path,
            training=training,
            train_ratio=train_ratio,
        )

        self.temporal_features = TemporalFeatureBuffer(window_size=12 * 5 * 6)

        # Reward settings
        self.reward_mode = reward_mode
        self.temperature_weight = temperature_weight
        self.economic_weight = economic_weight
        # Track total energy used so far (for observation purposes)
        self.cumulative_energy_Wh = 0.0
        # Dynamically compute observation space dimension
        dummy_reset_obs = self.reset()[0]
        obs_dim = dummy_reset_obs.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.current_step = 0
        self.prev_action = 0

        self.outdoor_temperature_path = outdoor_temperature_path
        self.energy_price_path = energy_price_path
        self.training = training
        self.train_ratio = train_ratio

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.

        Randomizes the initial indoor temperature between 20degC and 40degC, and
        resets building states. Returns the initial observation.
        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to 58.
            options (dict, optional): Additional environment reset options
                (not used here). Defaults to None.
        Returns:
            tuple:
                - observation (np.array): Initial observation vector.
                - info (dict): Diagnostic information, such as the random seed.
        """
        if seed is None:
            seed = np.random.randint(0, 10000)  # global RNG
        super().reset(seed=seed, options=options)
        # Randomize initial indoor temperature
        initial_Tin = np.random.randint(low=20, high=40)  # global RNG
        self.building.reset(T_in=initial_Tin, seed=seed)
        self.current_step = 0
        self.prev_action = 0
        obs = self._get_observation()
        self.cumulative_energy_Wh = 0.0
        return obs, {"seed": seed}

    def step(self, action):
        """
        Execute one control step in the environment by applying the selected
        action to the heat pump.

        Calculates:
          - The new indoor temperature
          - The immediate reward based on the chosen reward_mode
          - Termination conditions

        Args:
            action (np.array): 1D array with the control action in [-1, 1].

        Returns:
            observation (np.array): The next observation.
            reward (float): The computed reward signal.
            terminated (bool): True if the episode has ended, else False.
            truncated (bool): False in this environment.
            info (dict): Additional information data.
        """
        # Store the current action
        self.prev_action = action[0]

        cop = self.COP_HEAT if action[0] >= 0 else self.COP_COOL
        cop_eff = max(cop, self.EPS)
        P_HP_el = abs(action[0]) * (self.building.Q_HP_Max / cop_eff)
        dt_h = self.building.timestep / 3600.0
        self.cumulative_energy_Wh += P_HP_el * dt_h
        # Ensure action is within the valid range
        self.building.update_Tin(action=action[0])

        # Update the building's energy price and set point
        self.building._update_energy_price()
        # Update Temp. Setpoints
        self._update_set_point()
        obs = self._get_observation()

        # Calculate the temperature deviation
        temp_deviation = self.building.T_in - self.building.T_set
        self.temporal_features.append(delta=temp_deviation, action=action[0])

        # 1.Temperature-based reward (comfort): exp(-|temp_deviation|)
        reward_temperature = np.exp(-abs(temp_deviation))
        reward_temperature_norm = reward_temperature  # Already in [0, 1] by design

        # 2.Economic-based reward: negative of energy cost (cost = price × power consumption)
        max_price = self.building.full_price_df["price_normalized"].max()
        reward_economic = -self.building.energy_price * abs(
            action[0]
        )  # Negative cost as reward
        reward_economic_norm = (
            -self.building.energy_price * abs(action[0]) / max_price
        )  # ∈ [-1, 0]

        # Compute scalar reward based on the selected reward mode
        if self.reward_mode == "temperature":
            # Single-objective reward: comfort-driven (temperature deviation minimized)
            reward = reward_temperature_norm
        elif self.reward_mode == "combined":
            # Multi-objective reward: weighted sum of normalized comfort and cost terms
            reward = (
                self.temperature_weight * reward_temperature_norm
                + self.economic_weight * reward_economic_norm
            )
        else:
            raise ValueError(f"Invalid reward mode: {self.reward_mode}")
        logger.debug(
            "act %.2f | price %.3f | cost %.1f | econ_r(norm) %.3f",
            action[0],
            self.building.energy_price,
            -reward_economic,
            reward_economic_norm,
        )
        # Check if episode should terminate
        terminated = self.building.is_done()
        truncated = False
        self.current_step += 1

        info = {
            "temp_deviation": temp_deviation,
            "action": action,
            "T_out": self.building.T_out[self.building.iteration - 1],
            "Q_HP_Max": self.building.Q_HP_Max,
            "controlled_Q_HP": action[0] * self.building.Q_HP_Max,
            "P_HP_el": P_HP_el,
            "E_HP_el_Wh": self.cumulative_energy_Wh,
            "cop_used": cop_eff,
            "reward": reward,
            "reward_temperature": reward_temperature,
            "reward_economic": reward_economic,
            "reward_temperature_norm": reward_temperature_norm,
            "reward_economic_norm": reward_economic_norm,
            "history_temp_deviations": self.temporal_features.get_padded_deltas(),
        }

        return obs, reward, terminated, truncated, info

    def _update_set_point(self) -> None:
        """
        Updates the building's current setpoint T_set from the profile.
        """
        idx = self.building.iteration
        if hasattr(self.building, "T_set_profile") and idx < len(self.building.T_set_profile):
            self.building.T_set = self.building.T_set_profile[idx]

    def _get_future_energy_prices(self):
        """
        Returns a list of forecasted normalized energy prices for the next
        `self.prediction_horizon` control steps.

        The forecast horizon is defined by `self.prediction_horizon`, which
        corresponds to a total time span of:

        prediction_horizon x control_step seconds.

        For example, with a prediction_horizon of 3 and a control_step of 300 seconds
        (5 minutes), the agent receives energy price forecasts for the next 15 minutes.

        Forecasts are taken from real market data if available, or approximated
        using a default time-of-use pricing scheme. A small Gaussian noise is added
        to each price to simulate realistic prediction uncertainty.

        Returns:
            List[float]: Forecasted energy prices for upcoming steps, clipped to [0.0, 1.0].
        """
        future_prices = []
        for i in range(1, self.prediction_horizon + 1):
            future_idx = self.building.iteration + i
            if self.building.energy_price_path is not None:
                # If real energy price data is available, use it for future predictions
                if self.building.start + future_idx < len(self.building.full_price_df):
                    future_price = self.building.full_price_df.iloc[
                        self.building.start + future_idx
                    ]["price_normalized"]
                else:
                    # If future index exceeds dataset, use the last available price
                    future_price = self.building.full_price_df.iloc[-1][
                        "price_normalized"
                    ]
            else:
                # If no external price data is provided, fallback to default time-of-day pricing
                future_seconds = (self.building.iteration + i) * self.building.timestep
                future_hour = (future_seconds % (24 * 3600)) / 3600.0
                if future_hour < 4:
                    future_price = 0.25
                elif future_hour < 8:
                    future_price = 0.50
                else:
                    future_price = 0.75
            # Add small Gaussian noise to future prices for realism
            noisy_price = future_price + self.building._rng.normal(0, 0.01)
            noisy_price = np.clip(noisy_price, 0.0, 1.0)
            future_prices.append(noisy_price)
        return future_prices

    def _get_observation(self):
        """
        Constructs the observation vector used by the reinforcement learning agent,
        depending on the selected `obs_variant`. This modular design supports
        ablation studies to analyze the impact of individual observation components
        on agent behavior.

        Observation Variants:
        ---------------------
        The components correspond to the observation entries defined in Table~\ref{tab:observations}:

        - T01: [#1] Noisy indoor temperature deviation from setpoint.
        - T02: [#1, #2] Adds normalized time of day.
        - T03: [#1, #3] Adds action-based energy consumption (derived from previous action).
        - T04: [#1, #2, #3] Combines thermal deviation, time of day, and energy usage.
        - C01: [#1, #4, #5] Combines thermal deviation with current and forecasted prices.
        - C02: [#1, #3, #4, #5] Adds energy use to C01.
        - C03: [#1, #2, #4, #5] Adds time of day to C01.
        - C04: [#1, #2, #3, #4, #5] Full feature set for temperature and economic control.

        Returns:
            np.ndarray: Observation vector of shape (n,), where n depends on `obs_variant`.

        Note:
            For variants involving future energy price forecasts (#5), the method appends
            `n_p` steps of normalized future prices to the observation vector.
        """
        # Compute temperature deviation with stochastic noise
        temp_deviation = self.building.T_in - self.building.T_set
        noise = self.building._rng.normal(loc=0, scale=0.1)
        noisy_temp_deviation = temp_deviation * (1 + noise)
        # Normalize current time of day to [0, 1]
        current_time_sec = self.building.iteration * self.building.timestep
        normalized_time_of_day = (current_time_sec % (24 * 3600)) / (24 * 3600)
        # Previous control action (normalized power input to heat pump)
        normalized_prev_action = self.prev_action  # in [-1, 1], unitless

        # Get selected observation variant
        variant = getattr(self, "obs_variant", "T01")  # fallback to T01 if undefined
        observation = []

        # Thermal-only observation variants (T*)
        if variant == "T01":
            observation = [noisy_temp_deviation]
        elif variant == "T02":
            observation = [noisy_temp_deviation, normalized_time_of_day]
        elif variant == "T03":
            observation = [noisy_temp_deviation, normalized_prev_action]
        elif variant == "T04":
            observation = [
                noisy_temp_deviation,
                normalized_time_of_day,
                normalized_prev_action,
            ]
        # Combined thermal + economic observation variants (C*)
        elif variant == "C01":
            observation = [noisy_temp_deviation, self.building.energy_price]
        elif variant == "C02":
            observation = [
                noisy_temp_deviation,
                normalized_prev_action,
                self.building.energy_price,
            ]
        elif variant == "C03":
            observation = [
                noisy_temp_deviation,
                normalized_time_of_day,
                self.building.energy_price,
            ]
        elif variant == "C04":
            observation = [
                noisy_temp_deviation,
                normalized_time_of_day,
                normalized_prev_action,
                self.building.energy_price,
            ]
        else:
            raise ValueError(f"Unknown obs_variant: {variant}")
        # Future: add forecasted energy prices to observation
        # Append future prices if required
        if variant in ("C01", "C02", "C03", "C04"):
            observation.extend(self._get_future_energy_prices())
        return np.array(observation, dtype=np.float32)

    def render(self, mode="human"):
        """Optional rendering logic. Not implemented in this environment."""
        pass

    def close(self):
        """Cleanup operations if necessary."""
        pass
