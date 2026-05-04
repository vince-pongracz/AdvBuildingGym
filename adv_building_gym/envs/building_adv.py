from typing import Any, Dict, Optional
from collections import OrderedDict
import logging

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict as SDict
import numpy as np

from adv_building_gym.data_combinator import DataCombinator
from adv_building_gym.devices.statesources import StateSource
from adv_building_gym.rewards import RewardFunction, RewardAggregator, SumRewardAggregator
from adv_building_gym.devices.infrastructure import Infrastructure

from adv_building_gym.utils.rng_service import RngService
from adv_building_gym.utils.warning_filters import setup_warning_filters
from adv_building_gym.utils.constants import SECONDS_PER_HOUR
from adv_building_gym.envs.data_variant import DataVariantProvider
from adv_building_gym.envs._data_variant_manager import DataVariantManager
from adv_building_gym.envs._action_history_buffer import ActionHistoryBuffer
from adv_building_gym.envs._energy_tracker import EnergyTracker
from adv_building_gym.envs._raw_state_collector import RawStateCollector

from adv_building_gym.config.env_config import EnvConfig

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True  # Override any existing logging configuration (e.g., from Ray/RLlib)
)
logger = logging.getLogger(__name__)


class AdvBuildingGym(gym.Env, DataVariantProvider):
    """Modular Gymnasium env for building-energy control.

    Composes Infrastructure, StateSource, and RewardFunction components and
    drives them through a fixed control-step loop. Internal action and
    observation spaces are Dict-shaped; external wrappers (FlattenAction +
    RescaleAction) convert to the flat ``Box(-1, 1)`` interface RL libraries
    expect.

    Data contract — observation vs. info
        1. ``self.state`` / ``observation_space`` — policy-visible normalised
           values (``s_*`` / ``ctxt_*`` / ``raw_sim_hour``).
        2. ``self._component_info`` — shared inter-component dict for raw
           physical values (``net_power_kW``, EV schedule, action history).
        3. step/reset ``info`` — diagnostics for callbacks and logging
           (``reward_breakdown``, ``cum_E_kWh``, ``raw``).

    Construction
        ``env_config`` (EnvConfig) supplies ``EPISODE_LENGTH``,
        ``CONTROL_STEP``, and ``ACTION_HISTORY_LENGTH``. Components are
        passed in as already-constructed instances (the env no longer
        reaches into a global singleton). ``data_combinator`` and
        ``reward_aggregator`` are pluggable strategies.
    """

    def __init__(
        self,
        env_config: EnvConfig,
        statesources: list[StateSource],
        infras: list[Infrastructure],
        rewards: list[RewardFunction],
        *,
        data_combinator: DataCombinator,
        reward_aggregator: RewardAggregator,
        instance_id: str | None = None,
        render_mode=None,
        **kwargs,
    ):
        """Initialise the building-energy gym environment.

        Args:
            infras: Infrastructure components defining controllable
                action/observation sub-spaces.
            statesources: External time-series providers and physics
                updaters. Must include ``BuildingHeatLoss``.
            rewards: Reward functions evaluated at each step.
            env_config: EnvConfig instance providing episode length,
                control step, and action-history window length.
            data_combinator: Optional pre-built DataCombinator. Defaults
                to an empty combinator (no per-episode CSV swapping).
            reward_aggregator: Optional aggregation strategy. Defaults to
                ``SumRewardAggregator`` (matches previous behaviour).
            instance_id: Caller id for the RngService registry. Each env
                instance must be unique to keep per-worker seed streams
                independent. Falls back to ``"AdvBuildingGym"`` for
                single-env paths.
            render_mode: Gymnasium render mode (currently unused).
        """

        # Setup warning filters for Ray workers (must be called early)
        setup_warning_filters()

        super().__init__()

        self.env_config = env_config
        self.iteration = 0
        self._reward_aggregator: RewardAggregator = reward_aggregator
        self._variant_manager = DataVariantManager(
            data_combinator=data_combinator,
            episode_length=env_config.EPISODE_LENGTH,
        )

        self._energy_tracker = EnergyTracker(control_step_s=env_config.CONTROL_STEP)
        self._raw_state_collector = RawStateCollector()

        # Each env instance is a distinct caller in the RngService registry, keyed
        # by instance_id (worker_index + vector_index passed by env_creator). That
        # gives every worker its own deterministic seed chain — independent of
        # which worker's RPC reaches the actor first. Falls back to "AdvBuildingGym"
        # for single-env paths (eval, tests) where there is no ambiguity.
        # Gymnasium's reset(seed=...) contract overrides this further down.
        self._rng_caller_id: str = instance_id or "AdvBuildingGym"
        self._rng: np.random.Generator = np.random.default_rng(
            RngService.get().get_random(self._rng_caller_id)
        )

        # Build observation and action spaces from components.
        # Time-varying signals are normalised to small ranges; raw scale
        # factors (e.g. temp_abs_max, E_price_max) are included
        # unnormalised so the policy can reconstruct physical units.
        observation_space: OrderedDict = OrderedDict()
        action_space: OrderedDict = OrderedDict()

        self.infras = infras
        for infr in self.infras:
            infr.setup_spaces(observation_space, action_space)
        self.action_space_keys = list(action_space.keys())

        # For each action key, publish a matching ``<key>_prev`` observation
        # carrying the most recent applied action. Policies that want action
        # history consume these as plain obs keys (e.g. via
        # StridedHistoryConnector, which only needs to handle obs keys), so
        # there is no need to reconstruct per-key actions from the flat
        # Box stored in episodes by FlattenAction + RescaleAction.
        # Link: docs/hst_mgmt.md
        for act_key, act_box in action_space.items():
            observation_space[f"{act_key}_prev"] = spaces.Box(
                low=act_box.low, high=act_box.high,
                shape=act_box.shape, dtype=act_box.dtype,
            )

        self.statesources = statesources
        for ds in self.statesources:
            ds.setup_spaces(observation_space, action_space)

        self.reward_functors = rewards

        # sim_hour: hour of day (0–24) derived from the current step
        # within the episode.  Used by statesource synthetic profiles
        # and SolarPanel for time-of-day logic.
        observation_space["raw_sim_hour"] = spaces.Box(low=0.0, high=24.0, shape=(1,), dtype=np.float32)

        # NOTE VP 2026.05.04.: Tracks actions from the last ACTION_HISTORY_LENGTH steps
        self._action_history = ActionHistoryBuffer(
            action_keys=self.action_space_keys,
            max_length=env_config.ACTION_HISTORY_LENGTH,
        )

        # Assign spaces
        self.observation_space = SDict(observation_space)
        self.state: OrderedDict = OrderedDict()
        for state_name, state_space in observation_space.items():
            # Initialize observation state arrays with the same dtype as the declared space
            self.state[state_name] = np.zeros(shape=state_space.shape, dtype=state_space.dtype)

        # Shared dict for inter-component data that is NOT part of the
        # observation space (raw kWh/kW/°C values, EV schedule parameters).
        self._component_info: dict = {}

        # Native Dict action space — each key maps to the component's real bounds.
        self.action_space = SDict(action_space)

        # When True, step()/reset() include a deep copy of the full named state
        # dict in info["state"]. Expensive in memory — enable for evaluation only.
        self.log_full_info: bool = False

        lines: list = [
            "AdvBuildingGym created!",
            f"      States: {[ds.name for ds in statesources]}",
            f"     Actions: {[infr.name for infr in infras]}",
            f"  Objectives: {[rew.name for rew in rewards]}",
        ]
        logger.info("\n%s", "\n".join(lines))

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def cum_E_kWh(self) -> float:
        return self._energy_tracker.cum_E_kWh

    @property
    def episode_count(self) -> int:
        return self._variant_manager.episode_count

    @property
    def action_history(self):
        return self._action_history.history

    @property
    def data_combinator(self) -> DataCombinator:
        return self._variant_manager.data_combinator

    # ------------------------------------------------------------------
    # Hot-swap APIs (callbacks)
    # ------------------------------------------------------------------

    def set_reward_funcs(self, rewards: list[RewardFunction]) -> None:
        """Hot-swap the active reward functions (reward_switch callback)."""
        self.reward_functors = rewards
        logger.debug("Reward functions updated: %s", [r.name for r in rewards])

    def set_infras(self, infras: list[Infrastructure]) -> None:
        """Hot-swap infrastructure components (infra_schedule callback).

        Names must match the existing infras — spaces are invariant.
        """
        old_names = set([i.name for i in self.infras])
        new_names = set([i.name for i in infras])
        if old_names != new_names:
            raise ValueError(f"Infrastructure names must match. Old: {old_names}, New: {new_names}")
        
        self.infras = infras
        logger.info("Infrastructure swapped: %s", [i.name for i in infras])

    def get_state_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space

    def apply_data_variant(self, variant: dict[str, str]) -> None:
        """Reload statesources whose names appear in *variant*."""
        for state_src in self.statesources:
            if state_src.name in variant:
                state_src.reload(variant[state_src.name])

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        self._maybe_reseed(seed)

        self._variant_manager.begin_episode()
        variant = self._select_variant(options)
        row_offset = self._variant_manager.compute_day_offset(
            self.statesources, self._rng, self.env_config.CONTROL_STEP, options,
        )

        if variant:
            logger.info(
                "Episode %d, date %s: data variant %s",
                self.episode_count,
                self.data_combinator.get_day_date(),
                variant,
            )

        self._reset_internal_state()
        self._sync_components(row_offset)
        self._populate_initial_observations()

        info = self._build_reset_info(seed, variant)
        return {k: np.array(v, copy=True) for k, v in self.state.items()}, info

    def _maybe_reseed(self, seed: int | None) -> None:
        """Apply Gymnasium's seeding contract.

        Only fires when an explicit seed is provided (typically the first
        reset per env runner). Subsequent resets must NOT recreate the
        RNG — that would destroy the deterministic sequence.

        Note: ``random.seed`` / ``np.random.seed`` (global RNGs) are NOT
        touched here. Components must use ``self._rng`` or
        ``RngService.get()`` instead — the global state is shared across
        Ray workers and would interfere across reset calls.
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def _select_variant(self, options: dict | None) -> dict[str, str] | None:
        variant = self._variant_manager.select_variant(options, self._rng)
        if variant is not None:
            self.apply_data_variant(variant)
        return variant

    def _reset_internal_state(self) -> None:
        self.iteration = 0
        self._energy_tracker.reset()
        self._action_history.clear()
        # Re-zero observation arrays in place (preserves dtype/shape).
        for k, v in self.state.items():
            if isinstance(v, np.ndarray):
                self.state[k] = np.zeros(v.shape, dtype=np.float32)
            else:
                logger.debug("Unidentified type: %s", type(v))
        # sim_hour starts at midnight.
        self.state["raw_sim_hour"][0] = np.float32(0.0)
        self._component_info.clear()

    def _sync_components(self, row_offset: int) -> None:
        for sync in self.infras + self.statesources:
            sync.synchronise(self.iteration, row_offset)

    def _populate_initial_observations(self) -> None:
        """Let statesources and infras publish their initial observations."""
        for ds in self.statesources:
            ds.reset(states=self.state, info=self._component_info)
        for infr in self.infras:
            infr.reset(self.state, info=self._component_info)

    def _build_reset_info(self, seed: int | None, variant: dict | None) -> dict:
        info = {
            "seed": seed,
            "episode_date": self._variant_manager.episode_date,
            "episode_day_mode": self._variant_manager.episode_day_mode,
            "data_variant": variant if variant else None,
            "raw": self._raw_state_collector.collect(self.statesources, self.infras, self.state),
        }
        if self.log_full_info:
            info["state"] = {
                k: np.array(v, copy=True)
                for k, v in self.state.items()
                if isinstance(v, np.ndarray)
            }
        return info

    def _get_observation(self) -> dict:
        # Start with the current env state so statesources have access to
        # bookkeeping keys such as "iteration" and "raw_sim_hour" during reset.
        state = OrderedDict(self.state) if isinstance(self.state, OrderedDict) else OrderedDict()
        for ds in self.statesources:
            ds.update_state(states=state, info=self._component_info)
        return state

    def is_done(self) -> bool:
        # Episode ends when: 
        #  (a) the natural horizon is hit 
        #  (b) any reward voted to terminate in the Phase-1 pre-pass. 
        # The pre-pass writes the consolidated verdict into 
        # _component_info["terminated"] before any get_reward runs, 
        # so reward functions can read it safely.
        if self._component_info.get("terminated", False):
            return True
        return bool(self.iteration >= self.env_config.EPISODE_LENGTH)

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action):
        """Execute one control step. See class docstring for the data contract."""
        self._execute_actions(action)
        self._advance_time_and_sync()
        self._update_state()
        total_power_kW, power_breakdown = self._compute_power_breakdown(action)
        self._publish_step_info(action, total_power_kW, power_breakdown)
        # Phase 1: termination pre-pass — consolidate every reward's verdict
        # into info["terminated"] before any get_reward runs. This decouples
        # rewards: episode-aggregated rewards (LTER) can read a definitive
        # flag instead of relying on YAML ordering vs. terminating rewards.
        self._component_info["terminated"] = self._compute_termination(action)
        # Phase 2: reward aggregation — sees the correct terminated flag.
        reward, reward_breakdown, max_reward_step = self._compute_rewards(action)

        # Append AFTER reward computation so ActionSmoothnessReward still
        # compares the just-passed ``actions`` to ``action_history[-1]``.
        # Link: docs/hst_mgmt.md
        self._action_history.append_and_mirror(action, self.state)

        self._guard_finite_state()

        terminated = self.is_done()
        truncated = False

        info = self._build_step_info(
            action, reward, reward_breakdown, max_reward_step,
            total_power_kW, power_breakdown,
        )
        return (
            {k: np.array(v, copy=True) for k, v in self.state.items()},
            reward,
            terminated,
            truncated,
            info,
        )

    def _execute_actions(self, action) -> None:
        for infr in self.infras:
            infr.exec_action(action, self.state, info=self._component_info)

    def _advance_time_and_sync(self) -> None:
        # Advance time: increment iteration, then synchronise all components so
        # update_state reads the correct (new) row from time-series data.
        # Previously synchronise was called AFTER update_state, causing exogenous
        # datasources (price, weather, EV schedule) to lag 2 iterations behind.
        self.iteration += 1
        self.state["raw_sim_hour"][0] = np.float32(
            (self.iteration * self.env_config.CONTROL_STEP) / SECONDS_PER_HOUR
        )
        for sync in self.infras + self.statesources:
            sync.synchronise(self.iteration)

    def _update_state(self) -> None:
        # Reset directional power bound accumulators before infras publish.
        self._component_info["max_consumption_kW"] = 0.0
        self._component_info["max_export_kW"] = 0.0
        for infr in self.infras:
            infr.update_state(self.state, info=self._component_info)
        for ds in self.statesources:
            ds.update_state(states=self.state, info=self._component_info)

    def _compute_power_breakdown(self, action) -> tuple[float, dict[str, float]]:
        power_breakdown = {
            infra.name: infra.get_electric_consumption(action)
            for infra in self.infras
        }
        total_power_kW, _energy_kWh = self._energy_tracker.advance(power_breakdown)
        return total_power_kW, power_breakdown

    def _publish_step_info(
        self,
        action,
        total_power_kW: float,
        power_breakdown: dict[str, float],
    ) -> None:
        # Reward functions read these from _component_info instead of holding
        # references to infrastructures.
        self._component_info["power_breakdown"] = power_breakdown
        self._component_info["net_power_kW"] = total_power_kW
        self._component_info["penalisable_power_kW"] = sum(
            infra.get_penalisable_consumption(action, self.state)
            for infra in self.infras
        )
        self._component_info["action_history"] = self._action_history.history
        self._component_info["episode_length"] = self.env_config.EPISODE_LENGTH
        self._component_info["iteration"] = self.iteration
        # info["terminated"] is set by _compute_termination() in Phase 1 of
        # step(), after _publish_step_info populates the rest of info.

    def _compute_termination(self, action) -> bool:
        """Phase-1 termination pass. Returns True iff the episode ends here.

        Naturally times out at max_iteration; otherwise polls each reward's
        ``should_terminate`` (default False). Pure: must not mutate state.
        """
        if self.iteration >= self.env_config.EPISODE_LENGTH:
            return True
        for rf in self.reward_functors:
            if rf.should_terminate(action, self.state, self._component_info):
                return True
        return False

    def _compute_rewards(self, action) -> tuple[float, dict[str, float], float]:
        return self._reward_aggregator.aggregate(
            self.reward_functors, action, self.state, self._component_info,
        )

    def _guard_finite_state(self) -> None:
        # NaN/Inf in state propagates through the network and crashes the
        # action distribution (std = NaN → RuntimeError). Replace with 0.
        for key, val in self.state.items():
            if isinstance(val, np.ndarray) and not np.all(np.isfinite(val)):
                logger.error(
                    "NaN/Inf in state['%s']: %s (episode %d, step %d) — replaced with 0",
                    key, val, self.episode_count, self.iteration,
                )
                self.state[key] = np.where(np.isfinite(val), val, np.zeros_like(val))

    def _build_step_info(
        self,
        action,
        reward: float,
        reward_breakdown: dict[str, float],
        max_reward_step: float,
        total_power_kW: float,
        power_breakdown: dict[str, float],
    ) -> dict:
        info = {
            "action": action,
            "reward": reward,
            "reward_breakdown": reward_breakdown,
            "max_reward_step": max_reward_step,
            "cum_E_kWh": self._energy_tracker.cum_E_kWh,
            "step_power_kW": total_power_kW,
            "power_breakdown": power_breakdown,
            "raw": self._raw_state_collector.collect(self.statesources, self.infras, self.state),
        }
        if self.log_full_info:
            info["state"] = {
                k: np.array(v, copy=True)
                for k, v in self.state.items()
                if isinstance(v, np.ndarray)
            }
        return info

    def render(self):
        """Render the environment."""
        pass
