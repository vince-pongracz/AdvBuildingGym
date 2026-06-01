from typing import Any, Dict, Optional
from collections import OrderedDict
import logging

import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict as SDict
import numpy as np

from adv_building_gym.config.data.data_combinator import DataCombinator
from adv_building_gym.components.statesources import StateSource
from adv_building_gym.components.rewards import RewardFunction, RewardAggregator, SumRewardAggregator
from adv_building_gym.components.infrastructure import Infrastructure

from adv_building_gym._common.warning_filters import setup_warning_filters
from adv_building_gym._common.constants import SECONDS_PER_HOUR
from adv_building_gym.core.data_variant import DataVariantProvider
from adv_building_gym.core._data_variant_manager import DataVariantManager
from adv_building_gym.core._action_history_buffer import ActionHistoryBuffer
from adv_building_gym.core._energy_tracker import EnergyTracker
from adv_building_gym.core._price_tracker import PriceTracker
from adv_building_gym.core._raw_state_collector import RawStateCollector

from adv_building_gym.config.env.env_config import EnvConfig

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
           values (``s_*`` / ``ctxt_*``).
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
            instance_id: Human-readable identifier for this env instance
                (e.g. ``AdvBuildingGym_w<worker>_v<vector>`` from the
                env_creator), used for logging. Per-env seeding is handled
                by ``reset(seed=...)``, not this id.
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
        self._price_tracker = PriceTracker(control_step_s=env_config.CONTROL_STEP)
        self._raw_state_collector = RawStateCollector()

        # Provisional generator for the construct→first-reset window only. The
        # authoritative per-env seed arrives via the first reset(seed=...):
        # RLlib (config.debugging seed → trial.seed + worker_index) and SB3
        # (model.set_random_seed → env.seed → seed+i) both override this. All
        # in-env randomness (variant/day selection, statesource offsets) flows
        # through self._rng, so it is reproducible per worker once seeded.
        self.instance_id: str = instance_id or "AdvBuildingGym"
        self._rng: np.random.Generator = np.random.default_rng()

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
        # history consume these as plain obs keys (e.g. via the env-side
        # HistoryWrapper, which only needs to handle obs keys), so there is
        # no need to reconstruct per-key actions from the flat Box stored in
        # episodes by FlattenAction + RescaleAction.
        # Link: docs/hst_mgmt.md
        for act_key, act_box in action_space.items():
            observation_space[f"{act_key}_prev"] = spaces.Box(
                low=act_box.low, high=act_box.high,
                shape=act_box.shape, dtype=act_box.dtype,
            )

        self.statesources = statesources
        for ds in self.statesources:
            ds.setup_spaces(observation_space, action_space)

        # Env-owned hour-of-day signal in [0, 1] (sim_hour mod 24 / 24).
        # No component owns it because it depends only on iteration × control_step.
        observation_space["s_sim_hour"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self.reward_functors = rewards

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

        # When True, this env is an evaluation runner. In eval mode every reset()
        # draws a *fresh random* data variant (overriding the combinator's
        # episode_count // swap_every_n_episodes cadence) so each episode of an
        # eval round samples an independent (variant, day) pair — giving broad,
        # unbiased coverage rather than 5 consecutive episodes on one variant.
        # Set by the env creators from the eval env-config override.
        self.eval_mode: bool = False

        # Tracks whether this env has consumed an explicit reset seed yet. Used
        # by _maybe_reseed to seed the RNG exactly once in eval mode (see there).
        self._has_seeded: bool = False

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
    def cum_price_EUR(self) -> float:
        return self._price_tracker.cum_price_EUR

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

    def set_statesources(self, statesources) -> None:
        """Hot-swap statesource components (statesource_schedule callback).

        Names must match the existing statesources — spaces are invariant.
        """
        old_names = {s.name for s in self.statesources}
        new_names = {s.name for s in statesources}
        if old_names != new_names:
            raise ValueError(f"Statesource names must match. Old: {old_names}, New: {new_names}")
        self.statesources = statesources
        logger.info("Statesources swapped: %s", [s.name for s in statesources])

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
        for rf in self.reward_functors:
            rf.on_reset(self.state, info=self._component_info)

        info = self._build_reset_info(seed, variant)
        return {k: np.array(v, copy=True) for k, v in self.state.items()}, info

    def _maybe_reseed(self, seed: int | None) -> None:
        """Apply Gymnasium's seeding contract.

        Only fires when an explicit seed is provided (typically the first
        reset per env runner). Subsequent resets must NOT recreate the
        RNG — that would destroy the deterministic sequence.

        Eval-mode exception: RLlib's SingleAgentEnvRunner samples eval by
        ``num_episodes``, which sets ``_needs_initial_reset=True`` after every
        eval round and therefore re-hands the env the SAME fixed ``self._seed``
        at the start of each round (single_agent_env_runner.py: the
        ``seed=self._seed if self._needs_initial_reset`` reset). Honouring that
        every round would restart ``self._rng`` from an identical state, so each
        eval round would replay the exact same (variant, day) sequence. In eval
        mode we therefore seed ONCE and then ignore further seeds, letting the
        RNG advance so every eval round samples fresh data. The run stays
        reproducible across reruns (same first seed → same advancing stream).
        Training is unaffected — it samples by timesteps and only ever seeds on
        its first reset. The standalone eval script (run_eval_ray) does not set
        eval_mode, so its per-episode seeding is preserved.

        Note: ``random.seed`` / ``np.random.seed`` (global RNGs) are NOT
        touched here. Components must draw from the env rng instead — it is
        published on the shared info channel as ``info["_rng"]`` each reset
        (see ``_populate_initial_observations``) so statesources get the same
        deterministic, per-worker stream without touching global state.
        """
        apply_seed = seed is not None and not (self.eval_mode and self._has_seeded)
        super().reset(seed=seed if apply_seed else None)
        if apply_seed:
            self._rng = np.random.default_rng(seed)
            self._has_seeded = True

    def _select_variant(self, options: dict | None) -> dict[str, str] | None:
        variant = self._variant_manager.select_variant(
            options, self._rng, eval_mode=self.eval_mode,
        )
        if variant is not None:
            self.apply_data_variant(variant)
        return variant

    def _reset_internal_state(self) -> None:
        self.iteration = 0
        self.sim_hour = 0
        self._energy_tracker.reset()
        self._price_tracker.reset()
        self._action_history.clear()
        # Re-zero observation arrays in place (preserves dtype/shape).
        for k, v in self.state.items():
            if isinstance(v, np.ndarray):
                self.state[k] = np.zeros(v.shape, dtype=np.float32)
            else:
                logger.debug("Unidentified type: %s", type(v))

        self._component_info.clear()

    def _sync_components(self, row_offset: int) -> None:
        for sync in self.infras + self.statesources:
            sync.synchronise(self.iteration, row_offset)

    def _populate_initial_observations(self) -> None:
        """Let statesources and infras publish their initial observations."""
        # Publish the env rng on the shared channel so components that need
        # per-episode randomness (e.g. InsideTemperature's initial offset) draw
        # from the same deterministic, per-worker stream as variant/day selection.
        self._component_info["_rng"] = self._rng
        for ds in self.statesources:
            ds.reset(states=self.state, info=self._component_info)
        for infr in self.infras:
            infr.reset(self.state, info=self._component_info)

    def _build_reset_info(self, seed: int | None, variant: dict | None) -> dict:
        info = {
            "seed": seed,
            "episode_count": self.episode_count,
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
        # bookkeeping keys such as "iteration" during reset.
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
        self.sim_hour = self.iteration * self.env_config.CONTROL_STEP / SECONDS_PER_HOUR
        self.state["s_sim_hour"][0] = np.float32((self.sim_hour % 24.0) / 24.0)

        for sync in self.infras + self.statesources:
            sync.synchronise(self.iteration)

    def _update_state(self) -> None:
        # Reset directional power bound accumulators before infras publish.
        self._component_info["max_consumption_kW"] = 0.0
        self._component_info["max_production_kW"] = 0.0
        for infr in self.infras:
            infr.update_state(self.state, info=self._component_info)
        for ds in self.statesources:
            ds.update_state(states=self.state, info=self._component_info)

    def _compute_power_breakdown(self, action) -> tuple[float, dict[str, tuple[float, float]]]:
        power_breakdown = {
            infra.name: (infra.get_E(action))
            for infra in self.infras
        }
        total_power_kW, _energy_kWh = self._energy_tracker.add_step_E_contrib(power_breakdown)
        self._price_tracker.add_step_contrib(power_breakdown, self._current_baseprice_ct_per_kWh())
        return total_power_kW, power_breakdown

    def _current_baseprice_ct_per_kWh(self) -> float | None:
        # Duck-type to avoid importing EnergyPriceDataSource (circular). The
        # statesource caches baseprice_raw during update_state each tick.
        for ds in self.statesources:
            price = getattr(ds, "baseprice_raw", None)
            if price is not None:
                return float(price)
        return None

    def _publish_step_info(
        self,
        action,
        total_power_kW: float,
        power_breakdown: dict[str, float],
    ) -> None:
        # Reward functions read these from _component_info instead of holding
        # references to infrastructures. power_breakdown is not republished
        # here — it is emitted only on the returned step info dict (consumed
        # by trajectory logging / plotting), no reward needs it internally.
        self._component_info["net_power_kW"] = total_power_kW
        self._component_info["penalisable_power_kW"] = sum(
            infra.get_penalisable_consumption(action, self.state)
            for infra in self.infras
        )
        self._component_info["action_history"] = self._action_history.history
        self._component_info["episode_length"] = self.env_config.EPISODE_LENGTH
        self._component_info["iteration"] = self.iteration
        self._component_info["control_step_s"] = self.env_config.CONTROL_STEP
        # Surface the env-level termination toggle so reward functors can gate
        # their terminal-only branches (huge penalties / success bonuses).
        self._component_info["allow_early_termination"] = self.env_config.allow_early_termination
        # info["terminated"] is set by _compute_termination() in Phase 1 of
        # step(), after _publish_step_info populates the rest of info.

    def _compute_termination(self, action) -> bool:
        """Phase-1 termination pass. Returns True iff the episode ends here.

        Naturally times out at max_iteration; otherwise polls each reward's
        ``should_terminate`` (default False). Pure: must not mutate state.
        """
        if self.iteration >= self.env_config.EPISODE_LENGTH:
            return True
        if not self.env_config.allow_early_termination:
            return False
        for rf in self.reward_functors:
            if rf.should_terminate(action, self.state, self._component_info):
                return True
        return False

    def _compute_rewards(self, action) -> tuple[float, dict[str, float], float]:
        # Fresh per-step diagnostics channel — reward functions populate it
        # (e.g. saturation / clip flags) and the EpisodeMetricsCallback sums
        # each key across the episode for TensorBoard. Cleared here so values
        # never leak from the previous step's _component_info.
        self._component_info["reward_diagnostics"] = {}
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
            "reward_diagnostics": self._component_info.get("reward_diagnostics", {}),
            "cum_E_kWh": self._energy_tracker.cum_E_kWh,
            "cum_price_EUR": self._price_tracker.cum_price_EUR,
            "net_power_kW": total_power_kW,
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
