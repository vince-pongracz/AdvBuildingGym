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
from adv_building_gym.core.data_variant import DataVariantConsumer
from adv_building_gym.core._data_variant_manager import DataVariantManager
from adv_building_gym.core._energy_tracker import EnergyTracker
from adv_building_gym.core._price_tracker import PriceTracker
from adv_building_gym.core._raw_state_tracker import RawStateTracker

from adv_building_gym.config.env.env_config import EnvConfig

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True  # override existing logging config (e.g. Ray/RLlib)
)
logger = logging.getLogger(__name__)


class AdvBuildingGym(gym.Env, DataVariantConsumer):
    """Modular Gymnasium env for building-energy control.

    Composes Infrastructure / StateSource / RewardFunction over a fixed control-step
    loop. 
    Internal Dict action/obs spaces; external wrappers (FlattenAction + RescaleAction)
    give the flat ``Box(-1, 1)`` interface.

    Data contract — observation vs. info
        1. ``self.state`` / ``observation_space`` — policy-visible normalised
           values (``s_*`` / ``ctxt_*``).
        2. ``self._component_info`` — shared inter-component dict for raw
           physical values (``net_power_kW``, EV schedule, action history).
        3. step/reset ``info`` — diagnostics for callbacks and logging
           (``reward_breakdown``, ``cum_E_kWh``, ``raw``).

    Construction
        ``env_config`` (EnvConfig) supplies ``EPISODE_LENGTH``,
        ``CONTROL_STEP``, and ``ACTION_HISTORY_LENGTH``.
        Components passed are already pre-constructed.
        ``data_combinator`` / ``reward_aggregator`` are pluggable.
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
        eval_mode: bool = False,
        **kwargs,
    ):
        """env_config: episode length, control step, action-history window.
        infras / statesources (must include BuildingHeatLoss) / rewards: components.
        data_combinator: per-episode CSV swapping. 
        reward_aggregator: aggregation strategy.
        instance_id: logging id (seeding is via reset(seed=...), not this).
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
        self._raw_state_tracker = RawStateTracker()

        # provisional RNG until the first reset(seed=...) (RLlib/SB3 set the real
        # per-env seed). All in-env randomness flows through self._rng → reproducible per worker.
        self.instance_id: str = instance_id or "AdvBuildingGym"
        self._rng: np.random.Generator = np.random.default_rng()

        # Build obs/action spaces from components. Time-varying signals are normalised;
        # raw scale factors (temp_abs_max, E_price_max) are unnormalised for reconstruction.
        observation_space: OrderedDict = OrderedDict()
        action_space: OrderedDict = OrderedDict()

        self.infras = infras
        for infr in self.infras:
            infr.setup_spaces(observation_space, action_space)
        self.action_space_keys = list(action_space.keys())

        self.statesources = statesources
        self._partition_statesources()
        for ds in self.statesources:
            ds.setup_spaces(observation_space, action_space)

        # env-owned hour-of-day [0, 1] (sim_hour mod 24 / 24); depends only on iteration × control_step
        observation_space["s_sim_hour"] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self.reward_functors = rewards

        # Assign spaces
        self.observation_space = SDict(observation_space)
        self.state: OrderedDict = OrderedDict()
        for state_name, state_space in observation_space.items():
            # init obs arrays with the declared dtype
            self.state[state_name] = np.zeros(shape=state_space.shape, dtype=state_space.dtype)

        # shared inter-component data not in the obs space (raw kWh/kW/°C, EV schedule)
        self._component_info: dict = {}

        # Native Dict action space — each key maps to the component's real bounds.
        self.action_space = SDict(action_space)

        # When True, step()/reset() put a copy of the full state in info["state"] (eval only — costly).
        self.log_full_info: bool = False

        # Eval runner flag: every reset() draws a fresh random (variant, day) for broad
        # coverage, overriding the combinator cadence. Set by the env creators.
        self.eval_mode: bool = eval_mode

        # whether an explicit reset seed was consumed; _maybe_reseed seeds once in eval mode
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
        """Hot-swap infrastructure (infra_schedule callback); names must match (spaces invariant)."""
        old_names = set([i.name for i in self.infras])
        new_names = set([i.name for i in infras])
        if old_names != new_names:
            raise ValueError(f"Infrastructure names must match. Old: {old_names}, New: {new_names}")
        
        self.infras = infras
        logger.info("Infrastructure swapped: %s", [i.name for i in infras])

    def set_statesources(self, statesources) -> None:
        """Hot-swap statesources (statesource_schedule callback); names must match (spaces invariant)."""
        old_names = {s.name for s in self.statesources}
        new_names = {s.name for s in statesources}
        if old_names != new_names:
            raise ValueError(f"Statesource names must match. Old: {old_names}, New: {new_names}")
        self.statesources = statesources
        self._partition_statesources()
        logger.info("Statesources swapped: %s", [s.name for s in statesources])

    def _partition_statesources(self) -> None:
        """Split statesources by ``UPDATE_PHASE`` into endogenous (run before reward) and
        exogenous (advance to next row after); order preserved within each group."""
        self._endogenous_statesources = [s for s in self.statesources if s.UPDATE_PHASE == "endogenous"]
        self._exogenous_statesources = [s for s in self.statesources if s.UPDATE_PHASE != "endogenous"]

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
                self.episode_count, self.data_combinator.get_day_date(), variant,
            )

        self._reset_internal_state()
        self._sync_components(row_offset)
        self._populate_initial_observations()
        for rf in self.reward_functors:
            rf.on_reset(self.state, info=self._component_info)

        info = self._build_reset_info(seed, variant)
        return {k: np.array(v, copy=True) for k, v in self.state.items()}, info

    def _maybe_reseed(self, seed: int | None) -> None:
        """Apply Gymnasium's seeding contract: seed only when an explicit seed is given
        (the first reset per runner); later resets must NOT recreate the RNG.

        Eval-mode exception: RLlib re-hands the same fixed seed each eval round, which
        would replay identical (variant, day) sequences. So in eval mode we seed ONCE and
        ignore later seeds, letting the RNG advance for fresh data (still reproducible across
        reruns). Training (timestep-sampled) and the standalone eval script are unaffected.

        Global RNGs (``random``/``np.random``) are NOT touched; components draw from the env
        rng via ``info["_rng"]`` (see ``_populate_initial_observations``).
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
        # publish env rng on the shared channel so components (e.g. InsideTemperature)
        # draw per-episode randomness from the same per-worker stream
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
            "raw": self._raw_state_tracker.collect(self.statesources, self.infras),
        }
        if self.log_full_info:
            info["state"] = {
                k: np.array(v, copy=True)
                for k, v in self.state.items()
                if isinstance(v, np.ndarray)
            }
        return info

    def _get_observation(self) -> dict:
        # start from current state so statesources can read bookkeeping keys (e.g. "iteration")
        state = OrderedDict(self.state) if isinstance(self.state, OrderedDict) else OrderedDict()
        for ds in self.statesources:
            ds.update_state(states=state, info=self._component_info)
        return state

    def is_done(self) -> bool:
        # Ends on (a) natural horizon or (b) any reward's Phase-1 terminate vote,
        # consolidated into _component_info["terminated"] before get_reward runs.
        if self._component_info.get("terminated", False):
            return True
        return bool(self.iteration >= self.env_config.EPISODE_LENGTH)

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action):
        """Execute one control step. See class docstring for the data contract."""
        # Snapshot observed state s_t before mutation: rewards read inputs from s and
        # action outcomes from s' (self.state) — a proper R(s, a, s') transition.
        state = {k: np.array(v, copy=True) for k, v in self.state.items()}

        self._execute_actions(action)        # exec_action reads observed row[t]
        self._advance_time_and_sync()        # iteration++, sim_hour, synchronise
        self._update_endogenous()            # infras + inner statesources -> action outcomes (still row[t] exogenous)

        # power/price under observed row[t]: price tracker reads baseprice_raw pre-advance
        total_power_kW, power_breakdown = self._compute_power_breakdown(action)

        self._update_exogenous()             # outer statesources -> row[t+1]; self.state is now s'

        self._publish_step_info(action, total_power_kW)
        # Phase 1: termination pre-pass — consolidate verdicts into info["terminated"]
        # before any get_reward, so aggregated rewards (LTER) read a definitive flag.
        self._component_info["terminated"] = self._compute_termination(action, state)
        # Phase 2: reward aggregation over the full (s, a, s') transition
        reward, reward_breakdown = self._compute_rewards(action, state)

        self._guard_finite_state()

        terminated = self.is_done()
        truncated = False

        info = self._build_step_info(
            action, reward, reward_breakdown,
            total_power_kW, power_breakdown,
        )

        return (
            {k: np.array(v, copy=True) for k, v in self.state.items()},
            reward,
            terminated, truncated, info,
        )

    def _execute_actions(self, action) -> None:
        for infr in self.infras:
            infr.exec_action(action, self.state, info=self._component_info)

    def _advance_time_and_sync(self) -> None:
        # increment iteration, then synchronise so update_state reads the new row
        # (synchronising after update_state used to lag exogenous sources 2 iterations)
        self.iteration += 1
        self.sim_hour = self.iteration * self.env_config.CONTROL_STEP / SECONDS_PER_HOUR
        self.state["s_sim_hour"][0] = np.float32((self.sim_hour % 24.0) / 24.0)

        for sync in self.infras + self.statesources:
            sync.synchronise(self.iteration)

    def _update_endogenous(self) -> None:
        # within-step transition under observed row[t]: infras apply effects, then
        # endogenous physics (BuildingHeatLoss). Reset power-bound accumulators first.
        self._component_info["max_consumption_kW"] = 0.0
        self._component_info["max_production_kW"] = 0.0
        for infr in self.infras:
            infr.update_state(self.state, info=self._component_info)
        for ds in self._endogenous_statesources:
            ds.update_state(states=self.state, info=self._component_info)

    def _update_exogenous(self) -> None:
        # advance external series to row[t+1] (after reward); synchronise already moved
        # the effective index, so these read the new row
        for ds in self._exogenous_statesources:
            ds.update_state(states=self.state, info=self._component_info)

    def _compute_power_breakdown(self, action) -> tuple[float, dict[str, tuple[float, float]]]:
        power_breakdown = {
            infra.name: (infra.get_E(action))
            for infra in self.infras
        }
        total_power_kW, _energy_kWh = self._energy_tracker.add_step_E_contribution(power_breakdown)
        self._price_tracker.add_step_contribution(power_breakdown, self._current_baseprice_ct_per_kWh())
        return total_power_kW, power_breakdown

    def _current_baseprice_ct_per_kWh(self) -> float | None:
        # duck-type (avoid circular import); statesource caches baseprice_raw each tick
        for ds in self.statesources:
            price = getattr(ds, "baseprice_raw", None)
            if price is not None:
                return float(price)
        return None

    def _publish_step_info(
        self,
        action,
        total_power_kW: float,
    ) -> None:
        # rewards read these from _component_info (no infra references). power_breakdown
        # is emitted only on the returned step info, not here.
        self._component_info["net_power_kW"] = total_power_kW
        self._component_info["episode_length"] = self.env_config.EPISODE_LENGTH
        self._component_info["iteration"] = self.iteration
        self._component_info["control_step_s"] = self.env_config.CONTROL_STEP
        # expose the termination toggle so rewards can gate terminal-only branches
        self._component_info["allow_early_termination"] = self.env_config.allow_early_termination
        # info["terminated"] is set later by _compute_termination() (Phase 1)

    def _compute_termination(self, action, state) -> bool:
        """Phase-1 termination pass: True iff the episode ends here. Times out at the horizon,
        else polls each reward's ``should_terminate`` over (s, a, s'). Pure — no state mutation."""
        if self.iteration >= self.env_config.EPISODE_LENGTH:
            return True
        if not self.env_config.allow_early_termination:
            return False
        for rf in self.reward_functors:
            if rf.should_terminate(action, state, self.state, self._component_info):
                return True
        return False

    def _compute_rewards(self, action, state) -> tuple[float, dict[str, float]]:
        # fresh per-step diagnostics channel (rewards populate; EpisodeMetricsCallback
        # sums per episode); cleared so values don't leak from the previous step
        self._component_info["reward_diagnostics"] = {}
        # full transition: state = s (observed), self.state = s' (resulting)
        return self._reward_aggregator.aggregate(
            self.reward_functors, action, state, self.state, self._component_info,
        )

    def _guard_finite_state(self) -> None:
        # NaN/Inf would crash the action distribution (std=NaN); replace with 0
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
        total_power_kW: float,
        power_breakdown: dict[str, float],
    ) -> dict:
        info = {
            "action": action,
            "reward": reward,
            "reward_breakdown": reward_breakdown,
            "reward_diagnostics": self._component_info.get("reward_diagnostics", {}),
            "cum_E_kWh": self._energy_tracker.cum_E_kWh,
            "cum_price_EUR": self._price_tracker.cum_price_EUR,
            "net_power_kW": total_power_kW,
            "power_breakdown": power_breakdown,
            "raw": self._raw_state_tracker.collect(self.statesources, self.infras),
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
