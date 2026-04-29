import logging
import numpy as np

from .base import RewardFunction
from adv_building_gym.utils.serializable import ComponentRegistry

logger = logging.getLogger(__name__)


class LongTermEconomicReward(RewardFunction):
    """Sparse, period-aggregated economic reward.

    Companion to ``EconomicReward``: while the dense version emits a
    cost/income signal every step, this one accumulates the same raw
    quantity over an episode-length window and emits a single bonus on
    the final step. Run both together for the hybrid scheme — dense
    keeps the per-step gradient alive, sparse aligns the objective with
    the actual billing-window outcome (load shifting within the window
    incurs no local penalty).

    Per-step accumulator (matches ``EconomicReward.get_reward``):

        raw_step = -net_power_kW * E_price / reference_power_kW
                   * (export_bonus if net_power_kW < 0 else 1)

    On the final step the emitted reward is
    ``weight * clip(sum(raw_step), -episode_length, episode_length)``
    with ``max_reward_step = weight * episode_length``. On all other
    steps the emitted pair is ``(0.0, 0.0)`` so the dense rewards still
    define ``reward_rate`` — at the window boundary both numerator and
    denominator jump together.

    The window length is read from ``info["episode_length"]`` (published
    each step by the env from ``EnvConfig.EPISODE_LENGTH``) so there is
    a single source of truth in ``configs/env_meta/*.yaml``. Reset is
    detected via a sentinel key planted in ``info``: the env clears
    ``_component_info`` on ``reset()`` (see ``building_adv.py``), so a
    missing sentinel signals a fresh episode and the accumulator is
    zeroed. This avoids needing a dedicated ``reset`` hook on
    ``RewardFunction`` while staying robust to early termination.
    """

    def __init__(self, weight: float, reference_power_kW: float = 15.0,
                name: str = "long_term_economic_reward",
                export_bonus: float = 3.0) -> None:
        """Initialize LongTermEconomicReward.

        Args:
            weight: Reward weight for multi-objective optimization.
            reference_power_kW: Fallback power scale (kW) used when
                ``ctxt_operator_max_power_kW`` is not present in
                ``states``. Otherwise that ctxt value is preferred so
                this reward auto-tracks the active grid-exchange limit
                (matching ``EconomicReward``).
            name: Reward function identifier.
            export_bonus: Multiplier applied when exporting
                (``net_power_kW < 0``); same semantics as
                ``EconomicReward.export_bonus``.
        """
        super().__init__(weight, name)
        if reference_power_kW <= 0:
            raise ValueError("reference_power_kW must be positive.")

        self.reference_power_kW = float(reference_power_kW)
        self.export_bonus = float(export_bonus)
        self._sentinel_key = f"_lt_econ_active_{id(self)}"
        self._step_in_window = 0
        self._accum = 0.0

    _exclude_params = {"_step_in_window", "_accum", "_sentinel_key"}

    def _reset_window(self) -> None:
        self._step_in_window = 0
        self._accum = 0.0

    def _resolve_reference_power_kW(self, states) -> float:
        ctxt = states.get("ctxt_operator_max_power_kW")
        if ctxt is not None:
            value = float(ctxt[0])
            if value > 0:
                return value
        return self.reference_power_kW

    def get_reward(self, actions, states, info: dict | None = None) -> tuple[float, float]:
        if info is None:
            logger.warning("LongTermEconomicReward: info dict is None, returning 0")
            return 0.0, 0.0

        # Env clears _component_info on reset; missing sentinel ⇒ new episode.
        if self._sentinel_key not in info:
            self._reset_window()
        info[self._sentinel_key] = True

        episode_length = info.get("episode_length")
        if episode_length is None:
            logger.warning("LongTermEconomicReward: missing episode_length in info, returning 0")
            return 0.0, 0.0
        episode_length = int(episode_length)

        net_power_kW = info.get("net_power_kW")
        if net_power_kW is None:
            logger.warning("LongTermEconomicReward: missing net_power_kW in info, returning 0")
            return 0.0, 0.0

        current_energy_price = float(states["s_E_price"][0])

        reference_power_kW = self._resolve_reference_power_kW(states)
        # Same sign convention as EconomicReward.
        raw_step = -net_power_kW * current_energy_price / reference_power_kW
        if net_power_kW < 0:  # export
            raw_step *= self.export_bonus

        self._accum += float(raw_step)
        self._step_in_window += 1

        if self._step_in_window < episode_length:
            return 0.0, 0.0

        max_step = self.weight * float(episode_length)
        window_reward = float(np.clip(self._accum,
                                    -float(episode_length),
                                    float(episode_length)))
        self._reset_window()
        return float(self.weight * window_reward), max_step


ComponentRegistry.register('reward', LongTermEconomicReward)
