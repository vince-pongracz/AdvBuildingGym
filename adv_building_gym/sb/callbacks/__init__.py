"""SB3 callbacks mirroring adv_building_gym.ray.callbacks.

These all read per-env ``infos`` published by ``AdvBuildingGym.step()``
(``reward_breakdown``, ``max_reward_step``, ``cum_E_kWh``,
``cum_price_EUR``, ``episode_count``) and write through the SB3
``Logger`` API (``self.logger.record(...)``). Schedule callbacks reach
sub-envs via ``vec_env.env_method("apply_data_variant", ...)`` which
goes through gymnasium's ``get_wrapper_attr`` chain.
"""

from .episode_metrics_callback import SBEpisodeMetricsCallback
from .best_checkpoint_callback import SBBestCheckpointCallback
from .eval_state_action_callback import SBEvalStateActionCallback
from .iter_timing_callback import SBIterTimingCallback, wrap_eval_callback_with_timer
from .data_schedule_callback import make_data_schedule_callback
from .reward_switch_callback import make_reward_switch_callback
from .infra_schedule_callback import make_infra_schedule_callback
from .statesource_schedule_callback import make_statesource_schedule_callback

__all__ = [
    "SBEpisodeMetricsCallback",
    "SBBestCheckpointCallback",
    "SBEvalStateActionCallback",
    "SBIterTimingCallback",
    "wrap_eval_callback_with_timer",
    "make_data_schedule_callback",
    "make_reward_switch_callback",
    "make_infra_schedule_callback",
    "make_statesource_schedule_callback",
]
