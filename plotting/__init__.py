"""Standalone plotting utilities for trajectory analysis.

The modular implementation lives in ``plotting.traj_plotting`` and
``plotting.data_plotting``; this top-level package re-exports the public API.
"""

from .utils import EpisodeData, find_latest_hdf5, load_episode

__all__ = [
    "EpisodeData",
    "find_latest_hdf5",
    "load_episode",
    "plot_states",
    "plot_actions",
    "plot_rewards",
    "plot_energy",
    "generate_all_plots",
]

_TRAJ_EXPORTS = {
    "plot_states", "plot_actions", "plot_rewards", "plot_energy",
    "generate_all_plots",
}


def __getattr__(name: str):
    if name in _TRAJ_EXPORTS:
        from . import traj_plotting as _tp
        return getattr(_tp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
