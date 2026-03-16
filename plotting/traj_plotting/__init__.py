"""Modular plotting utilities for trajectory analysis.

Submodules
----------
utils        – shared helpers (data loading, discovery, styling)
plot_states  – state variable plots
plot_actions – action dimension plots
plot_rewards – reward breakdown plots
plot_energy  – energy / power plots
"""

from plotting.utils import EpisodeData, find_latest_hdf5, load_episode
from .plot_states import plot_states
from .plot_actions import plot_actions
from .plot_rewards import plot_rewards
from .plot_energy import plot_energy

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


def __getattr__(name: str):
    # Lazy import to avoid RuntimeWarning when running
    # `python -m plotting.traj_plotting.trajectory_plot`
    if name == "generate_all_plots":
        from .trajectory_plot import generate_all_plots
        return generate_all_plots
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
