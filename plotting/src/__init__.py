"""Modular plotting utilities for trajectory analysis.

Submodules
----------
utils        – shared helpers (data loading, discovery, styling)
plot_states  – state variable plots
plot_actions – action dimension plots
plot_rewards – reward breakdown plots
plot_energy  – energy / power plots
"""

from .utils import EpisodeData, find_latest_hdf5, load_episode
from .plot_states import plot_states
from .plot_actions import plot_actions
from .plot_rewards import plot_rewards
from .plot_energy import plot_energy
from .trajectory_plot import generate_all_plots

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
