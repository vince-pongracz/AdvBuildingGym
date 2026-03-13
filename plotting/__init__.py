"""Standalone plotting utilities for trajectory analysis.

The modular implementation lives in ``plotting.src``; this top-level
package re-exports the public API for backwards compatibility.
"""

from .src import (
    EpisodeData,
    find_latest_hdf5,
    generate_all_plots,
    load_episode,
    plot_actions,
    plot_energy,
    plot_rewards,
    plot_states,
)

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
