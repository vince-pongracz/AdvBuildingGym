"""Self-contained, offline 3-panel evaluation dashboard for a single episode.

See ``build_dashboard.write_episode_dashboard_html`` for the entry point. The
output is one HTML file that inlines Plotly.js, SortableJS and Prism plus all
figure JSON, so it opens from ``file://`` with no server and no network.
"""

from .build_dashboard import write_episode_dashboard_html

__all__ = ["write_episode_dashboard_html"]
