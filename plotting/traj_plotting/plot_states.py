"""Plot state variables from an episode trajectory."""

from __future__ import annotations

import logging

import numpy as np
import plotly.graph_objects as go

from plotting.utils import COLORS, EpisodeData, apply_day_xaxis, load_plot_config, style_figure

logger = logging.getLogger(__name__)


def plot_states(episode: EpisodeData) -> list[go.Figure]:
    """One independent plot per state variable (or group of variables).

    Keys listed in ``grouped_keys`` (from ``traj_plot_config.yaml``) are merged
    into a single plot.
    Returns a list of figures to be rendered sequentially in one HTML file.
    """
    plot_config = load_plot_config().get("states", {})
    skip_keys: set[str] = set(plot_config.get("skip_keys", []))
    # ``ctxt_`` state keys are static physical context (e.g. battery capacity);
    # they are constant within an episode, so they are excluded from plots.
    ctxt_prefix = "ctxt_"
    grouped_keys: list[list[str]] = plot_config.get("grouped_keys", [])

    states = episode.states
    time = episode.time_minutes
    suffix = episode.title_suffix()
    time_hhmm = episode.time_hhmm

    # Mask EV-related keys with NaN when ev_connected < 0.5
    # so Plotly renders non-continuous lines (gaps when disconnected).
    mask_cfg = plot_config.get("mask_when_disconnected", {})
    cond_key = mask_cfg.get("condition_key")
    mask_keys: set[str] = set(mask_cfg.get("keys", []))
    disconnected_mask: np.ndarray | None = None
    if cond_key and cond_key in states:
        cond = states[cond_key]
        disconnected_mask = (cond < 0.5) if cond.ndim == 1 else (cond[:, 0] < 0.5)

    # Build ordered list of plot specs.  Each entry is a list of keys.
    grouped_flat = {k for group in grouped_keys for k in group}
    plot_specs: list[list[str]] = []

    seen_groups: set[int] = set()
    for key, val in states.items():
        if key in skip_keys or key.startswith(ctxt_prefix) or val.ndim != 1:
            continue
        if key in grouped_flat:
            for gi, group in enumerate(grouped_keys):
                if key in group and gi not in seen_groups:
                    present = [k for k in group if k in states]
                    missing = [k for k in group if k not in states]
                    if missing:
                        logger.info(
                            "Grouped key(s) %s missing from data, skipping.",
                            missing,
                        )
                    if present:
                        plot_specs.append(present)
                        seen_groups.add(gi)
        else:
            plot_specs.append([key])

    # Multi-dim keys (2-D with few columns)
    for key, val in states.items():
        if key in skip_keys or key.startswith(ctxt_prefix) or key in grouped_flat:
            continue
        if val.ndim == 2 and val.shape[1] <= 4:
            plot_specs.append([key])

    if not plot_specs:
        logger.warning("No plottable state variables found.")
        return []

    figures: list[go.Figure] = []
    for keys in plot_specs:
        fig = go.Figure()
        title = " + ".join(keys)
        all_data: list[np.ndarray] = []
        trace_idx = 0

        for key in keys:
            if key not in states:
                logger.info("State key '%s' missing from data, skipping.", key)
                continue
            arr = states[key].copy()

            # Replace values with NaN when disconnected (non-continuous line)
            if key in mask_keys and disconnected_mask is not None:
                if arr.ndim == 1:
                    arr = arr.astype(np.float64)
                    arr[disconnected_mask] = np.nan
                elif arr.ndim == 2:
                    arr = arr.astype(np.float64)
                    arr[disconnected_mask, :] = np.nan

            if arr.ndim == 1:
                fig.add_trace(go.Scatter(
                    x=time, y=arr, mode="lines",
                    name=key,
                    line=dict(color=COLORS[trace_idx % len(COLORS)]),
                    customdata=time_hhmm,
                    hovertemplate=(
                        f"{key}<br>"
                        "time: %{customdata}<br>"
                        "value: %{y:.4f}"
                        "<extra></extra>"
                    ),
                ))
                all_data.append(arr)
                trace_idx += 1
            elif arr.ndim == 2:
                for col_idx in range(arr.shape[1]):
                    label = f"{key}[{col_idx}]"
                    fig.add_trace(go.Scatter(
                        x=time, y=arr[:, col_idx], mode="lines",
                        name=label,
                        line=dict(
                            color=COLORS[trace_idx % len(COLORS)],
                        ),
                        customdata=time_hhmm,
                        hovertemplate=(
                            f"{label}<br>"
                            "time: %{customdata}<br>"
                            "value: %{y:.4f}"
                            "<extra></extra>"
                        ),
                    ))
                    trace_idx += 1
                all_data.append(arr)

        if all_data:
            combined = np.concatenate([d.ravel() for d in all_data])
            if np.all(np.isnan(combined)):
                # All values masked (e.g. EV disconnected entire episode)
                fig.update_yaxes(range=[0.0, 1.0])
            else:
                y_lo = -1.0 if float(np.nanmin(combined)) < 0 else 0.0
                fig.update_yaxes(
                    range=[min(y_lo, float(np.nanmin(combined))),
                           max(1.0, float(np.nanmax(combined)))],
                )

        apply_day_xaxis(fig)
        fig.update_layout(title=f"{title}  —  {suffix}", height=350)
        figures.append(style_figure(fig, n_legend_items=trace_idx))

    return figures
