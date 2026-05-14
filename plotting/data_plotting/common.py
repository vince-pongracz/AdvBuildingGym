"""Shared constants, config helpers, date utilities, and output writers
for the data-plotting scripts.

Centralises code that was duplicated across ``plot_day_data``,
``plot_monthly_overview``, and ``plot_cross_year_combined``.
"""

from __future__ import annotations

import argparse
import calendar
import logging
import zlib
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import plotly.graph_objects as go
import yaml

from plotting.utils import ensure_chrome_for_kaleido, write_figure_list_html

from .loaders import set_warn_future_data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "data_plot_config.yaml"

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

WEATHER_DATASETS = ["dwd", "zenodo"]
PRICE_DATASETS = ["awattar", "e_charts"]

# Distinct colour per dataset — hue separated, semi-transparent for bands.
DATASET_STYLES: dict[str, dict] = {
    "dwd":      {"color": "#636EFA", "rgba_std": "rgba(99,110,250,0.20)",  "rgba_mm": "rgba(99,110,250,0.08)"},
    "zenodo":   {"color": "#EF553B", "rgba_std": "rgba(239,85,59,0.20)",   "rgba_mm": "rgba(239,85,59,0.08)"},
    "awattar":  {"color": "#00CC96", "rgba_std": "rgba(0,204,150,0.20)",   "rgba_mm": "rgba(0,204,150,0.08)"},
    "e_charts": {"color": "#AB63FA", "rgba_std": "rgba(171,99,250,0.20)",  "rgba_mm": "rgba(171,99,250,0.08)"},
}

# Distinct colours for synthesised configurations, kept clear of the
# DATASET_STYLES hues above so the original-vs-synthesised distinction stays
# visually obvious. Cycled deterministically by syn_cfg name.
_SYN_CFG_PALETTE: list[tuple[str, str]] = [
    ("#FFA15A", "255,161,90"),    # orange
    ("#19D3F3", "25,211,243"),    # cyan
    ("#FF6692", "255,102,146"),   # pink
    ("#B6E880", "182,232,128"),   # lime
    ("#FECB52", "254,203,82"),    # yellow
    ("#FF9DA6", "255,157,166"),   # salmon
    ("#1F77B4", "31,119,180"),    # steel blue
    ("#BCBD22", "188,189,34"),    # olive
]


def syn_cfg_style(cfg_name: str, *, index: int | None = None) -> dict[str, str]:
    """Return ``{color, rgba_std, rgba_mm}`` for a syn_cfg trace group.

    Uses a deterministic hash of *cfg_name* so the same cfg always renders
    in the same colour across plots, unless an explicit *index* overrides
    the assignment (useful when cycling through a known list).
    """
    if index is None:
        # zlib.crc32 is stable across processes, unlike builtin hash().
        index = zlib.crc32(cfg_name.encode("utf-8")) % len(_SYN_CFG_PALETTE)
    hex_color, rgb = _SYN_CFG_PALETTE[index % len(_SYN_CFG_PALETTE)]
    return {
        "color": hex_color,
        "rgba_std": f"rgba({rgb},0.20)",
        "rgba_mm": f"rgba({rgb},0.08)",
    }


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load a YAML config file and return the parsed dict."""
    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def get_data_figure_config() -> dict:
    """Return the ``figure`` section of the default data-plot config.

    Cached so figure_builders can call it cheaply per figure. Mirrors the
    pattern in ``plotting.utils._fig_cfg`` — always reads ``DEFAULT_CONFIG``,
    not whatever ``--config`` was passed on the CLI.
    """
    try:
        cfg = load_config(DEFAULT_CONFIG)
    except OSError:
        return {}
    return (cfg.get("figure") or {})


def resolve_source(cfg_section: dict, section_name: str, key: str | None = None) -> dict | None:
    """Resolve a dataset inside a config section.

    When *key* is given, return the sub-dict for that specific dataset
    (or ``None`` if absent).

    When *key* is ``None``, use the ``use`` selector in the section.
    Falls back to the flat layout (dir, file_pattern, timestamp_col at the
    top level) when no ``use`` key exists.
    """
    if key is not None:
        if key not in cfg_section:
            return None
        return cfg_section[key]

    if "use" not in cfg_section:
        return cfg_section

    selected = cfg_section["use"]
    if selected not in cfg_section:
        available = [k for k in cfg_section if k != "use"]
        raise KeyError(
            f"{section_name}.use = {selected!r} but available datasets are: "
            + ", ".join(available)
        )
    return cfg_section[selected]


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def days_in_month(year: int, month: int) -> list[datetime]:
    """Return a datetime for every day in a given year-month."""
    n_days = calendar.monthrange(year, month)[1]
    return [datetime(year, month, d) for d in range(1, n_days + 1)]


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------

def load_profiles_from_cfg(
    cfg: dict,
    dates: list[datetime],
) -> dict[str, dict[str, object]]:
    """Load profile sources (desired_temp_in, ev_schedule, user_energy_need).

    Returns ``{source_name: {label: DataFrame}}``.
    """
    from .loaders import load_days, load_profiles

    sources: dict[str, dict[str, object]] = {}

    if "desired_temp_in" in cfg:
        dt_cfg = cfg["desired_temp_in"]
        sources["desired_temp_in"] = load_profiles(
            REPO_ROOT / dt_cfg["dir"],
            dt_cfg["files"],
            dt_cfg["timestamp_col"],
        )

    if "ev_schedule" in cfg:
        ev_cfg = cfg["ev_schedule"]
        sources["ev_schedule"] = load_profiles(
            REPO_ROOT / ev_cfg["dir"],
            ev_cfg["files"],
            ev_cfg["timestamp_col"],
        )

    if "user_energy_need" in cfg:
        ue_cfg = cfg["user_energy_need"]
        ue_dir = REPO_ROOT / ue_cfg["dir"]
        pattern = ue_cfg["file_pattern"]
        frames: dict[str, object] = {}
        for profile in ue_cfg["profiles"]:
            profile_pattern = pattern.replace("{profile}", profile)
            day_frames = load_days(ue_dir, profile_pattern, ue_cfg["timestamp_col"], dates)
            for date_label, df in day_frames.items():
                frames[f"{profile} ({date_label})"] = df
        sources["user_energy_need"] = frames

    return sources


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_output(
    figures: list[go.Figure],
    out_dir: Path,
    base_name: str,
    output_formats: list[str],
) -> None:
    """Write figures to disk in the requested formats."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if "html" in output_formats:
        html_path = out_dir / f"{base_name}.html"
        write_figure_list_html(figures, str(html_path))
        logger.info("Wrote %s", html_path)

    static_formats = [fmt for fmt in output_formats if fmt != "html"]
    if static_formats:
        ensure_chrome_for_kaleido()
        for fmt in static_formats:
            for i, fig in enumerate(figures):
                title_obj = fig.layout.title
                title = getattr(title_obj, "text", None) or str(title_obj) or f"fig{i}"

                tag = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
                img_path = out_dir / f"{base_name}_{tag}.{fmt}"
                fig.write_image(str(img_path))
                logger.info("Wrote %s", img_path)


# ---------------------------------------------------------------------------
# Shared CLI plumbing
# ---------------------------------------------------------------------------

def add_shared_cli_args(parser: argparse.ArgumentParser) -> None:
    """Attach --config, --format, --warn-future-data to a data-plotting parser."""
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to data_plot_config.yaml (default: %(default)s).",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        default=["html"],
        choices=["html", "png", "svg", "pdf"],
        help="Output format(s). Default: html.",
    )
    parser.add_argument(
        "--warn-future-data",
        action="store_true",
        default=False,
        help="Emit 'No data for <date>' warnings for dates that have not yet "
             "occurred. Off by default — future dates are silently skipped.",
    )


def apply_shared_cli_args(args: argparse.Namespace, script_name: str) -> None:
    """Configure logging and apply the --warn-future-data toggle."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info(
        "%s started at %s",
        script_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    set_warn_future_data(args.warn_future_data)
    if not args.warn_future_data:
        logger.info(
            "Future-date warnings are OFF: 'No data for <date>' messages "
            "will be suppressed for dates after today. Pass --warn-future-data to enable."
        )
