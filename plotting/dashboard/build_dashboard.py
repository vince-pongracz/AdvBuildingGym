"""Build a self-contained, offline 3-panel evaluation dashboard for one episode.

Aggregates every trajectory figure (states, actions, rewards, energy, raw, …) onto
a single IDE-like page:

  * left   — a 2-level ticker tree (group → per-subplot key names) to toggle plots,
  * middle — a reorderable flexbox of plot cards (drag a card header to reorder,
             drag a card corner to resize); multiple plots per row,
  * right  — read-only ``manifest.json`` (top) over the trial-config YAML (bottom),
             both syntax-highlighted and independently scrollable.

The page inlines Plotly.js (once), SortableJS and Prism plus all figure JSON and the
manifest/config text, so it opens from ``file://`` with no server and no network.
"""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as po

from plotting.dashboard import vendor_assets
from plotting.utils import (
    EpisodeData,
    PLOT_CARD_CSS,
    get_width_multiplier,
    load_plot_config,
    short_label_from_fig,
)

# Horizontal gap (px) between plot cards; mirrors the ``gap`` in PLOT_CARD_CSS.
_CARD_GAP_PX = 14

logger = logging.getLogger(__name__)

_ASSETS = Path(__file__).resolve().parent / "assets"


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _js_string(text: str) -> str:
    """Serialise ``text`` as a JS string literal safe to embed inside ``<script>``.

    ``json.dumps`` handles quoting/escaping; replacing ``</`` with ``<\\/`` prevents
    an embedded ``</script>`` from prematurely closing the script element.
    """
    return json.dumps(text).replace("</", "<\\/")


def _js_json(obj) -> str:
    """Serialise ``obj`` to a compact JSON literal safe to embed inside ``<script>``."""
    return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")


def _asset(name: str) -> str:
    return (_ASSETS / name).read_text(encoding="utf-8")


def _root_vars_css() -> str:
    """CSS custom properties driving layout density, read from traj_plot_config.yaml.

    ``--card-basis`` sets the default flex width so ``default_per_row`` cards fit a
    row (each still drag-resizable); ``--left-w`` / ``--right-w`` size the panels.
    """
    cfg = load_plot_config().get("dashboard") or {}
    per_row = max(1, int(cfg.get("default_per_row", 2)))
    left = int(cfg.get("left_panel_px", 260))
    right = int(cfg.get("right_panel_px", 400))
    basis = f"calc((100% - {(per_row - 1) * _CARD_GAP_PX}px) / {per_row})"
    return f":root{{--card-basis:{basis};--left-w:{left}px;--right-w:{right}px;}}"


def _safe_read(path: Path | None) -> str:
    if path is not None and Path(path).is_file():
        return Path(path).read_text(encoding="utf-8")
    return ""


# ---------------------------------------------------------------------------
# Manifest / config source discovery
# ---------------------------------------------------------------------------

def _walk_up(start: Path, max_levels: int = 9):
    """Yield ``start`` and its parents up to ``max_levels`` (bounded so we never
    wander above the repo into unrelated directories)."""
    p = Path(start).resolve()
    chain = [p, *p.parents]
    return chain[:max_levels]


def _resolve_manifest(output_dir: Path) -> Path | None:
    """Return the nearest ``manifest.json`` walking up from ``output_dir``.

    Present for snapshot runs (``snapshots/<snap>/manifest.json``); absent for a
    plain ``eval_results/<ts>_eval`` run, in which case the pane is hidden.
    """
    for parent in _walk_up(output_dir):
        cand = parent / "manifest.json"
        if cand.is_file():
            return cand
    return None


def _resolve_config(output_dir: Path, manifest_path: Path | None) -> Path | None:
    """Return the trial-config YAML used for this eval.

    Primary source: the copy written into the eval-results dir next to the
    ``eval_*.json`` summary (everything but ``provenance.yaml``). Fallback: the
    snapshot's ``code/<source_trial_path>`` resolved from the manifest.
    """
    for parent in _walk_up(output_dir):
        if any(parent.glob("eval_*.json")):
            yamls = sorted(
                y for y in parent.glob("*.yaml") if y.name != "provenance.yaml"
            )
            if yamls:
                return yamls[0]
            break

    if manifest_path is not None:
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            src = manifest.get("source_trial_path")
            if src:
                cand = Path(manifest_path).parent / "code" / src
                if cand.is_file():
                    return cand
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return None


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------

def _build_payload(
    all_figures: dict[str, list[go.Figure]],
    footnotes: dict[str, str],
    episode: EpisodeData,
) -> dict:
    """Serialise every figure to a plotly JSON spec, grouped by plot group.

    ``wide`` mirrors the per-group width multiplier (energy bar charts get a full
    row). Capturing ``height`` here keeps the card's initial size even though the
    spec's ``layout.height`` is stripped client-side for responsive sizing.
    """
    groups: list[dict] = []
    for name, figs in all_figures.items():
        wide = get_width_multiplier(name) > 1.0
        figures: list[dict] = []
        for fig in figs:
            spec = json.loads(pio.to_json(fig))
            height = int(fig.layout.height) if fig.layout.height else 450
            title = ""
            if fig.layout.title is not None and fig.layout.title.text:
                title = fig.layout.title.text
            figures.append({
                "label": short_label_from_fig(fig),
                "title": title,
                "height": height,
                "wide": wide,
                "spec": spec,
            })
        groups.append({
            "name": name,
            "footnote": footnotes.get(name, ""),
            "figures": figures,
        })

    ep = episode
    return {
        "episode": {
            "id": ep.episode_id,
            "date": ep.episode_date,
            "seed": ep.seed,
            "length": ep.length,
            "achieved_reward": ep.summary.get("achieved_reward"),
            "cum_E_kWh": ep.summary.get("cum_E_kWh"),
            "cum_price_EUR": ep.summary.get("cum_price_EUR"),
        },
        "groups": groups,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def write_episode_dashboard_html(
    all_figures: dict[str, list[go.Figure]],
    footnotes: dict[str, str],
    episode: EpisodeData,
    filepath: str,
    *,
    output_dir: str | None = None,
    manifest_path: str | None = None,
    config_path: str | None = None,
) -> str:
    """Write the aggregated 3-panel dashboard for one episode to ``filepath``.

    ``all_figures`` / ``footnotes`` are exactly the structures assembled in
    ``generate_all_plots``. Manifest and config sources are auto-discovered from
    ``output_dir`` (defaults to the output file's directory) unless given explicitly.
    Returns the written path.
    """
    base = Path(output_dir) if output_dir else Path(filepath).parent

    manifest = Path(manifest_path) if manifest_path else _resolve_manifest(base)
    config = Path(config_path) if config_path else _resolve_config(base, manifest)
    manifest_text = _safe_read(manifest)
    config_text = _safe_read(config)
    logger.debug("Dashboard sources — manifest: %s, config: %s", manifest, config)

    payload = _build_payload(all_figures, footnotes, episode)

    # Third-party libs come from `npm install` (node_modules, not committed); the
    # build reads + inlines them so the output HTML stays self-contained. See README.
    vendor_assets.require(vendor_assets.CSS_ASSETS + vendor_assets.JS_ASSETS)
    prism_css = "\n".join(vendor_assets.read(n) for n in vendor_assets.CSS_ASSETS)
    vendor_js = "\n".join(vendor_assets.read(n) for n in vendor_assets.JS_ASSETS)
    data_js = (
        "var DASHBOARD_DATA=" + _js_json(payload) + ";\n"
        "var MANIFEST_TEXT=" + _js_string(manifest_text) + ";\n"
        "var CONFIG_TEXT=" + _js_string(config_text) + ";\n"
    )
    page_title = f"Episode {episode.episode_id} — evaluation dashboard"

    html_out = (
        _asset("dashboard.html")
        .replace("{{PAGE_TITLE}}", html.escape(page_title))
        .replace("{{CARD_CSS}}", PLOT_CARD_CSS)
        .replace("{{ROOT_VARS}}", _root_vars_css())
        .replace("{{PRISM_CSS}}", prism_css)
        .replace("{{APP_CSS}}", _asset("dashboard.css"))
        .replace("{{PLOTLY_JS}}", po.get_plotlyjs())
        .replace("{{VENDOR_JS}}", vendor_js)
        .replace("{{DATA_JS}}", data_js)
        .replace("{{APP_JS}}", _asset("dashboard.js"))
    )

    Path(filepath).write_text(html_out, encoding="utf-8")
    logger.info("Saved dashboard: %s", filepath)
    return filepath
