"""Resolve the dashboard's third-party browser libraries from ``node_modules``.

SortableJS (drag-reorder) and Prism (read-only syntax highlighting) are managed as
npm dependencies declared in ``plotting/dashboard/package.json``. Run ``npm install``
in ``plotting/dashboard/`` once (see README, 'Dashboard assets'); the Python build
reads the pinned files listed here from ``plotting/dashboard/node_modules/`` and
**inlines** them into each dashboard HTML — so the generated file stays fully
self-contained / offline-portable and never needs ``node_modules`` to be *viewed*.

``node_modules`` is gitignored; ``package.json`` + ``package-lock.json`` pin the
versions and are committed.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

NODE_MODULES = Path(__file__).resolve().parent / "node_modules"

# Logical name -> path within node_modules (pinned package layout).
# Prism is loaded as minified core + the JSON/YAML grammars + the line-numbers
# plugin (the npm package ships no bundled prism.min.js — only prism-core.min.js).
ASSET_PATHS: dict[str, str] = {
    "sortable.js": "sortablejs/Sortable.min.js",
    "prism-core.js": "prismjs/components/prism-core.min.js",
    "prism-json.js": "prismjs/components/prism-json.min.js",
    "prism-yaml.js": "prismjs/components/prism-yaml.min.js",
    "prism-line-numbers.js": "prismjs/plugins/line-numbers/prism-line-numbers.min.js",
    "prism.css": "prismjs/themes/prism.min.css",
    "prism-line-numbers.css": "prismjs/plugins/line-numbers/prism-line-numbers.min.css",
}

# Ordered groups the build inlines. Prism core MUST precede its grammars/plugins.
JS_ASSETS: tuple[str, ...] = (
    "sortable.js",
    "prism-core.js",
    "prism-json.js",
    "prism-yaml.js",
    "prism-line-numbers.js",
)
CSS_ASSETS: tuple[str, ...] = ("prism.css", "prism-line-numbers.css")
SORTABLE: tuple[str, ...] = ("sortable.js",)


def path(name: str) -> Path:
    """Absolute path to a vendored asset inside ``node_modules``."""
    return NODE_MODULES / ASSET_PATHS[name]


def require(names: Iterable[str]) -> None:
    """Verify the named assets are installed; raise an actionable error if not.

    This only checks — installation is the user's one-time ``npm install`` step,
    so plot generation never reaches the network.
    """
    missing = [ASSET_PATHS[n] for n in names if not path(n).is_file()]
    if missing:
        raise RuntimeError(
            f"Dashboard third-party libs not installed: {missing}. Run `npm install` "
            "in plotting/dashboard/ (see README, 'Dashboard assets'). These are npm "
            "dependencies (SortableJS, Prism); node_modules/ is not committed to git."
        )


def read(name: str) -> str:
    """Read a vendored asset's text from ``node_modules``."""
    return path(name).read_text(encoding="utf-8")
