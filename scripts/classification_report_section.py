# Optional "Classification utility" section for the linkage / MIA attack reports.
#
# When a per-experiment JSON produced by run_classification_benchmark.py is
# available alongside an attack run, the attack report should embed a
# Classification utility section. When it is not available, no section is added
# and the rest of the report is unchanged.
#
# This module DELEGATES the actual HTML / Chart.js rendering to the colleague's
# ``generate_anonymization_report.build_clf_detail_sections`` function so that
# the rendering style stays consistent across the anonymization report and the
# attack reports. We re-use that function unchanged by:
#   1. Locating the per-experiment classification JSON.
#   2. Building a small temporary directory tree that mirrors the conventional
#      layout (``outputs/classification/<exp>_classification.json`` and any
#      ``outputs/figures/<exp>/`` PNGs), but containing ONLY the JSON for the
#      current attack — so the colleague's function, which scans
#      ``outputs/classification/*.json``, naturally renders only that one
#      experiment.
#   3. Calling ``build_clf_detail_sections`` on that temporary tree.
#   4. Rewriting the wrapper ``<section>`` tag to fit the attack report's own
#      section numbering and anchor.
#
# The colleague's section relies on a few CSS classes and JS helpers
# (``.chart-wrap-tall``, ``.btn-reset``, the ``_charts`` registry, ``zoomPlugin``,
# ``resetZoom``) plus two extra CDN scripts (``hammer.js``, ``chartjs-plugin-zoom``).
# We expose those bits as constants so the attack report generators can inject
# them when (and only when) the section is rendered.

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import generate_anonymization_report as _gar


# ---------------------------------------------------------------------------
# Extra CSS / JS / head scripts required by the colleague's classification
# rendering. They are safe to inject regardless of whether the section is
# present (the CSS classes simply go unused, the JS helpers do nothing without
# any registered chart), but to keep the no-classification HTML byte-identical
# to the previous behavior we still gate the injection on the section being
# present.
# ---------------------------------------------------------------------------

CLF_EXTRA_CSS = """
  .chart-wrap-tall { position: relative; height: 380px; }
  .btn-reset {
    margin-top: 8px;
    padding: 4px 12px;
    font-size: .78rem;
    font-weight: 600;
    color: var(--accent);
    background: var(--accent-soft);
    border: 1px solid rgba(122,162,255,.3);
    border-radius: 999px;
    cursor: pointer;
  }
  .btn-reset:hover { background: rgba(122,162,255,.22); }
"""

# These two scripts must be loaded BEFORE the page's main <script> block so
# that the zoom plugin is registered by the time chart instances are created.
CLF_EXTRA_HEAD_SCRIPTS = (
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/hammer.js/2.0.8/hammer.min.js"></script>\n'
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-zoom/2.0.1/chartjs-plugin-zoom.min.js"></script>'
)

# JS helpers that the colleague's chart blocks reference: ``_charts`` (chart
# registry), ``zoomPlugin`` (Chart.js options shared across charts) and
# ``resetZoom`` (called by the per-chart "Reset zoom" buttons).
CLF_EXTRA_JS_HELPERS = """
const _charts = (typeof window !== 'undefined' && window._charts) || {};
const zoomPlugin = {
  zoom: {
    wheel: { enabled: true },
    pinch: { enabled: true },
    mode: 'xy',
  },
  pan: {
    enabled: true,
    mode: 'xy',
  },
};
function resetZoom(id) { if (_charts[id]) _charts[id].resetZoom(); }
"""


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _experiment_id_from_attack_id(attack_id: str) -> str | None:
    """Extract the anonymization experiment_id from a linkage / MIA attack_id.

    Linkage attack_ids follow the pattern  ``<experiment_id>__known_<...>``.
    MIA attack_ids follow the pattern      ``<experiment_id>__mia_<...>``.
    A few legacy linkage runs use         ``<experiment_id>__atk_<...>``.
    """
    if not attack_id:
        return None
    for sep in ("__mia_", "__known_", "__atk_"):
        if sep in attack_id:
            return attack_id.split(sep)[0]
    return None


def resolve_classification_json(
    *,
    cli_path: Path | None,
    project_root: Path,
    summary: dict[str, Any],
    attack_dir: Path | None = None,
) -> Path | None:
    """Resolve the path to the classification JSON for this attack.

    Resolution order:
      1. Explicit CLI path (``--classification-json``).
      2. Path stored in summary.json under the key
         ``classification_results_json`` (set by future runners).
      3. Conventional location:
         ``<project_root>/outputs/classification/<experiment_id>_classification.json``
         where ``experiment_id`` is derived from ``summary["attack_id"]``.
      4. Fallback inside the attack directory (rare).
    Returns ``None`` if nothing exists.
    """
    if cli_path is not None:
        candidate = Path(cli_path).resolve()
        if candidate.exists():
            return candidate

    summary_path = summary.get("classification_results_json")
    if summary_path:
        candidate = Path(str(summary_path))
        if candidate.exists():
            return candidate

    attack_id = str(summary.get("attack_id") or "")
    experiment_id = _experiment_id_from_attack_id(attack_id)
    if experiment_id:
        candidate = (
            project_root / "outputs" / "classification"
            / f"{experiment_id}_classification.json"
        )
        if candidate.exists():
            return candidate

    if attack_dir is not None:
        fallback = attack_dir / "classification_results.json"
        if fallback.exists():
            return fallback

    return None


# ---------------------------------------------------------------------------
# Section rendering — delegated to the colleague's renderer
# ---------------------------------------------------------------------------

# Pattern used to rewrite the colleague's hard-coded section header so that
# the attack report can use its own numbering and anchor. The colleague's
# wrapper opens with:
#     <section id="clf-detail" class="card" ...>
#       <h2 class="section-title">4. Classification — Per-experiment detail</h2>
#       <p class="section-subtitle">...</p>
# We replace those three opening lines, keeping everything inside the section
# intact (per-model cards, ROC charts, etc.).
_WRAPPER_OPEN_RE = re.compile(
    r'<section\s+id="clf-detail"\s+class="card"[^>]*>\s*'
    r'<h2\s+class="section-title">[^<]*</h2>\s*'
    r'<p\s+class="section-subtitle">.*?</p>',
    re.DOTALL,
)


def _isolate_experiment(
    classification_json: Path,
    experiment_id: str,
    project_root: Path,
    tmp_root: Path,
) -> None:
    """Mirror the conventional layout for a single experiment under ``tmp_root``.

    Copies the classification JSON and any matching PNG figures so that the
    colleague's renderer (which scans ``outputs/classification/`` and embeds
    PNGs from ``outputs/figures/<exp>/``) finds exactly one experiment.
    """
    fake_clf_dir = tmp_root / "outputs" / "classification"
    fake_clf_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(classification_json, fake_clf_dir / f"{experiment_id}_classification.json")

    figures_src = project_root / "outputs" / "figures" / experiment_id
    if figures_src.is_dir():
        figures_dst = tmp_root / "outputs" / "figures" / experiment_id
        figures_dst.mkdir(parents=True, exist_ok=True)
        for png in figures_src.iterdir():
            if png.is_file():
                shutil.copy2(png, figures_dst / png.name)


def build_classification_section(
    classification_json: Path | None,
    *,
    section_number: int = 9,
    anchor_id: str = "classification",
    project_root: Path | None = None,
) -> tuple[str, str]:
    """Return ``(html_section, chart_js_init)`` for the classification section.

    Both strings are empty when ``classification_json`` is ``None`` or fails to
    load. The HTML is produced by the colleague's ``build_clf_detail_sections``
    function so that the visual style matches the anonymization report exactly.

    Parameters
    ----------
    classification_json : the per-experiment ``*_classification.json`` to
        render. If ``None``, returns ``("", "")``.
    section_number      : the section number to display in the rewritten
        header (e.g. 9 in the attack reports).
    anchor_id           : the HTML anchor id to use for the section so that
        the report's nav can link to it.
    project_root        : project root used to locate matching PNG figures
        under ``<project_root>/outputs/figures/<experiment_id>/``. Defaults
        to the parent of the JSON file's grandparent (i.e. the conventional
        ``<project_root>/outputs/classification/<file>.json`` layout).
    """
    if classification_json is None:
        return "", ""

    classification_json = Path(classification_json)
    if not classification_json.exists():
        return "", ""

    # Conventional layout: <project_root>/outputs/classification/<file>.json
    if project_root is None:
        try:
            project_root = classification_json.resolve().parents[2]
        except IndexError:
            project_root = classification_json.parent

    experiment_id = classification_json.stem
    if experiment_id.endswith("_classification"):
        experiment_id = experiment_id[: -len("_classification")]

    with tempfile.TemporaryDirectory(prefix="clf_section_") as tmp:
        tmp_root = Path(tmp)
        try:
            _isolate_experiment(classification_json, experiment_id, project_root, tmp_root)
        except OSError:
            return "", ""

        try:
            html, js_blocks = _gar.build_clf_detail_sections(tmp_root, [0])
        except Exception:
            # If anything goes wrong inside the colleague's renderer, we fall
            # back to no section rather than break the attack report.
            return "", ""

    if not html:
        return "", ""

    # Rewrite the colleague's hard-coded "4. Classification — Per-experiment
    # detail" header to fit the attack report's own numbering / anchor.
    new_header = (
        f'<section id="{anchor_id}" class="card" style="margin-bottom:22px;">\n'
        f'  <h2 class="section-title">{int(section_number)}. Classification utility</h2>\n'
        f'  <p class="section-subtitle">'
        f'Classification-based utility benchmark for the anonymization configuration '
        f'of this attack: per-class metrics (Sensitivity, Specificity, Brier, AUC) and '
        f'ROC curves on <span style="color:#2563eb">Input</span> vs '
        f'<span style="color:#16a34a">Output</span> data. '
        f'Provided by <code>run_classification_benchmark.py</code>.'
        f'</p>'
    )
    html_rewritten, n = _WRAPPER_OPEN_RE.subn(new_header, html, count=1)
    if n == 0:
        # Pattern did not match — fall back to using the colleague's HTML
        # as-is so the section is still visible.
        html_rewritten = html

    chart_js = "\n".join(js_blocks)
    return html_rewritten, chart_js
