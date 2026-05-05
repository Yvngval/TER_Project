
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from scipy.stats import ks_2samp  # type: ignore
except Exception:  # pragma: no cover
    ks_2samp = None

from classification_report_section import (
    build_classification_section,
    resolve_classification_json,
)


STYLE = r"""
  :root {
    --bg: #ffffff;
    --panel: #ffffff;
    --panel-2: #f6f8fa;
    --line: #d0d7de;
    --text: #1f2328;
    --muted: #57606a;
    --accent: #0969da;
    --accent-soft: rgba(9, 105, 218, 0.08);
    --good: #1a7f37;
    --good-soft: rgba(26, 127, 55, 0.12);
    --warn: #9a6700;
    --warn-soft: rgba(154, 103, 0, 0.12);
    --danger: #cf222e;
    --danger-soft: rgba(207, 34, 46, 0.12);
    --radius: 18px;
    --shadow: 0 4px 14px rgba(31, 35, 40, 0.06);
    --font: Inter, system-ui, sans-serif;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0;
    font-family: var(--font);
    color: var(--text);
    background:
      radial-gradient(circle at top right, rgba(9, 105, 218, 0.05), transparent 24%),
      #ffffff;
    line-height: 1.65;
  }
  header {
    padding: 52px 28px 36px;
    border-bottom: 1px solid var(--line);
    background: linear-gradient(135deg, rgba(9, 105, 218, 0.06), rgba(9, 105, 218, 0) 45%);
  }
  .wrap { max-width: 1220px; margin: 0 auto; }
  .eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    background: var(--accent-soft);
    color: var(--accent);
    border: 1px solid rgba(9, 105, 218, 0.22);
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .04em;
    text-transform: uppercase;
  }
  h1 {
    margin: 18px 0 12px;
    font-size: clamp(2rem, 4vw, 3rem);
    line-height: 1.08;
    letter-spacing: -0.03em;
  }
  .lead { max-width: 920px; color: var(--muted); font-size: 1rem; }
  nav {
    position: sticky;
    top: 0;
    z-index: 10;
    backdrop-filter: blur(10px);
    background: rgba(255, 255, 255, 0.88);
    border-bottom: 1px solid var(--line);
  }
  nav .wrap { display: flex; gap: 18px; flex-wrap: wrap; padding: 14px 28px; }
  nav a { color: var(--muted); text-decoration: none; font-size: 13px; font-weight: 600; }
  nav a:hover { color: var(--accent); }
  main { padding: 28px; }
  section { margin: 0 0 22px; }
  .card {
    background-color: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 24px;
  }
  .section-title { margin: 0 0 16px; font-size: 1.55rem; letter-spacing: -0.02em; }
  .section-subtitle { margin: -4px 0 18px; color: var(--muted); font-size: .96rem; }
  .grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }
  .metric {
    background: var(--panel-2);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 16px 16px 14px;
  }
  .metric .k {
    color: var(--muted);
    font-size: .76rem;
    text-transform: uppercase;
    letter-spacing: .06em;
  }
  .metric .v {
    margin-top: 6px;
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.03em;
  }
  .good { color: var(--good); }
  .warn { color: var(--warn); }
  .danger { color: var(--danger); }
  .accent { color: var(--accent); }
  .two { display: grid; gap: 16px; grid-template-columns: 1fr 1fr; }
  .callout {
    border-left: 4px solid var(--accent);
    background: var(--accent-soft);
    border-radius: 0 16px 16px 0;
    padding: 14px 16px;
    color: var(--text);
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: .9rem;
    overflow: hidden;
    border-radius: 16px;
    background: var(--panel-2);
    border: 1px solid var(--line);
  }
  th, td {
    padding: 11px 14px;
    border-bottom: 1px solid var(--line);
    text-align: left;
    vertical-align: top;
  }
  tr:last-child td { border-bottom: none; }
  th {
    color: var(--accent);
    font-size: .75rem;
    letter-spacing: .05em;
    text-transform: uppercase;
    background: rgba(9, 105, 218, 0.06);
  }
  code {
    font-family: var(--mono);
    font-size: .82rem;
    color: var(--accent);
    background: rgba(9, 105, 218, 0.08);
    padding: 2px 6px;
    border-radius: 8px;
  }
  .small { color: var(--muted); font-size: .88rem; }
  .chart-card {
    background: var(--panel-2);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 16px;
  }
  .chart-wrap { position: relative; height: 340px; }
  .cm {
    display: grid;
    grid-template-columns: 110px 1fr 1fr;
    gap: 10px;
    align-items: stretch;
    font-size: .92rem;
  }
  .cm .label {
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--muted);
    font-weight: 600;
  }
  .cm .cell {
    border-radius: 16px;
    border: 1px solid var(--line);
    padding: 16px;
    min-height: 98px;
    background: var(--panel-2);
  }
  .cm .cell strong {
    display: block;
    font-size: 1.6rem;
    line-height: 1.1;
    margin-bottom: 6px;
  }
  .cell.tp { background: rgba(26, 127, 55, 0.10); }
  .cell.tn { background: rgba(9, 105, 218, 0.10); }
  .cell.fp { background: rgba(207, 34, 46, 0.10); }
  .cell.fn { background: rgba(154, 103, 0, 0.10); }
  ul.clean { margin: 8px 0 0 18px; padding: 0; }
  @media (max-width: 980px) { .two { grid-template-columns: 1fr; } }
  @media print {
    nav { display: none; }
    body { background: white; color: black; }
    .card, .metric, table, .chart-card { box-shadow: none; }
  }
"""

SUMMARY_KEYS = [
    "attack_id",
    "known_qids",
    "qid_filter_qids",
    "refine_qids",
    "unknown_visibility_qids",
    "target_id_col",
    "member_col",
    "n_targets",
    "n_members",
    "n_non_members",
    "seed",
    "max_compatible_fraction",
    "use_privjedai_fuzzy",
    "candidate_prefilter_mode",
    "schema_matching_enabled",
    "schema_matcher_name",
    "schema_matcher_min_score",
    "schema_matching_results_json",
    "schema_matching_pairs_csv",
    "tp",
    "tn",
    "fp",
    "fn",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "member_recall",
    "member_false_negative_rate",
    "non_member_true_negative_rate",
    "non_member_false_positive_rate",
    "member_avg_stage1_equivalence_class_size",
    "non_member_avg_stage1_equivalence_class_size",
    "member_avg_compatible_candidate_count",
    "non_member_avg_compatible_candidate_count",
    "member_avg_equivalence_class_reduction",
    "non_member_avg_equivalence_class_reduction",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate an HTML report for a MIA.")
    p.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root directory")
    p.add_argument("--attack-dir", type=Path, default=None, help="MIA attack directory")
    p.add_argument("--summary-json", type=Path, default=None, help="Direct path to summary.json")
    p.add_argument("--targets-csv", type=Path, default=None, help="Direct path to targets.csv")
    p.add_argument("--metrics-json", type=Path, default=None, help="Direct path to the anonymization metrics JSON")
    p.add_argument("--config-json", type=Path, default=None, help="Direct path to the anonymization config JSON")
    p.add_argument("--schema-results-json", type=Path, default=None, help="Direct path to schema_matching_results.json")
    p.add_argument("--schema-pairs-csv", type=Path, default=None, help="Direct path to schema_matching_pairs.csv")
    p.add_argument("--classification-json", type=Path, default=None, help="Direct path to <experiment_id>_classification.json (auto-resolved by default)")
    p.add_argument("--output", type=Path, default=None, help="Output HTML file")
    p.add_argument("--title", type=str, default=None, help="Custom title")
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def fmt_int(value: Any) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{int(round(float(value))):,}"
    except Exception:
        return html.escape(str(value))


def fmt_float(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return html.escape(str(value))


def fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except Exception:
        return html.escape(str(value))


def fmt_list(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, list):
        return ", ".join(str(x) for x in value) if value else "—"
    return str(value)


def escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def find_latest_mia_dir(project_root: Path) -> Path:
    base = project_root / "outputs" / "attacks" / "mia"
    candidates = [p.parent for p in base.rglob("summary.json")]
    if not candidates:
        raise FileNotFoundError(f"No summary.json found under {base}")
    return max(candidates, key=lambda p: (p / "summary.json").stat().st_mtime)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None, Path, Path | None, Path | None, Path | None, Path | None]:
    project_root = args.project_root.resolve()

    if args.summary_json:
        summary_json = args.summary_json.resolve()
        attack_dir = summary_json.parent
    else:
        attack_dir = args.attack_dir.resolve() if args.attack_dir else find_latest_mia_dir(project_root)
        summary_json = attack_dir / "summary.json"

    targets_csv = args.targets_csv.resolve() if args.targets_csv else attack_dir / "targets.csv"
    summary = read_json(summary_json)

    if args.metrics_json:
        metrics_json = args.metrics_json.resolve()
    else:
        attack_id = str(summary.get("attack_id", attack_dir.name))
        experiment_id = attack_id.split("__mia_")[0]
        candidate = project_root / "outputs" / "metrics" / f"{experiment_id}.json"
        metrics_json = candidate if candidate.exists() else None

    if args.config_json:
        config_json = args.config_json.resolve()
    else:
        config_json = None
        config_path_in_summary = summary.get("config_path")
        if config_path_in_summary:
            candidate = Path(str(config_path_in_summary))
            if candidate.exists():
                config_json = candidate
            else:
                basename_candidate = project_root / "outputs" / "configs" / candidate.name
                if basename_candidate.exists():
                    config_json = basename_candidate
        if config_json is None:
            attack_id = str(summary.get("attack_id", attack_dir.name))
            experiment_id = attack_id.split("__mia_")[0]
            fallback = project_root / "outputs" / "configs" / f"{experiment_id}.json"
            if fallback.exists():
                config_json = fallback

    # Schema-matching artefacts (optional). Resolution order:
    #   1) explicit CLI flag
    #   2) path stored in summary.json
    #   3) conventional file inside attack_dir
    schema_results_json: Path | None = None
    if args.schema_results_json:
        candidate = args.schema_results_json.resolve()
        if candidate.exists():
            schema_results_json = candidate
    else:
        summary_path_candidate = summary.get("schema_matching_results_json")
        if summary_path_candidate:
            candidate = Path(str(summary_path_candidate))
            if candidate.exists():
                schema_results_json = candidate
        if schema_results_json is None:
            fallback = attack_dir / "schema_matching_results.json"
            if fallback.exists():
                schema_results_json = fallback

    schema_pairs_csv: Path | None = None
    if args.schema_pairs_csv:
        candidate = args.schema_pairs_csv.resolve()
        if candidate.exists():
            schema_pairs_csv = candidate
    else:
        summary_path_candidate = summary.get("schema_matching_pairs_csv")
        if summary_path_candidate:
            candidate = Path(str(summary_path_candidate))
            if candidate.exists():
                schema_pairs_csv = candidate
        if schema_pairs_csv is None:
            fallback = attack_dir / "schema_matching_pairs.csv"
            if fallback.exists():
                schema_pairs_csv = fallback

    # Classification-utility artefact (optional). Resolved against the
    # conventional outputs/classification/<experiment_id>_classification.json
    # when no explicit path is given. ``summary`` is loaded above.
    classification_json = resolve_classification_json(
        cli_path=args.classification_json,
        project_root=project_root,
        summary=summary,
        attack_dir=attack_dir,
    )

    return project_root, summary_json, metrics_json, targets_csv, config_json, schema_results_json, schema_pairs_csv, classification_json


def rel_or_abs(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except Exception:
        return str(path)


def make_table(headers: list[str], rows: list[list[str]]) -> str:
    thead = "<thead><tr>" + "".join(f"<th>{escape(h)}</th>" for h in headers) + "</tr></thead>"
    body_rows = []
    for row in rows:
        body_rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"
    return f"<table>{thead}{tbody}</table>"


def grouped_histogram(
    in_values: list[float],
    out_values: list[float],
    n_bins: int,
    value_min: float | None = None,
    value_max: float | None = None,
) -> tuple[list[str], list[int], list[int]]:
    values = [*in_values, *out_values]
    if not values:
        return [], [], []
    lo = min(values) if value_min is None else value_min
    hi = max(values) if value_max is None else value_max
    if math.isclose(lo, hi):
        return [f"{100.0 * lo:.1f}%"], [len(in_values)], [len(out_values)]

    width = (hi - lo) / n_bins
    labels: list[str] = []
    in_counts = [0 for _ in range(n_bins)]
    out_counts = [0 for _ in range(n_bins)]
    for i in range(n_bins):
        a = lo + i * width
        b = lo + (i + 1) * width
        labels.append(f"{100.0 * a:.1f}–{100.0 * b:.1f}")

    for v in in_values:
        idx = int((v - lo) / width)
        if idx == n_bins:
            idx -= 1
        in_counts[idx] += 1
    for v in out_values:
        idx = int((v - lo) / width)
        if idx == n_bins:
            idx -= 1
        out_counts[idx] += 1

    return labels, in_counts, out_counts


def compute_ks_pvalue(a: list[float], b: list[float]) -> float | None:
    if not a or not b or ks_2samp is None:
        return None
    try:
        return float(ks_2samp(a, b).pvalue)
    except Exception:
        return None


def compute_best_fraction_threshold(rows: list[dict[str, str]]) -> tuple[float | None, float | None]:
    candidates = []
    for row in rows:
        frac = to_float(row.get("compatible_candidate_fraction"))
        count = to_int(row.get("compatible_candidate_count")) or 0
        truth = to_int(row.get("ground_truth_member"))
        if frac is None or truth is None:
            continue
        candidates.append((frac, count, truth))
    if not candidates:
        return None, None

    thresholds = sorted({frac for frac, _, _ in candidates})
    thresholds = [min(thresholds) - 1e-12, *thresholds, max(thresholds) + 1e-12]

    best_acc = -1.0
    best_thr: float | None = None
    for thr in thresholds:
        correct = 0
        for frac, count, truth in candidates:
            pred = 1 if (count > 0 and frac <= thr) else 0
            if pred == truth:
                correct += 1
        acc = correct / len(candidates)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return best_thr, best_acc


def build_report(
    project_root: Path,
    summary_json: Path,
    metrics_json: Path | None,
    config_json: Path | None,
    targets_csv: Path,
    output_path: Path,
    title: str | None,
    schema_results_json: Path | None = None,
    schema_pairs_csv: Path | None = None,
    classification_json: Path | None = None,
) -> Path:
    summary = read_json(summary_json)
    metrics = read_json(metrics_json) if metrics_json and metrics_json.exists() else {}
    config = read_json(config_json) if config_json and config_json.exists() else {}
    rows = read_csv_rows(targets_csv) if targets_csv.exists() else []
    schema_results = read_json(schema_results_json) if schema_results_json and schema_results_json.exists() else {}
    schema_pairs_rows: list[dict[str, str]] = (
        read_csv_rows(schema_pairs_csv) if schema_pairs_csv and schema_pairs_csv.exists() else []
    )

    # When ``build_report`` is called programmatically (e.g. from
    # run_mia_attack.py) without passing classification_json, we still try to
    # auto-resolve it via the conventional outputs/classification path.
    if classification_json is None:
        classification_json = resolve_classification_json(
            cli_path=None,
            project_root=project_root,
            summary=summary,
            attack_dir=summary_json.parent,
        )

    classification_section, classification_chart_js = build_classification_section(
        classification_json,
        section_number=9,
        anchor_id="classification",
        project_root=project_root,
    )

    # When the classification section is rendered, we must also pull in the
    # extra CSS / head scripts / JS helpers it depends on (chartjs-plugin-zoom
    # for ROC chart panning, ``_charts`` registry, ``resetZoom``). When the
    # section is absent, these are empty strings so the resulting HTML stays
    # byte-identical to the previous behavior.
    if classification_section:
        from classification_report_section import (
            CLF_EXTRA_CSS,
            CLF_EXTRA_HEAD_SCRIPTS,
            CLF_EXTRA_JS_HELPERS,
        )
        classification_extra_css = CLF_EXTRA_CSS
        classification_extra_head = CLF_EXTRA_HEAD_SCRIPTS
        classification_extra_js_helpers = CLF_EXTRA_JS_HELPERS
    else:
        classification_extra_css = ""
        classification_extra_head = ""
        classification_extra_js_helpers = ""

    attack_id = str(summary.get("attack_id", summary_json.parent.name))
    report_title = title or f"MIA Report – {attack_id}"

    tp = to_int(summary.get("tp")) or 0
    tn = to_int(summary.get("tn")) or 0
    fp = to_int(summary.get("fp")) or 0
    fn = to_int(summary.get("fn")) or 0

    accuracy = to_float(summary.get("accuracy"))
    precision = to_float(summary.get("precision"))
    recall = to_float(summary.get("recall"))
    f1 = to_float(summary.get("f1"))
    tnr = to_float(summary.get("non_member_true_negative_rate"))
    max_compatible_fraction = to_float(summary.get("max_compatible_fraction"))

    n_targets = to_int(summary.get("n_targets")) or len(rows)
    n_members = to_int(summary.get("n_members")) or sum(1 for r in rows if to_int(r.get("ground_truth_member")) == 1)
    n_non_members = to_int(summary.get("n_non_members")) or max(n_targets - n_members, 0)

    member_frac = [
        frac for row in rows
        if to_int(row.get("ground_truth_member")) == 1
        for frac in [to_float(row.get("compatible_candidate_fraction"))]
        if frac is not None
    ]
    non_member_frac = [
        frac for row in rows
        if to_int(row.get("ground_truth_member")) == 0
        for frac in [to_float(row.get("compatible_candidate_fraction"))]
        if frac is not None
    ]
    member_count = [
        cnt for row in rows
        if to_int(row.get("ground_truth_member")) == 1
        for cnt in [to_float(row.get("compatible_candidate_count"))]
        if cnt is not None
    ]
    non_member_count = [
        cnt for row in rows
        if to_int(row.get("ground_truth_member")) == 0
        for cnt in [to_float(row.get("compatible_candidate_count"))]
        if cnt is not None
    ]

    # Stage 1 (generalized QI filtering) per-target class sizes.
    stage1_member = [
        size for row in rows
        if to_int(row.get("ground_truth_member")) == 1
        for size in [to_float(row.get("stage1_equivalence_class_size"))]
        if size is not None
    ]
    stage1_non_member = [
        size for row in rows
        if to_int(row.get("ground_truth_member")) == 0
        for size in [to_float(row.get("stage1_equivalence_class_size"))]
        if size is not None
    ]
    # Stage 2 (clear-text refinement) per-target class sizes.
    stage2_member = member_count
    stage2_non_member = non_member_count
    # Reduction between stage 1 and stage 2.
    reduction_member = [
        red for row in rows
        if to_int(row.get("ground_truth_member")) == 1
        for red in [to_float(row.get("equivalence_class_reduction"))]
        if red is not None
    ]
    reduction_non_member = [
        red for row in rows
        if to_int(row.get("ground_truth_member")) == 0
        for red in [to_float(row.get("equivalence_class_reduction"))]
        if red is not None
    ]
    reduction_rate_member = [
        rate for row in rows
        if to_int(row.get("ground_truth_member")) == 1
        for rate in [to_float(row.get("equivalence_class_reduction_rate"))]
        if rate is not None
    ]
    reduction_rate_non_member = [
        rate for row in rows
        if to_int(row.get("ground_truth_member")) == 0
        for rate in [to_float(row.get("equivalence_class_reduction_rate"))]
        if rate is not None
    ]
    # Per-target counters for completeness (stage 1 containment flags).
    true_record_in_stage1_flags = [
        to_int(row.get("true_record_in_stage1_class"))
        for row in rows
        if to_int(row.get("ground_truth_member")) == 1
    ]
    true_record_in_stage1_member_count = sum(1 for f in true_record_in_stage1_flags if f == 1)

    ks_pvalue = compute_ks_pvalue(member_frac, non_member_frac)
    best_thr, best_acc = compute_best_fraction_threshold(rows)

    hist_labels, hist_in, hist_out = grouped_histogram(member_frac, non_member_frac, 18, 0.0, max([*member_frac, *non_member_frac], default=0.0))
    pred_counts = Counter(to_int(row.get("predicted_member")) for row in rows)
    predicted_in_count = pred_counts.get(1, 0)
    predicted_out_count = pred_counts.get(0, 0)

    synthesis_parts = []
    if accuracy is not None:
        synthesis_parts.append(f"The overall MIA accuracy is <strong>{fmt_pct(accuracy)}</strong>.")
    if precision is not None and recall is not None:
        synthesis_parts.append(
            f"Member detection achieves a precision of <strong>{fmt_pct(precision)}</strong> and a recall of <strong>{fmt_pct(recall)}</strong>."
        )
    if ks_pvalue is not None:
        synthesis_parts.append(
            f"The KS test on the compatible candidate fraction yields <strong>p = {fmt_float(ks_pvalue, 3)}</strong>."
        )
    if best_acc is not None and best_thr is not None:
        synthesis_parts.append(
            f"Using only the <code>compatible_candidate_fraction</code> variable, the best observed accuracy is <strong>{fmt_pct(best_acc)}</strong> with a threshold around <strong>{fmt_pct(best_thr)}</strong>."
        )
    synthesis_html = " ".join(synthesis_parts) if synthesis_parts else "No synthesis available."

    metrics_cards = [
        ("MIA targets", fmt_int(n_targets), ""),
        ("IN", fmt_int(n_members), ""),
        ("OUT", fmt_int(n_non_members), ""),
        ("Accuracy", fmt_pct(accuracy), "warn"),
        ("Precision", fmt_pct(precision), ""),
        ("Member recall", fmt_pct(recall), ""),
        ("Non-member TNR", fmt_pct(tnr), ""),
        ("F1-score", fmt_pct(f1), "accent"),
    ]
    metrics_html = "".join(
        f"<div class='metric'><div class='k'>{escape(k)}</div><div class='v {cls}'>{v}</div></div>"
        for k, v, cls in metrics_cards
    )

    config_rows = [
        ["Attack ID", f"<code>{escape(attack_id)}</code>"],
        ["Attacker's known QIDs", escape(fmt_list(summary.get("known_qids")))],
        ["Stage 1 QIDs", escape(fmt_list(summary.get("qid_filter_qids")))],
        ["Stage 2 QIDs", escape(fmt_list(summary.get("refine_qids")))],
        ["Targets", fmt_int(n_targets)],
        ["Members / non-members", f"{fmt_int(n_members)} / {fmt_int(n_non_members)}"],
        ["Max compatible fraction threshold", fmt_pct(max_compatible_fraction)],
        ["PrivJedAI fuzzy", escape(str(summary.get("use_privjedai_fuzzy", False)).lower())],
        ["Seed", fmt_int(summary.get("seed"))],
    ]

    if metrics or config:
        transformations = (metrics.get("transformations") if metrics else None) or {}
        config_rows.extend(
            [
                ["Anonymization QIs", escape(fmt_list((config.get("quasi_identifiers") if config else None) or (metrics.get("quasi_identifiers") if metrics else None))) if ((config.get("quasi_identifiers") if config else None) or (metrics.get("quasi_identifiers") if metrics else None)) else "—"],
                ["k / l / t", f"<code>k={escape((config.get('k') if config else None) if (config.get('k') if config else None) is not None else (metrics.get('k') if metrics else None))}</code> · <code>l={escape((config.get('l') if config else None) if (config.get('l') if config else None) is not None else (metrics.get('l') if metrics else None))}</code> · <code>t={escape((config.get('t') if config else None) if (config.get('t') if config else None) is not None else (metrics.get('t') if metrics else None))}</code>"],
                ["Suppression limit", f"<code>{escape((config.get('suppression_limit') if config else None) if (config.get('suppression_limit') if config else None) is not None else (metrics.get('suppression_limit') if metrics else None))}</code>" if (((config.get("suppression_limit") if config else None) is not None) or ((metrics.get("suppression_limit") if metrics else None) is not None)) else "—"],
                ["Transformations", "<code>" + escape(json.dumps(transformations, ensure_ascii=False)) + "</code>" if transformations else "—"],
            ]
        )

    files_rows = [
        ["summary.json", f"<code>{escape(rel_or_abs(summary_json, project_root))}</code>"],
        ["targets.csv", f"<code>{escape(rel_or_abs(targets_csv, project_root))}</code>"],
    ]
    if metrics_json and metrics_json.exists():
        files_rows.append(["metrics.json", f"<code>{escape(rel_or_abs(metrics_json, project_root))}</code>"])
    if config_json and config_json.exists():
        files_rows.append(["config.json", f"<code>{escape(rel_or_abs(config_json, project_root))}</code>"])
    if schema_results_json and schema_results_json.exists():
        files_rows.append(["schema_matching_results.json", f"<code>{escape(rel_or_abs(schema_results_json, project_root))}</code>"])
    if schema_pairs_csv and schema_pairs_csv.exists():
        files_rows.append(["schema_matching_pairs.csv", f"<code>{escape(rel_or_abs(schema_pairs_csv, project_root))}</code>"])

    summary_table_rows = []
    for key in SUMMARY_KEYS:
        if key in summary:
            value = summary[key]
            if isinstance(value, list):
                value_str = fmt_list(value)
            elif isinstance(value, bool):
                value_str = "true" if value else "false"
            elif isinstance(value, float) and (
                key.endswith("_rate")
                or key in {"accuracy", "precision", "recall", "f1", "member_recall", "non_member_true_negative_rate", "non_member_false_positive_rate", "member_false_negative_rate", "max_compatible_fraction"}
            ):
                value_str = fmt_pct(value)
            elif isinstance(value, (int, float)):
                value_str = fmt_float(value) if isinstance(value, float) and not float(value).is_integer() else fmt_int(value)
            else:
                value_str = escape(value)
            summary_table_rows.append([f"<code>{escape(key)}</code>", value_str])

    op_counter = summary.get("operation_counter") or {}
    op_rows: list[list[str]] = []
    if isinstance(op_counter, dict):
        for k, v in op_counter.items():
            if isinstance(v, float) and not float(v).is_integer():
                val = fmt_float(v, 4)
            elif isinstance(v, (int, float)):
                val = fmt_int(v)
            else:
                val = escape(v)
            op_rows.append([f"<code>{escape(k)}</code>", val])

    op_est = to_float(op_counter.get("estimated_total_operations")) if isinstance(op_counter, dict) else None
    op_per_target = (op_est / n_targets) if op_est and n_targets else None

    anonymization_block = ""
    if metrics:
        anonymization_block = f"""
<section id="anonymization" class="card">
  <h2 class="section-title">3. Anonymization results</h2>
  <div class="grid">
    <div class="metric"><div class="k">Equivalence classes</div><div class="v">{fmt_int(metrics.get('number_of_equivalence_classes'))}</div></div>
    <div class="metric"><div class="k">Average size</div><div class="v">{fmt_float(metrics.get('average_equivalence_class_size'))}</div></div>
    <div class="metric"><div class="k">Min size</div><div class="v">{fmt_int(metrics.get('min_equivalence_class_size'))}</div></div>
    <div class="metric"><div class="k">Max size</div><div class="v">{fmt_int(metrics.get('max_equivalence_class_size'))}</div></div>
    <div class="metric"><div class="k">Suppressed records</div><div class="v warn">{fmt_int(metrics.get('number_of_suppressed_records'))}</div></div>
    <div class="metric"><div class="k">After full suppression drop</div><div class="v">{fmt_int(metrics.get('n_rows_after_full_suppression_drop'))}</div></div>
  </div>
</section>
"""

    distribution_table = make_table(
        ["Group", "Avg. compatible count", "Median compatible count", "Avg. compatible fraction", "Median compatible fraction"],
        [
            [
                "IN",
                fmt_float(statistics.mean(member_count) if member_count else None),
                fmt_float(statistics.median(member_count) if member_count else None),
                fmt_pct(statistics.mean(member_frac) if member_frac else None),
                fmt_pct(statistics.median(member_frac) if member_frac else None),
            ],
            [
                "OUT",
                fmt_float(statistics.mean(non_member_count) if non_member_count else None),
                fmt_float(statistics.median(non_member_count) if non_member_count else None),
                fmt_pct(statistics.mean(non_member_frac) if non_member_frac else None),
                fmt_pct(statistics.median(non_member_frac) if non_member_frac else None),
            ],
        ],
    )

    op_section = ""
    if op_rows:
        op_section = f"""
<section id="complexity" class="card">
  <h2 class="section-title">8. Complexity counters</h2>
  <div class="grid">
    <div class="metric"><div class="k">Estimated operations</div><div class="v">{fmt_int(op_est)}</div></div>
    <div class="metric"><div class="k">Operations per target</div><div class="v">{fmt_float(op_per_target)}</div></div>
    <div class="metric"><div class="k">Membership decisions</div><div class="v">{fmt_int(op_counter.get('membership_decisions') if isinstance(op_counter, dict) else None)}</div></div>
  </div>
  <div style="margin-top:16px">{make_table(['Counter', 'Value'], op_rows)}</div>
</section>
"""

    # ------------------------------------------------------------------
    # Stage 1 (generalized QIDs) vs Stage 2 (clear-text refinement) section
    # ------------------------------------------------------------------
    stage1_attrs = summary.get("qid_filter_qids") or []
    stage2_attrs = summary.get("refine_qids") or []
    candidate_prefilter_mode = summary.get("candidate_prefilter_mode") or ""

    member_avg_stage1 = to_float(summary.get("member_avg_stage1_equivalence_class_size"))
    non_member_avg_stage1 = to_float(summary.get("non_member_avg_stage1_equivalence_class_size"))
    member_avg_stage2 = to_float(summary.get("member_avg_compatible_candidate_count"))
    non_member_avg_stage2 = to_float(summary.get("non_member_avg_compatible_candidate_count"))
    member_avg_reduction = to_float(summary.get("member_avg_equivalence_class_reduction"))
    non_member_avg_reduction = to_float(summary.get("non_member_avg_equivalence_class_reduction"))

    stage_breakdown_table = make_table(
        [
            "Group",
            "Stage 1 avg size",
            "Stage 1 median size",
            "Stage 2 avg size",
            "Stage 2 median size",
            "Avg. absolute reduction",
            "Avg. reduction rate",
        ],
        [
            [
                "IN (members)",
                fmt_float(statistics.mean(stage1_member) if stage1_member else member_avg_stage1),
                fmt_float(statistics.median(stage1_member) if stage1_member else None),
                fmt_float(statistics.mean(stage2_member) if stage2_member else member_avg_stage2),
                fmt_float(statistics.median(stage2_member) if stage2_member else None),
                fmt_float(statistics.mean(reduction_member) if reduction_member else member_avg_reduction),
                fmt_pct(statistics.mean(reduction_rate_member) if reduction_rate_member else None),
            ],
            [
                "OUT (non-members)",
                fmt_float(statistics.mean(stage1_non_member) if stage1_non_member else non_member_avg_stage1),
                fmt_float(statistics.median(stage1_non_member) if stage1_non_member else None),
                fmt_float(statistics.mean(stage2_non_member) if stage2_non_member else non_member_avg_stage2),
                fmt_float(statistics.median(stage2_non_member) if stage2_non_member else None),
                fmt_float(statistics.mean(reduction_non_member) if reduction_non_member else non_member_avg_reduction),
                fmt_pct(statistics.mean(reduction_rate_non_member) if reduction_rate_non_member else None),
            ],
        ],
    )

    stage_attrs_table = make_table(
        ["Stage", "Attributes used", "#"],
        [
            [
                "Stage 1 (generalized QIDs)",
                escape(fmt_list(stage1_attrs)),
                fmt_int(len(stage1_attrs)),
            ],
            [
                "Stage 2 (clear-text refinement)",
                escape(fmt_list(stage2_attrs)),
                fmt_int(len(stage2_attrs)),
            ],
        ],
    )

    stage_section = f"""
<section id="stages" class="card">
  <h2 class="section-title">5. Stage 1 / Stage 2 equivalence class analysis</h2>
  <p class="section-subtitle">Two-phase candidate reduction performed by the attack: stage 1 filters the anonymized table on generalized QIDs, then stage 2 refines the resulting equivalence class on clear-text attributes (with optional privJedAI fuzzy matching).</p>
  <div class="grid">
    <div class="metric"><div class="k">Stage 1 avg size (IN)</div><div class="v">{fmt_float(statistics.mean(stage1_member) if stage1_member else member_avg_stage1)}</div></div>
    <div class="metric"><div class="k">Stage 1 avg size (OUT)</div><div class="v">{fmt_float(statistics.mean(stage1_non_member) if stage1_non_member else non_member_avg_stage1)}</div></div>
    <div class="metric"><div class="k">Stage 2 avg size (IN)</div><div class="v">{fmt_float(statistics.mean(stage2_member) if stage2_member else member_avg_stage2)}</div></div>
    <div class="metric"><div class="k">Stage 2 avg size (OUT)</div><div class="v">{fmt_float(statistics.mean(stage2_non_member) if stage2_non_member else non_member_avg_stage2)}</div></div>
    <div class="metric"><div class="k">True record in stage 1 class (IN)</div><div class="v">{fmt_int(true_record_in_stage1_member_count)} / {fmt_int(len(true_record_in_stage1_flags))}</div></div>
    <div class="metric"><div class="k">Prefilter mode</div><div class="v" style="font-size:0.95rem">{escape(candidate_prefilter_mode) or '—'}</div></div>
  </div>
  <div class="two" style="margin-top:16px;">
    <div>
      <h3 style="margin-top:0">Attributes per stage</h3>
      {stage_attrs_table}
    </div>
    <div>
      <h3 style="margin-top:0">Per-group size and reduction</h3>
      {stage_breakdown_table}
      <div class="callout" style="margin-top:14px;">
        A larger stage 1 class means the attacker starts with more noise; a smaller stage 2 class after refinement means the clear-text attributes were highly discriminative. Systematically larger stage 2 classes for the OUT group (than for the IN group) indicate that non-members are harder to match, which directly drives the MIA signal.
      </div>
    </div>
  </div>
</section>
"""

    # ------------------------------------------------------------------
    # Schema matching (column matching) section
    # ------------------------------------------------------------------
    # schema_chart_js holds JS to be injected into the bottom <script>
    # block (next to chart_js for the histogram).
    schema_chart_js = ""
    schema_section = ""
    if schema_results or schema_pairs_rows:
        sm_matcher = schema_results.get("matcher") or summary.get("schema_matcher_name") or "—"
        sm_min_score = schema_results.get("min_score")
        if sm_min_score is None:
            sm_min_score = summary.get("schema_matcher_min_score")
        sm_obfuscated_columns = schema_results.get("obfuscated_columns") or []
        sm_anon_unknown = schema_results.get("anon_unknown_cols") or []
        sm_kb_candidates = schema_results.get("kb_candidate_cols") or []
        sm_metrics = schema_results.get("metrics") or {}

        sm_pairs = schema_results.get("pairs") or []
        # Use CSV pairs if the JSON block does not carry them.
        if not sm_pairs and schema_pairs_rows:
            sm_pairs = schema_pairs_rows

        def _sm_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes"}

        def _sm_score(value: Any) -> float | None:
            if value is None or value == "":
                return None
            try:
                return float(value)
            except Exception:
                return None

        n_obfuscated = to_int(sm_metrics.get("n_obfuscated")) if sm_metrics else None
        n_mapped = to_int(sm_metrics.get("n_mapped")) if sm_metrics else None
        n_correct = to_int(sm_metrics.get("n_correct")) if sm_metrics else None
        coverage = to_float(sm_metrics.get("coverage")) if sm_metrics else None
        accuracy_on_mapped = to_float(sm_metrics.get("accuracy_on_mapped")) if sm_metrics else None
        sm_recall = to_float(sm_metrics.get("recall")) if sm_metrics else None
        # Fallbacks derived from pairs if metrics are missing.
        if sm_pairs and (n_obfuscated is None or n_mapped is None or n_correct is None):
            n_obfuscated = n_obfuscated if n_obfuscated is not None else len(sm_pairs)
            n_mapped = n_mapped if n_mapped is not None else sum(1 for p in sm_pairs if _sm_bool(p.get("is_mapped")))
            n_correct = n_correct if n_correct is not None else sum(1 for p in sm_pairs if _sm_bool(p.get("is_correct")))
            if coverage is None and n_obfuscated:
                coverage = n_mapped / n_obfuscated
            if accuracy_on_mapped is None and n_mapped:
                accuracy_on_mapped = n_correct / n_mapped
            if sm_recall is None and n_obfuscated:
                sm_recall = n_correct / n_obfuscated

        sm_pair_rows_html = []
        for pair in sm_pairs:
            anon_col = pair.get("anon_column") or pair.get("anon_col") or "—"
            true_col = pair.get("true_column") or pair.get("true_col") or "—"
            predicted_col = pair.get("predicted_column")
            if predicted_col in (None, "", "None"):
                predicted_label = "<span class=\"warn\">unmapped</span>"
            else:
                predicted_label = f"<code>{escape(predicted_col)}</code>"
            score = _sm_score(pair.get("score"))
            is_correct = _sm_bool(pair.get("is_correct"))
            is_mapped = _sm_bool(pair.get("is_mapped")) or predicted_col not in (None, "", "None")
            if not is_mapped:
                status_label = "<span class=\"warn\">unmapped</span>"
            elif is_correct:
                status_label = "<span class=\"good\">correct</span>"
            else:
                status_label = "<span class=\"danger\">incorrect</span>"
            sm_pair_rows_html.append(
                [
                    f"<code>{escape(anon_col)}</code>",
                    f"<code>{escape(true_col)}</code>",
                    predicted_label,
                    fmt_float(score, 4) if score is not None else "—",
                    status_label,
                ]
            )

        sm_pairs_table = (
            make_table(
                ["Obfuscated (anon) column", "True column", "Predicted column", "Score", "Status"],
                sm_pair_rows_html,
            )
            if sm_pair_rows_html
            else "<p class=\"small\">No per-column pairs available.</p>"
        )

        sm_config_table = make_table(
            ["Parameter", "Value"],
            [
                ["Matcher", f"<code>{escape(sm_matcher)}</code>"],
                ["Min score", fmt_float(sm_min_score, 4) if sm_min_score is not None else "—"],
                ["Obfuscated columns", escape(fmt_list(sm_obfuscated_columns))],
                ["Anonymized unknown columns", escape(fmt_list(sm_anon_unknown))],
                ["KB candidate columns", escape(fmt_list(sm_kb_candidates))],
            ],
        )

        # ----- New: derived stats, verdict, downstream impact, charts -----
        n_obf_safe = n_obfuscated or 0
        n_map_safe = n_mapped or 0
        n_cor_safe = n_correct or 0
        n_incorrect = max(0, n_map_safe - n_cor_safe)
        n_unmapped = max(0, n_obf_safe - n_map_safe)

        # Verdict (auto-interpretation of the matching outcome).
        if n_obf_safe == 0:
            verdict_text = "No QI columns were obfuscated; schema matching was a no-op."
            verdict_class = "warn"
        elif n_cor_safe == n_obf_safe:
            verdict_text = (
                f"All {n_obf_safe} obfuscated QI(s) were recovered correctly. "
                f"Schema matching does not weaken the MIA: known_qids is unchanged."
            )
            verdict_class = "good"
        elif n_cor_safe == 0:
            verdict_text = (
                f"None of the {n_obf_safe} obfuscated QI(s) were recovered correctly. "
                f"The attacker loses access to all of them; the MIA proceeds with a "
                f"strictly smaller known_qids set (or none at all)."
            )
            verdict_class = "danger"
        else:
            verdict_text = (
                f"{n_cor_safe} of {n_obf_safe} obfuscated QI(s) recovered correctly. "
                f"{n_incorrect} mapped to a wrong KB column (values now silently misaligned in "
                f"the candidate filter), and {n_unmapped} stayed unmapped (dropped from "
                f"known_qids — the MIA can no longer use them)."
            )
            verdict_class = "warn"

        # Downstream impact: which true (KB) columns are usable, misrouted, or lost.
        # In run_mia_attack, a true_col survives in known_qids iff some anon column was
        # predicted to map to it. Misrouted columns (mapped to the wrong KB name) survive
        # but with corrupted values flowing into the equivalence-class filter.
        correct_cols: list[str] = []
        misrouted_cols: list[str] = []
        unmapped_cols: list[str] = []
        for pair in sm_pairs:
            true_col = pair.get("true_column") or pair.get("true_col") or "—"
            mapped = _sm_bool(pair.get("is_mapped")) or pair.get("predicted_column") not in (None, "", "None")
            correct = _sm_bool(pair.get("is_correct"))
            if not mapped:
                unmapped_cols.append(true_col)
            elif correct:
                correct_cols.append(true_col)
            else:
                misrouted_cols.append(true_col)

        def _cols_html(items: list[str], css_class: str) -> str:
            if not items:
                return "<span class=\"small\">none</span>"
            return ", ".join(
                f"<code class=\"{css_class}\">{escape(c)}</code>"
                for c in sorted(set(items))
            )

        # Score statistics by status.
        def _score_stats(scores: list[float]) -> dict[str, float] | None:
            scores = [s for s in scores if s is not None]
            if not scores:
                return None
            scores_sorted = sorted(scores)
            n = len(scores_sorted)
            mid = n // 2
            median = scores_sorted[mid] if n % 2 == 1 else (scores_sorted[mid - 1] + scores_sorted[mid]) / 2.0
            return {
                "n": n,
                "min": scores_sorted[0],
                "max": scores_sorted[-1],
                "mean": sum(scores_sorted) / n,
                "median": median,
            }

        correct_scores = [_sm_score(p.get("score")) for p in sm_pairs if _sm_bool(p.get("is_correct"))]
        incorrect_scores = [
            _sm_score(p.get("score")) for p in sm_pairs
            if (_sm_bool(p.get("is_mapped")) or p.get("predicted_column") not in (None, "", "None"))
            and not _sm_bool(p.get("is_correct"))
        ]
        unmapped_scores = [
            _sm_score(p.get("score")) for p in sm_pairs
            if not (_sm_bool(p.get("is_mapped")) or p.get("predicted_column") not in (None, "", "None"))
        ]

        stats_correct = _score_stats(correct_scores)
        stats_incorrect = _score_stats(incorrect_scores)
        stats_unmapped = _score_stats(unmapped_scores)

        def _stats_row(label: str, css_class: str, stats: dict[str, float] | None) -> list[str]:
            if not stats:
                return [
                    f"<span class=\"{css_class}\">{escape(label)}</span>",
                    "0", "—", "—", "—", "—",
                ]
            return [
                f"<span class=\"{css_class}\">{escape(label)}</span>",
                str(stats["n"]),
                fmt_float(stats["min"], 4),
                fmt_float(stats["mean"], 4),
                fmt_float(stats["median"], 4),
                fmt_float(stats["max"], 4),
            ]

        score_stats_table = make_table(
            ["Status", "N", "Min", "Mean", "Median", "Max"],
            [
                _stats_row("Correct", "good", stats_correct),
                _stats_row("Incorrect (mapped to wrong column)", "danger", stats_incorrect),
                _stats_row("Unmapped (no prediction)", "warn", stats_unmapped),
            ],
        )

        # Per-pair score chart (color-coded) and outcome distribution doughnut.
        GOOD_COLOR, WARN_COLOR, DANGER_COLOR = "#1a7f37", "#9a6700", "#cf222e"
        chart_labels: list[str] = []
        chart_scores: list[float] = []
        chart_colors: list[str] = []
        for pair in sm_pairs:
            anon = pair.get("anon_column") or pair.get("anon_col") or "—"
            true_col = pair.get("true_column") or pair.get("true_col") or "—"
            chart_labels.append(f"{anon} → {true_col}")
            score = _sm_score(pair.get("score"))
            chart_scores.append(score if score is not None else 0.0)
            mapped = _sm_bool(pair.get("is_mapped")) or pair.get("predicted_column") not in (None, "", "None")
            if not mapped:
                chart_colors.append(WARN_COLOR)
            elif _sm_bool(pair.get("is_correct")):
                chart_colors.append(GOOD_COLOR)
            else:
                chart_colors.append(DANGER_COLOR)

        score_chart_block = ""
        score_chart_js_part = ""
        if chart_labels:
            score_chart_js_part = (
                "new Chart(document.getElementById('schemaScoreChart'), "
                "{type:'bar', data:{labels:%s, datasets:[{label:'Score', data:%s, "
                "backgroundColor:%s, borderColor:%s, borderWidth:1}]}, "
                "options:{responsive:true, maintainAspectRatio:false, "
                "plugins:{legend:{display:false}}, "
                "scales:{x:{ticks:{color:textColor, maxRotation:45, minRotation:45, autoSkip:false, font:{size:11}}, grid:{display:false}}, "
                "y:{title:{display:true, text:'Matching score', color:textColor, font:{size:12}}, "
                "suggestedMin:0, suggestedMax:1, ticks:{color:textColor, font:{size:11}}, grid:{color:gridColor}}}}});\n"
                % (
                    json.dumps(chart_labels, ensure_ascii=False),
                    json.dumps(chart_scores),
                    json.dumps(chart_colors),
                    json.dumps(chart_colors),
                )
            )
            score_chart_block = """
    <div class=\"chart-card\">
      <h3 style=\"margin-top:0\">Per-pair matching score</h3>
      <div class=\"chart-wrap\"><canvas id=\"schemaScoreChart\"></canvas></div>
      <p class=\"small\" style=\"margin-top:8px;\">One bar per obfuscated QI. Color: <span class=\"good\">correct</span>, <span class=\"danger\">mapped to wrong KB column</span>, <span class=\"warn\">unmapped</span>. Min-score threshold: """ + (fmt_float(sm_min_score, 4) if sm_min_score is not None else "—") + """.</p>
    </div>
"""

        status_chart_js_part = (
            "new Chart(document.getElementById('schemaStatusChart'), "
            "{type:'doughnut', data:{labels:['Correct', 'Incorrect (mapped)', 'Unmapped'], "
            "datasets:[{data:%s, backgroundColor:['%s','%s','%s']}]}, "
            "options:{responsive:true, maintainAspectRatio:false, "
            "plugins:{legend:{position:'bottom', labels:{color:textColor, font:{size:11}}}}}});\n"
            % (
                json.dumps([n_cor_safe, n_incorrect, n_unmapped]),
                GOOD_COLOR, DANGER_COLOR, WARN_COLOR,
            )
        )

        schema_chart_js = score_chart_js_part + status_chart_js_part

        schema_section = f"""
<section id=\"schema-matching\" class=\"card\">
  <h2 class=\"section-title\">6. Schema matching (column matching)</h2>
  <p class=\"section-subtitle\">When the attacker's known QI columns were obfuscated in the anonymized release, a schema matcher attempts to recover the mapping between the anonymized (renamed) columns and the attacker's knowledge base columns (i.e. the targets CSV) before the MIA runs. QIs whose name cannot be recovered are dropped from <code>known_qids</code>: the attacker has lost the ability to use them.</p>
  <div class=\"callout\" style=\"margin-bottom:14px;\"><strong class=\"{verdict_class}\">Verdict:</strong> {escape(verdict_text)}</div>
  <div class=\"grid\">
    <div class=\"metric\"><div class=\"k\">Obfuscated columns</div><div class=\"v\">{fmt_int(n_obfuscated)}</div></div>
    <div class=\"metric\"><div class=\"k\">Mapped</div><div class=\"v accent\">{fmt_int(n_mapped)}</div></div>
    <div class=\"metric\"><div class=\"k\">Correctly mapped</div><div class=\"v good\">{fmt_int(n_correct)}</div></div>
    <div class=\"metric\"><div class=\"k\">Incorrectly mapped</div><div class=\"v danger\">{fmt_int(n_incorrect)}</div></div>
    <div class=\"metric\"><div class=\"k\">Unmapped</div><div class=\"v warn\">{fmt_int(n_unmapped)}</div></div>
    <div class=\"metric\"><div class=\"k\">Coverage</div><div class=\"v\">{fmt_pct(coverage)}</div></div>
    <div class=\"metric\"><div class=\"k\">Accuracy on mapped</div><div class=\"v\">{fmt_pct(accuracy_on_mapped)}</div></div>
    <div class=\"metric\"><div class=\"k\">Recall</div><div class=\"v\">{fmt_pct(sm_recall)}</div></div>
  </div>
  <div class=\"two\" style=\"margin-top:16px;\">
{score_chart_block}
    <div class=\"chart-card\">
      <h3 style=\"margin-top:0\">Outcome distribution</h3>
      <div class=\"chart-wrap\"><canvas id=\"schemaStatusChart\"></canvas></div>
    </div>
  </div>
  <div class=\"two\" style=\"margin-top:16px;\">
    <div>
      <h3 style=\"margin-top:0\">Downstream impact on the MIA</h3>
      <div class=\"callout\">
        <p style=\"margin:0 0 8px 0;\"><strong class=\"good\">Recovered (still in known_qids):</strong> {_cols_html(correct_cols, 'good')}</p>
        <p style=\"margin:0 0 8px 0;\"><strong class=\"danger\">Misrouted (silently corrupt values):</strong> {_cols_html(misrouted_cols, 'danger')}</p>
        <p style=\"margin:0;\"><strong class=\"warn\">Lost (dropped from known_qids):</strong> {_cols_html(unmapped_cols, 'warn')}</p>
        <p class=\"small\" style=\"margin-top:10px;\">A misrouted column is renamed to a real KB column name, so the MIA still uses it as a QI — but the values now come from a different obfuscated column. This corrupts the equivalence-class filter without raising any error.</p>
      </div>
    </div>
    <div>
      <h3 style=\"margin-top:0\">Score statistics by status</h3>
      {score_stats_table}
    </div>
  </div>
  <div class=\"two\" style=\"margin-top:16px;\">
    <div>
      <h3 style=\"margin-top:0\">Matcher configuration</h3>
      {sm_config_table}
    </div>
    <div>
      <h3 style=\"margin-top:0\">Per-column matching outcome</h3>
      {sm_pairs_table}
    </div>
  </div>
</section>
"""

    hist_section = ""
    if hist_labels:
        hist_section = f"""
<section id="distribution" class="card">
  <h2 class="section-title">7. IN vs OUT analysis</h2>
  <div class="grid">
    <div class="metric"><div class="k">KS p-value</div><div class="v {'good' if ks_pvalue is not None else ''}">{fmt_float(ks_pvalue, 3)}</div></div>
    <div class="metric"><div class="k">Best accuracy (fraction only)</div><div class="v warn">{fmt_pct(best_acc)}</div></div>
    <div class="metric"><div class="k">Best threshold</div><div class="v">{fmt_pct(best_thr)}</div></div>
    <div class="metric"><div class="k">Threshold used in the attack</div><div class="v">{fmt_pct(max_compatible_fraction)}</div></div>
  </div>
  <div class="two" style="margin-top:16px;">
    <div class="chart-card">
      <h3 style="margin-top:0">Distribution of compatible candidate fraction</h3>
      <div class="chart-wrap"><canvas id="histChart"></canvas></div>
    </div>
    <div>
      {distribution_table}
      <div class="callout" style="margin-top:14px;">
        This section considers only the <code>compatible_candidate_fraction</code> variable. The more IN and OUT distributions overlap, the harder it is for the MIA to distinguish members from non-members.
      </div>
    </div>
  </div>
</section>
"""

    chart_js = ""
    if hist_labels:
        chart_js = f"""
new Chart(document.getElementById('histChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(hist_labels, ensure_ascii=False)},
    datasets: [
      {{
        label: 'IN',
        data: {json.dumps(hist_in)},
        backgroundColor: 'rgba(122,162,255,0.58)',
        borderColor: 'rgba(122,162,255,0.95)',
        borderWidth: 1
      }},
      {{
        label: 'OUT',
        data: {json.dumps(hist_out)},
        backgroundColor: 'rgba(255,123,123,0.46)',
        borderColor: 'rgba(255,123,123,0.95)',
        borderWidth: 1
      }}
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ color: textColor }} }} }},
    scales: {{
      x: {{
        title: {{ display: true, text: 'Compatible candidate fraction (%)', color: textColor, font: {{ size: 12 }} }},
        ticks: {{ color: textColor, maxRotation: 0, autoSkip: true, maxTicksLimit: 8, font: {{ size: 11 }} }},
        grid: {{ display: false }}
      }},
      y: {{
        title: {{ display: true, text: 'Number of targets', color: textColor, font: {{ size: 12 }} }},
        ticks: {{ color: textColor, font: {{ size: 11 }} }},
        grid: {{ color: gridColor }}
      }}
    }}
  }}
}});
"""

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{escape(report_title)}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
{classification_extra_head}
<style>{STYLE}{classification_extra_css}</style>
</head>
<body>
<header>
  <div class="wrap">
    <div class="eyebrow">TER</div>
    <h1>{escape(report_title)}</h1>
    <p class="lead">Report automatically generated from a membership inference attack <code>summary.json</code> and the <code>targets.csv</code> file. It summarizes the attack configuration, its performance, the IN vs OUT comparison and complexity counters.</p>
  </div>
</header>

<nav>
  <div class="wrap">
    <a href="#summary">Summary</a>
    <a href="#protocol">Configuration</a>
    {'<a href="#anonymization">Anonymization</a>' if metrics else ''}
    <a href="#mia">MIA</a>
    <a href="#stages">Stage 1 / Stage 2</a>
    {'<a href="#schema-matching">Schema matching</a>' if schema_section else ''}
    <a href="#distribution">IN vs OUT analysis</a>
    {'<a href="#complexity">Complexity</a>' if op_rows else ''}
    {'<a href="#classification">Classification</a>' if classification_section else ''}
    <a href="#details">Raw details</a>
  </div>
</nav>

<main class="wrap">
<section id="summary" class="card">
  <h2 class="section-title">1. Executive summary</h2>
  <div class="grid">{metrics_html}</div>
  <div class="two" style="margin-top:16px;">
    <div class="callout"><strong>Synthesis:</strong> {synthesis_html}</div>
    <div class="callout"><strong>Files used:</strong><br>{make_table(['File', 'Path'], files_rows)}</div>
  </div>
</section>

<section id="protocol" class="card">
  <h2 class="section-title">2. Configuration</h2>
  {make_table(['Parameter', 'Value'], config_rows)}
</section>

{anonymization_block}

<section id="mia" class="card">
  <h2 class="section-title">4. Membership inference attack analysis</h2>
  <div class="grid">
    <div class="metric"><div class="k">TP</div><div class="v good">{fmt_int(tp)}</div></div>
    <div class="metric"><div class="k">TN</div><div class="v accent">{fmt_int(tn)}</div></div>
    <div class="metric"><div class="k">FP</div><div class="v danger">{fmt_int(fp)}</div></div>
    <div class="metric"><div class="k">FN</div><div class="v warn">{fmt_int(fn)}</div></div>
  </div>

  <div class="two" style="margin-top:16px;">
    <div>
      <h3 style="margin-top:0">Confusion matrix</h3>
      <div class="cm">
        <div></div>
        <div class="label">Predicted IN</div>
        <div class="label">Predicted OUT</div>
        <div class="label">Actual IN</div>
        <div class="cell tp"><strong>{fmt_int(tp)}</strong>TP · correctly detected members</div>
        <div class="cell fn"><strong>{fmt_int(fn)}</strong>FN · missed members</div>
        <div class="label">Actual OUT</div>
        <div class="cell fp"><strong>{fmt_int(fp)}</strong>FP · false members</div>
        <div class="cell tn"><strong>{fmt_int(tn)}</strong>TN · correctly rejected non-members</div>
      </div>
    </div>
    <div>
      {make_table(
        ['Indicator', 'Value'],
        [
          ['Accuracy', fmt_pct(accuracy)],
          ['Precision', fmt_pct(precision)],
          ['Member recall', fmt_pct(recall)],
          ['Non-member TNR', fmt_pct(tnr)],
          ['F1-score', fmt_pct(f1)],
          ['Predicted IN / OUT', f'{fmt_int(predicted_in_count)} / {fmt_int(predicted_out_count)}'],
          ['Avg. compatible count (IN)', fmt_float(statistics.mean(member_count) if member_count else None)],
          ['Avg. compatible count (OUT)', fmt_float(statistics.mean(non_member_count) if non_member_count else None)],
          ['Avg. compatible fraction (IN)', fmt_pct(statistics.mean(member_frac) if member_frac else None)],
          ['Avg. compatible fraction (OUT)', fmt_pct(statistics.mean(non_member_frac) if non_member_frac else None)],
        ],
      )}
    </div>
  </div>
</section>

{stage_section}
{schema_section}
{hist_section}
{op_section}
{classification_section}

<section id="details" class="card">
  <h2 class="section-title">10. Raw details of summary.json</h2>
  <p class="section-subtitle">This section displays the main fields of the attack summary, without detailed interpretation.</p>
  {make_table(['Field', 'Value'], summary_table_rows)}
  <p class="small" style="margin-top:14px;">Report generated on {escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>
</section>
</main>

<script>
const textColor = '#57606a';
const gridColor = 'rgba(31, 35, 40, 0.08)';
{classification_extra_js_helpers}
{chart_js}
{schema_chart_js}
{classification_chart_js}
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc, encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    project_root, summary_json, metrics_json, targets_csv, config_json, schema_results_json, schema_pairs_csv, classification_json = resolve_paths(args)
    summary = read_json(summary_json)
    attack_id = str(summary.get("attack_id", summary_json.parent.name))

    if args.output:
        output_path = args.output.resolve()
    else:
        output_path = summary_json.parent / "report.html"

    report_path = build_report(
        project_root=project_root,
        summary_json=summary_json,
        metrics_json=metrics_json,
        config_json=config_json,
        targets_csv=targets_csv,
        output_path=output_path,
        title=args.title,
        schema_results_json=schema_results_json,
        schema_pairs_csv=schema_pairs_csv,
        classification_json=classification_json,
    )
    print(f"[OK] Report generated: {report_path}")


if __name__ == "__main__":
    main()
