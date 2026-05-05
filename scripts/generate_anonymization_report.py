#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


STYLE = r'''
  :root {
    --bg: #ffffff;
    --panel: #f5f7fa;
    --panel-2: #eef1f6;
    --line: #d0d7e3;
    --text: #1a2233;
    --muted: #5a6580;
    --accent: #2563eb;
    --accent-soft: rgba(37, 99, 235, 0.08);
    --good: #16a34a;
    --good-soft: rgba(22, 163, 74, 0.10);
    --warn: #b45309;
    --warn-soft: rgba(180, 83, 9, 0.10);
    --danger: #dc2626;
    --danger-soft: rgba(220, 38, 38, 0.10);
    --purple: #7c3aed;
    --radius: 18px;
    --shadow: 0 4px 20px rgba(0,0,0,.08);
    --font: Inter, system-ui, sans-serif;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0;
    font-family: var(--font);
    color: var(--text);
    background: #ffffff;
    line-height: 1.65;
  }
  header {
    padding: 52px 28px 36px;
    border-bottom: 1px solid var(--line);
    background: linear-gradient(135deg, rgba(37,99,235,.05), rgba(37,99,235,0) 45%);
  }
  .wrap { max-width: 1220px; margin: 0 auto; }
  .eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    background: var(--accent-soft);
    color: var(--accent);
    border: 1px solid rgba(122,162,255,.22);
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
    background: rgba(13,17,23,.88);
    border-bottom: 1px solid var(--line);
  }
  nav .wrap { display: flex; gap: 18px; flex-wrap: wrap; padding: 14px 28px; }
  nav a { color: var(--muted); text-decoration: none; font-size: 13px; font-weight: 600; }
  nav a:hover { color: var(--accent); }
  main { padding: 28px; }
  section { margin: 0 0 22px; }
  .card {
    background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0));
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
  .purple { color: var(--purple); }
  .two { display: grid; gap: 16px; grid-template-columns: 1fr 1fr; }
  .three { display: grid; gap: 16px; grid-template-columns: 1fr 1fr 1fr; }
  .callout {
    border-left: 4px solid var(--accent);
    background: var(--accent-soft);
    border-radius: 0 16px 16px 0;
    padding: 14px 16px;
    color: #1e3a8a;
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
    background: rgba(122,162,255,0.06);
  }
  code {
    font-family: var(--mono);
    font-size: .82rem;
    color: var(--accent);
    background: rgba(122,162,255,0.10);
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
  .chart-wrap { position: relative; height: 300px; }
  .chart-wrap-tall { position: relative; height: 380px; }
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 700;
  }
  .badge-good { background: var(--good-soft); color: var(--good); }
  .badge-warn { background: var(--warn-soft); color: var(--warn); }
  .badge-accent { background: var(--accent-soft); color: var(--accent); }
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
  @media (max-width: 980px) {
    .two { grid-template-columns: 1fr; }
    .three { grid-template-columns: 1fr 1fr; }
  }
  @media (max-width: 640px) { .three { grid-template-columns: 1fr; } }
  @media print {
    nav { display: none; }
    body { background: white; color: black; }
    .card, .metric, table, .chart-card { box-shadow: none; }
  }
'''


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Génère un rapport HTML d'anonymisation et de classification.")
    p.add_argument("--project-root", type=Path, default=Path.cwd())
    p.add_argument("--benchmark-csv", type=Path, default=None)
    p.add_argument("--classification-csv", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--experiment-id", type=str, default=None, help="Filter report to a single experiment ID.")
    return p.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float | None:
    if value is None or value == "" or value == "None":
        return None
    try:
        return float(value)
    except Exception:
        return None


def fmt_int(value: Any) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{int(round(float(value))):,}".replace(",", "\u202f")
    except Exception:
        return escape(str(value))


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.{digits}f}".replace(".", ",")
    except Exception:
        return escape(str(value))


def fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{100.0 * float(value):.{digits}f}\xa0%".replace(".", ",")
    except Exception:
        return escape(str(value))


def fmt_ms(value: Any) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.0f}\xa0ms"
    except Exception:
        return escape(str(value))


def escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def make_table(headers: list[str], rows: list[list[str]]) -> str:
    thead = "<thead><tr>" + "".join(f"<th>{escape(h)}</th>" for h in headers) + "</tr></thead>"
    tbody = "<tbody>" + "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    ) + "</tbody>"
    return f"<table>{thead}{tbody}</table>"


def json_mean(val: Any) -> float | None:
    try:
        d = json.loads(val) if isinstance(val, str) else val
        if isinstance(d, dict) and d:
            vals = [v for v in d.values() if v is not None]
            return statistics.mean(vals) if vals else None
    except Exception:
        pass
    return None


def json_attr_table(val: Any) -> list[tuple[str, float]]:
    try:
        d = json.loads(val) if isinstance(val, str) else val
        if isinstance(d, dict):
            return sorted(d.items())
    except Exception:
        pass
    return []


def _safe_mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.mean(clean) if clean else None


def _group_mean(rows: list[dict], group_key: str, value_key: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        g = row.get(group_key, "?") or "?"
        v = to_float(row.get(value_key))
        if v is not None:
            buckets[g].append(v)
    return {k: statistics.mean(vs) for k, vs in buckets.items()}


def _group_by(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k, "") for k in keys)
        out[key].append(row)
    return out


CHART_COLORS = ["#7aa2ff", "#2ecc71", "#f5b942", "#ff7b7b", "#c792ea", "#56b6c2", "#e5c07b"]


def _chart_color(i: int, alpha: float = 0.85) -> str:
    c = CHART_COLORS[i % len(CHART_COLORS)]
    r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


MODEL_LABELS: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest",
    "zero_r": "ZeroR",
}

ROC_COLORS = ["#7aa2ff", "#2ecc71", "#f5b942", "#ff7b7b", "#c792ea", "#56b6c2", "#e5c07b", "#ff9e64"]


def _embed_png(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def _round_list(lst: list, digits: int = 4) -> list:
    return [round(v, digits) if v is not None else None for v in lst]


def build_clf_detail_sections(project_root: Path, chart_counter: list[int]) -> tuple[str, list[str]]:
    """Build HTML + Chart.js blocks for all classification JSON files."""
    clf_dir = project_root / "outputs" / "classification"
    if not clf_dir.exists():
        return "", []

    json_files = sorted(clf_dir.glob("*_classification.json"))
    if not json_files:
        return "", []

    all_html: list[str] = []
    all_chart_js: list[str] = []

    for json_path in json_files:
        try:
            with json_path.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        exp_id = data.get("experiment_id", json_path.stem)
        results = data.get("results", {})
        baseline_roc = data.get("baseline_roc_data") or {}

        anon_cfg_rows = [
            ["Privacy model", f"k={escape(str(data.get('k', '—')))} / l={escape(str(data.get('l') or '—'))} / t={escape(str(data.get('t') or '—'))}"],
            ["Suppression limit", f"{escape(str(data.get('suppression_limit', '—')))} %"],
            ["Suppressed records", f"{fmt_int(data.get('n_suppressed'))} ({fmt_pct(data.get('suppression_rate'))})"],
            ["Utility measure", escape(str(data.get("utility_measure", "—")))],
            ["Aggregation", escape(str(data.get("utility_aggregate", "—")))],
            ["Quasi-identifiers", escape(", ".join(data.get("quasi_identifiers") or []) or "—")],
            ["Target variable", escape(str(data.get("target", "—")))],
            ["Number of folds", escape(str(data.get("n_folds", "—")))],
        ]

        for model_name, model_res in results.items():
            if model_name == "zero_r":
                continue

            model_label = MODEL_LABELS.get(model_name, model_name)
            inp = model_res.get("input_data", {})
            out = model_res.get("output_data", {})
            inp_pc = inp.get("per_class", {})
            out_pc = out.get("per_class", {})
            inp_sum = inp.get("summary", {})
            out_sum = out.get("summary", {})
            inp_roc = inp.get("roc_data") or {}
            out_roc = out.get("roc_data") or {}
            classes = sorted(inp_pc.keys())

            # ── Summary tables ───────────────────────────────────────────────
            inp_sum_rows = [
                ["Baseline accuracy", fmt_pct(inp.get("baseline_accuracy"))],
                ["Accuracy", fmt_pct(inp.get("accuracy"))],
            ]
            out_sum_rows = [
                ["Baseline accuracy", fmt_pct(out.get("baseline_accuracy"))],
                ["Accuracy", fmt_pct(out.get("accuracy"))],
                ["Original accuracy", fmt_pct(out.get("original_accuracy"))],
                ["Relative accuracy", fmt_float(out.get("relative_accuracy"))],
                ["Brier skill score", fmt_float(out.get("brier_skill_score"))],
            ]

            # ── Per-class tables ─────────────────────────────────────────────
            inp_pc_rows = []
            for cls in classes:
                m = inp_pc.get(cls, {})
                inp_pc_rows.append([
                    f"<code>{escape(cls)}</code>",
                    fmt_float(m.get("sensitivity")),
                    fmt_float(m.get("specificity")),
                    fmt_float(m.get("brier_score")),
                    fmt_float(m.get("auc")),
                    fmt_float(m.get("auc_baseline")),
                ])
            for stat, prefix in [("Minimum", "min"), ("Average", "avg"), ("Maximum", "max")]:
                inp_pc_rows.append([
                    f"<span style='color:var(--warn)'>{stat}</span>",
                    fmt_float(inp_sum.get(f"{prefix}_sensitivity")),
                    fmt_float(inp_sum.get(f"{prefix}_specificity")),
                    fmt_float(inp_sum.get(f"{prefix}_brier")),
                    fmt_float(inp_sum.get(f"{prefix}_auc")),
                    fmt_float(inp_sum.get("auc_baseline")) if prefix == "avg" else "—",
                ])

            out_pc_rows = []
            for cls in classes:
                m = out_pc.get(cls, {})
                out_pc_rows.append([
                    f"<code>{escape(cls)}</code>",
                    fmt_float(m.get("sensitivity")),
                    fmt_float(m.get("specificity")),
                    fmt_float(m.get("brier_score")),
                    fmt_float(m.get("auc")),
                    fmt_float(m.get("original_auc")),
                    fmt_float(m.get("relative_auc")),
                ])
            for stat, prefix in [("Minimum", "min"), ("Average", "avg"), ("Maximum", "max")]:
                out_pc_rows.append([
                    f"<span style='color:var(--warn)'>{stat}</span>",
                    fmt_float(out_sum.get(f"{prefix}_sensitivity")),
                    fmt_float(out_sum.get(f"{prefix}_specificity")),
                    fmt_float(out_sum.get(f"{prefix}_brier")),
                    fmt_float(out_sum.get(f"{prefix}_auc")),
                    "—", "—",
                ])

            # ── ROC chart IDs ────────────────────────────────────────────────
            chart_counter[0] += 1
            cid_inp = f"roc_inp_{chart_counter[0]}"
            chart_counter[0] += 1
            cid_out = f"roc_out_{chart_counter[0]}"

            # ── ROC datasets ─────────────────────────────────────────────────
            def _roc_datasets(side: str) -> list[dict]:
                datasets: list[dict] = []
                # diagonal
                datasets.append({
                    "label": "Random (AUC = 0.50)",
                    "data": [{"x": 0, "y": 0}, {"x": 1, "y": 1}],
                    "borderColor": "rgba(150,150,150,0.5)",
                    "borderDash": [4, 4],
                    "borderWidth": 1,
                    "pointRadius": 0,
                    "showLine": True,
                    "fill": False,
                })
                for i, cls in enumerate(classes):
                    color = ROC_COLORS[i % len(ROC_COLORS)]
                    # baseline
                    if baseline_roc.get(cls):
                        br = baseline_roc[cls]
                        pts = [{"x": round(x, 4), "y": round(y, 4)}
                               for x, y in zip(br["fpr"], br["tpr"])]
                        base_auc = inp_pc.get(cls, {}).get("auc_baseline", 0.5)
                        datasets.append({
                            "label": f"Baseline {cls} AUC={base_auc:.3f}",
                            "data": pts,
                            "borderColor": "rgba(150,150,150,0.6)",
                            "borderWidth": 1,
                            "borderDash": [2, 3],
                            "pointRadius": 0,
                            "showLine": True,
                            "fill": False,
                        })
                    if side == "input":
                        if inp_roc.get(cls):
                            r = inp_roc[cls]
                            auc = inp_pc.get(cls, {}).get("auc", 0)
                            pts = [{"x": round(x, 4), "y": round(y, 4)}
                                   for x, y in zip(r["fpr"], r["tpr"])]
                            datasets.append({
                                "label": f"Input {cls} AUC={auc:.3f}",
                                "data": pts,
                                "borderColor": color,
                                "borderWidth": 2,
                                "pointRadius": 0,
                                "showLine": True,
                                "fill": False,
                            })
                    else:
                        if inp_roc.get(cls):
                            r = inp_roc[cls]
                            auc = inp_pc.get(cls, {}).get("auc", 0)
                            pts = [{"x": round(x, 4), "y": round(y, 4)}
                                   for x, y in zip(r["fpr"], r["tpr"])]
                            datasets.append({
                                "label": f"Original {cls} AUC={auc:.3f}",
                                "data": pts,
                                "borderColor": color,
                                "borderWidth": 2,
                                "pointRadius": 0,
                                "showLine": True,
                                "fill": False,
                            })
                        if out_roc.get(cls):
                            r = out_roc[cls]
                            auc = out_pc.get(cls, {}).get("auc", 0)
                            pts = [{"x": round(x, 4), "y": round(y, 4)}
                                   for x, y in zip(r["fpr"], r["tpr"])]
                            datasets.append({
                                "label": f"Anonymized {cls} AUC={auc:.3f}",
                                "data": pts,
                                "borderColor": color,
                                "borderWidth": 2,
                                "borderDash": [5, 3],
                                "pointRadius": 0,
                                "showLine": True,
                                "fill": False,
                            })
                return datasets

            roc_opts = (
                "{"
                "responsive:true, maintainAspectRatio:false,"
                "plugins:{"
                "legend:{labels:{color:textColor,font:{size:10},boxWidth:20}},"
                "zoom:zoomPlugin"
                "},"
                "scales:{"
                "x:{title:{display:true,text:'False positive rate',color:textColor},"
                "ticks:{color:textColor,font:{size:10}},grid:{color:gridColor},min:0,max:1},"
                "y:{title:{display:true,text:'True positive rate',color:textColor},"
                "ticks:{color:textColor,font:{size:10}},grid:{color:gridColor},min:0,max:1}"
                "}}"
            )

            all_chart_js.append(
                "_charts['{cid}'] = new Chart(document.getElementById('{cid}'), {{type:'scatter',"
                "data:{{datasets:{ds}}},"
                "options:{opts}}});".format(
                    cid=cid_inp,
                    ds=json.dumps(_roc_datasets("input"), ensure_ascii=False),
                    opts=roc_opts,
                )
            )
            all_chart_js.append(
                "_charts['{cid}'] = new Chart(document.getElementById('{cid}'), {{type:'scatter',"
                "data:{{datasets:{ds}}},"
                "options:{opts}}});".format(
                    cid=cid_out,
                    ds=json.dumps(_roc_datasets("output"), ensure_ascii=False),
                    opts=roc_opts,
                )
            )

            # ── Embedded PNG ─────────────────────────────────────────────────
            png_path = (
                project_root / "outputs" / "figures"
                / exp_id / f"{exp_id}__{model_name}.png"
            )
            png_html = ""
            if png_path.exists():
                png_src = _embed_png(png_path)
                png_html = f"""
<details style="margin-top:16px;">
  <summary style="cursor:pointer;color:var(--accent);font-size:.9rem;font-weight:600;">
    View original PNG figure ({escape(png_path.name)})
  </summary>
  <div style="margin-top:12px;overflow-x:auto;">
    <img src="{png_src}" style="max-width:100%;border-radius:12px;border:1px solid var(--line);" alt="Classification figure"/>
  </div>
</details>"""

            section_html = f"""
<section class="card" style="margin-bottom:22px;">
  <h2 class="section-title">{escape(model_label)} — {escape(exp_id[:60])}{'…' if len(exp_id)>60 else ''}</h2>
  <p class="section-subtitle">Experiment: <code>{escape(exp_id)}</code></p>

  <div style="margin-bottom:20px;">
    <h3 style="font-size:1rem;margin:0 0 8px;">Anonymization configuration</h3>
    {make_table(["Parameter", "Value"], anon_cfg_rows)}
  </div>

  <div class="two" style="margin-bottom:20px;">
    <div>
      <h3 style="font-size:1rem;margin:0 0 8px;color:#2563eb;">Input data</h3>
      {make_table(["Metric", "Value"], inp_sum_rows)}
      <div style="margin-top:12px;overflow-x:auto;">
        {make_table(["Class","Sensitivity","Specificity","Brier","AUC","Baseline AUC"], inp_pc_rows)}
      </div>
    </div>
    <div>
      <h3 style="font-size:1rem;margin:0 0 8px;color:#16a34a;">Output data</h3>
      {make_table(["Metric", "Value"], out_sum_rows)}
      <div style="margin-top:12px;overflow-x:auto;">
        {make_table(["Class","Sensitivity","Specificity","Brier","AUC","Original AUC","Relative AUC"], out_pc_rows)}
      </div>
    </div>
  </div>

  <div class="two">
    <div class="chart-card">
      <h3 style="margin-top:0;font-size:.95rem;color:#2563eb;">ROC Curves — Input data</h3>
      <div class="chart-wrap-tall"><canvas id="{escape(cid_inp)}"></canvas></div>
      <button class="btn-reset" onclick="resetZoom('{escape(cid_inp)}')">↺ Reset zoom</button>
    </div>
    <div class="chart-card">
      <h3 style="margin-top:0;font-size:.95rem;color:#16a34a;">ROC Curves — Output data</h3>
      <div class="chart-wrap-tall"><canvas id="{escape(cid_out)}"></canvas></div>
      <button class="btn-reset" onclick="resetZoom('{escape(cid_out)}')">↺ Reset zoom</button>
    </div>
  </div>
  {png_html}
</section>"""
            all_html.append(section_html)

    wrapper = f"""
<section id="clf-detail" class="card" style="margin-bottom:22px;">
  <h2 class="section-title">4. Classification — Per-experiment detail</h2>
  <p class="section-subtitle">
    Per-class metrics (Sensitivity, Specificity, Brier, AUC) and ROC curves
    for <span style="color:#2563eb">Input</span> and
    <span style="color:#16a34a">Output</span> anonymized data.
  </p>
  {"".join(all_html)}
</section>
""" if all_html else ""

    return wrapper, all_chart_js


def build_report(
    project_root: Path,
    benchmark_csv: Path,
    classification_csv: Path,
    output_path: Path,
    title: str | None,
    experiment_id_filter: str | None = None,
) -> Path:
    bench_rows = read_csv_rows(benchmark_csv) if benchmark_csv.exists() else []
    clf_rows = read_csv_rows(classification_csv) if classification_csv.exists() else []

    bench_rows = [r for r in bench_rows if r.get("status") == "success"]
    if experiment_id_filter:
        bench_rows = [r for r in bench_rows if r.get("experiment_id") == experiment_id_filter]
        clf_rows = [r for r in clf_rows if r.get("experiment_id") == experiment_id_filter]
    # deduplicate classification rows
    seen: set[tuple] = set()
    clf_dedup: list[dict] = []
    for r in clf_rows:
        key = (r.get("experiment_id"), r.get("model"))
        if key not in seen:
            seen.add(key)
            clf_dedup.append(r)
    clf_rows = clf_dedup

    report_title = title or "Anonymization & Classification Report"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── KPI summary ───────────────────────────────────────────────────────────
    n_experiments = len(bench_rows)
    k_values = sorted(set(r.get("k", "") for r in bench_rows), key=lambda x: (int(x) if x.isdigit() else 0))
    utility_measures = sorted(set(r.get("utility_measure", "") for r in bench_rows))
    models = sorted(set(r.get("model", "") for r in clf_rows))

    avg_opt_score = _safe_mean([to_float(r.get("optimization_score_min")) for r in bench_rows])
    avg_time_ms = _safe_mean([to_float(r.get("anonymization_time_ms")) for r in bench_rows])
    avg_suppressed = _safe_mean([to_float(r.get("number_of_suppressed_records")) for r in bench_rows])
    avg_eq_classes = _safe_mean([to_float(r.get("number_of_equivalence_classes")) for r in bench_rows])

    # ── Anonymization quality metrics ─────────────────────────────────────────
    quality_metrics_defs = [
        ("discernibility_metric", "Discernibility"),
        ("ambiguity_metric", "Ambiguity"),
        ("average_class_size_metric", "Avg. class size (norm.)"),
        ("granularity_metric", "Granularity (avg.)"),
        ("non_uniform_entropy_metric", "Entropy (avg.)"),
        ("generalization_intensity_metric", "Generalization intensity (avg.)"),
    ]

    def get_metric_value(row: dict, col: str) -> float | None:
        val = row.get(col)
        if not val or val == "None":
            return None
        raw = to_float(val)
        if raw is not None:
            return raw
        return json_mean(val)

    # ── Per-experiment quality table ──────────────────────────────────────────
    bench_table_rows: list[list[str]] = []
    for row in bench_rows:
        exp_id_short = row.get("experiment_id", "")[:60] + ("…" if len(row.get("experiment_id", "")) > 60 else "")
        opt = to_float(row.get("optimization_score_min"))
        disc = get_metric_value(row, "discernibility_metric")
        amb = get_metric_value(row, "ambiguity_metric")
        gran = get_metric_value(row, "granularity_metric")
        entropy = get_metric_value(row, "non_uniform_entropy_metric")
        bench_table_rows.append([
            f"<code title='{escape(row.get('experiment_id', ''))}'>{escape(exp_id_short)}</code>",
            f"<code>{escape(row.get('k', ''))}</code>",
            f"<code>{escape(row.get('l', '') or '—')}</code>",
            escape(row.get("utility_measure", "")),
            escape(row.get("utility_aggregate", "")),
            fmt_float(opt),
            fmt_ms(row.get("anonymization_time_ms")),
            fmt_int(row.get("number_of_suppressed_records")),
            fmt_int(row.get("number_of_equivalence_classes")),
            fmt_float(disc),
            fmt_float(amb),
            fmt_float(gran),
            fmt_float(entropy),
        ])

    # ── Per-attribute quality breakdown (last experiment) ─────────────────────
    attr_quality_html = ""
    if bench_rows:
        last = bench_rows[-1]
        attr_cols = [
            ("granularity_metric", "Granularity"),
            ("non_uniform_entropy_metric", "Non-uniform entropy"),
            ("generalization_intensity_metric", "Generalization intensity"),
            ("attribute_level_squared_error_metric", "Squared Error"),
        ]
        combined: dict[str, dict[str, float]] = {}
        for col, label in attr_cols:
            for attr, val in json_attr_table(last.get(col, "")):
                combined.setdefault(attr, {})
                combined[attr][label] = val

        if combined:
            labels_order = [label for _, label in attr_cols]
            attr_table_rows = []
            for attr, vals in sorted(combined.items()):
                attr_table_rows.append(
                    [f"<code>{escape(attr)}</code>"]
                    + [fmt_float(vals.get(l)) for l in labels_order]
                )
            attr_quality_html = make_table(["Attribute"] + labels_order, attr_table_rows)

    # ── Transformations table (last experiment) ───────────────────────────────
    transformations_html = ""
    if bench_rows:
        last = bench_rows[-1]
        try:
            tr = json.loads(last.get("transformations", "{}") or "{}")
            if tr:
                tr_rows = [[f"<code>{escape(attr)}</code>", fmt_int(level)] for attr, level in sorted(tr.items())]
                transformations_html = make_table(["Quasi-identifier", "Generalization level"], tr_rows)
        except Exception:
            pass

    # ── Classification table ──────────────────────────────────────────────────
    clf_table_rows: list[list[str]] = []
    for row in clf_rows:
        base_acc = to_float(row.get("baseline_accuracy"))
        in_acc = to_float(row.get("input_accuracy"))
        out_acc = to_float(row.get("output_accuracy"))
        delta = (out_acc - base_acc) if (out_acc is not None and base_acc is not None) else None
        delta_str = (
            f"<span class='good'>+{fmt_pct(delta)}</span>"
            if delta is not None and delta >= 0
            else f"<span class='danger'>{fmt_pct(delta)}</span>"
            if delta is not None
            else "—"
        )
        clf_table_rows.append([
            f"<code>{escape(row.get('model', ''))}</code>",
            f"<code>{escape(row.get('k', ''))}</code>",
            fmt_pct(row.get("suppression_rate")),
            fmt_pct(base_acc),
            fmt_pct(in_acc),
            fmt_pct(out_acc),
            delta_str,
            fmt_float(row.get("input_roc_auc_avg")),
            fmt_float(row.get("output_roc_auc_avg")),
            fmt_float(row.get("brier_skill_score")),
        ])

    # ── Classification detail sections ───────────────────────────────────────
    chart_counter = [100]
    clf_detail_html, clf_detail_js = build_clf_detail_sections(project_root, chart_counter)

    # ── Chart data ────────────────────────────────────────────────────────────
    chart_js_blocks: list[str] = []

    # Chart 5: Classification accuracy (baseline / input / output) by model
    if clf_rows:
        clf_by_model = _group_by(clf_rows, "model")
        clf_model_labels = sorted(clf_by_model.keys(), key=lambda t: t[0])
        clf_label_strs = [t[0] for t in clf_model_labels]

        def _clf_mean(key: str) -> list[float | None]:
            result = []
            for model_key in clf_model_labels:
                bucket = clf_by_model[model_key]
                vals = [to_float(r.get(key)) for r in bucket]
                vals = [v for v in vals if v is not None]
                result.append(round(statistics.mean(vals), 6) if vals else None)
            return result

        datasets_acc = [
            {"label": "Baseline", "data": _clf_mean("baseline_accuracy"), "backgroundColor": _chart_color(5, 0.7)},
            {"label": "Input (on original data)", "data": _clf_mean("input_accuracy"), "backgroundColor": _chart_color(0)},
            {"label": "Output (on anonymized data)", "data": _clf_mean("output_accuracy"), "backgroundColor": _chart_color(1)},
        ]
        chart_js_blocks.append(
            "_charts['chartAccuracy'] = new Chart(document.getElementById('chartAccuracy'), {{"
            "type:'bar', data:{{labels:{labels}, datasets:{ds}}}, "
            "options:barOptions('Average accuracy')}});"
            .format(
                labels=json.dumps(clf_label_strs),
                ds=json.dumps(datasets_acc),
            )
        )

        # Chart 6: ROC AUC input vs output by model
        datasets_auc = [
            {"label": "ROC AUC Input", "data": _clf_mean("input_roc_auc_avg"), "backgroundColor": _chart_color(0)},
            {"label": "ROC AUC Output", "data": _clf_mean("output_roc_auc_avg"), "backgroundColor": _chart_color(1)},
        ]
        chart_js_blocks.append(
            "_charts['chartAUC'] = new Chart(document.getElementById('chartAUC'), {{"
            "type:'bar', data:{{labels:{labels}, datasets:{ds}}}, "
            "options:barOptions('Average ROC AUC')}});"
            .format(
                labels=json.dumps(clf_label_strs),
                ds=json.dumps(datasets_auc),
            )
        )

    # ── Synthesis text ────────────────────────────────────────────────────────
    synthesis_parts: list[str] = []
    if n_experiments:
        synthesis_parts.append(
            f"<strong>{n_experiments}</strong> anonymization experiment(s) analysed, "
            f"with k ∈ {{{', '.join(k_values)}}} and "
            f"utility measure(s): <em>{', '.join(utility_measures)}</em>."
        )
    if avg_opt_score is not None:
        synthesis_parts.append(
            f"Average optimization score: <strong>{fmt_float(avg_opt_score)}</strong>."
        )
    if avg_suppressed is not None:
        synthesis_parts.append(
            f"On average, <strong>{fmt_int(avg_suppressed)}</strong> records are suppressed per anonymization."
        )
    if clf_rows:
        synthesis_parts.append(
            f"Classification evaluated on <strong>{len(models)}</strong> model(s): {', '.join(f'<code>{escape(m)}</code>' for m in models)}."
        )
    synthesis_html = " ".join(synthesis_parts) if synthesis_parts else "No data available."

    # ── KPI cards HTML ────────────────────────────────────────────────────────
    kpi_cards = [
        ("Experiments", fmt_int(n_experiments), "accent"),
        ("Avg. optim. score", fmt_float(avg_opt_score), "good"),
        ("Avg. time", fmt_ms(avg_time_ms), ""),
        ("Suppressed records (avg.)", fmt_int(avg_suppressed), "warn"),
        ("Equiv. classes (avg.)", fmt_int(avg_eq_classes), ""),
        ("Evaluated models", fmt_int(len(models)), "purple"),
    ]
    kpi_html = "".join(
        f"<div class='metric'><div class='k'>{escape(k)}</div><div class='v {cls}'>{v}</div></div>"
        for k, v, cls in kpi_cards
    )

    # ── HTML assembly ─────────────────────────────────────────────────────────
    anon_section = ""
    if bench_rows:
        anon_section = f"""
<section id="anonymization" class="card">
  <h2 class="section-title">2. Anonymization Results</h2>
  <p class="section-subtitle">Quality metrics for each successful experiment.</p>
  <div style="overflow-x:auto;">
  {make_table(
      ["Experiment", "k", "l", "Utility measure", "Aggregation",
       "Optim. score", "Time", "Suppressed", "Equiv. classes",
       "Discernibility", "Ambiguity", "Granularity", "Entropy"],
      bench_table_rows,
  )}
  </div>

  <div style="margin-top:20px">
    <h3 style="margin:0 0 12px;font-size:1.1rem;">Applied transformations</h3>
    {transformations_html if transformations_html else "<p class='small'>No transformation data available.</p>"}
  </div>

  <div style="margin-top:20px">
    <h3 style="margin:0 0 12px;font-size:1.1rem;">Per-attribute metrics</h3>
    {attr_quality_html if attr_quality_html else "<p class='small'>No per-attribute data available.</p>"}
  </div>
</section>
"""

    charts_anon_section = ""

    clf_section = ""
    if clf_rows:
        clf_section = f"""
<section id="classification" class="card">
  <h2 class="section-title">3. Classification Results</h2>
  <p class="section-subtitle">Accuracy and ROC AUC comparison before and after anonymization.</p>
  <div style="overflow-x:auto;">
  {make_table(
      ["Model", "k", "Suppression rate", "Baseline acc.", "Input acc.",
       "Output acc.", "Δ vs baseline", "ROC AUC input", "ROC AUC output", "Brier skill"],
      clf_table_rows,
  )}
  </div>
</section>
"""

    charts_clf_section = ""
    if clf_rows:
        charts_clf_section = f"""
<section id="charts-clf" class="card">
  <h2 class="section-title">4. Visualizations — Classification</h2>
  <div class="two">
    <div class="chart-card">
      <h3 style="margin-top:0;font-size:1rem;">Accuracy : Baseline / Input / Output</h3>
      <div class="chart-wrap-tall"><canvas id="chartAccuracy"></canvas></div>
      <button class="btn-reset" onclick="resetZoom('chartAccuracy')">↺ Reset zoom</button>
    </div>
    <div class="chart-card">
      <h3 style="margin-top:0;font-size:1rem;">ROC AUC : Input vs Output</h3>
      <div class="chart-wrap-tall"><canvas id="chartAUC"></canvas></div>
      <button class="btn-reset" onclick="resetZoom('chartAUC')">↺ Reset zoom</button>
    </div>
  </div>
</section>
"""

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{escape(report_title)}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/hammer.js/2.0.8/hammer.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chartjs-plugin-zoom/2.0.1/chartjs-plugin-zoom.min.js"></script>
<style>{STYLE}</style>
</head>
<body>
<header>
  <div class="wrap">
    <div class="eyebrow">TER — Anonymization</div>
    <h1>{escape(report_title)}</h1>
    <p class="lead">
      Automated report of ARX anonymization experiments and utility evaluation via classification.
      Dataset: <code>Adult</code> — Framework: <code>RECITALS</code>.
    </p>
  </div>
</header>

<nav>
  <div class="wrap">
    <a href="#summary">Summary</a>
    {'<a href="#anonymization">Anonymization</a>' if bench_rows else ''}
    {'<a href="#classification">Classification</a>' if clf_rows else ''}
    {'<a href="#charts-clf">Charts</a>' if clf_rows else ''}
    {'<a href="#clf-detail">Per-class detail</a>' if clf_rows else ''}
  </div>
</nav>

<main class="wrap">

<section id="summary" class="card">
  <h2 class="section-title">1. Executive Summary</h2>
  <div class="grid">{kpi_html}</div>
  <div class="callout" style="margin-top:16px;">
    <strong>Summary:</strong> {synthesis_html}
  </div>
  <p class="small" style="margin-top:12px;">
    Sources:
    <code>{escape(str(benchmark_csv))}</code> ·
    <code>{escape(str(classification_csv))}</code>
  </p>
</section>

{anon_section}
{charts_anon_section}
{clf_section}
{charts_clf_section}
{clf_detail_html}

<p class="small" style="text-align:center; padding:24px 0 8px;">
  Report generated on {escape(generated_at)}
</p>
</main>

<script>
const textColor = '#4b5563';
const gridColor = 'rgba(0,0,0,0.08)';
const _charts = {{}};

const zoomPlugin = {{
  zoom: {{
    wheel: {{ enabled: true }},
    pinch: {{ enabled: true }},
    mode: 'xy',
  }},
  pan: {{
    enabled: true,
    mode: 'xy',
  }},
}};

function barOptions(yTitle) {{
  return {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ labels: {{ color: textColor, font: {{ size: 12 }} }} }},
      tooltip: {{ mode: 'index', intersect: false }},
      zoom: zoomPlugin,
    }},
    scales: {{
      x: {{
        ticks: {{ color: textColor, font: {{ size: 11 }} }},
        grid: {{ display: false }},
      }},
      y: {{
        title: {{ display: true, text: yTitle, color: textColor, font: {{ size: 12 }} }},
        ticks: {{ color: textColor, font: {{ size: 11 }} }},
        grid: {{ color: gridColor }},
      }},
    }},
  }};
}}

function resetZoom(id) {{ if (_charts[id]) _charts[id].resetZoom(); }}

{chr(10).join(chart_js_blocks)}
{chr(10).join(clf_detail_js)}
</script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc, encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    benchmark_csv = (
        args.benchmark_csv.resolve()
        if args.benchmark_csv
        else project_root / "outputs" / "benchmark_summary.csv"
    )
    classification_csv = (
        args.classification_csv.resolve()
        if args.classification_csv
        else project_root / "outputs" / "classification_summary.csv"
    )

    if args.output:
        output_path = args.output.resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = project_root / "outputs" / "reports" / f"anonymization_report_{ts}.html"

    report_path = build_report(
        project_root=project_root,
        benchmark_csv=benchmark_csv,
        classification_csv=classification_csv,
        output_path=output_path,
        title=args.title,
        experiment_id_filter=args.experiment_id,
    )
    print(f"[OK] Rapport généré : {report_path}")


if __name__ == "__main__":
    main()
