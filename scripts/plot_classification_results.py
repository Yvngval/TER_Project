"""Visualise classification benchmark results — ARX-style page layout.

For each non-ZeroR classifier, generates one page (PNG) showing:
  ┌───────────────────────────┬───────────────────────────┐
  │       INPUT DATA          │       OUTPUT DATA         │
  │  summary metrics          │  summary metrics          │
  │  per-class table          │  per-class table          │
  │  ROC curves               │  ROC curves               │
  │  (Baseline + Input)       │  (Baseline+Original+Anon) │
  └───────────────────────────┴───────────────────────────┘

Usage
-----
  python plot_classification_results.py \\
      --json outputs/classification/<id>_classification.json \\
      [--output-dir outputs/figures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_LABELS: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "naive_bayes":         "Naive Bayes",
    "random_forest":       "Random Forest",
    "zero_r":              "ZeroR",
}

COLOR_BASELINE = "#888888"
COLOR_INPUT    = "#1565C0"   # dark blue  — original data
COLOR_OUTPUT   = "#E65100"   # dark orange — anonymized data

# One distinct color per class (for multi-class datasets)
CLASS_COLORS = ["#1565C0", "#2E7D32", "#6A1B9A", "#BF360C", "#00695C"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe(value, fmt: str = ".4f", fallback: str = "—") -> str:
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return fallback


def _non_zero_r_models(results: dict) -> list[str]:
    return [m for m in results if m != "zero_r"]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_summary(ax: plt.Axes, data: dict, side: str,
                  anon_config: dict | None = None) -> None:
    """Render summary metrics as a small readable table inside an axes.

    For the output side, also shows the anonymization configuration
    (privacy model, utility measure, quasi-identifiers, suppression).
    """
    ax.axis("off")

    if side == "input":
        rows = [
            ["Baseline accuracy", _safe(data.get("baseline_accuracy"))],
            ["Accuracy",          _safe(data.get("accuracy"))],
        ]
        col_labels = ["Metric", "Value"]
    else:
        rows = [
            ["Baseline accuracy",  _safe(data.get("baseline_accuracy"))],
            ["Accuracy",           _safe(data.get("accuracy"))],
            ["Original accuracy",  _safe(data.get("original_accuracy"))],
            ["Relative accuracy",  _safe(data.get("relative_accuracy"))],
            ["Brier skill score",  _safe(data.get("brier_skill_score"))],
        ]
        col_labels = ["Metric", "Value"]

        # Anonymization configuration block
        if anon_config:
            cfg = anon_config
            k   = cfg.get("k") or "—"
            l   = cfg.get("l") or "—"
            t   = cfg.get("t") or "—"
            supp = f"{cfg.get('suppression_limit', '—')} %"
            priv = f"k={k}"
            if str(l) not in ("", "None", "—"):
                priv += f"  l={l}"
            if str(t) not in ("", "None", "—"):
                priv += f"  t={t}"

            qi_list   = cfg.get("quasi_identifiers") or []
            qi_str    = ", ".join(qi_list) if qi_list else "—"
            measure   = cfg.get("utility_measure")   or "—"
            aggregate = cfg.get("utility_aggregate")  or "—"
            n_sup     = cfg.get("n_suppressed", 0)
            sup_rate  = cfg.get("suppression_rate", 0)

            rows += [
                ["", ""],   # separator
                ["— Anonymization config —", ""],
                ["Privacy model",       priv],
                ["Suppression limit",   supp],
                ["Suppressed records",  f"{n_sup}  ({sup_rate*100:.2f} %)"],
                ["Utility measure",     measure],
                ["Aggregation",         aggregate],
                ["Quasi-identifiers",   qi_str],
            ]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.35)

    # Style header row
    for col in range(2):
        table[(0, col)].set_facecolor("#CFD8DC")
        table[(0, col)].set_text_props(fontweight="bold")

    # Style the "Anonymization config" section header (indices are dynamic)
    if side == "output" and anon_config:
        # n_metrics rows + 1 header row = first separator index
        n_metrics = 5  # baseline_accuracy, accuracy, original_accuracy, relative_accuracy, brier_skill_score
        sep_row   = n_metrics + 1   # 1-based because row 0 is the table header
        title_row = sep_row + 1
        for r in [sep_row, title_row]:
            for col in range(2):
                table[(r, col)].set_facecolor("#E3F2FD")
        for col in range(2):
            table[(title_row, col)].set_text_props(fontweight="bold", color="#1A237E")


def _draw_per_class_table(ax: plt.Axes, per_class: dict,
                          summary: dict, side: str) -> None:
    """Render the per-class metrics table + min/avg/max summary rows."""
    ax.axis("off")

    if side == "input":
        col_labels = ["Class", "Sensitivity", "Specificity", "Brier", "AUC", "Baseline AUC"]
        def row_data(cls, m):
            return [
                cls,
                _safe(m.get("sensitivity")),
                _safe(m.get("specificity")),
                _safe(m.get("brier_score")),
                _safe(m.get("auc")),
                _safe(m.get("auc_baseline")),
            ]
    else:
        col_labels = ["Class", "Sensitivity", "Specificity", "Brier", "AUC",
                      "Original AUC", "Relative AUC"]
        def row_data(cls, m):
            return [
                cls,
                _safe(m.get("sensitivity")),
                _safe(m.get("specificity")),
                _safe(m.get("brier_score")),
                _safe(m.get("auc")),
                _safe(m.get("original_auc")),
                _safe(m.get("relative_auc")),
            ]

    cell_text = [row_data(cls, m) for cls, m in per_class.items()]

    # Append min / avg / max rows
    for stat, keys in [
        ("Minimum", ("min_sensitivity", "min_specificity", "min_brier")),
        ("Average", ("avg_sensitivity", "avg_specificity", "avg_brier")),
        ("Maximum", ("max_sensitivity", "max_specificity", "max_brier")),
    ]:
        s, sp, b = [_safe(summary.get(k)) for k in keys]
        auc_avg = _safe(summary.get("avg_auc"))
        if side == "input":
            cell_text.append([stat, s, sp, b, auc_avg, _safe(summary.get("auc_baseline"))])
        else:
            cell_text.append([stat, s, sp, b, auc_avg, "—", "—"])

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)

    n_classes = len(per_class)
    n_rows    = len(cell_text)

    # Style header
    for col in range(len(col_labels)):
        table[(0, col)].set_facecolor("#CFD8DC")
        table[(0, col)].set_text_props(fontweight="bold")

    # Style summary rows (min/avg/max) in light yellow
    for r in range(n_classes + 1, n_rows + 1):
        for col in range(len(col_labels)):
            table[(r, col)].set_facecolor("#FFF9C4")


def _draw_roc(ax: plt.Axes,
              baseline_roc: dict,
              input_roc: dict,
              output_roc: dict | None,
              input_pc: dict,
              output_pc: dict,
              classes: list[str],
              side: str) -> None:
    """Draw ROC curves on the given axes.

    Input side  : Baseline ROC + Input ROC (one curve per class).
    Output side : Baseline ROC + Original ROC (input) + Anonymized ROC (output).
    """
    # Diagonal reference
    ax.plot([0, 1], [0, 1], color="black", linewidth=0.8,
            linestyle="--", zorder=0, label="Random (AUC = 0.50)")

    for cls_idx, cls in enumerate(classes):
        color = CLASS_COLORS[cls_idx % len(CLASS_COLORS)]

        # Baseline ROC — ZeroR
        if baseline_roc and baseline_roc.get(cls):
            base_auc = float(input_pc.get(cls, {}).get("auc_baseline", 0.5))
            roc = baseline_roc[cls]
            lbl = f"Baseline ({cls})  AUC={base_auc:.3f}" if len(classes) > 1 else f"Baseline  AUC={base_auc:.3f}"
            ax.plot(roc["fpr"], roc["tpr"],
                    color=COLOR_BASELINE, linewidth=1.2, linestyle="-.",
                    label=lbl)

        if side == "input":
            # Input ROC
            if input_roc and input_roc.get(cls):
                in_auc = float(input_pc.get(cls, {}).get("auc", float("nan")))
                roc = input_roc[cls]
                lbl = f"Input ({cls})  AUC={in_auc:.3f}" if len(classes) > 1 else f"Input  AUC={in_auc:.3f}"
                ax.plot(roc["fpr"], roc["tpr"],
                        color=color, linewidth=2, label=lbl)

        else:  # output side
            # Original ROC (= input ROC, blue)
            if input_roc and input_roc.get(cls):
                in_auc = float(input_pc.get(cls, {}).get("auc", float("nan")))
                roc = input_roc[cls]
                lbl = f"Original ({cls})  AUC={in_auc:.3f}" if len(classes) > 1 else f"Original  AUC={in_auc:.3f}"
                ax.plot(roc["fpr"], roc["tpr"],
                        color=COLOR_INPUT, linewidth=2, linestyle="-",
                        label=lbl)

            # Anonymized ROC (orange)
            if output_roc and output_roc.get(cls):
                out_auc = float(output_pc.get(cls, {}).get("auc", float("nan")))
                roc = output_roc[cls]
                lbl = f"Anonymized ({cls})  AUC={out_auc:.3f}" if len(classes) > 1 else f"Anonymized  AUC={out_auc:.3f}"
                ax.plot(roc["fpr"], roc["tpr"],
                        color=COLOR_OUTPUT, linewidth=2, linestyle="--",
                        label=lbl)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False positive rate", fontsize=9)
    ax.set_ylabel("True positive rate", fontsize=9)
    ax.set_title("ROC curves", fontsize=9)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main page builder
# ---------------------------------------------------------------------------

def plot_model_page(
    model_name: str,
    model_res: dict,
    baseline_roc: dict,
    classes: list[str],
    experiment_id: str,
    output_dir: Path,
    anon_config: dict | None = None,
) -> None:
    """Generate one ARX-style page for a single classifier."""

    input_data  = model_res.get("input_data",  {})
    output_data = model_res.get("output_data", {})
    input_roc   = input_data.get("roc_data")  or {}
    output_roc  = output_data.get("roc_data") or {}
    input_pc    = input_data.get("per_class",  {})
    output_pc   = output_data.get("per_class", {})
    input_sum   = input_data.get("summary",    {})
    output_sum  = output_data.get("summary",   {})

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 17))
    fig.suptitle(
        f"{MODEL_LABELS.get(model_name, model_name)}\n"
        f"Target variable: income  |  Experiment: {experiment_id}",
        fontsize=11, y=0.99,
    )

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[2.2, 2.2, 4.5],
        hspace=0.55,
        wspace=0.08,
    )

    ax_sum_in   = fig.add_subplot(gs[0, 0])
    ax_sum_out  = fig.add_subplot(gs[0, 1])
    ax_tab_in   = fig.add_subplot(gs[1, 0])
    ax_tab_out  = fig.add_subplot(gs[1, 1])
    ax_roc_in   = fig.add_subplot(gs[2, 0])
    ax_roc_out  = fig.add_subplot(gs[2, 1])

    # Column headers
    for ax, title in [(ax_sum_in, "Input data"), (ax_sum_out, "Output data")]:
        ax.set_title(title, fontsize=11, fontweight="bold",
                     loc="left", pad=6, color="#1A237E")

    # ── Draw sections ─────────────────────────────────────────────────────
    _draw_summary(ax_sum_in,  input_data,  side="input")
    _draw_summary(ax_sum_out, output_data, side="output", anon_config=anon_config)

    _draw_per_class_table(ax_tab_in,  input_pc,  input_sum,  side="input")
    _draw_per_class_table(ax_tab_out, output_pc, output_sum, side="output")

    _draw_roc(ax_roc_in,  baseline_roc, input_roc, None,       input_pc, output_pc, classes, "input")
    _draw_roc(ax_roc_out, baseline_roc, input_roc, output_roc, input_pc, output_pc, classes, "output")

    # Separator line between left and right panels
    fig.add_artist(plt.Line2D(
        [0.505, 0.505], [0.02, 0.96],
        transform=fig.transFigure,
        color="#90A4AE", linewidth=1,
    ))

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = output_dir / f"{experiment_id}__{model_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ARX-style classification results from a JSON file."
    )
    parser.add_argument("--json", required=True,
                        help="Path to a *_classification.json file.")
    parser.add_argument("--output-dir", default="outputs/figures",
                        help="Directory for output PNGs (default: outputs/figures).")
    args = parser.parse_args()

    json_path  = Path(args.json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data          = _load_json(json_path)
    experiment_id = data.get("experiment_id", json_path.stem)
    results       = data.get("results", {})
    baseline_roc  = data.get("baseline_roc_data") or {}

    # Derive class list
    classes = sorted(
        next(iter(results.values()), {})
        .get("input_data", {}).get("per_class", {}).keys()
    )

    # Collect anonymization configuration from JSON top-level fields
    anon_config = {
        "k":                 data.get("k"),
        "l":                 data.get("l"),
        "t":                 data.get("t"),
        "suppression_limit": data.get("suppression_limit"),
        "n_suppressed":      data.get("n_suppressed"),
        "suppression_rate":  data.get("suppression_rate"),
        "utility_measure":   data.get("utility_measure"),
        "utility_aggregate": data.get("utility_aggregate"),
        "quasi_identifiers": data.get("quasi_identifiers"),
    }

    exp_output_dir = output_dir / experiment_id
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment : {experiment_id}")
    print(f"Classes    : {classes}")
    print(f"Output dir : {exp_output_dir}\n")

    for model_name in _non_zero_r_models(results):
        print(f"  → {MODEL_LABELS.get(model_name, model_name)}")
        plot_model_page(
            model_name=model_name,
            model_res=results[model_name],
            baseline_roc=baseline_roc,
            classes=classes,
            experiment_id=experiment_id,
            output_dir=exp_output_dir,
            anon_config=anon_config,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
