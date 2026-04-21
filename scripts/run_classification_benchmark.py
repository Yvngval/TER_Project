# Classification-based utility benchmark (ARX-faithful Python implementation).
#
# For each successful anonymisation experiment in benchmark_summary.csv:
#   1. Load original + anonymised datasets (row-aligned, same index).
#   2. Create KFold splits once on the original dataset (mirrors ARX getFolds).
#   3. Per fold:
#        - Train on original        → predict on original test
#        - Train on anonymised      → predict on anonymised test
#          (suppressed rows excluded from train, kept in test)
#   4. Accumulate all predictions across folds.
#   5. Compute metrics globally on accumulated predictions (ARX-faithful).
#   6. Save per-experiment JSON + global classification_summary.csv.

from __future__ import annotations

import argparse
import csv
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

from classification_models import (
    ADULT_FEATURE_COLS,
    get_model_builders,
    split_num_cat,
)
from compute_classification_metrics import compute_metrics
from common import ensure_dir, load_json, save_json


SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_CONFIG = "configs/classification_config.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(raw: str | Path, base: Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (base / p).resolve()


def _parse_qi(raw: str) -> list[str]:
    return [q.strip() for q in str(raw).split("|") if q.strip()]


def _load_benchmark_summary(summary_path: Path) -> pd.DataFrame:
    """Load benchmark_summary.csv robustly (handles extra columns in data rows)."""
    with open(summary_path, "r", encoding="utf-8") as f:
        reader  = csv.reader(f)
        headers = next(reader)
        n       = len(headers)
        rows    = []
        for line in reader:
            if len(line) >= n:
                rows.append(dict(zip(headers, line[:n])))
            elif line:
                padded = line + [""] * (n - len(line))
                rows.append(dict(zip(headers, padded)))
    return pd.DataFrame(rows, columns=headers)


def _suppressed_mask(anon_df: pd.DataFrame, qi_cols: list[str]) -> pd.Series:
    """True for rows where ALL QI columns contain '*' (suppressed by ARX)."""
    present = [c for c in qi_cols if c in anon_df.columns]
    if not present:
        return pd.Series(False, index=anon_df.index)
    return anon_df[present].eq("*").all(axis=1)


def _safe_proba(model, X: pd.DataFrame, classes: np.ndarray) -> np.ndarray:
    """Return predict_proba aligned to the global `classes` array.

    Some models trained on a fold may have fewer classes than the global set.
    Missing columns are filled with zeros.
    """
    proba_raw     = model.predict_proba(X)
    model_classes = np.asarray(model.classes_).astype(str)
    classes_str   = classes.astype(str)

    if np.array_equal(model_classes, classes_str):
        return proba_raw

    n_samples = proba_raw.shape[0]
    out = np.zeros((n_samples, len(classes)), dtype=float)
    for col_idx, cls in enumerate(model_classes):
        target_idx = np.where(classes_str == cls)[0]
        if target_idx.size > 0:
            out[:, target_idx[0]] = proba_raw[:, col_idx]
    return out


# ---------------------------------------------------------------------------
# Core: one experiment
# ---------------------------------------------------------------------------

def run_classification_for_experiment(
    row: dict,
    orig_df: pd.DataFrame,
    output_root: Path,
    config: dict,
    feature_cols: list[str] | None = None,
    skip_existing: bool = False,
) -> dict:
    """Run classification benchmark for a single anonymisation experiment."""
    experiment_id = row["experiment_id"]
    classif_dir   = ensure_dir(output_root / "classification")
    out_path      = classif_dir / f"{experiment_id}_classification.json"

    if skip_existing and out_path.exists():
        print(f"  [SKIP] {experiment_id}")
        return {"experiment_id": experiment_id, "status": "skipped"}

    csv_path = _resolve(row["csv_path"], PROJECT_ROOT)
    if not csv_path.exists():
        print(f"  [MISSING CSV] {experiment_id}")
        return {"experiment_id": experiment_id, "status": "missing_csv"}

    try:
        target     = config.get("target", "income")
        n_folds    = config.get("n_folds", 10)
        rand_state = config.get("random_state", 42)

        anon_df  = pd.read_csv(csv_path, dtype=str)
        qi_cols  = _parse_qi(row.get("quasi_identifiers", ""))
        supp_mask = _suppressed_mask(anon_df, qi_cols)
        n_suppressed = int(supp_mask.sum())

        # Feature columns: CLI → config → default ADULT_FEATURE_COLS
        if feature_cols is not None:
            feat_cols = list(feature_cols)
        elif config.get("feature_cols"):
            feat_cols = [c for c in config["feature_cols"] if c in orig_df.columns]
        else:
            feat_cols = [c for c in ADULT_FEATURE_COLS if c in orig_df.columns]
        feat_cols = [c for c in feat_cols if c != target]

        num_cols, cat_cols = split_num_cat(feat_cols)
        classes = np.sort(orig_df[target].astype(str).unique())

        builders   = get_model_builders(num_cols, cat_cols, config, rand_state)
        kf         = KFold(n_splits=n_folds, shuffle=True, random_state=rand_state)
        orig_str   = orig_df.astype(str)
        anon_str   = anon_df.astype(str)

        # When suppressed rows were dropped from the public CSV, anon_df is
        # shorter than orig_df and the orig KFold indices are out of bounds.
        # In that case use a separate KFold on anon_str; supp_mask is trivially
        # all-False so no per-fold masking is needed.
        anon_row_aligned = len(anon_df) == len(orig_df)
        kf_anon = kf if anon_row_aligned else KFold(n_splits=n_folds, shuffle=True, random_state=rand_state)

        X_orig = orig_str[feat_cols]
        y_orig = orig_str[target].values

        # Per-model results
        store: dict[str, dict] = {}

        for name, build_fn in builders.items():

            # ── Input (original) side — sklearn cross_val_predict ────────
            # cross_val_predict handles KFold splits internally: for each
            # fold it trains on train split and predicts on test split,
            # then concatenates all out-of-fold predictions in original
            # row order — faithful to ARX's global accumulation logic.
            y_proba_orig = cross_val_predict(
                build_fn(), X_orig, y_orig,
                cv=kf, method="predict_proba",
            )
            # Derive hard predictions from predicted probabilities (argmax).
            # This avoids a second cross_val_predict call and is equivalent
            # to calling predict() directly.
            y_pred_orig = classes[np.argmax(y_proba_orig, axis=1)]

            # ── Output (anonymised) side — custom loop for suppression ───
            # Cannot use cross_val_predict here: training must exclude
            # suppressed rows (mirrors ARX isOutlier check), which requires
            # a fold-level mask applied to the anonymised dataset.
            y_true_anon_list:  list[np.ndarray] = []
            y_pred_anon_list:  list[np.ndarray] = []
            y_proba_anon_list: list[np.ndarray] = []

            for train_idx, test_idx in kf_anon.split(anon_str):
                mask_train   = ~supp_mask.iloc[train_idx].values
                X_anon_train = anon_str.iloc[train_idx][mask_train][feat_cols]
                y_anon_train = anon_str.iloc[train_idx][mask_train][target].values
                # All train rows suppressed — skip this fold
                if len(X_anon_train) == 0:
                    continue
                # Test: keep ALL rows (mirrors ARX outputHandle evaluation)
                X_anon_test = anon_str.iloc[test_idx][feat_cols]
                y_anon_test = anon_str.iloc[test_idx][target].values

                model_anon = build_fn()
                model_anon.fit(X_anon_train, y_anon_train)
                y_pred_anon_list.append(model_anon.predict(X_anon_test))
                y_proba_anon_list.append(_safe_proba(model_anon, X_anon_test, classes))
                y_true_anon_list.append(y_anon_test)

            store[name] = {
                "y_true_orig":  y_orig,
                "y_pred_orig":  y_pred_orig,
                "y_proba_orig": y_proba_orig,
                "y_true_anon":  y_true_anon_list,
                "y_pred_anon":  y_pred_anon_list,
                "y_proba_anon": y_proba_anon_list,
            }

        # ── Compute global metrics ───────────────────────────────────────
        # ZeroR baseline accuracy derived from its out-of-fold predictions
        zero_r_store = store.get("zero_r", {})
        if zero_r_store and len(zero_r_store.get("y_true_orig", [])) > 0:
            baseline_acc = float(np.mean(
                zero_r_store["y_true_orig"] == zero_r_store["y_pred_orig"]
            ))
        else:
            # Fallback: majority class ratio
            vals, counts = np.unique(orig_str[target].values, return_counts=True)
            baseline_acc = float(counts.max() / counts.sum())

        # Pre-compute ZeroR metrics to extract empirical auc_baseline + baseline ROC
        # (mirrors ARX: baseline AUC is ZeroR's actual AUC, not theoretical 0.5)
        baseline_aucs:    dict[str, float] | None = None
        baseline_roc_data: dict[str, dict] | None  = None
        if zero_r_store and len(zero_r_store.get("y_true_orig", [])) > 0:
            _zr_tmp = compute_metrics(
                y_true=zero_r_store["y_true_orig"],
                y_pred=zero_r_store["y_pred_orig"],
                y_proba=zero_r_store["y_proba_orig"],
                classes=classes,
                baseline_accuracy=baseline_acc,
            )
            baseline_aucs     = _zr_tmp.get("_per_class_aucs")
            baseline_roc_data = _zr_tmp.get("_roc_data")

        results: dict[str, dict] = {}
        summary_rows: list[dict] = []

        for name, s in store.items():

            if len(s["y_true_orig"]) == 0:
                continue

            yt_orig  = s["y_true_orig"]
            yp_orig  = s["y_pred_orig"]
            ypr_orig = s["y_proba_orig"]

            # ── Input data metrics ───────────────────────────────────────
            input_metrics = compute_metrics(
                y_true=yt_orig,
                y_pred=yp_orig,
                y_proba=ypr_orig,
                classes=classes,
                baseline_accuracy=baseline_acc,
                input_accuracy=None,
                input_brier=None,
                input_aucs=None,
                baseline_aucs=baseline_aucs,
            )
            input_brier     = input_metrics.pop("_global_brier")
            input_aucs_dict = input_metrics.pop("_per_class_aucs", {})
            input_metrics["roc_data"] = input_metrics.pop("_roc_data", None)

            # ── Output data metrics ──────────────────────────────────────
            output_metrics: dict = {}
            if s["y_true_anon"]:
                yt_anon  = np.concatenate(s["y_true_anon"])
                yp_anon  = np.concatenate(s["y_pred_anon"])
                ypr_anon = np.concatenate(s["y_proba_anon"])

                output_metrics = compute_metrics(
                    y_true=yt_anon,
                    y_pred=yp_anon,
                    y_proba=ypr_anon,
                    classes=classes,
                    baseline_accuracy=baseline_acc,
                    input_accuracy=input_metrics["accuracy"],
                    input_brier=input_brier,
                    input_aucs=input_aucs_dict,
                    baseline_aucs=baseline_aucs,
                )
                output_metrics.pop("_global_brier", None)
                output_metrics.pop("_per_class_aucs", None)
                output_metrics["roc_data"] = output_metrics.pop("_roc_data", None)

            results[name] = {
                "input_data":  input_metrics,
                "output_data": output_metrics,
            }

            # Flat row for CSV summary
            summary_rows.append({
                "experiment_id":      experiment_id,
                "model":              name,
                "quasi_identifiers":  "|".join(qi_cols),
                "k":                  row.get("k"),
                "l":                  row.get("l"),
                "t":                  row.get("t"),
                "suppression_limit":  row.get("suppression_limit"),
                "backend":            row.get("backend"),
                "n_suppressed":       n_suppressed,
                "suppression_rate":   round(n_suppressed / len(orig_df), 6),
                "baseline_accuracy":  input_metrics.get("baseline_accuracy"),
                "input_accuracy":     input_metrics.get("accuracy"),
                "output_accuracy":    output_metrics.get("accuracy"),
                "relative_accuracy":  output_metrics.get("relative_accuracy"),
                "brier_skill_score":  output_metrics.get("brier_skill_score"),
                "input_roc_auc_avg":  _avg_auc(input_metrics),
                "output_roc_auc_avg": _avg_auc(output_metrics),
            })

        # ── Save JSON ────────────────────────────────────────────────────
        payload = {
            "experiment_id":      experiment_id,
            "target":             target,
            "feature_cols":       feat_cols,
            "n_folds":            n_folds,
            "n_records_orig":     len(orig_df),
            "n_suppressed":       n_suppressed,
            "suppression_rate":   round(n_suppressed / len(orig_df), 6),
            "quasi_identifiers":  qi_cols,
            "k":                  row.get("k"),
            "l":                  row.get("l"),
            "t":                  row.get("t"),
            "suppression_limit":  row.get("suppression_limit"),
            "backend":            row.get("backend"),
            "utility_measure":    row.get("utility_measure"),
            "utility_aggregate":  row.get("utility_aggregate"),
            "classifiers_config": config.get("classifiers", {}),
            "baseline_roc_data":  baseline_roc_data,
            "results":            results,
        }
        save_json(out_path, payload)

        # Print quick summary for logistic regression
        lr = results.get("logistic_regression", {})
        inp_acc = lr.get("input_data",  {}).get("accuracy", float("nan"))
        out_acc = lr.get("output_data", {}).get("accuracy", float("nan"))
        rel_acc = lr.get("output_data", {}).get("relative_accuracy", float("nan"))
        print(
            f"  [OK] {experiment_id}"
            f"  |  LR input={inp_acc:.4f}"
            f"  output={out_acc:.4f}"
            f"  relative={rel_acc:.4f}"
            f"  suppressed={n_suppressed}"
        )

        return {
            "experiment_id": experiment_id,
            "status":        "success",
            "summary_rows":  summary_rows,
            "output_path":   str(out_path),
        }

    except Exception as exc:
        print(f"  [ERROR] {experiment_id}: {exc}")
        traceback.print_exc()
        return {
            "experiment_id": experiment_id,
            "status":        "error",
            "error":         str(exc),
        }


def _avg_auc(metrics: dict) -> float | None:
    """Average AUC across classes from per_class auc values."""
    per_class = metrics.get("per_class", {})
    aucs = [
        v["auc"] for v in per_class.values()
        if isinstance(v.get("auc"), float) and not np.isnan(v["auc"])
    ]
    return round(float(np.mean(aucs)), 8) if aucs else None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_classification_benchmark(
    *,
    summary_path: Path,
    orig_data_path: Path,
    output_root: Path,
    config_path: Path,
    feature_cols: list[str] | None = None,
    skip_existing: bool = False,
    config_override: dict | None = None,
    experiment_id: str | None = None,
) -> None:
    """Run classification benchmark for every successful experiment.

    Parameters
    ----------
    experiment_id : if provided, only the experiment with this ID is processed.
    """
    config = config_override if config_override is not None else load_json(config_path)
    summary_df = _load_benchmark_summary(summary_path)
    successful = summary_df[summary_df["status"] == "success"].copy()

    # Filter to a single experiment when --experiment-id is given
    if experiment_id:
        mask = successful["experiment_id"] == experiment_id
        if not mask.any():
            print(f"[ERROR] experiment_id '{experiment_id}' not found in {summary_path}.")
            return
        successful = successful[mask].copy()

    print(f"Config           : {config_path}")
    print(f"Original dataset : {orig_data_path}")
    print(f"Experiments      : {len(successful)} selected / {len(summary_df)} total")
    print(f"Folds            : {config.get('n_folds', 10)}  |  target={config.get('target', 'income')}")

    orig_df = pd.read_csv(orig_data_path, dtype=str)

    all_summary_rows: list[dict] = []

    for _, row in successful.iterrows():
        print(f"\n{row['experiment_id']}")
        result = run_classification_for_experiment(
            row=row.to_dict(),
            orig_df=orig_df,
            output_root=output_root,
            config=config,
            feature_cols=feature_cols,
            skip_existing=skip_existing,
        )
        if result.get("status") == "success":
            all_summary_rows.extend(result.get("summary_rows", []))

    # ── Write classification_summary.csv ────────────────────────────────
    summary_out = output_root / "classification_summary.csv"
    if all_summary_rows:
        fieldnames: list[str] = []
        for r in all_summary_rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(summary_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_summary_rows)

    print("\n" + "=" * 60)
    print("Classification benchmark done.")
    print(f"  Summary rows : {len(all_summary_rows)}")
    print(f"  Saved to     : {summary_out}")

    # ── Auto-generate visualisations ─────────────────────────────────────
    try:
        from plot_classification_results import plot_model_page, _load_json, _non_zero_r_models
        figures_dir = ensure_dir(output_root / "figures")
        classif_dir = output_root / "classification"
        # When a single experiment is targeted, only regenerate its figure.
        if experiment_id:
            json_files = sorted(classif_dir.glob(f"{experiment_id}_classification.json"))
        else:
            json_files = sorted(classif_dir.glob("*_classification.json"))

        if json_files:
            print("\n" + "=" * 60)
            print("Generating classification figures...")
            for json_path in json_files:
                data          = _load_json(json_path)
                experiment_id = data.get("experiment_id", json_path.stem)
                results       = data.get("results", {})
                baseline_roc  = data.get("baseline_roc_data") or {}
                classes       = sorted(
                    next(iter(results.values()), {})
                    .get("input_data", {}).get("per_class", {}).keys()
                )
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
                exp_figures_dir = ensure_dir(figures_dir / experiment_id)
                print(f"\n  {experiment_id}")
                for model_name in _non_zero_r_models(results):
                    plot_model_page(
                        model_name=model_name,
                        model_res=results[model_name],
                        baseline_roc=baseline_roc,
                        classes=classes,
                        experiment_id=experiment_id,
                        output_dir=exp_figures_dir,
                        anon_config=anon_config,
                    )
            print(f"\n  Figures saved to: {figures_dir}")
    except ImportError:
        print("\n  [INFO] matplotlib not available — skipping figure generation.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classification-based utility benchmark (ARX-faithful)."
    )
    parser.add_argument(
        "--summary",
        default="outputs/benchmark_summary.csv",
        help="Path to benchmark_summary.csv (default: outputs/benchmark_summary.csv)",
    )
    parser.add_argument(
        "--original-data",
        default="data/adult.csv",
        help="Path to the original dataset (default: data/adult.csv)",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root output directory (default: outputs)",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to classification_config.json (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--features",
        default=None,
        help=(
            "Comma-separated list of feature columns to use. "
            "If omitted, all Adult columns except the target are used."
        ),
    )
    parser.add_argument(
        "--exclude-features",
        default=None,
        help="Comma-separated list of columns to exclude from features.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target column name. Overrides the value in classification_config.json.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments whose classification JSON already exists.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Run classification only for this experiment ID (default: all).",
    )
    args = parser.parse_args()

    # Build explicit feature list from CLI args
    feature_cols: list[str] | None = None
    if args.features:
        feature_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    if args.exclude_features:
        exclude      = {c.strip() for c in args.exclude_features.split(",") if c.strip()}
        base         = feature_cols if feature_cols is not None else ADULT_FEATURE_COLS
        feature_cols = [c for c in base if c not in exclude]

    config_path = _resolve(args.config, PROJECT_ROOT)

    # Load config and apply CLI overrides before passing to benchmark
    config = load_json(config_path)
    if args.target is not None:
        config["target"] = args.target

    run_classification_benchmark(
        summary_path=_resolve(args.summary,       PROJECT_ROOT),
        orig_data_path=_resolve(args.original_data, PROJECT_ROOT),
        output_root=_resolve(args.output_root,    PROJECT_ROOT),
        config_path=config_path,
        feature_cols=feature_cols,
        skip_existing=args.skip_existing,
        config_override=config,
        experiment_id=args.experiment_id,
    )


if __name__ == "__main__":
    main()
