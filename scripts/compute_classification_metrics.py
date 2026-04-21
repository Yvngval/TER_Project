# Classification metrics mirroring ARX's StatisticsClassification output.
#
# All metrics are computed on the FULL set of accumulated predictions
# (not averaged per fold) — faithful to ARX's global accumulation logic.
#
# Metrics produced:
#   - baseline_accuracy       (ZeroR)
#   - accuracy
#   - original_accuracy       (output side only)
#   - relative_accuracy       (output side only)
#   - brier_skill_score       (output side only)
#   - per_class: sensitivity, specificity, brier_score
#   - summary: min/avg/max of sensitivity, specificity, brier_score
#   - roc_curves: baseline_roc + classifier_roc per class

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

# Number of points used when sampling ROC curves for JSON storage
_ROC_SAMPLE_POINTS = 100


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    baseline_accuracy: float,
    input_accuracy: float | None = None,
    input_brier: float | None = None,
    input_aucs: dict[str, float] | None = None,
    baseline_aucs: dict[str, float] | None = None,
) -> dict:
    """Compute ARX-equivalent classification metrics on all predictions.

    Parameters
    ----------
    y_true           : all true labels concatenated across folds
    y_pred           : all predicted labels concatenated across folds
    y_proba          : all predicted probabilities (n_samples, n_classes)
    classes          : class labels matching columns of y_proba
    baseline_accuracy: accuracy of ZeroR on this dataset
    input_accuracy   : accuracy on input data — required for output side only.
                       If provided, activates relative_accuracy + brier_skill_score.
    input_brier      : global brier score on input data — required for brier_skill_score.
    input_aucs       : per-class AUC from input side — if provided, activates
                       original_auc and relative_auc per class (output side only).
    baseline_aucs    : per-class AUC of ZeroR on original data.  When provided,
                       used as auc_baseline instead of the theoretical 0.5 so
                       the value matches ARX's empirical baseline.

    Returns
    -------
    dict with keys: baseline_accuracy, accuracy, [original_accuracy,
    relative_accuracy, brier_skill_score], per_class, summary, roc_curves
    """
    y_true  = np.asarray(y_true)
    y_pred  = np.asarray(y_pred)
    y_proba = np.asarray(y_proba, dtype=float)
    classes = np.asarray(classes)

    n_classes = len(classes)

    # ── accuracy ──────────────────────────────────────────────────────────
    accuracy = float(accuracy_score(y_true, y_pred))

    # ── one-hot encode true labels (label_binarize) ────────────────────────
    # Returns (n_samples, n_classes); for binary sklearn returns (n_samples, 1)
    # — we expand it to two columns to keep the per-class loop uniform.
    y_true_onehot = label_binarize(y_true, classes=classes)
    if n_classes == 2:
        y_true_onehot = np.hstack([1 - y_true_onehot, y_true_onehot])

    # ── output-only metrics ───────────────────────────────────────────────
    result: dict = {
        "baseline_accuracy": round(baseline_accuracy, 8),
        "accuracy":          round(accuracy, 8),
    }

    if input_accuracy is not None:
        result["original_accuracy"] = round(input_accuracy, 8)
        denom = input_accuracy - baseline_accuracy
        result["relative_accuracy"] = (
            round((accuracy - baseline_accuracy) / denom, 8)
            if abs(denom) > 1e-12 else None
        )

    # ── per-class metrics (one-vs-all) ────────────────────────────────────
    per_class: dict[str, dict] = {}
    sensitivities: list[float] = []
    specificities: list[float] = []
    brier_scores:  list[float] = []
    aucs:          list[float] = []
    roc_data:      dict[str, dict | None] = {}

    for idx, cls in enumerate(classes):
        cls_str    = str(cls)
        y_bin      = y_true_onehot[:, idx].astype(int)
        y_pred_bin = (y_pred == cls).astype(int)
        y_score    = y_proba[:, idx]

        # sensitivity = recall for the positive class
        sensitivity = float(recall_score(y_bin, y_pred_bin, zero_division=0.0))

        # specificity = TN / (TN + FP) — derived from confusion matrix
        tn, fp, _fn, _tp = confusion_matrix(y_bin, y_pred_bin, labels=[0, 1]).ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # per-class brier score
        brier = float(brier_score_loss(y_bin, y_score))

        # AUC + ROC curve points (sampled to _ROC_SAMPLE_POINTS for compact storage)
        try:
            auc = float(roc_auc_score(y_bin, y_score))
            fpr_arr, tpr_arr, _ = roc_curve(y_bin, y_score)
            fpr_samp = np.linspace(0.0, 1.0, _ROC_SAMPLE_POINTS)
            tpr_samp = np.interp(fpr_samp, fpr_arr, tpr_arr)
            roc_data[cls_str] = {
                "fpr": fpr_samp.round(6).tolist(),
                "tpr": tpr_samp.round(6).tolist(),
            }
        except ValueError:
            auc = float("nan")
            roc_data[cls_str] = None

        sensitivities.append(sensitivity)
        specificities.append(specificity)
        brier_scores.append(brier)
        aucs.append(auc)

        # Per-class baseline AUC: use ZeroR's empirical AUC when available,
        # otherwise fall back to the theoretical random-classifier value of 0.5.
        auc_base = (
            float(baseline_aucs.get(cls_str, 0.5))
            if baseline_aucs else 0.5
        )

        # Per-class entry — AUC values only, no fpr/tpr arrays
        entry: dict = {
            "sensitivity":  round(sensitivity, 8),
            "specificity":  round(specificity, 8),
            "brier_score":  round(brier, 8),
            "auc_baseline": round(auc_base, 8),
            "auc":          round(auc, 8),
        }

        # Output-side only: original_auc + relative_auc
        if input_aucs is not None:
            orig_auc = input_aucs.get(cls_str)
            entry["original_auc"] = round(orig_auc, 8) if orig_auc is not None else None
            if orig_auc is not None and not np.isnan(orig_auc) and not np.isnan(auc):
                denom_auc = orig_auc - auc_base
                entry["relative_auc"] = (
                    round((auc - auc_base) / denom_auc, 8)
                    if abs(denom_auc) > 1e-12 else None
                )
            else:
                entry["relative_auc"] = None

        per_class[cls_str] = entry

    # ── global brier score = sum of per-class brier scores ─────────────────
    # Equivalent to mean(sum_k (p_k - y_k)^2) across all samples, derived
    # from sklearn's per-class brier_score_loss values.
    global_brier = float(sum(brier_scores))

    # ── brier_skill_score (output side only) ──────────────────────────────
    if input_brier is not None:
        result["brier_skill_score"] = (
            round(1.0 - global_brier / input_brier, 8)
            if input_brier > 1e-12 else None
        )

    # ── summary: min / avg / max across classes ───────────────────────────
    valid_aucs = [a for a in aucs if not np.isnan(a)]

    # Average baseline AUC across all classes (mirrors ARX summary output)
    if baseline_aucs:
        baseline_auc_vals = [
            float(baseline_aucs.get(str(c), 0.5)) for c in classes
        ]
        summary_auc_baseline = round(float(np.mean(baseline_auc_vals)), 8)
    else:
        summary_auc_baseline = 0.5

    result["per_class"] = per_class
    result["summary"] = {
        "min_sensitivity":  round(float(np.min(sensitivities)),  8),
        "avg_sensitivity":  round(float(np.mean(sensitivities)), 8),
        "max_sensitivity":  round(float(np.max(sensitivities)),  8),
        "min_specificity":  round(float(np.min(specificities)),  8),
        "avg_specificity":  round(float(np.mean(specificities)), 8),
        "max_specificity":  round(float(np.max(specificities)),  8),
        "min_brier":        round(float(np.min(brier_scores)),   8),
        "avg_brier":        round(float(np.mean(brier_scores)),  8),
        "max_brier":        round(float(np.max(brier_scores)),   8),
        "auc_baseline":     summary_auc_baseline,
        "avg_auc":          round(float(np.mean(valid_aucs)), 8) if valid_aucs else None,
    }

    # Internal keys — extracted by run_classification_benchmark.py, not in final JSON
    result["_global_brier"]    = global_brier
    result["_per_class_aucs"]  = {str(cls): auc for cls, auc in zip(classes, aucs)}
    result["_roc_data"]        = roc_data

    return result
