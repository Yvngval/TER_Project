"""
attacks.py — Simulation d'attaques de ré-identification (version optimisée).

Implémente :
  - Linkage Attack : l'attaquant connaît un sous-ensemble de QI
    d'une cible et essaie de la retrouver dans le dataset anonymisé.
  - Membership Inference Attack (simplifié).
"""

import pandas as pd
import numpy as np
from collections import Counter


def select_targets(df_original: pd.DataFrame, n_targets: int,
                   seed: int = 42) -> pd.DataFrame:
    """Sélectionne aléatoirement N cibles dans le dataset original."""
    rng = np.random.RandomState(seed)
    n = min(n_targets, len(df_original))
    indices = rng.choice(len(df_original), size=n, replace=False)
    return df_original.iloc[indices].reset_index(drop=True)


def _parse_interval(val: str):
    """Parse '30-34' → (30, 34) ou None."""
    if "-" in val and val[0].isdigit():
        parts = val.split("-")
        if len(parts) == 2:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                return None
    return None


def _build_match_index(df_anon: pd.DataFrame, qi_columns: list[str]) -> dict:
    """Pré-calcule un index pour matcher rapidement les cibles."""
    index = {}
    for qi in qi_columns:
        col_vals = df_anon[qi].astype(str).str.strip().values
        unique_vals = set(col_vals)

        has_intervals = any(
            _parse_interval(v) is not None for v in unique_vals if v != "*"
        )

        wildcard_indices = set()
        for idx, v in enumerate(col_vals):
            if v == "*":
                wildcard_indices.add(idx)

        if has_intervals:
            intervals = []
            for idx, v in enumerate(col_vals):
                if v != "*":
                    parsed = _parse_interval(v)
                    if parsed:
                        intervals.append((parsed[0], parsed[1], idx))
            index[qi] = {"type": "interval", "intervals": intervals,
                         "wildcards": wildcard_indices}
        else:
            val_to_indices = {}
            for idx, v in enumerate(col_vals):
                if v != "*":
                    if v not in val_to_indices:
                        val_to_indices[v] = set()
                    val_to_indices[v].add(idx)
            index[qi] = {"type": "exact", "mapping": val_to_indices,
                         "wildcards": wildcard_indices}
    return index


def _find_candidates(target_row: pd.Series, qi_columns: list[str],
                     match_index: dict, n_rows: int) -> set:
    """Trouve les indices candidats pour une cible."""
    candidate_set = None

    for qi in qi_columns:
        target_val = str(target_row[qi]).strip()
        qi_idx = match_index[qi]
        matching = set(qi_idx["wildcards"])

        if qi_idx["type"] == "exact":
            if target_val in qi_idx["mapping"]:
                matching |= qi_idx["mapping"][target_val]
        else:
            try:
                target_num = int(target_val)
                for low, high, idx in qi_idx["intervals"]:
                    if low <= target_num <= high:
                        matching.add(idx)
            except (ValueError, TypeError):
                pass

        if candidate_set is None:
            candidate_set = matching
        else:
            candidate_set &= matching

        if not candidate_set:
            return set()

    return candidate_set or set()


def linkage_attack(targets: pd.DataFrame, df_anonymized: pd.DataFrame,
                   qi_columns: list[str],
                   sensitive_attr: str = "income") -> dict:
    """
    Simule une attaque par linkage (optimisée avec index).
    """
    n_rows = len(df_anonymized)
    match_index = _build_match_index(df_anonymized, qi_columns)
    sensitive_values = df_anonymized[sensitive_attr].astype(str).str.strip().values

    n_unique = 0
    n_ambiguous = 0
    n_no_match = 0
    n_sensitive_ok = 0
    candidate_sizes = []

    for idx in range(len(targets)):
        target = targets.iloc[idx]
        candidates = _find_candidates(target, qi_columns, match_index, n_rows)
        n_cand = len(candidates)
        actual = str(target[sensitive_attr]).strip()

        if n_cand == 0:
            n_no_match += 1
        elif n_cand == 1:
            n_unique += 1
            cand_idx = next(iter(candidates))
            if sensitive_values[cand_idx] == actual:
                n_sensitive_ok += 1
            candidate_sizes.append(n_cand)
        else:
            n_ambiguous += 1
            cand_vals = [sensitive_values[i] for i in candidates]
            most_common = Counter(cand_vals).most_common(1)[0][0]
            if most_common == actual:
                n_sensitive_ok += 1
            candidate_sizes.append(n_cand)

    n_total = len(targets)
    mean_cand = np.mean(candidate_sizes) if candidate_sizes else 0

    return {
        "re_identification_rate": round(n_unique / n_total, 4) if n_total else 0,
        "ambiguity_rate": round(n_ambiguous / n_total, 4) if n_total else 0,
        "no_match_rate": round(n_no_match / n_total, 4) if n_total else 0,
        "attribute_inference_rate": round(n_sensitive_ok / n_total, 4) if n_total else 0,
        "mean_candidate_set_size": round(mean_cand, 2),
        "n_targets": n_total,
        "n_unique_matches": n_unique,
        "n_ambiguous_matches": n_ambiguous,
        "n_no_matches": n_no_match,
    }


def membership_inference_attack(targets_in: pd.DataFrame,
                                targets_out: pd.DataFrame,
                                df_anonymized: pd.DataFrame,
                                qi_columns: list[str]) -> dict:
    """
    Attaque MIA simplifiée : décide "IN" si au moins un match trouvé.
    """
    n_rows = len(df_anonymized)
    match_index = _build_match_index(df_anonymized, qi_columns)

    tp = fp = tn = fn = 0

    for df_targets, true_label in [(targets_in, 1), (targets_out, 0)]:
        for idx in range(len(df_targets)):
            target = df_targets.iloc[idx]
            candidates = _find_candidates(target, qi_columns, match_index, n_rows)
            predicted = 1 if len(candidates) > 0 else 0

            if predicted == 1 and true_label == 1:
                tp += 1
            elif predicted == 1 and true_label == 0:
                fp += 1
            elif predicted == 0 and true_label == 0:
                tn += 1
            else:
                fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }
