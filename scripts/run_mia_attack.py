from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import ensure_dir, load_json, save_json

EPS = 1e-12
SUPPRESSION_SCORE = 0.25
GENERALIZED_SCORE = 0.50
SUPPRESSION_TOKENS = {"", "*"}


# Run one membership inference attack against an anonymized dataset and save attack results.


# Parse a comma-separated string into a list of values.
def parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


# Build a unique name for one MIA run.
def make_attack_id(anonymized_path: Path, known_qids: list[str], n_targets: int, seed: int) -> str:
    qid_part = "-".join(known_qids)
    return f"{anonymized_path.stem}__mia_{qid_part}__targets_{n_targets}__seed_{seed}"


# Load a CSV file as strings only and normalize headers / cell values.
def read_csv_str(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [str(col).strip() for col in df.columns]
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df


# Load and validate the runtime anonymization config.
def load_runtime_config(config_path: Path) -> dict[str, Any]:
    payload = load_json(config_path)
    if "hierarchies" not in payload:
        raise ValueError(
            "The config passed to run_mia_attack.py must be a runtime config containing a 'hierarchies' mapping. "
            "Use a generated config from outputs/configs/... ."
        )
    return payload


# Check whether one anonymized value is an explicit suppression token.
def is_suppressed_value(value: str) -> bool:
    value = str(value).strip()
    if value in SUPPRESSION_TOKENS:
        return True
    return bool(value) and set(value) == {"*"}


# Load all rows from one hierarchy CSV file.
def load_hierarchy_rows(hierarchy_path: str | Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with Path(hierarchy_path).open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            cleaned = [str(cell).strip() for cell in row if str(cell).strip() != ""]
            if cleaned:
                rows.append(cleaned)
    return rows


# Infer the most specific hierarchy level compatible with the values actually visible in the anonymized CSV.
def infer_last_visible_level(hierarchy_rows: list[list[str]], observed_values: list[str]) -> int:
    observed = {
        str(value).strip()
        for value in observed_values
        if not is_suppressed_value(value)
    }
    if not observed:
        return 0

    max_depth = max(len(row) for row in hierarchy_rows)
    compatible_levels: list[int] = []
    for level in range(max_depth):
        labels = {
            str(row[level]).strip()
            for row in hierarchy_rows
            if level < len(row)
        }
        if observed.issubset(labels):
            compatible_levels.append(level)

    if not compatible_levels:
        return 0

    return min(compatible_levels)


# Build the reduced attacker knowledge for one attribute.
def build_attacker_projection_for_attr(
    *,
    attr: str,
    hierarchy_path: str | Path,
    observed_anonymized_values: pd.Series,
) -> dict[str, Any]:
    hierarchy_rows = load_hierarchy_rows(hierarchy_path)
    observed_values = observed_anonymized_values.astype(str).unique().tolist()
    visible_level = infer_last_visible_level(hierarchy_rows, observed_values)

    projection: dict[str, str] = {}
    for row in hierarchy_rows:
        raw_value = str(row[0]).strip()
        exposed_value = str(row[min(visible_level, len(row) - 1)]).strip()
        projection[raw_value] = exposed_value

    return {
        "attribute": attr,
        "hierarchy_path": str(hierarchy_path),
        "visible_level": int(visible_level),
        "observed_values": sorted({str(v).strip() for v in observed_values}),
        "projection": projection,
        "n_projected_values": int(len(projection)),
    }


# Build the reduced attacker knowledge for all known QIs.
def build_attacker_knowledge(
    *,
    runtime: dict[str, Any],
    known_qids: list[str],
    df_public: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    hierarchies = runtime.get("hierarchies", {})
    knowledge: dict[str, dict[str, Any]] = {}
    for qid in known_qids:
        hierarchy_path = hierarchies.get(qid)
        if hierarchy_path:
            knowledge[qid] = build_attacker_projection_for_attr(
                attr=qid,
                hierarchy_path=hierarchy_path,
                observed_anonymized_values=df_public[qid].astype(str),
            )
        else:
            projection = {str(v).strip(): str(v).strip() for v in df_public[qid].astype(str).unique().tolist()}
            knowledge[qid] = {
                "attribute": qid,
                "hierarchy_path": None,
                "visible_level": 0,
                "observed_values": sorted({str(v).strip() for v in df_public[qid].astype(str).tolist()}),
                "projection": projection,
                "n_projected_values": int(len(projection)),
            }
    return knowledge


# Compute the compatibility score for one attribute value using only reduced attacker knowledge.
def attribute_score(raw_value: str, anonymized_value: str, attacker_attr_knowledge: dict[str, Any] | None) -> float:
    raw_value = str(raw_value).strip()
    anonymized_value = str(anonymized_value).strip()

    if raw_value == anonymized_value:
        return 1.0

    if is_suppressed_value(anonymized_value):
        return SUPPRESSION_SCORE

    if attacker_attr_knowledge is None:
        return 1.0 if raw_value == anonymized_value else 0.0

    projection = attacker_attr_knowledge.get("projection", {})
    exposed_value = projection.get(raw_value)
    if exposed_value is None:
        return 1.0 if raw_value == anonymized_value else 0.0

    if anonymized_value == exposed_value:
        return 1.0 if exposed_value == raw_value else GENERALIZED_SCORE

    return 0.0


# Compute the score vector for one target value over all anonymized rows.
def score_vector_for_target_value(
    qid: str,
    target_value: str,
    anonymized_values_str: pd.Series,
    attacker_knowledge: dict[str, dict[str, Any]],
    score_cache: dict[tuple[str, str], dict[str, float]],
) -> np.ndarray:
    cache_key = (qid, str(target_value).strip())
    if cache_key not in score_cache:
        attacker_attr_knowledge = attacker_knowledge.get(qid)
        mapping = {
            anonymized_value: attribute_score(target_value, anonymized_value, attacker_attr_knowledge)
            for anonymized_value in anonymized_values_str.unique().tolist()
        }
        score_cache[cache_key] = mapping
    return anonymized_values_str.map(score_cache[cache_key]).astype(float).to_numpy()


# Build an inverted index of row positions for each visible anonymized value.
def build_row_index_by_qid_value(
    working_df_str: pd.DataFrame,
    known_qids: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    row_index: dict[str, dict[str, np.ndarray]] = {}
    for qid in known_qids:
        positions: dict[str, list[int]] = {}
        for idx, value in enumerate(working_df_str[qid].astype(str).tolist()):
            positions.setdefault(str(value).strip(), []).append(idx)
        row_index[qid] = {
            value: np.asarray(sorted(indices), dtype=int)
            for value, indices in positions.items()
        }
    return row_index


# List visible anonymized values that remain potentially compatible with one raw target value.
def compatible_anonymized_values_for_target_value(
    *,
    qid: str,
    target_value: str,
    attacker_knowledge: dict[str, dict[str, Any]],
    row_index_by_qid_value: dict[str, dict[str, np.ndarray]],
    compatible_values_cache: dict[tuple[str, str], list[str]],
) -> list[str]:
    cache_key = (qid, str(target_value).strip())
    if cache_key in compatible_values_cache:
        return compatible_values_cache[cache_key]

    attacker_attr_knowledge = attacker_knowledge.get(qid)
    compatible_values = [
        anonymized_value
        for anonymized_value in row_index_by_qid_value[qid].keys()
        if attribute_score(target_value, anonymized_value, attacker_attr_knowledge) > EPS
    ]
    compatible_values_cache[cache_key] = compatible_values
    return compatible_values


# Prefilter the anonymized dataset to rows that do not contradict any known victim attribute.
def prefilter_candidate_indices_for_target(
    *,
    known_qids: list[str],
    known_values: dict[str, str],
    attacker_knowledge: dict[str, dict[str, Any]],
    row_index_by_qid_value: dict[str, dict[str, np.ndarray]],
    compatible_values_cache: dict[tuple[str, str], list[str]],
) -> np.ndarray:
    candidate_indices: np.ndarray | None = None

    for qid in known_qids:
        compatible_values = compatible_anonymized_values_for_target_value(
            qid=qid,
            target_value=known_values[qid],
            attacker_knowledge=attacker_knowledge,
            row_index_by_qid_value=row_index_by_qid_value,
            compatible_values_cache=compatible_values_cache,
        )
        if not compatible_values:
            return np.asarray([], dtype=int)

        qid_candidate_indices = np.unique(
            np.concatenate([row_index_by_qid_value[qid][value] for value in compatible_values])
        )
        if candidate_indices is None:
            candidate_indices = qid_candidate_indices
        else:
            candidate_indices = np.intersect1d(candidate_indices, qid_candidate_indices, assume_unique=False)

        if len(candidate_indices) == 0:
            return np.asarray([], dtype=int)

    return candidate_indices if candidate_indices is not None else np.asarray([], dtype=int)


# Infer the known QIs from the targets dataset when they are not passed explicitly.
def infer_known_qids(df_targets: pd.DataFrame, explicit_known_qids: list[str], target_id_col: str, member_col: str) -> list[str]:
    if explicit_known_qids:
        return explicit_known_qids
    return [col for col in df_targets.columns if col not in {target_id_col, member_col}]


# Append one MIA result row to the global summary CSV.
def append_attack_summary(summary_csv: Path, row: dict[str, Any]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# Predict membership from a combination of score and candidate-set criteria.
def decide_membership(
    *,
    best_score: float,
    compatible_candidate_count: int,
    total_rows: int,
    min_best_score: float,
    max_compatible_candidates: int | None,
    max_compatible_fraction: float | None,
) -> tuple[bool, str]:
    has_candidate = compatible_candidate_count > 0
    score_ok = best_score >= min_best_score
    count_ok = max_compatible_candidates is None or compatible_candidate_count <= max_compatible_candidates
    fraction = (compatible_candidate_count / total_rows) if total_rows else 0.0
    fraction_ok = max_compatible_fraction is None or fraction <= max_compatible_fraction

    predicted_member = has_candidate and score_ok and count_ok and fraction_ok
    reason = (
        f"has_candidate={has_candidate};best_score={best_score:.6f};"
        f"score_ok={score_ok};compatible_candidate_count={compatible_candidate_count};"
        f"count_ok={count_ok};compatible_fraction={fraction:.6f};fraction_ok={fraction_ok}"
    )
    return predicted_member, reason


# Validate all attack inputs before running the MIA.
def _validate_inputs(
    *,
    runtime: dict[str, Any],
    known_qids: list[str],
    target_id_col: str,
    member_col: str,
    df_targets: pd.DataFrame,
    df_public: pd.DataFrame,
    df_eval: pd.DataFrame,
    min_best_score: float,
) -> None:
    if not known_qids:
        raise ValueError("known_qids cannot be empty.")

    runtime_qids = set(runtime.get("quasi_identifiers", []))
    extra_known_attrs = [qi for qi in known_qids if qi not in runtime_qids]

    if target_id_col not in df_targets.columns:
        raise ValueError(f"Targets dataset must contain '{target_id_col}'.")
    if member_col not in df_targets.columns:
        raise ValueError(f"Targets dataset must contain '{member_col}'.")
    if target_id_col not in df_eval.columns:
        raise ValueError(f"Evaluation anonymized dataset must contain '{target_id_col}'.")

    missing_targets_cols = [qi for qi in known_qids if qi not in df_targets.columns]
    missing_public_cols = [qi for qi in known_qids if qi not in df_public.columns]
    if missing_targets_cols:
        raise ValueError(f"Targets dataset misses known QI columns: {missing_targets_cols}")
    if missing_public_cols:
        raise ValueError(f"Public anonymized dataset misses known QI columns: {missing_public_cols}")

    if len(df_public) != len(df_eval):
        raise ValueError("Public and eval anonymized CSV files must have the same row order and row count.")

    if not 0.0 <= float(min_best_score) <= 1.0:
        raise ValueError("min_best_score must be in [0, 1].")


# Compute standard binary-classification metrics for the MIA.
def compute_classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    total = len(y_true)
    pos = sum(y_true)
    neg = total - pos

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    non_member_tnr = tn / neg if neg else 0.0
    non_member_fpr = fp / neg if neg else 0.0
    member_fnr = fn / pos if pos else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "member_recall": recall,
        "member_false_negative_rate": member_fnr,
        "non_member_true_negative_rate": non_member_tnr,
        "non_member_false_positive_rate": non_member_fpr,
    }


# Run the full MIA workflow and save all outputs.
def run_mia_attack(
    *,
    runtime: dict[str, Any],
    df_targets: pd.DataFrame,
    df_public: pd.DataFrame,
    df_eval: pd.DataFrame,
    known_qids: list[str] | None = None,
    target_id_col: str = "record_id",
    member_col: str = "is_member",
    min_best_score: float = 0.5,
    max_compatible_candidates: int | None = 100,
    max_compatible_fraction: float | None = 0.01,
    output_root: str | Path = "outputs",
    name: str | None = None,
    config_path: str | Path | None = None,
    targets_path: str | Path | None = None,
    anonymized_path: str | Path | None = None,
    anonymized_eval_path: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    known_qids = infer_known_qids(df_targets, list(known_qids or []), target_id_col, member_col)

    _validate_inputs(
        runtime=runtime,
        known_qids=known_qids,
        target_id_col=target_id_col,
        member_col=member_col,
        df_targets=df_targets,
        df_public=df_public,
        df_eval=df_eval,
        min_best_score=min_best_score,
    )

    attacker_knowledge = build_attacker_knowledge(
        runtime=runtime,
        known_qids=known_qids,
        df_public=df_public,
    )

    working_df = df_public.copy().reset_index(drop=True)
    working_df[target_id_col] = df_eval[target_id_col].astype(str).reset_index(drop=True)

    working_df_str = working_df.copy()
    for qi in known_qids:
        working_df_str[qi] = working_df_str[qi].astype(str).str.strip()
    working_df_str[target_id_col] = working_df_str[target_id_col].astype(str).str.strip()

    targets_df = df_targets.copy().reset_index(drop=True)
    for col in [target_id_col, member_col] + known_qids:
        targets_df[col] = targets_df[col].astype(str).str.strip()

    id_to_eval_index = {str(record_id): idx for idx, record_id in enumerate(working_df_str[target_id_col].tolist())}

    qid_series = {qi: working_df_str[qi] for qi in known_qids}
    row_index_by_qid_value = build_row_index_by_qid_value(working_df_str, known_qids)
    compatible_values_cache: dict[tuple[str, str], list[str]] = {}
    score_cache: dict[tuple[str, str], dict[str, float]] = {}

    per_target_rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    n_rows = len(working_df_str)

    for _, target in targets_df.iterrows():
        target_id = str(target[target_id_col]).strip()
        truth_member = int(str(target[member_col]).strip())
        known_values = {qi: str(target[qi]).strip() for qi in known_qids}

        candidate_indices = prefilter_candidate_indices_for_target(
            known_qids=known_qids,
            known_values=known_values,
            attacker_knowledge=attacker_knowledge,
            row_index_by_qid_value=row_index_by_qid_value,
            compatible_values_cache=compatible_values_cache,
        )

        compatible_candidate_count = int(len(candidate_indices))
        compatible_candidate_fraction = (compatible_candidate_count / n_rows) if n_rows else 0.0

        normalized_score = np.asarray([], dtype=float)
        if compatible_candidate_count > 0:
            filtered_total_score = np.zeros(compatible_candidate_count, dtype=float)
            for qi in known_qids:
                filtered_qi_series = qid_series[qi].iloc[candidate_indices]
                qi_scores = score_vector_for_target_value(
                    qid=qi,
                    target_value=known_values[qi],
                    anonymized_values_str=filtered_qi_series,
                    attacker_knowledge=attacker_knowledge,
                    score_cache=score_cache,
                )
                filtered_total_score += qi_scores
            normalized_score = filtered_total_score / len(known_qids)

        best_score = float(np.max(normalized_score)) if len(normalized_score) else 0.0
        best_candidate_count = int(np.isclose(normalized_score, best_score, atol=EPS).sum()) if len(normalized_score) else 0
        target_present_in_anonymized = target_id in id_to_eval_index

        predicted_member, decision_reason = decide_membership(
            best_score=best_score,
            compatible_candidate_count=compatible_candidate_count,
            total_rows=n_rows,
            min_best_score=min_best_score,
            max_compatible_candidates=max_compatible_candidates,
            max_compatible_fraction=max_compatible_fraction,
        )

        true_record_best_score = None
        true_record_full_compatible = None
        if target_present_in_anonymized:
            true_idx = id_to_eval_index[target_id]
            true_record_full_compatible = bool(np.any(candidate_indices == true_idx))
            if true_record_full_compatible:
                candidate_pos = int(np.where(candidate_indices == true_idx)[0][0])
                true_record_best_score = float(normalized_score[candidate_pos])
            else:
                true_record_best_score = 0.0

        row = {
            "target_id": target_id,
            "ground_truth_member": truth_member,
            "predicted_member": int(predicted_member),
            "known_qids": "|".join(known_qids),
            "best_score": best_score,
            "best_candidate_count": best_candidate_count,
            "compatible_candidate_count": compatible_candidate_count,
            "compatible_candidate_fraction": compatible_candidate_fraction,
            "target_present_in_anonymized": target_present_in_anonymized,
            "true_record_best_score": true_record_best_score,
            "true_record_full_compatible": true_record_full_compatible,
            "decision_reason": decision_reason,
        }
        for qi in known_qids:
            row[f"known_{qi}"] = known_values[qi]
        per_target_rows.append(row)
        y_true.append(truth_member)
        y_pred.append(int(predicted_member))

    metrics = compute_classification_metrics(y_true, y_pred)

    targets_results_df = pd.DataFrame(per_target_rows)
    member_mask = targets_results_df["ground_truth_member"].astype(int) == 1
    non_member_mask = ~member_mask

    summary = {
        "attack_id": name or make_attack_id(Path(anonymized_path or "anonymized.csv"), known_qids, len(df_targets), seed),
        "config_path": str(config_path) if config_path else None,
        "targets_path": str(targets_path) if targets_path else None,
        "anonymized_path": str(anonymized_path) if anonymized_path else None,
        "anonymized_eval_path": str(anonymized_eval_path) if anonymized_eval_path else None,
        "known_qids": known_qids,
        "target_id_col": target_id_col,
        "member_col": member_col,
        "seed": seed,
        "n_targets": len(df_targets),
        "n_members": int(member_mask.sum()),
        "n_non_members": int(non_member_mask.sum()),
        "min_best_score": min_best_score,
        "max_compatible_candidates": max_compatible_candidates,
        "max_compatible_fraction": max_compatible_fraction,
        "attacker_knowledge": attacker_knowledge,
        "candidate_prefilter_mode": "remove_clearly_contradictory_rows",
        **metrics,
        "member_avg_compatible_candidate_count": float(targets_results_df.loc[member_mask, "compatible_candidate_count"].mean()) if member_mask.any() else None,
        "non_member_avg_compatible_candidate_count": float(targets_results_df.loc[non_member_mask, "compatible_candidate_count"].mean()) if non_member_mask.any() else None,
        "member_avg_best_score": float(targets_results_df.loc[member_mask, "best_score"].mean()) if member_mask.any() else None,
        "non_member_avg_best_score": float(targets_results_df.loc[non_member_mask, "best_score"].mean()) if non_member_mask.any() else None,
    }

    output_root = ensure_dir(Path(output_root))
    attacks_root = ensure_dir(output_root / "attacks" / "mia")
    attack_id = str(summary["attack_id"])
    attack_dir = ensure_dir(attacks_root / attack_id)
    summary_path = attack_dir / "summary.json"
    targets_path_out = attack_dir / "targets.csv"

    save_json(summary_path, summary)
    targets_results_df.to_csv(targets_path_out, index=False)

    summary_row = {
        "attack_id": attack_id,
        "config_path": summary.get("config_path"),
        "targets_path": summary.get("targets_path"),
        "anonymized_path": summary.get("anonymized_path"),
        "anonymized_eval_path": summary.get("anonymized_eval_path"),
        "known_qids": "|".join(known_qids),
        "n_targets": summary["n_targets"],
        "n_members": summary["n_members"],
        "n_non_members": summary["n_non_members"],
        "min_best_score": min_best_score,
        "max_compatible_candidates": max_compatible_candidates,
        "max_compatible_fraction": max_compatible_fraction,
        "accuracy": summary["accuracy"],
        "precision": summary["precision"],
        "recall": summary["recall"],
        "f1": summary["f1"],
        "tp": summary["tp"],
        "tn": summary["tn"],
        "fp": summary["fp"],
        "fn": summary["fn"],
        "member_recall": summary["member_recall"],
        "non_member_true_negative_rate": summary["non_member_true_negative_rate"],
        "summary_json": str(summary_path),
        "targets_csv": str(targets_path_out),
        "candidate_prefilter_mode": "remove_clearly_contradictory_rows",
    }
    append_attack_summary(attacks_root / "mia_summary.csv", summary_row)

    print(f"[OK] {attack_id}")
    print(f"Summary   : {summary_path}")
    print(f"Targets   : {targets_path_out}")

    return {
        "summary": summary,
        "summary_path": summary_path,
        "targets_path": targets_path_out,
    }


# Load all input files and run one MIA from file paths.
def run_mia_attack_from_paths(
    *,
    config_path: str | Path,
    targets_path: str | Path,
    anonymized_path: str | Path,
    anonymized_eval_path: str | Path,
    known_qids: list[str] | None = None,
    target_id_col: str = "record_id",
    member_col: str = "is_member",
    min_best_score: float = 0.5,
    max_compatible_candidates: int | None = 100,
    max_compatible_fraction: float | None = 0.01,
    output_root: str | Path = "outputs",
    name: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    targets_path = Path(targets_path).resolve()
    anonymized_path = Path(anonymized_path).resolve()
    anonymized_eval_path = Path(anonymized_eval_path).resolve()

    runtime = load_runtime_config(config_path)
    df_targets = read_csv_str(targets_path)
    df_public = read_csv_str(anonymized_path)
    df_eval = read_csv_str(anonymized_eval_path)

    return run_mia_attack(
        runtime=runtime,
        df_targets=df_targets,
        df_public=df_public,
        df_eval=df_eval,
        known_qids=known_qids,
        target_id_col=target_id_col,
        member_col=member_col,
        min_best_score=min_best_score,
        max_compatible_candidates=max_compatible_candidates,
        max_compatible_fraction=max_compatible_fraction,
        output_root=output_root,
        name=name,
        config_path=config_path,
        targets_path=targets_path,
        anonymized_path=anonymized_path,
        anonymized_eval_path=anonymized_eval_path,
        seed=seed,
    )


# Parse CLI arguments and launch the MIA.
def main() -> None:
    parser = argparse.ArgumentParser(description="Run one membership inference attack on an anonymized dataset.")
    parser.add_argument("--config", required=True, help="Path to the runtime config JSON.")
    parser.add_argument("--targets", required=True, help="Path to the balanced MIA targets CSV.")
    parser.add_argument("--anonymized", required=True, help="Path to the public anonymized CSV.")
    parser.add_argument("--anonymized-eval", required=True, help="Path to the eval anonymized CSV with record_id.")
    parser.add_argument("--known-qids", default=None, help="Optional comma-separated known QIs. If omitted, infer them from the targets CSV.")
    parser.add_argument("--target-id-col", default="record_id", help="Internal identifier column name.")
    parser.add_argument("--member-col", default="is_member", help="Ground-truth membership label column name.")
    parser.add_argument("--min-best-score", type=float, default=0.5, help="Minimum best compatibility score required to predict member.")
    parser.add_argument("--max-compatible-candidates", type=int, default=100, help="Maximum number of fully compatible candidates allowed to predict member.")
    parser.add_argument("--max-compatible-fraction", type=float, default=0.01, help="Maximum fraction of fully compatible candidates allowed to predict member.")
    parser.add_argument("--output-root", default="outputs", help="Root directory for attack outputs.")
    parser.add_argument("--name", default=None, help="Optional attack name override.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used only for naming consistency.")
    args = parser.parse_args()

    run_mia_attack_from_paths(
        config_path=args.config,
        targets_path=args.targets,
        anonymized_path=args.anonymized,
        anonymized_eval_path=args.anonymized_eval,
        known_qids=parse_csv_list(args.known_qids),
        target_id_col=args.target_id_col,
        member_col=args.member_col,
        min_best_score=args.min_best_score,
        max_compatible_candidates=args.max_compatible_candidates,
        max_compatible_fraction=args.max_compatible_fraction,
        output_root=args.output_root,
        name=args.name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
