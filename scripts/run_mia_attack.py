from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import hashlib

from attack_common import (
    append_attack_summary,
    build_attacker_knowledge,
    is_suppressed_value,
    load_runtime_config,
    read_csv_str,
)
from common import ensure_dir, parse_csv_list, save_json
from linkage_helpers import refinement_match_result
from privjedai_utils import build_privjedai_fuzzy_config
from generate_mia_attack_report import build_report

EPS = 1e-12
SUPPRESSION_SCORE = 0.25
GENERALIZED_SCORE = 0.50


def make_operation_counter() -> dict[str, int]:
    return {
        "row_index_row_visits": 0,
        "compatible_value_cache_hits": 0,
        "compatible_value_cache_misses": 0,
        "compatible_value_tests": 0,
        "targets_evaluated": 0,
        "qid_stage1_cache_hits": 0,
        "qid_stage1_cache_misses": 0,
        "candidate_row_refs_loaded": 0,
        "candidate_intersections": 0,
        "candidate_intersection_input_total": 0,
        "refinement_candidate_row_visits": 0,
        "refinement_exact_tests": 0,
        "refinement_fuzzy_tests": 0,
        "refinement_mask_cells": 0,
        "membership_decisions": 0,
    }


def estimate_total_operations(op_counter: dict[str, int]) -> int:
    return int(
        op_counter["row_index_row_visits"]
        + op_counter["compatible_value_tests"]
        + op_counter["candidate_row_refs_loaded"]
        + op_counter["candidate_intersection_input_total"]
        + op_counter["refinement_candidate_row_visits"]
        + op_counter["refinement_exact_tests"]
        + op_counter["refinement_fuzzy_tests"]
        + op_counter["refinement_mask_cells"]
        + op_counter["membership_decisions"]
    )


HIDDEN_OPERATION_COUNTER_KEYS = {
    "compatible_value_cache_hits",
    "compatible_value_cache_misses",
    "compatible_value_tests",
    "qid_stage1_cache_hits",
    "qid_stage1_cache_misses",
    "candidate_intersections",
    "candidate_intersection_input_total",
    "operation_model",
}


def build_public_operation_counter(op_counter: dict[str, int], *, use_privjedai_fuzzy: bool) -> dict[str, int]:
    public_counter = {
        key: value
        for key, value in op_counter.items()
        if key not in HIDDEN_OPERATION_COUNTER_KEYS
    }
    public_counter["estimated_total_operations"] = estimate_total_operations(op_counter)
    return public_counter


def _maybe_generate_mia_report(
    *,
    output_root: Path,
    summary_path: Path,
    attack_id: str,
    report_title: str | None = None,
) -> Path | None:
    """Generate the HTML MIA report for the current attack if possible."""
    project_root = output_root.parent if output_root.name == "outputs" else output_root
    attack_dir = summary_path.parent
    targets_csv = attack_dir / "targets.csv"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    experiment_id = str(summary.get("attack_id", attack_id)).split("__mia_")[0]

    metrics_json = project_root / "outputs" / "metrics" / f"{experiment_id}.json"
    if not metrics_json.exists():
        metrics_json = None

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
        fallback = project_root / "outputs" / "configs" / f"{experiment_id}.json"
        if fallback.exists():
            config_json = fallback

    report_path = attack_dir / 'report.html'
    return build_report(
        project_root=project_root,
        summary_json=summary_path,
        metrics_json=metrics_json,
        config_json=config_json,
        targets_csv=targets_csv,
        output_path=report_path,
        title=report_title,
    )


# Run one membership inference attack against an anonymized dataset and save attack results.


# Build a unique name for one MIA run.
def make_attack_id(
    anonymized_path: Path,
    known_qids: list[str],
    n_targets: int,
    seed: int,
    use_privjedai_fuzzy: bool = False,
) -> str:
    payload = (
        f"qids={'-'.join(known_qids)}|n={n_targets}|seed={seed}"
        f"|fuzzy={int(use_privjedai_fuzzy)}"
    )
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]
    return f"{anonymized_path.stem}__mia_{digest}"



# Build the optional privJedAI fuzzy configuration used during stage-2 refinement.
def _maybe_build_fuzzy_config(
    *,
    use_privjedai_fuzzy: bool,
    privjedai_src: str | Path | None,
    privjedai_fuzzy_threshold: float,
    privjedai_fuzzy_metric: str,
    privjedai_bloom_size: int,
    privjedai_bloom_num_hashes: int,
    privjedai_bloom_qgrams: int,
    privjedai_bloom_hashing_type: str,
) -> dict[str, Any] | None:
    if not use_privjedai_fuzzy:
        return None
    return build_privjedai_fuzzy_config(
        privjedai_src=privjedai_src,
        threshold=privjedai_fuzzy_threshold,
        metric=privjedai_fuzzy_metric,
        bloom_size=privjedai_bloom_size,
        bloom_num_hashes=privjedai_bloom_num_hashes,
        bloom_qgrams=privjedai_bloom_qgrams,
        bloom_hashing_type=privjedai_bloom_hashing_type,
    )

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


# Build an inverted index of row positions for each visible anonymized value.
def build_row_index_by_qid_value(
    working_df_str: pd.DataFrame,
    known_qids: list[str],
    op_counter: dict[str, int] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    row_index: dict[str, dict[str, np.ndarray]] = {}
    for qid in known_qids:
        if op_counter is not None:
            op_counter["row_index_row_visits"] += int(len(working_df_str))
        positions: dict[str, list[int]] = {}
        for idx, value in enumerate(working_df_str[qid].astype(str).tolist()):
            positions.setdefault(str(value).strip(), []).append(idx)
        row_index[qid] = {
            value: np.asarray(sorted(indices), dtype=np.int32)
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
    op_counter: dict[str, int] | None = None,
) -> list[str]:
    cache_key = (qid, str(target_value).strip())
    if cache_key in compatible_values_cache:
        if op_counter is not None:
            op_counter["compatible_value_cache_hits"] += 1
        return compatible_values_cache[cache_key]

    if op_counter is not None:
        op_counter["compatible_value_cache_misses"] += 1
        op_counter["compatible_value_tests"] += int(len(row_index_by_qid_value[qid]))

    attacker_attr_knowledge = attacker_knowledge.get(qid)
    compatible_values = [
        anonymized_value
        for anonymized_value in row_index_by_qid_value[qid].keys()
        if attribute_score(target_value, anonymized_value, attacker_attr_knowledge) > EPS
    ]
    compatible_values_cache[cache_key] = compatible_values
    return compatible_values


# Prefilter the anonymized dataset to rows that do not contradict any stage-1 known target attribute.
def prefilter_candidate_indices_for_target(
    *,
    qid_filter_attrs: list[str],
    qid_known_values: dict[str, str],
    attacker_knowledge: dict[str, dict[str, Any]],
    row_index_by_qid_value: dict[str, dict[str, np.ndarray]],
    compatible_values_cache: dict[tuple[str, str], list[str]],
    n_rows: int,
    op_counter: dict[str, int] | None = None,
) -> np.ndarray:
    if not qid_filter_attrs:
        return np.arange(n_rows, dtype=np.int32)

    candidate_indices: np.ndarray | None = None

    for qid in qid_filter_attrs:
        compatible_values = compatible_anonymized_values_for_target_value(
            qid=qid,
            target_value=qid_known_values[qid],
            attacker_knowledge=attacker_knowledge,
            row_index_by_qid_value=row_index_by_qid_value,
            compatible_values_cache=compatible_values_cache,
            op_counter=op_counter,
        )
        if not compatible_values:
            return np.asarray([], dtype=np.int32)

        if op_counter is not None:
            op_counter["candidate_row_refs_loaded"] += int(
                sum(len(row_index_by_qid_value[qid][value]) for value in compatible_values)
            )

        qid_candidate_indices = np.unique(
            np.concatenate([row_index_by_qid_value[qid][value] for value in compatible_values])
        ).astype(np.int32, copy=False)
        if candidate_indices is None:
            candidate_indices = qid_candidate_indices
        else:
            if op_counter is not None:
                op_counter["candidate_intersections"] += 1
                op_counter["candidate_intersection_input_total"] += int(
                    len(candidate_indices) + len(qid_candidate_indices)
                )
            candidate_indices = np.intersect1d(candidate_indices, qid_candidate_indices, assume_unique=False).astype(np.int32, copy=False)

        if len(candidate_indices) == 0:
            return np.asarray([], dtype=np.int32)

    return candidate_indices if candidate_indices is not None else np.asarray([], dtype=np.int32)


# Infer the known QIs from the targets dataset when they are not passed explicitly.
def infer_known_qids(df_targets: pd.DataFrame, explicit_known_qids: list[str], target_id_col: str, member_col: str) -> list[str]:
    if explicit_known_qids:
        return explicit_known_qids
    return [col for col in df_targets.columns if col not in {target_id_col, member_col}]


# Split attacker-known attributes into stage 1 (generalized) and stage 2 (clear-text) attributes.
def _split_attack_attributes(
    *,
    known_qids: list[str],
    attacker_knowledge: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    qid_filter_attrs: list[str] = []
    refine_attrs: list[str] = []
    unknown_visibility_attrs: list[str] = []

    for qid in known_qids:
        attr_knowledge = attacker_knowledge.get(qid)
        if attr_knowledge is None or "visible_level" not in attr_knowledge:
            unknown_visibility_attrs.append(qid)
            continue

        visible_level = int(attr_knowledge.get("visible_level", 0))
        if visible_level == 0:
            refine_attrs.append(qid)
        else:
            qid_filter_attrs.append(qid)

    return qid_filter_attrs, refine_attrs, unknown_visibility_attrs


# Project one raw stage-1 value to the attacker-visible generalized value.
def _project_qid_filter_value(
    *,
    qid: str,
    raw_value: str,
    attacker_knowledge: dict[str, dict[str, Any]],
) -> str:
    attr_knowledge = attacker_knowledge.get(qid, {})
    projection = attr_knowledge.get("projection", {})
    return str(projection.get(str(raw_value).strip(), str(raw_value).strip())).strip()


# Build the cache key for stage-1 equivalence classes.
def _make_qid_stage1_cache_key(
    *,
    known_values: dict[str, str],
    qid_filter_attrs: list[str],
    attacker_knowledge: dict[str, dict[str, Any]],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            qid,
            _project_qid_filter_value(
                qid=qid,
                raw_value=known_values[qid],
                attacker_knowledge=attacker_knowledge,
            ),
        )
        for qid in qid_filter_attrs
    )


# Build one stage-1 cache entry for the generalized-QI equivalence class.
def _build_qid_stage1_cache_entry(
    *,
    qid_known_values: dict[str, str],
    qid_filter_attrs: list[str],
    attacker_knowledge: dict[str, dict[str, Any]],
    row_index_by_qid_value: dict[str, dict[str, np.ndarray]],
    compatible_values_cache: dict[tuple[str, str], list[str]],
    n_rows: int,
    op_counter: dict[str, int] | None = None,
) -> dict[str, Any]:
    qid_kept_indices = prefilter_candidate_indices_for_target(
        qid_filter_attrs=qid_filter_attrs,
        qid_known_values=qid_known_values,
        attacker_knowledge=attacker_knowledge,
        row_index_by_qid_value=row_index_by_qid_value,
        compatible_values_cache=compatible_values_cache,
        n_rows=n_rows,
        op_counter=op_counter,
    )
    return {
        "qid_kept_indices": qid_kept_indices,
    }


# Apply stage-2 reduction on clear-text attributes with exact matching and optional privJedAI fuzzy fallback.
def refine_candidate_indices_for_target(
    *,
    candidate_indices: np.ndarray,
    refine_attrs: list[str],
    refine_known_values: dict[str, str],
    working_df_str: pd.DataFrame,
    fuzzy_config: dict[str, Any] | None,
    fuzzy_pair_cache: dict[tuple[str, str], float],
    fuzzy_hash_cache: dict[str, frozenset[int]],
    use_privjedai_fuzzy: bool,
    op_counter: dict[str, int] | None = None,
) -> dict[str, Any]:
    if len(candidate_indices) == 0:
        return {
            "refined_candidate_indices": np.asarray([], dtype=np.int32),
            "refine_exact_match_count_kept": np.zeros(0, dtype=np.int16),
            "refine_privjedai_fuzzy_match_count_kept": np.zeros(0, dtype=np.int16),
            "refine_privjedai_fuzzy_score_sum_kept": np.zeros(0, dtype=np.float32),
            "refine_privjedai_fuzzy_score_mean_kept": np.zeros(0, dtype=np.float32),
        }

    refine_input_count = int(len(candidate_indices))
    refine_exact_count = np.zeros(refine_input_count, dtype=np.int16)
    refine_fuzzy_count = np.zeros(refine_input_count, dtype=np.int16)
    refine_fuzzy_score_sum = np.zeros(refine_input_count, dtype=np.float32)
    refine_full_mask = np.ones(refine_input_count, dtype=bool)

    if not refine_attrs:
        return {
            "refined_candidate_indices": candidate_indices.astype(np.int32, copy=False),
            "refine_exact_match_count_kept": refine_exact_count,
            "refine_privjedai_fuzzy_match_count_kept": refine_fuzzy_count,
            "refine_privjedai_fuzzy_score_sum_kept": refine_fuzzy_score_sum,
            "refine_privjedai_fuzzy_score_mean_kept": np.full(refine_input_count, np.nan, dtype=np.float32),
        }

    refined_df = working_df_str.iloc[candidate_indices].copy()

    for attr in refine_attrs:
        attr_positive_mask = np.zeros(refine_input_count, dtype=bool)
        if op_counter is not None:
            op_counter["refinement_mask_cells"] += refine_input_count

        candidate_values = refined_df[attr].astype(str).str.strip().tolist()
        target_value = str(refine_known_values[attr]).strip()
        for idx, candidate_value in enumerate(candidate_values):
            if op_counter is not None:
                op_counter["refinement_candidate_row_visits"] += 1
                op_counter["refinement_exact_tests"] += 1
                if fuzzy_config is not None and target_value != candidate_value and not is_suppressed_value(candidate_value):
                    op_counter["refinement_fuzzy_tests"] += 1

            result = refinement_match_result(
                target_value,
                candidate_value,
                fuzzy_config=fuzzy_config,
                fuzzy_pair_cache=fuzzy_pair_cache,
                fuzzy_hash_cache=fuzzy_hash_cache,
            )
            if result is None:
                continue

            attr_positive_mask[idx] = True
            kind = str(result["kind"])
            score = result.get("score")
            if kind == "exact":
                refine_exact_count[idx] += 1
            elif kind == "privjedai_fuzzy":
                refine_fuzzy_count[idx] += 1
                if score is not None:
                    refine_fuzzy_score_sum[idx] += float(score)

        refine_full_mask &= attr_positive_mask
        if not bool(np.any(refine_full_mask)):
            return {
                "refined_candidate_indices": np.asarray([], dtype=np.int32),
                "refine_exact_match_count_kept": np.zeros(0, dtype=np.int16),
                "refine_privjedai_fuzzy_match_count_kept": np.zeros(0, dtype=np.int16),
                "refine_privjedai_fuzzy_score_sum_kept": np.zeros(0, dtype=np.float32),
                "refine_privjedai_fuzzy_score_mean_kept": np.zeros(0, dtype=np.float32),
            }

    kept_indices = candidate_indices[refine_full_mask].astype(np.int32, copy=False)
    kept_exact_count = refine_exact_count[refine_full_mask].astype(np.int16, copy=False)
    kept_fuzzy_count = refine_fuzzy_count[refine_full_mask].astype(np.int16, copy=False)
    kept_fuzzy_score_sum = refine_fuzzy_score_sum[refine_full_mask].astype(np.float32, copy=False)
    kept_fuzzy_score_mean = np.full(int(np.sum(refine_full_mask)), np.nan, dtype=np.float32)
    np.divide(
        kept_fuzzy_score_sum,
        kept_fuzzy_count,
        out=kept_fuzzy_score_mean,
        where=kept_fuzzy_count > 0,
    )

    return {
        "refined_candidate_indices": kept_indices,
        "refine_exact_match_count_kept": kept_exact_count,
        "refine_privjedai_fuzzy_match_count_kept": kept_fuzzy_count,
        "refine_privjedai_fuzzy_score_sum_kept": kept_fuzzy_score_sum,
        "refine_privjedai_fuzzy_score_mean_kept": kept_fuzzy_score_mean,
    }


# Predict membership from the reduced candidate-set criteria.
def decide_membership(
    *,
    compatible_candidate_count: int,
    total_rows: int,
    max_compatible_fraction: float | None,
) -> tuple[bool, str]:
    has_candidate = compatible_candidate_count > 0
    fraction = (compatible_candidate_count / total_rows) if total_rows else 0.0
    fraction_ok = max_compatible_fraction is None or fraction <= max_compatible_fraction

    predicted_member = has_candidate and fraction_ok
    reason = (
        f"has_candidate={has_candidate};"
        f"compatible_candidate_count={compatible_candidate_count};"
        f"compatible_fraction={fraction:.6f};fraction_ok={fraction_ok}"
    )
    return predicted_member, reason


# Validate all attack inputs before running the MIA.
def _validate_inputs(
    *,
    known_qids: list[str],
    target_id_col: str,
    member_col: str,
    df_targets: pd.DataFrame,
    df_public: pd.DataFrame,
    df_eval: pd.DataFrame,
) -> None:
    if not known_qids:
        raise ValueError("known_qids cannot be empty.")

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
    max_compatible_fraction: float | None = 0.09,
    output_root: str | Path = "outputs",
    name: str | None = None,
    config_path: str | Path | None = None,
    targets_path: str | Path | None = None,
    anonymized_path: str | Path | None = None,
    anonymized_eval_path: str | Path | None = None,
    seed: int = 42,
    use_privjedai_fuzzy: bool = False,
    privjedai_src: str | Path | None = None,
    privjedai_fuzzy_threshold: float = 0.9,
    privjedai_fuzzy_metric: str = "dice",
    privjedai_bloom_size: int = 1024,
    privjedai_bloom_num_hashes: int = 15,
    privjedai_bloom_qgrams: int = 4,
    privjedai_bloom_hashing_type: str = "salted_qgrams",
    generate_report: bool = True,
    report_title: str | None = None,
    obfuscate_known_qids: str | None = None,
    schema_matcher_name: str = "jaccard",
    schema_matcher_min_score: float = 0.0,
) -> dict[str, Any]:
    known_qids = infer_known_qids(df_targets, list(known_qids or []), target_id_col, member_col)

    _validate_inputs(
        known_qids=known_qids,
        target_id_col=target_id_col,
        member_col=member_col,
        df_targets=df_targets,
        df_public=df_public,
        df_eval=df_eval,
    )

    attacker_knowledge = build_attacker_knowledge(runtime=runtime, known_attrs=known_qids, df_public=df_public)

    # ------------------------------------------------------------------ #
    # Optional schema matching step.                                      #
    # Models a realistic attacker who does NOT know the column names of   #
    # the anonymized release: the listed QI columns are renamed col_0,    #
    # col_1, ... in df_public, then a schema matcher (Valentine-based or  #
    # Jaccard baseline) recovers a mapping back to the attacker's         #
    # vocabulary (df_targets columns) before the rest of the MIA runs.    #
    # QIs whose column name cannot be recovered are dropped from          #
    # known_qids: the attacker has lost the ability to use them.          #
    # ------------------------------------------------------------------ #
    schema_matching_results: dict[str, Any] | None = None
    schema_matching_results_path: Path | None = None
    schema_matching_pairs_path: Path | None = None

    if obfuscate_known_qids:
        from schema_matcher import (
            obfuscate_columns,
            recover_column_mapping,
            jaccard_baseline,
            apply_recovered_mapping,
            evaluate_mapping,
        )

        cols_to_hide = [c.strip() for c in obfuscate_known_qids.split(",") if c.strip()]
        # Sanity check: every column to obfuscate must currently exist in df_public.
        missing_in_public = [c for c in cols_to_hide if c not in df_public.columns]
        if missing_in_public:
            raise ValueError(
                f"--obfuscate-known-qids references columns not present in the anonymized public CSV: {missing_in_public}"
            )

        df_public, truth = obfuscate_columns(df_public, cols_to_hide, prefix="col_")

        # Attacker's knowledge base for matching = df_targets minus the label column
        # and the internal record_id column (neither would appear in a real release).
        kb_candidates = [
            c for c in df_targets.columns
            if c not in (target_id_col, member_col)
        ]

        if schema_matcher_name == "baseline_jaccard":
            recovered = jaccard_baseline(
                df_anon=df_public, df_kb=df_targets,
                anon_unknown_cols=list(truth.keys()),
                kb_candidate_cols=kb_candidates,
                min_score=schema_matcher_min_score,
            )
        else:
            recovered = recover_column_mapping(
                df_anon=df_public, df_kb=df_targets,
                anon_unknown_cols=list(truth.keys()),
                kb_candidate_cols=kb_candidates,
                matcher_name=schema_matcher_name,
                min_score=schema_matcher_min_score,
            )

        metrics_sm = evaluate_mapping(recovered, truth)
        print(f"[schema-matching] {metrics_sm}")

        recovered_kb_cols = {kb for _, (kb, _) in recovered.items()}

        schema_matching_pairs: list[dict[str, Any]] = []
        for anon_col, true_col in truth.items():
            match = recovered.get(anon_col)
            predicted_col = None
            score = None
            if match is not None:
                predicted_col = str(match[0])
                score = None if match[1] is None else float(match[1])
            schema_matching_pairs.append(
                {
                    "anon_column": str(anon_col),
                    "true_column": str(true_col),
                    "predicted_column": predicted_col,
                    "score": score,
                    "is_mapped": predicted_col is not None,
                    "is_correct": (predicted_col == true_col) if predicted_col is not None else False,
                    "matcher": schema_matcher_name,
                    "min_score": float(schema_matcher_min_score),
                }
            )

        schema_matching_results = {
            "enabled": True,
            "matcher": schema_matcher_name,
            "min_score": float(schema_matcher_min_score),
            "obfuscated_columns": cols_to_hide,
            "anon_unknown_cols": [str(c) for c in truth.keys()],
            "kb_candidate_cols": [str(c) for c in kb_candidates],
            "truth": {str(k): str(v) for k, v in truth.items()},
            "recovered": {
                str(anon_col): {
                    "predicted_column": str(kb_col),
                    "score": None if score is None else float(score),
                }
                for anon_col, (kb_col, score) in recovered.items()
            },
            "metrics": metrics_sm,
            "pairs": schema_matching_pairs,
        }

        # Rename df_public columns back so the rest of the pipeline works unchanged.
        # Note: df_eval is not renamed because no QI column from df_eval is read by
        # name later in this function (only target_id_col is read from df_eval).
        df_public, _ = apply_recovered_mapping(df_public, recovered)

        # known_qids must drop any obfuscated QI whose name could not be recovered:
        # the attacker can no longer exploit it.
        unrecoverable = {orig for _, orig in truth.items() if orig not in recovered_kb_cols}
        if unrecoverable:
            print(f"[schema-matching] Unrecoverable QIs dropped from known_qids: {sorted(unrecoverable)}")
        known_qids = [a for a in known_qids if a not in unrecoverable]

    qid_filter_attrs, refine_attrs, unknown_visibility_attrs = _split_attack_attributes(
        known_qids=known_qids,
        attacker_knowledge=attacker_knowledge,
    )
    fuzzy_config = _maybe_build_fuzzy_config(
        use_privjedai_fuzzy=bool(use_privjedai_fuzzy and refine_attrs),
        privjedai_src=privjedai_src,
        privjedai_fuzzy_threshold=privjedai_fuzzy_threshold,
        privjedai_fuzzy_metric=privjedai_fuzzy_metric,
        privjedai_bloom_size=privjedai_bloom_size,
        privjedai_bloom_num_hashes=privjedai_bloom_num_hashes,
        privjedai_bloom_qgrams=privjedai_bloom_qgrams,
        privjedai_bloom_hashing_type=privjedai_bloom_hashing_type,
    )
    op_counter = make_operation_counter()

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

    row_index_by_qid_value = build_row_index_by_qid_value(working_df_str, known_qids, op_counter=op_counter)
    compatible_values_cache: dict[tuple[str, str], list[str]] = {}
    qid_stage1_cache: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
    fuzzy_pair_cache: dict[tuple[str, str], float] = {}
    fuzzy_hash_cache: dict[str, frozenset[int]] = {}

    per_target_rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    targets_with_any_fuzzy_candidate: list[bool] = []
    targets_with_true_record_fuzzy_kept: list[bool] = []

    n_rows = len(working_df_str)

    for _, target in targets_df.iterrows():
        op_counter["targets_evaluated"] += 1
        target_id = str(target[target_id_col]).strip()
        truth_member = int(str(target[member_col]).strip())
        known_values = {qi: str(target[qi]).strip() for qi in known_qids}
        qid_known_values = {qi: known_values[qi] for qi in qid_filter_attrs}
        refine_known_values = {qi: known_values[qi] for qi in refine_attrs}

        qid_stage1_key = _make_qid_stage1_cache_key(
            known_values=known_values,
            qid_filter_attrs=qid_filter_attrs,
            attacker_knowledge=attacker_knowledge,
        )
        if qid_stage1_key not in qid_stage1_cache:
            op_counter["qid_stage1_cache_misses"] += 1
            qid_stage1_cache[qid_stage1_key] = _build_qid_stage1_cache_entry(
                qid_known_values=qid_known_values,
                qid_filter_attrs=qid_filter_attrs,
                attacker_knowledge=attacker_knowledge,
                row_index_by_qid_value=row_index_by_qid_value,
                compatible_values_cache=compatible_values_cache,
                n_rows=n_rows,
                op_counter=op_counter,
            )
        else:
            op_counter["qid_stage1_cache_hits"] += 1

        qid_candidate_indices = qid_stage1_cache[qid_stage1_key]["qid_kept_indices"]
        qid_candidate_count = int(len(qid_candidate_indices))

        refinement_result = refine_candidate_indices_for_target(
            candidate_indices=qid_candidate_indices,
            refine_attrs=refine_attrs,
            refine_known_values=refine_known_values,
            working_df_str=working_df_str,
            fuzzy_config=fuzzy_config,
            fuzzy_pair_cache=fuzzy_pair_cache,
            fuzzy_hash_cache=fuzzy_hash_cache,
            use_privjedai_fuzzy=use_privjedai_fuzzy,
            op_counter=op_counter,
        )
        refined_candidate_indices = refinement_result["refined_candidate_indices"]
        refined_fuzzy_match_count_kept = refinement_result["refine_privjedai_fuzzy_match_count_kept"]
        compatible_candidate_count = int(len(refined_candidate_indices))
        compatible_candidate_fraction = (compatible_candidate_count / n_rows) if n_rows else 0.0

        target_present_in_anonymized = target_id in id_to_eval_index

        predicted_member, decision_reason = decide_membership(
            compatible_candidate_count=compatible_candidate_count,
            total_rows=n_rows,
            max_compatible_fraction=max_compatible_fraction,
        )
        op_counter["membership_decisions"] += 1

        true_record_in_qid_class = None
        true_record_in_reduced_class = None
        true_record_fuzzy_kept = False
        if target_present_in_anonymized:
            true_idx = id_to_eval_index[target_id]
            true_record_in_qid_class = bool(np.any(qid_candidate_indices == true_idx))
            true_record_in_reduced_class = bool(np.any(refined_candidate_indices == true_idx))
            if use_privjedai_fuzzy and true_record_in_reduced_class:
                true_kept_positions = np.flatnonzero(refined_candidate_indices == true_idx)
                true_record_fuzzy_kept = bool(
                    len(true_kept_positions) > 0
                    and (refined_fuzzy_match_count_kept[true_kept_positions] > 0).any()
                )

        has_any_fuzzy_candidate = bool(
            use_privjedai_fuzzy
            and compatible_candidate_count > 0
            and (refined_fuzzy_match_count_kept > 0).any()
        )
        targets_with_any_fuzzy_candidate.append(has_any_fuzzy_candidate)
        targets_with_true_record_fuzzy_kept.append(bool(true_record_fuzzy_kept))

        stage1_reduction = qid_candidate_count - compatible_candidate_count
        stage1_reduction_rate = None if qid_candidate_count == 0 else float(stage1_reduction) / float(qid_candidate_count)

        row = {
            "target_id": target_id,
            "ground_truth_member": truth_member,
            "predicted_member": int(predicted_member),
            "known_qids": "|".join(known_qids),
            "qid_filter_qids": "|".join(qid_filter_attrs),
            "refine_qids": "|".join(refine_attrs),
            "stage1_equivalence_class_size": qid_candidate_count,
            "compatible_candidate_count": compatible_candidate_count,
            "reduced_equivalence_class_size": compatible_candidate_count,
            "equivalence_class_reduction": stage1_reduction,
            "equivalence_class_reduction_rate": None if stage1_reduction_rate is None else round(stage1_reduction_rate, 6),
            "compatible_candidate_fraction": compatible_candidate_fraction,
            "target_present_in_anonymized": target_present_in_anonymized,
            "true_record_in_stage1_class": true_record_in_qid_class,
            "true_record_in_reduced_class": true_record_in_reduced_class,
            "decision_reason": decision_reason,
        }
        if use_privjedai_fuzzy:
            row["reduced_candidates_with_privjedai_fuzzy"] = int((refined_fuzzy_match_count_kept > 0).sum())
            row["true_record_used_privjedai_fuzzy"] = bool(true_record_fuzzy_kept)

        for qi in known_qids:
            row[f"known_{qi}"] = known_values[qi]
        per_target_rows.append(row)
        y_true.append(truth_member)
        y_pred.append(int(predicted_member))

    metrics = compute_classification_metrics(y_true, y_pred)

    targets_results_df = pd.DataFrame(per_target_rows)
    member_mask = targets_results_df["ground_truth_member"].astype(int) == 1
    non_member_mask = ~member_mask

    candidate_prefilter_mode = (
        "stage1_generalized_qids_then_stage2_cleartext_exact_or_privjedai_fuzzy_refinement"
        if use_privjedai_fuzzy
        else "stage1_generalized_qids_then_stage2_cleartext_refinement"
    )

    summary = {
        "attack_id": name or make_attack_id(Path(anonymized_path or "anonymized.csv"), known_qids, len(df_targets), seed, use_privjedai_fuzzy=use_privjedai_fuzzy),
        "config_path": str(config_path) if config_path else None,
        "targets_path": str(targets_path) if targets_path else None,
        "anonymized_path": str(anonymized_path) if anonymized_path else None,
        "anonymized_eval_path": str(anonymized_eval_path) if anonymized_eval_path else None,
        "known_qids": known_qids,
        "qid_filter_qids": qid_filter_attrs,
        "refine_qids": refine_attrs,
        "unknown_visibility_qids": unknown_visibility_attrs,
        "target_id_col": target_id_col,
        "member_col": member_col,
        "seed": seed,
        "n_targets": len(df_targets),
        "n_members": int(member_mask.sum()),
        "n_non_members": int(non_member_mask.sum()),
        "max_compatible_fraction": max_compatible_fraction,
        "attacker_knowledge": attacker_knowledge,
        "candidate_prefilter_mode": candidate_prefilter_mode,
        "use_privjedai_fuzzy": bool(use_privjedai_fuzzy),
        "schema_matching_enabled": schema_matching_results is not None,
        "schema_matcher_name": schema_matcher_name if schema_matching_results is not None else None,
        "schema_matcher_min_score": float(schema_matcher_min_score) if schema_matching_results is not None else None,
        "schema_matching_metrics": (schema_matching_results.get("metrics") if schema_matching_results is not None else None),
        **({
            "privjedai_fuzzy_threshold": float(fuzzy_config["threshold"]),
            "privjedai_fuzzy_metric": str(fuzzy_config["metric"]),
            "privjedai_bloom_size": int(fuzzy_config["bloom_size"]),
            "privjedai_bloom_num_hashes": int(fuzzy_config["num_hashes"]),
            "privjedai_bloom_qgrams": int(fuzzy_config["qgrams"]),
            "privjedai_bloom_hashing_type": str(fuzzy_config["hashing_type"]),
            "targets_with_any_privjedai_fuzzy_candidate_rate": round(sum(targets_with_any_fuzzy_candidate) / len(targets_with_any_fuzzy_candidate), 6) if targets_with_any_fuzzy_candidate else 0.0,
            "targets_with_true_record_kept_by_privjedai_fuzzy_rate": round(sum(targets_with_true_record_fuzzy_kept) / len(targets_with_true_record_fuzzy_kept), 6) if targets_with_true_record_fuzzy_kept else 0.0,
        } if use_privjedai_fuzzy and fuzzy_config is not None else {}),
        "operation_counter": build_public_operation_counter(
            op_counter,
            use_privjedai_fuzzy=use_privjedai_fuzzy,
        ),
        **metrics,
        "member_avg_stage1_equivalence_class_size": float(targets_results_df.loc[member_mask, "stage1_equivalence_class_size"].mean()) if member_mask.any() else None,
        "non_member_avg_stage1_equivalence_class_size": float(targets_results_df.loc[non_member_mask, "stage1_equivalence_class_size"].mean()) if non_member_mask.any() else None,
        "member_avg_compatible_candidate_count": float(targets_results_df.loc[member_mask, "compatible_candidate_count"].mean()) if member_mask.any() else None,
        "non_member_avg_compatible_candidate_count": float(targets_results_df.loc[non_member_mask, "compatible_candidate_count"].mean()) if non_member_mask.any() else None,
        "member_avg_equivalence_class_reduction": float(targets_results_df.loc[member_mask, "equivalence_class_reduction"].mean()) if member_mask.any() else None,
        "non_member_avg_equivalence_class_reduction": float(targets_results_df.loc[non_member_mask, "equivalence_class_reduction"].mean()) if non_member_mask.any() else None,
    }

    output_root = ensure_dir(Path(output_root))
    attacks_root = ensure_dir(output_root / "attacks" / "mia")
    attack_id = str(summary["attack_id"])
    attack_dir = ensure_dir(attacks_root / attack_id.replace(".", "_"))
    summary_path = attack_dir / "summary.json"
    targets_path_out = attack_dir / "targets.csv"

    if schema_matching_results is not None:
        schema_matching_results_path = attack_dir / "schema_matching_results.json"
        schema_matching_pairs_path = attack_dir / "schema_matching_pairs.csv"
        save_json(schema_matching_results_path, schema_matching_results)
        pd.DataFrame(schema_matching_results["pairs"]).to_csv(schema_matching_pairs_path, index=False)
        summary["schema_matching_results_json"] = str(schema_matching_results_path)
        summary["schema_matching_pairs_csv"] = str(schema_matching_pairs_path)
    else:
        summary["schema_matching_results_json"] = None
        summary["schema_matching_pairs_csv"] = None

    save_json(summary_path, summary)
    targets_results_df.to_csv(targets_path_out, index=False)

    report_path: Path | None = None
    if generate_report:
        try:
            report_path = _maybe_generate_mia_report(
                output_root=output_root,
                summary_path=summary_path,
                attack_id=attack_id,
                report_title=report_title,
            )
        except Exception as exc:
            print(f"[WARN] MIA HTML report generation failed: {exc}")

    summary_row = {
        "attack_id": attack_id,
        "config_path": summary.get("config_path"),
        "targets_path": summary.get("targets_path"),
        "anonymized_path": summary.get("anonymized_path"),
        "anonymized_eval_path": summary.get("anonymized_eval_path"),
        "known_qids": "|".join(known_qids),
        "qid_filter_qids": "|".join(qid_filter_attrs),
        "refine_qids": "|".join(refine_attrs),
        "n_targets": summary["n_targets"],
        "n_members": summary["n_members"],
        "n_non_members": summary["n_non_members"],
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
        "report_html": str(report_path) if report_path is not None else "",
        "candidate_prefilter_mode": candidate_prefilter_mode,
        "use_privjedai_fuzzy": bool(use_privjedai_fuzzy),
        "schema_matching_enabled": schema_matching_results is not None,
        "schema_matcher_name": schema_matcher_name if schema_matching_results is not None else "",
        "schema_matching_results_json": str(schema_matching_results_path) if schema_matching_results_path is not None else "",
        "schema_matching_pairs_csv": str(schema_matching_pairs_path) if schema_matching_pairs_path is not None else "",
        **({
            "privjedai_fuzzy_threshold": summary["privjedai_fuzzy_threshold"],
            "privjedai_fuzzy_metric": summary["privjedai_fuzzy_metric"],
            "targets_with_any_privjedai_fuzzy_candidate_rate": summary["targets_with_any_privjedai_fuzzy_candidate_rate"],
            "targets_with_true_record_kept_by_privjedai_fuzzy_rate": summary["targets_with_true_record_kept_by_privjedai_fuzzy_rate"],
        } if use_privjedai_fuzzy and fuzzy_config is not None else {}),
    }
    append_attack_summary(attacks_root / "mia_summary.csv", summary_row)

    print(f"[OK] {attack_id}")
    print(f"Summary   : {summary_path}")
    if report_path is not None:
        print(f"HTML report : {report_path}")
    print(f"Targets   : {targets_path_out}")
    print(f"Logical ops (est.) : {summary['operation_counter']['estimated_total_operations']}")
    if use_privjedai_fuzzy and fuzzy_config is not None:
        print(f"Targets with fuzzy cand.  : {summary['targets_with_any_privjedai_fuzzy_candidate_rate']}")
        print(f"True records rescued fuzz.: {summary['targets_with_true_record_kept_by_privjedai_fuzzy_rate']}")

    return {
        "summary": summary,
        "summary_path": summary_path,
        "targets_path": targets_path_out,
        "report_path": report_path,
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
    max_compatible_fraction: float | None = 0.09,
    output_root: str | Path = "outputs",
    name: str | None = None,
    seed: int = 42,
    use_privjedai_fuzzy: bool = False,
    privjedai_src: str | Path | None = None,
    privjedai_fuzzy_threshold: float = 0.9,
    privjedai_fuzzy_metric: str = "dice",
    privjedai_bloom_size: int = 1024,
    privjedai_bloom_num_hashes: int = 15,
    privjedai_bloom_qgrams: int = 4,
    privjedai_bloom_hashing_type: str = "salted_qgrams",
    generate_report: bool = True,
    report_title: str | None = None,
    obfuscate_known_qids: str | None = None,
    schema_matcher_name: str = "jaccard",
    schema_matcher_min_score: float = 0.0,
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
        max_compatible_fraction=max_compatible_fraction,
        output_root=output_root,
        name=name,
        config_path=config_path,
        targets_path=targets_path,
        anonymized_path=anonymized_path,
        anonymized_eval_path=anonymized_eval_path,
        seed=seed,
        use_privjedai_fuzzy=use_privjedai_fuzzy,
        privjedai_src=privjedai_src,
        privjedai_fuzzy_threshold=privjedai_fuzzy_threshold,
        privjedai_fuzzy_metric=privjedai_fuzzy_metric,
        privjedai_bloom_size=privjedai_bloom_size,
        privjedai_bloom_num_hashes=privjedai_bloom_num_hashes,
        privjedai_bloom_qgrams=privjedai_bloom_qgrams,
        privjedai_bloom_hashing_type=privjedai_bloom_hashing_type,
        generate_report=generate_report,
        report_title=report_title,
        obfuscate_known_qids=obfuscate_known_qids,
        schema_matcher_name=schema_matcher_name,
        schema_matcher_min_score=schema_matcher_min_score,
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
    parser.add_argument("--max-compatible-fraction", type=float, default=0.09, help="Maximum fraction of fully compatible candidates allowed to predict member.")
    parser.add_argument(
        "--use-privjedai-fuzzy",
        action="store_true",
        help="Enable privJedAI Bloom-filter fuzzy matching during stage-2 clear-text refinement.",
    )
    parser.add_argument("--privjedai-src", default=None, help="Path to privJedAI-main/src if privjedai is not installed.")
    parser.add_argument("--privjedai-fuzzy-threshold", type=float, default=0.9)
    parser.add_argument("--privjedai-fuzzy-metric", choices=["dice", "jaccard", "cosine", "scm"], default="dice")
    parser.add_argument("--privjedai-bloom-size", type=int, default=1024)
    parser.add_argument("--privjedai-bloom-num-hashes", type=int, default=15)
    parser.add_argument("--privjedai-bloom-qgrams", type=int, default=4)
    parser.add_argument(
        "--privjedai-bloom-hashing-type",
        choices=["dh", "rh", "edh", "th", "random", "doubleHash", "enhancedDoubleHash", "tripleHash", "salted_qgrams"],
        default="salted_qgrams",
    )
    parser.add_argument("--output-root", default="outputs", help="Root directory for attack outputs.")
    parser.add_argument("--name", default=None, help="Optional attack name override.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used only for naming consistency.")
    parser.add_argument(
        "--no-generate-report",
        action="store_true",
        help="Do not auto-generate the HTML MIA report at the end of the attack.",
    )
    parser.add_argument(
        "--report-title",
        default=None,
        help="Optional custom title for the generated HTML report.",
    )
    parser.add_argument(
        "--obfuscate-known-qids",
        default=None,
        help=(
            "Comma-separated list of known QI columns to treat as unknown-named "
            "in the public anonymized CSV. The schema matcher will recover their "
            "names against the targets CSV before the MIA runs. QIs whose name "
            "cannot be recovered are dropped from the attacker's known_qids."
        ),
    )
    parser.add_argument(
        "--schema-matcher",
        choices=["coma", "jaccard", "distribution", "baseline_jaccard"],
        default="jaccard",
        help="Schema matcher used to recover obfuscated column names.",
    )
    parser.add_argument(
        "--schema-matcher-min-score",
        type=float,
        default=0.0,
        help="Minimum similarity score for a recovered (anon_col -> kb_col) match to be kept.",
    )
    args = parser.parse_args()

    run_mia_attack_from_paths(
        config_path=args.config,
        targets_path=args.targets,
        anonymized_path=args.anonymized,
        anonymized_eval_path=args.anonymized_eval,
        known_qids=parse_csv_list(args.known_qids),
        target_id_col=args.target_id_col,
        member_col=args.member_col,
        max_compatible_fraction=args.max_compatible_fraction,
        output_root=args.output_root,
        name=args.name,
        seed=args.seed,
        use_privjedai_fuzzy=args.use_privjedai_fuzzy,
        privjedai_src=args.privjedai_src,
        privjedai_fuzzy_threshold=args.privjedai_fuzzy_threshold,
        privjedai_fuzzy_metric=args.privjedai_fuzzy_metric,
        privjedai_bloom_size=args.privjedai_bloom_size,
        privjedai_bloom_num_hashes=args.privjedai_bloom_num_hashes,
        privjedai_bloom_qgrams=args.privjedai_bloom_qgrams,
        privjedai_bloom_hashing_type=args.privjedai_bloom_hashing_type,
        generate_report=not args.no_generate_report,
        report_title=args.report_title,
        obfuscate_known_qids=args.obfuscate_known_qids,
        schema_matcher_name=args.schema_matcher,
        schema_matcher_min_score=args.schema_matcher_min_score,
    )


if __name__ == "__main__":
    main()
