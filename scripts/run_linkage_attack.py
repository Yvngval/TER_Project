# Run one strict linkage attack against an anonymized dataset.
# In this strict setting, every target is guaranteed to be published in the
# anonymized release. The script therefore treats target presence and true-record
# retention as internal consistency checks rather than output metrics.

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
from linkage_helpers import (
    build_value_indices,
    get_match_mapping_for_target_value,
    refinement_match_result,
    summarize_sensitive_prediction,
)
from privjedai_utils import build_privjedai_fuzzy_config
from generate_linkage_attack_report import build_report


def make_operation_counter() -> dict[str, int]:
    return {
        "value_index_row_visits": 0,
        "match_cache_hits": 0,
        "match_cache_misses": 0,
        "compatible_value_tests": 0,
        "targets_evaluated": 0,
        "qid_stage1_cache_hits": 0,
        "qid_stage1_cache_misses": 0,
        "array_cells_initialized": 0,
        "attribute_positive_mask_cells": 0,
        "matching_row_visits": 0,
        "mask_and_cells": 0,
        "final_mask_reads": 0,
        "equivalence_class_candidate_rows_output": 0,
        "refinement_candidate_row_visits": 0,
        "refinement_exact_tests": 0,
        "refinement_fuzzy_tests": 0,
        "refinement_mask_cells": 0,
        "reduced_equivalence_class_candidate_rows_output": 0,
    }


def estimate_total_operations(op_counter: dict[str, int]) -> int:
    return int(
        op_counter["value_index_row_visits"]
        + op_counter["compatible_value_tests"]
        + op_counter["array_cells_initialized"]
        + op_counter["attribute_positive_mask_cells"]
        + op_counter["matching_row_visits"]
        + op_counter["mask_and_cells"]
        + op_counter["final_mask_reads"]
        + op_counter["equivalence_class_candidate_rows_output"]
        + op_counter["refinement_candidate_row_visits"]
        + op_counter["refinement_exact_tests"]
        + op_counter["refinement_fuzzy_tests"]
        + op_counter["refinement_mask_cells"]
        + op_counter["reduced_equivalence_class_candidate_rows_output"]
    )


HIDDEN_OPERATION_COUNTER_KEYS = {
    "match_cache_hits",
    "match_cache_misses",
    "compatible_value_tests",
    "qid_stage1_cache_hits",
    "qid_stage1_cache_misses",
    "operation_model",
}


def build_public_operation_counter(op_counter: dict[str, int]) -> dict[str, int]:
    public_counter = {
        key: value
        for key, value in op_counter.items()
        if key not in HIDDEN_OPERATION_COUNTER_KEYS
    }
    public_counter["estimated_total_operations"] = estimate_total_operations(op_counter)
    return public_counter


def _maybe_generate_linkage_report(
    *,
    output_root: Path,
    summary_path: Path,
    attack_id: str,
    report_title: str | None = None,
) -> Path | None:
    """Generate the HTML linkage report for the current attack if possible."""
    project_root = output_root.parent if output_root.name == 'outputs' else output_root
    attack_dir = summary_path.parent
    targets_csv = attack_dir / 'targets.csv'

    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    experiment_id = str(summary.get('attack_id', attack_id)).split('__known_')[0]

    metrics_json = project_root / 'outputs' / 'metrics' / f'{experiment_id}.json'
    if not metrics_json.exists():
        metrics_json = None

    config_json = None
    config_path_in_summary = summary.get('config_path')
    if config_path_in_summary:
        candidate = Path(str(config_path_in_summary))
        if candidate.exists():
            config_json = candidate
        else:
            basename_candidate = project_root / 'outputs' / 'configs' / candidate.name
            if basename_candidate.exists():
                config_json = basename_candidate

    if config_json is None:
        fallback = project_root / 'outputs' / 'configs' / f'{experiment_id}.json'
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

# Parse --n-targets that can be either an integer or 'all'.
def parse_n_targets_arg(raw: str) -> int | str:
    value = str(raw).strip()
    if value.lower() == "all":
        return "all"

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--n-targets must be a positive integer or 'all'.") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("--n-targets must be a positive integer or 'all'.")
    return parsed


# Resolve the actual number of targets from the auxiliary dataset size.
def resolve_n_targets(n_targets: int | str, df_aux: pd.DataFrame) -> int:
    if isinstance(n_targets, str):
        if n_targets.strip().lower() == "all":
            return int(len(df_aux))
        raise ValueError("n_targets must be a positive integer or 'all'.")
    return int(n_targets)


# Build a unique name for one linkage attack run.
def make_attack_id(
    anonymized_path: Path,
    known_attrs: list[str],
    n_targets: int | str,
    seed: int,
    use_privjedai_fuzzy: bool = False,
) -> str:
    payload = (
        f"known={'-'.join(known_attrs)}|n={n_targets}|seed={seed}"
        f"|fuzzy={int(use_privjedai_fuzzy)}"
    )
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]
    return f"{anonymized_path.stem}__known_{digest}"

# Validate all attack inputs before running the linkage attack.
def _validate_inputs(
    *,
    known_attrs: list[str],
    target_id_col: str,
    sensitive_attr: str,
    df_aux: pd.DataFrame,
    df_public: pd.DataFrame,
    df_eval: pd.DataFrame,
    n_targets: int,
) -> None:
    if not known_attrs:
        raise ValueError("--known-attrs cannot be empty.")

    if target_id_col not in df_aux.columns:
        raise ValueError(f"Auxiliary dataset must contain '{target_id_col}'.")
    if target_id_col not in df_eval.columns:
        raise ValueError(
            f"Evaluation anonymized dataset must contain '{target_id_col}'. "
            "Save an internal eval copy with record_id kept."
        )

    missing_aux_cols = [attr for attr in known_attrs if attr not in df_aux.columns]
    if missing_aux_cols:
        raise ValueError(f"Auxiliary dataset misses known attacker attributes: {missing_aux_cols}")

    missing_public_cols = [attr for attr in known_attrs if attr not in df_public.columns]
    if missing_public_cols:
        raise ValueError(
            "The public anonymized dataset does not contain these attacker-known attributes: "
            f"{missing_public_cols}"
        )

    if sensitive_attr in known_attrs:
        raise ValueError(f"The sensitive attribute '{sensitive_attr}' cannot be part of --known-attrs.")

    if sensitive_attr not in df_eval.columns:
        raise ValueError(f"Sensitive attribute '{sensitive_attr}' not found in anonymized eval dataset.")

    if len(df_public) != len(df_eval):
        raise ValueError("Public and eval anonymized CSV files must have the same number of rows and the same row order.")

    if n_targets <= 0:
        raise ValueError("--n-targets must be > 0.")
    if n_targets > len(df_aux):
        raise ValueError(f"--n-targets ({n_targets}) is larger than the auxiliary dataset size ({len(df_aux)}).")


# Build the optional privJedAI fuzzy configuration.
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


# Ensure all sampled targets are truly part of the anonymized release.
def _validate_strict_target_set(
    *,
    sampled_targets: pd.DataFrame,
    target_id_col: str,
    id_to_eval_index: dict[str, int],
) -> None:
    sampled_ids = sampled_targets[target_id_col].astype(str).str.strip().tolist()
    missing_ids = [target_id for target_id in sampled_ids if target_id not in id_to_eval_index]
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        raise ValueError(
            "Strict linkage expects every sampled target to be present in anonymized_eval, "
            f"but {len(missing_ids)} target(s) were missing. Example ids: {preview}"
        )


def _column_is_fully_suppressed(df: pd.DataFrame, attr: str) -> bool:
    if attr not in df.columns:
        return False
    values = df[attr].astype(str).str.strip()
    if len(values) == 0:
        return False
    return bool(values.map(is_suppressed_value).all())


# Normalize attacker-visible levels from the actual published column values.
# A column that is entirely suppressed ("*", "**", empty string, ...) should
# not be treated as visible_level == 0, because it is not visible in clear text.
def _normalize_attacker_visible_levels(
    *,
    attacker_knowledge: dict[str, dict[str, Any]],
    known_attrs: list[str],
    df_public: pd.DataFrame,
) -> None:
    for attr in known_attrs:
        attr_knowledge = attacker_knowledge.get(attr)
        if attr_knowledge is None or "visible_level" not in attr_knowledge:
            continue

        raw_visible_level = int(attr_knowledge.get("visible_level", 0))
        attr_knowledge["raw_visible_level"] = raw_visible_level

        if raw_visible_level == 0 and _column_is_fully_suppressed(df_public, attr):
            attr_knowledge["visible_level"] = 1
            attr_knowledge["visible_level_override_reason"] = "fully_suppressed_column"


def _split_attack_attributes(
    *,
    runtime: dict[str, Any],
    known_attrs: list[str],
    attacker_knowledge: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    # Stage split is attacker-view based, not config-based.
    # Stage 1: attributes that appear generalized/suppressed in the release
    #          (visible_level != 0) and therefore should be used to build the
    #          initial equivalence class.
    # Stage 2: attributes that remain visible in clear text (visible_level == 0)
    #          and can refine the stage-1 class with exact or fuzzy matching.
    filter_attrs: list[str] = []
    refine_attrs: list[str] = []
    unknown_visibility_attrs: list[str] = []

    for attr in known_attrs:
        attr_knowledge = attacker_knowledge.get(attr)
        if attr_knowledge is None or "visible_level" not in attr_knowledge:
            unknown_visibility_attrs.append(attr)
            continue

        visible_level = int(attr_knowledge.get("visible_level", 0))
        if visible_level == 0:
            refine_attrs.append(attr)
        else:
            filter_attrs.append(attr)

    return filter_attrs, refine_attrs, unknown_visibility_attrs


def _project_qid_filter_value(
    *,
    attr: str,
    raw_value: str,
    attacker_knowledge: dict[str, dict[str, Any]],
) -> str:
    attr_knowledge = attacker_knowledge.get(attr, {})
    projection = attr_knowledge.get("projection", {})
    return str(projection.get(str(raw_value).strip(), str(raw_value).strip())).strip()


def _make_qid_stage1_cache_key(
    *,
    known_values: dict[str, str],
    qid_filter_attrs: list[str],
    attacker_knowledge: dict[str, dict[str, Any]],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            attr,
            _project_qid_filter_value(
                attr=attr,
                raw_value=known_values[attr],
                attacker_knowledge=attacker_knowledge,
            ),
        )
        for attr in qid_filter_attrs
    )


def _build_qid_stage1_cache_entry(
    *,
    qid_known_values: dict[str, str],
    qid_filter_attrs: list[str],
    attacker_knowledge: dict[str, dict[str, Any]],
    working_df_str: pd.DataFrame,
    value_indices: dict[str, dict[str, np.ndarray]],
    attr_unique_values: dict[str, list[str]],
    match_cache: dict[tuple[str, str, str], dict[str, dict[str, Any]]],
    fuzzy_pair_cache: dict[tuple[str, str], float],
    fuzzy_hash_cache: dict[str, frozenset[int]],
    op_counter: dict[str, int] | None = None,
) -> dict[str, Any]:
    n_rows = len(working_df_str)
    qid_compatible_count = np.zeros(n_rows, dtype=np.int16)
    qid_exact_count = np.zeros(n_rows, dtype=np.int16)
    qid_generalized_count = np.zeros(n_rows, dtype=np.int16)
    qid_suppressed_count = np.zeros(n_rows, dtype=np.int16)
    qid_full_compatible_mask = np.ones(n_rows, dtype=bool)
    if op_counter is not None:
        op_counter["array_cells_initialized"] += int(4 * n_rows)

    qid_compatible_values_debug: dict[str, dict[str, dict[str, Any]]] = {}
    for attr in qid_filter_attrs:
        mapping = get_match_mapping_for_target_value(
            attr=attr,
            target_value=qid_known_values[attr],
            possible_anonymized_values=attr_unique_values[attr],
            attacker_knowledge=attacker_knowledge,
            match_cache=match_cache,
            fuzzy_config=None,
            fuzzy_pair_cache=fuzzy_pair_cache,
            fuzzy_hash_cache=fuzzy_hash_cache,
            op_counter=op_counter,
        )
        qid_compatible_values_debug[attr] = mapping
        attr_positive_mask = np.zeros(n_rows, dtype=bool)
        if op_counter is not None:
            op_counter["attribute_positive_mask_cells"] += int(n_rows)

        for anonymized_value, result in mapping.items():
            row_ids = value_indices[attr].get(anonymized_value)
            if row_ids is None or len(row_ids) == 0:
                continue

            kind = str(result["kind"])
            if op_counter is not None:
                op_counter["matching_row_visits"] += int(len(row_ids))
            attr_positive_mask[row_ids] = True
            qid_compatible_count[row_ids] += 1

            if kind == "exact":
                qid_exact_count[row_ids] += 1
            elif kind == "generalized":
                qid_generalized_count[row_ids] += 1
            elif kind == "suppressed":
                qid_suppressed_count[row_ids] += 1

        qid_full_compatible_mask &= attr_positive_mask
        if op_counter is not None:
            op_counter["mask_and_cells"] += int(n_rows)

    if op_counter is not None:
        op_counter["final_mask_reads"] += int(n_rows)

    qid_kept_indices = np.flatnonzero(qid_full_compatible_mask).astype(np.int32)
    return {
        "qid_kept_indices": qid_kept_indices,
        "qid_matched_attr_count": qid_compatible_count[qid_kept_indices].astype(np.int16, copy=False),
        "qid_exact_match_count": qid_exact_count[qid_kept_indices].astype(np.int16, copy=False),
        "qid_generalized_match_count": qid_generalized_count[qid_kept_indices].astype(np.int16, copy=False),
        "qid_suppressed_match_count": qid_suppressed_count[qid_kept_indices].astype(np.int16, copy=False),
        "qid_compatible_values_debug": qid_compatible_values_debug,
    }


def _materialize_qid_df_from_cache_entry(
    *,
    cache_entry: dict[str, Any],
    working_df_str: pd.DataFrame,
) -> pd.DataFrame:
    qid_kept_indices = cache_entry["qid_kept_indices"]
    qid_df = working_df_str.iloc[qid_kept_indices].copy()
    qid_df["qid_matched_attr_count"] = cache_entry["qid_matched_attr_count"]
    qid_df["qid_exact_match_count"] = cache_entry["qid_exact_match_count"]
    qid_df["qid_generalized_match_count"] = cache_entry["qid_generalized_match_count"]
    qid_df["qid_suppressed_match_count"] = cache_entry["qid_suppressed_match_count"]
    return qid_df


# Compute one target's equivalence class and all derived reporting fields.
def _evaluate_target(
    *,
    target: pd.Series,
    target_id_col: str,
    sensitive_attr: str,
    known_attrs: list[str],
    qid_filter_attrs: list[str],
    refine_attrs: list[str],
    attacker_knowledge: dict[str, dict[str, Any]],
    working_df_str: pd.DataFrame,
    value_indices: dict[str, dict[str, np.ndarray]],
    attr_unique_values: dict[str, list[str]],
    match_cache: dict[tuple[str, str, str], dict[str, dict[str, Any]]],
    qid_stage1_cache: dict[tuple[tuple[str, str], ...], dict[str, Any]],
    fuzzy_config: dict[str, Any] | None,
    fuzzy_pair_cache: dict[tuple[str, str], float],
    fuzzy_hash_cache: dict[str, frozenset[int]],
    save_prefilter_debug: bool,
    prefilter_debug_dir: Path | None,
    id_to_eval_index: dict[str, int],
    use_privjedai_fuzzy: bool,
    op_counter: dict[str, int] | None = None,
) -> dict[str, Any]:
    n_rows = len(working_df_str)
    if op_counter is not None:
        op_counter["targets_evaluated"] += 1

    target_id = str(target[target_id_col]).strip()
    known_values = {attr: str(target[attr]).strip() for attr in known_attrs}
    qid_known_values = {attr: known_values[attr] for attr in qid_filter_attrs}
    refine_known_values = {attr: known_values[attr] for attr in refine_attrs}

    if target_id not in id_to_eval_index:
        raise ValueError(f"Strict linkage violation: target '{target_id}' is missing from anonymized_eval.")

    qid_stage1_key = _make_qid_stage1_cache_key(
        known_values=known_values,
        qid_filter_attrs=qid_filter_attrs,
        attacker_knowledge=attacker_knowledge,
    )
    if qid_stage1_key not in qid_stage1_cache:
        if op_counter is not None:
            op_counter["qid_stage1_cache_misses"] += 1
        qid_stage1_cache[qid_stage1_key] = _build_qid_stage1_cache_entry(
            qid_known_values=qid_known_values,
            qid_filter_attrs=qid_filter_attrs,
            attacker_knowledge=attacker_knowledge,
            working_df_str=working_df_str,
            value_indices=value_indices,
            attr_unique_values=attr_unique_values,
            match_cache=match_cache,
            fuzzy_pair_cache=fuzzy_pair_cache,
            fuzzy_hash_cache=fuzzy_hash_cache,
            op_counter=op_counter,
        )
    else:
        if op_counter is not None:
            op_counter["qid_stage1_cache_hits"] += 1

    qid_cache_entry = qid_stage1_cache[qid_stage1_key]
    qid_compatible_values_debug = qid_cache_entry["qid_compatible_values_debug"]
    qid_kept_indices = qid_cache_entry["qid_kept_indices"]
    qid_df = _materialize_qid_df_from_cache_entry(
        cache_entry=qid_cache_entry,
        working_df_str=working_df_str,
    )

    qid_candidate_count = int(len(qid_df))
    if op_counter is not None:
        op_counter["equivalence_class_candidate_rows_output"] += qid_candidate_count

    true_record_in_qid_class = bool(qid_candidate_count > 0 and (qid_df[target_id_col] == target_id).any())
    if not true_record_in_qid_class:
        raise ValueError(
            "Strict linkage invariant violated: the true target record is absent from its QI equivalence class "
            f"for target '{target_id}'."
        )

    refined_df = qid_df.copy()
    refine_input_count = int(len(refined_df))
    if refine_input_count > 0:
        refine_exact_count = np.zeros(refine_input_count, dtype=np.int16)
        refine_fuzzy_count = np.zeros(refine_input_count, dtype=np.int16)
        refine_fuzzy_score_sum = np.zeros(refine_input_count, dtype=np.float32)
        refine_full_mask = np.ones(refine_input_count, dtype=bool)
        if op_counter is not None:
            op_counter["array_cells_initialized"] += int(3 * refine_input_count)
    else:
        refine_exact_count = np.zeros(0, dtype=np.int16)
        refine_fuzzy_count = np.zeros(0, dtype=np.int16)
        refine_fuzzy_score_sum = np.zeros(0, dtype=np.float32)
        refine_full_mask = np.ones(0, dtype=bool)

    for attr in refine_attrs:
        attr_positive_mask = np.zeros(refine_input_count, dtype=bool)
        if op_counter is not None:
            op_counter["refinement_mask_cells"] += int(refine_input_count)

        candidate_values = refined_df[attr].astype(str).str.strip().tolist()
        target_value = refine_known_values[attr]
        for idx, candidate_value in enumerate(candidate_values):
            if op_counter is not None:
                op_counter["refinement_candidate_row_visits"] += 1
                op_counter["refinement_exact_tests"] += 1
            if fuzzy_config is not None and target_value != candidate_value and not is_suppressed_value(candidate_value):
                if op_counter is not None:
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

    refined_df = refined_df.loc[refine_full_mask].copy()
    refined_df["refine_exact_match_count"] = refine_exact_count[refine_full_mask]
    refined_df["refine_privjedai_fuzzy_match_count"] = refine_fuzzy_count[refine_full_mask]
    refined_df["refine_privjedai_fuzzy_score_sum"] = refine_fuzzy_score_sum[refine_full_mask]
    refine_fuzzy_score_mean = np.full(int(np.sum(refine_full_mask)), np.nan, dtype=np.float32)
    np.divide(
        refine_fuzzy_score_sum[refine_full_mask],
        refine_fuzzy_count[refine_full_mask],
        out=refine_fuzzy_score_mean,
        where=refine_fuzzy_count[refine_full_mask] > 0,
    )
    refined_df["refine_privjedai_fuzzy_score_mean"] = refine_fuzzy_score_mean
    refined_df["total_matched_attr_count"] = (
        refined_df["qid_matched_attr_count"].fillna(0).astype(int)
        + refined_df["refine_exact_match_count"].fillna(0).astype(int)
        + refined_df["refine_privjedai_fuzzy_match_count"].fillna(0).astype(int)
    )

    refined_candidate_count = int(len(refined_df))
    if op_counter is not None:
        op_counter["reduced_equivalence_class_candidate_rows_output"] += refined_candidate_count

    true_record_in_refined_class = bool(refined_candidate_count > 0 and (refined_df[target_id_col] == target_id).any())
    unique_reidentified = bool(refined_candidate_count == 1 and true_record_in_refined_class)
    false_unique_match = bool(refined_candidate_count == 1 and not true_record_in_refined_class)
    has_any_fuzzy_candidate = bool(
        use_privjedai_fuzzy
        and refined_candidate_count > 0
        and (refined_df["refine_privjedai_fuzzy_match_count"] > 0).any()
    )
    true_record_fuzzy_kept = bool(
        use_privjedai_fuzzy
        and refined_candidate_count > 0
        and (
            refined_df.loc[refined_df[target_id_col] == target_id, "refine_privjedai_fuzzy_match_count"]
            .fillna(0)
            .astype(int)
            .gt(0)
            .any()
        )
    )

    prediction = summarize_sensitive_prediction(refined_df, sensitive_attr)
    target_eval_idx = id_to_eval_index[target_id]
    true_sensitive_value = str(working_df_str.iloc[target_eval_idx][sensitive_attr]).strip()
    true_sensitive_probability = prediction["distribution"].get(true_sensitive_value)

    qid_exposed_values = {
        attr: attacker_knowledge[attr].get("projection", {}).get(qid_known_values[attr], qid_known_values[attr])
        for attr in qid_filter_attrs
    }

    prefilter_artifacts: dict[str, Any] = {}
    if save_prefilter_debug and prefilter_debug_dir is not None:
        qid_kept_debug_df = pd.DataFrame({
            "row_index": qid_kept_indices.astype(int),
            "record_id": working_df_str.loc[qid_full_compatible_mask, target_id_col].astype(str).tolist(),
        })
        qid_kept_debug_path = prefilter_debug_dir / f"target_{target_id}__qid_kept_rows.csv"
        qid_kept_debug_df.to_csv(qid_kept_debug_path, index=False)

        refined_kept_ids = refined_df[target_id_col].astype(str).tolist()
        refined_kept_debug_path = prefilter_debug_dir / f"target_{target_id}__refined_kept_rows.csv"
        pd.DataFrame({
            "record_id": refined_kept_ids,
        }).to_csv(refined_kept_debug_path, index=False)

        qid_compatible_values_by_attr: dict[str, list[str]] = {}
        qid_compatible_kinds_by_attr: dict[str, dict[str, str]] = {}
        for attr in qid_filter_attrs:
            qid_compatible_values_by_attr[attr] = sorted(qid_compatible_values_debug[attr].keys())
            qid_compatible_kinds_by_attr[attr] = {
                key: str(payload["kind"])
                for key, payload in qid_compatible_values_debug[attr].items()
            }

        debug_payload = {
            "target_id": target_id,
            "known_attrs": known_attrs,
            "qid_filter_attrs": qid_filter_attrs,
            "refine_attrs": refine_attrs,
            "known_values": known_values,
            "qid_exposed_values": qid_exposed_values,
            "qid_compatible_values_by_attr": qid_compatible_values_by_attr,
            "qid_compatible_kinds_by_attr": qid_compatible_kinds_by_attr,
            "qid_kept_row_indices": qid_kept_indices.astype(int).tolist(),
            "qid_kept_record_ids": working_df_str.loc[qid_full_compatible_mask, target_id_col].astype(str).tolist(),
            "qid_kept_count": int(len(qid_kept_indices)),
            "refined_kept_record_ids": refined_kept_ids,
            "refined_kept_count": int(len(refined_kept_ids)),
        }
        debug_json_path = prefilter_debug_dir / f"target_{target_id}__filter.json"
        save_json(debug_json_path, debug_payload)
        prefilter_artifacts = {
            "qid_kept_debug_path": qid_kept_debug_path,
            "refined_kept_debug_path": refined_kept_debug_path,
            "debug_json_path": debug_json_path,
        }

    reduction_absolute = qid_candidate_count - refined_candidate_count
    reduction_rate = None if qid_candidate_count == 0 else float(reduction_absolute) / float(qid_candidate_count)

    per_target_row = {
        "target_id": target_id,
        "known_values": " | ".join(f"{attr}={known_values[attr]}" for attr in known_attrs),
        "qid_filter_values": " | ".join(f"{attr}={qid_known_values[attr]}" for attr in qid_filter_attrs),
        "refine_values": " | ".join(f"{attr}={refine_known_values[attr]}" for attr in refine_attrs) if refine_attrs else "",
        "qid_exposed_values": " | ".join(f"{attr}={qid_exposed_values[attr]}" for attr in qid_filter_attrs),
        "qid_equivalence_class_size": qid_candidate_count,
        "stage1_equivalence_class_size": qid_candidate_count,
        "reduced_equivalence_class_size": refined_candidate_count,
        "stage2_equivalence_class_size": refined_candidate_count,
        "equivalence_class_size": refined_candidate_count,
        "equivalence_class_reduction": reduction_absolute,
        "equivalence_class_reduction_rate": None if reduction_rate is None else round(reduction_rate, 6),
        "true_record_in_qid_class": true_record_in_qid_class,
        "true_record_in_reduced_class": true_record_in_refined_class,
        "unique_reidentified": unique_reidentified,
        "false_unique_match": false_unique_match,
        "true_sensitive_value": true_sensitive_value,
        "predicted_sensitive_top_value": prediction["top_value"],
        "predicted_sensitive_top_probability": None if prediction["top_probability"] is None else round(float(prediction["top_probability"]), 6),
        "true_sensitive_probability": None if true_sensitive_probability is None else round(float(true_sensitive_probability), 6),
        "sensitive_value_certain": prediction["is_certain"],
        "n_distinct_sensitive_values": prediction["n_distinct_sensitive_values"],
        "sensitive_distribution_json": json.dumps(
            {key: round(float(value), 6) for key, value in prediction["distribution"].items()},
            ensure_ascii=False,
        ),
    }
    if use_privjedai_fuzzy:
        per_target_row["reduced_candidates_with_privjedai_fuzzy"] = int(
            (refined_df["refine_privjedai_fuzzy_match_count"] > 0).sum()
        ) if refined_candidate_count > 0 else 0
        per_target_row["true_record_used_privjedai_fuzzy"] = true_record_fuzzy_kept

    equivalence_class_rows = []
    for _, candidate in refined_df.iterrows():
        row = {
            "target_id": target_id,
            "candidate_record_id": str(candidate[target_id_col]),
            "candidate_sensitive_value": str(candidate[sensitive_attr]),
            "candidate_qid_matched_attr_count": int(candidate["qid_matched_attr_count"]),
            "candidate_qid_exact_match_count": int(candidate["qid_exact_match_count"]),
            "candidate_qid_generalized_match_count": int(candidate["qid_generalized_match_count"]),
            "candidate_qid_suppressed_match_count": int(candidate["qid_suppressed_match_count"]),
            "candidate_refine_exact_match_count": int(candidate["refine_exact_match_count"]),
            "candidate_refine_privjedai_fuzzy_match_count": int(candidate["refine_privjedai_fuzzy_match_count"]),
            "candidate_total_matched_attr_count": int(candidate["total_matched_attr_count"]),
            "candidate_is_true_record": str(candidate[target_id_col]) == target_id,
        }
        if use_privjedai_fuzzy:
            row.update(
                {
                    "candidate_refine_privjedai_fuzzy_score_sum": None if pd.isna(candidate["refine_privjedai_fuzzy_score_sum"]) else round(float(candidate["refine_privjedai_fuzzy_score_sum"]), 6),
                    "candidate_refine_privjedai_fuzzy_score_mean": None if pd.isna(candidate["refine_privjedai_fuzzy_score_mean"]) else round(float(candidate["refine_privjedai_fuzzy_score_mean"]), 6),
                }
            )
        equivalence_class_rows.append(row)

    return {
        "target_id": target_id,
        "qid_candidate_count": qid_candidate_count,
        "compatible_candidate_count": refined_candidate_count,
        "refined_candidate_count": refined_candidate_count,
        "unique_reidentified": unique_reidentified,
        "false_unique_match": false_unique_match,
        "true_record_in_qid_class": true_record_in_qid_class,
        "true_record_in_refined_class": true_record_in_refined_class,
        "has_any_fuzzy_candidate": has_any_fuzzy_candidate,
        "true_record_fuzzy_kept": true_record_fuzzy_kept,
        "prediction": prediction,
        "true_sensitive_probability": true_sensitive_probability,
        "per_target_row": per_target_row,
        "equivalence_class_rows": equivalence_class_rows,
        **prefilter_artifacts,
    }


# Run the full linkage attack workflow and save all outputs.
def run_linkage_attack(
    *,
    runtime: dict[str, Any],
    df_aux: pd.DataFrame,
    df_public: pd.DataFrame,
    df_eval: pd.DataFrame,
    known_attrs: list[str],
    target_id_col: str = "record_id",
    sensitive_attr: str | None = None,
    n_targets: int | str = 500,
    seed: int = 42,
    output_root: str | Path = "outputs",
    name: str | None = None,
    config_path: str | Path | None = None,
    auxiliary_path: str | Path | None = None,
    anonymized_path: str | Path | None = None,
    anonymized_eval_path: str | Path | None = None,
    save_prefilter_debug: bool = False,
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
    obfuscate_refine_attrs: str | None = None,
    schema_matcher_name: str = "jaccard",
    schema_matcher_min_score: float = 0.0,
) -> dict[str, Any]:
    sensitive_attr = sensitive_attr or (runtime.get("sensitive_attributes") or [None])[0]
    if not sensitive_attr:
        raise ValueError("No sensitive attribute available. Pass --sensitive-attr explicitly.")

    resolved_n_targets = resolve_n_targets(n_targets, df_aux)
    n_targets_label = "all" if isinstance(n_targets, str) and n_targets.strip().lower() == "all" else str(resolved_n_targets)

    _validate_inputs(
        known_attrs=known_attrs,
        target_id_col=target_id_col,
        sensitive_attr=sensitive_attr,
        df_aux=df_aux,
        df_public=df_public,
        df_eval=df_eval,
        n_targets=resolved_n_targets,
    )

    fuzzy_config = _maybe_build_fuzzy_config(
        use_privjedai_fuzzy=use_privjedai_fuzzy,
        privjedai_src=privjedai_src,
        privjedai_fuzzy_threshold=privjedai_fuzzy_threshold,
        privjedai_fuzzy_metric=privjedai_fuzzy_metric,
        privjedai_bloom_size=privjedai_bloom_size,
        privjedai_bloom_num_hashes=privjedai_bloom_num_hashes,
        privjedai_bloom_qgrams=privjedai_bloom_qgrams,
        privjedai_bloom_hashing_type=privjedai_bloom_hashing_type,
    )

    attacker_knowledge = build_attacker_knowledge(runtime=runtime, known_attrs=known_attrs, df_public=df_public)

    _normalize_attacker_visible_levels(
        attacker_knowledge=attacker_knowledge,
        known_attrs=known_attrs,
        df_public=df_public,
    )

    schema_matching_results: dict[str, Any] | None = None
    schema_matching_results_path: Path | None = None
    schema_matching_pairs_path: Path | None = None

    # --- Schema matching step (between phase 1 and phase 2) ------------------
    if obfuscate_refine_attrs:
        from schema_matcher import (
            obfuscate_columns,
            recover_column_mapping,
            jaccard_baseline,
            apply_recovered_mapping,
            evaluate_mapping,
        )

        cols_to_hide = [c.strip() for c in obfuscate_refine_attrs.split(",") if c.strip()]
        df_public, truth = obfuscate_columns(df_public, cols_to_hide, prefix="col_")

        kb_candidates = [c for c in df_aux.columns
                         if c not in (target_id_col, sensitive_attr)]

        if schema_matcher_name == "baseline_jaccard":
            recovered = jaccard_baseline(
                df_anon=df_public, df_kb=df_aux,
                anon_unknown_cols=list(truth.keys()),
                kb_candidate_cols=kb_candidates,
                min_score=schema_matcher_min_score,
            )
        else:
            recovered = recover_column_mapping(
                df_anon=df_public, df_kb=df_aux,
                anon_unknown_cols=list(truth.keys()),
                kb_candidate_cols=kb_candidates,
                matcher_name=schema_matcher_name,
                min_score=schema_matcher_min_score,
            )

        metrics = evaluate_mapping(recovered, truth)
        print(f"[schema-matching] {metrics}")

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
            "metrics": metrics,
            "pairs": schema_matching_pairs,
        }

        # Rename back so the rest of the pipeline works unchanged.
        df_public, _ = apply_recovered_mapping(df_public, recovered)

        # known_attrs must drop any originally-obfuscated column whose name
        # could not be recovered: the attacker cannot use it in phase 2.
        unrecoverable = {orig for _, orig in truth.items() if orig not in recovered_kb_cols}
        known_attrs = [a for a in known_attrs if a not in unrecoverable]

    qid_filter_attrs, refine_attrs, skipped_refine_attrs = _split_attack_attributes(
        runtime=runtime,
        known_attrs=known_attrs,
        attacker_knowledge=attacker_knowledge,
    )
    op_counter = make_operation_counter()

    working_df = df_public.copy().reset_index(drop=True)
    working_df[target_id_col] = df_eval[target_id_col].astype(str).reset_index(drop=True)
    if sensitive_attr not in working_df.columns:
        working_df[sensitive_attr] = df_eval[sensitive_attr].astype(str).reset_index(drop=True)

    working_df_str = working_df.copy()
    for attr in known_attrs:
        working_df_str[attr] = working_df[attr].astype(str).str.strip()
    working_df_str[target_id_col] = working_df[target_id_col].astype(str).str.strip()
    working_df_str[sensitive_attr] = working_df[sensitive_attr].astype(str).str.strip()

    sampled_targets = df_aux.sample(n=resolved_n_targets, random_state=seed, replace=False).reset_index(drop=True)
    for attr in known_attrs:
        sampled_targets[attr] = sampled_targets[attr].astype(str).str.strip()
    sampled_targets[target_id_col] = sampled_targets[target_id_col].astype(str).str.strip()

    id_to_eval_index = {str(record_id): idx for idx, record_id in enumerate(working_df_str[target_id_col].tolist())}
    _validate_strict_target_set(
        sampled_targets=sampled_targets,
        target_id_col=target_id_col,
        id_to_eval_index=id_to_eval_index,
    )

    output_root = ensure_dir(Path(output_root).resolve())
    attack_root = ensure_dir(output_root / "attacks" / "linkage")
    inferred_anonymized_path = Path(anonymized_path).resolve() if anonymized_path else Path("in_memory_public.csv")
    attack_id = name or make_attack_id(inferred_anonymized_path, known_attrs, n_targets_label, seed, use_privjedai_fuzzy=use_privjedai_fuzzy)
    attack_dir = ensure_dir(attack_root / attack_id.replace(".", "_"))
    prefilter_debug_dir = ensure_dir(attack_dir / "prefilter_debug") if save_prefilter_debug else None

    if schema_matching_results is not None:
        schema_matching_results_path = attack_dir / "schema_matching_results.json"
        schema_matching_pairs_path = attack_dir / "schema_matching_pairs.csv"

        save_json(schema_matching_results_path, schema_matching_results)
        pd.DataFrame(schema_matching_results["pairs"]).to_csv(schema_matching_pairs_path, index=False)

    match_cache: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    qid_stage1_cache: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
    fuzzy_pair_cache: dict[tuple[str, str], float] = {}
    fuzzy_hash_cache: dict[str, frozenset[int]] = {}

    per_target_rows: list[dict[str, Any]] = []
    equivalence_class_rows: list[dict[str, Any]] = []
    qid_candidate_counts: list[int] = []
    compatible_candidate_counts: list[int] = []
    unique_reidentified_flags: list[bool] = []
    false_unique_match_flags: list[bool] = []
    true_record_kept_after_refinement_flags: list[bool] = []
    certainty_sensitive_flags: list[bool] = []
    true_sensitive_probabilities: list[float] = []
    top_sensitive_probabilities: list[float] = []
    targets_with_any_fuzzy_candidate: list[bool] = []
    targets_with_true_record_fuzzy_kept: list[bool] = []

    attr_unique_values = {attr: sorted({str(v).strip() for v in working_df_str[attr].astype(str).tolist()}) for attr in qid_filter_attrs}
    value_indices = build_value_indices(working_df_str, qid_filter_attrs, op_counter=op_counter)

    for _, target in sampled_targets.iterrows():
        result = _evaluate_target(
            target=target,
            target_id_col=target_id_col,
            sensitive_attr=sensitive_attr,
            known_attrs=known_attrs,
            qid_filter_attrs=qid_filter_attrs,
            refine_attrs=refine_attrs,
            attacker_knowledge=attacker_knowledge,
            working_df_str=working_df_str,
            value_indices=value_indices,
            attr_unique_values=attr_unique_values,
            match_cache=match_cache,
            qid_stage1_cache=qid_stage1_cache,
            fuzzy_config=fuzzy_config,
            fuzzy_pair_cache=fuzzy_pair_cache,
            fuzzy_hash_cache=fuzzy_hash_cache,
            save_prefilter_debug=save_prefilter_debug,
            prefilter_debug_dir=prefilter_debug_dir,
            id_to_eval_index=id_to_eval_index,
            use_privjedai_fuzzy=use_privjedai_fuzzy,
            op_counter=op_counter,
        )
        qid_candidate_counts.append(result["qid_candidate_count"])
        compatible_candidate_counts.append(result["compatible_candidate_count"])
        unique_reidentified_flags.append(result["unique_reidentified"])
        false_unique_match_flags.append(result["false_unique_match"])
        true_record_kept_after_refinement_flags.append(result["true_record_in_refined_class"])
        targets_with_any_fuzzy_candidate.append(result["has_any_fuzzy_candidate"])
        targets_with_true_record_fuzzy_kept.append(result["true_record_fuzzy_kept"])
        if result["prediction"]["is_certain"] is not None:
            certainty_sensitive_flags.append(bool(result["prediction"]["is_certain"]))
        if result["prediction"]["top_probability"] is not None:
            top_sensitive_probabilities.append(float(result["prediction"]["top_probability"]))
        if result["true_sensitive_probability"] is not None:
            true_sensitive_probabilities.append(float(result["true_sensitive_probability"]))
        per_target_rows.append(result["per_target_row"])
        equivalence_class_rows.extend(result["equivalence_class_rows"])

    per_target_path = attack_dir / "targets.csv"
    equivalence_class_path = attack_dir / "equivalence_class_candidates.csv"
    attacker_knowledge_path = attack_dir / "attacker_knowledge.json"
    summary_path = attack_dir / "summary.json"

    pd.DataFrame(per_target_rows).to_csv(per_target_path, index=False)
    pd.DataFrame(equivalence_class_rows).to_csv(equivalence_class_path, index=False)
    save_json(attacker_knowledge_path, attacker_knowledge)

    certainty_sensitive_rate = sum(certainty_sensitive_flags) / len(certainty_sensitive_flags) if certainty_sensitive_flags else None
    avg_true_sensitive_probability = float(np.mean(true_sensitive_probabilities)) if true_sensitive_probabilities else None
    median_true_sensitive_probability = float(np.median(true_sensitive_probabilities)) if true_sensitive_probabilities else None
    avg_top_sensitive_probability = float(np.mean(top_sensitive_probabilities)) if top_sensitive_probabilities else None

    avg_qid_equivalence_class_size = float(np.mean(qid_candidate_counts)) if qid_candidate_counts else None
    median_qid_equivalence_class_size = float(np.median(qid_candidate_counts)) if qid_candidate_counts else None
    avg_reduction_rate = (
        float(np.mean([(before - after) / before for before, after in zip(qid_candidate_counts, compatible_candidate_counts) if before > 0]))
        if qid_candidate_counts
        else None
    )
    unique_reidentification_rate = round(sum(unique_reidentified_flags) / len(unique_reidentified_flags), 6)
    false_unique_match_rate = round(sum(false_unique_match_flags) / len(false_unique_match_flags), 6)
    true_record_kept_after_refinement_rate = round(
        sum(true_record_kept_after_refinement_flags) / len(true_record_kept_after_refinement_flags), 6
    )

    summary = {
        "attack_id": attack_id,
        "config_path": str(config_path) if config_path else "",
        "auxiliary_path": str(auxiliary_path) if auxiliary_path else "",
        "anonymized_public_path": str(anonymized_path) if anonymized_path else "",
        "anonymized_eval_path": str(anonymized_eval_path) if anonymized_eval_path else "",
        "known_attrs": known_attrs,
        "qid_filter_attrs": qid_filter_attrs,
        "stage1_filter_attrs": qid_filter_attrs,
        "refine_attrs": refine_attrs,
        "stage2_refine_attrs": refine_attrs,
        "skipped_refine_attrs": skipped_refine_attrs,
        "target_id_col": target_id_col,
        "sensitive_attr": sensitive_attr,
        "n_targets": int(resolved_n_targets),
        "n_targets_requested": n_targets_label,
        "seed": seed,
        "n_anonymized_rows": int(len(working_df_str)),
        "attacker_knowledge_json": str(attacker_knowledge_path),
        "prefilter_debug_dir": str(prefilter_debug_dir) if prefilter_debug_dir is not None else None,
        "attacker_visible_levels": {attr: int(attacker_knowledge[attr]["visible_level"]) for attr in known_attrs},
        "attacker_raw_visible_levels": {
            attr: int(attacker_knowledge[attr].get("raw_visible_level", attacker_knowledge[attr]["visible_level"]))
            for attr in known_attrs
        },
        "use_privjedai_fuzzy": bool(use_privjedai_fuzzy),
        "n_distinct_stage1_groups": int(len(qid_stage1_cache)),
        "operation_counter": build_public_operation_counter(op_counter),
        "unique_reidentification_rate": unique_reidentification_rate,
        "avg_qid_equivalence_class_size": None if avg_qid_equivalence_class_size is None else round(avg_qid_equivalence_class_size, 6),
        "avg_stage1_equivalence_class_size": None if avg_qid_equivalence_class_size is None else round(avg_qid_equivalence_class_size, 6),
        "median_qid_equivalence_class_size": None if median_qid_equivalence_class_size is None else round(median_qid_equivalence_class_size, 6),
        "median_stage1_equivalence_class_size": None if median_qid_equivalence_class_size is None else round(median_qid_equivalence_class_size, 6),
        "avg_equivalence_class_size": round(float(np.mean(compatible_candidate_counts)), 6),
        "median_equivalence_class_size": round(float(np.median(compatible_candidate_counts)), 6),
        "avg_reduction_rate": None if avg_reduction_rate is None else round(avg_reduction_rate, 6),
        "max_equivalence_class_size": int(np.max(compatible_candidate_counts)) if compatible_candidate_counts else 0,
        "certainty_sensitive_inference_rate": None if certainty_sensitive_rate is None else round(certainty_sensitive_rate, 6),
        "avg_true_sensitive_probability": None if avg_true_sensitive_probability is None else round(avg_true_sensitive_probability, 6),
        "median_true_sensitive_probability": None if median_true_sensitive_probability is None else round(median_true_sensitive_probability, 6),
        "avg_top_sensitive_probability": None if avg_top_sensitive_probability is None else round(avg_top_sensitive_probability, 6),
        "targets_csv": str(per_target_path),
        "equivalence_class_candidates_csv": str(equivalence_class_path),
        "schema_matching_enabled": schema_matching_results is not None,
        "schema_matcher_name": schema_matcher_name if schema_matching_results is not None else None,
        "schema_matcher_min_score": float(schema_matcher_min_score) if schema_matching_results is not None else None,
        "schema_matching_results_json": str(schema_matching_results_path) if schema_matching_results_path is not None else None,
        "schema_matching_pairs_csv": str(schema_matching_pairs_path) if schema_matching_pairs_path is not None else None,
    }
    if use_privjedai_fuzzy and fuzzy_config is not None:
        summary.update(
            {
                "privjedai_fuzzy_threshold": float(fuzzy_config["threshold"]),
                "privjedai_fuzzy_metric": str(fuzzy_config["metric"]),
                "privjedai_bloom_size": int(fuzzy_config["bloom_size"]),
                "privjedai_bloom_num_hashes": int(fuzzy_config["num_hashes"]),
                "privjedai_bloom_qgrams": int(fuzzy_config["qgrams"]),
                "privjedai_bloom_hashing_type": str(fuzzy_config["hashing_type"]),
                "targets_with_any_privjedai_fuzzy_candidate_rate": round(sum(targets_with_any_fuzzy_candidate) / len(targets_with_any_fuzzy_candidate), 6),
                "targets_with_true_record_kept_by_privjedai_fuzzy_rate": round(sum(targets_with_true_record_fuzzy_kept) / len(targets_with_true_record_fuzzy_kept), 6),
            }
        )
    save_json(summary_path, summary)

    report_path: Path | None = None
    if generate_report:
        try:
            report_path = _maybe_generate_linkage_report(
                output_root=output_root,
                summary_path=summary_path,
                attack_id=attack_id,
                report_title=report_title,
            )
        except Exception as exc:
            print(f"[WARN] Linkage HTML report generation failed: {exc}")

    summary_row = {
        "attack_id": attack_id,
        "config_path": str(config_path) if config_path else "",
        "anonymized_public_path": str(anonymized_path) if anonymized_path else "",
        "anonymized_eval_path": str(anonymized_eval_path) if anonymized_eval_path else "",
        "auxiliary_path": str(auxiliary_path) if auxiliary_path else "",
        "known_attrs": "|".join(known_attrs),
        "qid_filter_attrs": "|".join(qid_filter_attrs),
        "refine_attrs": "|".join(refine_attrs),
        "sensitive_attr": sensitive_attr,
        "n_targets": resolved_n_targets,
        "n_targets_requested": n_targets_label,
        "seed": seed,
        "attacker_visible_levels": "|".join(f"{attr}:{attacker_knowledge[attr]['visible_level']}" for attr in known_attrs),
        "attacker_raw_visible_levels": "|".join(
            f"{attr}:{attacker_knowledge[attr].get('raw_visible_level', attacker_knowledge[attr]['visible_level'])}"
            for attr in known_attrs
        ),
        "use_privjedai_fuzzy": bool(use_privjedai_fuzzy),
        "n_distinct_stage1_groups": summary["n_distinct_stage1_groups"],
        "estimated_total_operations": summary["operation_counter"]["estimated_total_operations"],
        "unique_reidentification_rate": summary["unique_reidentification_rate"],
        "false_unique_match_rate": false_unique_match_rate,
        "avg_qid_equivalence_class_size": summary["avg_qid_equivalence_class_size"],
        "avg_equivalence_class_size": summary["avg_equivalence_class_size"],
        "avg_reduction_rate": summary["avg_reduction_rate"],
        "certainty_sensitive_inference_rate": summary["certainty_sensitive_inference_rate"],
        "avg_true_sensitive_probability": summary["avg_true_sensitive_probability"],
        "avg_top_sensitive_probability": summary["avg_top_sensitive_probability"],
        "summary_json": str(summary_path),
        "report_html": str(report_path) if report_path is not None else "",
        "avg_stage1_equivalence_class_size": summary["avg_stage1_equivalence_class_size"],
        "schema_matching_enabled": schema_matching_results is not None,
        "schema_matcher_name": schema_matcher_name if schema_matching_results is not None else "",
        "schema_matching_results_json": str(schema_matching_results_path) if schema_matching_results_path is not None else "",
        "schema_matching_pairs_csv": str(schema_matching_pairs_path) if schema_matching_pairs_path is not None else "",
    }
    if use_privjedai_fuzzy and fuzzy_config is not None:
        summary_row.update(
            {
                "privjedai_fuzzy_threshold": float(fuzzy_config["threshold"]),
                "privjedai_fuzzy_metric": str(fuzzy_config["metric"]),
                "targets_with_any_privjedai_fuzzy_candidate_rate": summary["targets_with_any_privjedai_fuzzy_candidate_rate"],
                "targets_with_true_record_kept_by_privjedai_fuzzy_rate": summary["targets_with_true_record_kept_by_privjedai_fuzzy_rate"],
            }
        )
    append_attack_summary(attack_root / "linkage_summary.csv", summary_row)

    print(f"Attack summary            : {summary_path}")
    if report_path is not None:
        print(f"HTML report               : {report_path}")
    print(f"Attacker knowledge        : {attacker_knowledge_path}")
    print(f"Per-target results        : {per_target_path}")
    print(f"Equivalence class records : {equivalence_class_path}")
    print(f"Avg QI equivalence size   : {summary['avg_qid_equivalence_class_size']}")
    print(f"Avg reduced eq size       : {summary['avg_equivalence_class_size']}")
    print(f"Avg reduction rate        : {summary['avg_reduction_rate']}")
    print(f"Distinct stage1 groups    : {summary['n_distinct_stage1_groups']}")
    print(f"Logical ops (est.)       : {summary['operation_counter']['estimated_total_operations']}")
    print(f"Unique reidentification   : {summary['unique_reidentification_rate']}")
    print(f"False unique matches      : {false_unique_match_rate}")
    print(f"Avg true sensitive prob   : {summary['avg_true_sensitive_probability']}")
    if use_privjedai_fuzzy:
        print(f"Targets with fuzzy cand.  : {summary['targets_with_any_privjedai_fuzzy_candidate_rate']}")
        print(f"True records rescued fuzz.: {summary['targets_with_true_record_kept_by_privjedai_fuzzy_rate']}")
    if schema_matching_results_path is not None:
        print(f"Schema matching JSON      : {schema_matching_results_path}")
        print(f"Schema matching pairs CSV : {schema_matching_pairs_path}")

    return {
        "summary": summary,
        "summary_path": summary_path,
        "attacker_knowledge_path": attacker_knowledge_path,
        "targets_path": per_target_path,
        "equivalence_class_path": equivalence_class_path,
        "report_path": report_path,
        "per_target_rows": per_target_rows,
        "equivalence_class_rows": equivalence_class_rows,
        "attacker_knowledge": attacker_knowledge,
        "prefilter_debug_dir": prefilter_debug_dir,
        "schema_matching_results_path": schema_matching_results_path,
        "schema_matching_pairs_path": schema_matching_pairs_path,
        "schema_matching_results": schema_matching_results,
    }


# Load all input files and run one linkage attack from file paths.
def run_linkage_attack_from_paths(
    *,
    config_path: str | Path,
    auxiliary_path: str | Path,
    anonymized_path: str | Path,
    anonymized_eval_path: str | Path,
    known_attrs: list[str],
    target_id_col: str = "record_id",
    sensitive_attr: str | None = None,
    n_targets: int | str = 500,
    seed: int = 42,
    output_root: str | Path = "outputs",
    name: str | None = None,
    save_prefilter_debug: bool = False,
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
    obfuscate_refine_attrs: str | None = None,
    schema_matcher_name: str = "jaccard",
    schema_matcher_min_score: float = 0.0,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    auxiliary_path = Path(auxiliary_path).resolve()
    anonymized_path = Path(anonymized_path).resolve()
    anonymized_eval_path = Path(anonymized_eval_path).resolve()

    runtime = load_runtime_config(config_path)
    df_aux = read_csv_str(auxiliary_path)
    df_public = read_csv_str(anonymized_path)
    df_eval = read_csv_str(anonymized_eval_path)

    return run_linkage_attack(
        runtime=runtime,
        df_aux=df_aux,
        df_public=df_public,
        df_eval=df_eval,
        known_attrs=known_attrs,
        target_id_col=target_id_col,
        sensitive_attr=sensitive_attr,
        n_targets=n_targets,
        seed=seed,
        output_root=output_root,
        name=name,
        config_path=config_path,
        auxiliary_path=auxiliary_path,
        anonymized_path=anonymized_path,
        anonymized_eval_path=anonymized_eval_path,
        save_prefilter_debug=save_prefilter_debug,
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
        obfuscate_refine_attrs=obfuscate_refine_attrs,
        schema_matcher_name=schema_matcher_name,
        schema_matcher_min_score=schema_matcher_min_score,
    )


# Parse CLI arguments and launch the linkage attack.
def main() -> None:
    parser = argparse.ArgumentParser(description="Run a strict linkage attack against one anonymized dataset.")
    parser.add_argument("--config", required=True, help="Generated runtime config JSON from outputs/configs/...")
    parser.add_argument("--auxiliary", required=True, help="Auxiliary attacker knowledge CSV.")
    parser.add_argument("--anonymized", required=True, help="Public anonymized CSV (without record_id ideally).")
    parser.add_argument("--anonymized-eval", required=True, help="Internal anonymized CSV with record_id kept for evaluation.")
    parser.add_argument(
        "--known-attrs",
        "--known-qids",
        dest="known_attrs_raw",
        required=True,
        help=(
            "Comma-separated list of attributes known by the attacker. "
            "The attack first filters with attacker-known attributes whose visible_level is not 0, "
            "then refines with attacker-known clear-text attributes whose visible_level is 0."
        ),
    )
    parser.add_argument("--target-id-col", default="record_id", help="Internal record identifier column.")
    parser.add_argument("--sensitive-attr", default=None, help="Sensitive attribute to inspect. Defaults to the first sensitive attribute in the runtime config.")
    parser.add_argument(
        "--n-targets",
        type=parse_n_targets_arg,
        default=500,
        help="Number of target individuals sampled from the auxiliary base, or 'all' to attack the whole auxiliary base.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for target sampling.")
    parser.add_argument("--output-root", default="outputs", help="Root output directory.")
    parser.add_argument("--name", default=None, help="Optional custom attack name.")
    parser.add_argument(
        "--save-prefilter-debug",
        action="store_true",
        help="Save one debug file per target with kept row indices and record_id after compatibility filtering.",
    )
    parser.add_argument(
        "--use-privjedai-fuzzy",
        action="store_true",
        help=(
            "Enable an optional privJedAI fallback during the refinement stage on attacker-known clear-text attributes: "
            "if an exact one-to-one match fails, try a privJedAI similarity check."
        ),
    )
    parser.add_argument("--privjedai-src", default=None, help="Path to privJedAI-main/src if privjedai is not installed.")
    parser.add_argument("--privjedai-fuzzy-threshold", type=float, default=0.9)
    parser.add_argument("--privjedai-fuzzy-metric", choices=["dice", "jaccard", "cosine", "scm"], default="dice")
    parser.add_argument("--privjedai-bloom-size", type=int, default=1024)
    parser.add_argument("--privjedai-bloom-num-hashes", type=int, default=15)
    parser.add_argument("--privjedai-bloom-qgrams", type=int, default=4)
    parser.add_argument(
        "--privjedai-bloom-hashing-type",
        choices=["salted_string", "salted_qgrams", "salted_skipqgrams", "salted_metaphone", "salted_tokens"],
        default="salted_qgrams",
    )
    parser.add_argument(
        "--no-generate-report",
        action="store_true",
        help="Do not auto-generate the HTML linkage report at the end of the attack.",
    )
    parser.add_argument(
        "--report-title",
        default=None,
        help="Optional custom title for the generated HTML report.",
    )

    parser.add_argument(
        "--obfuscate-refine-attrs",
        default=None,
        help="Comma-separated list of refine_attrs to treat as unknown-named "
          "in df_public; the schema matcher will recover them before phase 2.",
    )   
    parser.add_argument(
        "--schema-matcher",
        choices=["coma", "jaccard", "distribution", "baseline_jaccard"],
        default="jaccard",
    )
    parser.add_argument(
        "--schema-matcher-min-score",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    run_linkage_attack_from_paths(
        config_path=args.config,
        auxiliary_path=args.auxiliary,
        anonymized_path=args.anonymized,
        anonymized_eval_path=args.anonymized_eval,
        known_attrs=parse_csv_list(args.known_attrs_raw),
        target_id_col=args.target_id_col,
        sensitive_attr=args.sensitive_attr,
        n_targets=args.n_targets,
        seed=args.seed,
        output_root=args.output_root,
        name=args.name,
        save_prefilter_debug=args.save_prefilter_debug,
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
        obfuscate_refine_attrs=args.obfuscate_refine_attrs,
        schema_matcher_name=args.schema_matcher,
        schema_matcher_min_score=args.schema_matcher_min_score,
    )


if __name__ == "__main__":
    main()
