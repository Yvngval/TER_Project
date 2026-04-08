# Run one linkage attack against an anonymized dataset.
# Attacker-known attributes are independent from the anonymization QI choice.
# Compatibility is binary: a record is either compatible with the target or not.
# Sensitive attribute inference is then computed from the target's equivalence class.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from attack_common import (
    append_attack_summary,
    build_attacker_knowledge,
    load_runtime_config,
    read_csv_str,
)
from common import ensure_dir, parse_csv_list, save_json
from linkage_helpers import (
    build_value_indices,
    compute_sensitive_distribution,
    get_match_mapping_for_target_value,
    summarize_sensitive_prediction,
)
from privjedai_utils import build_privjedai_fuzzy_config


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
    attr_part = "-".join(known_attrs)
    suffix = "__privjedai_fuzzy" if use_privjedai_fuzzy else ""
    return f"{anonymized_path.stem}__known_{attr_part}__targets_{n_targets}__seed_{seed}{suffix}"


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


# Compute one target's equivalence class and all derived reporting fields.
def _evaluate_target(
    *,
    target: pd.Series,
    target_id_col: str,
    sensitive_attr: str,
    known_attrs: list[str],
    attacker_knowledge: dict[str, dict[str, Any]],
    working_df_str: pd.DataFrame,
    value_indices: dict[str, dict[str, np.ndarray]],
    attr_unique_values: dict[str, list[str]],
    match_cache: dict[tuple[str, str, str], dict[str, dict[str, Any]]],
    fuzzy_config: dict[str, Any] | None,
    fuzzy_pair_cache: dict[tuple[str, str], float],
    fuzzy_hash_cache: dict[str, frozenset[int]],
    save_prefilter_debug: bool,
    prefilter_debug_dir: Path | None,
    id_to_eval_index: dict[str, int],
) -> dict[str, Any]:
    n_rows = len(working_df_str)
    target_id = str(target[target_id_col]).strip()
    known_values = {attr: str(target[attr]).strip() for attr in known_attrs}

    compatible_count = np.zeros(n_rows, dtype=np.int16)
    exact_count = np.zeros(n_rows, dtype=np.int16)
    generalized_count = np.zeros(n_rows, dtype=np.int16)
    suppressed_count = np.zeros(n_rows, dtype=np.int16)
    fuzzy_count = np.zeros(n_rows, dtype=np.int16)
    fuzzy_score_sum = np.zeros(n_rows, dtype=np.float32)

    full_compatible_mask = np.ones(n_rows, dtype=bool)
    compatible_values_debug: dict[str, dict[str, dict[str, Any]]] = {}

    for attr in known_attrs:
        mapping = get_match_mapping_for_target_value(
            attr=attr,
            target_value=known_values[attr],
            possible_anonymized_values=attr_unique_values[attr],
            attacker_knowledge=attacker_knowledge,
            match_cache=match_cache,
            fuzzy_config=fuzzy_config,
            fuzzy_pair_cache=fuzzy_pair_cache,
            fuzzy_hash_cache=fuzzy_hash_cache,
        )
        compatible_values_debug[attr] = mapping
        attr_positive_mask = np.zeros(n_rows, dtype=bool)

        for anonymized_value, result in mapping.items():
            row_ids = value_indices[attr].get(anonymized_value)
            if row_ids is None or len(row_ids) == 0:
                continue

            kind = str(result["kind"])
            score = result.get("score")
            attr_positive_mask[row_ids] = True
            compatible_count[row_ids] += 1

            if kind == "exact":
                exact_count[row_ids] += 1
            elif kind == "generalized":
                generalized_count[row_ids] += 1
            elif kind == "suppressed":
                suppressed_count[row_ids] += 1
            elif kind == "privjedai_fuzzy":
                fuzzy_count[row_ids] += 1
                if score is not None:
                    fuzzy_score_sum[row_ids] += float(score)

        full_compatible_mask &= attr_positive_mask

    compatible_df = working_df_str.loc[full_compatible_mask].copy()
    compatible_df["matched_attr_count"] = compatible_count[full_compatible_mask]
    compatible_df["exact_match_count"] = exact_count[full_compatible_mask]
    compatible_df["generalized_match_count"] = generalized_count[full_compatible_mask]
    compatible_df["suppressed_match_count"] = suppressed_count[full_compatible_mask]
    compatible_df["privjedai_fuzzy_match_count"] = fuzzy_count[full_compatible_mask]
    compatible_df["privjedai_fuzzy_score_sum"] = fuzzy_score_sum[full_compatible_mask]
    fuzzy_score_mean = np.full(int(np.sum(full_compatible_mask)), np.nan, dtype=np.float32)
    np.divide(
        fuzzy_score_sum[full_compatible_mask],
        fuzzy_count[full_compatible_mask],
        out=fuzzy_score_mean,
        where=fuzzy_count[full_compatible_mask] > 0,
    )
    compatible_df["privjedai_fuzzy_score_mean"] = fuzzy_score_mean

    compatible_candidate_count = int(len(compatible_df))
    target_present_in_anonymized = target_id in id_to_eval_index
    true_record_in_compatible = bool(
        target_present_in_anonymized
        and compatible_candidate_count > 0
        and (compatible_df[target_id_col] == target_id).any()
    )
    exact_reidentified = bool(target_present_in_anonymized and compatible_candidate_count == 1 and true_record_in_compatible)
    has_any_fuzzy_candidate = bool(compatible_candidate_count > 0 and (compatible_df["privjedai_fuzzy_match_count"] > 0).any())
    true_record_fuzzy_kept = bool(
        target_present_in_anonymized
        and compatible_candidate_count > 0
        and (
            compatible_df.loc[compatible_df[target_id_col] == target_id, "privjedai_fuzzy_match_count"]
            .fillna(0)
            .astype(int)
            .gt(0)
            .any()
        )
    )

    prediction = summarize_sensitive_prediction(compatible_df, sensitive_attr)
    true_sensitive_value = None
    true_sensitive_probability = None
    if target_present_in_anonymized:
        target_eval_idx = id_to_eval_index[target_id]
        true_sensitive_value = str(working_df_str.iloc[target_eval_idx][sensitive_attr]).strip()
        true_sensitive_probability = prediction["distribution"].get(true_sensitive_value)

    exposed_values = {
        attr: attacker_knowledge[attr].get("projection", {}).get(known_values[attr], known_values[attr])
        for attr in known_attrs
    }

    prefilter_artifacts: dict[str, Any] = {}
    if save_prefilter_debug and prefilter_debug_dir is not None:
        kept_indices = np.flatnonzero(full_compatible_mask)
        kept_debug_df = pd.DataFrame({
            "row_index": kept_indices.astype(int),
            "record_id": working_df_str.loc[full_compatible_mask, target_id_col].astype(str).tolist(),
        })
        kept_debug_path = prefilter_debug_dir / f"target_{target_id}__kept_rows.csv"
        kept_debug_df.to_csv(kept_debug_path, index=False)

        compatible_values_by_attr: dict[str, list[str]] = {}
        compatible_kinds_by_attr: dict[str, dict[str, str]] = {}
        compatible_scores_by_attr: dict[str, dict[str, float]] = {}
        for attr in known_attrs:
            compatible_values_by_attr[attr] = sorted(compatible_values_debug[attr].keys())
            compatible_kinds_by_attr[attr] = {key: str(payload["kind"]) for key, payload in compatible_values_debug[attr].items()}
            compatible_scores_by_attr[attr] = {
                key: round(float(payload["score"]), 6)
                for key, payload in compatible_values_debug[attr].items()
                if payload.get("score") is not None
            }

        debug_payload = {
            "target_id": target_id,
            "known_attrs": known_attrs,
            "known_values": known_values,
            "exposed_values": exposed_values,
            "compatible_values_by_attr": compatible_values_by_attr,
            "compatible_kinds_by_attr": compatible_kinds_by_attr,
            "compatible_scores_by_attr": compatible_scores_by_attr,
            "kept_row_indices": kept_indices.astype(int).tolist(),
            "kept_record_ids": working_df_str.loc[full_compatible_mask, target_id_col].astype(str).tolist(),
            "kept_count": int(len(kept_indices)),
        }
        debug_json_path = prefilter_debug_dir / f"target_{target_id}__filter.json"
        save_json(debug_json_path, debug_payload)
        prefilter_artifacts = {
            "kept_debug_path": kept_debug_path,
            "debug_json_path": debug_json_path,
        }

    per_target_row = {
        "target_id": target_id,
        "target_present_in_anonymized": target_present_in_anonymized,
        "known_attrs": "|".join(known_attrs),
        "known_values": " | ".join(f"{attr}={known_values[attr]}" for attr in known_attrs),
        "exposed_values": " | ".join(f"{attr}={exposed_values[attr]}" for attr in known_attrs),
        "equivalence_class_size": compatible_candidate_count,
        "equivalence_class_candidates_with_privjedai_fuzzy": int((compatible_df["privjedai_fuzzy_match_count"] > 0).sum()) if compatible_candidate_count > 0 else 0,
        "true_record_in_equivalence_class": true_record_in_compatible,
        "true_record_used_privjedai_fuzzy": true_record_fuzzy_kept,
        "exact_reidentified": exact_reidentified,
        "true_sensitive_value": true_sensitive_value,
        "predicted_sensitive_top_value": prediction["top_value"],
        "predicted_sensitive_top_probability": None if prediction["top_probability"] is None else round(float(prediction["top_probability"]), 6),
        "true_sensitive_probability": None if true_sensitive_probability is None else round(float(true_sensitive_probability), 6),
        "sensitive_value_certain": prediction["is_certain"],
        "n_distinct_sensitive_values": prediction["n_distinct_sensitive_values"],
        "sensitive_distribution_json": json.dumps({key: round(float(value), 6) for key, value in prediction["distribution"].items()}, ensure_ascii=False),
    }

    equivalence_class_rows = []
    for _, candidate in compatible_df.iterrows():
        equivalence_class_rows.append(
            {
                "target_id": target_id,
                "candidate_record_id": str(candidate[target_id_col]),
                "candidate_sensitive_value": str(candidate[sensitive_attr]),
                "candidate_matched_attr_count": int(candidate["matched_attr_count"]),
                "candidate_exact_match_count": int(candidate["exact_match_count"]),
                "candidate_generalized_match_count": int(candidate["generalized_match_count"]),
                "candidate_suppressed_match_count": int(candidate["suppressed_match_count"]),
                "candidate_privjedai_fuzzy_match_count": int(candidate["privjedai_fuzzy_match_count"]),
                "candidate_privjedai_fuzzy_score_sum": None if pd.isna(candidate["privjedai_fuzzy_score_sum"]) else round(float(candidate["privjedai_fuzzy_score_sum"]), 6),
                "candidate_privjedai_fuzzy_score_mean": None if pd.isna(candidate["privjedai_fuzzy_score_mean"]) else round(float(candidate["privjedai_fuzzy_score_mean"]), 6),
                "candidate_is_true_record": str(candidate[target_id_col]) == target_id,
            }
        )

    return {
        "target_id": target_id,
        "compatible_candidate_count": compatible_candidate_count,
        "target_present_in_anonymized": target_present_in_anonymized,
        "true_record_in_compatible": true_record_in_compatible,
        "exact_reidentified": exact_reidentified,
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

    output_root = ensure_dir(Path(output_root).resolve())
    attack_root = ensure_dir(output_root / "attacks" / "linkage")
    inferred_anonymized_path = Path(anonymized_path).resolve() if anonymized_path else Path("in_memory_public.csv")
    attack_id = name or make_attack_id(inferred_anonymized_path, known_attrs, n_targets_label, seed, use_privjedai_fuzzy=use_privjedai_fuzzy)
    attack_dir = ensure_dir(attack_root / attack_id)
    prefilter_debug_dir = ensure_dir(attack_dir / "prefilter_debug") if save_prefilter_debug else None

    match_cache: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    fuzzy_pair_cache: dict[tuple[str, str], float] = {}
    fuzzy_hash_cache: dict[str, frozenset[int]] = {}

    per_target_rows: list[dict[str, Any]] = []
    equivalence_class_rows: list[dict[str, Any]] = []
    target_presence_flags: list[bool] = []
    compatible_candidate_counts: list[int] = []
    true_in_compatible_flags: list[bool] = []
    unique_exact_flags: list[bool] = []
    certainty_sensitive_flags: list[bool] = []
    true_sensitive_probabilities: list[float] = []
    top_sensitive_probabilities: list[float] = []
    targets_with_any_fuzzy_candidate: list[bool] = []
    targets_with_true_record_fuzzy_kept: list[bool] = []

    attr_unique_values = {attr: sorted({str(v).strip() for v in working_df_str[attr].astype(str).tolist()}) for attr in known_attrs}
    value_indices = build_value_indices(working_df_str, known_attrs)

    for _, target in sampled_targets.iterrows():
        result = _evaluate_target(
            target=target,
            target_id_col=target_id_col,
            sensitive_attr=sensitive_attr,
            known_attrs=known_attrs,
            attacker_knowledge=attacker_knowledge,
            working_df_str=working_df_str,
            value_indices=value_indices,
            attr_unique_values=attr_unique_values,
            match_cache=match_cache,
            fuzzy_config=fuzzy_config,
            fuzzy_pair_cache=fuzzy_pair_cache,
            fuzzy_hash_cache=fuzzy_hash_cache,
            save_prefilter_debug=save_prefilter_debug,
            prefilter_debug_dir=prefilter_debug_dir,
            id_to_eval_index=id_to_eval_index,
        )
        compatible_candidate_counts.append(result["compatible_candidate_count"])
        target_presence_flags.append(result["target_present_in_anonymized"])
        true_in_compatible_flags.append(result["true_record_in_compatible"])
        unique_exact_flags.append(result["exact_reidentified"])
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

    summary = {
        "attack_id": attack_id,
        "config_path": str(config_path) if config_path else "",
        "auxiliary_path": str(auxiliary_path) if auxiliary_path else "",
        "anonymized_public_path": str(anonymized_path) if anonymized_path else "",
        "anonymized_eval_path": str(anonymized_eval_path) if anonymized_eval_path else "",
        "known_attrs": known_attrs,
        "target_id_col": target_id_col,
        "sensitive_attr": sensitive_attr,
        "n_targets": int(resolved_n_targets),
        "n_targets_requested": n_targets_label,
        "seed": seed,
        "n_anonymized_rows": int(len(working_df_str)),
        "attacker_knowledge_json": str(attacker_knowledge_path),
        "prefilter_debug_dir": str(prefilter_debug_dir) if prefilter_debug_dir is not None else None,
        "attacker_visible_levels": {attr: int(attacker_knowledge[attr]["visible_level"]) for attr in known_attrs},
        "use_privjedai_fuzzy": bool(use_privjedai_fuzzy),
        "privjedai_fuzzy_threshold": None if fuzzy_config is None else float(fuzzy_config["threshold"]),
        "privjedai_fuzzy_metric": None if fuzzy_config is None else str(fuzzy_config["metric"]),
        "privjedai_bloom_size": None if fuzzy_config is None else int(fuzzy_config["bloom_size"]),
        "privjedai_bloom_num_hashes": None if fuzzy_config is None else int(fuzzy_config["num_hashes"]),
        "privjedai_bloom_qgrams": None if fuzzy_config is None else int(fuzzy_config["qgrams"]),
        "privjedai_bloom_hashing_type": None if fuzzy_config is None else str(fuzzy_config["hashing_type"]),
        "target_survival_rate": round(sum(target_presence_flags) / len(target_presence_flags), 6),
        "nonempty_equivalence_class_rate": round(sum(count > 0 for count in compatible_candidate_counts) / len(compatible_candidate_counts), 6),
        "true_record_in_equivalence_class_rate": round(sum(true_in_compatible_flags) / len(true_in_compatible_flags), 6),
        "unique_exact_reidentification_rate": round(sum(unique_exact_flags) / len(unique_exact_flags), 6),
        "avg_equivalence_class_size": round(float(np.mean(compatible_candidate_counts)), 6),
        "median_equivalence_class_size": round(float(np.median(compatible_candidate_counts)), 6),
        "max_equivalence_class_size": int(np.max(compatible_candidate_counts)) if compatible_candidate_counts else 0,
        "targets_with_any_privjedai_fuzzy_candidate_rate": round(sum(targets_with_any_fuzzy_candidate) / len(targets_with_any_fuzzy_candidate), 6),
        "targets_with_true_record_kept_by_privjedai_fuzzy_rate": round(sum(targets_with_true_record_fuzzy_kept) / len(targets_with_true_record_fuzzy_kept), 6),
        "certainty_sensitive_inference_rate": None if certainty_sensitive_rate is None else round(certainty_sensitive_rate, 6),
        "avg_true_sensitive_probability": None if avg_true_sensitive_probability is None else round(avg_true_sensitive_probability, 6),
        "median_true_sensitive_probability": None if median_true_sensitive_probability is None else round(median_true_sensitive_probability, 6),
        "avg_top_sensitive_probability": None if avg_top_sensitive_probability is None else round(avg_top_sensitive_probability, 6),
        "targets_csv": str(per_target_path),
        "equivalence_class_candidates_csv": str(equivalence_class_path),
    }
    save_json(summary_path, summary)

    append_attack_summary(
        attack_root / "linkage_summary.csv",
        {
            "attack_id": attack_id,
            "config_path": str(config_path) if config_path else "",
            "anonymized_public_path": str(anonymized_path) if anonymized_path else "",
            "anonymized_eval_path": str(anonymized_eval_path) if anonymized_eval_path else "",
            "auxiliary_path": str(auxiliary_path) if auxiliary_path else "",
            "known_attrs": "|".join(known_attrs),
            "sensitive_attr": sensitive_attr,
            "n_targets": resolved_n_targets,
            "n_targets_requested": n_targets_label,
            "seed": seed,
            "attacker_visible_levels": "|".join(f"{attr}:{attacker_knowledge[attr]['visible_level']}" for attr in known_attrs),
            "use_privjedai_fuzzy": bool(use_privjedai_fuzzy),
            "privjedai_fuzzy_threshold": None if fuzzy_config is None else float(fuzzy_config["threshold"]),
            "privjedai_fuzzy_metric": None if fuzzy_config is None else str(fuzzy_config["metric"]),
            "target_survival_rate": summary["target_survival_rate"],
            "nonempty_equivalence_class_rate": summary["nonempty_equivalence_class_rate"],
            "true_record_in_equivalence_class_rate": summary["true_record_in_equivalence_class_rate"],
            "unique_exact_reidentification_rate": summary["unique_exact_reidentification_rate"],
            "avg_equivalence_class_size": summary["avg_equivalence_class_size"],
            "targets_with_any_privjedai_fuzzy_candidate_rate": summary["targets_with_any_privjedai_fuzzy_candidate_rate"],
            "targets_with_true_record_kept_by_privjedai_fuzzy_rate": summary["targets_with_true_record_kept_by_privjedai_fuzzy_rate"],
            "certainty_sensitive_inference_rate": summary["certainty_sensitive_inference_rate"],
            "avg_true_sensitive_probability": summary["avg_true_sensitive_probability"],
            "avg_top_sensitive_probability": summary["avg_top_sensitive_probability"],
            "summary_json": str(summary_path),
        },
    )

    print(f"Attack summary            : {summary_path}")
    print(f"Attacker knowledge        : {attacker_knowledge_path}")
    print(f"Per-target results        : {per_target_path}")
    print(f"Equivalence class records : {equivalence_class_path}")
    print(f"Avg equivalence size      : {summary['avg_equivalence_class_size']}")
    print(f"Unique exact rate         : {summary['unique_exact_reidentification_rate']}")
    print(f"Avg true sensitive prob   : {summary['avg_true_sensitive_probability']}")
    if use_privjedai_fuzzy:
        print(f"Targets with fuzzy cand.  : {summary['targets_with_any_privjedai_fuzzy_candidate_rate']}")
        print(f"True records rescued fuzz.: {summary['targets_with_true_record_kept_by_privjedai_fuzzy_rate']}")

    return {
        "summary": summary,
        "summary_path": summary_path,
        "attacker_knowledge_path": attacker_knowledge_path,
        "targets_path": per_target_path,
        "equivalence_class_path": equivalence_class_path,
        "per_target_rows": per_target_rows,
        "equivalence_class_rows": equivalence_class_rows,
        "attacker_knowledge": attacker_knowledge,
        "prefilter_debug_dir": prefilter_debug_dir,
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
    )


# Parse CLI arguments and launch the linkage attack.
def main() -> None:
    parser = argparse.ArgumentParser(description="Run a linkage attack against one anonymized dataset.")
    parser.add_argument("--config", required=True, help="Generated runtime config JSON from outputs/configs/...")
    parser.add_argument("--auxiliary", required=True, help="Auxiliary attacker knowledge CSV.")
    parser.add_argument("--anonymized", required=True, help="Public anonymized CSV (without record_id ideally).")
    parser.add_argument("--anonymized-eval", required=True, help="Internal anonymized CSV with record_id kept for evaluation.")
    parser.add_argument(
        "--known-attrs",
        "--known-qids",
        dest="known_attrs_raw",
        required=True,
        help="Comma-separated list of attributes known by the attacker.",
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
            "Enable an optional privJedAI fallback only for clear attributes: "
            "if the normal ARX-aware match fails on an exact-value attribute, try a privJedAI similarity check."
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
    )


if __name__ == "__main__":
    main()
