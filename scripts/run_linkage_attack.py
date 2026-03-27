# Run one linkage attack against an anonymized dataset.
# Attacker-known attributes are independent from the anonymization QI choice.
# Optimized with value-indexed candidate filtering and sparse score accumulation.

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


# Parse a comma-separated string into a list of values.
def parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


# Build a unique name for one linkage attack run.
def make_attack_id(anonymized_path: Path, known_attrs: list[str], n_targets: int, seed: int) -> str:
    attr_part = "-".join(known_attrs)
    return f"{anonymized_path.stem}__known_{attr_part}__targets_{n_targets}__seed_{seed}"


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
            "The config passed to run_linkage_attack.py must be a runtime config containing a 'hierarchies' mapping. "
            "Use the generated config from outputs/configs/..., not the initial base_config.json."
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

    # Choose the most specific level that still explains the observed anonymized values.
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


# Build the reduced attacker knowledge for all known attributes.
def build_attacker_knowledge(
    *,
    runtime: dict[str, Any],
    known_attrs: list[str],
    df_public: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    hierarchies = runtime.get("hierarchies", {})
    knowledge: dict[str, dict[str, Any]] = {}
    for attr in known_attrs:
        hierarchy_path = hierarchies.get(attr)
        if hierarchy_path:
            knowledge[attr] = build_attacker_projection_for_attr(
                attr=attr,
                hierarchy_path=hierarchy_path,
                observed_anonymized_values=df_public[attr].astype(str),
            )
        else:
            projection = {str(v).strip(): str(v).strip() for v in df_public[attr].astype(str).unique().tolist()}
            knowledge[attr] = {
                "attribute": attr,
                "hierarchy_path": None,
                "visible_level": 0,
                "observed_values": sorted({str(v).strip() for v in df_public[attr].astype(str).tolist()}),
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


# Compute and cache the score mapping for one target attribute value.
def get_score_mapping_for_target_value(
    attr: str,
    target_value: str,
    possible_anonymized_values: list[str],
    attacker_knowledge: dict[str, dict[str, Any]],
    score_cache: dict[tuple[str, str], dict[str, float]],
) -> dict[str, float]:
    cache_key = (attr, str(target_value).strip())
    if cache_key not in score_cache:
        attacker_attr_knowledge = attacker_knowledge.get(attr)
        mapping = {
            str(anonymized_value).strip(): attribute_score(target_value, anonymized_value, attacker_attr_knowledge)
            for anonymized_value in possible_anonymized_values
        }
        score_cache[cache_key] = mapping
    return score_cache[cache_key]


# Build an inverted index: for each attribute value, store the row indices where it appears.
def build_value_indices(df: pd.DataFrame, attrs: list[str]) -> dict[str, dict[str, np.ndarray]]:
    indices: dict[str, dict[str, np.ndarray]] = {}
    for attr in attrs:
        per_value: dict[str, list[int]] = {}
        for idx, value in enumerate(df[attr].astype(str).tolist()):
            per_value.setdefault(str(value).strip(), []).append(idx)
        indices[attr] = {
            value: np.asarray(row_ids, dtype=np.int32)
            for value, row_ids in per_value.items()
        }
    return indices


# Append one attack result row to the global linkage summary CSV.
def append_attack_summary(summary_csv: Path, row: dict[str, Any]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# Check whether candidate records reveal a unique sensitive value.
def candidate_sensitive_inference(candidate_df: pd.DataFrame, sensitive_attr: str) -> tuple[bool | None, str | None]:
    if candidate_df.empty:
        return None, None
    unique_sensitive = sorted(candidate_df[sensitive_attr].astype(str).unique().tolist())
    if len(unique_sensitive) == 1:
        return True, unique_sensitive[0]
    return False, None


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
        raise ValueError(
            f"Sensitive attribute '{sensitive_attr}' not found in anonymized eval dataset."
        )

    if len(df_public) != len(df_eval):
        raise ValueError(
            "Public and eval anonymized CSV files must have the same number of rows and the same row order."
        )

    if n_targets <= 0:
        raise ValueError("--n-targets must be > 0.")
    if n_targets > len(df_aux):
        raise ValueError(
            f"--n-targets ({n_targets}) is larger than the auxiliary dataset size ({len(df_aux)})."
        )


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
    n_targets: int = 500,
    seed: int = 42,
    output_root: str | Path = "outputs",
    name: str | None = None,
    config_path: str | Path | None = None,
    auxiliary_path: str | Path | None = None,
    anonymized_path: str | Path | None = None,
    anonymized_eval_path: str | Path | None = None,
    save_prefilter_debug: bool = False,
) -> dict[str, Any]:
    sensitive_attr = sensitive_attr or (runtime.get("sensitive_attributes") or [None])[0]
    if not sensitive_attr:
        raise ValueError("No sensitive attribute available. Pass --sensitive-attr explicitly.")

    _validate_inputs(
        known_attrs=known_attrs,
        target_id_col=target_id_col,
        sensitive_attr=sensitive_attr,
        df_aux=df_aux,
        df_public=df_public,
        df_eval=df_eval,
        n_targets=n_targets,
    )

    attacker_knowledge = build_attacker_knowledge(
        runtime=runtime,
        known_attrs=known_attrs,
        df_public=df_public,
    )

    working_df = df_public.copy().reset_index(drop=True)
    working_df[target_id_col] = df_eval[target_id_col].astype(str).reset_index(drop=True)
    if sensitive_attr not in working_df.columns:
        working_df[sensitive_attr] = df_eval[sensitive_attr].astype(str).reset_index(drop=True)

    working_df_str = working_df.copy()
    for attr in known_attrs:
        working_df_str[attr] = working_df[attr].astype(str).str.strip()
    working_df_str[target_id_col] = working_df[target_id_col].astype(str).str.strip()
    working_df_str[sensitive_attr] = working_df[sensitive_attr].astype(str).str.strip()

    sampled_targets = df_aux.sample(n=n_targets, random_state=seed, replace=False).reset_index(drop=True)
    for attr in known_attrs:
        sampled_targets[attr] = sampled_targets[attr].astype(str).str.strip()
    sampled_targets[target_id_col] = sampled_targets[target_id_col].astype(str).str.strip()

    id_to_eval_index = {
        str(record_id): idx for idx, record_id in enumerate(working_df_str[target_id_col].tolist())
    }

    output_root = ensure_dir(Path(output_root).resolve())
    attack_root = ensure_dir(output_root / "attacks" / "linkage")
    inferred_anonymized_path = Path(anonymized_path).resolve() if anonymized_path else Path("in_memory_public.csv")
    attack_id = name or make_attack_id(inferred_anonymized_path, known_attrs, n_targets, seed)
    attack_dir = ensure_dir(attack_root / attack_id)
    prefilter_debug_dir = ensure_dir(attack_dir / "prefilter_debug") if save_prefilter_debug else None

    score_cache: dict[tuple[str, str], dict[str, float]] = {}
    per_target_rows: list[dict[str, Any]] = []
    best_candidates_rows: list[dict[str, Any]] = []

    n_rows = len(working_df_str)
    target_presence_flags: list[bool] = []
    compatible_candidate_counts: list[int] = []
    best_candidate_counts: list[int] = []
    true_in_compatible_flags: list[bool] = []
    true_in_best_flags: list[bool] = []
    unique_exact_flags: list[bool] = []
    compatible_disclosure_values: list[bool] = []
    best_disclosure_values: list[bool] = []
    prefilter_candidate_counts: list[int] = []

    attr_unique_values = {
        attr: sorted({str(v).strip() for v in working_df_str[attr].astype(str).tolist()})
        for attr in known_attrs
    }
    value_indices = build_value_indices(working_df_str, known_attrs)

    for _, target in sampled_targets.iterrows():
        target_id = str(target[target_id_col]).strip()
        known_values = {attr: str(target[attr]).strip() for attr in known_attrs}

        total_score = np.zeros(n_rows, dtype=float)
        compatible_count = np.zeros(n_rows, dtype=np.int16)
        exact_count = np.zeros(n_rows, dtype=np.int16)
        generalized_count = np.zeros(n_rows, dtype=np.int16)
        suppressed_count = np.zeros(n_rows, dtype=np.int16)

        full_compatible_mask = np.ones(n_rows, dtype=bool)

        for attr in known_attrs:
            mapping = get_score_mapping_for_target_value(
                attr=attr,
                target_value=known_values[attr],
                possible_anonymized_values=attr_unique_values[attr],
                attacker_knowledge=attacker_knowledge,
                score_cache=score_cache,
            )

            attr_positive_mask = np.zeros(n_rows, dtype=bool)

            for anonymized_value, score in mapping.items():
                if score <= EPS:
                    continue
                row_ids = value_indices[attr].get(anonymized_value)
                if row_ids is None or len(row_ids) == 0:
                    continue

                attr_positive_mask[row_ids] = True
                total_score[row_ids] += score
                compatible_count[row_ids] += 1

                if score >= 1.0 - EPS:
                    exact_count[row_ids] += 1
                elif np.isclose(score, GENERALIZED_SCORE, atol=EPS):
                    generalized_count[row_ids] += 1
                elif np.isclose(score, SUPPRESSION_SCORE, atol=EPS):
                    suppressed_count[row_ids] += 1

            full_compatible_mask &= attr_positive_mask

        normalized_score = total_score / len(known_attrs)
        prefilter_candidate_count = int(full_compatible_mask.sum())
        prefilter_candidate_counts.append(prefilter_candidate_count)

        best_score = float(np.max(normalized_score)) if len(normalized_score) else 0.0
        if best_score <= EPS:
            best_mask = np.zeros(n_rows, dtype=bool)
        else:
            best_mask = np.isclose(normalized_score, best_score, atol=EPS)

        compatible_df = working_df_str.loc[full_compatible_mask].copy()
        compatible_df["normalized_score"] = normalized_score[full_compatible_mask]
        compatible_df["exact_match_count"] = exact_count[full_compatible_mask]
        compatible_df["generalized_match_count"] = generalized_count[full_compatible_mask]
        compatible_df["suppressed_match_count"] = suppressed_count[full_compatible_mask]

        best_df = working_df_str.loc[best_mask].copy()
        best_df["normalized_score"] = normalized_score[best_mask]
        best_df["exact_match_count"] = exact_count[best_mask]
        best_df["generalized_match_count"] = generalized_count[best_mask]
        best_df["suppressed_match_count"] = suppressed_count[best_mask]
        best_df["is_full_compatible"] = full_compatible_mask[best_mask]

        target_present_in_anonymized = target_id in id_to_eval_index
        target_presence_flags.append(target_present_in_anonymized)

        compatible_candidate_count = int(len(compatible_df))
        best_candidate_count = int(len(best_df))
        compatible_candidate_counts.append(compatible_candidate_count)
        best_candidate_counts.append(best_candidate_count)

        if save_prefilter_debug and prefilter_debug_dir is not None:
            kept_indices = np.flatnonzero(full_compatible_mask)
            kept_debug_df = pd.DataFrame({
                "row_index": kept_indices.astype(int),
                "record_id": working_df_str.loc[full_compatible_mask, target_id_col].astype(str).tolist(),
            })
            kept_debug_path = prefilter_debug_dir / f"target_{target_id}__kept_rows.csv"
            kept_debug_df.to_csv(kept_debug_path, index=False)

            compatible_values_by_attr: dict[str, list[str]] = {}
            for attr in known_attrs:
                mapping = get_score_mapping_for_target_value(
                    attr=attr,
                    target_value=known_values[attr],
                    possible_anonymized_values=attr_unique_values[attr],
                    attacker_knowledge=attacker_knowledge,
                    score_cache=score_cache,
                )
                compatible_values_by_attr[attr] = sorted([
                    value for value, score in mapping.items() if score > EPS
                ])

            debug_payload = {
                "target_id": target_id,
                "known_attrs": known_attrs,
                "known_values": known_values,
                "exposed_values": {
                    attr: attacker_knowledge[attr].get("projection", {}).get(known_values[attr], known_values[attr])
                    for attr in known_attrs
                },
                "compatible_values_by_attr": compatible_values_by_attr,
                "kept_row_indices": kept_indices.astype(int).tolist(),
                "kept_record_ids": working_df_str.loc[full_compatible_mask, target_id_col].astype(str).tolist(),
                "kept_count": int(len(kept_indices)),
            }
            debug_json_path = prefilter_debug_dir / f"target_{target_id}__filter.json"
            save_json(debug_json_path, debug_payload)

        true_record_in_compatible = bool(
            target_present_in_anonymized
            and compatible_candidate_count > 0
            and (compatible_df[target_id_col] == target_id).any()
        )
        true_record_in_best = bool(
            target_present_in_anonymized
            and best_candidate_count > 0
            and (best_df[target_id_col] == target_id).any()
        )
        exact_reidentified = bool(
            target_present_in_anonymized and compatible_candidate_count == 1 and true_record_in_compatible
        )

        true_in_compatible_flags.append(true_record_in_compatible)
        true_in_best_flags.append(true_record_in_best)
        unique_exact_flags.append(exact_reidentified)

        compatible_disclosure, compatible_sensitive_value = candidate_sensitive_inference(compatible_df, sensitive_attr)
        best_disclosure, best_sensitive_value = candidate_sensitive_inference(best_df, sensitive_attr)

        if compatible_disclosure is not None:
            compatible_disclosure_values.append(bool(compatible_disclosure))
        if best_disclosure is not None:
            best_disclosure_values.append(bool(best_disclosure))

        exposed_values = {
            attr: attacker_knowledge[attr].get("projection", {}).get(known_values[attr], known_values[attr])
            for attr in known_attrs
        }

        per_target_rows.append(
            {
                "target_id": target_id,
                "target_present_in_anonymized": target_present_in_anonymized,
                "known_attrs": "|".join(known_attrs),
                "known_values": " | ".join(f"{attr}={known_values[attr]}" for attr in known_attrs),
                "exposed_values": " | ".join(f"{attr}={exposed_values[attr]}" for attr in known_attrs),
                "best_score": round(best_score, 6),
                "prefilter_candidate_count": prefilter_candidate_count,
                "compatible_candidate_count": compatible_candidate_count,
                "best_candidate_count": best_candidate_count,
                "true_record_in_compatible": true_record_in_compatible,
                "true_record_in_best": true_record_in_best,
                "exact_reidentified": exact_reidentified,
                "compatible_sensitive_disclosure": compatible_disclosure,
                "compatible_inferred_sensitive_value": compatible_sensitive_value,
                "best_sensitive_disclosure": best_disclosure,
                "best_inferred_sensitive_value": best_sensitive_value,
            }
        )

        for _, candidate in best_df.iterrows():
            best_candidates_rows.append(
                {
                    "target_id": target_id,
                    "candidate_record_id": str(candidate[target_id_col]),
                    "candidate_sensitive_value": str(candidate[sensitive_attr]),
                    "candidate_score": round(float(candidate["normalized_score"]), 6),
                    "candidate_exact_match_count": int(candidate["exact_match_count"]),
                    "candidate_generalized_match_count": int(candidate["generalized_match_count"]),
                    "candidate_suppressed_match_count": int(candidate["suppressed_match_count"]),
                    "candidate_is_full_compatible": bool(candidate["is_full_compatible"]),
                    "candidate_is_true_record": str(candidate[target_id_col]) == target_id,
                }
            )

    per_target_path = attack_dir / "targets.csv"
    best_candidates_path = attack_dir / "best_candidates.csv"
    attacker_knowledge_path = attack_dir / "attacker_knowledge.json"
    summary_path = attack_dir / "summary.json"

    pd.DataFrame(per_target_rows).to_csv(per_target_path, index=False)
    pd.DataFrame(best_candidates_rows).to_csv(best_candidates_path, index=False)
    save_json(attacker_knowledge_path, attacker_knowledge)

    compatible_disclosure_rate = (
        sum(compatible_disclosure_values) / len(compatible_disclosure_values)
        if compatible_disclosure_values
        else None
    )
    best_disclosure_rate = (
        sum(best_disclosure_values) / len(best_disclosure_values)
        if best_disclosure_values
        else None
    )

    summary = {
        "attack_id": attack_id,
        "config_path": str(config_path) if config_path else "",
        "auxiliary_path": str(auxiliary_path) if auxiliary_path else "",
        "anonymized_public_path": str(anonymized_path) if anonymized_path else "",
        "anonymized_eval_path": str(anonymized_eval_path) if anonymized_eval_path else "",
        "known_attrs": known_attrs,
        "known_qids_legacy": known_attrs,
        "target_id_col": target_id_col,
        "sensitive_attr": sensitive_attr,
        "n_targets": int(n_targets),
        "seed": seed,
        "n_anonymized_rows": int(len(working_df_str)),
        "attacker_knowledge_json": str(attacker_knowledge_path),
        "prefilter_debug_dir": str(prefilter_debug_dir) if prefilter_debug_dir is not None else None,
        "attacker_visible_levels": {attr: int(attacker_knowledge[attr]["visible_level"]) for attr in known_attrs},
        "target_survival_rate": round(sum(target_presence_flags) / len(target_presence_flags), 6),
        "avg_prefilter_candidate_count": round(float(np.mean(prefilter_candidate_counts)), 6),
        "full_compatibility_rate": round(sum(count > 0 for count in compatible_candidate_counts) / len(compatible_candidate_counts), 6),
        "true_record_in_compatible_rate": round(sum(true_in_compatible_flags) / len(true_in_compatible_flags), 6),
        "true_record_in_best_rate": round(sum(true_in_best_flags) / len(true_in_best_flags), 6),
        "unique_exact_reidentification_rate": round(sum(unique_exact_flags) / len(unique_exact_flags), 6),
        "avg_compatible_candidate_count": round(float(np.mean(compatible_candidate_counts)), 6),
        "median_compatible_candidate_count": round(float(np.median(compatible_candidate_counts)), 6),
        "max_compatible_candidate_count": int(np.max(compatible_candidate_counts)),
        "avg_best_candidate_count": round(float(np.mean(best_candidate_counts)), 6),
        "median_best_candidate_count": round(float(np.median(best_candidate_counts)), 6),
        "max_best_candidate_count": int(np.max(best_candidate_counts)),
        "compatible_sensitive_disclosure_rate": None if compatible_disclosure_rate is None else round(compatible_disclosure_rate, 6),
        "best_sensitive_disclosure_rate": None if best_disclosure_rate is None else round(best_disclosure_rate, 6),
        "targets_csv": str(per_target_path),
        "best_candidates_csv": str(best_candidates_path),
    }
    save_json(summary_path, summary)

    attack_summary_csv = attack_root / "linkage_summary.csv"
    append_attack_summary(
        attack_summary_csv,
        {
            "attack_id": attack_id,
            "config_path": str(config_path) if config_path else "",
            "anonymized_public_path": str(anonymized_path) if anonymized_path else "",
            "anonymized_eval_path": str(anonymized_eval_path) if anonymized_eval_path else "",
            "auxiliary_path": str(auxiliary_path) if auxiliary_path else "",
            "known_attrs": "|".join(known_attrs),
            "sensitive_attr": sensitive_attr,
            "n_targets": n_targets,
            "seed": seed,
            "attacker_visible_levels": "|".join(f"{attr}:{attacker_knowledge[attr]['visible_level']}" for attr in known_attrs),
            "avg_prefilter_candidate_count": summary["avg_prefilter_candidate_count"],
            "target_survival_rate": summary["target_survival_rate"],
            "full_compatibility_rate": summary["full_compatibility_rate"],
            "true_record_in_compatible_rate": summary["true_record_in_compatible_rate"],
            "true_record_in_best_rate": summary["true_record_in_best_rate"],
            "unique_exact_reidentification_rate": summary["unique_exact_reidentification_rate"],
            "avg_compatible_candidate_count": summary["avg_compatible_candidate_count"],
            "avg_best_candidate_count": summary["avg_best_candidate_count"],
            "compatible_sensitive_disclosure_rate": summary["compatible_sensitive_disclosure_rate"],
            "best_sensitive_disclosure_rate": summary["best_sensitive_disclosure_rate"],
            "summary_json": str(summary_path),
        },
    )

    print(f"Attack summary      : {summary_path}")
    print(f"Attacker knowledge  : {attacker_knowledge_path}")
    print(f"Per-target results  : {per_target_path}")
    print(f"Best candidates     : {best_candidates_path}")
    print(f"Avg prefilter count : {summary['avg_prefilter_candidate_count']}")
    print(f"Unique exact rate   : {summary['unique_exact_reidentification_rate']}")
    print(f"True record in best : {summary['true_record_in_best_rate']}")

    return {
        "summary": summary,
        "summary_path": summary_path,
        "attacker_knowledge_path": attacker_knowledge_path,
        "targets_path": per_target_path,
        "best_candidates_path": best_candidates_path,
        "per_target_rows": per_target_rows,
        "best_candidates_rows": best_candidates_rows,
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
    n_targets: int = 500,
    seed: int = 42,
    output_root: str | Path = "outputs",
    name: str | None = None,
    save_prefilter_debug: bool = False,
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
    )


# Parse CLI arguments and launch the linkage attack.
def main() -> None:
    parser = argparse.ArgumentParser(description="Run a linkage attack against one anonymized dataset.")
    parser.add_argument("--config", required=True, help="Generated runtime config JSON from outputs/configs/...")
    parser.add_argument("--auxiliary", required=True, help="Auxiliary attacker knowledge CSV.")
    parser.add_argument("--anonymized", required=True, help="Public anonymized CSV (without record_id ideally).")
    parser.add_argument(
        "--anonymized-eval",
        required=True,
        help="Internal anonymized CSV with record_id kept for evaluation.",
    )
    parser.add_argument(
        "--known-attrs",
        "--known-qids",
        dest="known_attrs_raw",
        required=True,
        help="Comma-separated list of attributes known by the attacker.",
    )
    parser.add_argument("--target-id-col", default="record_id", help="Internal record identifier column.")
    parser.add_argument(
        "--sensitive-attr",
        default=None,
        help="Sensitive attribute to inspect. Defaults to the first sensitive attribute in the runtime config.",
    )
    parser.add_argument(
        "--n-targets",
        type=int,
        default=500,
        help="Number of target individuals sampled from the auxiliary base.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for target sampling.")
    parser.add_argument("--output-root", default="outputs", help="Root output directory.")
    parser.add_argument("--name", default=None, help="Optional custom attack name.")
    parser.add_argument(
        "--save-prefilter-debug",
        action="store_true",
        help="Save one debug file per target with kept row indices and record_id after prefiltering.",
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
    )


if __name__ == "__main__":
    main()
