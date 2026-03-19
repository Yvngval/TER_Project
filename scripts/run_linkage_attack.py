from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import ensure_dir, load_json, save_json

EPS = 1e-12


def parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def make_attack_id(anonymized_path: Path, known_qids: list[str], n_targets: int, seed: int) -> str:
    qid_part = "-".join(known_qids)
    return f"{anonymized_path.stem}__known_{qid_part}__targets_{n_targets}__seed_{seed}"


def read_csv_str(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_runtime_config(config_path: Path) -> dict[str, Any]:
    payload = load_json(config_path)
    if "hierarchies" not in payload:
        raise ValueError(
            "The config passed to run_linkage_attack.py must be a runtime config containing a 'hierarchies' mapping. "
            "Use the generated config from outputs/configs/..., not the initial base_config.json."
        )
    return payload


def load_hierarchy_lookup(hierarchy_path: str | Path) -> dict[str, dict[str, int]]:
    lookup: dict[str, dict[str, int]] = {}
    with Path(hierarchy_path).open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            cleaned = [str(cell).strip() for cell in row if str(cell).strip() != ""]
            if not cleaned:
                continue
            exact = cleaned[0]
            lookup[exact] = {value: idx for idx, value in enumerate(cleaned)}
    return lookup


def attribute_score(raw_value: str, anonymized_value: str, hierarchy_lookup: dict[str, dict[str, int]] | None) -> float:
    raw_value = str(raw_value)
    anonymized_value = str(anonymized_value)

    if hierarchy_lookup is None:
        return 1.0 if raw_value == anonymized_value else 0.0

    path_positions = hierarchy_lookup.get(raw_value)
    if path_positions is None:
        return 1.0 if raw_value == anonymized_value else 0.0

    pos = path_positions.get(anonymized_value)
    if pos is None:
        return 0.0

    max_pos = max(path_positions.values()) if path_positions else 0
    if max_pos == 0:
        return 1.0

    return max(0.0, 1.0 - (pos / max_pos))


def score_vector_for_target_value(
    qid: str,
    target_value: str,
    anonymized_values_str: pd.Series,
    hierarchy_lookups: dict[str, dict[str, dict[str, int]]],
    score_cache: dict[tuple[str, str], dict[str, float]],
) -> np.ndarray:
    cache_key = (qid, str(target_value))
    if cache_key not in score_cache:
        hierarchy_lookup = hierarchy_lookups.get(qid)
        mapping = {
            anonymized_value: attribute_score(target_value, anonymized_value, hierarchy_lookup)
            for anonymized_value in anonymized_values_str.unique().tolist()
        }
        score_cache[cache_key] = mapping
    return anonymized_values_str.map(score_cache[cache_key]).astype(float).to_numpy()


def append_attack_summary(summary_csv: Path, row: dict[str, Any]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def candidate_sensitive_inference(candidate_df: pd.DataFrame, sensitive_attr: str) -> tuple[bool | None, str | None]:
    if candidate_df.empty:
        return None, None
    unique_sensitive = sorted(candidate_df[sensitive_attr].astype(str).unique().tolist())
    if len(unique_sensitive) == 1:
        return True, unique_sensitive[0]
    return False, None


def _validate_inputs(
    *,
    runtime: dict[str, Any],
    known_qids: list[str],
    target_id_col: str,
    sensitive_attr: str,
    df_aux: pd.DataFrame,
    df_public: pd.DataFrame,
    df_eval: pd.DataFrame,
    n_targets: int,
) -> None:
    if not known_qids:
        raise ValueError("--known-qids cannot be empty.")

    invalid_qids = [qi for qi in known_qids if qi not in runtime["quasi_identifiers"]]
    if invalid_qids:
        raise ValueError(
            f"These known QIs are not quasi-identifiers in the anonymization config: {invalid_qids}"
        )

    if target_id_col not in df_aux.columns:
        raise ValueError(f"Auxiliary dataset must contain '{target_id_col}'.")
    if target_id_col not in df_eval.columns:
        raise ValueError(
            f"Evaluation anonymized dataset must contain '{target_id_col}'. "
            "Save an internal eval copy with record_id kept."
        )

    missing_aux_cols = [qi for qi in known_qids if qi not in df_aux.columns]
    if missing_aux_cols:
        raise ValueError(f"Auxiliary dataset misses known QI columns: {missing_aux_cols}")

    missing_public_cols = [qi for qi in known_qids if qi not in df_public.columns]
    if missing_public_cols:
        raise ValueError(f"Public anonymized dataset misses known QI columns: {missing_public_cols}")

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


def run_linkage_attack(
    *,
    runtime: dict[str, Any],
    df_aux: pd.DataFrame,
    df_public: pd.DataFrame,
    df_eval: pd.DataFrame,
    known_qids: list[str],
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
) -> dict[str, Any]:
    sensitive_attr = sensitive_attr or (runtime.get("sensitive_attributes") or [None])[0]
    if not sensitive_attr:
        raise ValueError("No sensitive attribute available. Pass --sensitive-attr explicitly.")

    _validate_inputs(
        runtime=runtime,
        known_qids=known_qids,
        target_id_col=target_id_col,
        sensitive_attr=sensitive_attr,
        df_aux=df_aux,
        df_public=df_public,
        df_eval=df_eval,
        n_targets=n_targets,
    )

    hierarchy_lookups = {
        qi: load_hierarchy_lookup(runtime["hierarchies"][qi])
        for qi in known_qids
    }

    working_df = df_public.copy().reset_index(drop=True)
    working_df[target_id_col] = df_eval[target_id_col].astype(str).reset_index(drop=True)
    if sensitive_attr not in working_df.columns:
        working_df[sensitive_attr] = df_eval[sensitive_attr].astype(str).reset_index(drop=True)

    working_df_str = working_df.copy()
    for qi in known_qids:
        working_df_str[qi] = working_df[qi].astype(str)
    working_df_str[target_id_col] = working_df[target_id_col].astype(str)
    working_df_str[sensitive_attr] = working_df[sensitive_attr].astype(str)

    sampled_targets = (
        df_aux.sample(n=n_targets, random_state=seed, replace=False)
        .reset_index(drop=True)
    )
    for qi in known_qids:
        sampled_targets[qi] = sampled_targets[qi].astype(str)
    sampled_targets[target_id_col] = sampled_targets[target_id_col].astype(str)

    id_to_eval_index = {
        str(record_id): idx for idx, record_id in enumerate(working_df_str[target_id_col].tolist())
    }

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

    qid_series = {qi: working_df_str[qi] for qi in known_qids}
    record_id_series = working_df_str[target_id_col]
    sensitive_series = working_df_str[sensitive_attr]

    for _, target in sampled_targets.iterrows():
        target_id = str(target[target_id_col])
        known_values = {qi: str(target[qi]) for qi in known_qids}

        total_score = np.zeros(n_rows, dtype=float)
        compatible_count = np.zeros(n_rows, dtype=int)
        exact_count = np.zeros(n_rows, dtype=int)

        for qi in known_qids:
            qi_scores = score_vector_for_target_value(
                qid=qi,
                target_value=known_values[qi],
                anonymized_values_str=qid_series[qi],
                hierarchy_lookups=hierarchy_lookups,
                score_cache=score_cache,
            )
            total_score += qi_scores
            compatible_count += (qi_scores > EPS).astype(int)
            exact_count += (qi_scores >= 1.0 - EPS).astype(int)

        normalized_score = total_score / len(known_qids)
        full_compatible_mask = compatible_count == len(known_qids)
        best_score = float(np.max(normalized_score)) if len(normalized_score) else 0.0
        best_mask = np.isclose(normalized_score, best_score, atol=EPS)

        compatible_df = working_df_str.loc[full_compatible_mask].copy()
        compatible_df["normalized_score"] = normalized_score[full_compatible_mask]
        compatible_df["exact_match_count"] = exact_count[full_compatible_mask]

        best_df = working_df_str.loc[best_mask].copy()
        best_df["normalized_score"] = normalized_score[best_mask]
        best_df["exact_match_count"] = exact_count[best_mask]
        best_df["is_full_compatible"] = full_compatible_mask[best_mask]

        target_present_in_anonymized = target_id in id_to_eval_index
        target_presence_flags.append(target_present_in_anonymized)

        compatible_candidate_count = int(len(compatible_df))
        best_candidate_count = int(len(best_df))
        compatible_candidate_counts.append(compatible_candidate_count)
        best_candidate_counts.append(best_candidate_count)

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

        compatible_disclosure, compatible_sensitive_value = candidate_sensitive_inference(
            compatible_df, sensitive_attr
        )
        best_disclosure, best_sensitive_value = candidate_sensitive_inference(best_df, sensitive_attr)

        if compatible_disclosure is not None:
            compatible_disclosure_values.append(bool(compatible_disclosure))
        if best_disclosure is not None:
            best_disclosure_values.append(bool(best_disclosure))

        per_target_rows.append(
            {
                "target_id": target_id,
                "target_present_in_anonymized": target_present_in_anonymized,
                "known_qids": "|".join(known_qids),
                "known_values": " | ".join(f"{qi}={known_values[qi]}" for qi in known_qids),
                "best_score": round(best_score, 6),
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
                    "candidate_is_full_compatible": bool(candidate["is_full_compatible"]),
                    "candidate_is_true_record": str(candidate[target_id_col]) == target_id,
                }
            )

    output_root = ensure_dir(Path(output_root).resolve())
    attack_root = ensure_dir(output_root / "attacks" / "linkage")
    inferred_anonymized_path = Path(anonymized_path).resolve() if anonymized_path else Path("in_memory_public.csv")
    attack_id = name or make_attack_id(inferred_anonymized_path, known_qids, n_targets, seed)
    attack_dir = ensure_dir(attack_root / attack_id)

    per_target_path = attack_dir / "targets.csv"
    best_candidates_path = attack_dir / "best_candidates.csv"
    summary_path = attack_dir / "summary.json"

    pd.DataFrame(per_target_rows).to_csv(per_target_path, index=False)
    pd.DataFrame(best_candidates_rows).to_csv(best_candidates_path, index=False)

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
        "known_qids": known_qids,
        "target_id_col": target_id_col,
        "sensitive_attr": sensitive_attr,
        "n_targets": int(n_targets),
        "seed": seed,
        "n_anonymized_rows": int(len(working_df_str)),
        "target_survival_rate": round(sum(target_presence_flags) / len(target_presence_flags), 6),
        "full_compatibility_rate": round(
            sum(count > 0 for count in compatible_candidate_counts) / len(compatible_candidate_counts), 6
        ),
        "true_record_in_compatible_rate": round(
            sum(true_in_compatible_flags) / len(true_in_compatible_flags), 6
        ),
        "true_record_in_best_rate": round(sum(true_in_best_flags) / len(true_in_best_flags), 6),
        "unique_exact_reidentification_rate": round(
            sum(unique_exact_flags) / len(unique_exact_flags), 6
        ),
        "avg_compatible_candidate_count": round(float(np.mean(compatible_candidate_counts)), 6),
        "median_compatible_candidate_count": round(float(np.median(compatible_candidate_counts)), 6),
        "max_compatible_candidate_count": int(np.max(compatible_candidate_counts)),
        "avg_best_candidate_count": round(float(np.mean(best_candidate_counts)), 6),
        "median_best_candidate_count": round(float(np.median(best_candidate_counts)), 6),
        "max_best_candidate_count": int(np.max(best_candidate_counts)),
        "compatible_sensitive_disclosure_rate": (
            None if compatible_disclosure_rate is None else round(compatible_disclosure_rate, 6)
        ),
        "best_sensitive_disclosure_rate": (
            None if best_disclosure_rate is None else round(best_disclosure_rate, 6)
        ),
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
            "known_qids": "|".join(known_qids),
            "sensitive_attr": sensitive_attr,
            "n_targets": n_targets,
            "seed": seed,
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
    print(f"Per-target results  : {per_target_path}")
    print(f"Best candidates     : {best_candidates_path}")
    print(f"Unique exact rate   : {summary['unique_exact_reidentification_rate']}")
    print(f"True record in best : {summary['true_record_in_best_rate']}")

    return {
        "summary": summary,
        "summary_path": summary_path,
        "targets_path": per_target_path,
        "best_candidates_path": best_candidates_path,
        "per_target_rows": per_target_rows,
        "best_candidates_rows": best_candidates_rows,
    }


def run_linkage_attack_from_paths(
    *,
    config_path: str | Path,
    auxiliary_path: str | Path,
    anonymized_path: str | Path,
    anonymized_eval_path: str | Path,
    known_qids: list[str],
    target_id_col: str = "record_id",
    sensitive_attr: str | None = None,
    n_targets: int = 500,
    seed: int = 42,
    output_root: str | Path = "outputs",
    name: str | None = None,
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
        known_qids=known_qids,
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
    )


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
        "--known-qids",
        required=True,
        help="Comma-separated list of QIs known by the attacker.",
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
    args = parser.parse_args()

    run_linkage_attack_from_paths(
        config_path=args.config,
        auxiliary_path=args.auxiliary,
        anonymized_path=args.anonymized,
        anonymized_eval_path=args.anonymized_eval,
        known_qids=parse_csv_list(args.known_qids),
        target_id_col=args.target_id_col,
        sensitive_attr=args.sensitive_attr,
        n_targets=args.n_targets,
        seed=args.seed,
        output_root=args.output_root,
        name=args.name,
    )


if __name__ == "__main__":
    main()
