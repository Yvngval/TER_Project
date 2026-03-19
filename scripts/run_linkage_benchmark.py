from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_benchmark import run_benchmark_grid
from run_linkage_attack import load_runtime_config, read_csv_str, run_linkage_attack
from common import ensure_dir, iter_qi_subsets, load_json, save_json


def parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def ensure_record_id_dataset(original_csv: Path, target_id_col: str, output_csv: Path) -> Path:
    df = pd.read_csv(original_csv, dtype=str, keep_default_na=False)
    df = df.copy()
    if target_id_col not in df.columns:
        df.insert(0, target_id_col, [str(i) for i in range(len(df))])
    else:
        df[target_id_col] = df[target_id_col].astype(str)
        if df[target_id_col].duplicated().any():
            raise ValueError(f"Column '{target_id_col}' already exists but is not unique.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


def create_auxiliary_base_from_df(
    *,
    full_df: pd.DataFrame,
    known_qids: list[str],
    target_id_col: str,
    output_csv: Path,
    sample_size: int | None,
    sample_frac: float | None,
    seed: int,
) -> Path:
    df = full_df.copy()

    if sample_size is not None and sample_frac is not None:
        raise ValueError("Use either sample_size or sample_frac, not both.")

    if sample_frac is not None:
        if not 0 < sample_frac <= 1:
            raise ValueError("sample_frac must be in (0, 1].")
        sample_size = max(1, int(round(len(df) * sample_frac)))

    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be > 0.")
        if sample_size > len(df):
            raise ValueError(f"sample_size ({sample_size}) is larger than dataset size ({len(df)}).")
        df = df.sample(n=sample_size, random_state=seed, replace=False)

    keep_cols = [target_id_col] + known_qids
    missing = [col for col in keep_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for auxiliary base: {missing}")

    aux_df = df.loc[:, keep_cols].sort_values(by=target_id_col).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    aux_df.to_csv(output_csv, index=False)
    return output_csv


def write_runtime_base_config(base_config: dict[str, Any], full_dataset_csv: Path, target_id_col: str, output_json: Path) -> Path:
    payload = dict(base_config)
    payload["data"] = str(full_dataset_csv)

    insensitive = list(payload.get("insensitive_attributes", []))
    if target_id_col not in insensitive:
        insensitive.append(target_id_col)
    payload["insensitive_attributes"] = insensitive

    identifiers = [col for col in payload.get("identifiers", []) if col != target_id_col]
    payload["identifiers"] = identifiers

    save_json(output_json, payload)
    return output_json


def write_runtime_benchmark_grid(benchmark_grid: dict[str, Any], base_config_json: Path, output_json: Path) -> Path:
    payload = dict(benchmark_grid)
    payload["base_config"] = str(Path("configs") / base_config_json.name)
    save_json(output_json, payload)
    return output_json


def _normalize_benchmark_row(row: dict[str, str]) -> dict[str, str]:
    out = {str(k): ("" if v is None else str(v)) for k, v in row.items() if k is not None}

    if not out.get("csv_path") and out.get("public_csv_path"):
        out["csv_path"] = out["public_csv_path"]

    if not out.get("eval_csv_path") and out.get("csv_path"):
        csv_path = out["csv_path"]
        if "anonymized\\" in csv_path:
            out["eval_csv_path"] = csv_path.replace("anonymized\\", "anonymized_eval\\")
        elif "anonymized/" in csv_path:
            out["eval_csv_path"] = csv_path.replace("anonymized/", "anonymized_eval/")

    return out


def _row_quality(row: dict[str, str]) -> tuple[int, int]:
    score = 0
    for key in ["status", "config_path", "csv_path", "eval_csv_path", "metrics_path"]:
        if row.get(key):
            score += 1
    return score, len(row)


def read_benchmark_rows(summary_csv: Path) -> list[dict[str, str]]:
    if not summary_csv.exists():
        return []

    rows: list[dict[str, str]] = []

    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return []

        legacy_header = list(header)

        extended_header = list(legacy_header)
        if "csv_path" in extended_header and "eval_csv_path" not in extended_header and "metrics_path" in extended_header:
            metrics_idx = extended_header.index("metrics_path")
            extended_header.insert(metrics_idx, "eval_csv_path")
        if "public_drop_columns" not in extended_header:
            extended_header.append("public_drop_columns")
        if "public_csv_path" not in extended_header:
            extended_header.append("public_csv_path")

        for raw in reader:
            if not raw:
                continue

            if len(raw) == len(legacy_header):
                parsed = dict(zip(legacy_header, raw))
            elif len(raw) == len(extended_header):
                parsed = dict(zip(extended_header, raw))
            elif len(raw) > len(extended_header):
                extra_names = [f"__extra_{i}" for i in range(len(raw) - len(extended_header))]
                parsed = dict(zip(extended_header + extra_names, raw))
            else:
                padded = raw + [""] * (len(legacy_header) - len(raw))
                parsed = dict(zip(legacy_header, padded))

            rows.append(_normalize_benchmark_row(parsed))

    best_by_experiment: dict[str, dict[str, str]] = {}
    for row in rows:
        exp_id = row.get("experiment_id", "")
        if not exp_id:
            continue
        current = best_by_experiment.get(exp_id)
        if current is None or _row_quality(row) > _row_quality(current):
            best_by_experiment[exp_id] = row

    return list(best_by_experiment.values())


def make_aux_name(known_qids: list[str], sample_size: int | None, sample_frac: float | None) -> str:
    q = "-".join(known_qids)
    if sample_size is not None:
        s = f"n_{sample_size}"
    else:
        s = f"frac_{str(sample_frac).replace('.', '_')}"
    return f"aux__known_{q}__{s}"


def make_attack_name(experiment_id: str, known_qids: list[str], n_targets: int, seed: int) -> str:
    return f"{experiment_id}__atk_{'-'.join(known_qids)}__targets_{n_targets}__seed_{seed}"


def run_linkage_benchmark(
    *,
    grid_path: str | Path,
    output_root: str | Path,
    skip_anonymization: bool = False,
    skip_existing_attacks: bool = False,
) -> dict[str, Any]:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    output_root = ensure_dir(Path(output_root).resolve())

    grid_path = Path(grid_path).resolve()
    linkage_grid = load_json(grid_path)

    benchmark_grid_path = (project_root / linkage_grid["anonymization_grid"]).resolve()
    benchmark_grid = load_json(benchmark_grid_path)

    base_config_rel = linkage_grid.get("base_config") or benchmark_grid["base_config"]
    base_config_path = (project_root / base_config_rel).resolve()
    base_config = load_json(base_config_path)

    original_dataset_path = (project_root / base_config["data"]).resolve()
    target_id_col = linkage_grid.get("target_id_col", "record_id")
    sensitive_attr = linkage_grid.get("sensitive_attr")
    n_targets = int(linkage_grid.get("n_targets", 500))
    seed = int(linkage_grid.get("seed", 42))
    sample_size = linkage_grid.get("sample_size")
    sample_frac = linkage_grid.get("sample_frac")
    stop_on_error = bool(linkage_grid.get("stop_on_error", False))

    attacker_qi_pool = linkage_grid.get("attacker_qi_pool") or benchmark_grid.get("qi_pool") or base_config.get("quasi_identifiers")
    if not attacker_qi_pool:
        raise ValueError("No attacker QI pool found in linkage grid, anonymization grid, or base config.")

    known_qid_subsets = linkage_grid.get("known_qid_subsets")
    if known_qid_subsets is None:
        known_qid_sizes = linkage_grid.get("known_qid_sizes")
        if not known_qid_sizes:
            raise ValueError("Provide either known_qid_subsets or known_qid_sizes in linkage grid.")
        known_qid_subsets = iter_qi_subsets(attacker_qi_pool, known_qid_sizes)

    prepared_data_dir = ensure_dir(output_root / "prepared_data")
    prepared_configs_dir = ensure_dir(output_root / "configs")
    auxiliary_dir = ensure_dir(output_root / "auxiliary")

    full_dataset_csv = prepared_data_dir / f"{original_dataset_path.stem}_with_{target_id_col}.csv"
    ensure_record_id_dataset(original_dataset_path, target_id_col, full_dataset_csv)

    runtime_base_config_path = prepared_configs_dir / "base_config_with_record_id.json"
    write_runtime_base_config(base_config, full_dataset_csv, target_id_col, runtime_base_config_path)

    runtime_benchmark_grid_path = prepared_configs_dir / "benchmark_grid_with_record_id.json"
    write_runtime_benchmark_grid(benchmark_grid, runtime_base_config_path, runtime_benchmark_grid_path)

    inproc_results_by_experiment: dict[str, dict[str, Any]] = {}
    if not skip_anonymization:
        benchmark_run = run_benchmark_grid(
            grid_path=runtime_benchmark_grid_path,
            output_root=output_root,
            save_anonymized_eval_csv=True,
            public_drop_columns=[target_id_col],
        )
        for result in benchmark_run["results"]:
            experiment_id = result.get("experiment_id")
            row = result.get("row", {})
            if experiment_id and row.get("status") == "success":
                inproc_results_by_experiment[experiment_id] = result

    benchmark_summary_csv = output_root / "benchmark_summary.csv"
    benchmark_rows = read_benchmark_rows(benchmark_summary_csv)
    success_rows = [
        row for row in benchmark_rows
        if row.get("status") == "success" and row.get("config_path") and row.get("csv_path") and row.get("eval_csv_path")
    ]
    if not success_rows:
        raise ValueError(
            f"No successful anonymization experiments with public/eval CSVs found in {benchmark_summary_csv}."
        )

    aux_manifest_rows: list[dict[str, Any]] = []
    aux_by_key: dict[tuple[str, ...], Path] = {}
    aux_df_by_key: dict[tuple[str, ...], pd.DataFrame] = {}
    full_df = pd.read_csv(full_dataset_csv, dtype=str, keep_default_na=False)
    full_columns = set(full_df.columns)
    for known_qids in known_qid_subsets:
        missing = [qi for qi in known_qids if qi not in full_columns]
        if missing:
            raise ValueError(f"Unknown attacker QIs in original dataset: {missing}")
        if sensitive_attr and sensitive_attr in known_qids:
            raise ValueError(f"Sensitive attribute '{sensitive_attr}' cannot be part of attacker known_qids.")
        aux_name = make_aux_name(known_qids, sample_size, sample_frac)
        aux_path = auxiliary_dir / f"{aux_name}.csv"
        create_auxiliary_base_from_df(
            full_df=full_df,
            known_qids=known_qids,
            target_id_col=target_id_col,
            output_csv=aux_path,
            sample_size=sample_size,
            sample_frac=sample_frac,
            seed=seed,
        )
        aux_df = pd.read_csv(aux_path, dtype=str, keep_default_na=False)
        aux_by_key[tuple(known_qids)] = aux_path
        aux_df_by_key[tuple(known_qids)] = aux_df
        aux_manifest_rows.append(
            {
                "aux_name": aux_name,
                "known_qids": "|".join(known_qids),
                "auxiliary_path": str(aux_path),
                "sample_size": sample_size,
                "sample_frac": sample_frac,
                "seed": seed,
            }
        )

    aux_manifest_path = output_root / "auxiliary" / "auxiliary_manifest.csv"
    pd.DataFrame(aux_manifest_rows).to_csv(aux_manifest_path, index=False)

    linkage_pairs_rows: list[dict[str, Any]] = []
    launched = 0
    skipped = 0
    for row in success_rows:
        experiment_id = row["experiment_id"]
        experiment_qids = [q for q in row.get("quasi_identifiers", "").split("|") if q]
        if not experiment_qids:
            continue
        experiment_qid_set = set(experiment_qids)

        inproc_result = inproc_results_by_experiment.get(experiment_id)
        runtime = None
        public_df = None
        eval_df = None
        if inproc_result is not None:
            runtime = inproc_result.get("runtime")
            public_df = inproc_result.get("public_df")
            eval_df = inproc_result.get("anonymized_df")

        if runtime is None:
            runtime = load_runtime_config(Path(row["config_path"]))

        for known_qids in known_qid_subsets:
            if not set(known_qids).issubset(experiment_qid_set):
                continue

            aux_path = aux_by_key[tuple(known_qids)]
            aux_df = aux_df_by_key[tuple(known_qids)]
            attack_name = make_attack_name(experiment_id, known_qids, n_targets, seed)
            attack_dir = output_root / "attacks" / "linkage" / attack_name
            attack_summary_path = attack_dir / "summary.json"

            linkage_pairs_rows.append(
                {
                    "experiment_id": experiment_id,
                    "experiment_qids": "|".join(experiment_qids),
                    "known_qids": "|".join(known_qids),
                    "auxiliary_path": str(aux_path),
                    "attack_name": attack_name,
                    "attack_summary_json": str(attack_summary_path),
                }
            )

            if skip_existing_attacks and attack_summary_path.exists():
                skipped += 1
                continue

            if public_df is None or eval_df is None:
                public_df = read_csv_str(Path(row["csv_path"]))
                eval_df = read_csv_str(Path(row["eval_csv_path"]))

            print("=" * 100)
            print(f"Running linkage attack: {attack_name}")
            run_linkage_attack(
                runtime=runtime,
                df_aux=aux_df,
                df_public=public_df,
                df_eval=eval_df,
                known_qids=known_qids,
                target_id_col=target_id_col,
                sensitive_attr=sensitive_attr,
                n_targets=n_targets,
                seed=seed,
                output_root=output_root,
                name=attack_name,
                config_path=row["config_path"],
                auxiliary_path=aux_path,
                anonymized_path=row["csv_path"],
                anonymized_eval_path=row["eval_csv_path"],
            )
            launched += 1

    linkage_plan_path = output_root / "attacks" / "linkage" / "linkage_plan.csv"
    ensure_dir(linkage_plan_path.parent)
    pd.DataFrame(linkage_pairs_rows).to_csv(linkage_plan_path, index=False)

    summary = {
        "runtime_base_config": str(runtime_base_config_path),
        "runtime_benchmark_grid": str(runtime_benchmark_grid_path),
        "full_dataset_with_record_id": str(full_dataset_csv),
        "auxiliary_manifest_csv": str(aux_manifest_path),
        "linkage_plan_csv": str(linkage_plan_path),
        "n_successful_anonymization_experiments": len(success_rows),
        "n_auxiliary_bases": len(aux_manifest_rows),
        "n_attack_pairs": len(linkage_pairs_rows),
        "n_attacks_launched_now": launched,
        "n_attacks_skipped_existing": skipped,
        "output_root": str(output_root),
    }
    summary_path = output_root / "attacks" / "linkage" / "linkage_benchmark_run_summary.json"
    save_json(summary_path, summary)

    print("=" * 100)
    print(f"Prepared dataset           : {full_dataset_csv}")
    print(f"Runtime base config        : {runtime_base_config_path}")
    print(f"Runtime anonymization grid : {runtime_benchmark_grid_path}")
    print(f"Auxiliary manifest         : {aux_manifest_path}")
    print(f"Linkage plan               : {linkage_plan_path}")
    print(f"Run summary                : {summary_path}")
    print(f"Successful anon experiments: {len(success_rows)}")
    print(f"Auxiliary bases            : {len(aux_manifest_rows)}")
    print(f"Attack pairs               : {len(linkage_pairs_rows)}")
    print(f"Launched now               : {launched}")
    print(f"Skipped existing           : {skipped}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full linkage-attack benchmark grid.")
    parser.add_argument("--grid", default="configs/linkage_benchmark_grid.json", help="Path to the linkage benchmark grid JSON.")
    parser.add_argument("--output-root", default="outputs", help="Root output directory.")
    parser.add_argument("--skip-anonymization", action="store_true", help="Do not run anonymization benchmark again; reuse existing outputs.")
    parser.add_argument("--skip-existing-attacks", action="store_true", help="Skip attacks whose summary.json already exists.")
    args = parser.parse_args()

    run_linkage_benchmark(
        grid_path=args.grid,
        output_root=args.output_root,
        skip_anonymization=args.skip_anonymization,
        skip_existing_attacks=args.skip_existing_attacks,
    )


if __name__ == "__main__":
    main()
