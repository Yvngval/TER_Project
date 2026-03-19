from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from common import ensure_dir, iter_qi_subsets, load_json, make_experiment_id, save_json
from make_mia_targets import ensure_record_id, split_publish_holdout, build_targets_df
from run_mia_attack import run_mia_attack_from_paths


# Run the full MIA benchmark pipeline over anonymization experiments and attacker knowledge settings.


# Parse a comma-separated string into a list of values.
def parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


# Resolve a path by searching through candidate base directories.
def resolve_existing_path(raw_path: str, *, candidates: list[Path]) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    for base in candidates:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (candidates[0] / path).resolve()


# Build one anonymization experiment config from the base config and benchmark parameters.
def build_experiment_payload(base_config: dict[str, Any], qi_subset: list[str], k: int, l: Any, t: Any, suppression_limit: Any, backend: str) -> dict[str, Any]:
    payload = dict(base_config)
    payload["quasi_identifiers"] = qi_subset
    all_qi = base_config["quasi_identifiers"]
    excluded_qi = [q for q in all_qi if q not in qi_subset]
    payload["insensitive_attributes"] = base_config["insensitive_attributes"] + excluded_qi
    payload["k"] = k
    payload["l"] = l
    payload["t"] = t
    payload["suppression_limit"] = suppression_limit
    payload["backend"] = backend
    return payload


# Run all anonymization experiments defined by the runtime benchmark grid.
def run_anonymization_grid(
    *,
    grid_path: str | Path,
    output_root: str | Path,
    save_anonymized_eval_csv: bool,
    public_drop_columns: list[str],
) -> dict[str, Any]:
    grid_path = Path(grid_path).resolve()
    output_root = Path(output_root).resolve()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    run_one_experiment_path = (script_dir / "run_one_experiment.py").resolve()

    grid = load_json(grid_path)
    base_config_path = resolve_existing_path(
        grid["base_config"],
        candidates=[project_root, grid_path.parent, grid_path.parent.parent, Path.cwd()],
    )
    base_config = load_json(base_config_path)

    qi_subsets = grid.get("qi_subsets")
    if qi_subsets is None:
        qi_subsets = iter_qi_subsets(grid["qi_pool"], grid["qi_subset_sizes"])

    generated_configs_dir = ensure_dir(output_root / "generated_configs")
    stop_on_error = bool(grid.get("stop_on_error", False))
    save_anonymized_csv = bool(grid.get("save_anonymized_csv", True))

    launched = 0
    results: list[dict[str, Any]] = []
    for qi_subset in qi_subsets:
        for k in grid["k_values"]:
            for l in grid["l_values"]:
                for t in grid["t_values"]:
                    for suppression_limit in grid["suppression_limits"]:
                        backend = grid["backend"]
                        payload = build_experiment_payload(
                            base_config=base_config,
                            qi_subset=qi_subset,
                            k=k,
                            l=l,
                            t=t,
                            suppression_limit=suppression_limit,
                            backend=backend,
                        )
                        experiment_id = make_experiment_id(qi_subset, k, l, t, suppression_limit, backend)
                        config_path = generated_configs_dir / f"{experiment_id}.json"
                        save_json(config_path, payload)

                        cmd = [
                            sys.executable,
                            str(run_one_experiment_path),
                            "--config",
                            str(config_path),
                            "--output-root",
                            str(output_root),
                        ]
                        if save_anonymized_csv:
                            cmd.append("--save-anonymized-csv")
                        if save_anonymized_eval_csv:
                            cmd.append("--save-anonymized-eval-csv")
                        if public_drop_columns:
                            cmd.extend(["--public-drop-columns", ",".join(public_drop_columns)])

                        print("=" * 100)
                        print(f"Running: {experiment_id}")
                        print(" ".join(cmd))
                        completed = subprocess.run(cmd, check=False)
                        launched += 1
                        results.append({"experiment_id": experiment_id, "returncode": completed.returncode, "config_path": str(config_path)})
                        if completed.returncode != 0 and stop_on_error:
                            raise SystemExit(completed.returncode)

    print("=" * 100)
    print(f"Benchmark finished. Total experiments launched: {launched}")
    return {"launched": launched, "results": results}


# Write a base config copy that points to the published subset.
def write_runtime_base_config(base_config: dict[str, Any], published_dataset_csv: str | Path, output_json: str | Path) -> dict[str, Any]:
    payload = dict(base_config)
    payload["data"] = str(Path(published_dataset_csv).resolve())
    save_json(output_json, payload)
    return payload


# Write a benchmark grid copy that uses the runtime base config.
def write_runtime_benchmark_grid(benchmark_grid: dict[str, Any], base_config_json: str | Path, output_json: str | Path) -> dict[str, Any]:
    payload = dict(benchmark_grid)
    payload["base_config"] = str(Path(base_config_json).resolve())
    save_json(output_json, payload)
    return payload


# Normalize one benchmark summary row into a consistent format.
def _normalize_benchmark_row(row: dict[str, str]) -> dict[str, str]:
    normalized = dict(row)
    normalized.setdefault("experiment_id", "")
    normalized.setdefault("status", "")
    normalized.setdefault("config_path", "")
    normalized.setdefault("quasi_identifiers", "")
    normalized.setdefault("csv_path", normalized.get("public_csv_path", ""))
    normalized.setdefault("eval_csv_path", "")
    normalized.setdefault("metrics_path", "")
    normalized.setdefault("error", "")
    if not normalized.get("csv_path") and normalized.get("public_csv_path"):
        normalized["csv_path"] = normalized["public_csv_path"]
    return normalized


# Score how complete one benchmark summary row is.
def _row_quality(row: dict[str, str]) -> int:
    score = 0
    if row.get("status") == "success":
        score += 100
    for key in ["config_path", "csv_path", "eval_csv_path", "metrics_path"]:
        if row.get(key):
            score += 10
    return score


# Read and deduplicate benchmark summary rows from the global CSV.
def read_benchmark_rows(summary_csv: str | Path) -> list[dict[str, str]]:
    summary_csv = Path(summary_csv)
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


# Build a unique name for one balanced MIA targets dataset.
def make_target_set_name(known_qids: list[str], targets_per_class: int, seed: int) -> str:
    return f"mia_targets__known_{'-'.join(known_qids)}__n_{targets_per_class}__seed_{seed}"


# Build a unique name for one MIA run inside the benchmark.
def make_attack_name(experiment_id: str, known_qids: list[str], targets_per_class: int, seed: int) -> str:
    return f"{experiment_id}__mia_{'-'.join(known_qids)}__n_{targets_per_class}__seed_{seed}"


# Run the full MIA benchmark workflow and save all generated outputs.
def run_mia_benchmark(
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
    mia_grid = load_json(grid_path)

    anonymization_grid_path = resolve_existing_path(
        mia_grid["anonymization_grid"],
        candidates=[project_root, grid_path.parent, grid_path.parent.parent, Path.cwd()],
    )
    anonymization_grid = load_json(anonymization_grid_path)

    base_config_ref = mia_grid.get("base_config") or anonymization_grid["base_config"]
    base_config_path = resolve_existing_path(
        base_config_ref,
        candidates=[project_root, anonymization_grid_path.parent, anonymization_grid_path.parent.parent, grid_path.parent, Path.cwd()],
    )
    base_config = load_json(base_config_path)

    original_dataset_path = resolve_existing_path(
        base_config["data"],
        candidates=[project_root, base_config_path.parent, Path.cwd()],
    )
    target_id_col = mia_grid.get("target_id_col", "record_id")
    member_col = mia_grid.get("member_col", "is_member")
    publish_size = mia_grid.get("publish_size")
    publish_frac = mia_grid.get("publish_frac")
    targets_per_class = int(mia_grid.get("targets_per_class", 500))
    seed = int(mia_grid.get("seed", 42))
    min_best_score = float(mia_grid.get("min_best_score", 0.5))
    max_compatible_candidates = mia_grid.get("max_compatible_candidates")
    max_compatible_fraction = mia_grid.get("max_compatible_fraction")

    attacker_qi_pool = mia_grid.get("attacker_qi_pool") or anonymization_grid.get("qi_pool") or base_config.get("quasi_identifiers")
    if not attacker_qi_pool:
        raise ValueError("No attacker_qi_pool found in MIA grid, anonymization grid, or base config.")

    known_qid_subsets = mia_grid.get("known_qid_subsets")
    if known_qid_subsets is None:
        known_qid_sizes = mia_grid.get("known_qid_sizes")
        if not known_qid_sizes:
            raise ValueError("Provide either known_qid_subsets or known_qid_sizes in the MIA grid.")
        known_qid_subsets = iter_qi_subsets(attacker_qi_pool, known_qid_sizes)

    prepared_data_dir = ensure_dir(output_root / "prepared_data")
    prepared_configs_dir = ensure_dir(output_root / "configs")
    targets_root = ensure_dir(output_root / "mia_targets")

    original_df = pd.read_csv(original_dataset_path, dtype=str, keep_default_na=False)
    original_df = ensure_record_id(original_df, target_id_col)
    published_df, holdout_df = split_publish_holdout(
        original_df,
        publish_size=publish_size,
        publish_frac=publish_frac,
        seed=seed,
    )

    published_csv = prepared_data_dir / f"{original_dataset_path.stem}_mia_published_with_{target_id_col}.csv"
    holdout_csv = prepared_data_dir / f"{original_dataset_path.stem}_mia_holdout_with_{target_id_col}.csv"
    published_df.to_csv(published_csv, index=False)
    holdout_df.to_csv(holdout_csv, index=False)

    runtime_base_config_path = prepared_configs_dir / "mia_base_config_published.json"
    write_runtime_base_config(base_config, published_csv, runtime_base_config_path)

    runtime_benchmark_grid_path = prepared_configs_dir / "mia_benchmark_grid_runtime.json"
    write_runtime_benchmark_grid(anonymization_grid, runtime_base_config_path, runtime_benchmark_grid_path)

    if not skip_anonymization:
        run_anonymization_grid(
            grid_path=runtime_benchmark_grid_path,
            output_root=output_root,
            save_anonymized_eval_csv=True,
            public_drop_columns=[target_id_col],
        )

    benchmark_summary_csv = output_root / "benchmark_summary.csv"
    benchmark_rows = read_benchmark_rows(benchmark_summary_csv)
    success_rows = [
        row for row in benchmark_rows
        if row.get("status") == "success" and row.get("config_path") and row.get("csv_path") and row.get("eval_csv_path")
    ]
    if not success_rows:
        raise ValueError(f"No successful anonymization experiments with public/eval CSVs found in {benchmark_summary_csv}.")

    targets_manifest_rows: list[dict[str, Any]] = []
    targets_by_key: dict[tuple[str, ...], Path] = {}
    for known_qids in known_qid_subsets:
        target_set_name = make_target_set_name(known_qids, targets_per_class, seed)
        targets_csv = targets_root / f"{target_set_name}.csv"
        targets_df = build_targets_df(
            published_df,
            holdout_df,
            known_qids=known_qids,
            target_id_col=target_id_col,
            member_col=member_col,
            targets_per_class=targets_per_class,
            seed=seed,
        )
        targets_df.to_csv(targets_csv, index=False)
        save_json(targets_csv.with_suffix(".json"), {
            "targets_csv": str(targets_csv),
            "known_qids": known_qids,
            "targets_per_class": targets_per_class,
            "seed": seed,
            "member_col": member_col,
            "target_id_col": target_id_col,
            "published_csv": str(published_csv),
            "holdout_csv": str(holdout_csv),
        })
        targets_by_key[tuple(known_qids)] = targets_csv
        targets_manifest_rows.append({
            "target_set_name": target_set_name,
            "known_qids": "|".join(known_qids),
            "targets_csv": str(targets_csv),
            "targets_per_class": targets_per_class,
            "seed": seed,
        })
    pd.DataFrame(targets_manifest_rows).to_csv(targets_root / "targets_manifest.csv", index=False)

    mia_plan_rows: list[dict[str, Any]] = []
    launched = 0
    skipped = 0
    for row in success_rows:
        experiment_id = row["experiment_id"]
        experiment_qids = [q for q in row.get("quasi_identifiers", "").split("|") if q]
        experiment_qid_set = set(experiment_qids)
        if not experiment_qids:
            continue

        for known_qids in known_qid_subsets:
            if not set(known_qids).issubset(experiment_qid_set):
                continue

            targets_csv = targets_by_key[tuple(known_qids)]
            attack_name = make_attack_name(experiment_id, known_qids, targets_per_class, seed)
            attack_dir = output_root / "attacks" / "mia" / attack_name
            attack_summary_path = attack_dir / "summary.json"

            mia_plan_rows.append({
                "experiment_id": experiment_id,
                "experiment_qids": "|".join(experiment_qids),
                "known_qids": "|".join(known_qids),
                "targets_csv": str(targets_csv),
                "anonymized_csv": row["csv_path"],
                "anonymized_eval_csv": row["eval_csv_path"],
                "attack_name": attack_name,
                "attack_summary_json": str(attack_summary_path),
            })

            if skip_existing_attacks and attack_summary_path.exists():
                skipped += 1
                continue

            print("=" * 100)
            print(f"Running MIA: {attack_name}")
            run_mia_attack_from_paths(
                config_path=row["config_path"],
                targets_path=targets_csv,
                anonymized_path=row["csv_path"],
                anonymized_eval_path=row["eval_csv_path"],
                known_qids=known_qids,
                target_id_col=target_id_col,
                member_col=member_col,
                min_best_score=min_best_score,
                max_compatible_candidates=max_compatible_candidates,
                max_compatible_fraction=max_compatible_fraction,
                output_root=output_root,
                name=attack_name,
                seed=seed,
            )
            launched += 1

    attacks_root = ensure_dir(output_root / "attacks" / "mia")
    pd.DataFrame(mia_plan_rows).to_csv(attacks_root / "mia_plan.csv", index=False)

    summary = {
        "grid_path": str(grid_path),
        "output_root": str(output_root),
        "published_csv": str(published_csv),
        "holdout_csv": str(holdout_csv),
        "benchmark_summary_csv": str(benchmark_summary_csv),
        "targets_manifest_csv": str(targets_root / "targets_manifest.csv"),
        "mia_plan_csv": str(attacks_root / "mia_plan.csv"),
        "n_successful_anonymization_experiments": len(success_rows),
        "n_target_sets": len(known_qid_subsets),
        "n_mia_runs_launched": launched,
        "n_mia_runs_skipped": skipped,
        "seed": seed,
        "targets_per_class": targets_per_class,
        "min_best_score": min_best_score,
        "max_compatible_candidates": max_compatible_candidates,
        "max_compatible_fraction": max_compatible_fraction,
    }
    save_json(attacks_root / "mia_benchmark_run_summary.json", summary)
    return summary


# Parse CLI arguments and launch the MIA benchmark.
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full MIA benchmark pipeline.")
    parser.add_argument("--grid", required=True, help="Path to the MIA benchmark grid JSON.")
    parser.add_argument("--output-root", default="outputs", help="Root directory for generated outputs.")
    parser.add_argument("--skip-anonymization", action="store_true", help="Reuse existing anonymization outputs.")
    parser.add_argument("--skip-existing-attacks", action="store_true", help="Skip MIA runs whose summary.json already exists.")
    args = parser.parse_args()

    summary = run_mia_benchmark(
        grid_path=args.grid,
        output_root=args.output_root,
        skip_anonymization=args.skip_anonymization,
        skip_existing_attacks=args.skip_existing_attacks,
    )
    print("=" * 100)
    print("MIA benchmark finished.")
    print(f"Launched attacks : {summary['n_mia_runs_launched']}")
    print(f"Skipped attacks  : {summary['n_mia_runs_skipped']}")


if __name__ == "__main__":
    main()
