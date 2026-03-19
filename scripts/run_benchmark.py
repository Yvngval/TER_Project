from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import ensure_dir, iter_qi_subsets, load_json, make_experiment_id, save_json
from run_one_experiment import run_one_experiment_from_config


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def resolve_existing_path(raw_path: str, *, candidates: list[Path]) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()

    for base in candidates:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate

    return (candidates[0] / path).resolve()


def build_experiment_payload(base_config: dict[str, Any], qi_subset: list[str], k, l, t, suppression_limit, backend):
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


def run_benchmark_grid(
    *,
    grid_path: str | Path,
    output_root: str | Path,
    save_anonymized_eval_csv: bool = False,
    public_drop_columns: list[str] | None = None,
) -> dict[str, Any]:
    grid_path = Path(grid_path).resolve()
    output_root = Path(output_root).resolve()
    public_drop_columns = list(public_drop_columns or [])

    grid = load_json(grid_path)
    base_config_path = resolve_existing_path(
        grid["base_config"],
        candidates=[PROJECT_ROOT, grid_path.parent, grid_path.parent.parent, Path.cwd()],
    )
    base_config = load_json(base_config_path)

    qi_subsets = grid.get("qi_subsets")
    if qi_subsets is None:
        qi_subsets = iter_qi_subsets(grid["qi_pool"], grid["qi_subset_sizes"])

    generated_configs_dir = ensure_dir(output_root / "generated_configs")
    stop_on_error = bool(grid.get("stop_on_error", False))
    save_anonymized_csv = bool(grid.get("save_anonymized_csv", True))

    count = 0
    failures = 0
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

                        print("=" * 100)
                        print(f"Running: {experiment_id}")

                        result = run_one_experiment_from_config(
                            config_path=config_path,
                            output_root=output_root,
                            save_anonymized_csv=save_anonymized_csv,
                            save_anonymized_eval_csv=save_anonymized_eval_csv,
                            public_drop_columns=public_drop_columns,
                            append_summary=True,
                        )
                        count += 1
                        results.append(result)

                        if result["row"].get("status") != "success":
                            failures += 1
                            if stop_on_error:
                                raise SystemExit(1)

    print("=" * 100)
    print(f"Benchmark finished. Total experiments launched: {count}")

    return {
        "grid_path": str(grid_path),
        "output_root": str(output_root),
        "n_experiments": count,
        "n_failures": failures,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full anonymization benchmark grid.")
    parser.add_argument("--grid", default="configs/benchmark_grid.json", help="Path to benchmark grid JSON.")
    parser.add_argument("--output-root", default="outputs", help="Root directory for outputs.")
    parser.add_argument(
        "--save-anonymized-eval-csv",
        action="store_true",
        help="Also save internal anonymized CSV files with evaluation columns preserved.",
    )
    parser.add_argument(
        "--public-drop-columns",
        default="",
        help="Comma-separated columns dropped only from the public anonymized CSV.",
    )
    args = parser.parse_args()

    run_benchmark_grid(
        grid_path=args.grid,
        output_root=args.output_root,
        save_anonymized_eval_csv=args.save_anonymized_eval_csv,
        public_drop_columns=parse_csv_list(args.public_drop_columns),
    )


if __name__ == "__main__":
    main()
