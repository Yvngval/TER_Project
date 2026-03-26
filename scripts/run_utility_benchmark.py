# Run statistical utility metrics for all successful anonymization experiments.
# Reads benchmark_summary.csv, loads each anonymized CSV, computes metrics,
# and saves per-experiment JSON + a global utility_summary.csv.

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from common import ensure_dir, load_json, save_json
from compute_utility_metrics import compute_utility_metrics


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve(raw: str | Path, base: Path) -> Path:
    """Resolve a path that may be relative to a given base directory."""
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _parse_qi(raw: str) -> list[str]:
    """Parse the pipe-separated quasi_identifiers field from benchmark_summary.csv."""
    return [q.strip() for q in str(raw).split("|") if q.strip()]


def _load_benchmark_summary(summary_path: Path) -> pd.DataFrame:
    """Load benchmark_summary.csv robustly.

    The file may have more columns in data rows than declared in the header
    (extra ARX metrics appended without updating the header). We read with
    the csv module, keep only the header-declared columns, and return a DataFrame.
    """
    with open(summary_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        n = len(headers)
        rows = []
        for line in reader:
            if len(line) >= n:
                rows.append(dict(zip(headers, line[:n])))
            elif len(line) > 0:
                # Pad short rows with empty strings
                padded = line + [""] * (n - len(line))
                rows.append(dict(zip(headers, padded)))
    return pd.DataFrame(rows, columns=headers)


# ---------------------------------------------------------------------------
# Per-experiment logic
# ---------------------------------------------------------------------------

def run_utility_for_experiment(
    row: dict,
    orig_df: pd.DataFrame,
    output_root: Path,
    skip_existing: bool = False,
) -> dict:
    """Compute and save utility metrics for a single experiment row."""
    experiment_id = row["experiment_id"]
    utility_dir = ensure_dir(output_root / "utility")
    out_path = utility_dir / f"{experiment_id}_utility.json"

    if skip_existing and out_path.exists():
        print(f"  [SKIP] {experiment_id}")
        return {"experiment_id": experiment_id, "status": "skipped"}

    csv_path = _resolve(row["csv_path"], PROJECT_ROOT)
    if not csv_path.exists():
        print(f"  [MISSING CSV] {experiment_id}")
        return {"experiment_id": experiment_id, "status": "missing_csv"}

    try:
        anon_df = pd.read_csv(csv_path)

        quasi_identifiers = _parse_qi(row.get("quasi_identifiers", ""))

        # Load the runtime config to get sensitive_attributes
        config_path = _resolve(row["config_path"], PROJECT_ROOT)
        config = load_json(config_path)
        sensitive_attributes = config.get("sensitive_attributes", [])

        metrics = compute_utility_metrics(
            orig_df=orig_df,
            anon_df=anon_df,
            quasi_identifiers=quasi_identifiers,
            sensitive_attributes=sensitive_attributes,
        )

        payload = {
            "experiment_id": experiment_id,
            "status": "success",
            "quasi_identifiers": quasi_identifiers,
            "k": row.get("k"),
            "l": row.get("l"),
            "t": row.get("t"),
            "suppression_limit": row.get("suppression_limit"),
            "backend": row.get("backend"),
            "metrics": metrics,
        }
        save_json(out_path, payload)

        corr = metrics["correlation_delta"]
        summary_row = {
            "experiment_id": experiment_id,
            "status": "success",
            "quasi_identifiers": "|".join(quasi_identifiers),
            "k": row.get("k"),
            "l": row.get("l"),
            "t": row.get("t"),
            "suppression_limit": row.get("suppression_limit"),
            "backend": row.get("backend"),
            "n_records_orig": metrics["n_records_orig"],
            "n_records_anon": metrics["n_records_anon"],
            "suppression_rate": metrics["suppression_rate"],
            "mean_tvd_qi": metrics["mean_tvd_qi"],
            "mean_tvd_sensitive": metrics["mean_tvd_sensitive"],
            "mean_tvd_all": metrics["mean_tvd_all"],
            "mean_kl_qi": metrics["mean_kl_qi"],
            "corr_frobenius": corr["frobenius_norm"],
            "corr_frobenius_normalized": corr["frobenius_norm_normalized"],
            "output_path": str(out_path),
        }
        print(f"  [OK] {experiment_id}  |  TVD_qi={metrics['mean_tvd_qi']:.4f}  KL_qi={metrics['mean_kl_qi']:.4f}")
        return summary_row

    except Exception as exc:
        print(f"  [ERROR] {experiment_id}: {exc}")
        return {"experiment_id": experiment_id, "status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------

def run_utility_benchmark(
    *,
    summary_path: Path,
    orig_data_path: Path,
    output_root: Path,
    skip_existing: bool = False,
) -> None:
    """Run utility metrics for every successful experiment in benchmark_summary.csv."""
    summary_df = _load_benchmark_summary(summary_path)
    successful = summary_df[summary_df["status"] == "success"].copy()

    print(f"Original dataset : {orig_data_path}")
    print(f"Experiments found: {len(successful)} successful / {len(summary_df)} total")

    orig_df = pd.read_csv(orig_data_path)

    results: list[dict] = []
    for _, row in successful.iterrows():
        print(f"\n{row['experiment_id']}")
        result = run_utility_for_experiment(
            row=row.to_dict(),
            orig_df=orig_df,
            output_root=output_root,
            skip_existing=skip_existing,
        )
        results.append(result)

    # Deduplicate: keep last result per experiment_id (in case summary has duplicates)
    seen: dict[str, dict] = {}
    for r in results:
        seen[r["experiment_id"]] = r
    results = list(seen.values())

    # Save utility_summary.csv
    utility_summary_path = output_root / "utility_summary.csv"
    if results:
        all_keys: list[str] = []
        for r in results:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        with open(utility_summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

    n_ok = sum(1 for r in results if r.get("status") == "success")
    n_skip = sum(1 for r in results if r.get("status") == "skipped")
    n_err = sum(1 for r in results if r.get("status") == "error")

    print("\n" + "=" * 60)
    print(f"Utility benchmark done.")
    print(f"  Success : {n_ok}")
    print(f"  Skipped : {n_skip}")
    print(f"  Errors  : {n_err}")
    print(f"Summary saved to: {utility_summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run statistical utility metrics benchmark.")
    parser.add_argument(
        "--summary",
        default="outputs/benchmark_summary.csv",
        help="Path to benchmark_summary.csv (default: outputs/benchmark_summary.csv)",
    )
    parser.add_argument(
        "--original-data",
        default="data/adult.csv",
        help="Path to the original (non-anonymized) dataset (default: data/adult.csv)",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root directory where outputs/utility/ will be created (default: outputs)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments whose utility JSON already exists",
    )
    args = parser.parse_args()

    run_utility_benchmark(
        summary_path=_resolve(args.summary, PROJECT_ROOT),
        orig_data_path=_resolve(args.original_data, PROJECT_ROOT),
        output_root=_resolve(args.output_root, PROJECT_ROOT),
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
