# Run a single anonymization experiment and save its outputs, metrics, and summary.

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path
from typing import Any


def _ensure_recitals_import_path() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    candidates = [
        project_root / "RECITALS-anonymization-manager" / "src",
        project_root.parent / "RECITALS-anonymization-manager" / "src",
    ]
    for candidate in candidates:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


_ensure_recitals_import_path()

from anonymization_manager import AnonymizationConfig, AnonymizationManager

from common import (
    build_hierarchy_mapping,
    collect_result_metrics,
    ensure_dir,
    load_json,
    make_experiment_id,
    sanitize_row_for_csv,
    save_json,
    timestamp,
    to_jsonable,
)


# Parse a comma-separated string into a list of values.
def parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


# Build the runtime config with resolved data and hierarchy paths.
def build_runtime_config(config_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    runtime = dict(payload)

    project_root = Path(__file__).resolve().parent.parent

    data_path = Path(payload["data"])
    if data_path.is_absolute():
        runtime["data"] = str(data_path)
    else:
        runtime["data"] = str((project_root / data_path).resolve())

    runtime["hierarchies"] = build_hierarchy_mapping(
        base_dir=project_root,
        hierarchy_dir=payload["hierarchy_dir"],
        quasi_identifiers=payload["quasi_identifiers"],
    )

    runtime.pop("hierarchy_dir", None)
    return runtime


# Append one experiment result row to the benchmark summary CSV.
def append_row_to_summary(summary_csv: Path, row: dict[str, Any]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    sanitized_row = sanitize_row_for_csv(row)

    if not summary_csv.exists():
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(sanitized_row.keys()))
            writer.writeheader()
            writer.writerow(sanitized_row)
        return

    # Read existing header and extend it with any new columns.
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        existing_header = next(reader, [])
        existing_rows = list(reader)

    new_cols = [k for k in sanitized_row if k not in existing_header]
    if new_cols:
        # Rewrite entire file with the extended schema.
        fieldnames = existing_header + new_cols
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in existing_rows:
                writer.writerow(dict(zip(existing_header, r)))
            writer.writerow(sanitized_row)
    else:
        with summary_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=existing_header, extrasaction="ignore"
            )
            writer.writerow(sanitized_row)


# Create the initial summary row for a new experiment.
def _make_initial_row(
    experiment_id: str, runtime_config_path: Path, runtime: dict[str, Any]
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "status": "started",
        "timestamp": timestamp(),
        "config_path": str(runtime_config_path),
        "quasi_identifiers": "|".join(runtime["quasi_identifiers"]),
        "k": runtime.get("k"),
        "l": runtime.get("l"),
        "t": runtime.get("t"),
        "suppression_limit": runtime.get("suppression_limit"),
        "utility_measure": runtime.get("utility_measure", "loss"),
        "utility_aggregate": runtime.get("utility_aggregate"),
        "backend": runtime.get("backend"),
        "csv_path": "",
        "eval_csv_path": "",
        "metrics_path": "",
        "error": "",
    }


# Run the anonymization process from a runtime config.
def _run_anonymization(runtime: dict[str, Any]):
    config = AnonymizationConfig(**runtime)
    return AnonymizationManager.anonymize(config)


# Remove records whose quasi-identifiers are all fully suppressed as '*'.
def drop_fully_suppressed_records(anonymized_df, quasi_identifiers: list[str]):
    missing_qis = [col for col in quasi_identifiers if col not in anonymized_df.columns]
    if missing_qis:
        raise ValueError(
            "Cannot drop fully suppressed records because these quasi-identifiers are missing "
            f"from the anonymized dataframe: {missing_qis}"
        )

    if not quasi_identifiers:
        return anonymized_df.copy(), {
            "fully_suppressed_drop_enabled": True,
            "fully_suppressed_qis": [],
            "n_rows_before_full_suppression_drop": int(len(anonymized_df)),
            "n_rows_dropped_full_suppression": 0,
            "n_rows_after_full_suppression_drop": int(len(anonymized_df)),
        }

    qi_df = anonymized_df[quasi_identifiers].astype(str).apply(lambda col: col.str.strip())
    fully_suppressed_mask = qi_df.eq("*").all(axis=1)
    filtered_df = anonymized_df.loc[~fully_suppressed_mask].copy().reset_index(drop=True)

    stats = {
        "fully_suppressed_drop_enabled": True,
        "fully_suppressed_qis": list(quasi_identifiers),
        "n_rows_before_full_suppression_drop": int(len(anonymized_df)),
        "n_rows_dropped_full_suppression": int(fully_suppressed_mask.sum()),
        "n_rows_after_full_suppression_drop": int(len(filtered_df)),
    }
    return filtered_df, stats


# Execute one anonymization experiment and save all generated artifacts.
def run_one_experiment(
    *,
    runtime: dict[str, Any],
    experiment_id: str,
    output_root: str | Path,
    save_anonymized_csv: bool = True,
    save_anonymized_eval_csv: bool = False,
    public_drop_columns: list[str] | None = None,
    append_summary: bool = True,
    drop_fully_suppressed_records_from_exports: bool = True,
) -> dict[str, Any]:
    public_drop_columns = list(public_drop_columns or [])

    output_root = ensure_dir(output_root)
    metrics_dir = ensure_dir(output_root / "metrics")
    configs_dir = ensure_dir(output_root / "configs")
    anonymized_dir = ensure_dir(output_root / "anonymized")
    anonymized_eval_dir = ensure_dir(output_root / "anonymized_eval")
    errors_dir = ensure_dir(output_root / "errors")
    summary_csv = output_root / "benchmark_summary.csv"

    runtime_config_path = configs_dir / f"{experiment_id}.json"
    save_json(runtime_config_path, runtime)

    row = _make_initial_row(experiment_id, runtime_config_path, runtime)
    details: dict[str, Any] = {
        "experiment_id": experiment_id,
        "row": row,
        "runtime": runtime,
        "runtime_config_path": runtime_config_path,
        "metrics_path": None,
        "public_csv_path": None,
        "eval_csv_path": None,
        "public_drop_columns": public_drop_columns,
        "anonymized_df": None,
        "export_df": None,
        "public_df": None,
        "error_path": None,
    }

    try:
        result = _run_anonymization(runtime)

        anonymized_df = None
        export_df = None
        public_df = None
        public_csv_path = anonymized_dir / f"{experiment_id}.csv"
        eval_csv_path = anonymized_eval_dir / f"{experiment_id}.csv"
        export_drop_stats = {
            "fully_suppressed_drop_enabled": bool(drop_fully_suppressed_records_from_exports),
            "fully_suppressed_qis": list(runtime.get("quasi_identifiers", [])),
            "n_rows_before_full_suppression_drop": None,
            "n_rows_dropped_full_suppression": 0,
            "n_rows_after_full_suppression_drop": None,
        }

        if save_anonymized_csv or save_anonymized_eval_csv:
            anonymized_df = result.get_anonymized_data_as_dataframe()

            if drop_fully_suppressed_records_from_exports:
                export_df, export_drop_stats = drop_fully_suppressed_records(
                    anonymized_df=anonymized_df,
                    quasi_identifiers=list(runtime.get("quasi_identifiers", [])),
                )
            else:
                export_df = anonymized_df.copy()
                export_drop_stats["n_rows_before_full_suppression_drop"] = int(len(anonymized_df))
                export_drop_stats["n_rows_after_full_suppression_drop"] = int(len(export_df))

            if save_anonymized_eval_csv:
                export_df.to_csv(eval_csv_path, index=False)
                row["eval_csv_path"] = str(eval_csv_path)
                details["eval_csv_path"] = eval_csv_path

            if save_anonymized_csv:
                public_df = export_df.drop(
                    columns=[col for col in public_drop_columns if col in export_df.columns],
                    errors="ignore",
                )
                public_df.to_csv(public_csv_path, index=False)
                row["csv_path"] = str(public_csv_path)
                details["public_csv_path"] = public_csv_path

        metrics = collect_result_metrics(result, runtime.get("quasi_identifiers", []))
        metrics = to_jsonable(metrics)
        metrics.update(
            {
                "experiment_id": experiment_id,
                "status": "success",
                "public_drop_columns": public_drop_columns,
                "public_csv_path": row["csv_path"],
                "eval_csv_path": row["eval_csv_path"],
                **export_drop_stats,
            }
        )

        metrics_path = metrics_dir / f"{experiment_id}.json"
        save_json(metrics_path, metrics)

        row["status"] = "success"
        row["metrics_path"] = str(metrics_path)
        row.update(sanitize_row_for_csv(metrics))

        details["metrics"] = metrics
        details["metrics_path"] = metrics_path
        details["anonymized_df"] = anonymized_df
        details["export_df"] = export_df
        details["public_df"] = public_df

        print(f"[OK] {experiment_id}")
        print(f"Config        : {runtime_config_path}")
        print(f"Metrics       : {metrics_path}")
        if export_drop_stats["fully_suppressed_drop_enabled"]:
            print(
                "Dropped fully suppressed QI rows : "
                f"{export_drop_stats['n_rows_dropped_full_suppression']}"
            )
        if row["csv_path"]:
            print(f"Public CSV    : {public_csv_path}")
        if row["eval_csv_path"]:
            print(f"Eval CSV      : {eval_csv_path}")

    except Exception as exc:  # noqa: BLE001
        error_txt = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        error_path = errors_dir / f"{experiment_id}.txt"
        error_path.write_text(error_txt, encoding="utf-8")

        row["status"] = "failed"
        row["error"] = str(exc)
        details["error_path"] = error_path

        print(f"[FAILED] {experiment_id}")
        print(error_txt)

    if append_summary:
        append_row_to_summary(summary_csv, row)

    return details


# Load a config file and run one experiment from it.
def run_one_experiment_from_config(
    *,
    config_path: str | Path,
    output_root: str | Path,
    name: str | None = None,
    save_anonymized_csv: bool = True,
    save_anonymized_eval_csv: bool = False,
    public_drop_columns: list[str] | None = None,
    append_summary: bool = True,
    drop_fully_suppressed_records_from_exports: bool = True,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    payload = load_json(config_path)
    runtime = build_runtime_config(config_path, payload)

    experiment_id = name or make_experiment_id(
        quasi_identifiers=runtime["quasi_identifiers"],
        k=runtime.get("k"),
        l=runtime.get("l"),
        t=runtime.get("t"),
        suppression_limit=runtime.get("suppression_limit"),
        backend=runtime.get("backend"),
        utility_measure=runtime.get("utility_measure"),
        utility_aggregate=runtime.get("utility_aggregate"),
    )

    return run_one_experiment(
        runtime=runtime,
        experiment_id=experiment_id,
        output_root=output_root,
        save_anonymized_csv=save_anonymized_csv,
        save_anonymized_eval_csv=save_anonymized_eval_csv,
        public_drop_columns=public_drop_columns,
        append_summary=append_summary,
        drop_fully_suppressed_records_from_exports=drop_fully_suppressed_records_from_exports,
    )


# Parse CLI arguments and launch one experiment.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one anonymization experiment with RECITALS."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the JSON experiment config."
    )
    parser.add_argument(
        "--output-root", default="outputs", help="Directory where outputs are stored."
    )
    parser.add_argument(
        "--name", default=None, help="Optional experiment name override."
    )
    parser.add_argument(
        "--save-anonymized-csv",
        action="store_true",
        help="Save the public anonymized CSV.",
    )
    parser.add_argument(
        "--save-anonymized-eval-csv",
        action="store_true",
        help="Save an internal anonymized CSV with evaluation columns still present.",
    )
    parser.add_argument(
        "--public-drop-columns",
        default="",
        help="Comma-separated columns dropped only from the public CSV (example: record_id).",
    )
    parser.add_argument(
        "--keep-fully-suppressed-records",
        action="store_true",
        help=(
            "Keep rows whose quasi-identifiers are all '*' in the exported anonymized CSV files. "
            "By default these rows are removed from both anonymized.csv and anonymized_eval.csv."
        ),
    )
    args = parser.parse_args()

    run_one_experiment_from_config(
        config_path=args.config,
        output_root=args.output_root,
        name=args.name,
        save_anonymized_csv=args.save_anonymized_csv,
        save_anonymized_eval_csv=args.save_anonymized_eval_csv,
        public_drop_columns=parse_csv_list(args.public_drop_columns),
        append_summary=True,
        drop_fully_suppressed_records_from_exports=not args.keep_fully_suppressed_records,
    )


if __name__ == "__main__":
    main()
