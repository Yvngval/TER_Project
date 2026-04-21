# Delete all generated outputs to restart the benchmark from scratch.

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_DIRS = [
    "anonymized",
    "anonymized_eval",
    "configs",
    "errors",
    "generated_configs",
    "metrics",
    "test_agg",
    "test_run_one",
    "tmp_measure_check",
    "utility",
]

OUTPUT_FILES = [
    "benchmark_summary.csv",
    "benchmark_summary 2.csv",
    "benchmark_summary 3.csv",
    "utility_summary.csv",
]


def clean_outputs(output_root: Path, dry_run: bool = False) -> None:
    print(f"Output root: {output_root}")
    print("=" * 60)

    for name in OUTPUT_DIRS:
        path = output_root / name
        if path.exists():
            print(f"  [DIR]  {path}")
            if not dry_run:
                shutil.rmtree(path)
        else:
            print(f"  [SKIP] {path} (not found)")

    for name in OUTPUT_FILES:
        path = output_root / name
        if path.exists():
            print(f"  [FILE] {path}")
            if not dry_run:
                path.unlink()
        else:
            print(f"  [SKIP] {path} (not found)")

    print("=" * 60)
    if dry_run:
        print("Dry run — nothing was deleted.")
    else:
        print("Done. All outputs have been deleted.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete all generated outputs to restart the benchmark."
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Path to the outputs directory (default: outputs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be deleted without actually deleting anything.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()

    if not output_root.exists():
        print(f"Output root not found: {output_root}")
        return

    if not args.dry_run:
        confirm = input(f"Delete all outputs in {output_root}? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    clean_outputs(output_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
