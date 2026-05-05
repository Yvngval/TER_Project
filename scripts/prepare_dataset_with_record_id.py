# Prepare a dataset with a stable internal record identifier for anonymization and evaluation.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from common import ensure_dir, load_json, save_json


# Build the default output path for the full dataset with record_id.
def default_full_output(original_path: Path) -> Path:
    return original_path.with_name(f"{original_path.stem}_with_record_id.csv")


# Ensure the dataset contains a unique internal record identifier column.
def ensure_record_id(df: pd.DataFrame, target_id_col: str) -> pd.DataFrame:
    df = df.copy()
    if target_id_col not in df.columns:
        df.insert(0, target_id_col, [str(i) for i in range(len(df))])
    else:
        df[target_id_col] = df[target_id_col].astype(str)
        if df[target_id_col].duplicated().any():
            raise ValueError(f"Column '{target_id_col}' already exists but is not unique.")
    return df


# Create an updated config copy that points to the dataset with record_id.
def update_config_copy(
    base_config_path: Path,
    config_output_path: Path,
    full_dataset_path: Path,
    target_id_col: str,
) -> None:
    payload = load_json(base_config_path)
    payload["data"] = str(full_dataset_path)

    insensitive = list(payload.get("insensitive_attributes", []))
    if target_id_col not in insensitive:
        insensitive.append(target_id_col)
    payload["insensitive_attributes"] = insensitive

    identifiers = [col for col in payload.get("identifiers", []) if col != target_id_col]
    payload["identifiers"] = identifiers

    save_json(config_output_path, payload)


# Prepare the full dataset with record_id and optional updated config copy.
def prepare_dataset_with_record_id(
    *,
    original_path: str | Path,
    target_id_col: str = "record_id",
    output_root: str | Path = "outputs",
    full_output: str | Path | None = None,
    base_config: str | Path | None = None,
    config_output: str | Path | None = None,
) -> dict[str, Any]:
    original_path = Path(original_path).resolve()
    output_root = ensure_dir(Path(output_root).resolve())

    df_original = pd.read_csv(original_path, dtype=str, keep_default_na=False)
    df_full = ensure_record_id(df_original, target_id_col)

    full_output_path = Path(full_output).resolve() if full_output else default_full_output(original_path)
    full_output_path.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(full_output_path, index=False)

    metadata = {
        "original_path": str(original_path),
        "full_dataset_with_record_id": str(full_output_path),
        "target_id_col": target_id_col,
        "n_rows": int(len(df_full)),
        "n_columns_original": int(len(df_original.columns)),
        "n_columns_with_record_id": int(len(df_full.columns)),
    }
    metadata_path = full_output_path.with_suffix(".json")
    save_json(metadata_path, metadata)

    config_output_path = None
    if base_config:
        base_config_path = Path(base_config).resolve()
        if config_output:
            config_output_path = Path(config_output).resolve()
        else:
            config_output_path = output_root / "configs" / f"{base_config_path.stem}_with_record_id.json"
        config_output_path.parent.mkdir(parents=True, exist_ok=True)
        update_config_copy(
            base_config_path=base_config_path,
            config_output_path=config_output_path,
            full_dataset_path=full_output_path,
            target_id_col=target_id_col,
        )

    return {
        "full_output_path": full_output_path,
        "metadata_path": metadata_path,
        "config_output_path": config_output_path,
        "metadata": metadata,
    }


# Parse CLI arguments and prepare the dataset with record_id.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a full CSV dataset with a stable internal record_id column for anonymization and evaluation."
    )
    parser.add_argument("--original", required=True, help="Path to the original CSV dataset.")
    parser.add_argument("--target-id-col", default="record_id", help="Name of the internal record identifier column.")
    parser.add_argument("--output-root", default="outputs", help="Root directory used for optional generated artifacts.")
    parser.add_argument("--full-output", default=None, help="Optional path for the CSV dataset with record_id added.")
    parser.add_argument(
        "--base-config",
        default=None,
        help="Optional anonymization base config to clone and update with the dataset-with-record_id path.",
    )
    parser.add_argument(
        "--config-output",
        default=None,
        help="Where to save the updated config copy. Only used with --base-config.",
    )
    args = parser.parse_args()

    result = prepare_dataset_with_record_id(
        original_path=args.original,
        target_id_col=args.target_id_col,
        output_root=args.output_root,
        full_output=args.full_output,
        base_config=args.base_config,
        config_output=args.config_output,
    )

    if result["config_output_path"] is not None:
        print(f"Updated config copy : {result['config_output_path']}")
    print(f"Full dataset        : {result['full_output_path']}")
    print(f"Metadata            : {result['metadata_path']}")
    print(f"Rows                : {result['metadata']['n_rows']}")


if __name__ == "__main__":
    main()
