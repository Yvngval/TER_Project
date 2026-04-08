# Create an attacker auxiliary dataset independent from the anonymization QI choice.
# The attacker can know any published attribute except the sensitive attribute.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from common import ensure_dir, load_json, parse_csv_list, save_json


# Build the default output path for the full dataset with record_id.
def default_full_output(original_path: Path) -> Path:
    return original_path.with_name(f"{original_path.stem}_with_record_id.csv")


# Build the default output path for the auxiliary dataset.
def default_aux_output(output_root: Path, original_path: Path, known_attrs: list[str], n_rows: int) -> Path:
    attr_part = "-".join(known_attrs)
    aux_dir = ensure_dir(output_root / "auxiliary")
    return aux_dir / f"{original_path.stem}__aux__known_{attr_part}__n_{n_rows}.csv"


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


# Sample a subset of rows from the dataset for the auxiliary base.
def sample_dataframe(df: pd.DataFrame, sample_size: int | None, sample_frac: float | None, seed: int) -> pd.DataFrame:
    if sample_size is not None and sample_frac is not None:
        raise ValueError("Use either --sample-size or --sample-frac, not both.")

    if sample_size is None and sample_frac is None:
        return df.copy()

    if sample_frac is not None:
        if not 0 < sample_frac <= 1:
            raise ValueError("--sample-frac must be in (0, 1].")
        sample_size = max(1, int(round(len(df) * sample_frac)))

    assert sample_size is not None
    if sample_size <= 0:
        raise ValueError("--sample-size must be > 0.")
    if sample_size > len(df):
        raise ValueError(f"--sample-size ({sample_size}) is larger than the dataset size ({len(df)}).")

    return df.sample(n=sample_size, random_state=seed, replace=False).sort_values(by=df.columns[0]).reset_index(drop=True)


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


# Build the auxiliary dataset and its metadata.
def build_auxiliary_base(
    *,
    original_path: str | Path,
    known_attrs: list[str],
    target_id_col: str = "record_id",
    sample_size: int | None = None,
    sample_frac: float | None = None,
    seed: int = 42,
    sensitive_attr: str | None = None,
    output_root: str | Path = "outputs",
    full_output: str | Path | None = None,
    aux_output: str | Path | None = None,
    base_config: str | Path | None = None,
    config_output: str | Path | None = None,
) -> dict[str, Any]:
    original_path = Path(original_path).resolve()
    output_root = ensure_dir(Path(output_root).resolve())
    if not known_attrs:
        raise ValueError("known_attrs cannot be empty.")

    df_original = pd.read_csv(original_path, dtype=str, keep_default_na=False)
    df_full = ensure_record_id(df_original, target_id_col)

    missing_attrs = [attr for attr in known_attrs if attr not in df_full.columns]
    if missing_attrs:
        raise ValueError(f"Unknown attacker-known attributes: {missing_attrs}")

    if sensitive_attr and sensitive_attr in known_attrs:
        raise ValueError(f"The sensitive attribute '{sensitive_attr}' cannot be part of known_attrs.")

    full_output_path = Path(full_output).resolve() if full_output else default_full_output(original_path)
    full_output_path.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(full_output_path, index=False)

    df_aux = sample_dataframe(df_full, sample_size, sample_frac, seed)
    aux_columns = [target_id_col] + known_attrs
    df_aux = df_aux[aux_columns].reset_index(drop=True)

    aux_output_path = Path(aux_output).resolve() if aux_output else default_aux_output(output_root, original_path, known_attrs, len(df_aux))
    aux_output_path.parent.mkdir(parents=True, exist_ok=True)
    df_aux.to_csv(aux_output_path, index=False)

    metadata = {
        "original_path": str(original_path),
        "full_dataset_with_record_id": str(full_output_path),
        "auxiliary_path": str(aux_output_path),
        "target_id_col": target_id_col,
        "known_attrs": known_attrs,
        "n_original_rows": int(len(df_full)),
        "n_auxiliary_rows": int(len(df_aux)),
        "sample_size": int(len(df_aux)),
        "sample_frac": None if len(df_full) == 0 else len(df_aux) / len(df_full),
        "seed": seed,
        "sensitive_attr": sensitive_attr,
    }
    metadata_path = aux_output_path.with_suffix(".json")
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
        "aux_output_path": aux_output_path,
        "metadata_path": metadata_path,
        "config_output_path": config_output_path,
        "metadata": metadata,
    }


# Parse CLI arguments and generate the auxiliary datasets and metadata.
def main() -> None:
    parser = argparse.ArgumentParser(description="Create an external auxiliary dataset for linkage attack evaluation.")
    parser.add_argument("--original", required=True, help="Path to the original CSV dataset.")
    parser.add_argument(
        "--known-attrs",
        "--known-qids",
        dest="known_attrs_raw",
        required=True,
        help="Comma-separated list of attributes known by the attacker.",
    )
    parser.add_argument("--target-id-col", default="record_id", help="Internal record identifier column added for evaluation.")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of individuals kept in the auxiliary base.")
    parser.add_argument("--sample-frac", type=float, default=None, help="Fraction of individuals kept in the auxiliary base.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for sampling.")
    parser.add_argument("--sensitive-attr", default=None, help="Optional sensitive attribute name, used only for validation.")
    parser.add_argument("--output-root", default="outputs", help="Root directory where auxiliary outputs will be stored.")
    parser.add_argument("--full-output", default=None, help="Optional path for the full dataset with record_id added.")
    parser.add_argument("--aux-output", default=None, help="Optional path for the auxiliary dataset CSV.")
    parser.add_argument("--base-config", default=None, help="Optional anonymization base config to clone and update with record_id dataset path.")
    parser.add_argument("--config-output", default=None, help="Where to save the updated config copy. Only used with --base-config.")
    args = parser.parse_args()

    result = build_auxiliary_base(
        original_path=args.original,
        known_attrs=parse_csv_list(args.known_attrs_raw),
        target_id_col=args.target_id_col,
        sample_size=args.sample_size,
        sample_frac=args.sample_frac,
        seed=args.seed,
        sensitive_attr=args.sensitive_attr,
        output_root=args.output_root,
        full_output=args.full_output,
        aux_output=args.aux_output,
        base_config=args.base_config,
        config_output=args.config_output,
    )

    if result["config_output_path"] is not None:
        print(f"Updated config copy : {result['config_output_path']}")
    print(f"Full dataset        : {result['full_output_path']}")
    print(f"Auxiliary dataset   : {result['aux_output_path']}")
    print(f"Metadata            : {result['metadata_path']}")
    print(f"Rows in auxiliary   : {result['metadata']['n_auxiliary_rows']}")


if __name__ == "__main__":
    main()
