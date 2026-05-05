from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from common import ensure_dir, parse_csv_list, save_json


# Build the default output path for the auxiliary dataset.
def default_aux_output(
    output_root: Path,
    full_dataset_path: Path,
    known_attrs: list[str],
    n_rows: int,
    released_only: bool,
) -> Path:
    attr_part = "-".join(known_attrs)
    scope_part = "released_only" if released_only else "all_records"
    aux_dir = ensure_dir(output_root / "auxiliary")
    return aux_dir / f"{full_dataset_path.stem}__aux__known_{attr_part}__{scope_part}__n_{n_rows}.csv"


# Ensure the prepared dataset already contains a unique internal record identifier column.
def validate_prepared_dataset(df: pd.DataFrame, target_id_col: str) -> pd.DataFrame:
    df = df.copy()
    if target_id_col not in df.columns:
        raise ValueError(
            f"Column '{target_id_col}' was not found. "
            "Run prepare_dataset_with_record_id.py first or provide a dataset that already contains this column."
        )

    df[target_id_col] = df[target_id_col].astype(str)
    if df[target_id_col].duplicated().any():
        raise ValueError(f"Column '{target_id_col}' exists but is not unique in the prepared dataset.")
    return df


# Sample a subset of rows from the dataset for the auxiliary base.
def sample_dataframe(df: pd.DataFrame, sample_size: int | None, sample_frac: float | None, seed: int) -> pd.DataFrame:
    if sample_size is not None and sample_frac is not None:
        raise ValueError("Use either --sample-size or --sample-frac, not both.")

    if sample_size is None and sample_frac is None:
        return df.copy().sort_values(by=df.columns[0]).reset_index(drop=True)

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


# Read anonymized_eval and keep only record_ids that are still published.
def filter_to_released_records(
    *,
    df_full: pd.DataFrame,
    released_eval_path: str | Path,
    target_id_col: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    released_eval_path = Path(released_eval_path).resolve()
    df_released = pd.read_csv(released_eval_path, dtype=str, keep_default_na=False)

    if target_id_col not in df_released.columns:
        raise ValueError(
            f"Column '{target_id_col}' was not found in released-eval file: {released_eval_path}"
        )

    released_ids = set(df_released[target_id_col].astype(str).tolist())
    if not released_ids:
        raise ValueError(f"No released record ids found in: {released_eval_path}")

    df_filtered = df_full[df_full[target_id_col].astype(str).isin(released_ids)].copy()
    if df_filtered.empty:
        raise ValueError(
            "The intersection between the prepared dataset and released-eval record_ids is empty."
        )

    stats = {
        "released_eval_path": str(released_eval_path),
        "n_released_rows_in_eval": int(len(df_released)),
        "n_unique_released_ids": int(len(released_ids)),
        "n_overlap_rows_with_full_dataset": int(len(df_filtered)),
        "n_missing_released_ids_in_full_dataset": int(len(released_ids) - len(df_filtered)),
    }
    return df_filtered, stats


# Build the auxiliary dataset and its metadata from a dataset already prepared with record_id.
def build_auxiliary_base(
    *,
    full_dataset_path: str | Path,
    known_attrs: list[str],
    target_id_col: str = "record_id",
    sample_size: int | None = None,
    sample_frac: float | None = None,
    seed: int = 42,
    sensitive_attr: str | None = None,
    output_root: str | Path = "outputs",
    aux_output: str | Path | None = None,
    released_eval: str | Path | None = None,
) -> dict[str, Any]:
    full_dataset_path = Path(full_dataset_path).resolve()
    output_root = ensure_dir(Path(output_root).resolve())
    if not known_attrs:
        raise ValueError("known_attrs cannot be empty.")

    df_full = pd.read_csv(full_dataset_path, dtype=str, keep_default_na=False)
    df_full = validate_prepared_dataset(df_full, target_id_col)

    missing_attrs = [attr for attr in known_attrs if attr not in df_full.columns]
    if missing_attrs:
        raise ValueError(f"Unknown attacker-known attributes: {missing_attrs}")

    if sensitive_attr and sensitive_attr in known_attrs:
        raise ValueError(f"The sensitive attribute '{sensitive_attr}' cannot be part of known_attrs.")

    population_df = df_full
    release_stats: dict[str, Any] = {
        "released_eval_path": None,
        "n_released_rows_in_eval": None,
        "n_unique_released_ids": None,
        "n_overlap_rows_with_full_dataset": None,
        "n_missing_released_ids_in_full_dataset": None,
    }
    released_only = released_eval is not None
    if released_eval is not None:
        population_df, release_stats = filter_to_released_records(
            df_full=df_full,
            released_eval_path=released_eval,
            target_id_col=target_id_col,
        )

    df_aux = sample_dataframe(population_df, sample_size, sample_frac, seed)
    aux_columns = [target_id_col] + known_attrs
    df_aux = df_aux[aux_columns].reset_index(drop=True)

    aux_output_path = (
        Path(aux_output).resolve()
        if aux_output
        else default_aux_output(output_root, full_dataset_path, known_attrs, len(df_aux), released_only)
    )
    aux_output_path.parent.mkdir(parents=True, exist_ok=True)
    df_aux.to_csv(aux_output_path, index=False)

    source_population_size = len(population_df)
    metadata = {
        "full_dataset_with_record_id": str(full_dataset_path),
        "auxiliary_path": str(aux_output_path),
        "target_id_col": target_id_col,
        "known_attrs": known_attrs,
        "n_full_dataset_rows": int(len(df_full)),
        "target_population_mode": "released_only" if released_only else "all_records",
        "target_population_size": int(source_population_size),
        "n_auxiliary_rows": int(len(df_aux)),
        "sample_size": int(len(df_aux)),
        "sample_frac_within_target_population": None if source_population_size == 0 else len(df_aux) / source_population_size,
        "sample_frac_within_full_dataset": None if len(df_full) == 0 else len(df_aux) / len(df_full),
        "seed": seed,
        "sensitive_attr": sensitive_attr,
        **release_stats,
    }
    metadata_path = aux_output_path.with_suffix(".json")
    save_json(metadata_path, metadata)

    return {
        "aux_output_path": aux_output_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }


# Parse CLI arguments and generate the auxiliary dataset and metadata.
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create an external auxiliary dataset for linkage attack evaluation from a dataset already prepared "
            "with record_id. Use --released-eval to sample only targets that are still present in anonymized_eval."
        )
    )
    parser.add_argument(
        "--full-dataset",
        "--original",
        dest="full_dataset",
        required=True,
        help="Path to the CSV dataset already containing record_id.",
    )
    parser.add_argument(
        "--known-attrs",
        "--known-qids",
        dest="known_attrs_raw",
        required=True,
        help="Comma-separated list of attributes known by the attacker.",
    )
    parser.add_argument(
        "--target-id-col",
        default="record_id",
        help="Internal record identifier column already present in the prepared dataset.",
    )
    parser.add_argument("--sample-size", type=int, default=None, help="Number of individuals kept in the auxiliary base.")
    parser.add_argument("--sample-frac", type=float, default=None, help="Fraction of individuals kept in the auxiliary base.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for sampling.")
    parser.add_argument("--sensitive-attr", default=None, help="Optional sensitive attribute name, used only for validation.")
    parser.add_argument(
        "--released-eval",
        default=None,
        help="Optional anonymized_eval CSV. If provided, only released record_ids are eligible as linkage targets.",
    )
    parser.add_argument("--output-root", default="outputs", help="Root directory where auxiliary outputs will be stored.")
    parser.add_argument("--aux-output", default=None, help="Optional path for the auxiliary dataset CSV.")
    args = parser.parse_args()

    result = build_auxiliary_base(
        full_dataset_path=args.full_dataset,
        known_attrs=parse_csv_list(args.known_attrs_raw),
        target_id_col=args.target_id_col,
        sample_size=args.sample_size,
        sample_frac=args.sample_frac,
        seed=args.seed,
        sensitive_attr=args.sensitive_attr,
        output_root=args.output_root,
        aux_output=args.aux_output,
        released_eval=args.released_eval,
    )

    print(f"Full dataset        : {result['metadata']['full_dataset_with_record_id']}")
    print(f"Auxiliary dataset   : {result['aux_output_path']}")
    print(f"Metadata            : {result['metadata_path']}")
    print(f"Target population   : {result['metadata']['target_population_mode']}")
    if result["metadata"]["released_eval_path"]:
        print(f"Released eval       : {result['metadata']['released_eval_path']}")
        print(f"Released overlap    : {result['metadata']['target_population_size']}")
    print(f"Rows in auxiliary   : {result['metadata']['n_auxiliary_rows']}")


if __name__ == "__main__":
    main()
