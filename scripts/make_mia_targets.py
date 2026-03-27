from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from common import ensure_dir, load_json, save_json


# Create a published subset, a holdout subset, and balanced MIA targets for one attacker knowledge setting.


# Parse a comma-separated string into a list of values.
def parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


# Build the default output path for the published dataset.
def default_publish_output(output_root: str | Path, original_path: str | Path, target_id_col: str) -> Path:
    output_root = Path(output_root)
    original_path = Path(original_path)
    return output_root / "prepared_data" / f"{original_path.stem}_published_with_{target_id_col}.csv"


# Build the default output path for the holdout dataset.
def default_holdout_output(output_root: str | Path, original_path: str | Path, target_id_col: str) -> Path:
    output_root = Path(output_root)
    original_path = Path(original_path)
    return output_root / "prepared_data" / f"{original_path.stem}_holdout_with_{target_id_col}.csv"


# Build the default output path for the balanced MIA targets dataset.
def default_targets_output(output_root: str | Path, known_qids: list[str], targets_per_class: int, seed: int) -> Path:
    output_root = Path(output_root)
    q_part = "-".join(known_qids)
    return output_root / "mia_targets" / f"mia_targets__known_{q_part}__n_{targets_per_class}__seed_{seed}.csv"


# Ensure the dataset contains a unique internal record identifier column.
def ensure_record_id(df: pd.DataFrame, target_id_col: str) -> pd.DataFrame:
    df = df.copy()
    if target_id_col in df.columns:
        values = df[target_id_col].astype(str)
        if values.nunique(dropna=False) != len(df):
            raise ValueError(f"Existing column '{target_id_col}' is not unique.")
        df[target_id_col] = values
        return df

    df.insert(0, target_id_col, [str(i) for i in range(len(df))])
    return df


# Split the full dataset into a published subset and a holdout subset.
def split_publish_holdout(
    df: pd.DataFrame,
    *,
    publish_size: int | None,
    publish_frac: float | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if publish_size is None and publish_frac is None:
        publish_frac = 0.5

    if publish_size is not None and publish_frac is not None:
        raise ValueError("Use either publish_size or publish_frac, not both.")

    if publish_frac is not None:
        if not 0.0 < float(publish_frac) < 1.0:
            raise ValueError("publish_frac must be in (0, 1).")
        publish_size = int(round(len(df) * float(publish_frac)))

    assert publish_size is not None
    if publish_size <= 0 or publish_size >= len(df):
        raise ValueError("publish_size must be between 1 and len(df)-1.")

    shuffled = df.sample(frac=1.0, random_state=seed, replace=False).reset_index(drop=True)
    publish_df = shuffled.iloc[:publish_size].copy().reset_index(drop=True)
    holdout_df = shuffled.iloc[publish_size:].copy().reset_index(drop=True)
    return publish_df, holdout_df


# Resolve one size parameter from either an absolute size or a fraction of the full dataset.
def _resolve_subset_size(
    *,
    n_total: int,
    subset_size: int | None,
    subset_frac: float | None,
    default_frac: float,
    subset_name: str,
) -> int:
    if subset_size is not None and subset_frac is not None:
        raise ValueError(f"Use either {subset_name}_size or {subset_name}_frac, not both.")

    if subset_size is None and subset_frac is None:
        subset_frac = default_frac

    if subset_frac is not None:
        subset_frac = float(subset_frac)
        if not 0.0 < subset_frac < 1.0:
            raise ValueError(f"{subset_name}_frac must be in (0, 1).")
        subset_size = int(round(n_total * subset_frac))

    assert subset_size is not None
    if subset_size <= 0:
        raise ValueError(f"{subset_name}_size must be > 0.")
    if subset_size >= n_total:
        raise ValueError(f"{subset_name}_size must be < len(df).")
    return subset_size


# Split the original dataset for MIA with a 5% OUT pool and a 5% IN pool sampled from the remaining published rows.
def split_mia_candidate_pools(
    df: pd.DataFrame,
    *,
    out_size: int | None = None,
    out_frac: float | None = 0.05,
    in_size: int | None = None,
    in_frac: float | None = 0.05,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_total = len(df)
    if n_total < 3:
        raise ValueError("The dataset must contain at least 3 rows for the MIA split.")

    out_size = _resolve_subset_size(
        n_total=n_total,
        subset_size=out_size,
        subset_frac=out_frac,
        default_frac=0.05,
        subset_name="out",
    )
    in_size = _resolve_subset_size(
        n_total=n_total,
        subset_size=in_size,
        subset_frac=in_frac,
        default_frac=0.05,
        subset_name="in",
    )

    if out_size + in_size >= n_total:
        raise ValueError(
            f"The requested OUT ({out_size}) and IN ({in_size}) auxiliary pools leave no published data."
        )

    shuffled = df.sample(frac=1.0, random_state=seed, replace=False).reset_index(drop=True)
    out_df = shuffled.iloc[:out_size].copy().reset_index(drop=True)
    published_df = shuffled.iloc[out_size:].copy().reset_index(drop=True)

    if in_size > len(published_df):
        raise ValueError(
            f"in_size ({in_size}) is larger than the published subset size ({len(published_df)})."
        )

    auxiliary_in_df = published_df.sample(n=in_size, random_state=seed + 1, replace=False).copy().reset_index(drop=True)
    return published_df, out_df, auxiliary_in_df


# Build a balanced targets dataset with members from one source and non-members from another.
def build_targets_df(
    member_source_df: pd.DataFrame,
    non_member_source_df: pd.DataFrame,
    *,
    known_qids: list[str],
    target_id_col: str,
    member_col: str,
    targets_per_class: int,
    seed: int,
) -> pd.DataFrame:
    if not known_qids:
        raise ValueError("known_qids cannot be empty.")

    member_missing = [qi for qi in known_qids if qi not in member_source_df.columns]
    non_member_missing = [qi for qi in known_qids if qi not in non_member_source_df.columns]
    if member_missing or non_member_missing:
        raise ValueError(
            f"Missing known QI columns. member_missing={member_missing}, non_member_missing={non_member_missing}"
        )

    if target_id_col not in member_source_df.columns or target_id_col not in non_member_source_df.columns:
        raise ValueError(f"Both datasets must contain '{target_id_col}'.")

    if targets_per_class <= 0:
        raise ValueError("targets_per_class must be > 0.")
    if targets_per_class > len(member_source_df):
        raise ValueError(
            f"targets_per_class ({targets_per_class}) is larger than the member source size ({len(member_source_df)})."
        )
    if targets_per_class > len(non_member_source_df):
        raise ValueError(
            f"targets_per_class ({targets_per_class}) is larger than the non-member source size ({len(non_member_source_df)})."
        )

    member_df = member_source_df.sample(n=targets_per_class, random_state=seed, replace=False).copy()
    non_member_df = non_member_source_df.sample(n=targets_per_class, random_state=seed, replace=False).copy()

    keep_cols = [target_id_col] + list(known_qids)
    member_targets = member_df[keep_cols].copy()
    member_targets[member_col] = 1
    non_member_targets = non_member_df[keep_cols].copy()
    non_member_targets[member_col] = 0

    targets_df = pd.concat([member_targets, non_member_targets], ignore_index=True)
    targets_df = targets_df.sample(frac=1.0, random_state=seed, replace=False).reset_index(drop=True)
    for col in targets_df.columns:
        targets_df[col] = targets_df[col].astype(str)
    return targets_df


# Create an updated config copy that points to the published subset.
def update_config_copy(
    base_config_path: str | Path,
    config_output_path: str | Path,
    published_dataset_path: str | Path,
) -> dict[str, Any]:
    payload = load_json(base_config_path)
    payload["data"] = str(Path(published_dataset_path).resolve())
    save_json(config_output_path, payload)
    return payload


# Parse CLI arguments and generate published, holdout, and target datasets for one MIA setup.
def main() -> None:
    parser = argparse.ArgumentParser(description="Create published/holdout subsets and balanced MIA targets.")
    parser.add_argument("--original", required=True, help="Path to the original CSV dataset.")
    parser.add_argument("--known-qids", required=True, help="Comma-separated attacker-known QIs.")
    parser.add_argument("--publish-size", type=int, default=None, help="Legacy mode only: number of rows kept in the published subset.")
    parser.add_argument("--publish-frac", type=float, default=None, help="Legacy mode only: fraction of rows kept in the published subset.")
    parser.add_argument("--out-size", type=int, default=None, help="Number of OUT candidates kept in the attacker auxiliary pool.")
    parser.add_argument("--out-frac", type=float, default=0.05, help="Fraction of the original dataset used as OUT candidates.")
    parser.add_argument("--in-size", type=int, default=None, help="Number of IN candidates kept in the attacker auxiliary pool.")
    parser.add_argument("--in-frac", type=float, default=0.05, help="Fraction of the original dataset used as IN candidates.")
    parser.add_argument("--legacy-publish-holdout", action="store_true", help="Use the old published/holdout split instead of the mixed IN/OUT auxiliary pool.")
    parser.add_argument("--targets-per-class", type=int, default=500, help="Number of member and non-member targets.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for splitting and sampling.")
    parser.add_argument("--target-id-col", default="record_id", help="Internal identifier column name.")
    parser.add_argument("--member-col", default="is_member", help="Ground-truth membership label column name.")
    parser.add_argument("--output-root", default="outputs", help="Root directory for generated files.")
    parser.add_argument("--published-output", default=None, help="Optional path for the published subset CSV.")
    parser.add_argument("--holdout-output", default=None, help="Optional path for the holdout subset CSV.")
    parser.add_argument("--targets-output", default=None, help="Optional path for the MIA targets CSV.")
    parser.add_argument("--base-config", default=None, help="Optional base config JSON to copy and retarget to the published subset.")
    parser.add_argument("--config-output", default=None, help="Optional output path for the updated config JSON.")
    args = parser.parse_args()

    original_path = Path(args.original).resolve()
    known_qids = parse_csv_list(args.known_qids)
    output_root = Path(args.output_root).resolve()

    published_output = (
        Path(args.published_output).resolve()
        if args.published_output
        else default_publish_output(output_root, original_path, args.target_id_col)
    )
    holdout_output = (
        Path(args.holdout_output).resolve()
        if args.holdout_output
        else default_holdout_output(output_root, original_path, args.target_id_col)
    )
    targets_output = (
        Path(args.targets_output).resolve()
        if args.targets_output
        else default_targets_output(output_root, known_qids, args.targets_per_class, args.seed)
    )

    df = pd.read_csv(original_path, dtype=str, keep_default_na=False)
    df = ensure_record_id(df, args.target_id_col)

    if args.legacy_publish_holdout:
        publish_df, holdout_df = split_publish_holdout(
            df,
            publish_size=args.publish_size,
            publish_frac=args.publish_frac,
            seed=args.seed,
        )
        member_source_df = publish_df
        non_member_source_df = holdout_df
        split_metadata = {
            "split_mode": "legacy_publish_holdout",
            "publish_size": len(publish_df),
            "holdout_size": len(holdout_df),
        }
    else:
        publish_df, holdout_df, auxiliary_in_df = split_mia_candidate_pools(
            df,
            out_size=args.out_size,
            out_frac=args.out_frac,
            in_size=args.in_size,
            in_frac=args.in_frac,
            seed=args.seed,
        )
        member_source_df = auxiliary_in_df
        non_member_source_df = holdout_df
        split_metadata = {
            "split_mode": "mixed_auxiliary_pool",
            "published_size": len(publish_df),
            "auxiliary_out_size": len(holdout_df),
            "auxiliary_in_size": len(auxiliary_in_df),
            "out_frac": args.out_frac,
            "in_frac": args.in_frac,
        }

    ensure_dir(published_output.parent)
    ensure_dir(holdout_output.parent)
    ensure_dir(targets_output.parent)
    publish_df.to_csv(published_output, index=False)
    holdout_df.to_csv(holdout_output, index=False)

    targets_df = build_targets_df(
        member_source_df,
        non_member_source_df,
        known_qids=known_qids,
        target_id_col=args.target_id_col,
        member_col=args.member_col,
        targets_per_class=args.targets_per_class,
        seed=args.seed,
    )
    targets_df.to_csv(targets_output, index=False)

    metadata = {
        "original_path": str(original_path),
        "published_output": str(published_output),
        "holdout_output": str(holdout_output),
        "targets_output": str(targets_output),
        "known_qids": known_qids,
        "targets_per_class": args.targets_per_class,
        "seed": args.seed,
        "target_id_col": args.target_id_col,
        "member_col": args.member_col,
        **split_metadata,
    }
    metadata_path = targets_output.with_suffix(".json")
    save_json(metadata_path, metadata)

    if args.base_config and args.config_output:
        update_config_copy(
            base_config_path=Path(args.base_config).resolve(),
            config_output_path=Path(args.config_output).resolve(),
            published_dataset_path=published_output,
        )

    print(f"[OK] Published subset : {published_output}")
    print(f"[OK] Holdout subset   : {holdout_output}")
    print(f"[OK] Targets CSV      : {targets_output}")
    print(f"[OK] Metadata JSON    : {metadata_path}")
    if args.base_config and args.config_output:
        print(f"[OK] Updated config   : {Path(args.config_output).resolve()}")


if __name__ == "__main__":
    main()
