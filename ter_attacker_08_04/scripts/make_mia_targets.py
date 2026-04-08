from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from common import ensure_dir, load_json, parse_csv_list, save_json


# Create the published subset, OUT subset, attacker base, and balanced MIA targets.


# Build the default output path for the published dataset.
def default_publish_output(output_root: str | Path, name: str) -> Path:
    return Path(output_root) / "prepared_data" / f"{name}.published.csv"


# Build the default output path for the OUT subset.
def default_out_output(output_root: str | Path, name: str) -> Path:
    return Path(output_root) / "prepared_data" / f"{name}.out.csv"


# Build the default output path for the attacker base.
def default_attacker_base_output(output_root: str | Path, name: str) -> Path:
    return Path(output_root) / "prepared_data" / f"{name}.attacker_base.csv"


# Build the default output path for the balanced MIA targets dataset.
def default_targets_output(output_root: str | Path, name: str) -> Path:
    return Path(output_root) / "mia_targets" / f"{name}.targets.csv"


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
# Kept for benchmark compatibility, even though the main CLI now uses the IN/OUT setup.
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


# Split the original dataset for MIA with an OUT pool and an IN pool sampled from the remaining published rows.
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
        raise ValueError(f"The requested OUT ({out_size}) and IN ({in_size}) pools leave no published data.")

    shuffled = df.sample(frac=1.0, random_state=seed, replace=False).reset_index(drop=True)
    out_df = shuffled.iloc[:out_size].copy().reset_index(drop=True)
    published_df = shuffled.iloc[out_size:].copy().reset_index(drop=True)

    if in_size > len(published_df):
        raise ValueError(f"in_size ({in_size}) is larger than the published subset size ({len(published_df)}).")

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
        raise ValueError(f"Missing known QI columns. member_missing={member_missing}, non_member_missing={non_member_missing}")

    if target_id_col not in member_source_df.columns or target_id_col not in non_member_source_df.columns:
        raise ValueError(f"Both datasets must contain '{target_id_col}'.")

    if targets_per_class <= 0:
        raise ValueError("targets_per_class must be > 0.")
    if targets_per_class > len(member_source_df):
        raise ValueError(f"targets_per_class ({targets_per_class}) is larger than the member source size ({len(member_source_df)}).")
    if targets_per_class > len(non_member_source_df):
        raise ValueError(f"targets_per_class ({targets_per_class}) is larger than the non-member source size ({len(non_member_source_df)}).")

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


# Prepare the standard MIA data split and target set.
def prepare_mia_targets(
    *,
    original_path: str | Path,
    known_qids: list[str],
    out_size: int | None = None,
    out_frac: float | None = 0.05,
    in_size: int | None = None,
    in_frac: float | None = 0.05,
    targets_per_class: int = 500,
    seed: int = 42,
    target_id_col: str = "record_id",
    member_col: str = "is_member",
    output_root: str | Path = "outputs",
    published_output: str | Path | None = None,
    out_output: str | Path | None = None,
    attacker_base_output: str | Path | None = None,
    targets_output: str | Path | None = None,
    base_config: str | Path | None = None,
    config_output: str | Path | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    if not known_qids:
        raise ValueError("known_qids cannot be empty.")

    original_path = Path(original_path).resolve()
    output_root = Path(output_root).resolve()

    df = pd.read_csv(original_path, dtype=str, keep_default_na=False)
    df = ensure_record_id(df, target_id_col)

    missing_known_qids = [qi for qi in known_qids if qi not in df.columns]
    if missing_known_qids:
        raise ValueError(f"Unknown known_qids: {missing_known_qids}")

    run_name = name or original_path.stem
    published_df, out_df, auxiliary_in_df = split_mia_candidate_pools(
        df,
        out_size=out_size,
        out_frac=out_frac,
        in_size=in_size,
        in_frac=in_frac,
        seed=seed,
    )
    attacker_base_df = pd.concat([out_df, auxiliary_in_df], ignore_index=True)
    attacker_base_df = attacker_base_df.sample(frac=1.0, random_state=seed + 2, replace=False).reset_index(drop=True)

    targets_df = build_targets_df(
        member_source_df=auxiliary_in_df,
        non_member_source_df=out_df,
        known_qids=known_qids,
        target_id_col=target_id_col,
        member_col=member_col,
        targets_per_class=targets_per_class,
        seed=seed,
    )

    published_output_path = Path(published_output).resolve() if published_output else default_publish_output(output_root, run_name)
    out_output_path = Path(out_output).resolve() if out_output else default_out_output(output_root, run_name)
    attacker_base_output_path = Path(attacker_base_output).resolve() if attacker_base_output else default_attacker_base_output(output_root, run_name)
    targets_output_path = Path(targets_output).resolve() if targets_output else default_targets_output(output_root, run_name)

    ensure_dir(published_output_path.parent)
    ensure_dir(out_output_path.parent)
    ensure_dir(attacker_base_output_path.parent)
    ensure_dir(targets_output_path.parent)
    published_df.to_csv(published_output_path, index=False)
    out_df.to_csv(out_output_path, index=False)
    attacker_base_df.to_csv(attacker_base_output_path, index=False)
    targets_df.to_csv(targets_output_path, index=False)

    metadata = {
        "original_path": str(original_path),
        "published_output": str(published_output_path),
        "out_output": str(out_output_path),
        "attacker_base_output": str(attacker_base_output_path),
        "targets_output": str(targets_output_path),
        "known_qids": known_qids,
        "targets_per_class": targets_per_class,
        "seed": seed,
        "target_id_col": target_id_col,
        "member_col": member_col,
        "split_mode": "mixed_auxiliary_pool",
        "published_size": len(published_df),
        "auxiliary_out_size": len(out_df),
        "auxiliary_in_size": len(auxiliary_in_df),
        "attacker_base_size": len(attacker_base_df),
        "out_frac": out_frac,
        "in_frac": in_frac,
    }
    metadata_path = targets_output_path.with_suffix(".json")
    save_json(metadata_path, metadata)

    config_output_path = None
    if base_config:
        if config_output:
            config_output_path = Path(config_output).resolve()
        else:
            config_output_path = output_root / "configs" / f"{run_name}.runtime.json"
        update_config_copy(
            base_config_path=Path(base_config).resolve(),
            config_output_path=config_output_path,
            published_dataset_path=published_output_path,
        )

    return {
        "published_output_path": published_output_path,
        "out_output_path": out_output_path,
        "attacker_base_output_path": attacker_base_output_path,
        "targets_output_path": targets_output_path,
        "metadata_path": metadata_path,
        "config_output_path": config_output_path,
        "metadata": metadata,
    }


# Parse CLI arguments and generate published/OUT/attacker-base/targets files.
def main() -> None:
    parser = argparse.ArgumentParser(description="Create the standard IN/OUT MIA split and balanced targets.")
    parser.add_argument("--original", required=True, help="Path to the original CSV dataset.")
    parser.add_argument("--known-qids", required=True, help="Comma-separated attacker-known QIs.")
    parser.add_argument("--out-size", type=int, default=None, help="Number of OUT candidates kept in the attacker auxiliary pool.")
    parser.add_argument("--out-frac", type=float, default=0.05, help="Fraction of the original dataset used as OUT candidates.")
    parser.add_argument("--in-size", type=int, default=None, help="Number of IN candidates kept in the attacker auxiliary pool.")
    parser.add_argument("--in-frac", type=float, default=0.05, help="Fraction of the original dataset used as IN candidates.")
    parser.add_argument("--targets-per-class", type=int, default=500, help="Number of member and non-member targets.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for splitting and sampling.")
    parser.add_argument("--target-id-col", default="record_id", help="Internal identifier column name.")
    parser.add_argument("--member-col", default="is_member", help="Ground-truth membership label column name.")
    parser.add_argument("--output-root", default="outputs", help="Root directory for generated files.")
    parser.add_argument("--name", default=None, help="Optional prefix for generated filenames.")
    parser.add_argument("--published-output", default=None, help="Optional path for the published subset CSV.")
    parser.add_argument("--out-output", default=None, help="Optional path for the OUT subset CSV.")
    parser.add_argument("--attacker-base-output", default=None, help="Optional path for the attacker base CSV.")
    parser.add_argument("--targets-output", default=None, help="Optional path for the MIA targets CSV.")
    parser.add_argument("--base-config", default=None, help="Optional base config JSON to copy and retarget to the published subset.")
    parser.add_argument("--config-output", default=None, help="Optional output path for the updated config JSON.")
    args = parser.parse_args()

    result = prepare_mia_targets(
        original_path=args.original,
        known_qids=parse_csv_list(args.known_qids),
        out_size=args.out_size,
        out_frac=args.out_frac,
        in_size=args.in_size,
        in_frac=args.in_frac,
        targets_per_class=args.targets_per_class,
        seed=args.seed,
        target_id_col=args.target_id_col,
        member_col=args.member_col,
        output_root=args.output_root,
        published_output=args.published_output,
        out_output=args.out_output,
        attacker_base_output=args.attacker_base_output,
        targets_output=args.targets_output,
        base_config=args.base_config,
        config_output=args.config_output,
        name=args.name,
    )

    print(f"[OK] Published subset : {result['published_output_path']}")
    print(f"[OK] OUT subset       : {result['out_output_path']}")
    print(f"[OK] Attacker base    : {result['attacker_base_output_path']}")
    print(f"[OK] Targets CSV      : {result['targets_output_path']}")
    print(f"[OK] Metadata JSON    : {result['metadata_path']}")
    if result['config_output_path'] is not None:
        print(f"[OK] Updated config   : {result['config_output_path']}")


if __name__ == "__main__":
    main()
