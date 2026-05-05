from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from common import ensure_dir, load_json, parse_csv_list, save_json


# Split the original dataset into a published subset and an OUT holdout pool
# for membership inference attack evaluation.
#
# Target creation is handled AFTER anonymization by make_mia_targets_post_ano.py,
# which verifies that IN candidates actually survived in the anonymized release.


# ---------------------------------------------------------------------------
# Default output paths
# ---------------------------------------------------------------------------


# Build the default output path for the published dataset.
def default_publish_output(output_root: str | Path, name: str) -> Path:
    return Path(output_root) / "prepared_data" / f"{name}.published.csv"


# Build the default output path for the OUT subset.
def default_out_output(output_root: str | Path, name: str) -> Path:
    return Path(output_root) / "prepared_data" / f"{name}.out.csv"


# ---------------------------------------------------------------------------
# Shared helpers (importable by run_mia_benchmark, make_mia_targets_post_ano)
# ---------------------------------------------------------------------------


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


# Build a balanced targets dataset with members from one source and non-members from another.
# Kept here so that run_mia_benchmark and make_mia_targets_post_ano can import it.
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
    targets_df = targets_df.sample(frac=1.0, random_state=seed + 1, replace=False).reset_index(drop=True)
    for col in targets_df.columns:
        targets_df[col] = targets_df[col].astype(str)
    return targets_df


# ---------------------------------------------------------------------------
# Legacy split functions (kept for run_mia_benchmark backward compatibility)
# ---------------------------------------------------------------------------


# Split the full dataset into a published subset and a holdout subset.
# Legacy helper kept for run_mia_benchmark compatibility.
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


# Split the original dataset with an OUT pool and an IN pool sampled from the published rows.
# Legacy helper kept for run_mia_benchmark compatibility.  The standalone CLI now uses
# split_published_out (2-way split) and defers target creation to post-anonymization.
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


# ---------------------------------------------------------------------------
# Current split function (2-way: published + OUT only)
# ---------------------------------------------------------------------------


# Split the original dataset into a published subset (sent to anonymization)
# and an OUT pool (never anonymized, used as non-member candidates post-ano).
def split_published_out(
    df: pd.DataFrame,
    *,
    out_size: int | None = None,
    out_frac: float | None = 0.05,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_total = len(df)
    if n_total < 2:
        raise ValueError("The dataset must contain at least 2 rows.")

    resolved_out_size = _resolve_subset_size(
        n_total=n_total,
        subset_size=out_size,
        subset_frac=out_frac,
        default_frac=0.05,
        subset_name="out",
    )

    shuffled = df.sample(frac=1.0, random_state=seed, replace=False).reset_index(drop=True)
    out_df = shuffled.iloc[:resolved_out_size].copy().reset_index(drop=True)
    published_df = shuffled.iloc[resolved_out_size:].copy().reset_index(drop=True)
    return published_df, out_df


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


# Create an updated config copy that points to the published subset.
# Also guarantees that the record_id column is treated as an insensitive,
# non-quasi-identifier passthrough attribute so that post-anonymization
# targeting can match surviving rows back to the original published dataset.
def update_config_copy(
    base_config_path: str | Path,
    config_output_path: str | Path,
    published_dataset_path: str | Path,
    target_id_col: str = "record_id",
) -> dict[str, Any]:
    payload = load_json(base_config_path)
    payload["data"] = str(Path(published_dataset_path).resolve())

    # Make sure the record-id column is NEVER listed as a quasi-identifier:
    # otherwise it would be generalized/suppressed and we would lose the
    # one-to-one mapping needed by make_mia_targets_post_ano.py.
    qis = payload.get("quasi_identifiers")
    if isinstance(qis, list) and target_id_col in qis:
        payload["quasi_identifiers"] = [q for q in qis if q != target_id_col]

    # Same for sensitive attributes (if this schema uses them): a record id
    # is an identifier, not a sensitive value.
    sens = payload.get("sensitive_attributes")
    if isinstance(sens, list) and target_id_col in sens:
        payload["sensitive_attributes"] = [s for s in sens if s != target_id_col]

    # Register the record-id column as an insensitive attribute so that any
    # downstream tool that inspects this list knows it must pass through
    # unchanged.  Creates the key if missing, appends otherwise (de-duplicated).
    insens = payload.get("insensitive_attributes")
    if not isinstance(insens, list):
        insens = []
    if target_id_col not in insens:
        insens.append(target_id_col)
    payload["insensitive_attributes"] = insens

    save_json(config_output_path, payload)
    return payload


# ---------------------------------------------------------------------------
# Main pipeline: pre-anonymization split only
# ---------------------------------------------------------------------------


# Derive the OUT fraction from the total attacker knowledge base fraction.
# The attacker KB is split evenly: half OUT (held out), half IN (sampled post-ano).
def resolve_out_frac_from_attacker(
    *,
    attacker_frac: float | None,
    attacker_size: int | None,
    n_total: int,
) -> tuple[float | None, int | None]:
    if attacker_frac is not None and attacker_size is not None:
        raise ValueError("Use either --attacker-frac or --attacker-size, not both.")

    if attacker_size is not None:
        if attacker_size <= 0:
            raise ValueError("--attacker-size must be > 0.")
        if attacker_size >= n_total:
            raise ValueError(f"--attacker-size ({attacker_size}) must be < dataset size ({n_total}).")
        out_size = attacker_size // 2
        if out_size <= 0:
            raise ValueError(f"--attacker-size ({attacker_size}) is too small to split into OUT and IN pools.")
        return None, out_size

    if attacker_frac is not None:
        if not 0.0 < attacker_frac < 1.0:
            raise ValueError("--attacker-frac must be in (0, 1).")
        out_frac = attacker_frac / 2.0
        return out_frac, None

    # Default: 10% attacker KB → 5% OUT
    return 0.05, None


# Split the original dataset into published and OUT subsets for MIA evaluation.
# Targets are created AFTER anonymization by make_mia_targets_post_ano.py.
def prepare_mia_split(
    *,
    original_path: str | Path,
    attacker_frac: float | None = 0.10,
    attacker_size: int | None = None,
    seed: int = 42,
    target_id_col: str = "record_id",
    output_root: str | Path = "outputs",
    published_output: str | Path | None = None,
    out_output: str | Path | None = None,
    base_config: str | Path | None = None,
    config_output: str | Path | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    original_path = Path(original_path).resolve()
    output_root = Path(output_root).resolve()

    df = pd.read_csv(original_path, dtype=str, keep_default_na=False)
    df = ensure_record_id(df, target_id_col)

    out_frac, out_size = resolve_out_frac_from_attacker(
        attacker_frac=attacker_frac,
        attacker_size=attacker_size,
        n_total=len(df),
    )

    run_name = name or original_path.stem
    published_df, out_df = split_published_out(
        df,
        out_size=out_size,
        out_frac=out_frac,
        seed=seed,
    )

    # The IN pool will be the same size as OUT (balanced attacker KB).
    expected_in_size = len(out_df)

    published_output_path = Path(published_output).resolve() if published_output else default_publish_output(output_root, run_name)
    out_output_path = Path(out_output).resolve() if out_output else default_out_output(output_root, run_name)

    ensure_dir(published_output_path.parent)
    ensure_dir(out_output_path.parent)
    published_df.to_csv(published_output_path, index=False)
    out_df.to_csv(out_output_path, index=False)

    metadata = {
        "original_path": str(original_path),
        "published_output": str(published_output_path),
        "out_output": str(out_output_path),
        "seed": seed,
        "target_id_col": target_id_col,
        "original_size": len(df),
        "published_size": len(published_df),
        "out_size": len(out_df),
        "expected_in_size": expected_in_size,
        "attacker_frac": attacker_frac,
        "attacker_size": attacker_size,
    }
    metadata_path = published_output_path.with_suffix(".json")
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
            target_id_col=target_id_col,
        )

    return {
        "published_output_path": published_output_path,
        "out_output_path": out_output_path,
        "metadata_path": metadata_path,
        "config_output_path": config_output_path,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# Parse CLI arguments and split the original dataset into published + OUT.
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split the original dataset into a published subset (input for anonymization) "
            "and an OUT holdout pool (non-members for MIA).  The attacker knowledge base "
            "fraction is split evenly: half OUT, half IN.  Balanced IN/OUT targets are "
            "created AFTER anonymization with make_mia_targets_post_ano.py."
        )
    )
    parser.add_argument("--original", required=True, help="Path to the original CSV dataset.")
    parser.add_argument(
        "--attacker-frac", type=float, default=0.10,
        help="Fraction of the original dataset in the attacker knowledge base (default 0.10). Split evenly into OUT and IN.",
    )
    parser.add_argument(
        "--attacker-size", type=int, default=None,
        help="Absolute number of records in the attacker knowledge base. Split evenly into OUT and IN.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split.")
    parser.add_argument("--target-id-col", default="record_id", help="Internal identifier column name.")
    parser.add_argument("--output-root", default="outputs", help="Root directory for generated files.")
    parser.add_argument("--name", default=None, help="Optional prefix for generated filenames.")
    parser.add_argument("--published-output", default=None, help="Optional path for the published subset CSV.")
    parser.add_argument("--out-output", default=None, help="Optional path for the OUT subset CSV.")
    parser.add_argument("--base-config", default=None, help="Optional base config JSON to retarget to the published subset.")
    parser.add_argument("--config-output", default=None, help="Optional output path for the updated config JSON.")
    args = parser.parse_args()

    result = prepare_mia_split(
        original_path=args.original,
        attacker_frac=args.attacker_frac if args.attacker_size is None else None,
        attacker_size=args.attacker_size,
        seed=args.seed,
        target_id_col=args.target_id_col,
        output_root=args.output_root,
        published_output=args.published_output,
        out_output=args.out_output,
        base_config=args.base_config,
        config_output=args.config_output,
        name=args.name,
    )

    m = result['metadata']
    print(f"[OK] Published subset : {result['published_output_path']}  ({m['published_size']} rows)")
    print(f"[OK] OUT holdout      : {result['out_output_path']}  ({m['out_size']} rows)")
    print(f"     Expected IN size : {m['expected_in_size']} (sampled post-ano from survivors)")
    print(f"     Attacker KB      : {m['out_size'] + m['expected_in_size']} rows ({m['out_size']} OUT + {m['expected_in_size']} IN)")
    print(f"[OK] Metadata JSON    : {result['metadata_path']}")
    if result['config_output_path'] is not None:
        print(f"[OK] Updated config   : {result['config_output_path']}")
    print()
    print("Next step: anonymize the published subset, then run make_mia_targets_post_ano.py")


if __name__ == "__main__":
    main()
