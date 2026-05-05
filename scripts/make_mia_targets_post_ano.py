# Build balanced MIA targets AFTER anonymization, so that IN labels reflect
# actual presence in the anonymized release (accounting for fully-suppressed
# row drops or any other post-anonymization filtering).
#
# Typical workflow:
#   1. make_mia_targets.py        → split original into published.csv + out.csv
#                                    (--attacker-frac controls KB size)
#   2. run_ano_modified.py        → anonymize published.csv → anonymized_eval.csv
#   3. THIS SCRIPT                → read anonymized_eval.csv, verify surviving
#                                    record_ids, build attacker base, sample
#                                    balanced IN/OUT targets.
#
# With --split-metadata the script auto-reads published/out paths and the
# expected IN pool size from the JSON produced by make_mia_targets.py.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from common import ensure_dir, load_json, parse_csv_list, save_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Build the default output path for the post-ano targets file.
def default_targets_output(output_root: str | Path, name: str) -> Path:
    return Path(output_root) / "mia_targets" / f"{name}.targets_post_ano.csv"


# Build the default output path for the attacker knowledge base.
def default_attacker_base_output(output_root: str | Path, name: str) -> Path:
    return Path(output_root) / "prepared_data" / f"{name}.attacker_base.csv"


# Load a CSV as all-string with stripped whitespace.
def _read_csv_str(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(path), dtype=str, keep_default_na=False)
    df.columns = [str(col).strip() for col in df.columns]
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


# Identify record_ids that survived in the anonymized eval export.
def get_surviving_record_ids(
    anonymized_eval_path: str | Path,
    target_id_col: str,
) -> set[str]:
    df_eval = _read_csv_str(anonymized_eval_path)
    if target_id_col not in df_eval.columns:
        raise ValueError(
            f"Column '{target_id_col}' not found in anonymized_eval: {anonymized_eval_path}"
        )
    return set(df_eval[target_id_col].tolist())


# Build the attacker knowledge base (IN sample + OUT pool), balanced targets,
# and all metadata from the post-anonymization state.
def build_post_ano_targets(
    *,
    published_path: str | Path,
    out_path: str | Path,
    anonymized_eval_path: str | Path,
    known_qids: list[str],
    expected_in_size: int | None = None,
    target_id_col: str = "record_id",
    member_col: str = "is_member",
    targets_per_class: int | None = None,
    seed: int = 42,
    output_root: str | Path = "outputs",
    targets_output: str | Path | None = None,
    attacker_base_output: str | Path | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    if not known_qids:
        raise ValueError("known_qids cannot be empty.")

    published_path = Path(published_path).resolve()
    out_path = Path(out_path).resolve()
    anonymized_eval_path = Path(anonymized_eval_path).resolve()
    output_root = Path(output_root).resolve()

    # ---- Load datasets ----
    df_published = _read_csv_str(published_path)
    df_out = _read_csv_str(out_path)

    for label, df in [("published", df_published), ("out", df_out)]:
        if target_id_col not in df.columns:
            raise ValueError(f"Column '{target_id_col}' not found in {label} dataset: {df.columns.tolist()}")
        missing = [qi for qi in known_qids if qi not in df.columns]
        if missing:
            raise ValueError(f"Missing known QI columns in {label} dataset: {missing}")

    # ---- Determine surviving IN candidates ----
    surviving_ids = get_surviving_record_ids(anonymized_eval_path, target_id_col)

    all_surviving_in = df_published[
        df_published[target_id_col].isin(surviving_ids)
    ].copy().reset_index(drop=True)

    # OUT candidates: none of them should be in anonymized_eval
    out_in_eval = df_out[df_out[target_id_col].isin(surviving_ids)]
    if len(out_in_eval) > 0:
        raise ValueError(
            f"{len(out_in_eval)} OUT-pool record(s) were found in anonymized_eval. "
            "This indicates a pipeline error: OUT records should never appear in the anonymized release."
        )
    out_candidates = df_out.copy().reset_index(drop=True)

    # ---- Determine IN pool size (balanced with OUT) ----
    if expected_in_size is None:
        expected_in_size = len(out_candidates)

    if expected_in_size > len(all_surviving_in):
        raise ValueError(
            f"expected_in_size ({expected_in_size}) exceeds the number of surviving "
            f"published records ({len(all_surviving_in)}). The anonymization suppression "
            "rate is too high for the requested attacker knowledge base size."
        )

    # Subsample surviving records to build the IN pool of the attacker KB
    in_candidates = all_surviving_in.sample(
        n=expected_in_size, random_state=seed, replace=False,
    ).copy().reset_index(drop=True)

    # ---- Dropped IN stats ----
    all_published_ids = set(df_published[target_id_col].tolist())
    dropped_ids = all_published_ids - surviving_ids
    n_published = len(all_published_ids)
    n_surviving = len(all_surviving_in)
    n_dropped = len(dropped_ids)

    print(f"Published records     : {n_published}")
    print(f"Surviving in eval     : {n_surviving}")
    print(f"Dropped (suppressed)  : {n_dropped}")
    print(f"OUT pool size         : {len(out_candidates)}")
    print(f"IN pool size          : {len(in_candidates)} (sampled from {n_surviving} survivors)")

    # ---- Build attacker knowledge base ----
    attacker_base_df = pd.concat([out_candidates, in_candidates], ignore_index=True)
    attacker_base_df = attacker_base_df.sample(
        frac=1.0, random_state=seed + 2, replace=False,
    ).reset_index(drop=True)

    # ---- Resolve targets_per_class (default: use the full attacker KB) ----
    max_per_class = min(len(in_candidates), len(out_candidates))
    if targets_per_class is None:
        targets_per_class = max_per_class
    if targets_per_class <= 0:
        raise ValueError("targets_per_class must be > 0.")
    if targets_per_class > max_per_class:
        raise ValueError(
            f"targets_per_class ({targets_per_class}) exceeds the smallest pool "
            f"(IN={len(in_candidates)}, OUT={len(out_candidates)}). "
            "Lower targets_per_class or increase --attacker-frac."
        )

    # ---- Sample balanced targets ----
    if targets_per_class == len(in_candidates):
        in_sample = in_candidates.copy()
    else:
        in_sample = in_candidates.sample(
            n=targets_per_class, random_state=seed + 3, replace=False,
        ).copy()
    if targets_per_class == len(out_candidates):
        out_sample = out_candidates.copy()
    else:
        out_sample = out_candidates.sample(
            n=targets_per_class, random_state=seed + 3, replace=False,
        ).copy()

    keep_cols = [target_id_col] + list(known_qids)
    in_targets = in_sample[keep_cols].copy()
    in_targets[member_col] = "1"
    out_targets = out_sample[keep_cols].copy()
    out_targets[member_col] = "0"

    targets_df = pd.concat([in_targets, out_targets], ignore_index=True)
    targets_df = targets_df.sample(frac=1.0, random_state=seed + 4, replace=False).reset_index(drop=True)
    for col in targets_df.columns:
        targets_df[col] = targets_df[col].astype(str)

    # ---- Save ----
    run_name = name or published_path.stem.replace(".published", "")
    targets_output_path = (
        Path(targets_output).resolve()
        if targets_output
        else default_targets_output(output_root, run_name)
    )
    attacker_base_output_path = (
        Path(attacker_base_output).resolve()
        if attacker_base_output
        else default_attacker_base_output(output_root, run_name)
    )

    ensure_dir(targets_output_path.parent)
    ensure_dir(attacker_base_output_path.parent)
    targets_df.to_csv(targets_output_path, index=False)
    attacker_base_df.to_csv(attacker_base_output_path, index=False)

    metadata = {
        "published_path": str(published_path),
        "out_path": str(out_path),
        "anonymized_eval_path": str(anonymized_eval_path),
        "known_qids": known_qids,
        "targets_per_class": targets_per_class,
        "seed": seed,
        "target_id_col": target_id_col,
        "member_col": member_col,
        "n_published_records": n_published,
        "n_surviving_in_eval": n_surviving,
        "n_dropped_by_suppression": n_dropped,
        "drop_rate": round(n_dropped / n_published, 6) if n_published else 0.0,
        "n_out_pool": len(out_candidates),
        "n_in_pool": len(in_candidates),
        "expected_in_size": expected_in_size,
        "attacker_base_size": len(attacker_base_df),
        "n_targets_in": targets_per_class,
        "n_targets_out": targets_per_class,
        "n_targets_total": 2 * targets_per_class,
        "targets_output": str(targets_output_path),
        "attacker_base_output": str(attacker_base_output_path),
    }
    metadata_path = targets_output_path.with_suffix(".json")
    save_json(metadata_path, metadata)

    print(f"[OK] Attacker base    : {attacker_base_output_path}  ({len(attacker_base_df)} rows = {len(out_candidates)} OUT + {len(in_candidates)} IN)")
    print(f"[OK] Targets CSV      : {targets_output_path}")
    print(f"[OK] Metadata JSON    : {metadata_path}")
    print(f"     IN  targets      : {targets_per_class}")
    print(f"     OUT targets      : {targets_per_class}")
    print(f"     Drop rate        : {metadata['drop_rate']:.2%}")

    return {
        "targets_output_path": targets_output_path,
        "attacker_base_output_path": attacker_base_output_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create balanced MIA targets AFTER anonymization.  IN targets are "
            "sampled only from published records whose record_id survived in the "
            "anonymized_eval export.  Use --split-metadata to auto-read paths and "
            "the expected IN pool size from the make_mia_targets.py output."
        )
    )
    parser.add_argument(
        "--split-metadata", default=None,
        help=(
            "Path to the JSON metadata produced by make_mia_targets.py.  When provided, "
            "--published, --out-pool, --seed, and --target-id-col are read from it "
            "automatically (CLI flags still override if given)."
        ),
    )
    parser.add_argument(
        "--published", default=None,
        help="Path to the published subset CSV (original values, with record_id).",
    )
    parser.add_argument(
        "--out-pool", default=None,
        help="Path to the OUT pool CSV (original values, with record_id).",
    )
    parser.add_argument(
        "--anonymized-eval", required=True,
        help="Path to the anonymized_eval CSV (post-anonymization, with record_id).",
    )
    parser.add_argument(
        "--known-qids", required=True,
        help="Comma-separated attacker-known quasi-identifiers.",
    )
    parser.add_argument("--targets-per-class", type=int, default=None, help="Number of IN and OUT targets (default: use the full attacker knowledge base).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling (default: read from split metadata).")
    parser.add_argument("--target-id-col", default=None, help="Record identifier column name (default: read from split metadata).")
    parser.add_argument("--member-col", default="is_member", help="Membership label column name.")
    parser.add_argument("--output-root", default="outputs", help="Root directory for outputs.")
    parser.add_argument("--targets-output", default=None, help="Optional path for the targets CSV.")
    parser.add_argument("--attacker-base-output", default=None, help="Optional path for the attacker knowledge base CSV.")
    parser.add_argument("--name", default=None, help="Optional name prefix for output files.")
    args = parser.parse_args()

    # ---- Resolve parameters from split metadata + CLI overrides ----
    split_meta: dict[str, Any] = {}
    if args.split_metadata:
        split_meta = load_json(args.split_metadata)

    published_path = args.published or split_meta.get("published_output")
    out_path = args.out_pool or split_meta.get("out_output")
    seed = args.seed if args.seed is not None else split_meta.get("seed", 42)
    target_id_col = args.target_id_col or split_meta.get("target_id_col", "record_id")
    expected_in_size = split_meta.get("expected_in_size")

    if not published_path:
        parser.error("--published is required (or provide --split-metadata).")
    if not out_path:
        parser.error("--out-pool is required (or provide --split-metadata).")

    build_post_ano_targets(
        published_path=published_path,
        out_path=out_path,
        anonymized_eval_path=args.anonymized_eval,
        known_qids=parse_csv_list(args.known_qids),
        expected_in_size=expected_in_size,
        targets_per_class=args.targets_per_class,
        seed=seed,
        target_id_col=target_id_col,
        member_col=args.member_col,
        output_root=args.output_root,
        targets_output=args.targets_output,
        attacker_base_output=args.attacker_base_output,
        name=args.name,
    )


if __name__ == "__main__":
    main()
