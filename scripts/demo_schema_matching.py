"""
demo_schema_matching.py

Standalone demo of the schema-matching step inserted before the phase-2
refinement of the linkage attack.

This script:
  1. Loads an anonymized release (public CSV) and the attacker's auxiliary KB.
  2. Picks a set of refine_attrs (clear-text in the release, visible_level==0).
  3. Simulates the realistic scenario where the attacker does NOT know the
     column names: the script renames those columns to col_0, col_1, ... in
     the anonymized DataFrame.
  4. Uses Valentine to recover the mapping obfuscated -> KB-named column
     from the values alone.
  5. Compares the recovered mapping to the ground truth and prints metrics.
  6. Shows how to plug the renamed DataFrame back into the attack pipeline.

Run:
    python demo_schema_matching.py \
        --anonymized outputs/anonymized_public.csv \
        --auxiliary outputs/auxiliary/auxiliary_base.csv \
        --refine-attrs age,hours-per-week,capital-gain \
        --matcher coma
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from schema_matcher import (
    apply_recovered_mapping,
    evaluate_mapping,
    jaccard_baseline,
    obfuscate_columns,
    recover_column_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: recover obfuscated column names in the anonymized "
                    "release before phase-2 refinement of the linkage attack."
    )
    parser.add_argument(
        "--anonymized",
        required=True,
        help="Anonymized public CSV (what the attacker sees).",
    )
    parser.add_argument(
        "--auxiliary",
        required=True,
        help="Attacker's auxiliary knowledge base CSV (named columns).",
    )
    parser.add_argument(
        "--refine-attrs",
        required=True,
        help="Comma-separated list of clear-text attributes to obfuscate in "
             "the anonymized release and try to recover.",
    )
    parser.add_argument(
        "--matcher",
        choices=["coma", "jaccard", "distribution", "baseline_jaccard"],
        default="coma",
        help="Valentine matcher, or 'baseline_jaccard' for the pure-Python baseline.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.1,
        help="Minimum similarity score to keep a mapping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---------- 1. Load the two DataFrames ----------------------------------
    df_anon = pd.read_csv(
        Path(args.anonymized), dtype=str, keep_default_na=False
    )
    df_kb = pd.read_csv(
        Path(args.auxiliary), dtype=str, keep_default_na=False
    )
    refine_attrs = [c.strip() for c in args.refine_attrs.split(",") if c.strip()]

    missing_in_anon = [c for c in refine_attrs if c not in df_anon.columns]
    missing_in_kb = [c for c in refine_attrs if c not in df_kb.columns]
    if missing_in_anon:
        raise ValueError(f"Missing in anonymized CSV: {missing_in_anon}")
    if missing_in_kb:
        raise ValueError(f"Missing in auxiliary CSV: {missing_in_kb}")

    print(f"[load] anonymized shape = {df_anon.shape}")
    print(f"[load] auxiliary  shape = {df_kb.shape}")
    print(f"[load] refine_attrs     = {refine_attrs}")

    # ---------- 2. Simulate attacker POV: obfuscate the column names --------
    # The attacker sees col_0, col_1, ... instead of the real names. The truth
    # mapping is kept aside purely for evaluation.
    df_anon_obf, truth = obfuscate_columns(df_anon, refine_attrs, prefix="col_")
    obfuscated_cols = list(truth.keys())  # col_0, col_1, ...

    print("\n[obfuscate] ground-truth mapping (hidden from the matcher):")
    for k, v in truth.items():
        print(f"  {k} -> {v}")

    # ---------- 3. Run the schema matcher -----------------------------------
    # Candidates on the KB side: all columns of df_kb EXCEPT record_id and
    # anything the attacker would already know (typically the QID stage-1
    # columns). In this demo we keep it simple and consider every KB column
    # except record_id.
    kb_candidate_cols = [c for c in df_kb.columns if c != "record_id"]

    print(f"\n[match] using matcher = {args.matcher}")
    print(f"[match] candidates on KB side = {kb_candidate_cols}")

    if args.matcher == "baseline_jaccard":
        recovered = jaccard_baseline(
            df_anon=df_anon_obf,
            df_kb=df_kb,
            anon_unknown_cols=obfuscated_cols,
            kb_candidate_cols=kb_candidate_cols,
            min_score=args.min_score,
        )
    else:
        recovered = recover_column_mapping(
            df_anon=df_anon_obf,
            df_kb=df_kb,
            anon_unknown_cols=obfuscated_cols,
            kb_candidate_cols=kb_candidate_cols,
            matcher_name=args.matcher,  # type: ignore[arg-type]
            min_score=args.min_score,
        )

    print("\n[match] recovered mapping:")
    for anon_col, (kb_col, score) in recovered.items():
        correct = "OK " if truth.get(anon_col) == kb_col else "KO "
        print(f"  {correct} {anon_col} -> {kb_col}  (score={score:.4f}, "
              f"truth={truth.get(anon_col)})")

    # ---------- 4. Evaluate the matcher -------------------------------------
    metrics = evaluate_mapping(recovered, truth)
    print("\n[eval] metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s} = {v:.4f}")
        else:
            print(f"  {k:25s} = {v}")

    # ---------- 5. Plug back into the attack pipeline -----------------------
    # Rename the obfuscated columns back to the KB names so phase 2 can
    # consume df_anon_recovered exactly as if the names had never been hidden.
    df_anon_recovered, recovered_kb_cols = apply_recovered_mapping(
        df_anon=df_anon_obf,
        recovered=recovered,
    )
    # refine_attrs that the attacker can actually use in phase 2:
    effective_refine_attrs = [c for c in refine_attrs if c in recovered_kb_cols]
    dropped_refine_attrs = [c for c in refine_attrs if c not in recovered_kb_cols]

    print("\n[integration] effective refine_attrs for phase 2 =",
          effective_refine_attrs)
    if dropped_refine_attrs:
        print("[integration] dropped (schema matcher failed)   =",
              dropped_refine_attrs)
    print("[integration] df_anon_recovered is now ready to replace df_public "
          "in run_linkage_attack._evaluate_target()")


if __name__ == "__main__":
    main()
