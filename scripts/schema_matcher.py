"""
schema_matcher.py

Schema matching module for the linkage attack pipeline.

Scenario
--------
In phase 2 of the linkage attack, the refinement step works on clear-text
attributes (visible_level == 0). We want to model a realistic attacker who
does NOT know the column names of the anonymized release (they are obfuscated)
but still wants to use those columns because their values are readable.

Given:
  - df_anon  : the anonymized public release (some columns renamed col_0, ...).
  - df_kb    : the attacker's auxiliary knowledge base (columns named).
  - anon_unknown_cols : columns in df_anon whose names are unknown.
  - kb_candidate_cols : columns in df_kb that could correspond to them.

This module recovers, via Valentine, a mapping
    anon_col -> (kb_col, score)
so that the rest of the pipeline can rename df_anon's columns back to the
attacker's vocabulary and proceed with phase 2 as if the names were known.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd


MatcherName = Literal["coma", "jaccard", "distribution"]


# ---------------------------------------------------------------------------
# Valentine matcher factory
# ---------------------------------------------------------------------------


# Instantiate a Valentine matcher by name.
# All matchers here support instance-based matching, which is what we need
# since column names are obfuscated (pure schema-based would fail).
def build_valentine_matcher(name: MatcherName = "coma", **kwargs: Any):
    if name == "coma":
        from valentine.algorithms import Coma
        # use_instances=True => COMA compares values, not just column names.
        # max_n=0 means "do not cap the number of matches returned".
        from valentine.algorithms import ComaPy
        return ComaPy(
            use_instances=kwargs.get("use_instances", True),
            use_schema=kwargs.get("use_schema", False),  # cf. note below
            max_n=kwargs.get("max_n", 0),
        )
    if name == "jaccard":
        from valentine.algorithms import JaccardDistanceMatcher
        return JaccardDistanceMatcher(
            threshold_dist=kwargs.get("threshold_dist", 0.8),
        )
    if name == "distribution":
        from valentine.algorithms import DistributionBased
        return DistributionBased(
            threshold1=kwargs.get("threshold1", 0.15),
            threshold2=kwargs.get("threshold2", 0.15),
        )
    raise ValueError(f"Unknown matcher: {name}")


# ---------------------------------------------------------------------------
# Obfuscation (attacker POV simulation)
# ---------------------------------------------------------------------------


# Rename a subset of columns to anonymous identifiers. Returns the renamed
# DataFrame and the ground-truth mapping obfuscated_name -> original_name.
# The ground truth is kept aside for evaluation of the schema matcher; it is
# NEVER passed to the matcher itself.
def obfuscate_columns(
    df: pd.DataFrame,
    cols_to_obfuscate: list[str],
    prefix: str = "col_",
) -> tuple[pd.DataFrame, dict[str, str]]:
    truth: dict[str, str] = {}
    rename: dict[str, str] = {}
    for i, col in enumerate(cols_to_obfuscate):
        new_name = f"{prefix}{i}"
        truth[new_name] = col
        rename[col] = new_name
    return df.rename(columns=rename), truth


# ---------------------------------------------------------------------------
# Valentine-based column recovery
# ---------------------------------------------------------------------------


# Run Valentine on the two column subsets and return a one-to-one mapping
# anon_column -> (kb_column, score).
#
# Only columns whose similarity score is >= min_score are kept. Columns that
# Valentine cannot confidently match are dropped from the returned mapping.
def recover_column_mapping(
    df_anon: pd.DataFrame,
    df_kb: pd.DataFrame,
    anon_unknown_cols: list[str],
    kb_candidate_cols: list[str],
    matcher_name: MatcherName = "coma",
    matcher_kwargs: dict[str, Any] | None = None,
    min_score: float = 0.0,
) -> dict[str, tuple[str, float]]:
    from valentine import valentine_match

    matcher = build_valentine_matcher(matcher_name, **(matcher_kwargs or {}))

    df_anon_sub = df_anon[anon_unknown_cols].copy()
    df_kb_sub = df_kb[kb_candidate_cols].copy()

    try:
        matches = valentine_match([df_anon_sub, df_kb_sub], matcher)
    except TypeError:
        matches = valentine_match(df_anon_sub, df_kb_sub, matcher)


    try:
        one_to_one = matches.one_to_one()
    except AttributeError:
        one_to_one = matches

    recovered: dict[str, tuple[str, float]] = {}
    for pair, score in one_to_one.items():
        if score < min_score:
            continue
        if hasattr(pair, "source_column"):
            source_col = pair.source_column
            target_col = pair.target_column
        else:
            source_col = pair[0][1]
            target_col = pair[1][1]
        recovered[source_col] = (target_col, float(score))
    return recovered


# ---------------------------------------------------------------------------
# Jaccard baseline (no external dependency)
# ---------------------------------------------------------------------------


# Greedy Jaccard baseline on value sets. Useful as a sanity check against
# Valentine and for categorical attributes where the attacker's KB and the
# published dataset share many overlapping values.
def jaccard_baseline(
    df_anon: pd.DataFrame,
    df_kb: pd.DataFrame,
    anon_unknown_cols: list[str],
    kb_candidate_cols: list[str],
    min_score: float = 0.1,
) -> dict[str, tuple[str, float]]:
    recovered: dict[str, tuple[str, float]] = {}
    for anon_col in anon_unknown_cols:
        set_a = {str(v).strip() for v in df_anon[anon_col].astype(str).unique()}
        best_kb: str | None = None
        best_score: float = -1.0
        for kb_col in kb_candidate_cols:
            set_b = {str(v).strip() for v in df_kb[kb_col].astype(str).unique()}
            union = set_a | set_b
            score = 0.0 if not union else len(set_a & set_b) / len(union)
            if score > best_score:
                best_score = score
                best_kb = kb_col
        if best_kb is not None and best_score >= min_score:
            recovered[anon_col] = (best_kb, best_score)
    return recovered


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


# Compare a recovered mapping to the obfuscation ground truth.
# Returns coverage (fraction of obfuscated cols for which a mapping was
# returned) and accuracy (fraction of returned mappings that are correct).
def evaluate_mapping(
    recovered: dict[str, tuple[str, float]],
    truth: dict[str, str],
) -> dict[str, Any]:
    n_truth = len(truth)
    n_mapped = len(recovered)
    n_correct = sum(
        1 for anon_col, (kb_col, _) in recovered.items()
        if truth.get(anon_col) == kb_col
    )
    return {
        "n_obfuscated": n_truth,
        "n_mapped": n_mapped,
        "n_correct": n_correct,
        "coverage": None if n_truth == 0 else n_mapped / n_truth,
        "accuracy_on_mapped": None if n_mapped == 0 else n_correct / n_mapped,
        "recall": None if n_truth == 0 else n_correct / n_truth,
    }


# ---------------------------------------------------------------------------
# Utility: apply a recovered mapping to rename anonymized columns back
# ---------------------------------------------------------------------------


# Rename the anonymized DataFrame's obfuscated columns back to the KB names.
# Columns that were not recovered are left with their obfuscated name so the
# caller can decide to drop them from refine_attrs.
def apply_recovered_mapping(
    df_anon: pd.DataFrame,
    recovered: dict[str, tuple[str, float]],
) -> tuple[pd.DataFrame, list[str]]:
    rename = {anon_col: kb_col for anon_col, (kb_col, _) in recovered.items()}
    df_renamed = df_anon.rename(columns=rename)
    recovered_kb_cols = list(rename.values())
    return df_renamed, recovered_kb_cols
