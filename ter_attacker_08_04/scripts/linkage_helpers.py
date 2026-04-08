# Matching and reporting helpers for the linkage attack.

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from attack_common import is_suppressed_value
from privjedai_utils import compute_privjedai_similarity, is_attr_clear_for_fuzzy


# Build an inverted index: for each attribute value, store the row indices where it appears.
def build_value_indices(df: pd.DataFrame, attrs: list[str]) -> dict[str, dict[str, np.ndarray]]:
    indices: dict[str, dict[str, np.ndarray]] = {}
    for attr in attrs:
        per_value: dict[str, list[int]] = {}
        for idx, value in enumerate(df[attr].astype(str).tolist()):
            per_value.setdefault(str(value).strip(), []).append(idx)
        indices[attr] = {
            value: np.asarray(row_ids, dtype=np.int32)
            for value, row_ids in per_value.items()
        }
    return indices


# Determine whether one attribute value is compatible and classify the match type.
def attribute_match_result(
    raw_value: str,
    anonymized_value: str,
    attacker_attr_knowledge: dict[str, Any] | None,
    *,
    fuzzy_config: dict[str, Any] | None,
    fuzzy_pair_cache: dict[tuple[str, str], float],
    fuzzy_hash_cache: dict[str, frozenset[int]],
) -> dict[str, Any] | None:
    raw_value = str(raw_value).strip()
    anonymized_value = str(anonymized_value).strip()

    if raw_value == anonymized_value:
        return {"kind": "exact", "score": None}

    if is_suppressed_value(anonymized_value):
        return {"kind": "suppressed", "score": None}

    projection = {}
    if attacker_attr_knowledge is not None:
        projection = attacker_attr_knowledge.get("projection", {})

    exposed_value = projection.get(raw_value, raw_value)
    if anonymized_value == exposed_value:
        return {
            "kind": "exact" if exposed_value == raw_value else "generalized",
            "score": None,
        }

    if fuzzy_config is None or not is_attr_clear_for_fuzzy(attacker_attr_knowledge):
        return None

    score = compute_privjedai_similarity(
        raw_value,
        anonymized_value,
        fuzzy_config=fuzzy_config,
        pair_cache=fuzzy_pair_cache,
        hash_cache=fuzzy_hash_cache,
    )
    if score >= float(fuzzy_config["threshold"]):
        return {"kind": "privjedai_fuzzy", "score": float(score)}
    return None


# Compute and cache the compatibility mapping for one target attribute value.
def get_match_mapping_for_target_value(
    attr: str,
    target_value: str,
    possible_anonymized_values: list[str],
    attacker_knowledge: dict[str, dict[str, Any]],
    match_cache: dict[tuple[str, str, str], dict[str, dict[str, Any]]],
    *,
    fuzzy_config: dict[str, Any] | None,
    fuzzy_pair_cache: dict[tuple[str, str], float],
    fuzzy_hash_cache: dict[str, frozenset[int]],
) -> dict[str, dict[str, Any]]:
    fuzzy_cache_tag = "no_fuzzy"
    if fuzzy_config is not None:
        fuzzy_cache_tag = (
            f"fuzzy::{fuzzy_config['metric']}::thr_{fuzzy_config['threshold']}::"
            f"size_{fuzzy_config['bloom_size']}::hashes_{fuzzy_config['num_hashes']}::"
            f"qgrams_{fuzzy_config['qgrams']}::hashing_{fuzzy_config['hashing_type']}"
        )

    cache_key = (attr, str(target_value).strip(), fuzzy_cache_tag)
    if cache_key not in match_cache:
        attacker_attr_knowledge = attacker_knowledge.get(attr)
        mapping: dict[str, dict[str, Any]] = {}
        for anonymized_value in possible_anonymized_values:
            result = attribute_match_result(
                target_value,
                anonymized_value,
                attacker_attr_knowledge,
                fuzzy_config=fuzzy_config,
                fuzzy_pair_cache=fuzzy_pair_cache,
                fuzzy_hash_cache=fuzzy_hash_cache,
            )
            if result is not None:
                mapping[str(anonymized_value).strip()] = result
        match_cache[cache_key] = mapping
    return match_cache[cache_key]


# Compute the sensitive-value probability distribution inside one equivalence class.
def compute_sensitive_distribution(candidate_df: pd.DataFrame, sensitive_attr: str) -> dict[str, float]:
    if candidate_df.empty:
        return {}
    counts = candidate_df[sensitive_attr].astype(str).value_counts(dropna=False)
    ordered_items = sorted(
        ((str(value), int(count)) for value, count in counts.items()),
        key=lambda item: (-item[1], item[0]),
    )
    total = sum(count for _, count in ordered_items)
    return {value: count / total for value, count in ordered_items}


# Summarize the sensitive inference made from one equivalence class.
def summarize_sensitive_prediction(candidate_df: pd.DataFrame, sensitive_attr: str) -> dict[str, Any]:
    distribution = compute_sensitive_distribution(candidate_df, sensitive_attr)
    if not distribution:
        return {
            "distribution": {},
            "top_value": None,
            "top_probability": None,
            "is_certain": None,
            "n_distinct_sensitive_values": 0,
        }

    top_value, top_probability = next(iter(distribution.items()))
    return {
        "distribution": distribution,
        "top_value": top_value,
        "top_probability": float(top_probability),
        "is_certain": len(distribution) == 1,
        "n_distinct_sensitive_values": int(len(distribution)),
    }
