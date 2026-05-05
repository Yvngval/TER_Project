# Optional privJedAI fuzzy-matching helpers used by the linkage attack.

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any


# Import only the Bloom encoder part of privJedAI.
def import_privjedai_bloom(privjedai_src: str | Path | None = None) -> dict[str, Any]:
    candidate_paths: list[Path] = []
    if privjedai_src:
        candidate_paths.append(Path(privjedai_src).resolve())

    env_path = os.environ.get("PRIVJEDAI_SRC")
    if env_path:
        candidate_paths.append(Path(env_path).resolve())

    script_dir = Path(__file__).resolve().parent
    candidate_paths.extend(
        [
            (script_dir.parent / "privJedAI-main" / "src").resolve(),
            (script_dir / "privJedAI-main" / "src").resolve(),
            (Path.cwd() / "privJedAI-main" / "src").resolve(),
        ]
    )

    for path in candidate_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    try:
        from privjedai.encoder import BloomFilter, BloomFilterConfig
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "privJedAI fuzzy fallback is enabled but privJedAI or one of its dependencies is missing. "
            "Install the required dependencies first, for example:\n"
            "  python -m pip install bitarray ray Metaphone tqdm pandas numpy\n"
            "Then pass --privjedai-src path\\to\\privJedAI-main\\src if the package is not installed.\n"
            f"Original import error: {exc}"
        ) from exc

    return {
        "BloomFilter": BloomFilter,
        "BloomFilterConfig": BloomFilterConfig,
    }


# Build the fuzzy configuration and the underlying Bloom encoder.
def build_privjedai_fuzzy_config(
    *,
    privjedai_src: str | Path | None,
    threshold: float,
    metric: str,
    bloom_size: int,
    bloom_num_hashes: int,
    bloom_qgrams: int,
    bloom_hashing_type: str,
) -> dict[str, Any]:
    imported = import_privjedai_bloom(privjedai_src)
    BloomFilter = imported["BloomFilter"]
    BloomFilterConfig = imported["BloomFilterConfig"]
    encoder = BloomFilter(
        BloomFilterConfig(
            size=bloom_size,
            offset=0,
            num_hashes=bloom_num_hashes,
            hashing_type=bloom_hashing_type,
            salt="",
            attributes=None,
            qgrams=bloom_qgrams,
        )
    )
    return {
        "encoder": encoder,
        "threshold": float(threshold),
        "metric": str(metric),
        "bloom_size": int(bloom_size),
        "num_hashes": int(bloom_num_hashes),
        "qgrams": int(bloom_qgrams),
        "hashing_type": str(bloom_hashing_type),
        "src": None if privjedai_src is None else str(privjedai_src),
    }


# Return True only for attributes that remain visible at raw level.
def is_attr_clear_for_fuzzy(attacker_attr_knowledge: dict[str, Any] | None) -> bool:
    if attacker_attr_knowledge is None:
        return True
    return int(attacker_attr_knowledge.get("visible_level", 0)) == 0


# Compute a set-based similarity from Bloom filter active positions.
def compute_bloom_similarity(
    bits_a: frozenset[int],
    bits_b: frozenset[int],
    *,
    metric: str,
    bloom_size: int,
) -> float:
    inter = len(bits_a & bits_b)
    a = len(bits_a)
    b = len(bits_b)

    if metric == "dice":
        denom = a + b
        return 0.0 if denom == 0 else (2.0 * inter) / denom

    if metric == "jaccard":
        union = len(bits_a | bits_b)
        return 0.0 if union == 0 else inter / union

    if metric == "cosine":
        denom = math.sqrt(a * b)
        return 0.0 if denom == 0 else inter / denom

    if metric == "scm":
        xor_count = a + b - (2 * inter)
        return max(0.0, (float(bloom_size) - float(xor_count)) / float(bloom_size))

    raise ValueError(f"Unknown privJedAI fuzzy metric: {metric}")


# Encode one string once and reuse it through a cache.
def get_privjedai_bits(
    value: str,
    *,
    encoder: Any,
    hash_cache: dict[str, frozenset[int]],
) -> frozenset[int]:
    key = str(value).strip()
    cached = hash_cache.get(key)
    if cached is None:
        cached = frozenset(int(bit) for bit in encoder.generate_hash(key))
        hash_cache[key] = cached
    return cached


# Compute a privJedAI similarity for one pair of strings.
def compute_privjedai_similarity(
    raw_value: str,
    anonymized_value: str,
    *,
    fuzzy_config: dict[str, Any],
    pair_cache: dict[tuple[str, str], float],
    hash_cache: dict[str, frozenset[int]],
) -> float:
    key = (str(raw_value).strip(), str(anonymized_value).strip())
    cached = pair_cache.get(key)
    if cached is not None:
        return cached

    bits_a = get_privjedai_bits(key[0], encoder=fuzzy_config["encoder"], hash_cache=hash_cache)
    bits_b = get_privjedai_bits(key[1], encoder=fuzzy_config["encoder"], hash_cache=hash_cache)
    score = compute_bloom_similarity(
        bits_a,
        bits_b,
        metric=str(fuzzy_config["metric"]),
        bloom_size=int(fuzzy_config["bloom_size"]),
    )
    pair_cache[key] = float(score)
    return float(score)
