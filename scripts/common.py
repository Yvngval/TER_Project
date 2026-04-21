# Shared utility functions for configuration, paths, metrics, and output serialization.

from __future__ import annotations

import itertools
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from collections.abc import Mapping

logger = logging.getLogger(__name__)


# Parse a comma-separated CLI list into a clean Python list.
def parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


# Convert an object into a JSON-serializable value.
def to_jsonable(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, Mapping) or hasattr(obj, "items"):
        try:
            items = obj.items()
        except Exception:
            items = list(obj.items())
        return {str(to_jsonable(k)): to_jsonable(v) for k, v in items}

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]

    if hasattr(obj, "__iter__") and not isinstance(obj, (bytes, bytearray)):
        try:
            return [to_jsonable(x) for x in obj]
        except Exception:
            pass

    for caster in (int, float):
        try:
            return caster(obj)
        except Exception:
            pass

    return str(obj)


# Load a JSON file from disk.
def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Save a Python object to a JSON file.
def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)


# Create a directory if it does not already exist.
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Generate a timestamp string for naming outputs.
def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# Resolve a path relative to a base directory.
def resolve_path(base_dir: str | Path, value: str | Path) -> Path:
    value = Path(value)
    if value.is_absolute():
        return value
    return (Path(base_dir) / value).resolve()


# Map each quasi-identifier to its hierarchy CSV file.
def build_hierarchy_mapping(base_dir: str | Path, hierarchy_dir: str | Path, quasi_identifiers: list[str]) -> dict[str, str]:
    hierarchy_root = resolve_path(base_dir, hierarchy_dir)
    mapping: dict[str, str] = {}
    for qi in quasi_identifiers:
        csv_path = hierarchy_root / f"{qi}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Hierarchy file not found for '{qi}': {csv_path}")
        mapping[qi] = str(csv_path)
    return mapping


# Build a unique name for an anonymization experiment.
def make_experiment_id(
    quasi_identifiers: list[str],
    k: int | None,
    l: int | None,
    t: float | None,
    suppression_limit: int | None,
    backend: str | None,
    utility_measure: str | None = None,
    utility_aggregate: str | None = None,
) -> str:
    qi_part = "-".join(quasi_identifiers)
    base = f"qi_{qi_part}__k_{k}__l_{l}__t_{t}__supp_{suppression_limit}__{backend}"
    if utility_measure and utility_measure != "loss":
        base += f"__m_{utility_measure}"
    if utility_aggregate:
        base += f"__agg_{utility_aggregate}"
    return base


# Generate all QI subsets for the requested subset sizes.
def iter_qi_subsets(qi_pool: list[str], subset_sizes: list[int]) -> list[list[str]]:
    subsets: list[list[str]] = []
    for size in subset_sizes:
        for combo in itertools.combinations(qi_pool, size):
            subsets.append(list(combo))
    return subsets


# Safely call a method and return its result, or None on any failure.
def safe_call(obj: Any, method_name: str, *args: Any) -> Any:
    method = getattr(obj, method_name, None)
    if callable(method):
        try:
            return method(*args)
        except Exception as exc:
            logger.warning("safe_call(%r) failed: %s", method_name, exc)
            return None
    return None


# Collect a per-attribute metric for each quasi-identifier.
def _collect_per_attribute_metric(
    result: Any, method_name: str, quasi_identifiers: list[str],
) -> dict[str, Any]:
    return {
        qi: safe_call(result, method_name, qi)
        for qi in quasi_identifiers
    }


# Collect standard metrics from an anonymization result object.
def collect_result_metrics(
    result: Any, quasi_identifiers: list[str] | None = None,
) -> dict[str, Any]:
    quasi_identifiers = quasi_identifiers or []

    metrics: dict[str, Any] = {
        # Optimization scores from the globally optimal ARX node
        "optimization_score_min": safe_call(result, "get_optimization_score_min"),
        "optimization_score_max": safe_call(result, "get_optimization_score_max"),
        "anonymization_time_ms": safe_call(result, "get_anonymization_time"),
        "transformations": safe_call(result, "get_transformations"),
        "number_of_equivalence_classes": safe_call(result, "get_number_of_equivalence_classes"),
        "average_equivalence_class_size": safe_call(result, "get_average_equivalence_class_size"),
        "min_equivalence_class_size": safe_call(result, "get_min_equivalence_class_size"),
        "max_equivalence_class_size": safe_call(result, "get_max_equivalence_class_size"),
        "number_of_suppressed_records": safe_call(result, "get_number_of_suppressed_records"),
        # Global metrics
        "discernibility_metric": safe_call(result, "get_discernibility_metric"),
        "ambiguity_metric": safe_call(result, "get_ambiguity_metric"),
        "average_class_size_metric": safe_call(result, "get_average_class_size_metric"),
        "ssesst_metric": safe_call(result, "get_ssesst_metric"),
        "record_level_squared_error_metric": safe_call(result, "get_record_level_squared_error_metric"),
        # Per-attribute metrics
        "granularity_metric": _collect_per_attribute_metric(result, "get_granularity_metric", quasi_identifiers),
        "non_uniform_entropy_metric": _collect_per_attribute_metric(result, "get_non_uniform_entropy_metric", quasi_identifiers),
        "generalization_intensity_metric": _collect_per_attribute_metric(result, "get_generalization_intensity_metric", quasi_identifiers),
        "attribute_level_squared_error_metric": _collect_per_attribute_metric(result, "get_attribute_level_squared_error_metric", quasi_identifiers),
        "missings_metric": _collect_per_attribute_metric(result, "get_missings_metric", quasi_identifiers),
    }

    return metrics


# Convert a row into a CSV-safe dictionary.
def sanitize_row_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in row.items():
        value = to_jsonable(value)
        if isinstance(value, (list, dict)):
            sanitized[key] = json.dumps(value, ensure_ascii=False)
        else:
            sanitized[key] = value
    return sanitized
