from __future__ import annotations

import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from collections.abc import Mapping


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


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_path(base_dir: str | Path, value: str | Path) -> Path:
    value = Path(value)
    if value.is_absolute():
        return value
    return (Path(base_dir) / value).resolve()


def build_hierarchy_mapping(base_dir: str | Path, hierarchy_dir: str | Path, quasi_identifiers: list[str]) -> dict[str, str]:
    hierarchy_root = resolve_path(base_dir, hierarchy_dir)
    mapping: dict[str, str] = {}
    for qi in quasi_identifiers:
        csv_path = hierarchy_root / f"{qi}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Hierarchy file not found for '{qi}': {csv_path}")
        mapping[qi] = str(csv_path)
    return mapping


def make_experiment_id(
    quasi_identifiers: list[str],
    k: int | None,
    l: int | None,
    t: float | None,
    suppression_limit: int | None,
    backend: str | None,
) -> str:
    qi_part = "-".join(quasi_identifiers)
    return f"qi_{qi_part}__k_{k}__l_{l}__t_{t}__supp_{suppression_limit}__{backend}"


def iter_qi_subsets(qi_pool: list[str], subset_sizes: list[int]) -> list[list[str]]:
    subsets: list[list[str]] = []
    for size in subset_sizes:
        for combo in itertools.combinations(qi_pool, size):
            subsets.append(list(combo))
    return subsets


def safe_call(obj: Any, method_name: str) -> Any:
    method = getattr(obj, method_name, None)
    if callable(method):
        try:
            return method()
        except Exception as exc:  # noqa: BLE001
            return f"<error: {exc}>"
    return None


def collect_result_metrics(result: Any) -> dict[str, Any]:
    return {
        "anonymization_time_ms": safe_call(result, "get_anonymization_time"),
        "transformations": safe_call(result, "get_transformations"),
        "number_of_equivalence_classes": safe_call(result, "get_number_of_equivalence_classes"),
        "average_equivalence_class_size": safe_call(result, "get_average_equivalence_class_size"),
        "min_equivalence_class_size": safe_call(result, "get_min_equivalence_class_size"),
        "max_equivalence_class_size": safe_call(result, "get_max_equivalence_class_size"),
        "number_of_suppressed_records": safe_call(result, "get_number_of_suppressed_records"),
        "discernibility_metric": safe_call(result, "get_discernibility_metric"),
        "ambiguity_metric": safe_call(result, "get_ambiguity_metric"),
        "average_class_size_metric": safe_call(result, "get_average_class_size_metric"),
        "granularity_metric": safe_call(result, "get_granularity_metric"),
        "non_uniform_entropy_metric": safe_call(result, "get_non_uniform_entropy_metric"),
    }


def sanitize_row_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in row.items():
        value = to_jsonable(value)
        if isinstance(value, (list, dict)):
            sanitized[key] = json.dumps(value, ensure_ascii=False)
        else:
            sanitized[key] = value
    return sanitized
