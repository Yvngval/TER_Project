# Shared helpers used by the linkage and MIA attacks.

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from common import load_json

SUPPRESSION_TOKENS = {"", "*"}


# Repair one runtime path when the config was generated on another machine.
def _repair_runtime_path(raw_path: str | Path, fallback_dirs: list[Path]) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path.resolve()
    raw_name = str(raw_path).replace("\\", "/").split("/")[-1]
    for base_dir in fallback_dirs:
        candidate = (base_dir / raw_name).resolve()
        if candidate.exists():
            return candidate
    return path


# Load a CSV file as strings only and normalize headers / cell values.
def read_csv_str(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(path), dtype=str, keep_default_na=False)
    df.columns = [str(col).strip() for col in df.columns]
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df


# Load and validate the runtime anonymization config.
def load_runtime_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    payload = load_json(config_path)
    if "hierarchies" not in payload:
        raise ValueError(
            "The supplied config must be a runtime config containing a 'hierarchies' mapping. "
            "Use a generated config from outputs/configs/... rather than the original base config."
        )

    project_root = config_path.parent.parent.parent
    hierarchy_dir_candidates = [project_root / "hierarchies", config_path.parent, Path.cwd() / "hierarchies"]
    data_dir_candidates = [project_root / "data", config_path.parent, Path.cwd() / "data"]

    if "data" in payload:
        payload["data"] = str(_repair_runtime_path(payload["data"], data_dir_candidates))

    repaired_hierarchies: dict[str, str] = {}
    for attr, hierarchy_path in payload.get("hierarchies", {}).items():
        repaired_hierarchies[str(attr)] = str(_repair_runtime_path(hierarchy_path, hierarchy_dir_candidates))
    payload["hierarchies"] = repaired_hierarchies
    return payload


# Check whether one anonymized value is an explicit suppression token.
def is_suppressed_value(value: str) -> bool:
    value = str(value).strip()
    if value in SUPPRESSION_TOKENS:
        return True
    return bool(value) and set(value) == {"*"}


# Load all rows from one hierarchy CSV file.
def load_hierarchy_rows(hierarchy_path: str | Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with Path(hierarchy_path).open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            cleaned = [str(cell).strip() for cell in row if str(cell).strip()]
            if cleaned:
                rows.append(cleaned)
    return rows


# Infer the most specific hierarchy level compatible with the values visible in the anonymized CSV.
def infer_last_visible_level(hierarchy_rows: list[list[str]], observed_values: list[str]) -> int:
    observed = {str(value).strip() for value in observed_values if not is_suppressed_value(value)}
    if not observed:
        return 0

    max_depth = max(len(row) for row in hierarchy_rows)
    compatible_levels: list[int] = []
    for level in range(max_depth):
        labels = {str(row[level]).strip() for row in hierarchy_rows if level < len(row)}
        if observed.issubset(labels):
            compatible_levels.append(level)

    if not compatible_levels:
        return 0
    return min(compatible_levels)


# Build the reduced attacker projection for one attribute.
def build_attacker_projection_for_attr(
    *,
    attr: str,
    hierarchy_path: str | Path,
    observed_anonymized_values: pd.Series,
) -> dict[str, Any]:
    hierarchy_rows = load_hierarchy_rows(hierarchy_path)
    observed_values = observed_anonymized_values.astype(str).unique().tolist()
    visible_level = infer_last_visible_level(hierarchy_rows, observed_values)

    projection: dict[str, str] = {}
    for row in hierarchy_rows:
        raw_value = str(row[0]).strip()
        exposed_value = str(row[min(visible_level, len(row) - 1)]).strip()
        projection[raw_value] = exposed_value

    return {
        "attribute": attr,
        "hierarchy_path": str(hierarchy_path),
        "visible_level": int(visible_level),
        "observed_values": sorted({str(v).strip() for v in observed_values}),
        "projection": projection,
        "n_projected_values": int(len(projection)),
    }


# Build the reduced attacker knowledge for all attacker-known attributes.
def build_attacker_knowledge(
    *,
    runtime: dict[str, Any],
    known_attrs: list[str],
    df_public: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    hierarchies = runtime.get("hierarchies", {})
    knowledge: dict[str, dict[str, Any]] = {}
    for attr in known_attrs:
        hierarchy_path = hierarchies.get(attr)
        if hierarchy_path:
            knowledge[attr] = build_attacker_projection_for_attr(
                attr=attr,
                hierarchy_path=hierarchy_path,
                observed_anonymized_values=df_public[attr].astype(str),
            )
        else:
            projection = {str(v).strip(): str(v).strip() for v in df_public[attr].astype(str).unique().tolist()}
            knowledge[attr] = {
                "attribute": attr,
                "hierarchy_path": None,
                "visible_level": 0,
                "observed_values": sorted({str(v).strip() for v in df_public[attr].astype(str).tolist()}),
                "projection": projection,
                "n_projected_values": int(len(projection)),
            }
    return knowledge


# Append one attack row to a CSV summary.
def append_attack_summary(summary_csv: str | Path, row: dict[str, Any]) -> None:
    summary_csv = Path(summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
