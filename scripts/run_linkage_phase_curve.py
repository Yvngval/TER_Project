r"""
run_linkage_phase_curve.py

Generate N random release configurations for the Adult dataset, run the full
pipeline per configuration (anonymization -> auxiliary base -> linkage attack),
then aggregate phase-1 and phase-2 equivalence class sizes grouped by the
number of attacker-known attributes that ARX ended up generalizing (visible
level != 0), and plot the two curves.

Sampling logic (one configuration = one experiment)
---------------------------------------------------
For each run we independently draw:

  1) A random number N_qi in [--min-qi, --max-qi] of quasi-identifiers,
     sampled uniformly without replacement from the QI pool (attributes that
     actually have a hierarchy CSV under the hierarchy directory).

  2) From the attributes *remaining* after the QI pick (i.e. every column in
     the dataset except the sensitive attribute, the base-config identifiers,
     record_id, and the problematic attributes listed via
     --problematic-insensitive), a random number N_ins in
     [--min-insensitive, --max-insensitive] of insensitive attributes,
     sampled uniformly without replacement.

  3) All other columns become *identifiers* and are stripped from the public
     release. record_id is always kept as insensitive so the linkage attack
     can evaluate matches against ground truth.

The attacker's known attributes for a run are  chosen_qi U chosen_insensitive
(everything that is visible in the release and could plausibly be linked on).

For each run the script records how many of those known attributes ARX
generalized or suppressed (visible_level != 0). The final plot groups
experiments by that number and shows the median stage-1 and stage-2
equivalence class sizes. The legend displays the k / l / suppression limit
read from the base config so the reader sees the anonymization regime at a
glance.

Outputs under  <output-root>/linkage_phase_curve/ :
    <stem>__plan.csv         : the N configurations (QI + insensitive sets)
    <stem>__runs.csv         : per-run metrics (written after each run so a
                               crash mid-sweep does not lose completed work)
    <stem>__aggregated.csv   : median/mean/std/min/max of stage1/stage2 per
                               n_generalized_known_attrs
    <stem>__plot.png         : the two-curve figure (x-axis 0..8 ascending)
    configs/<run_name>.json  : the runtime config used for each run
    auxiliary/<run_name>__aux.csv : the auxiliary base built for each run

Example (Windows PowerShell, from the project root):
    python .\scripts\run_linkage_phase_curve.py `
        --base-config .\configs\adult_base.json `
        --full-dataset .\data\adult_with_record_id.csv `
        --output-root .\outputs `
        --n-configurations 50 `
        --min-qi 2 --max-qi 8 `
        --min-insensitive 0 --max-insensitive 5 `
        --problematic-insensitive fnlwgt,education-num `
        --sample-frac 0.01 `
        --n-targets all `
        --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

# Allow importing the project's own scripts/ modules when launched from anywhere.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import ensure_dir, load_json, save_json  # type: ignore
from make_auxiliary_base import build_auxiliary_base  # type: ignore
from run_ano import run_one_experiment_from_config  # type: ignore
from run_linkage_attack import run_linkage_attack_from_paths  # type: ignore


# Adult's 9 quasi-identifiers (must have a matching <qi>.csv under hierarchies/).
DEFAULT_QI_POOL = [
    "native-country",
    "sex",
    "race",
    "marital-status",
    "workclass",
    "education",
    "occupation",
    "relationship",
    "age",
]

# Insensitive attributes that are "too problematic" for Adult and should never
# be passed clear in the release:
#   - fnlwgt       : census final weight, near-unique per record.
#   - education-num: ordinal numeric duplicate of `education`, trivial leakage.
# The CLI exposes this as a configurable list; pass "" to disable.
DEFAULT_PROBLEMATIC_INSENSITIVE = ["fnlwgt", "education-num"]

# Columns that must always be protected as insensitive (kept clear, not generalized).
# record_id is the linkage ground-truth key; it must reach the eval CSV intact
# but must NOT leak into the public release, hence the explicit drop in
# run_one_experiment_from_config(public_drop_columns=["record_id"]).
ALWAYS_INSENSITIVE = ["record_id"]


# ---------------------------------------------------------------------------
# Configuration sampling
# ---------------------------------------------------------------------------


def sample_random_configurations(
    *,
    qi_pool: list[str],
    all_columns: list[str],
    sensitive_attributes: list[str],
    base_identifiers: list[str],
    problematic_insensitive: list[str],
    n_configurations: int,
    min_qi: int,
    max_qi: int,
    min_insensitive: int,
    max_insensitive: int | None,
    seed: int,
) -> list[dict[str, list[str]]]:
    """
    Return `n_configurations` unique {qi_subset, insensitive_subset} draws.

    For each draw:
      * qi_subset          : rng.randint(min_qi, max_qi) names sampled without
                             replacement from qi_pool.
      * insensitive_subset : rng.randint(min_insensitive, effective_max_ins)
                             names sampled without replacement from columns
                             that are NOT in {qi_subset, sensitive_attributes,
                             base_identifiers, record_id, problematic}.

    Uniqueness is enforced on the unordered pair (frozenset(qi), frozenset(ins)).
    """
    if min_qi < 2:
        raise ValueError("min_qi must be >= 2.")
    if max_qi < min_qi:
        raise ValueError(f"max_qi ({max_qi}) must be >= min_qi ({min_qi}).")
    if max_qi > len(qi_pool):
        raise ValueError(
            f"max_qi ({max_qi}) cannot exceed the QI pool size ({len(qi_pool)})."
        )
    if min_insensitive < 0:
        raise ValueError("min_insensitive must be >= 0.")
    if max_insensitive is not None and max_insensitive < min_insensitive:
        raise ValueError(
            f"max_insensitive ({max_insensitive}) must be >= min_insensitive "
            f"({min_insensitive})."
        )

    protected = (
        set(sensitive_attributes)
        | set(base_identifiers)
        | set(ALWAYS_INSENSITIVE)
        | set(problematic_insensitive)
    )

    rng = random.Random(seed)
    seen: set[tuple[frozenset[str], frozenset[str]]] = set()
    configs: list[dict[str, list[str]]] = []

    # Upper bound on attempts: defend against pathologically small search spaces.
    max_attempts = max(n_configurations * 200, 5_000)
    attempts = 0
    while len(configs) < n_configurations and attempts < max_attempts:
        attempts += 1
        n_qi = rng.randint(min_qi, max_qi)
        qi_subset = rng.sample(qi_pool, n_qi)

        eligible_insensitive = [
            c
            for c in all_columns
            if c not in protected and c not in qi_subset
        ]
        effective_max_ins = (
            len(eligible_insensitive)
            if max_insensitive is None
            else min(max_insensitive, len(eligible_insensitive))
        )
        lo = min(min_insensitive, effective_max_ins)
        n_ins = rng.randint(lo, effective_max_ins) if effective_max_ins >= lo else 0
        insensitive_subset = rng.sample(eligible_insensitive, n_ins)

        key = (frozenset(qi_subset), frozenset(insensitive_subset))
        if key in seen:
            continue
        seen.add(key)

        # Report QIs size-descending then alphabetical for stable plan output.
        qi_sorted = sorted(qi_subset)
        ins_sorted = sorted(insensitive_subset)
        configs.append({"quasi_identifiers": qi_sorted, "insensitive_attributes": ins_sorted})

    if len(configs) < n_configurations:
        raise RuntimeError(
            f"Could only draw {len(configs)} unique configurations after "
            f"{attempts} attempts; reduce --n-configurations or widen the "
            f"QI/insensitive ranges."
        )

    # Plan ordering: QI size descending, then alphabetical on the joined names.
    configs.sort(
        key=lambda c: (-len(c["quasi_identifiers"]),
                        "|".join(c["quasi_identifiers"]),
                        "|".join(c["insensitive_attributes"]))
    )
    return configs


# ---------------------------------------------------------------------------
# Per-run runtime-config construction
# ---------------------------------------------------------------------------


def build_runtime_config_payload(
    *,
    base_config: dict[str, Any],
    full_dataset_path: Path,
    qi_subset: list[str],
    insensitive_subset: list[str],
    all_columns: list[str],
) -> dict[str, Any]:
    """
    Build the config payload for one anonymization run:

    - `quasi_identifiers`      = the chosen QI subset
    - `sensitive_attributes`   = inherited from the base config (e.g. ["income"])
    - `insensitive_attributes` = the chosen insensitive subset, plus any column
                                 in ALWAYS_INSENSITIVE (record_id) that actually
                                 exists in the dataset
    - `identifiers`            = base identifiers + every remaining column
                                 (those columns are stripped from the release)
    """
    base_identifiers = list(base_config.get("identifiers") or [])
    sensitive_attributes = list(base_config.get("sensitive_attributes") or [])

    always_ins_present = [c for c in ALWAYS_INSENSITIVE if c in all_columns]
    insensitive_attributes = sorted(set(insensitive_subset) | set(always_ins_present))

    classified = (
        set(base_identifiers)
        | set(qi_subset)
        | set(sensitive_attributes)
        | set(insensitive_attributes)
    )
    # Anything not classified yet becomes an identifier (dropped from the release).
    extra_identifiers = [c for c in all_columns if c not in classified]
    identifiers = list(base_identifiers) + extra_identifiers

    payload = dict(base_config)
    payload.pop("hierarchies", None)  # re-built by run_ano.build_runtime_config
    payload["data"] = str(full_dataset_path)
    payload["identifiers"] = identifiers
    payload["quasi_identifiers"] = list(qi_subset)
    payload["sensitive_attributes"] = sensitive_attributes
    payload["insensitive_attributes"] = insensitive_attributes
    return payload


# ---------------------------------------------------------------------------
# Single pipeline: anonymize -> auxiliary -> linkage
# ---------------------------------------------------------------------------


def run_one_configuration(
    *,
    output_root: Path,
    full_dataset_path: Path,
    base_config: dict[str, Any],
    all_columns: list[str],
    sensitive_attr: str,
    qi_subset: list[str],
    insensitive_subset: list[str],
    run_name: str,
    sample_frac: float,
    n_targets: int | str,
    seed: int,
) -> dict[str, Any]:
    phase_curve_dir = ensure_dir(output_root / "linkage_phase_curve")
    configs_dir = ensure_dir(phase_curve_dir / "configs")
    aux_dir = ensure_dir(phase_curve_dir / "auxiliary")

    # 1) Build and persist the runtime config for this configuration.
    config_payload = build_runtime_config_payload(
        base_config=base_config,
        full_dataset_path=full_dataset_path,
        qi_subset=qi_subset,
        insensitive_subset=insensitive_subset,
        all_columns=all_columns,
    )
    config_path = configs_dir / f"{run_name}.json"
    save_json(config_path, config_payload)

    # 2) Anonymization. Public CSV drops record_id; eval CSV keeps it.
    ano = run_one_experiment_from_config(
        config_path=config_path,
        output_root=output_root,
        name=run_name,
        save_anonymized_csv=True,
        save_anonymized_eval_csv=True,
        public_drop_columns=["record_id"],
        append_summary=True,
        drop_fully_suppressed_records_from_exports=True,
    )
    if ano["row"].get("status") != "success":
        raise RuntimeError(
            f"Anonymization failed for {run_name}: "
            f"{ano['row'].get('error', '(no error message)')}"
        )

    public_csv = Path(ano["public_csv_path"])
    eval_csv = Path(ano["eval_csv_path"])
    runtime_config_path = Path(ano["runtime_config_path"])

    # Attacker knowledge for this run = everything in the release that the
    # attacker can plausibly link on: the generalized QIs + the clear
    # insensitives. record_id and the sensitive attribute are excluded.
    attacker_knowledge = list(qi_subset) + list(insensitive_subset)

    # 3) Auxiliary base: sample targets only among records that actually survived.
    aux = build_auxiliary_base(
        full_dataset_path=full_dataset_path,
        known_attrs=attacker_knowledge,
        target_id_col="record_id",
        sample_frac=sample_frac,
        seed=seed,
        sensitive_attr=sensitive_attr,
        output_root=output_root,
        aux_output=aux_dir / f"{run_name}__aux.csv",
        released_eval=eval_csv,
    )

    # 4) Linkage attack. We skip the per-run HTML report because this is a sweep.
    attack = run_linkage_attack_from_paths(
        config_path=runtime_config_path,
        auxiliary_path=aux["aux_output_path"],
        anonymized_path=public_csv,
        anonymized_eval_path=eval_csv,
        known_attrs=attacker_knowledge,
        target_id_col="record_id",
        sensitive_attr=sensitive_attr,
        n_targets=n_targets,
        seed=seed,
        output_root=output_root,
        name=None,
        save_prefilter_debug=False,
        use_privjedai_fuzzy=False,
        generate_report=False,
    )

    summary = attack["summary"]
    attacker_knowledge_data = attack["attacker_knowledge"]

    # Count how many attacker-known attributes ended up generalized / suppressed.
    visible_levels = {
        attr: int(attacker_knowledge_data.get(attr, {}).get("visible_level", 0))
        for attr in attacker_knowledge
    }
    n_generalized_known_attrs = sum(1 for lvl in visible_levels.values() if lvl != 0)

    return {
        "status": "success",
        "run_name": run_name,
        "quasi_identifiers": "|".join(qi_subset),
        "n_qi": len(qi_subset),
        "insensitive_attributes_chosen": "|".join(insensitive_subset),
        "n_insensitive_chosen": len(insensitive_subset),
        "attacker_knowledge": "|".join(attacker_knowledge),
        "n_generalized_known_attrs": n_generalized_known_attrs,
        "visible_levels": "|".join(f"{k}:{v}" for k, v in visible_levels.items()),
        "avg_stage1_equivalence_class_size": summary.get("avg_stage1_equivalence_class_size"),
        "median_stage1_equivalence_class_size": summary.get("median_stage1_equivalence_class_size"),
        "avg_stage2_equivalence_class_size": summary.get("avg_equivalence_class_size"),
        "median_stage2_equivalence_class_size": summary.get("median_equivalence_class_size"),
        "n_targets": summary.get("n_targets"),
        "linkage_summary_path": str(attack["summary_path"]),
        "runtime_config_path": str(runtime_config_path),
        "anonymized_path": str(public_csv),
        "anonymized_eval_path": str(eval_csv),
        "auxiliary_path": str(aux["aux_output_path"]),
    }


# ---------------------------------------------------------------------------
# Aggregation + plot
# ---------------------------------------------------------------------------


def _format_anonymization_label(base_config: dict[str, Any]) -> str:
    """
    Best-effort extraction of k / l / suppression-limit from the base config for
    the plot legend. Looks at the top level first, then at any nested
    `privacy_models` or `anonymization` / `privacy` sub-dict. Missing keys are
    simply omitted; everything found is formatted on a single line.
    """
    def _lookup(keys: list[str]) -> Any:
        for k in keys:
            if k in base_config and base_config[k] is not None:
                return base_config[k]
        for nest_key in ("privacy_models", "privacy", "anonymization", "params"):
            nest = base_config.get(nest_key)
            if isinstance(nest, dict):
                for k in keys:
                    if k in nest and nest[k] is not None:
                        return nest[k]
        return None

    k_val = _lookup(["k", "k_anonymity"])
    l_val = _lookup(["l", "l_diversity"])
    s_val = _lookup(["suppression_limit", "suppression", "max_outliers"])

    parts: list[str] = []
    if k_val is not None:
        parts.append(f"k = {k_val}")
    if l_val is not None:
        parts.append(f"l = {l_val}")
    if s_val is not None:
        try:
            s_float = float(s_val)
        except (TypeError, ValueError):
            parts.append(f"suppression limit = {s_val}")
        else:
            if 0 < s_float <= 1:
                pct = s_float * 100
                parts.append(
                    f"suppression limit = {pct:.0f}%"
                    if abs(pct - round(pct)) < 1e-6
                    else f"suppression limit = {pct:.1f}%"
                )
            else:
                parts.append(f"suppression limit = {s_val}")

    if not parts:
        return "Anonymization parameters: (unknown)"
    return "Anonymization parameters: " + ", ".join(parts)


def aggregate_and_plot(
    *,
    run_rows: list[dict[str, Any]],
    base_config: dict[str, Any],
    output_dir: Path,
    stem: str,
    metric: str,
) -> None:
    """Aggregate phase-1/phase-2 per x-axis bucket and render the 2-curve plot."""
    import matplotlib

    matplotlib.use("Agg")  # headless safe
    import matplotlib.pyplot as plt

    if metric == "median":
        col_s1 = "median_stage1_equivalence_class_size"
        col_s2 = "median_stage2_equivalence_class_size"
    else:
        col_s1 = "avg_stage1_equivalence_class_size"
        col_s2 = "avg_stage2_equivalence_class_size"

    successes = [r for r in run_rows if r.get("status") == "success"]
    if not successes:
        print("[WARN] No successful run; skipping aggregation and plot.")
        return

    df_runs = pd.DataFrame(successes)
    if col_s1 not in df_runs.columns or col_s2 not in df_runs.columns:
        print(f"[WARN] Missing metric columns ({col_s1}, {col_s2}); skipping aggregation.")
        return

    # Across-runs aggregation: we report the MEDIAN as the curve point (the
    # central value the user asked for) plus mean/std/min/max for diagnostics.
    agg = (
        df_runs.groupby("n_generalized_known_attrs", as_index=False)
        .agg(
            n_runs=("run_name", "count"),
            stage1_median=(col_s1, "median"),
            stage1_mean=(col_s1, "mean"),
            stage1_std=(col_s1, "std"),
            stage1_min=(col_s1, "min"),
            stage1_max=(col_s1, "max"),
            stage2_median=(col_s2, "median"),
            stage2_mean=(col_s2, "mean"),
            stage2_std=(col_s2, "std"),
            stage2_min=(col_s2, "min"),
            stage2_max=(col_s2, "max"),
        )
        .sort_values("n_generalized_known_attrs", ascending=True)  # left to right: 0 -> 8
        .fillna(0.0)
    )

    agg_csv = output_dir / f"{stem}__aggregated.csv"
    agg.to_csv(agg_csv, index=False)
    print(f"[OK] Aggregated CSV : {agg_csv}")

    anon_label = _format_anonymization_label(base_config)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        agg["n_generalized_known_attrs"],
        agg["stage1_median"],
        marker="o",
        linewidth=2,
        label="Phase 1 (QI equivalence class)",
    )
    ax.plot(
        agg["n_generalized_known_attrs"],
        agg["stage2_median"],
        marker="s",
        linewidth=2,
        label="Phase 2 (after refinement on clear-text attributes)",
    )
    # x-axis ascending 0 -> 8 by default.
    ax.set_xlabel("Number of attacker-known attributes with visible_level \u2260 0")
    ax.set_ylabel(f"Equivalence class size ({metric})")
    ax.set_title("Adult \u2014 linkage attack: phase 1 vs phase 2")
    ax.grid(True, alpha=0.3)
    leg = ax.legend(
        loc="upper right",
        title=anon_label,
        title_fontsize=10,
        framealpha=0.95,
    )
    # Left-align the legend title with the entries for a cleaner look.
    leg._legend_box.align = "left"

    plt.tight_layout()
    plot_path = output_dir / f"{stem}__plot.png"
    plt.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Plot           : {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_csv_list(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


def _parse_n_targets(raw: str) -> int | str:
    value = str(raw).strip()
    if value.lower() == "all":
        return "all"
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--n-targets must be a positive integer or 'all'.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--n-targets must be a positive integer or 'all'.")
    return parsed


def _parse_optional_nonneg_int(raw: str) -> int | None:
    value = str(raw).strip()
    if value == "" or value.lower() in {"none", "all"}:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a non-negative integer or 'none'/'all'."
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0.")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-curve experiment on Adult: draw N random release configurations "
            "(random QI subset + random insensitive subset), run one anonymization "
            "and one linkage attack per configuration, and plot median phase-1 / "
            "phase-2 equivalence class sizes grouped by number of attacker-known "
            "attributes that got generalized."
        )
    )
    parser.add_argument("--base-config", default="configs/adult_base.json",
                        help="Base JSON config; its sensitive/identifiers/hierarchy_dir/k/l/suppression fields are preserved.")
    parser.add_argument("--full-dataset", default="data/adult_with_record_id.csv",
                        help="CSV that contains the record_id column.")
    parser.add_argument("--output-root", default="outputs",
                        help="Root directory where outputs will be written.")
    parser.add_argument("--project-root", default=None,
                        help="Project root. Defaults to the parent of this script's directory.")
    parser.add_argument("--qi-pool", default=",".join(DEFAULT_QI_POOL),
                        help="Comma-separated QI pool. Each QI must have a matching <qi>.csv under the hierarchy dir.")
    parser.add_argument("--sensitive-attr", default="income", help="Sensitive attribute name.")
    parser.add_argument("--n-configurations", type=int, default=50,
                        help="Number of unique random configurations (QI subset + insensitive subset) to run.")
    parser.add_argument("--min-qi", type=int, default=2,
                        help="Minimum number of QIs per configuration (must be >= 2).")
    parser.add_argument("--max-qi", type=int, default=8,
                        help=(
                            "Maximum number of QIs per configuration. Default 8 leaves at least "
                            "one QI pool member available as a potential insensitive, and stays "
                            "under the ARX solver's comfort zone on deep Adult hierarchies."
                        ))
    parser.add_argument("--min-insensitive", type=int, default=0,
                        help="Minimum number of insensitive attributes drawn from the remaining pool.")
    parser.add_argument("--max-insensitive", type=_parse_optional_nonneg_int, default=None,
                        help=(
                            "Maximum number of insensitive attributes drawn from the remaining "
                            "pool. Pass 'none' (or omit) for no cap."
                        ))
    parser.add_argument("--problematic-insensitive",
                        default=",".join(DEFAULT_PROBLEMATIC_INSENSITIVE),
                        help=(
                            "Comma-separated list of non-sensitive attributes that must never "
                            "be drawn as insensitive (too high-cardinality or redundant with a "
                            "QI). Pass an empty string to disable the filter."
                        ))
    parser.add_argument("--sample-frac", type=float, default=0.01,
                        help="Fraction of released records sampled as linkage targets for each configuration.")
    parser.add_argument("--n-targets", type=_parse_n_targets, default="all",
                        help="Number of linkage targets per configuration, or 'all'.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for configuration sampling and attacks.")
    parser.add_argument("--stem", default="adult_linkage_phase_curve",
                        help="File-name stem for plan/runs/aggregated/plot outputs.")
    parser.add_argument("--metric", choices=["mean", "median"], default="median",
                        help="Per-run reducer used for the y-values that are then aggregated across runs.")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop immediately if one configuration fails. Default: record and continue.")
    args = parser.parse_args()

    project_root = (
        Path(args.project_root).resolve() if args.project_root else SCRIPT_DIR.parent.resolve()
    )
    base_config_path = (project_root / args.base_config).resolve()
    full_dataset_path = (project_root / args.full_dataset).resolve()
    output_root = ensure_dir((project_root / args.output_root).resolve())

    qi_pool = _parse_csv_list(args.qi_pool)
    problematic_insensitive = _parse_csv_list(args.problematic_insensitive)
    if not qi_pool:
        raise SystemExit("--qi-pool cannot be empty.")

    base_config = load_json(base_config_path)
    sensitive_attributes_cfg = list(base_config.get("sensitive_attributes") or [])
    base_identifiers_cfg = list(base_config.get("identifiers") or [])
    if args.sensitive_attr and args.sensitive_attr not in sensitive_attributes_cfg:
        # Keep downstream sensitive-attr usage consistent with the CLI override.
        sensitive_attributes_cfg.append(args.sensitive_attr)

    # Read only the header to get the list of columns in the dataset.
    df_head = pd.read_csv(full_dataset_path, dtype=str, keep_default_na=False, nrows=0)
    all_columns = list(df_head.columns)
    missing_qis = [q for q in qi_pool if q not in all_columns]
    if missing_qis:
        raise SystemExit(f"QI pool contains columns not in the dataset: {missing_qis}")
    unknown_problematic = [p for p in problematic_insensitive if p not in all_columns]
    if unknown_problematic:
        print(
            f"[WARN] --problematic-insensitive lists columns not in the dataset "
            f"(ignored): {unknown_problematic}"
        )

    configs = sample_random_configurations(
        qi_pool=qi_pool,
        all_columns=all_columns,
        sensitive_attributes=sensitive_attributes_cfg,
        base_identifiers=base_identifiers_cfg,
        problematic_insensitive=problematic_insensitive,
        n_configurations=args.n_configurations,
        min_qi=args.min_qi,
        max_qi=args.max_qi,
        min_insensitive=args.min_insensitive,
        max_insensitive=args.max_insensitive,
        seed=args.seed,
    )

    phase_curve_dir = ensure_dir(output_root / "linkage_phase_curve")

    # Save the experiment plan up front.
    plan_rows = [
        {
            "run_index": i + 1,
            "run_name": f"{args.stem}__{i+1:03d}",
            "quasi_identifiers": "|".join(cfg["quasi_identifiers"]),
            "n_qi": len(cfg["quasi_identifiers"]),
            "insensitive_attributes_chosen": "|".join(cfg["insensitive_attributes"]),
            "n_insensitive_chosen": len(cfg["insensitive_attributes"]),
        }
        for i, cfg in enumerate(configs)
    ]
    plan_csv = phase_curve_dir / f"{args.stem}__plan.csv"
    pd.DataFrame(plan_rows).to_csv(plan_csv, index=False)
    print(f"[PLAN] {len(plan_rows)} configurations -> {plan_csv}")

    run_rows: list[dict[str, Any]] = []
    runs_csv = phase_curve_dir / f"{args.stem}__runs.csv"

    for i, cfg in enumerate(configs, start=1):
        run_name = f"{args.stem}__{i:03d}"
        qi_subset = list(cfg["quasi_identifiers"])
        insensitive_subset = list(cfg["insensitive_attributes"])
        print()
        print(f"[{i}/{len(configs)}] {run_name}")
        print(f"    QI ({len(qi_subset)}): {', '.join(qi_subset) or '(none)'}")
        print(
            f"    INS ({len(insensitive_subset)}): "
            f"{', '.join(insensitive_subset) or '(none)'}"
        )
        try:
            row = run_one_configuration(
                output_root=output_root,
                full_dataset_path=full_dataset_path,
                base_config=base_config,
                all_columns=all_columns,
                sensitive_attr=args.sensitive_attr,
                qi_subset=qi_subset,
                insensitive_subset=insensitive_subset,
                run_name=run_name,
                sample_frac=args.sample_frac,
                n_targets=args.n_targets,
                seed=args.seed,
            )
            stage1_key = (
                "median_stage1_equivalence_class_size"
                if args.metric == "median"
                else "avg_stage1_equivalence_class_size"
            )
            stage2_key = (
                "median_stage2_equivalence_class_size"
                if args.metric == "median"
                else "avg_stage2_equivalence_class_size"
            )
            print(
                f"    -> n_generalized_known_attrs = {row['n_generalized_known_attrs']}, "
                f"stage1 ({args.metric}) = {row.get(stage1_key)}, "
                f"stage2 ({args.metric}) = {row.get(stage2_key)}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    [FAILED] {exc}")
            if args.stop_on_error:
                traceback.print_exc()
                raise
            row = {
                "status": "failed",
                "run_name": run_name,
                "quasi_identifiers": "|".join(qi_subset),
                "n_qi": len(qi_subset),
                "insensitive_attributes_chosen": "|".join(insensitive_subset),
                "n_insensitive_chosen": len(insensitive_subset),
                "error": str(exc),
            }
        run_rows.append(row)

        # Persist after every run so a crash does not lose already-completed work.
        pd.DataFrame(run_rows).to_csv(runs_csv, index=False)

    print()
    print(f"[RUNS] {runs_csv}")

    aggregate_and_plot(
        run_rows=run_rows,
        base_config=base_config,
        output_dir=phase_curve_dir,
        stem=args.stem,
        metric=args.metric,
    )

    n_ok = sum(1 for r in run_rows if r.get("status") == "success")
    n_ko = sum(1 for r in run_rows if r.get("status") == "failed")
    print()
    print(f"[DONE] success = {n_ok}, failed = {n_ko}")


if __name__ == "__main__":
    main()
