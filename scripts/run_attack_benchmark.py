r"""
run_attack_benchmark.py

Orchestrate a structured campaign of linkage and MIA attacks against
ARX-anonymized releases of the Adult dataset, organized as three studies plus
a non-anonymized baseline.

Studies
-------
- Study 1: sensitivity to attacker profile, anonymization fixed.
    One canonical release (R3 = {age, sex, race, native-country}, k=5, l=2,
    suppression=10%), five attacker profiles A1-A5.

- Study 2: sensitivity to QI declaration, attacker profile fixed (A5 = strong).
    Six representative releases R1-R6 with the same k/l/suppression so that
    only the choice of generalized attributes varies.

- Study 3: sensitivity to k, QI declaration and attacker profile fixed.
    Five values of k on R3, with A5 (strong attacker).

- Baseline: no anonymization (published.csv passed through), all five profiles.
    Provides the upper bound on what every attack would achieve in the absence
    of any privacy mechanism.

Pipeline
--------
1. Split the original dataset once: published.csv + out.csv. Same split is
   reused for every release so that linkage and MIA always reason about the
   same population.

2. Build the unique set of releases needed by the three studies + baseline,
   anonymize each one once. Anonymization results are cached on disk so that
   re-running the script does not re-anonymize unless --force-rebuild is set.

3. For each (release, attacker_profile) pair declared by the studies, build
   the linkage auxiliary and the MIA targets, then run both attacks. Each
   attack writes its own per-target details and HTML report under outputs/.

4. Aggregate every (study, release, profile) row into two master CSVs
   (linkage + mia) plus a manifest JSON describing the campaign.

Usage (Windows PowerShell, from project root)
---------------------------------------------
    python .\scripts\run_attack_benchmark.py `
        --base-config .\configs\adult_demo_with_record_id.json `
        --full-dataset .\data\adult_with_record_id.csv `
        --output-root .\outputs `
        --campaign-name attack_benchmark_v1 `
        --seed 42

Re-running the same command will skip already-anonymized releases. Pass
--force-rebuild to redo the anonymizations from scratch.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# Allow importing the project's own scripts/ modules when launched from anywhere.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import ensure_dir, load_json, save_json  # type: ignore
from make_auxiliary_base import build_auxiliary_base  # type: ignore
from make_mia_targets import prepare_mia_split  # type: ignore
from make_mia_targets_post_ano import build_post_ano_targets  # type: ignore
from run_ano import build_runtime_config, run_one_experiment_from_config  # type: ignore
from run_linkage_attack import run_linkage_attack_from_paths  # type: ignore
from run_mia_attack import run_mia_attack_from_paths  # type: ignore


# ---------------------------------------------------------------------------
# Experimental design: attacker profiles, releases, studies
# ---------------------------------------------------------------------------

# Pool of attributes that have a hierarchy CSV and can therefore be declared
# as quasi-identifiers. Anything not declared as QI in a release is published
# in clear as an insensitive attribute.
GENERALIZABLE_ATTRIBUTES = [
    "age", "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]

# Non-generalizable columns that are always published as insensitive.
# fnlwgt and education-num are numeric IDs/scales; capital-* and hours-per-week
# are continuous and have no hierarchy in the project.
ALWAYS_INSENSITIVE_NON_GENERALIZABLE = [
    "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week",
]

# Every column that is NOT the sensitive attribute and NOT the record id.
# Used to build the strong attacker profile A5 (worst-case knowledge).
ALL_NON_SENSITIVE_NON_ID = (
    GENERALIZABLE_ATTRIBUTES + ALWAYS_INSENSITIVE_NON_GENERALIZABLE
)


# Fixed attacker profiles (Study 1 + Baseline). A5 is computed dynamically per
# release because "strong attacker" means "all release columns except the
# sensitive one", which depends on what the release actually publishes.
ATTACKER_PROFILES: dict[str, list[str] | None] = {
    "A1_demographic":     ["age", "sex", "race"],
    "A2_demographic_geo": ["age", "sex", "race", "native-country"],
    "A3_professional":    ["age", "sex", "education", "occupation", "workclass", "hours-per-week"],
    "A4_social":          ["age", "sex", "race", "marital-status", "relationship"],
    # A5 is dynamic: resolved per release as "everything in the release that is
    # not the sensitive attribute and not the record id".
    "A5_strong":          None,
}


# Default ARX parameters for the canonical release (R3) and Study 2.
DEFAULT_K = 5
DEFAULT_L = 2
DEFAULT_T = None
DEFAULT_SUPPRESSION_LIMIT = 0.1
DEFAULT_BACKEND = "arx"


@dataclass(frozen=True)
class Release:
    """One ARX anonymization configuration declared by a study.

    The pair (qi, k, l, t, suppression_limit) is what is varied across
    releases. Two studies that share the same Release object will reuse a
    single anonymization run thanks to the deduplication step below.
    """
    name: str                       # human-readable id (e.g. "R3", "R3_k=10")
    qi: tuple[str, ...]             # quasi-identifiers in the ARX config
    k: int = DEFAULT_K
    l: int | None = DEFAULT_L
    t: float | None = DEFAULT_T
    suppression_limit: float = DEFAULT_SUPPRESSION_LIMIT
    is_baseline: bool = False       # True => skip ARX, use published.csv as-is

    # Stable hash key used to deduplicate releases across studies.
    @property
    def key(self) -> tuple:
        return (
            self.qi, self.k, self.l, self.t, self.suppression_limit,
            self.is_baseline,
        )


# Six representative QI declarations covering the meaningful subspace.
# R3 is the user's current declaration; it appears in Study 1, 2, and 3.
R1 = Release(name="R1_minimal",         qi=("age", "sex"))
R2 = Release(name="R2_sweeney",         qi=("age", "sex", "native-country"))
R3 = Release(name="R3_current",         qi=("age", "sex", "race", "native-country"))
R4 = Release(name="R4_demo_work",       qi=("age", "sex", "race", "occupation", "workclass"))
R5 = Release(name="R5_broad",           qi=("age", "sex", "race", "native-country",
                                            "marital-status", "education", "occupation"))
R6 = Release(
    name="R6_maximal",
    qi=tuple(c for c in GENERALIZABLE_ATTRIBUTES if c != "relationship"),
)

# Variations of R3 with k swept from 2 to 50 (k=5 is R3 itself).
R3_K2  = Release(name="R3_k=2",  qi=R3.qi, k=2)
R3_K10 = Release(name="R3_k=10", qi=R3.qi, k=10)
R3_K20 = Release(name="R3_k=20", qi=R3.qi, k=20)
R3_K50 = Release(name="R3_k=50", qi=R3.qi, k=50)

# Baseline: no anonymization. The "release" is just published.csv unchanged.
BASELINE = Release(
    name="baseline_no_anonymization",
    qi=(),  # ignored when is_baseline=True
    k=1, l=None, t=None, suppression_limit=0.0,
    is_baseline=True,
)


@dataclass(frozen=True)
class StudyRun:
    """One (release, profile) pair declared by a study."""
    study: str
    release: Release
    profile: str                    # key in ATTACKER_PROFILES


def _study_1_runs() -> list[StudyRun]:
    """Study 1: canonical release, all five profiles."""
    return [StudyRun(study="study_1", release=R3, profile=p)
            for p in ATTACKER_PROFILES]


def _study_2_runs() -> list[StudyRun]:
    """Study 2: vary the QI declaration, A5 (strong attacker) fixed."""
    return [StudyRun(study="study_2", release=r, profile="A5_strong")
            for r in (R1, R2, R3, R4, R5, R6)]


def _study_3_runs() -> list[StudyRun]:
    """Study 3: vary k on the R3 QI declaration, A5 fixed."""
    return [StudyRun(study="study_3", release=r, profile="A5_strong")
            for r in (R3_K2, R3, R3_K10, R3_K20, R3_K50)]


def _baseline_runs() -> list[StudyRun]:
    """Baseline: no anonymization, all five profiles."""
    return [StudyRun(study="baseline", release=BASELINE, profile=p)
            for p in ATTACKER_PROFILES]


def all_runs() -> list[StudyRun]:
    """Concatenate every study's runs in execution order."""
    return (
        _study_1_runs() + _study_2_runs() + _study_3_runs() + _baseline_runs()
    )


# ---------------------------------------------------------------------------
# Release config builder
# ---------------------------------------------------------------------------


# Build the per-release ARX runtime config payload. record_id is forced to be
# insensitive (never QI, never sensitive) so that survivor lookup keeps working.
def build_release_config_payload(
    *,
    base_config: dict[str, Any],
    published_dataset_path: Path,
    release: Release,
    target_id_col: str = "record_id",
) -> dict[str, Any]:
    sensitive_attrs = list(base_config.get("sensitive_attributes") or [])
    base_identifiers = list(base_config.get("identifiers") or [])

    qi = list(release.qi)
    # Every other generalizable attribute that was not declared QI is published
    # in clear as an insensitive attribute. Same for the always-insensitive
    # non-generalizable columns. record_id is always passthrough.
    non_qi_generalizable = [c for c in GENERALIZABLE_ATTRIBUTES if c not in qi]
    insensitive = sorted({
        *non_qi_generalizable,
        *ALWAYS_INSENSITIVE_NON_GENERALIZABLE,
        target_id_col,
    })

    payload = dict(base_config)
    payload.pop("hierarchies", None)
    payload["data"] = str(published_dataset_path.resolve())
    payload["identifiers"] = list(base_identifiers)
    payload["quasi_identifiers"] = qi
    payload["sensitive_attributes"] = sensitive_attrs
    payload["insensitive_attributes"] = insensitive
    payload["k"] = release.k
    payload["l"] = release.l
    payload["t"] = release.t
    payload["suppression_limit"] = release.suppression_limit
    payload["backend"] = base_config.get("backend", DEFAULT_BACKEND)
    return payload


# Attacker profile A5 is dynamic: the strong attacker knows everything that
# is published in the release except the sensitive attribute and the record
# id. For every other profile we also drop attributes that the release does
# not actually publish.
def resolve_known_attrs(
    *,
    profile: str,
    release_payload: dict[str, Any],
    target_id_col: str,
) -> list[str]:
    sensitive = set(release_payload.get("sensitive_attributes") or [])
    qi = list(release_payload.get("quasi_identifiers") or [])
    insens = list(release_payload.get("insensitive_attributes") or [])
    visible_in_release = [
        c for c in (qi + insens)
        if c not in sensitive and c != target_id_col
    ]

    if profile == "A5_strong":
        return visible_in_release

    declared = ATTACKER_PROFILES[profile]
    if declared is None:
        raise ValueError(f"Profile {profile!r} has no static attribute list.")
    visible_set = set(visible_in_release)
    return [c for c in declared if c in visible_set]


# ---------------------------------------------------------------------------
# Pipeline steps: split, anonymize, attacks
# ---------------------------------------------------------------------------


# Run the MIA pre-split once for the whole campaign so every release uses the
# same published / out partition.
def run_mia_split_once(
    *,
    full_dataset_path: Path,
    campaign_root: Path,
    attacker_frac: float,
    seed: int,
) -> dict[str, Any]:
    split_dir = ensure_dir(campaign_root / "mia_split")
    metadata_path = split_dir / "split_metadata.json"
    published_path = split_dir / "published.csv"
    out_path = split_dir / "out.csv"

    if metadata_path.exists() and published_path.exists() and out_path.exists():
        meta = load_json(metadata_path)
        if (meta.get("attacker_frac") == attacker_frac
                and meta.get("seed") == seed):
            print(f"[CACHE] MIA split reused: {metadata_path}")
            return meta

    result = prepare_mia_split(
        original_path=full_dataset_path,
        attacker_frac=attacker_frac,
        seed=seed,
        target_id_col="record_id",
        output_root=campaign_root,
        published_output=published_path,
        out_output=out_path,
        name="campaign",
    )
    meta = result["metadata"]
    save_json(metadata_path, meta)
    return meta


# Anonymize a single release, or, for the baseline, just stage published.csv
# under the same path naming convention so downstream attacks treat both
# uniformly. Cached on disk via release.key.
def prepare_release(
    *,
    release: Release,
    base_config: dict[str, Any],
    published_dataset_path: Path,
    campaign_root: Path,
    force_rebuild: bool,
) -> dict[str, Any]:
    releases_root = ensure_dir(campaign_root / "releases")
    release_dir = ensure_dir(releases_root / release.name)
    config_path = release_dir / "config.json"
    public_csv = release_dir / "anonymized.csv"
    eval_csv = release_dir / "anonymized_eval.csv"
    metrics_path = release_dir / "metrics.json"
    info_path = release_dir / "release_info.json"
    runtime_config_path = release_dir / "runtime_config.json"

    payload = build_release_config_payload(
        base_config=base_config,
        published_dataset_path=published_dataset_path,
        release=release,
    )

    cached = (
        not force_rebuild
        and public_csv.exists()
        and eval_csv.exists()
        and config_path.exists()
        and runtime_config_path.exists()
        and info_path.exists()
    )
    if cached:
        info = load_json(info_path)
        # Verify the cached payload still matches the requested release.
        # Both sides are normalized through JSON because release.key contains
        # a tuple of strings (qi) that becomes a list after round-tripping
        # through to_jsonable / load_json.
        cached_key = info.get("release_key")
        wanted_key = json.loads(json.dumps(list(release.key), default=str))
        if cached_key == wanted_key:
            print(f"[CACHE] Release reused: {release.name}")
            return info

    save_json(config_path, payload)
    runtime_payload = build_runtime_config(config_path, payload)
    save_json(runtime_config_path, runtime_payload)

    if release.is_baseline:
        # No ARX call. published.csv is both the public and the eval release.
        df_pub = pd.read_csv(published_dataset_path, dtype=str, keep_default_na=False)
        # Public release omits record_id (consistent with anonymized public CSVs).
        df_pub.drop(columns=["record_id"]).to_csv(public_csv, index=False)
        df_pub.to_csv(eval_csv, index=False)
        save_json(metrics_path, {"baseline": True})
        ano_status = "baseline_passthrough"
    else:
        ano = run_one_experiment_from_config(
            config_path=config_path,
            output_root=release_dir,
            name=release.name,
            save_anonymized_csv=True,
            save_anonymized_eval_csv=True,
            public_drop_columns=["record_id"],
            append_summary=False,
            drop_fully_suppressed_records_from_exports=True,
        )
        if ano["row"].get("status") != "success":
            err_msg = ano["row"].get("error", "(no message)")
            print(f"[FAILED ANO] {release.name}: {err_msg}")
            failed_info = {
                "release_name": release.name,
                "release_key": list(release.key),
                "is_baseline": release.is_baseline,
                "qi": list(release.qi),
                "k": release.k,
                "l": release.l,
                "t": release.t,
                "suppression_limit": release.suppression_limit,
                "config_path": str(config_path),
                "status": "anonymization_failed",
                "error": str(err_msg),
            }
            save_json(info_path, failed_info)
            return failed_info
        # run_ano writes anonymized CSV under release_dir/anonymized/<name>.csv;
        # promote to release_dir/anonymized.csv so paths are stable.
        Path(ano["public_csv_path"]).replace(public_csv)
        Path(ano["eval_csv_path"]).replace(eval_csv)
        if ano.get("metrics_path"):
            Path(ano["metrics_path"]).replace(metrics_path)
        ano_status = "anonymized"

    info = {
        "release_name": release.name,
        "release_key": list(release.key),
        "is_baseline": release.is_baseline,
        "qi": list(release.qi),
        "k": release.k,
        "l": release.l,
        "t": release.t,
        "suppression_limit": release.suppression_limit,
        "config_path": str(runtime_config_path),
        "public_csv_path": str(public_csv),
        "eval_csv_path": str(eval_csv),
        "metrics_path": str(metrics_path),
        "status": ano_status,
        "payload": payload,
    }
    save_json(info_path, info)
    return info


# Run the linkage attack for one (release, profile) pair. Builds the auxiliary
# base on the fly, restricted to records that survived in the eval release.
def _run_linkage_for_run(
    *,
    run: StudyRun,
    release_info: dict[str, Any],
    full_dataset_path: Path,
    sensitive_attr: str,
    output_root: Path,
    seed: int,
    linkage_aux_frac: float,
    linkage_n_targets: int | str,
) -> dict[str, Any]:
    payload = release_info["payload"]
    known_attrs = resolve_known_attrs(
        profile=run.profile, release_payload=payload, target_id_col="record_id",
    )
    if not known_attrs:
        return {
            "status": "skipped",
            "reason": "profile has no overlap with release columns",
            "known_attrs": [],
        }

    run_id = f"{run.study}__{run.release.name}__{run.profile}"
    aux_dir = ensure_dir(output_root / "linkage_aux")
    aux_output = aux_dir / f"{run_id}.aux.csv"

    aux = build_auxiliary_base(
        full_dataset_path=full_dataset_path,
        known_attrs=known_attrs,
        target_id_col="record_id",
        sample_frac=linkage_aux_frac,
        seed=seed,
        sensitive_attr=sensitive_attr,
        output_root=output_root,
        aux_output=aux_output,
        released_eval=release_info["eval_csv_path"],
    )

    attack = run_linkage_attack_from_paths(
        config_path=release_info["config_path"],
        auxiliary_path=aux["aux_output_path"],
        anonymized_path=release_info["public_csv_path"],
        anonymized_eval_path=release_info["eval_csv_path"],
        known_attrs=known_attrs,
        target_id_col="record_id",
        sensitive_attr=sensitive_attr,
        n_targets=linkage_n_targets,
        seed=seed,
        output_root=output_root,
        name=run_id,
        save_prefilter_debug=False,
        use_privjedai_fuzzy=False,
        generate_report=True,
        report_title=f"Linkage - {run.study} - {run.release.name} - {run.profile}",
    )

    s = attack["summary"]
    return {
        "status": "success",
        "run_id": run_id,
        "study": run.study,
        "release_name": run.release.name,
        "profile": run.profile,
        "known_attrs": known_attrs,
        "n_known_attrs": len(known_attrs),
        "n_targets": s.get("n_targets"),
        "unique_reidentification_rate": s.get("unique_reidentification_rate"),
        "avg_qid_equivalence_class_size": s.get("avg_qid_equivalence_class_size"),
        "avg_equivalence_class_size": s.get("avg_equivalence_class_size"),
        "avg_reduction_rate": s.get("avg_reduction_rate"),
        "certainty_sensitive_inference_rate": s.get("certainty_sensitive_inference_rate"),
        "avg_true_sensitive_probability": s.get("avg_true_sensitive_probability"),
        "avg_top_sensitive_probability": s.get("avg_top_sensitive_probability"),
        "avg_stage1_equivalence_class_size": s.get("avg_stage1_equivalence_class_size"),
        "n_distinct_stage1_groups": s.get("n_distinct_stage1_groups"),
        "estimated_total_operations": s.get("operation_counter", {}).get("estimated_total_operations"),
        "summary_path": str(attack["summary_path"]),
        "report_path": str(attack["report_path"]) if attack.get("report_path") else "",
        "auxiliary_path": str(aux["aux_output_path"]),
    }


# Run the MIA attack for one (release, profile) pair. Builds the post-ano
# attacker KB and balanced IN/OUT targets on the fly.
def _run_mia_for_run(
    *,
    run: StudyRun,
    release_info: dict[str, Any],
    published_path: Path,
    out_path: Path,
    expected_in_size: int,
    output_root: Path,
    seed: int,
    mia_targets_per_class: int,
) -> dict[str, Any]:
    payload = release_info["payload"]
    known_qids = resolve_known_attrs(
        profile=run.profile, release_payload=payload, target_id_col="record_id",
    )
    if not known_qids:
        return {
            "status": "skipped",
            "reason": "profile has no overlap with release columns",
            "known_qids": [],
        }

    run_id = f"{run.study}__{run.release.name}__{run.profile}"
    targets_dir = ensure_dir(output_root / "mia_targets")
    targets_out = targets_dir / f"{run_id}.targets.csv"
    attacker_base_out = ensure_dir(output_root / "prepared_data") / f"{run_id}.attacker_base.csv"

    targets = build_post_ano_targets(
        published_path=published_path,
        out_path=out_path,
        anonymized_eval_path=release_info["eval_csv_path"],
        known_qids=known_qids,
        expected_in_size=expected_in_size,
        target_id_col="record_id",
        member_col="is_member",
        targets_per_class=mia_targets_per_class,
        seed=seed,
        output_root=output_root,
        targets_output=targets_out,
        attacker_base_output=attacker_base_out,
        name=run_id,
    )

    attack = run_mia_attack_from_paths(
        config_path=release_info["config_path"],
        targets_path=targets["targets_output_path"],
        anonymized_path=release_info["public_csv_path"],
        anonymized_eval_path=release_info["eval_csv_path"],
        known_qids=known_qids,
        target_id_col="record_id",
        member_col="is_member",
        output_root=output_root,
        name=run_id,
        seed=seed,
        use_privjedai_fuzzy=False,
        generate_report=True,
        report_title=f"MIA - {run.study} - {run.release.name} - {run.profile}",
    )

    s = attack["summary"]
    accuracy = s.get("accuracy")
    advantage = (accuracy - 0.5) if isinstance(accuracy, (int, float)) else None
    return {
        "status": "success",
        "run_id": run_id,
        "study": run.study,
        "release_name": run.release.name,
        "profile": run.profile,
        "known_qids": known_qids,
        "n_known_qids": len(known_qids),
        "n_targets": s.get("n_targets"),
        "n_members": s.get("n_members"),
        "n_non_members": s.get("n_non_members"),
        "accuracy": accuracy,
        "mia_advantage": advantage,
        "precision": s.get("precision"),
        "recall": s.get("recall"),
        "f1": s.get("f1"),
        "tp": s.get("tp"),
        "tn": s.get("tn"),
        "fp": s.get("fp"),
        "fn": s.get("fn"),
        "member_recall": s.get("member_recall"),
        "non_member_true_negative_rate": s.get("non_member_true_negative_rate"),
        "estimated_total_operations": s.get("operation_counter", {}).get("estimated_total_operations"),
        "summary_path": str(attack["summary_path"]),
        "report_path": str(attack["report_path"]) if attack.get("report_path") else "",
        "targets_path": str(targets["targets_output_path"]),
        "attacker_base_path": str(targets["attacker_base_output_path"]),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    df = pd.DataFrame(rows)
    # Stringify list columns so they survive CSV round-trips.
    for col in df.columns:
        if df[col].apply(lambda v: isinstance(v, list)).any():
            df[col] = df[col].apply(
                lambda v: "|".join(map(str, v)) if isinstance(v, list) else v
            )
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_campaign(
    *,
    base_config_path: Path,
    full_dataset_path: Path,
    output_root: Path,
    campaign_name: str,
    seed: int,
    attacker_frac: float,
    linkage_aux_frac: float,
    linkage_n_targets: int | str,
    mia_targets_per_class: int,
    force_rebuild: bool,
    only_studies: list[str] | None,
) -> dict[str, Any]:
    base_config = load_json(base_config_path)
    sensitive_attr = (base_config.get("sensitive_attributes") or [None])[0]
    if not sensitive_attr:
        raise ValueError("Base config must declare at least one sensitive_attributes entry.")

    output_root = ensure_dir(Path(output_root).resolve())
    campaign_root = ensure_dir(output_root / "campaigns" / campaign_name)
    save_json(campaign_root / "settings.json", {
        "base_config_path": str(base_config_path),
        "full_dataset_path": str(full_dataset_path),
        "output_root": str(output_root),
        "campaign_name": campaign_name,
        "seed": seed,
        "attacker_frac": attacker_frac,
        "linkage_aux_frac": linkage_aux_frac,
        "linkage_n_targets": linkage_n_targets,
        "mia_targets_per_class": mia_targets_per_class,
        "force_rebuild": force_rebuild,
        "only_studies": only_studies,
    })

    # 1) MIA split (shared by every release).
    split_meta = run_mia_split_once(
        full_dataset_path=full_dataset_path,
        campaign_root=campaign_root,
        attacker_frac=attacker_frac,
        seed=seed,
    )
    published_path = Path(split_meta["published_output"])
    out_pool_path = Path(split_meta["out_output"])
    expected_in_size = int(split_meta["expected_in_size"])

    # 2) Build the run plan and the unique release set.
    runs = all_runs()
    if only_studies:
        runs = [r for r in runs if r.study in set(only_studies)]
    if not runs:
        raise SystemExit("Empty run plan. Check --only-studies.")

    unique_releases: dict[tuple, Release] = {}
    for r in runs:
        unique_releases.setdefault(r.release.key, r.release)
    release_infos: dict[tuple, dict[str, Any]] = {}

    print("=" * 100)
    print(f"Campaign            : {campaign_name}")
    print(f"Run plan size       : {len(runs)} (release, profile) pairs")
    print(f"Unique releases     : {len(unique_releases)}")
    print(f"Sensitive attribute : {sensitive_attr}")
    print(f"Output root         : {campaign_root}")
    print("=" * 100)

    # 3) Anonymize each unique release once.
    for key, rel in unique_releases.items():
        print(f"\n[release] {rel.name}")
        info = prepare_release(
            release=rel,
            base_config=base_config,
            published_dataset_path=published_path,
            campaign_root=campaign_root,
            force_rebuild=force_rebuild,
        )
        release_infos[key] = info

    # 4) For each (release, profile) pair: run linkage + MIA.
    linkage_rows: list[dict[str, Any]] = []
    mia_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    attacks_root = ensure_dir(campaign_root / "attacks")

    for idx, run in enumerate(runs, start=1):
        print("\n" + "=" * 100)
        print(f"[{idx}/{len(runs)}] {run.study} | release={run.release.name} | profile={run.profile}")
        print("=" * 100)
        info = release_infos[run.release.key]

        if info.get("status") not in ("anonymized", "baseline_passthrough"):
            skip_reason = info.get("status", "unknown")
            print(f"[SKIP] release {run.release.name} not usable ({skip_reason}); skipping linkage and MIA.")
            for rows, csv_name in (
                (linkage_rows, "linkage_results.csv"),
                (mia_rows, "mia_results.csv"),
            ):
                rows.append({
                    "study": run.study,
                    "release_name": run.release.name,
                    "profile": run.profile,
                    "status": "skipped_release_failed",
                    "reason": skip_reason,
                })
                _write_csv(rows, campaign_root / csv_name)
            failures.append({
                "stage": "release_setup",
                "study": run.study,
                "release_name": run.release.name,
                "profile": run.profile,
                "error": f"release status = {skip_reason}",
            })
            continue

        # Linkage
        try:
            row = _run_linkage_for_run(
                run=run,
                release_info=info,
                full_dataset_path=published_path,  # auxiliary built from the published pool
                sensitive_attr=sensitive_attr,
                output_root=attacks_root,
                seed=seed,
                linkage_aux_frac=linkage_aux_frac,
                linkage_n_targets=linkage_n_targets,
            )
            row.update({"study": run.study, "release_name": run.release.name, "profile": run.profile})
            linkage_rows.append(row)
            _write_csv(linkage_rows, campaign_root / "linkage_results.csv")
        except Exception as exc:
            err = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            print(f"[FAILED linkage] {run.study} {run.release.name} {run.profile}\n{err}")
            failures.append({
                "stage": "linkage", "study": run.study,
                "release_name": run.release.name, "profile": run.profile,
                "error": str(exc),
            })

        # MIA
        try:
            row = _run_mia_for_run(
                run=run,
                release_info=info,
                published_path=published_path,
                out_path=out_pool_path,
                expected_in_size=expected_in_size,
                output_root=attacks_root,
                seed=seed,
                mia_targets_per_class=mia_targets_per_class,
            )
            row.update({"study": run.study, "release_name": run.release.name, "profile": run.profile})
            mia_rows.append(row)
            _write_csv(mia_rows, campaign_root / "mia_results.csv")
        except Exception as exc:
            err = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            print(f"[FAILED mia] {run.study} {run.release.name} {run.profile}\n{err}")
            failures.append({
                "stage": "mia", "study": run.study,
                "release_name": run.release.name, "profile": run.profile,
                "error": str(exc),
            })

    # 5) Final manifest.
    manifest = {
        "campaign_name": campaign_name,
        "seed": seed,
        "n_runs_planned": len(runs),
        "n_unique_releases": len(unique_releases),
        "n_linkage_success": sum(1 for r in linkage_rows if r.get("status") == "success"),
        "n_mia_success": sum(1 for r in mia_rows if r.get("status") == "success"),
        "n_failures": len(failures),
        "linkage_results_csv": str(campaign_root / "linkage_results.csv"),
        "mia_results_csv": str(campaign_root / "mia_results.csv"),
        "split_metadata": split_meta,
        "studies": {
            "study_1": [r.release.name + "/" + r.profile for r in _study_1_runs()],
            "study_2": [r.release.name + "/" + r.profile for r in _study_2_runs()],
            "study_3": [r.release.name + "/" + r.profile for r in _study_3_runs()],
            "baseline": [r.release.name + "/" + r.profile for r in _baseline_runs()],
        },
        "failures": failures,
    }
    save_json(campaign_root / "campaign_manifest.json", manifest)
    if failures:
        save_json(campaign_root / "failures.json", failures)

    print("\n" + "=" * 100)
    print(f"Campaign finished. Linkage success: {manifest['n_linkage_success']} | "
          f"MIA success: {manifest['n_mia_success']} | failures: {manifest['n_failures']}")
    print(f"Manifest    : {campaign_root / 'campaign_manifest.json'}")
    print(f"Linkage CSV : {campaign_root / 'linkage_results.csv'}")
    print(f"MIA CSV     : {campaign_root / 'mia_results.csv'}")
    return manifest


def _parse_csv_list(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def _parse_n_targets(raw: str) -> int | str:
    if raw.strip().lower() == "all":
        return "all"
    return int(raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the structured linkage + MIA attack benchmark campaign.",
    )
    parser.add_argument(
        "--base-config", required=True,
        help="Path to the base ARX config JSON (e.g. configs/adult_demo_with_record_id.json).",
    )
    parser.add_argument(
        "--full-dataset", required=True,
        help="Path to the original CSV with a record_id column.",
    )
    parser.add_argument(
        "--output-root", default="outputs",
        help="Project-wide outputs root (campaigns are stored under outputs/campaigns/).",
    )
    parser.add_argument(
        "--campaign-name", default="attack_benchmark_v1",
        help="Subdirectory name under outputs/campaigns/.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attacker-frac", type=float, default=0.10,
        help="Total attacker KB fraction for the MIA split (half OUT, half IN).",
    )
    parser.add_argument(
        "--linkage-aux-frac", type=float, default=0.05,
        help="Sampling fraction for the linkage auxiliary base, within survivors.",
    )
    parser.add_argument(
        "--linkage-n-targets", type=_parse_n_targets, default=500,
        help="Number of linkage targets per attack, or 'all'.",
    )
    parser.add_argument(
        "--mia-targets-per-class", type=int, default=500,
        help="Balanced IN/OUT target count per MIA attack.",
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Re-anonymize releases even if cached on disk.",
    )
    parser.add_argument(
        "--only-studies", default="",
        help="Comma-separated subset of studies to run "
             "(study_1, study_2, study_3, baseline). Empty = all.",
    )
    args = parser.parse_args()

    only_studies = _parse_csv_list(args.only_studies) or None

    run_campaign(
        base_config_path=Path(args.base_config).resolve(),
        full_dataset_path=Path(args.full_dataset).resolve(),
        output_root=Path(args.output_root).resolve(),
        campaign_name=args.campaign_name,
        seed=args.seed,
        attacker_frac=args.attacker_frac,
        linkage_aux_frac=args.linkage_aux_frac,
        linkage_n_targets=args.linkage_n_targets,
        mia_targets_per_class=args.mia_targets_per_class,
        force_rebuild=args.force_rebuild,
        only_studies=only_studies,
    )


if __name__ == "__main__":
    main()
