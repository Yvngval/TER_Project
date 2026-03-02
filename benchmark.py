"""
benchmark.py — Orchestrateur principal du benchmark Privacy vs Utility.

Exécute la grille d'expériences :
  - Horizontal (variation des QI connus par l'attaquant)
  - Vertical (variation de k pour k-anonymity)

Pour chaque configuration : anonymise, attaque, mesure l'utilité, log.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime

from config import (
    QUASI_IDENTIFIERS, SENSITIVE_ATTRIBUTE, DROP_COLUMNS,
    OTHER_COLUMNS, K_VALUES, QI_SUBSETS, N_TARGETS, RANDOM_SEED,
    ML_TARGET_COLUMN,
)
from anonymizer import anonymize
from attacks import select_targets, linkage_attack, membership_inference_attack
from utility import compute_statistical_utility, compute_ml_utility


def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Charge et nettoie le dataset Adult."""
    print(f"[DATA] Chargement de {filepath}...")
    df = pd.read_csv(filepath)

    # Nettoyer les espaces
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()

    # Supprimer les colonnes non nécessaires
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Supprimer les lignes avec '?'
    for col in df.columns:
        if df[col].dtype == "object":
            df = df[df[col] != "?"]

    df = df.reset_index(drop=True)
    print(f"[DATA] Dataset prêt : {len(df)} lignes, {len(df.columns)} colonnes")
    print(f"[DATA] Colonnes : {list(df.columns)}")
    return df


def run_single_experiment(df: pd.DataFrame, qi_subset: list[str],
                          k: int, targets: pd.DataFrame,
                          targets_out: pd.DataFrame,
                          run_ml: bool = True) -> dict:
    """
    Exécute UNE expérience (un point de la grille).

    Args:
        df: Dataset original.
        qi_subset: Sous-ensemble de QI pour cette config.
        k: Valeur de k pour k-anonymity.
        targets: Cibles pour l'attaque (échantillon du dataset).
        targets_out: Individus hors dataset (pour MIA).
        run_ml: Si True, exécute aussi le benchmark ML (plus lent).

    Returns:
        Dict complet des résultats.
    """
    experiment_id = f"k{k}_qi{len(qi_subset)}"
    print(f"\n  ┌─ Expérience {experiment_id} : k={k}, QI={qi_subset}")

    result = {
        "experiment_id": experiment_id,
        "k": k,
        "qi_subset": qi_subset,
        "n_qi": len(qi_subset),
        "timestamp": datetime.now().isoformat(),
    }

    # ── 1. Anonymisation ──────────────────────────────────
    t0 = time.time()
    df_anon, anon_meta = anonymize(df, qi_subset, k)
    anon_time = time.time() - t0

    result["anonymization"] = anon_meta
    result["anonymization"]["time_seconds"] = round(anon_time, 3)

    if df_anon is None:
        print(f"  │  ✗ Anonymisation échouée (suppression trop élevée)")
        result["status"] = "anonymization_failed"
        return result

    print(f"  │  ✓ Anonymisation réussie : {len(df_anon)} lignes "
          f"(suppression {anon_meta['suppression_rate']*100:.1f}%), "
          f"coût={anon_meta['cost']}")

    # ── 2. Attaque par linkage ────────────────────────────
    t0 = time.time()
    linkage_results = linkage_attack(targets, df_anon, qi_subset,
                                      SENSITIVE_ATTRIBUTE)
    attack_time = time.time() - t0

    # Retirer les résultats détaillés (trop volumineux pour le log)
    detailed = linkage_results.pop("detailed_results", [])
    result["linkage_attack"] = linkage_results
    result["linkage_attack"]["time_seconds"] = round(attack_time, 3)

    print(f"  │  ✓ Linkage Attack : re-id={linkage_results['re_identification_rate']*100:.1f}%, "
          f"ambiguous={linkage_results['ambiguity_rate']*100:.1f}%, "
          f"no-match={linkage_results['no_match_rate']*100:.1f}%")

    # ── 3. Attaque MIA (simplifié) ────────────────────────
    t0 = time.time()
    mia_results = membership_inference_attack(
        targets, targets_out, df_anon, qi_subset
    )
    mia_time = time.time() - t0
    result["mia_attack"] = mia_results
    result["mia_attack"]["time_seconds"] = round(mia_time, 3)

    print(f"  │  ✓ MIA : accuracy={mia_results['accuracy']*100:.1f}%, "
          f"precision={mia_results['precision']*100:.1f}%")

    # ── 4. Utilité statistique ────────────────────────────
    stat_util = compute_statistical_utility(df, df_anon, qi_subset)
    result["statistical_utility"] = stat_util
    print(f"  │  ✓ Info Loss : mean_KL={stat_util['global']['mean_kl_divergence']:.4f}, "
          f"mean_JS={stat_util['global']['mean_js_divergence']:.4f}")

    # ── 5. Utilité ML ─────────────────────────────────────
    if run_ml:
        t0 = time.time()
        all_features = qi_subset + [c for c in OTHER_COLUMNS if c in df.columns]
        ml_util = compute_ml_utility(df, df_anon, all_features, ML_TARGET_COLUMN)
        ml_time = time.time() - t0
        result["ml_utility"] = ml_util
        result["ml_utility_time_seconds"] = round(ml_time, 3)

        for model_name, metrics in ml_util.items():
            if "accuracy_loss" in metrics:
                print(f"  │  ✓ ML ({model_name}) : "
                      f"acc_orig={metrics['accuracy_original']:.3f}, "
                      f"acc_anon={metrics['accuracy_anonymized']:.3f}, "
                      f"loss={metrics['accuracy_loss']:.3f}")
    else:
        result["ml_utility"] = {"skipped": True}

    result["status"] = "success"
    print(f"  └─ Terminé")
    return result


def run_full_benchmark(data_filepath: str, output_dir: str = "results",
                       run_ml: bool = True) -> list[dict]:
    """
    Exécute le benchmark complet sur toute la grille (horizontal × vertical).

    Args:
        data_filepath: Chemin vers le CSV du dataset Adult.
        output_dir: Dossier de sortie pour les résultats.
        run_ml: Si True, inclut le benchmark ML (plus lent).

    Returns:
        Liste de tous les résultats d'expériences.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Charger les données
    df = load_and_prepare_data(data_filepath)

    # Sélectionner les cibles
    targets = select_targets(df, N_TARGETS, seed=RANDOM_SEED)

    # Créer des "faux" individus pour MIA (perturbation aléatoire)
    rng = np.random.RandomState(RANDOM_SEED + 1)
    targets_out = targets.copy()
    for col in QUASI_IDENTIFIERS:
        if col in targets_out.columns:
            unique_vals = df[col].unique()
            targets_out[col] = rng.choice(unique_vals, size=len(targets_out))

    print("\n" + "=" * 70)
    print("  BENCHMARK PRIVACY vs UTILITY — Dataset Adult")
    print("=" * 70)
    print(f"  Valeurs de k  : {K_VALUES}")
    print(f"  Sous-ensembles QI : {len(QI_SUBSETS)} configs "
          f"({[len(s) for s in QI_SUBSETS]} QI)")
    print(f"  Nombre de cibles : {N_TARGETS}")
    print(f"  Total expériences : {len(K_VALUES) * len(QI_SUBSETS)}")
    print(f"  ML activé : {run_ml}")
    print("=" * 70)

    all_results = []
    t_start = time.time()

    for qi_idx, qi_subset in enumerate(QI_SUBSETS):
        print(f"\n{'─' * 60}")
        print(f"HORIZONTAL [{qi_idx+1}/{len(QI_SUBSETS)}] : "
              f"{len(qi_subset)} QI = {qi_subset}")
        print(f"{'─' * 60}")

        for k in K_VALUES:
            result = run_single_experiment(
                df, qi_subset, k, targets, targets_out, run_ml=run_ml
            )
            all_results.append(result)

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK TERMINÉ en {total_time:.1f}s")
    print(f"  {len(all_results)} expériences exécutées")
    print(f"{'=' * 70}")

    # ── Sauvegarder les résultats ─────────────────────────
    # JSON complet
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[SAVE] Résultats JSON : {json_path}")

    # CSV résumé
    csv_rows = []
    for r in all_results:
        row = {
            "k": r["k"],
            "n_qi": r["n_qi"],
            "qi_subset": str(r["qi_subset"]),
            "status": r["status"],
        }
        if r["status"] == "success":
            row.update({
                "suppression_rate": r["anonymization"]["suppression_rate"],
                "cost": r["anonymization"]["cost"],
                "n_eq_classes": r["anonymization"]["n_equivalence_classes"],
                "re_id_rate": r["linkage_attack"]["re_identification_rate"],
                "ambiguity_rate": r["linkage_attack"]["ambiguity_rate"],
                "no_match_rate": r["linkage_attack"]["no_match_rate"],
                "attr_inference_rate": r["linkage_attack"]["attribute_inference_rate"],
                "mean_candidate_set": r["linkage_attack"]["mean_candidate_set_size"],
                "mia_accuracy": r["mia_attack"]["accuracy"],
                "mia_precision": r["mia_attack"]["precision"],
                "mean_kl": r["statistical_utility"]["global"]["mean_kl_divergence"],
                "mean_js": r["statistical_utility"]["global"]["mean_js_divergence"],
                "gen_intensity": r["statistical_utility"]["global"]["generalization_intensity"],
            })
            # ML
            ml = r.get("ml_utility", {})
            for model in ["GradientBoosting", "RandomForest"]:
                if model in ml and "accuracy_loss" in ml[model]:
                    row[f"{model}_acc_orig"] = ml[model]["accuracy_original"]
                    row[f"{model}_acc_anon"] = ml[model]["accuracy_anonymized"]
                    row[f"{model}_acc_loss"] = ml[model]["accuracy_loss"]

        csv_rows.append(row)

    df_results = pd.DataFrame(csv_rows)
    csv_path = os.path.join(output_dir, "benchmark_summary.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"[SAVE] Résumé CSV   : {csv_path}")

    return all_results


# ══════════════════════════════════════════════════════════════════════
# Point d'entrée
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark Privacy vs Utility sur le dataset Adult"
    )
    parser.add_argument("--data", type=str, default="data/adult.csv",
                        help="Chemin vers le fichier adult.csv")
    parser.add_argument("--output", type=str, default="results",
                        help="Dossier de sortie")
    parser.add_argument("--no-ml", action="store_true",
                        help="Désactiver le benchmark ML (plus rapide)")
    parser.add_argument("--quick", action="store_true",
                        help="Mode rapide : k=[2,5,10], 2 subsets de QI")

    args = parser.parse_args()

    if args.quick:
        # Override pour un test rapide
        from config import K_VALUES as _k, QI_SUBSETS as _q
        import config
        config.K_VALUES = [2, 5, 10]
        config.QI_SUBSETS = [
            ["age", "sex"],
            ["age", "sex", "race", "marital-status", "education"],
        ]
        config.N_TARGETS = 50

    results = run_full_benchmark(
        data_filepath=args.data,
        output_dir=args.output,
        run_ml=not args.no_ml
    )
