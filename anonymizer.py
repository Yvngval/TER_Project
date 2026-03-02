"""
anonymizer.py — Moteur d'anonymisation par k-anonymity.

Implémente la généralisation hiérarchique des quasi-identifiants
et la suppression des lignes qui ne satisfont pas k-anonymity.
"""

import pandas as pd
import numpy as np
from itertools import product as iter_product

from config import GENERALIZATION_HIERARCHIES, MAX_SUPPRESSION_RATE


def apply_generalization(df: pd.DataFrame, qi_columns: list[str],
                         levels: dict[str, int]) -> pd.DataFrame:
    """
    Applique la généralisation sur les QI selon les niveaux donnés.

    Args:
        df: DataFrame original.
        qi_columns: Liste des quasi-identifiants à généraliser.
        levels: Dict {qi_name: level} indiquant le niveau de généralisation.

    Returns:
        DataFrame avec les QI généralisés.
    """
    df_gen = df.copy()
    for qi in qi_columns:
        if qi in levels and qi in GENERALIZATION_HIERARCHIES:
            func = GENERALIZATION_HIERARCHIES[qi]["func"]
            lvl = levels[qi]
            df_gen[qi] = df_gen[qi].apply(lambda v, f=func, l=lvl: f(v, l))
    return df_gen


def check_k_anonymity(df: pd.DataFrame, qi_columns: list[str], k: int) -> bool:
    """
    Vérifie si le DataFrame satisfait k-anonymity sur les QI donnés.

    Chaque classe d'équivalence (combinaison unique de QI) doit contenir
    au moins k lignes.
    """
    if df.empty:
        return False
    group_sizes = df.groupby(qi_columns, observed=True).size()
    return group_sizes.min() >= k


def get_equivalence_classes(df: pd.DataFrame,
                            qi_columns: list[str]) -> pd.Series:
    """
    Retourne la taille de chaque classe d'équivalence.
    """
    return df.groupby(qi_columns, observed=True).size()


def suppress_small_classes(df: pd.DataFrame, qi_columns: list[str],
                           k: int, max_suppression: float = MAX_SUPPRESSION_RATE
                           ) -> pd.DataFrame:
    """
    Supprime les lignes appartenant à des classes d'équivalence de taille < k.

    Si la suppression dépasse max_suppression, retourne None (échec).
    """
    group_sizes = df.groupby(qi_columns, observed=True).size()
    small_classes = group_sizes[group_sizes < k].index

    if len(small_classes) == 0:
        return df

    # Identifier les lignes à supprimer
    mask = df.set_index(qi_columns).index.isin(small_classes)
    n_suppress = mask.sum()
    suppression_rate = n_suppress / len(df)

    if suppression_rate > max_suppression:
        return None  # Trop de suppression

    return df[~mask].reset_index(drop=True)


def find_optimal_generalization(df: pd.DataFrame, qi_columns: list[str],
                                 k: int) -> dict | None:
    """
    Recherche le niveau minimal de généralisation (greedy bottom-up)
    qui satisfait k-anonymity avec un taux de suppression acceptable.

    Stratégie : on augmente progressivement le niveau de généralisation
    en commençant par les QI ayant le plus de valeurs distinctes.

    Returns:
        Dict {qi: level} ou None si impossible.
    """
    # Initialiser tous les niveaux à 0
    levels = {qi: 0 for qi in qi_columns}
    max_levels = {qi: GENERALIZATION_HIERARCHIES.get(qi, {}).get("max_level", 1)
                  for qi in qi_columns}

    # Essayer d'abord sans généralisation (juste suppression)
    df_gen = apply_generalization(df, qi_columns, levels)
    df_anon = suppress_small_classes(df_gen, qi_columns, k)
    if df_anon is not None and check_k_anonymity(df_anon, qi_columns, k):
        return levels

    # Stratégie greedy : augmenter le QI avec le plus de valeurs distinctes
    for _ in range(50):  # max iterations de sécurité
        # Calculer le nombre de valeurs distinctes par QI
        df_gen = apply_generalization(df, qi_columns, levels)
        n_distinct = {qi: df_gen[qi].nunique() for qi in qi_columns
                      if levels[qi] < max_levels[qi]}

        if not n_distinct:
            break  # Plus rien à généraliser

        # Généraliser le QI avec le plus de valeurs distinctes
        qi_to_gen = max(n_distinct, key=n_distinct.get)
        levels[qi_to_gen] += 1

        # Tester
        df_gen = apply_generalization(df, qi_columns, levels)
        df_anon = suppress_small_classes(df_gen, qi_columns, k)
        if df_anon is not None and check_k_anonymity(df_anon, qi_columns, k):
            return levels

    return None


def anonymize(df: pd.DataFrame, qi_columns: list[str], k: int
              ) -> tuple[pd.DataFrame | None, dict]:
    """
    Pipeline complet d'anonymisation :
      1. Trouver le niveau de généralisation optimal
      2. Appliquer la généralisation
      3. Supprimer les classes restantes < k

    Args:
        df: DataFrame original.
        qi_columns: Liste des QI.
        k: Paramètre de k-anonymity.

    Returns:
        (df_anonymized, metadata) où metadata contient :
          - levels : niveaux de généralisation appliqués
          - suppression_rate : taux de lignes supprimées
          - n_equivalence_classes : nombre de classes d'équivalence
          - min_class_size : taille min des classes
          - cost : score de coût (somme des niveaux de généralisation)
    """
    metadata = {
        "k": k,
        "qi_columns": qi_columns,
        "n_qi": len(qi_columns),
        "levels": {},
        "suppression_rate": 0.0,
        "n_equivalence_classes": 0,
        "min_class_size": 0,
        "cost": 0,
        "success": False,
    }

    levels = find_optimal_generalization(df, qi_columns, k)
    if levels is None:
        return None, metadata

    df_gen = apply_generalization(df, qi_columns, levels)
    df_anon = suppress_small_classes(df_gen, qi_columns, k)

    if df_anon is None or df_anon.empty:
        return None, metadata

    # Calculer les métadonnées
    eq_classes = get_equivalence_classes(df_anon, qi_columns)
    suppression_rate = 1.0 - len(df_anon) / len(df)

    metadata.update({
        "levels": levels,
        "suppression_rate": round(suppression_rate, 4),
        "n_equivalence_classes": len(eq_classes),
        "min_class_size": int(eq_classes.min()),
        "mean_class_size": round(eq_classes.mean(), 2),
        "max_class_size": int(eq_classes.max()),
        "cost": sum(levels.values()),
        "success": True,
    })

    return df_anon, metadata
