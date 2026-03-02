"""
utility.py — Mesure de l'utilité des données anonymisées.

Deux axes :
  A) Information Loss (comparaison statistique original ↔ anonymisé)
  B) ML Utility Loss (accuracy XGBoost/RandomForest sur original vs anonymisé)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

from config import ML_TARGET_COLUMN, ML_TEST_SIZE, RANDOM_SEED


# ══════════════════════════════════════════════════════════════════════
# A) Métriques d'Information Loss (statistiques)
# ══════════════════════════════════════════════════════════════════════

def compute_kl_divergence(p: np.ndarray, q: np.ndarray,
                          epsilon: float = 1e-10) -> float:
    """
    Calcule la divergence KL entre deux distributions discrètes.
    KL(P || Q) = sum( P(x) * log(P(x) / Q(x)) )
    """
    p = np.asarray(p, dtype=float) + epsilon
    q = np.asarray(q, dtype=float) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def compute_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Divergence de Jensen-Shannon (symétrique, bornée [0, ln2]).
    JS(P, Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = (P+Q)/2
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)
    m = 0.5 * (p + q)
    return 0.5 * compute_kl_divergence(p, m) + 0.5 * compute_kl_divergence(q, m)


def distribution_similarity(col_orig: pd.Series,
                             col_anon: pd.Series) -> dict:
    """
    Compare la distribution d'une colonne entre original et anonymisé.
    """
    # Aligner les catégories
    all_values = set(col_orig.astype(str).unique()) | set(col_anon.astype(str).unique())
    all_values = sorted(all_values)

    counts_orig = col_orig.astype(str).value_counts()
    counts_anon = col_anon.astype(str).value_counts()

    p = np.array([counts_orig.get(v, 0) for v in all_values], dtype=float)
    q = np.array([counts_anon.get(v, 0) for v in all_values], dtype=float)

    # Normaliser
    p_norm = p / (p.sum() + 1e-10)
    q_norm = q / (q.sum() + 1e-10)

    return {
        "kl_divergence": round(compute_kl_divergence(p_norm, q_norm), 6),
        "js_divergence": round(compute_js_divergence(p_norm, q_norm), 6),
        "n_categories_orig": len(counts_orig),
        "n_categories_anon": len(counts_anon),
    }


def compute_statistical_utility(df_original: pd.DataFrame,
                                 df_anonymized: pd.DataFrame,
                                 qi_columns: list[str]) -> dict:
    """
    Calcule les métriques d'information loss entre original et anonymisé.

    Returns:
        Dict avec pour chaque QI : KL, JS, et des métriques globales.
    """
    results = {"per_column": {}, "global": {}}

    kl_values = []
    js_values = []

    for qi in qi_columns:
        if qi in df_original.columns and qi in df_anonymized.columns:
            sim = distribution_similarity(df_original[qi], df_anonymized[qi])
            results["per_column"][qi] = sim
            kl_values.append(sim["kl_divergence"])
            js_values.append(sim["js_divergence"])

    # Métriques globales
    results["global"] = {
        "mean_kl_divergence": round(np.mean(kl_values), 6) if kl_values else 0,
        "mean_js_divergence": round(np.mean(js_values), 6) if js_values else 0,
        "max_kl_divergence": round(np.max(kl_values), 6) if kl_values else 0,
        "max_js_divergence": round(np.max(js_values), 6) if js_values else 0,
        "n_rows_original": len(df_original),
        "n_rows_anonymized": len(df_anonymized),
        "row_retention_rate": round(len(df_anonymized) / len(df_original), 4),
    }

    # Proportion de valeurs généralisées (remplacées par "*")
    n_stars = 0
    n_total = 0
    for qi in qi_columns:
        if qi in df_anonymized.columns:
            n_stars += (df_anonymized[qi].astype(str) == "*").sum()
            n_total += len(df_anonymized)
    results["global"]["generalization_intensity"] = round(
        n_stars / n_total, 4) if n_total > 0 else 0

    return results


# ══════════════════════════════════════════════════════════════════════
# B) ML Utility Loss
# ══════════════════════════════════════════════════════════════════════

def _prepare_ml_data(df: pd.DataFrame, feature_cols: list[str],
                     target_col: str) -> tuple:
    """
    Prépare les données pour l'entraînement ML.
    Encode les variables catégorielles via LabelEncoder.
    """
    df_ml = df.copy()

    # Garder seulement les colonnes nécessaires
    cols_to_use = [c for c in feature_cols if c in df_ml.columns]
    cols_to_use.append(target_col)
    df_ml = df_ml[cols_to_use].dropna()

    # Encoder les catégorielles
    encoders = {}
    for col in cols_to_use:
        if df_ml[col].dtype == "object" or df_ml[col].dtype.name == "category":
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            encoders[col] = le

    X = df_ml.drop(columns=[target_col])
    y = df_ml[target_col]

    return X, y, encoders


def compute_ml_utility(df_original: pd.DataFrame,
                       df_anonymized: pd.DataFrame,
                       feature_cols: list[str],
                       target_col: str = ML_TARGET_COLUMN) -> dict:
    """
    Compare la performance ML entre original et anonymisé.

    Protocole :
      1. Entraîner un modèle sur original → accuracy_original
      2. Entraîner le même modèle sur anonymisé → accuracy_anonymized
      3. Mesurer la perte = accuracy_original - accuracy_anonymized

    Utilise GradientBoosting (similaire XGBoost) et RandomForest.
    """
    results = {}

    # Préparer les données originales
    try:
        X_orig, y_orig, _ = _prepare_ml_data(df_original, feature_cols, target_col)
    except Exception as e:
        return {"error": f"Erreur préparation données originales: {str(e)}"}

    # Préparer les données anonymisées
    try:
        X_anon, y_anon, _ = _prepare_ml_data(df_anonymized, feature_cols, target_col)
    except Exception as e:
        return {"error": f"Erreur préparation données anonymisées: {str(e)}"}

    # Split des données originales (test set fixe)
    X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(
        X_orig, y_orig, test_size=ML_TEST_SIZE, random_state=RANDOM_SEED,
        stratify=y_orig
    )

    # Pour le modèle anonymisé : train sur anonymisé, test sur données originales
    # (on veut mesurer la qualité prédictive sur des vraies données)
    X_anon_train, _, y_anon_train, _ = train_test_split(
        X_anon, y_anon, test_size=ML_TEST_SIZE, random_state=RANDOM_SEED,
        stratify=y_anon
    )

    models = {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_SEED
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_SEED
        ),
    }

    for name, model_cls in models.items():
        try:
            # Entraîner sur original
            model_orig = model_cls.__class__(**model_cls.get_params())
            model_orig.fit(X_orig_train, y_orig_train)
            y_pred_orig = model_orig.predict(X_orig_test)
            acc_orig = accuracy_score(y_orig_test, y_pred_orig)
            f1_orig = f1_score(y_orig_test, y_pred_orig, average="weighted")

            # Entraîner sur anonymisé, tester sur même test set original
            model_anon = model_cls.__class__(**model_cls.get_params())

            # Aligner les colonnes (l'anonymisé peut avoir des features encodées différemment)
            common_cols = sorted(set(X_orig_train.columns) & set(X_anon_train.columns))
            model_anon.fit(X_anon_train[common_cols], y_anon_train)
            y_pred_anon = model_anon.predict(X_orig_test[common_cols])
            acc_anon = accuracy_score(y_orig_test, y_pred_anon)
            f1_anon = f1_score(y_orig_test, y_pred_anon, average="weighted")

            results[name] = {
                "accuracy_original": round(acc_orig, 4),
                "accuracy_anonymized": round(acc_anon, 4),
                "accuracy_loss": round(acc_orig - acc_anon, 4),
                "f1_original": round(f1_orig, 4),
                "f1_anonymized": round(f1_anon, 4),
                "f1_loss": round(f1_orig - f1_anon, 4),
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return results
