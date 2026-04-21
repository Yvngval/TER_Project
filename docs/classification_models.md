# classification_models — Guide d'intégration

> **Fichier source :** `scripts/classification_models.py`  
> **Rôle :** Construit des pipelines scikit-learn qui reproduisent le comportement des classifieurs ARX, afin de mesurer l'utilité d'un jeu de données anonymisé par sa précision de classification.

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Dépendances](#2-dépendances)
3. [Format des données attendu](#3-format-des-données-attendu)
4. [Configuration — `classification_config.json`](#4-configuration--classification_configjson)
5. [API publique](#5-api-publique)
6. [Pipelines de prétraitement par modèle](#6-pipelines-de-prétraitement-par-modèle)
7. [Exemple d'intégration minimal](#7-exemple-dintégration-minimal)
8. [Valeurs supprimées et généralisées](#8-valeurs-supprimées-et-généralisées)
9. [Correspondance ARX ↔ scikit-learn](#9-correspondance-arx--scikit-learn)
10. [Points d'attention pour l'intégration](#10-points-dattention-pour-lintégration)

---

## 1. Vue d'ensemble

Le module expose une **fabrique de modèles** (`get_model_builders`) qui retourne un dictionnaire de callables. Chaque callable, appelé sans argument, produit un **pipeline sklearn frais** prêt à être entraîné sur un fold de validation croisée.

Les quatre classifieurs disponibles :

| Clé dans le registre    | Classe sklearn              | Équivalent ARX                          |
|-------------------------|-----------------------------|-----------------------------------------|
| `zero_r`                | `DummyClassifier("prior")`  | ZeroR (baseline)                        |
| `logistic_regression`   | `LogisticRegression`        | Mahout `OnlineLogisticRegression`       |
| `naive_bayes`           | `MultinomialNB`/`BernoulliNB` | SMILE `NaiveBayes`                    |
| `random_forest`         | `RandomForestClassifier`    | SMILE `RandomForest`                    |

---

## 2. Dépendances

```
scikit-learn >= 1.3
numpy >= 1.24
```

Aucune autre dépendance externe. Le module n'importe pas pandas — les DataFrames passés en entrée sont acceptés mais convertis en tableaux numpy en interne.

---

## 3. Format des données attendu

### Colonnes

Les **14 colonnes de features** du dataset Adult (dans n'importe quel ordre) :

```
age, workclass, fnlwgt, education, education-num,
marital-status, occupation, relationship, race, sex,
capital-gain, capital-loss, hours-per-week, native-country
```

La colonne cible (`income` par défaut, ou celle spécifiée dans `classification_config.json`) doit être fournie séparément comme vecteur `y`.

### Types de valeurs acceptées

Les colonnes peuvent contenir des valeurs :

| Type de valeur        | Exemple         | Traitement appliqué        |
|-----------------------|-----------------|----------------------------|
| Numérique exact       | `"37"`, `37`    | Converti en `float`        |
| Intervalle généralisé | `"30-39"`       | Remplacé par le midpoint → `34.5` |
| Supprimé              | `"*"`           | Traité comme valeur inconnue, puis imputé |
| Inconnu               | `"?"`, `""`     | Traité comme valeur inconnue, puis imputé |

> Les colonnes numériques concernées par la conversion midpoint sont :  
> `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`

---

## 4. Configuration — `classification_config.json`

Chemin attendu : `configs/classification_config.json`

### Structure complète

```json
{
  "n_folds": 10,
  "random_state": 42,
  "target": "income",
  "active_models": "all",
  "feature_cols": ["age", "workclass", "fnlwgt", "..."],
  "classifiers": {
    "logistic_regression": {
      "enabled": true,
      "params": {
        "C": 100000,
        "penalty": "elasticnet",
        "l1_ratio": 0.5,
        "solver": "saga",
        "max_iter": 2000
      }
    },
    "naive_bayes": {
      "enabled": true,
      "model": "MULTINOMIAL",
      "preprocessing": {
        "n_bins": 10,
        "bin_strategy": "uniform"
      },
      "params": {
        "alpha": 1.0
      }
    },
    "random_forest": {
      "enabled": true,
      "params": {
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 5,
        "max_leaf_nodes": 100,
        "bootstrap": true,
        "max_samples": 1.0,
        "criterion": "gini"
      }
    },
    "zero_r": {
      "enabled": true
    }
  }
}
```

### Champ `active_models`

| Valeur               | Effet                                    |
|----------------------|------------------------------------------|
| `"all"`              | Active les 4 classifieurs (défaut)       |
| `"naive_bayes"`      | Active uniquement Naive Bayes            |
| `["zero_r", "random_forest"]` | Active les modèles listés      |

### Priorité des hyperparamètres

Les paramètres du config **surchargent** les défauts internes. Si une clé est absente du config, la valeur par défaut est utilisée :

```
params_effectifs = {**DEFAULT_*_PARAMS, **config["classifiers"][nom]["params"]}
```

---

## 5. API publique

### `split_num_cat(feature_cols)`

Divise une liste de colonnes en colonnes numériques et catégorielles.

```python
from classification_models import split_num_cat

num_cols, cat_cols = split_num_cat(feature_cols)
# num_cols → ["age", "fnlwgt", "education-num", ...]
# cat_cols → ["workclass", "education", "marital-status", ...]
```

---

### `get_model_builders(num_cols, cat_cols, config, random_state=42)`

Point d'entrée principal. Retourne un dictionnaire `{nom → factory}`.

```python
from classification_models import get_model_builders, split_num_cat

num_cols, cat_cols = split_num_cat(config["feature_cols"])
builders = get_model_builders(num_cols, cat_cols, config)

# builders est de la forme :
# {
#   "zero_r":              <function>,
#   "logistic_regression": <function>,
#   "naive_bayes":         <function>,
#   "random_forest":       <function>,
# }

# Créer un pipeline frais pour un fold :
pipeline = builders["random_forest"]()
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

> **Important :** appeler `builders["random_forest"]()` à chaque fold pour obtenir une instance fraîche — ne jamais réutiliser un pipeline déjà entraîné.

---

### `build_zero_r()` / `build_logistic_regression(...)` / `build_naive_bayes(...)` / `build_random_forest(...)`

Ces fonctions sont utilisables directement si vous avez besoin de construire un modèle spécifique sans passer par le registre.

```python
from classification_models import build_naive_bayes, split_num_cat

num_cols, cat_cols = split_num_cat(feature_cols)
pipeline = build_naive_bayes(
    num_cols=num_cols,
    cat_cols=cat_cols,
    model="MULTINOMIAL",
    params={"alpha": 1.0},
    n_bins=10,
    bin_strategy="uniform",
)
pipeline.fit(X_train, y_train)
```

---

## 6. Pipelines de prétraitement par modèle

### ZeroR

Aucun prétraitement. Le `DummyClassifier` prédit directement les fréquences de classe du fold d'entraînement.

---

### Logistic Regression

```
Colonnes numériques :
  valeur brute → midpoint (ex. "30-39" → 34.5) → imputation (moyenne) → StandardScaler

Colonnes catégorielles :
  valeur brute → OrdinalEncoder ("*" → -1) → imputation (mode) → OneHotEncoder
```

**Pourquoi StandardScaler ?** La descente de gradient (`solver="saga"`) converge beaucoup plus rapidement avec des features normalisées.

---

### Naive Bayes

```
Colonnes numériques :
  valeur brute → midpoint → imputation (moyenne) → KBinsDiscretizer (10 bins, one-hot-dense)

Colonnes catégorielles :
  valeur brute → OrdinalEncoder ("*" → -1) → imputation (mode) → OneHotEncoder
```

**Pourquoi discrétiser les numériques ?** `MultinomialNB` requiert des features entières non-négatives. La discrétisation en bins one-hot satisfait cette contrainte tout en restant fidèle au comportement ARX/SMILE.

---

### Random Forest

```
Colonnes numériques :
  valeur brute → midpoint → imputation (moyenne)   ← pas de scaling

Colonnes catégorielles :
  valeur brute → OrdinalEncoder ("*" → -1) → imputation (mode)   ← pas de one-hot
```

**Pourquoi pas de scaling ni de one-hot ?** Les arbres de décision splitent sur des seuils et sont invariants à l'échelle. L'encodage ordinal suffit ; le one-hot augmenterait inutilement la dimensionnalité.

---

## 7. Exemple d'intégration minimal

```python
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from classification_models import get_model_builders, split_num_cat

# --- 1. Charger config et données ---
with open("configs/classification_config.json") as f:
    config = json.load(f)

df = pd.read_csv("outputs/anonymized.csv")
target = config["target"]                       # ex. "income"
feature_cols = config["feature_cols"]

X = df[feature_cols]
y = df[target]

# --- 2. Construire le registre de modèles ---
num_cols, cat_cols = split_num_cat(feature_cols)
builders = get_model_builders(num_cols, cat_cols, config, random_state=42)

# --- 3. Validation croisée ---
cv = StratifiedKFold(n_splits=config["n_folds"], shuffle=True,
                     random_state=config["random_state"])

for model_name, factory in builders.items():
    all_preds, all_true = [], []

    for train_idx, test_idx in cv.split(X, y):
        pipe = factory()                        # pipeline frais à chaque fold
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        all_preds.extend(pipe.predict(X.iloc[test_idx]))
        all_true.extend(y.iloc[test_idx])

    acc = accuracy_score(all_true, all_preds)
    print(f"{model_name}: accuracy = {acc:.4f}")
```

---

## 8. Valeurs supprimées et généralisées

ARX produit des CSV dont certaines cellules contiennent des valeurs généralisées ou supprimées. Le module les gère automatiquement :

| Valeur dans le CSV | Colonne numérique        | Colonne catégorielle          |
|--------------------|--------------------------|-------------------------------|
| `"30-39"`          | Midpoint → `34.5`        | Traité comme catégorie `"30-39"` |
| `"*"`              | → `NaN` → imputation (moyenne) | → code `-1` → imputation (mode) |
| `"?"`              | → `NaN` → imputation (moyenne) | → code `-1` → imputation (mode) |
| `""`               | → `NaN` → imputation (moyenne) | → code `-1` → imputation (mode) |

> L'imputation est **apprise sur le fold d'entraînement uniquement** et appliquée au fold de test — aucune fuite de données.

---

## 9. Correspondance ARX ↔ scikit-learn

| Paramètre ARX                       | Valeur scikit-learn                        |
|-------------------------------------|--------------------------------------------|
| LR : pas de régularisation forte    | `C=100_000` (régularisation quasi-nulle)   |
| LR : ElasticNet                     | `penalty="elasticnet"`, `l1_ratio=0.5`     |
| NB : Laplace smoothing              | `alpha=1.0`                                |
| NB : discrétisation numérique       | `KBinsDiscretizer(n_bins=10, strategy="uniform")` |
| RF : `sqrt` features                | `max_features="sqrt"`                      |
| RF : bootstrap complet              | `bootstrap=True`, `max_samples=1.0`        |
| RF : 500 arbres                     | `n_estimators=500`                         |
| RF : Gini                           | `criterion="gini"`                         |
| CV : accumulation globale           | OOF concatenation (non-moyenne par fold)   |

---

## 10. Points d'attention pour l'intégration

1. **Ne pas réutiliser un pipeline entre folds.** Appeler `factory()` à chaque fold pour garantir un état propre.

2. **Les colonnes doivent être présentes dans le DataFrame.** Si une colonne de `feature_cols` est absente du CSV anonymisé, `ColumnTransformer` lèvera une `ValueError`.

3. **Le champ `target` du config est la colonne à prédire**, pas nécessairement `"income"`. Vérifier que cette colonne existe dans le CSV et qu'elle n'est pas dans `feature_cols`.

4. **`active_models` dans le config** contrôle quels modèles sont inclus dans le registre retourné par `get_model_builders`. Si vous ne voulez faire tourner qu'un modèle, mettez `"active_models": "naive_bayes"` plutôt que de filtrer en dehors.

5. **Parallélisme :** `RandomForestClassifier` est construit avec `n_jobs=-1` (tous les cœurs disponibles). Si vous parallélisez vous-même la boucle de CV avec `joblib`, préférez `n_jobs=1` dans le config pour éviter une sursouscription CPU.

6. **Reproductibilité :** le `random_state` du config est transmis à `RandomForestClassifier` et au `StratifiedKFold`. Pour reproduire exactement les résultats, utiliser les mêmes valeurs que dans le config de référence (`random_state: 42`).
