# Benchmark Privacy vs Utility — TER Project

## Vue d'ensemble

Ce pipeline automatise le **benchmarking des attaques de ré-identification** et des **métriques d'utilité** sur des données tabulaires anonymisées. Il implémente exactement le protocole décrit par l'encadrant :

- **Assessment Horizontal** : varier le nombre de quasi-identifiants (QI) connus par l'attaquant (attaquant faible → fort)
- **Assessment Vertical** : varier la force d'anonymisation (k-anonymity : k = 2, 3, 5, 10, 15, 20)

Le tout est exécuté automatiquement sur une grille complète, avec logs CSV/JSON et génération de courbes.

---

## Structure du projet

```
ter-benchmark/
├── run_benchmark.py      ← Script principal (point d'entrée)
├── config.py             ← Configuration (QI, k-values, hiérarchies)
├── anonymizer.py         ← Moteur k-anonymity (généralisation + suppression)
├── attacks.py            ← Simulations d'attaques (linkage + MIA)
├── utility.py            ← Métriques d'utilité (statistiques + ML)
├── visualize.py          ← Génération des graphiques
├── benchmark.py          ← Orchestrateur de la grille d'expériences
├── requirements.txt      ← Dépendances Python
├── data/
│   └── adult.csv         ← Dataset Adult (à placer ici)
└── results/              ← Dossier de sortie (créé automatiquement)
    ├── benchmark_results.json
    ├── benchmark_summary.csv
    ├── plot_reid_vs_k.png
    ├── plot_ml_loss_vs_k.png
    ├── plot_privacy_vs_utility.png
    ├── plot_heatmap_reid.png
    ├── plot_mia_vs_k.png
    └── plot_cost_analysis.png
```

---

## Installation et lancement

### Prérequis

- Python 3.9 ou supérieur
- pip

### Étapes

```bash
# 1. Cloner ou copier le dossier du projet
cd ter-benchmark

# 2. (Optionnel) Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Placer le dataset
#    Copier adult.csv dans le dossier data/
mkdir -p data
cp /chemin/vers/adult.csv data/adult.csv

# 5. Lancer le benchmark
python run_benchmark.py --data data/adult.csv
```

### Options de lancement

| Commande | Description |
|----------|-------------|
| `python run_benchmark.py` | Benchmark complet (toutes configs, ML inclus) |
| `python run_benchmark.py --quick` | Mode rapide : seulement k={2,5,10} et 2 subsets QI |
| `python run_benchmark.py --no-ml` | Sans benchmark ML (beaucoup plus rapide) |
| `python run_benchmark.py --quick --no-ml` | Le plus rapide possible (test) |
| `python run_benchmark.py --plots-only` | Regénérer les graphiques à partir de résultats existants |
| `python run_benchmark.py --data chemin/vers/data.csv` | Spécifier un chemin personnalisé |

**Temps estimés :**
- `--quick --no-ml` : ~1-2 minutes
- `--quick` : ~5-10 minutes
- benchmark complet : ~30-60 minutes (dépend de la machine)

---

## Fonctionnement détaillé

### 1. Configuration (`config.py`)

Le fichier `config.py` définit tout ce qui est paramétrable :

- **Quasi-Identifiants (QI)** : `age, sex, race, marital-status, education, native-country, workclass, occupation`
- **Attribut sensible** : `income` (<=50K ou >50K)
- **Hiérarchies de généralisation** : chaque QI a une fonction qui définit comment le généraliser par niveau (ex : age → intervalle de 5 → de 10 → de 20 → `*`)
- **Grille d'expériences** : combinaisons de `K_VALUES` × `QI_SUBSETS`

Pour modifier la configuration, éditez directement `config.py`. Par exemple, pour ajouter une valeur de k :

```python
K_VALUES = [2, 3, 5, 10, 15, 20, 50]  # ajout de k=50
```

### 2. Anonymisation (`anonymizer.py`)

Implémente la **k-anonymity** via :

1. **Généralisation hiérarchique** : remplace les valeurs précises par des catégories plus larges (ex : age 35 → "35-39" → "30-39" → "20-39" → "*")
2. **Suppression** : élimine les lignes qui restent dans des classes d'équivalence de taille < k (max 15% de suppression par défaut)
3. **Recherche greedy** : trouve automatiquement le niveau minimal de généralisation nécessaire pour atteindre k-anonymity

### 3. Attaques (`attacks.py`)

Deux types d'attaques sont simulés :

**Linkage Attack** : L'attaquant connaît un sous-ensemble des QI d'une personne cible et essaie de la retrouver dans le dataset anonymisé.

- Sélection de N cibles aléatoires dans le dataset original
- Pour chaque cible, recherche de correspondances dans l'anonymisé via les QI connus
- Résultat : match unique (ré-identifié), match multiple (ambigu), ou pas de match (protégé)
- Métrique principale : **taux de ré-identification** (% de matchs uniques)

**Membership Inference Attack (MIA)** : L'attaquant essaie de déterminer si un individu faisait partie du dataset original.

- Utilise des cibles "in" (vraiment dans le dataset) et "out" (perturbées)
- Métriques : accuracy, precision, recall, F1

### 4. Métriques d'utilité (`utility.py`)

**A) Information Loss (statistique)** :
- **Divergence KL** entre les distributions originale et anonymisée de chaque QI
- **Divergence Jensen-Shannon** (version symétrique et bornée de KL)
- Intensité de généralisation (proportion de valeurs remplacées par `*`)

**B) ML Utility Loss** :
- Entraînement de GradientBoosting et RandomForest sur les données originales → accuracy de référence
- Entraînement des mêmes modèles sur les données anonymisées → accuracy dégradée
- Mesure de la perte : `accuracy_loss = acc_original - acc_anonymized`
- Le test set est toujours composé de données originales (non anonymisées) pour un benchmark équitable

### 5. Orchestration (`benchmark.py`)

Pour chaque point de la grille `(k, qi_subset)` :

1. Anonymise le dataset avec la config donnée
2. Lance l'attaque linkage sur les cibles pré-sélectionnées
3. Lance l'attaque MIA
4. Calcule les métriques statistiques d'information loss
5. (Optionnel) Entraîne et compare les modèles ML
6. Log tout dans un dict structuré

Résultats sauvegardés en **JSON** (détails complets) et **CSV** (résumé tabulaire).

### 6. Visualisation (`visualize.py`)

Génère 6 graphiques :

| Graphique | Description |
|-----------|-------------|
| `plot_reid_vs_k.png` | Taux de ré-identification en fonction de k, par nombre de QI |
| `plot_ml_loss_vs_k.png` | Perte d'accuracy ML en fonction de k |
| `plot_privacy_vs_utility.png` | Scatter trade-off Privacy vs Utility |
| `plot_heatmap_reid.png` | Heatmap ré-identification (k × nombre de QI) |
| `plot_mia_vs_k.png` | Accuracy de l'attaque MIA en fonction de k |
| `plot_cost_analysis.png` | Coût d'anonymisation et taux de suppression |

---

## Lien avec l'outil RECITALS

Le projet [RECITALS Anonymization Manager](https://github.com/AI-team-UoA/RECITALS-anonymization-manager) est un outil Python de gestion d'anonymisation. Ce benchmark est conçu pour être **compatible** avec RECITALS :

- Le fichier `anonymizer.py` peut être remplacé par les fonctions de RECITALS si vous souhaitez utiliser leur moteur d'anonymisation.
- Le format de données (CSV tabulaire avec QI et attributs sensibles) est identique.
- Pour intégrer RECITALS, installez-le via `uv sync --all-extras` dans leur repo, puis importez leurs fonctions d'anonymisation à la place des nôtres dans `anonymizer.py`.

---

## Interpréter les résultats

### Fichier CSV (`benchmark_summary.csv`)

Chaque ligne correspond à une expérience `(k, qi_subset)` avec :

| Colonne | Signification |
|---------|---------------|
| `k` | Paramètre de k-anonymity |
| `n_qi` | Nombre de QI connus par l'attaquant |
| `re_id_rate` | Taux de ré-identification (0 = parfait, 1 = tout ré-identifié) |
| `ambiguity_rate` | % de cibles avec plusieurs correspondances |
| `no_match_rate` | % de cibles sans correspondance |
| `attr_inference_rate` | % de cibles dont l'attribut sensible est correctement deviné |
| `mia_accuracy` | Accuracy de l'attaque MIA (0.5 = hasard) |
| `mean_kl` / `mean_js` | Information loss moyenne |
| `cost` | Somme des niveaux de généralisation appliqués |
| `suppression_rate` | Fraction de lignes supprimées |
| `GradientBoosting_acc_loss` | Perte d'accuracy du modèle ML |

### Ce qu'on s'attend à voir

- **Plus k augmente** → re_id_rate diminue (meilleure privacy), mais utility loss augmente
- **Plus l'attaquant connaît de QI** → re_id_rate augmente (privacy plus difficile à maintenir)
- Le **trade-off idéal** est un k qui minimise la ré-identification tout en conservant une utilité acceptable

---

## Personnalisation

### Ajouter un nouveau dataset

1. Placez le CSV dans `data/`
2. Modifiez `config.py` pour ajuster `QUASI_IDENTIFIERS`, `SENSITIVE_ATTRIBUTE`, `DROP_COLUMNS`
3. Ajoutez les fonctions de généralisation adaptées dans `GENERALIZATION_HIERARCHIES`

### Ajouter une technique d'anonymisation

1. Créez une nouvelle fonction dans `anonymizer.py` (ex : l-diversity, t-closeness, differential privacy)
2. Modifiez `benchmark.py` pour l'appeler en plus ou à la place de k-anonymity

### Ajouter un modèle ML

Éditez `utility.py`, fonction `compute_ml_utility()` — ajoutez simplement un nouveau modèle dans le dict `models`.

---

## Résumé des commandes

```bash
# Installation
pip install -r requirements.txt

# Test rapide (2 min)
python run_benchmark.py --data data/adult.csv --quick --no-ml

# Benchmark complet
python run_benchmark.py --data data/adult.csv

# Regénérer les plots
python run_benchmark.py --plots-only
```
