"""
config.py — Configuration du benchmark Privacy vs Utility.

Définit les quasi-identifiants, attributs sensibles, hiérarchies de
généralisation, et la grille d'expériences (horizontal + vertical).
"""

# ──────────────────────────────────────────────────────────────────────
# 1. Colonnes du dataset Adult
# ──────────────────────────────────────────────────────────────────────

# Quasi-Identifiants (QI) — ordonnés du plus risqué au moins risqué
QUASI_IDENTIFIERS = [
    "age",
    "sex",
    "race",
    "marital-status",
    "education",
    "native-country",
    "workclass",
    "occupation",
]

# Attribut(s) sensible(s)
SENSITIVE_ATTRIBUTE = "income"

# Colonnes à ignorer (identifiants directs ou non pertinents)
DROP_COLUMNS = ["fnlwgt", "education-num", "capital-gain", "capital-loss"]

# Colonnes "other" conservées telles quelles
OTHER_COLUMNS = ["relationship", "hours-per-week"]

# ──────────────────────────────────────────────────────────────────────
# 2. Hiérarchies de généralisation
# ──────────────────────────────────────────────────────────────────────
# Chaque QI a N niveaux : 0 = valeur originale, 1 = première généralisation, etc.
# Le dernier niveau est toujours "*" (suppression totale).

def generalize_age(value, level):
    """Généralise l'âge par intervalles croissants."""
    try:
        age = int(value)
    except (ValueError, TypeError):
        return "*"
    if level == 0:
        return str(age)
    elif level == 1:
        # Intervalle de 5 ans
        lower = (age // 5) * 5
        return f"{lower}-{lower+4}"
    elif level == 2:
        # Intervalle de 10 ans
        lower = (age // 10) * 10
        return f"{lower}-{lower+9}"
    elif level == 3:
        # Intervalle de 20 ans
        lower = (age // 20) * 20
        return f"{lower}-{lower+19}"
    else:
        return "*"


def generalize_education(value, level):
    """Généralise education en catégories plus larges."""
    mapping_l1 = {
        "Preschool": "Low", "1st-4th": "Low", "5th-6th": "Low",
        "7th-8th": "Low", "9th": "Low", "10th": "Medium",
        "11th": "Medium", "12th": "Medium", "HS-grad": "Medium",
        "Some-college": "High", "Assoc-voc": "High", "Assoc-acdm": "High",
        "Bachelors": "Higher", "Masters": "Higher",
        "Prof-school": "Higher", "Doctorate": "Higher",
    }
    if level == 0:
        return value
    elif level == 1:
        return mapping_l1.get(str(value).strip(), "Other")
    else:
        return "*"


def generalize_marital(value, level):
    """Généralise marital-status."""
    mapping_l1 = {
        "Married-civ-spouse": "Married", "Married-spouse-absent": "Married",
        "Married-AF-spouse": "Married", "Divorced": "Single",
        "Never-married": "Single", "Separated": "Single",
        "Widowed": "Single",
    }
    if level == 0:
        return value
    elif level == 1:
        return mapping_l1.get(str(value).strip(), "Other")
    else:
        return "*"


def generalize_native_country(value, level):
    """Généralise native-country par continent."""
    north_america = {"United-States", "Canada", "Mexico", "Cuba",
                     "Jamaica", "Dominican-Republic", "Haiti",
                     "Guatemala", "Honduras", "El-Salvador",
                     "Nicaragua", "Puerto-Rico", "Trinadad&Tobago",
                     "Outlying-US(Guam-USVI-etc)"}
    asia = {"China", "Japan", "India", "Iran", "Philippines",
            "Vietnam", "Taiwan", "Thailand", "Hong",
            "Cambodia", "Laos", "South"}
    europe = {"England", "Germany", "Italy", "Poland", "Portugal",
              "France", "Greece", "Ireland", "Hungary",
              "Holand-Netherlands", "Scotland", "Yugoslavia"}
    south_america = {"Columbia", "Ecuador", "Peru"}

    val = str(value).strip()
    if level == 0:
        return val
    elif level == 1:
        if val in north_america:
            return "North-America"
        elif val in asia:
            return "Asia"
        elif val in europe:
            return "Europe"
        elif val in south_america:
            return "South-America"
        else:
            return "Other"
    else:
        return "*"


def generalize_categorical_simple(value, level):
    """Généralisation simple : 0 = original, 1 = '*'."""
    if level == 0:
        return value
    return "*"


# Mapping QI → fonction de généralisation + nombre max de niveaux
GENERALIZATION_HIERARCHIES = {
    "age":              {"func": generalize_age,              "max_level": 4},
    "sex":              {"func": generalize_categorical_simple, "max_level": 1},
    "race":             {"func": generalize_categorical_simple, "max_level": 1},
    "marital-status":   {"func": generalize_marital,          "max_level": 2},
    "education":        {"func": generalize_education,        "max_level": 2},
    "native-country":   {"func": generalize_native_country,   "max_level": 2},
    "workclass":        {"func": generalize_categorical_simple, "max_level": 1},
    "occupation":       {"func": generalize_categorical_simple, "max_level": 1},
}

# ──────────────────────────────────────────────────────────────────────
# 3. Grille d'expériences
# ──────────────────────────────────────────────────────────────────────

# Vertical : valeurs de k à tester pour k-anonymity
K_VALUES = [2, 3, 5, 10, 15, 20]

# Horizontal : sous-ensembles croissants de QI (attaquant faible → fort)
QI_SUBSETS = [
    ["age", "sex"],                                          # 2 QI
    ["age", "sex", "race"],                                  # 3 QI
    ["age", "sex", "race", "marital-status"],                # 4 QI
    ["age", "sex", "race", "marital-status", "education"],   # 5 QI
    ["age", "sex", "race", "marital-status", "education",
     "native-country"],                                      # 6 QI
    ["age", "sex", "race", "marital-status", "education",
     "native-country", "workclass", "occupation"],           # 8 QI (all)
]

# Nombre de cibles (targets) pour l'attaque de ré-identification
N_TARGETS = 100

# Seed pour reproductibilité
RANDOM_SEED = 42

# Taux de suppression max autorisé (fraction de lignes supprimées)
MAX_SUPPRESSION_RATE = 0.15

# Modèle ML pour le benchmark d'utilité
ML_TARGET_COLUMN = "income"
ML_TEST_SIZE = 0.3
