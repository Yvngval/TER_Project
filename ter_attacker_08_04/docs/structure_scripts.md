# Structure des scripts

## Objectif de cette page

Cette page permet de donner une vue claire de l'organisation des scripts du projet, pour comprendre rapidement :

- quels fichiers jouent un rôle central ;
- quels scripts servent à lancer les étapes principales ;
- quels fichiers servent surtout de support ;

---

## Vue d'ensemble

Le dossier `scripts/` contient :

1. les scripts d'exécution principaux ;
2. les scripts de préparation des données d'attaque ;
3. les fichiers utilitaires partagés ;
5. les fichiers annexes.

---

## Scripts principaux

Les scripts les plus importants dans l'état actuel du projet sont les suivants :

- `run_ano.py`
- `make_auxiliary_base.py`
- `run_linkage_attack.py`
- `make_mia_targets.py`
- `run_mia_attack.py`

Ce sont eux qui correspondent directement à la logique principale documentée dans les pages précédentes.

---

## 1. Scripts d'exécution principaux

## `run_ano.py`

C'est le point d'entrée principal pour l'anonymisation.

### Rôle
- charger une configuration d'expérience ;
- préparer la configuration runtime ;
- lancer l'anonymisation ;
- sauvegarder les sorties produites.

### Entrées typiques
- un fichier de configuration JSON ;
- un dataset source ;
- des hiérarchies de généralisation.

### Sorties typiques
- configuration exécutée ;
- dataset anonymisé public ;
- dataset anonymisé d'évaluation ;
- métriques.

C'est le script central de la phase d'anonymisation.

---

## `run_linkage_attack.py`

C'est le point d'entrée principal pour exécuter la linkage attack.

### Rôle
- charger la base auxiliaire ;
- charger les datasets anonymisés ;
- traiter les cibles ;
- filtrer les candidats compatibles ;
- construire les classes d'équivalence ;
- inférer l'attribut sensible ;
- sauvegarder les résultats.

### Idée générale
Ce script contient la logique principale de l'attaque de linkage.

---

## `run_mia_attack.py`

C'est le point d'entrée principal pour exécuter la membership inference attack.

### Rôle
- charger les cibles MIA ;
- charger les datasets anonymisés ;
- tester la compatibilité avec les attributs connus ;
- calculer des signaux de membership ;
- prédire IN ou OUT ;
- sauvegarder les résultats.

### Idée générale
Ce script contient la logique principale de la MIA.

---

## 2. Scripts de préparation des données d'attaque

Ces scripts ne réalisent pas directement les attaques, mais ils préparent les fichiers nécessaires à leur exécution.

## `make_auxiliary_base.py`

Ce script prépare la base auxiliaire utilisée par la linkage attack.

### Rôle
- partir du dataset original ;
- sélectionner un sous-ensemble d'individus ;
- conserver seulement les colonnes connues par l'attaquant ;
- produire un fichier auxiliaire exploitable pour l'attaque.

### Pourquoi il est important
Sans base auxiliaire, il n'y a pas de connaissance attaquant à exploiter dans la linkage attack.

---

## `make_mia_targets.py`

Ce script prépare les cibles utilisées par la MIA.

### Rôle
- construire les groupes IN et OUT ;
- échantillonner les cibles ;
- associer les attributs connus ;
- ajouter le label de vérité terrain `is_member` ;
- sauvegarder le fichier de cibles.

### Pourquoi il est important
Il définit précisément ce que la MIA devra prédire.

---

## 3. Fichiers utilitaires partagés

Ces fichiers ne sont pas lancés directement par l'utilisateur.  
Ils servent de couche de support pour factoriser la logique ou les fonctions techniques.

## `common.py`

Ce fichier regroupe les utilitaires communs au projet.

### Rôle
- gestion de chemins ;
- lecture et écriture de fichiers ;
- helpers de configuration ;
- fonctions réutilisées par plusieurs scripts.

C'est la boîte à outils générale du projet.

---

## `attack_common.py`

Ce fichier regroupe les utilitaires communs aux attaques.

### Rôle
- logique partagée entre linkage attack et MIA ;
- fonctions de chargement ou de validation ;
- compatibilité entre cibles et lignes anonymisées ;
- calculs communs liés aux attaques.

C'est la couche partagée par les scripts d'attaque.

---

## `linkage_helpers.py`

Ce fichier contient les helpers spécifiques à la linkage attack.

### Rôle
- fonctions de filtrage des candidats ;
- logique de compatibilité orientée linkage ;
- construction ou manipulation des classes d'équivalence ;
- traitement de l'inférence de l'attribut sensible.

Il isole la logique propre à la linkage attack pour éviter de surcharger le script principal.

---

## `privjedai_utils.py`

Ce fichier regroupe les fonctions liées à l'utilisation de `privJedAI`.

### Rôle
- transformations de données pour les étapes fuzzy ;
- fonctions d'intégration avec `privJedAI` ;
- logique de support pour les variantes de linkage plus souples.

Ce fichier sert d'extension utile pour certaines variantes de la linkage attack.

---

## Organisation logique


### Bloc anonymisation
- `run_ano.py`

### Bloc préparation linkage
- `make_auxiliary_base.py`

### Bloc attaque linkage
- `run_linkage_attack.py`
- `linkage_helpers.py`
- `privjedai_utils.py`

### Bloc préparation MIA
- `make_mia_targets.py`

### Bloc attaque MIA
- `run_mia_attack.py`

### Bloc utilitaires partagés
- `common.py`
- `attack_common.py`

### Bloc benchmarks
- `run_benchmark.py`
- `run_linkage_benchmark.py`
- `run_mia_benchmark.py`
