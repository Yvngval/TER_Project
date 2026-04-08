# Formats des fichiers d'entrée / sortie

## Objectif de cette page

Cette page décrit les principaux fichiers manipulés par le projet.

L'objectif est de comprendre :

- quels fichiers servent d'entrée ;
- quels fichiers sont produits ;
- à quoi ils servent dans le pipeline ;
- quelle est la différence entre les versions publiques et les versions d'évaluation.

---

## Vue d'ensemble

Le projet manipule plusieurs familles de fichiers :

- les **datasets source** ;
- les **fichiers de configuration** ;
- les **hiérarchies de généralisation** ;
- les **fichiers intermédiaires** pour les attaques ;
- les **fichiers de sortie** produits après anonymisation ou après attaque.

On peut résumer la logique ainsi :

1. on part d'un dataset source ;
2. on utilise une configuration d'anonymisation ;
3. on produit des datasets anonymisés et des métriques ;
4. on prépare des fichiers spécifiques aux attaques ;
5. on produit les résultats des attaques.

---

## 1. Fichiers d'entrée principaux

## Dataset source

Le dataset source est le point de départ de tout le projet.

Exemples typiques :

- `data/adult.csv`
- `data/adult_with_record_id.csv`

### Rôle
Ces fichiers contiennent les données brutes avant anonymisation.

### Format attendu
Ce sont des fichiers tabulaires de type CSV, avec une ligne d'en-tête contenant les noms de colonnes.

### Colonnes typiques
Selon la version utilisée, on peut y trouver :

- les quasi-identifiants ;
- l'attribut sensible ;
- d'autres attributs descriptifs ;
- éventuellement un identifiant interne comme `record_id`.


---

## Fichiers de configuration

Les expériences d'anonymisation sont pilotées par des fichiers JSON.

Exemple de dossier :

- `outputs/configs/`

### Rôle
Ces fichiers décrivent une expérience d'anonymisation.

### Contenu typique
On y retrouve notamment :

- le chemin du dataset ;
- les quasi-identifiants ;
- l'attribut sensible ;
- les attributs insensibles ;
- les chemins vers les hiérarchies ;
- les paramètres comme `k`, `l`, `t` ;
- la limite de suppression ;
- parfois d'autres paramètres utiles à l'exécution.

### Remarque
Dans le projet, il faut distinguer :

- la **configuration de départ** ;
- la **configuration runtime** réellement exécutée.

La configuration runtime est souvent plus importante pour l'analyse, car elle reflète exactement ce qui a été lancé.

---

## Hiérarchies de généralisation

Les hiérarchies sont stockées dans des fichiers CSV séparés.

Exemples typiques :

- `hierarchies/age.csv`
- `hierarchies/sex.csv`
- `hierarchies/race.csv`
- `hierarchies/native-country.csv`

### Rôle
Ces fichiers décrivent comment généraliser une valeur précise en valeurs plus larges.

### Format attendu
Ce sont des CSV dans lesquels chaque ligne représente une valeur source et ses niveaux successifs de généralisation.

### Exemple conceptuel
Une valeur précise comme un âge ou un pays peut être reliée à :

- une catégorie intermédiaire ;
- puis une catégorie plus générale.

### Utilisation
Ces fichiers sont utilisés pendant l'anonymisation.

---

## 2. Fichiers produits par l'anonymisation

L'anonymisation produit plusieurs types de fichiers importants.

---

## Configuration runtime

Dossier typique :

- `outputs/configs/`

### Rôle
Conserver une trace exacte des paramètres réellement utilisés pendant l'exécution.

### Pourquoi c'est utile
Ce fichier permet :

- de reproduire une expérience ;
- de comprendre précisément quelles colonnes ont été utilisées ;
- de vérifier les paramètres d'anonymisation ;
- d'éviter les ambiguïtés entre configuration théorique et configuration exécutée.

### Format
Un fichier JSON.

---

## Dataset anonymisé public

Dossier typique :

- `outputs/anonymized/`

### Rôle
Représenter le dataset publié après anonymisation.

### Point de vue
C'est la version censée être visible par l'attaquant.

### Format
CSV.

### Contenu
On y trouve les colonnes utiles à la publication, après généralisation ou suppression.

Certaines colonnes internes, comme `record_id`, peuvent être supprimées de cette version.


---

## Dataset anonymisé d'évaluation

Dossier typique :

- `outputs/anonymized_eval/`

### Rôle
Conserver une version interne du dataset anonymisé pour l'évaluation des attaques.

### Point de vue
Ce fichier n'est pas censé être publié à l'attaquant.

### Format
CSV.

### Contenu
Il ressemble au dataset anonymisé public, mais peut conserver des colonnes internes utiles à l'évaluation, comme `record_id`.

### Pourquoi ce fichier est important
Il permet notamment de :

- vérifier les correspondances réelles ;
- savoir si une cible a bien été retrouvée ;
- calculer des métriques internes fiables.

---

## 3. Fichiers préparés pour les attaques

## Base auxiliaire pour la linkage attack

Dossier typique :

- `outputs/auxiliary/`

### Rôle
Contenir les informations connues par l'attaquant pour la linkage attack.

### Format
CSV.

### Contenu
On y trouve :

- un identifiant interne de cible ;
- uniquement les colonnes que l'attaquant est supposé connaître ;
- un sous-ensemble d'individus du dataset original.

### Utilité
Cette base sert d'entrée à `run_linkage_attack.py`.


---

## Cibles de la MIA

Dossier typique :

- `outputs/mia_targets/`

### Rôle
Contenir les individus testés par la membership inference attack.

### Format
CSV.

### Contenu
On y trouve :

- un identifiant de cible ;
- les attributs connus par l'attaquant ;
- une colonne indiquant si la cible est réellement membre ou non, par exemple `is_member`.

### Utilité
Ce fichier sert d'entrée à `run_mia_attack.py`.

---

## 4. Fichiers produits par les attaques

Les attaques produisent leurs résultats dans un sous-dossier de :

- `outputs/attacks/`

Le détail exact peut varier selon le script, mais la logique générale reste la même.

---

## Résultats de linkage attack

Dossier typique :

- `outputs/attacks/linkage/`

### Formats possibles
- CSV
- JSON
- résumés par cible
- `summary.json`

### Contenu typique
On peut y trouver :

- le nombre de candidats compatibles par cible ;
- l'existence ou non d'un vrai match ;
- la taille de la classe d'équivalence ;
- la distribution prédite de l'attribut sensible ;
- des indicateurs agrégés sur l'ensemble de l'attaque.

### Utilité
Ces fichiers servent à analyser le risque de liaison et le risque d'inférence sensible.

---

## Résultats de MIA

Dossier typique :

- `outputs/attacks/mia/`

### Formats possibles
- CSV
- JSON
- `summary.json`
- résultats détaillés par cible

### Contenu typique
On peut y trouver :

- la vérité terrain `is_member` ;
- la prédiction IN ou OUT ;
- le nombre de candidats compatibles ;
- le meilleur score ;
- la fraction compatible dans le dataset ;
- des statistiques globales de performance.

### Utilité
Ces fichiers servent à analyser le risque de fuite d'appartenance.

---

## 5. Format logique des colonnes les plus importantes

Cette section ne fixe pas tous les noms exacts possibles, mais rappelle le rôle des colonnes les plus importantes.

## `record_id`

### Rôle
Identifiant interne d'un enregistrement.

### Utilisation
Très utile pour l'évaluation interne.

### Attention
Il ne doit pas être considéré comme une information réellement publiée à l'attaquant.

---

## `income` ou autre attribut sensible

### Rôle
Attribut sensible à protéger.

### Utilisation
- pendant l'anonymisation, il intervient dans certaines contraintes ;
- pendant la linkage attack, il peut être la cible d'inférence.

---

## `is_member`

### Rôle
Label de vérité terrain pour la MIA.

### Valeurs typiques
- `1` : la cible est membre ;
- `0` : la cible n'est pas membre.

### Utilité
Permet d'évaluer la qualité des prédictions de la MIA.

---

## Attributs connus par l'attaquant

Exemples :

- `age`
- `sex`
- `race`
- `marital-status`
- `native-country`

### Rôle
Servir de base à la comparaison entre une cible et les lignes du dataset anonymisé.

### Utilisation
- dans la base auxiliaire pour la linkage attack ;
- dans les cibles MIA pour la membership inference attack.
