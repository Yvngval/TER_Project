# Documentation TER

## Objectif de cette documentation

Cette documentation a pour but de présenter le fonctionnement global du projet.

---

## Contexte du projet

Le projet étudie l'effet de l'anonymisation d'un dataset sur la protection de la vie privée.

À partir d'un dataset original, on applique une anonymisation, puis on cherche à évaluer ce qu'un attaquant peut encore apprendre à partir des données publiées.

Le travail se concentre principalement sur trois parties :

- l'anonymisation ;
- la **linkage attack** ;
- la **membership inference attack (MIA)**.

Les scripts de benchmark existent dans le projet, mais ils ne constituent pas le cœur de cette documentation.

---

## Organisation de la documentation

La documentation est organisée autour des pages suivantes :

### Vue générale du projet
Présente le pipeline global, l'objectif du projet et l'enchaînement des grandes étapes.

### Anonymisation
Explique comment un dataset source est transformé en dataset anonymisé, avec production d'une version publique et d'une version d'évaluation.

### Linkage attack
Décrit comment un attaquant utilisant une base auxiliaire peut chercher des enregistrements compatibles dans le dataset anonymisé et en déduire un attribut sensible.

### Membership Inference Attack (MIA)
Décrit comment un attaquant peut essayer de prédire si une cible appartenait ou non au dataset ayant servi à produire les données publiées.

### Structure des scripts
Donne une vue claire de l'organisation du dossier `scripts/` et du rôle des principaux fichiers.

### Formats des fichiers d'entrée / sortie
Présente les principaux fichiers du projet, leur rôle et leur place dans le pipeline.


---

## Pipeline global

Le projet suit globalement la logique suivante :

1. partir d'un dataset source ;
2. charger une configuration d'anonymisation ;
3. exécuter l'anonymisation ;
4. produire un dataset anonymisé public et un dataset anonymisé d'évaluation ;
5. préparer les fichiers nécessaires aux attaques ;
6. exécuter la linkage attack et la MIA ;
7. analyser les résultats produits.
