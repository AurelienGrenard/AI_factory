# Tableau de bord de l'audit principal

Référentiel actif : **v10.0 — 2026-09-29**.

## État du passage v10

Le référentiel à six axes est adopté, mais aucun passage complet v10 n'a encore
été exécuté. Ce document décrit donc l'état de couverture actuel sans reprendre
comme conclusions courantes les comptes rendus historiques de l'ancien
référentiel.

- **État global :** indéterminé.
- **Dernière action :** refonte du contrat d'audit uniquement.
- **Audit du code et des résultats :** non exécuté sous v10.
- **Validation indépendante :** hors de ce passage ; elle possède son
  [propre référentiel](../validation/query.md).
- **Campagne GPU ou performance :** aucune.
- **Nettoyage, suppression ou remédiation :** aucun.

## Snapshot documentaire

- **Date :** 2026-09-29.
- **Branche observée :** `main`.
- **HEAD observé :** `27ac29535573557fc8e3e1fa44ac72c94c026d51`.
- **Worktree :** fortement modifié ; le commit seul ne suffit pas à reconstruire
  l'état audité.
- **Portée du snapshot :** refonte de
  [query.md](query.md), [status.md](status.md) et de l'en-tête de
  [response.md](response.md). Un passage v10 devra produire un snapshot
  reconstructible du code, des fichiers non suivis pertinents, du build et des
  binaires réellement examinés.

Les preuves et comptes rendus autrefois accumulés dans ce fichier restent
consultables dans l'historique Git et dans les documents auxquels renvoient les
constats. Ils sont des preuves historiques : leur compatibilité avec le
worktree courant doit être démontrée avant réemploi.

## Couverture des six axes

| Axe | Couverture v10 | Verdict v10 | État |
|---|---|---|---|
| I. Véracité | non examinée | indéterminé | modèles, produits, algorithmes, dérivées et RNG à auditer |
| II. Factorisation | non examinée | indéterminé | frontières prix/gradients/Hessiennes/codegen à auditer |
| III. Arborescence et lisibilité | non examinée | indéterminé | parcours, ownership et naming à auditer |
| IV. Performance | non examinée | indéterminé | algorithmes, kernels, mémoire et temps complets à mesurer |
| V. CMake | non examinée | indéterminé | graphe, incrémentalité et coût de build à auditer |
| VI. Hygiène et artefacts | non examinée | indéterminé | orphelins, sorties suivies et binaires périmés à auditer |

## Constats reportés dans le référentiel v10

Les huit constats de [response.md](response.md) restent ouverts jusqu'à leur
réexamen. Leur texte peut être antérieur au code courant ; ce report maintient
le risque et son identifiant, pas la validité automatique de chaque observation.

| Constat | Axe principal | Axes secondaires |
|---|---|---|
| `NUM-031` | I. Véracité | II. Factorisation, IV. Performance |
| `NUM-032` | I. Véracité | II. Factorisation, IV. Performance |
| `DELTA-001` | II. Factorisation | I. Véracité, III. Arborescence, IV. Performance |
| `PRODUCT-001` | I. Véracité | II. Factorisation, III. Arborescence |
| `NUM-028` | I. Véracité | IV. Performance |
| `NUM-029` | I. Véracité | IV. Performance |
| `NUM-030` | I. Véracité | IV. Performance |
| `PERF-016` | IV. Performance | VI. Hygiène et artefacts |

L'absence de constat principal reporté pour les axes III, V ou VI ne vaut pas
conformité. Ces axes n'ont pas encore été examinés sous v10.

## Prochain passage

Le prochain audit complet doit, dans cet ordre :

1. geler un snapshot reconstructible du worktree, du build et de la toolchain ;
2. établir les inventaires indépendants des capacités, recettes, targets et
   artefacts ;
3. examiner les six axes selon [query.md](query.md) ;
4. revalider, reformuler, clôturer ou remplacer chacun des huit constats
   reportés à partir de preuves courantes ;
5. publier ici la matrice de couverture et le verdict de chaque axe ;
6. conserver dans [response.md](response.md) uniquement les constats ouverts et
   transférer les clôtures prouvées dans [closed.md](closed.md).
