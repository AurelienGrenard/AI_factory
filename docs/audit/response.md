# Constats d'audit non résolus

## Objet

Ce document contient exclusivement les constats non résolus du passage
indépendant version 7 exécuté le 2026-08-30 sur le worktree courant attaché à
`5d65c23cf709173a65ab809baff0da2566b58acf`. La provenance, les inventaires,
les matrices de couverture, les exclusions et les commandes sont consignés
dans `docs/audit/status.md`.

Un constat nouveau reste ouvert.

| Sévérité | Constats |
|---|---|
| Moyenne | `STRUCT-019` |

## Priorité haute

### STRUCT-019 — Donner au gate Performance un propriétaire dans le périmètre principal

- **État :** ouvert.
- **Sévérité :** moyenne.
- **Priorité :** haute.
- **Confiance :** prouvée.
- **Propriétaire :** non attribué.
- **Dernière vérification :** 2026-08-30.
- **Cause primaire :** l'infrastructure qui prétend produire la décision de
  performance du dépôt vit exclusivement sous `validation/**`, alors que le
  référentiel version 7 impose les sources, fixtures, manifestes et baselines
  principales sous `tests/performance`, et les runners/checkers sous
  `tools/performance`.
- **Fichiers et preuve :** `tests/performance` et `tools/performance` sont
  absents. Les onze fichiers concernés sont sous `validation/performance`,
  dont `validation/performance/baseline_sm89_v2.json`, quatre sources de
  benchmark, `validation/performance/run_baseline.py` et
  `validation/performance/check_baseline.py`. `CMakeLists.txt:411-570` et
  `CMakeLists.txt:1177-1200` enregistrent directement ces composants. Le
  propriétaire est même revendiqué par `validation/performance/README.md:1-6`.
- **Impact :** aucun résultat de cette infrastructure ne peut constituer la
  preuve principale des quatre sous-audits Performance version 7. Le gate
  `P0-SCOPE` et la couverture Performance restent donc partiels, même si les
  binaires et la baseline de validation peuvent fournir des éléments
  complémentaires.
- **Correction minimale :** posséder benchmarks, fixtures, manifeste et
  baseline principaux sous `tests/performance`; posséder runners, checkers et
  outils de profilage sous `tools/performance`; conserver le protocole durable
  sous `docs`; remplacer la sélection best-of-N par l'agrégation complète
  pré-déclarée, séparer les quatre temps et appliquer tous les budgets version
  7; adapter le graphe CMake et les imports sans perdre de clé.
- **Clôture :** les propriétaires principaux existent dans le périmètre, aucun
  composant décisionnel principal ne dépend exclusivement de `validation/**`,
  et le déplacement n'a pas simplement reproduit le protocole complémentaire.
  Le manifeste et le checker rejettent toute clé absente/inconnue/dupliquée,
  séparent kernel/API publique/pipeline/publication, interdisent best-of-N,
  minimum par clé et baseline recomposée, conservent les campagnes brutes et
  appliquent les budgets numériques, ressources, VRAM et profils par
  architecture de la version 7. Les quatre rapports sont reproductibles depuis
  ces propriétaires sans rebaseline auto-validant.
- **Historique :** aucune signature équivalente trouvée dans les registres.
