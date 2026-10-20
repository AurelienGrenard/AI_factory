# Revue ciblée de l'architecture multi-actifs — 2026-10-20

## Snapshot et portée

- Branche `main`, commit `0c9dace1d8a23692cbd5a960de037ed5f4bf26e0`, arbre propre avant rédaction. PR #34 fusionnée.
- Revue statique des axes II (factorisation) et III (arborescence et lisibilité) selon `query.md` v10.2. Le périmètre inclut les contrats de transitions, les moteurs mono et multi-actifs, les graphes de dérivées, les deux préparations rough, les bindings de scénarios, le manifeste et le raccordement CMake du multi-actif. Les milliers de wrappers générés ont été inspectés par leurs templates et quelques instances représentatives.
- Environnement observé : RTX 4090 Laptop SM89, `nvcc` 13.3.73, CMake 3.22.1. Aucun build, test, prix, profil GPU ou dataset n'a été exécuté pour ce passage. Les conclusions de performance sont donc des contraintes de remédiation, pas des gains mesurés.
- Inventaire suivi : `src/common` 256 fichiers, `src/model` 2 837, `src/product` 176, `tools` 283, `tests` 171. Cette taille reflète notamment les bindings générés ; elle n'est pas un indice de duplication en soi.

## Graphe des responsabilités

```text
catalog → tools/pricing → src/common/monte_carlo → src/common/simulation
                     ↘ src/correlation       ↘ src/common/volterra
                     ↘ modèles et produits   ↘ policies modèle et produit
tools/codegen → templates → bindings src/model/**/product + recettes catalog
src/common/price_gradients → préparation/stencils/reconstruction
                             ↘ graphes MC/FFT → dynamics + payoff existants
src/common/sample → path_simulation/FFT → dynamics modèle existants
```

Les `one_step_transition` des modèles sont réellement partagés entre le mono-actif et `ModelGroup` ; les adaptateurs de gradients ajoutent le couplage des innovations et réutilisent les transitions ou leur noyau numérique. Les samples passent par `path_simulation` et les dynamics des modèles. Le prix, les gradients et les Hessiennes ne partagent pas une topologie de kernel : leurs unités de travail sont respectivement prix, sensibilité/nœud et graphe de nœuds. La réduction et les produits sont partagés à leur niveau. Ce découpage est cohérent ; une fusion des moteurs n'est pas recommandée.

`MultiAssetPack` assemble les groupes markoviens ; `MultiRoughPack` assemble les spots rough et l'ordre des facteurs. `BrownianNormals` et `CorrelatedBrownianStep` sont la primitive commune de tirage/corrélation, avec un adressage Philox indépendant de `TeamSize`. Les moteurs purement markoviens et avec taux rough consomment ces mêmes briques. La FFT rough reste une phase distincte ; `causal_fft_volatility` et `hybrid_fft_volatility` préparent seulement la volatilité, puis le scan joint rejoue les innovations pour tous les spots. Le produit porte l'observation et le payoff.

La famille factorisée laisse les paramètres endogènes dans les bases `factor_parameters` et place taux/corrélations dans des datasets distincts. `assemble_factorized_scenarios` reconstitue des paramètres complets avant le calcul. La famille mono conserve ses paramètres complets ; cette coexistence a un propriétaire explicite et évite de changer l'API historique.

## Comparaisons représentatives

| Chemin | Entrée et préparation | Transition / produit | Kernel et raison de séparation |
|---|---|---|---|
| Prix mono Black–Scholes/Heston | recette générée, `tools/pricing`, schedule et dynamics modèle | `simulate_one_step` délègue à `one_step_transition`, policy produit | `monte_carlo_price_kernel`, un bloc par ligne de prix |
| Samples de modèle | binding `sample`, `path_simulation` ou FFT | mêmes dynamics, observation de sample | kernels de samples car sortie = trajectoire/calendrier, pas payoff |
| Prix + gradients / Hessiennes | `SensitivityRequest`, préparation centrale et stencils | adaptateur CRN modèle, payoff produit, reconstruction commune | kernels par sensibilité/nœud ou graphe ; état et mémoire distincts du prix seul |
| Panier markovien | `MultiAssetPack`, matrice corrélée, policy produit | transitions modèle avec innovations explicites | `multi_asset_price_kernel`, un bloc par prix et `TeamSize` configurable |
| Panier rough | volatilité causal/hybrid préparée en amont | `MultiRoughPack::advance_spots`, observation conjointe | scan spot séparé après FFT ; deux versions du même scan sont relevées sous `FACTOR-007` |
| Panier + taux | `RateAdjustedModelGroup`, policy de taux et éventuellement rough | taux intégré puis diffusé aux actions, discount final | `cross_asset_price_kernel` : plusieurs blocs par prix et workspace rough ; fusion avec le kernel markovien sans taux non motivée |

Les chemins directs `src/common → tools/catalog/maintainer` sont absents dans la recherche d'includes. Un include inverse `src/common → src/product/range_accrual` subsiste (`BOUNDARY-009`). Le manifeste typé déduit les bindings mono et leurs recettes ; les trois recettes multi-actifs sont des compositions sélectionnées explicitement, avec des générateurs courts. `AIFactoryMultiAsset.cmake` enregistre aussi explicitement les recettes `factor_parameters` sélectionnées. C'est une liste de sélection CMake, non une table de capacités des dynamics ; l'absence de génération automatique n'est pas, seule, un constat.

## Tests, outils et héritage encore actif

Les tests livrés sont rangés par contrat sous tests/simulation, tests/model,
tests/price_gradients, tests/sampling et tests/datasets ; les benchmarks et
références indépendantes consultés sont sous maintainer/tests. tools/experiments
ne contient que le schéma réutilisable des expériences locales ; les campagnes
sont sous work/experiments. Aucun déplacement vers maintainer n'est motivé par
l'inventaire de ce passage. La présence du mot legacy n'indique pas du code
mort : legacy_spot_delta_graph.cuh est appelé par de nombreuses compositions
rough de prix + delta et délègue au graphe de nœuds actuel. Ses consommateurs
interdisent une suppression sur le seul critère du nom.

## Exercices d'extension à blanc

| Extension | Propriétaire naturel | Friction observée |
|---|---|---|
| Paramètre modèle dérivable | modèle, domaine, préparation de sensibilité, loi du manifeste | coût local ; pas de transition dupliquée dans le moteur de gradients |
| Paramètre produit ou courbe | `src/product` ou `src/curve`, loader et préparation | coût local ; la quote relative a toutefois une exception produit codée dans `common` (`BOUNDARY-009`) |
| Produit terminal ou observé | policy et paramètres sous `src/product`, recette/binding | le handler donne observation et payoff sans changer la dynamique |
| Nouveau modèle MC | dossier modèle et `one_step_transition`, manifeste si mono, groupe si multi | un modèle avec taux doit satisfaire des champs implicites (`FACTOR-006`) |
| Nouvelle loi auxiliaire ou FFT | policy modèle, domaine Philox ou moteur Volterra concerné | la phase de volatilité reste distincte du scan spot |
| Ajouter un actif rough et un taux corrélé | `MultiRoughPack`, groupe, matrice réordonnée, `RatePolicy` | facteur et scène composables, mais injection du taux structurelle (`FACTOR-006`) |
| Sensibilité multi-actifs | nouvelle préparation de nœuds et stratégie CRN | capacité absente aujourd'hui ; les transitions existantes sont réutilisables, aucune duplication de dynamics requise |

## Parcours de découverte

1. Modèle : `src/model/<classe>/<famille>/<modèle>/parameters.hpp`, `parameter_domain.hpp`, `dynamics.cuh` et `dynamics_impl.cuh` ; adapter `correlated_dynamics.cuh` pour les taux.
2. Produit : `src/product/<produit>/parameters.hpp` et `pricing_policy.cuh` ; les compositions mono résident sous `src/model/**/product`.
3. Prix : `catalog/**/recipe.yaml` puis `generator.cpp`, `tools/pricing/*generation.cuh`, `src/common/monte_carlo/*kernel.cuh`.
4. Gradient : recette et binding modèle/produit, `src/common/price_gradients`, adaptateur `price_gradients/coupled_dynamics`, puis kernel terminal/path/node graph et reconstruction.
5. Cible et campagne : manifeste/codegen et `cmake/generated` pour le mono ; `cmake/AIFactoryMultiAsset.cmake` pour les compositions multi ; `docs/dataset-generation-workflow.md` pour lancer/reprendre ; `docs/audit` et `project_journal` pour limites et mesures historiques.

Les noms `ModelGroup`, `MultiAssetPack`, `MultiRoughPack`, `CorrelatedBrownianStep` et `RatePolicy` permettent de retrouver les responsabilités. `RateAdjustedModelGroup` est plus trompeur : il hérite du groupe générique mais réimplémente intégralement son ordonnanceur de transitions pour gérer des champs de modèles concrets (`FACTOR-006`). Les deux fichiers `multi_asset_rough_kernel.cuh` et `multi_asset_heterogeneous_rough_kernel.cuh` se distinguent par le type du pack, malgré la même topologie de scan (`FACTOR-007`).

## Verdict et limites

Trois constats nouveaux : `FACTOR-006`, `FACTOR-007`, `BOUNDARY-009`. Les axes II et III sont **non conformes dans ce périmètre**. Les principaux propriétaires (dynamics, produits, FFT, kernels, préparation des dérivées) sont néanmoins placés correctement ; la correction doit rester ciblée. Les axes numériques, performance, CMake complet, datasets et hygiène ne sont pas certifiés par cette revue statique. En particulier, fusionner les scans rough ou changer l'injection du taux exige une parité de prix/trajectoires et une mesure GPU avant adoption.
