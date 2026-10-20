# Reprise de l'audit du src — 2026-10-20

## Snapshot, mandat et méthode

- Branche audit/src-reprise-20261020 créée depuis main, commit 050d759d464d39f091da66959eca362ad2b86db0 ; arbre propre avant rédaction. La revue précédente est dans architecture-multi-asset-2026-10-20.md.
- Mandat : terminer l'audit des axes II (factorisation) et III (arborescence/lisibilité) avant toute correction. Inspection statique uniquement. Aucun code source, test, build, prix ou profil GPU n'a été modifié ou exécuté.
- L'audit du 2026-10-06 couvrait déjà les unités architecturales mono. Cette reprise confronte ce baseline aux ajouts multi-actifs, au moteur rough hybride, au cross-asset, aux chemins de dérivées et aux recettes sélectionnées. Les milliers de bindings générés sont examinés par leurs templates et consommateurs, sans lecture ligne par ligne de chaque instance.

## Couverture des frontières

| Zone | Chemins représentatifs contrôlés | Conclusion |
|---|---|---|
| Prix et samples mono | common/simulation/path_simulation, common/monte_carlo/monte_carlo_kernel, common/sample, dynamics markoviennes et rough | La transition du modèle reste l'autorité. Les kernels de samples ont une sortie et une géométrie distinctes du prix ; fusion non motivée. |
| Prix, gradients et Hessiennes | common/price_gradients, common/monte_carlo/price_gradients, common/equity/price_gradients, adapters model/price_gradients, policies produit | Les graphes, stencils et réductions ont une topologie propre. Le planning des graphes est toutefois répété quatre fois (FACTOR-009). |
| Multi-actifs markoviens et taux | common/simulation/{multi_asset_path,rate_adjusted_equity_path,correlated_brownian}, common/monte_carlo/{multi_asset_kernel,cross_asset_kernel}, fixed_income/correlated_* | Pack, flux Philox, Cholesky et réduction ont des owners cohérents. Injection du taux encore implicite (FACTOR-006). |
| Rough causal et hybride | common/volterra/causal_fft_volatility, hybrid_fft_volatility, common/simulation/multi_rough_path, model/equity/rough/hybrid_multi_asset, scans MC | La phase de préparation FFT est justifiée. Les deux scans voisins relèvent de FACTOR-007 ; les formules rough hybrides ont deux propriétaires (FACTOR-008). |
| Produits, courbes et corrélations | src/product, src/curve, src/correlation, quote relative, recettes multi-actifs | Le produit possède payoff et calendrier. L'exception Range Accrual dans common reste BOUNDARY-009. Les deux recettes multi pilotes à paramètres complets ne respectent pas toutes la source de taux commune (FACTOR-010). |
| Codegen et structure livrable | tools/codegen, manifestes, templates, cmake/AIFactoryMultiAsset.cmake, catalog, tools, tests, maintainer | Le manifeste reste l'autorité des bindings mono. La liste CMake multi sélectionne des recettes, sans prétendre décrire toutes les capacités. Aucun nouveau constat sur les frontières tests/tools/maintainer. |

Un balayage des includes directs de src donne common → common 620, common → product 1, model → common 7331, model → curve 108, model → model 4543, model → product 2536, product → common 240, product → product 156, curve → common 3, curve → curve 26, correlation → correlation 1. Le seul include inverse commun → produit est celui de BOUNDARY-009. Une recherche de consommateurs sous src, tools, catalog, tests, learning et maintainer n'a pas trouvé de header direct orphelin dans common, correlation, curve ou product. Cela n'établit pas que chaque wrapper modèle est utile ; l'interface legacy_spot_delta_graph est consommée par les bindings rough et n'est pas du code mort.

## Constats supplémentaires

1. **FACTOR-008, duplication des équations rough hybrides.** Le mono et le multi possèdent chacun les formules spot/vol de quatre modèles. L'adaptateur doit garder la séparation FFT/scan, mais la formule doit rester au modèle.
2. **FACTOR-009, planification de graphes répétée.** Deux planners equity et un planner Monte Carlo (plus sa branche mixte) répètent le même choix chunk/budget/shards. Les layouts et la règle LSM de réduction complète restent des paramètres réels ; seuls le calcul géométrique et son propriétaire doivent converger.
3. **FACTOR-010, taux incohérents dans un panier sélectionné.** BS/rough Heston lit deux bases complètes, utilise leurs deux taux en drift et le seul taux BS en discount. Les huit lignes actuelles diffèrent toutes, de -6,9921 à +6,3557 points de pourcentage annuels pour taux rough moins taux BS. Une autre recette factorisée BS/Heston/Vasicek a déjà le mécanisme de composition requis.

Ces constats s'ajoutent à FACTOR-006, FACTOR-007 et BOUNDARY-009. Aucun de ces six constats n'a été corrigé pendant la revue. L'exigence utilisateur selon laquelle strikes et barrières de prix sont, sauf exception intrinsèque, exprimés en pourcentage du spot initial reste la convention des produits equity ; FACTOR-010 ne change pas cette convention.

## Parcours et exercices d'extension

1. **Nouveau modèle mono :** paramètres/domaines et dynamics sous src/model, binding via manifeste et template, samples et prix via moteurs communs. Pour le multi markovien, one_step_transition prend les innovations explicites ; l'ajout d'un taux exige actuellement les conventions implicites de FACTOR-006.
2. **Nouveau rough hybride :** dynamics et noyau Volterra propres au modèle, puis politique hybride multi. Cette dernière force aujourd'hui à recopier les formules (FACTOR-008) ; la préparation de volatilité et le scan restent deux phases.
3. **Nouveau produit terminal ou à calendrier :** paramètres et policy sous src/product ; le moteur appelle observation/payoff. La quote relative traverse common, avec l'exception concrète BOUNDARY-009. Une barrière intrinsèquement relative doit déclarer sa convention sans ajouter un test de type à common.
4. **Sensibilité mono :** sélection et préparation centrale, adapter de dynamique CRN, graphe terminal/path/mixte et reconstruction. Le passage prix → gradients/Hessiennes partage la dynamique et le payoff, mais pas la topologie des kernels. Les plans équivalents sont dispersés (FACTOR-009).
5. **Panier corrélé avec taux :** recette → datasets facteur/taux/corrélation/produit → assemble_factorized_scenarios → cross_asset_kernel. Ce parcours réussit pour BS/Heston/Vasicek ; BS/rough Heston utilise encore deux paramètres complets incompatibles (FACTOR-010).
6. **Sensibilité multi-actifs future :** le pack et les transitions offrent une base, mais aucun binding prix-gradients multi n'est livré. Son coût d'intégration reste à mesurer ; il ne justifie pas aujourd'hui de fusionner les kernels prix et graphes de sensibilités.

Pour découvrir une recette, suivre catalog/multi_asset, son generator.cpp, tools/pricing, common/monte_carlo, puis common/simulation et les policies modèles/produits. Les contrats de modèle se trouvent sous src/model ; les limites et preuves restent dans docs/audit et project_journal. Les nouveaux constats correspondent aux trois endroits où ce parcours amène à deux autorités ou à une donnée de scénario contradictoire.

## Verdict et limites

Axes II et III : **non conformes** pour le périmètre revu, avec six constats architecturaux ouverts. La plupart des frontières restent cohérentes ; aucune refonte générale du src ou fusion globale des kernels n'est étayée. Le constat de taux est une incohérence d'entrées démontrée, pas une comparaison de prix. Les propositions de factorisation doivent préserver les trajectoires/flux Philox et être qualifiées en GPU avant clôture ; cette revue n'établit aucun gain de performance, ni conformité CMake, ni exactitude des datasets historiques.
