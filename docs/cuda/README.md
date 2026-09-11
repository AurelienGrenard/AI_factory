# Documentation CUDA

Ce dossier contient les contrats normatifs et les références d'architecture
des backends CUDA. Commencer par le besoin à traiter ; les détails de
catalogue, de publication et de validation indépendante restent dans
[`docs/README.md`](../README.md).

## Trouver le bon document

| Besoin | Document autoritatif |
|---|---|
| Comprendre l'assemblage complet d'un prix ou d'un sample | [Composition des politiques de pricing](pricing-policy-composition.md) |
| Ajouter ou modifier une dynamique | [Contrat des dynamiques](model-dynamics-contract.md) |
| Ajouter ou modifier une fonction analytique | [Contrat des analytics](model-analytics-contract.md) |
| Ajouter un pricer fermé ou Monte Carlo | [Contrat closed form et Monte Carlo](closed-form-and-monte-carlo-pricing-contract.md) |
| Calculer prix et delta equity par bump de S0 | [Contrat prix-delta et périmètre pilote](equity-price-delta-contract.md) |
| Ajouter un produit américain ou bermudéen | [Contrat Longstaff--Schwartz](american-and-bermudan-pricing-contract.md) |
| Planifier un pricing, valider un lancement ou inspecter ses ressources | [Validation et diagnostic des kernels](launch-validation-and-kernel-diagnostics.md) |
| Mesurer une régression ou qualifier un GPU | [Protocole de performance](../performance-regression-protocol.md) |

## Composition

[La référence de composition](pricing-policy-composition.md) couvre les moteurs
actifs : transitions markoviennes exactes ou à pas fixe, Volterra FFT, lift
rough N-facteurs, formules fermées equity et fixed income,
Longstaff--Schwartz, et sampling modèle. Elle montre les frontières entre :

- dynamique et préparation du modèle ;
- calendrier et schedule numérique ;
- état de chemin et payoff produit ;
- politique de pricing ou d'observation ;
- moteur, workspace, kernel et launcher public.

Les documents spécialisés définissent ensuite les signatures et invariants
obligatoires. Une description locale ne doit pas recopier ces contrats.

## Matériel et performance

Le [notebook SM89](rtx4090-laptop-memory-map.ipynb) décrit uniquement la RTX
4090 Laptop de référence. Ses valeurs de mémoire, géométrie et occupation ne
sont pas des valeurs universelles.

Le [protocole de performance](../performance-regression-protocol.md) explique
comment produire une baseline et des profils distincts pour un autre GPU ou
toolchain. Aucun changement de tuning n'est accepté sur la seule base des
mesures SM89.

## Frontières documentaires

- Les contrats de ce dossier possèdent les interfaces et invariants CUDA.
- Le [workflow d'extension](../catalog-extension-and-validation-workflow.md)
  possède la séquence complète modèle, produit, recette, test et publication.
- Le [contrat des samples](../model-sample-dataset-generation.md) possède la
  forme des datasets d'entraînement.
- Le [pipeline de validation](../independent-price-validation-pipeline.md)
  possède les références Premia/QuantLib et les caches.
- Le [manifeste typé](../../tools/codegen/pricing_bindings/capability_manifest.py)
  possède la matrice active des modèles, produits, moteurs et bindings.

Une nouvelle règle est ajoutée au document qui la possède ; les autres pages
la résument uniquement pour orienter le lecteur.
