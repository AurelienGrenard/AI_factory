# Prix, gradients et diagonales de Hessienne fixed income

## Périmètre actif

Deux stratégies fixed income utilisent le même contrat de sensibilités
sélectionnées :

1. la swaption européenne régulière sous CIR, payer ou receiver, par
   décomposition de Jamshidian ;
2. la swaption bermudéenne co-terminale, payer ou receiver, par
   Longstaff--Schwartz avec politique d'exercice centrale gelée.

Les deux stratégies publient le prix central, les dérivées premières
sélectionnées et, sur demande, la diagonale de Hessienne. Les dérivées croisées
ne font pas partie de ce contrat.

La voie Bermudan couvre les dix compositions déjà disponibles pour le prix :

- CIR autonome ;
- CIR++ avec courbe Nelson--Siegel ou Svensson ;
- G2 autonome ;
- G2++ avec courbe Nelson--Siegel ou Svensson ;
- Hull--White avec courbe Nelson--Siegel ou Svensson ;
- Ornstein--Uhlenbeck autonome ;
- Vasicek autonome.

Le calendrier reste discret. `first_exercise_time_days`,
`payment_interval_days`, `payment_count`, `exercise_count` et la maturité
d'exercice anticipé ne sont pas des coordonnées différentiables. Les
coordonnées produit continues sont `product.notional`, `product.strike` et
`product.accrual_fraction`.

Les adaptateurs exposent toutes les coordonnées continues de leur modèle :

| Modèle | Coordonnées modèle |
|---|---|
| CIR, CIR++ | `mean_reversion`, `long_term_mean`, `volatility`, `initial_state` |
| G2 | `mean_reversion_x`, `volatility_x`, `mean_reversion_y`, `volatility_y`, `correlation`, `initial_state_x`, `initial_state_y` |
| G2++ | `mean_reversion_x`, `volatility_x`, `mean_reversion_y`, `volatility_y`, `correlation` |
| Hull--White | `mean_reversion`, `volatility` |
| Ornstein--Uhlenbeck | `mean_reversion`, `volatility`, `initial_state` |
| Vasicek | `mean_reversion`, `long_term_mean`, `volatility`, `initial_state` |

Les modèles ajustés ajoutent les coordonnées de courbe : `beta0`, `beta1`,
`beta2`, `tau` pour Nelson--Siegel ; `beta0`, `beta1`, `beta2`, `beta3`,
`tau1`, `tau2` pour Svensson. Les recettes générées sélectionnent ces
coordonnées avec des bumps explicites. Les prédicats de domaine appartiennent
au modèle, à la courbe ou au produit ; le loader JSON et la préparation GPU
appellent le même prédicat.

## Plan compact et préparation des sensibilités

L'hôte conserve uniquement :

```text
models[M]
curves[C]                    # modèles ajustés uniquement
products[P]
sensitivities[K] = {paramètre résolu, BumpConfiguration}
construction = aligned | cartesian
SensitivityRequest = first | second | first_and_second
```

Il ne matérialise pas `N*(1+2*K)` scénarios. Le mapping canonique retrouve le
modèle, la courbe éventuelle et le produit de chaque ligne. Les noms sont
résolus une fois sur l'hôte en identifiants typés ; aucun kernel ne compare de
chaînes.

Pour chaque couple `(ligne, sensibilité)`, le thread 0 du bloc appelle
`build_sensitivity_task`. Le couple formé par les paramètres centraux et le
`SensitivitySpec` construit :

- trois nœuds pour l'ordre un et les stencils diagonaux centrés ;
- quatre nœuds pour une diagonale unilatérale d'ordre deux ;
- le `SensitivityStencil` avec les valeurs FP32 et les poids effectivement
  utilisés ;
- le besoin éventuel du nœud central pour reconstruire la dérivée.

Une frontière produit un stencil unilatéral d'ordre deux. Aucun clamp
silencieux n'est admis. Les `N*K` stencils sont des sorties de preuve ; les
entrées restent en `O(M+C+P+K)`.

## Jamshidian coopératif

Les voies scalaire et coopérative utilisent les mêmes analytics CIR, le même
calendrier et la même reconstruction :

- `scalar` : un thread possède une ligne et évalue ses sensibilités
  successivement ;
- `cooperative` : un bloc possède une ligne et coopère sur le solveur de
  Jamshidian.

Le central est écrit une fois. Le kernel coopératif réutilise son workspace
partagé pour chaque nœud ; le launcher peut employer la voie scalaire si le
workspace coopératif n'est pas admissible. `reconstruct_sensitivity` reste
l'unique logique de reconstruction.

Les mesures historiques du 22 septembre 2026 sur SM89, pour 1 000 lignes,
sept sensibilités et au plus douze paiements, donnent 1,114 ms pour l'ordre un
et 1,841 ms pour ordre un plus diagonale sur la voie coopérative. Elles ne
qualifient ni la voie Bermudan ni un autre GPU.

## Longstaff--Schwartz à exercice gelé

La pipeline centrale reste celle du prix seul : simulation forward, régressions
backward, décisions d'exercice puis réduction du prix. Elle enregistre pour
chaque chemin l'indice d'exercice central. Le prix et son erreur standard sont
ainsi calculés une seule fois.

Si `K>0`, deux kernels sont ajoutés après la pipeline centrale :

1. `frozen_sensitivity_moments_kernel` utilise une grille
   `(path shard, row*sensitivity)`. Un bloc possède une sensibilité d'une ligne
   et ses threads se partagent les chemins. Pour chaque chemin, les nœuds sont
   rejoués successivement avec la même clé Philox et la décision d'exercice
   centrale ; cette exécution séquentielle borne les états vivants et la
   pression registre. Le nœud central est rejoué uniquement quand le stencil
   en a besoin.
2. `finalize_frozen_sensitivities_kernel` réduit en FP64 les moments partiels,
   puis écrit gradient, diagonale et erreurs standards en ordre de sélection.

Le cas `K=0` délègue à la voie prix seul. Le planner conserve la géométrie LSM
native : 128 threads, 64 blocs de chemins par prix fixed income. Le planner de
workspace natif choisit le nombre de prix résidents d'après la VRAM disponible.
Le générateur borne séparément à 16 lignes l'intervalle durable entre deux
checkpoints. Le workspace ajoute l'indice d'exercice par chemin, les moments
par sensibilité et les sorties `N*K` ; il ne réserve aucun tableau de scénarios
bumpés.

CIR/CIR++ utilisent la voie de mesure terminal-forward avec tables
d'observation par nœud. G2, G2++, Hull--White, OU et Vasicek utilisent la voie
jointe état/intégrale. Cette différence appartient aux policies de pricing ;
les kernels de moments et de finalisation restent communs.

## Génération et provenance

Le codegen produit, pour chacune des dix compositions Bermudan : payer et
receiver, construction alignée et cartésienne, ordre un et ordre un plus
second diagonal. Cela représente 80 recettes permanentes sous
`catalog/model/fixed_income/<model>/price_gradients/`.

Chaque recette contient l'URL `/v2/`, les datasets modèle/courbe/produit, la
seed, le mapping `philox_source_step_v2`, les bumps sélectionnés et la recette
de prix source. La génération stochastique écrit un checkpoint durable après
chaque lot et peut reprendre au premier préfixe incomplet. Le writer conserve
les identifiants `model_id`, `curve_id` et `product_id` dans l'ordre canonique
`model, curve, product`.

## Responsabilités

| Responsabilité | Emplacement |
|---|---|
| Specs, tâches, stencils et reconstruction | `src/common/price_gradients/` |
| Plan compact modèle/courbe/produit | `src/common/fixed_income/price_gradients/` |
| Workspace et kernels LSM gelés communs | `src/common/longstaff_schwartz/price_gradients/` |
| Composition Bermudan et replay du payoff | `src/product/bermudan_swaption/price_gradients/` |
| Accès et domaines modèle | `src/model/fixed_income/<model>/{parameter_domain.hpp,price_gradients/}` |
| Accès et domaines courbe | `src/curve/<curve>/{parameter_domain.hpp,price_gradients/}` |
| Bindings publics | `src/model/fixed_income/<model>/product/.../bermudan_swaption_price_gradients.cuh/.cu` |
| Génération, checkpoint et artefact | `tools/pricing/price_gradients/`, `tools/datasets/price_gradients/` |
| Templates, manifeste et recettes | `tools/codegen/pricing_bindings/`, `catalog/model/fixed_income/` |

## Preuves et limites

Le test CUDA permanent couvre CIR terminal-forward, Vasicek joint un facteur,
G2 joint deux facteurs et CIR++/Nelson--Siegel ajusté. Il vérifie la parité
bitwise du prix et de son erreur standard avec le pricer central, la finitude
des ordres un et deux, les propriétaires modèle/courbe/produit, les sélections
réordonnées et l'invariance des gradients quand la diagonale est aussi
demandée. Les dix bibliothèques de binding et des générateurs représentatifs
des dix compositions compilent. La régénération compare exactement les 80
recettes, 20 wrappers et deux manifestes.

Sur SM89, l'inspection statique du binaire de test donne pour G2
`frozen_sensitivity_moments_kernel` 168 registres/thread, 32 octets de stack,
1 056 octets de shared et zéro octet local à l'ordre un. L'ordre deux diagonal
et l'ordre un plus deux atteignent 242 registres/thread, 464 octets de stack,
1 296 octets de shared et zéro octet local déclaré. Le retour à une boucle
forcée non déroulée a augmenté simultanément registres et stack ; la version
compilée conserve donc le déroulage actuel. Cette pression reste une limite à
mesurer de bout en bout sur secteur.

Ces contrôles qualifient l'intégration et la reproductibilité. Ils ne
certifient pas encore les bumps, le biais de politique gelée, la convergence
LSM ni la performance de production. Une campagne numérique multi-seeds et
une mesure complète sur secteur sont requises avant publication d'une base de
Hessiennes Bermudan.
