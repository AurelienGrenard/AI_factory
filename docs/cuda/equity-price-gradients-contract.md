# Prix et gradients equity sélectionnés

## Périmètre de la première intégration

Les launchers `european_option_price_gradients` couvrent les calls et puts
Black–Scholes (formule fermée et Monte Carlo terminal exact), Heston
(Monte Carlo QE-M à pas fixe), CEV (Milstein absorbant à pas fixe) et Merton
(Monte Carlo terminal exact à horizon fixe). Les paramètres supportés sont déclarés dans
`price_gradients/parameter_policy.cuh` de chaque modèle et produit.

Pour Merton, l'intensité de saut est exclue et la maturité européenne reste une
fonctionnalité explicitement ouverte, non une exclusion définitive :
le premier lot partage le compte de Poisson et les normales canoniques à
horizon et intensité inchangés. La maturité sera ajoutée seulement avec un
couplage exact des comptes et marques entre horizons. Cette intégration ne
migre pas encore les autres modèles et produits. Les intensités de saut sont
exclues du chantier.

Les calls et puts américains Heston disposent d'un premier moteur sélectionné.
Il résout une seule stratégie Longstaff--Schwartz centrale, enregistre sa date
et son spot d'exercice par chemin, puis rejoue chaque paire bumpée avec cette
stratégie gelée. Il couvre spot, taux, dividende, variance initiale, `kappa`,
`theta`, `gamma`, `rho` et strike. La maturité et l'intervalle d'exercice ne
sont pas admissibles. Les moteurs `price_delta` restent les
références de compatibilité pendant cette migration ; leur suppression exige
la couverture et la validation de tous leurs consommateurs.

## Sélection et préparation

`PriceGradientConfiguration` contient une liste ordonnée de `Sensitivity`.
Chaque entrée identifie une coordonnée (`model.spot`, `product.strike`, etc.)
et son `BumpConfiguration`. Une liste vide demande seulement le prix.
Aucune sensibilité n'est inférée de la valeur d'un paramètre : notamment,
`r = d = 0` n'ajoute pas de dérivées par rapport aux taux.

Exemple de sélection :

```cpp
price_gradients::PriceGradientConfiguration selection{{
    {"model.spot", {.005, price_gradients::BumpScale::relative}},
    {"model.risk_free_rate", {.0005, price_gradients::BumpScale::absolute}},
    {"product.strike", {.005, price_gradients::BumpScale::relative}}
}};
```

Le déplacement est **celui d'un seul côté**. `.005` relatif pour le spot
correspond donc à la largeur totale historique `.01` de `price_delta`.
La configuration conserve la valeur demandée en double ; sa représentation
FP32 est utilisée pour construire les scénarios, conformément au pricing.

Le mode est propre à chaque coordonnée sélectionnée : `relative` donne
`h = déplacement × abs(paramètre)`, tandis que `absolute` donne directement
`h = déplacement`. Les points de base sont seulement une unité pratique pour
un déplacement absolu de taux : 5 pb correspondent à `0.0005` par côté.
Les paramètres positifs comme spot, strike ou volatilité peuvent utiliser une
échelle relative ; un déplacement absolu reste défini à zéro et permet une
échelle fixe pour la variance initiale ou la corrélation. Ce sont des choix de
recette configurables, pas des tailles optimales garanties par le type du
paramètre. Les limites du domaine restent contrôlées dans les deux modes.

La préparation hôte résout les noms, valide le domaine complet de chaque
scénario et construit `1 + 2*K` scénarios par ligne. Les entrées sont rangées
par ligne, puis scénario central, première et seconde extrémités de chaque
coordonnée dans l'ordre demandé. Les sorties gradients et erreurs standards
sont rangées par ligne puis coordonnée. Les noms ne sont pas interprétés dans
les boucles CUDA. Le dispatch MC compile des largeurs de lots bornées, sans
instancier toutes les cardinalités ou combinaisons de paramètres. La formule
fermée conserve son dispatch par cardinalité et évalue les scénarios successivement.

Le plan et ses buffers CUDA sont des miroirs appartenant à l'appelant : après
préparation, ne pas modifier les scénarios ou stencils et téléverser exactement
ce plan. Le launcher vérifie dimensions, capacités, plages, géométrie et
absence de recouvrement entre entrées et sorties. Il ne relit pas les buffers
GPU pour prouver leur égalité au miroir hôte.

## Frontières et calendriers

Le stencil est centré si les deux points sont admissibles. Sinon, la politique
`central_then_one_sided_order2` utilise deux points du même côté à `h` et `2h`,
avec les coefficients d'ordre deux calculés sur les **points représentés**.
Si aucun côté n'est admissible, la préparation échoue avant lancement.
`central_only` interdit ce repli. Aucun clamp ou rétrécissement silencieux
n'est effectué. Un bump relatif à zéro échoue : choisir explicitement un
bump absolu.

Le dataset enregistre les extrémités FP32, la largeur représentée, le stencil
et ses coefficients. L'erreur standard MC est celle de la différence couplée
par trajectoire, et non celle obtenue en combinant des erreurs de prix
indépendants. Elle ne mesure pas le biais du bump.

La coordonnée `product.maturity_years` est distincte du champ d'entrée entier
`maturity_days` (clé JSON `maturity`). Sa perturbation doit correspondre à un
nombre entier de pas `dt`. Le minimum est donc un pas, par défaut `1/504` an.
Les scénarios portent leurs dates numériques en nombres entiers de pas ; on
ne tronque pas une demi-journée dans un champ entier de jours. Cette convention
est actuellement réservée au produit européen terminal.

## CRN et responsabilité des dynamiques

Chaque ligne conserve la clé Philox `base_seed + global_row`, avec compteurs
`(path_index, local_group_index)`. La sélection, la largeur `B`, le numéro du lot de sensibilités et le découpage
des lignes ne changent pas cette adresse. L'ordre des tirages Heston est celui du
moteur existant. Les `coupled_dynamics` adaptent les transitions canoniques ;
elles ne recopient ni équations du modèle ni préparation de ses coefficients.

Pour un horizon fixe, les scénarios consomment les mêmes innovations. Heston
évolue dans chaque lot sur la grille commune jusqu'à sa plus grande maturité et fige chaque
état terminal à sa propre date. Pour Black–Scholes exact, le premier normal
conserve le terminal central ; deux normales supplémentaires construisent
les autres dates par pont/extension browniens. Cela assure la covariance
`Cov(W(s), W(t)) = min(s,t)`, y compris pour un stencil de maturité unilatéral.

Black–Scholes et Heston sont homogènes en spot : la trajectoire
peut partir du spot central et l'observation appliquer le rapport des spots.
CEV déclare explicitement `kMultiplicativeSpot=false` : chaque spot bumpé
initialise son propre état, puis reçoit les mêmes normales à chaque pas.
Son domaine `beta ∈ [0.5, 1)` est conservé, avec stencil unilatéral si nécessaire.
Cette propriété doit être déclarée par modèle avant toute généralisation.
Merton est également homogène en spot. Son adaptateur appelle le tirage
canonique à consommation variable une seule fois, puis partage le compte de
Poisson, la normale de diffusion et la normale de somme des sauts entre les
scénarios admissibles. La factorisation est valide parce que l'intensité et
l'horizon restent fixes ; elle ne synchronise jamais plusieurs suites Philox
indépendantes.
Les sept gradients calls/puts sont comparés au même stencil évalué par une
formule Merton indépendante en double, somme de Poisson de prix lognormaux.
Le cas d'intensité nulle vérifie séparément les sensibilités de sauts nulles à
la précision FP32. Un probe avant/après conserve tous les terminaux bit à bit,
54 registres et zéro octet local sur SM89 ; ses timings alternés ne résolvent
aucune régression. Ces preuves ne qualifient pas encore la performance du
launcher complet.
Les scénarios sont logiquement distincts. La préparation repère ceux dont
la dynamique et la maturité coïncident avec le central : leur payoff réutilise
alors son état terminal. Les autres scénarios partagent les innovations.
La mutualisation des coefficients préparés reste à approfondir.

La dernière addition diffusion/dérive du pas QE-M Heston emploie désormais
un `fmaf` explicite dans la dynamique canonique. Cette contraction était déjà
présente dans le binaire `price_delta` de référence ; l'expliciter empêche le
compilateur de la changer lorsqu'il déroule plusieurs scénarios. La vérification
compare aussi le binaire antérieur, pas seulement deux launchers recompilés.

Les états et payoffs restent FP32 ; les sommes et carrés MC sont FP64.
L'équivalence bit à bit avec `price_delta` nécessite aussi les mêmes points
représentés, outil de compilation, GPU, géométrie, ordre de réduction et
contractions arithmétiques. Les tests publics vérifient la compatibilité du
prix, du delta et de leurs erreurs standards pour les cas couverts.

## Arborescence et génération

| Responsabilité | Emplacement |
|---|---|
| Sélection, stencils, buffers et dispatch | `src/common/price_gradients/` |
| Préparation des scénarios et couplage terminal | `src/common/equity/price_gradients/` |
| Kernels MC et formule fermée | `src/common/{monte_carlo,closed_form}/price_gradients/` |
| Paramètres et adaptation des transitions | `src/model/equity/markovian/<model>/price_gradients/` |
| Adaptateurs du produit européen | `src/product/european_option/price_gradients/` |
| Trace d'exercice gelée | `src/common/longstaff_schwartz/frozen_exercise_trace.cuh` |
| Workspace, planner et kernels LSM gradients | `src/common/longstaff_schwartz/price_gradients/` |
| Règles du produit américain | `src/product/american_option/price_gradients/` |
| Launchers générés | `src/model/equity/markovian/<model>/product/` |
| Manifeste et rendu dédiés | `tools/codegen/pricing_bindings/price_gradients/` |
| Templates dédiés | `tools/codegen/pricing_bindings/templates/**/price_gradients/` |
| Planification, exécution, sérialisation | `tools/{cuda,pricing,datasets}/price_gradients/` |
| Recettes calls/puts, alignées/cartésiennes | `catalog/model/equity/markovian/<model>/price_gradients/` |
| Tests de configuration, CUDA, artefacts | `tests/price_gradients/` |

Le manifeste général agrège ces déclarations. Les launchers et recettes sont
régénérés par la commande habituelle `generate.py --family all --output .`.
Le contrôleur `tools/datasets/generate_catalog.py` connaît `price_gradients`
et réutilise la provenance, les checkpoints et la progression existants.
Aucun dataset n'est certifié du seul fait de sa génération : la validation
indépendante reste explicitement en attente.

Heston américain génère quatre recettes permanentes : calls/puts, construction
alignée/cartésienne. Le runner découpe la campagne en tranches durables de 16
lignes ; chaque appel est ensuite replanifié par le LSM selon la VRAM libre.
`result_offset` décale ensemble les scénarios, stencils, clés Philox et indices
de sortie. Un checkpoint n'est validé qu'après la copie des prix, erreurs et
colonnes gradients de toute la tranche. Une interruption reprend donc au
dernier multiple de 16 confirmé, ou à la dernière tranche plus courte.

## Géométrie et qualification

Un bloc MC traite **un couple (prix, lot de sensibilités)**. Chaque thread
simule ses indices de trajectoire habituels, avec au plus `1+2*B` états et
`B+1` canaux de moments FP64. `K` est le nombre total de coordonnées choisies ;
`LaunchConfiguration::sensitivity_batch_size` est une limite indépendante,
valant **1 par défaut**, avec les choix explicites **1, 2 ou 4**.
Chaque tâche de bloc traite donc par défaut une seule coordonnée d'un prix.

Les lots complets sont aplatis dans une grille `(ligne, lot)` et parcourus
par pas de `gridDim.x`. Un éventuel reste est soumis dans une seconde grille,
spécialisée à sa taille exacte (1, 2 ou 3). Cela évite le remplissage artificiel
et les masques dans la boucle des trajectoires. Si `K<B`, seule cette petite
grille est lancée. Si `K=0`, une grille prix seul est lancée. Le plafond
`block_count` s'applique à chaque grille, limité à ses tâches réelles ; il peut
maintenant dépasser le nombre de lignes. Il ne peut dépasser le nombre total
de couples `(ligne, lot)` de la soumission.

Les scénarios hôtes restent stockés en `1+2*K` : chaque lot utilise une vue
sur le central et ses paires. Il rejoue le même flux Philox ; son horizon peut
être plus court, mais son préfixe d'innovations est identique. Seul le lot
commençant à la coordonnée zéro écrit le prix et son erreur standard. Chaque
lot écrit ses colonnes dans l'ordre de la sélection. Les réductions conservent
l'ordre des chemins et l'accumulation FP64 du moteur précédent.

La préparation hôte classe chaque paire dans `CentralRequirement` :

- `none` : stencil centré et aucune trajectoire mutualisée ;
- `state` : trajectoire centrale réutilisée, sans besoin de son payoff ;
- `payoff` : stencil unilatéral nécessitant aussi le payoff central par chemin.

Ce classement réutilise le stencil et les indicateurs `reuse_central` déjà
calculés. Le bloc prend le besoin maximal de ses coordonnées ; le premier lot
prend toujours `payoff` pour produire le prix. Une branche uniforme, **avant**
la boucle des chemins, sélectionne une spécialisation compilée. Elle conserve
le flux Philox, sans ajouter de branche de sélection dans cette boucle.

**Le prix moyen et son erreur standard sont calculés et écrits une seule fois
par ligne**, dans le premier lot. Les autres lots ont uniquement `B` canaux de
moments FP64. Leur travail central dépend du besoin préparé :

| Lot | États logiques | Payoff central | Moments du prix |
|---|---:|---|---|
| Premier lot | `1+2*B` | Oui | Oui |
| Autre lot, `none` | `2*B` | Non | Non |
| Autre lot, `state` | `1+2*B` | Non | Non |
| Autre lot, `payoff` | `1+2*B` | Oui, pour le gradient | Non |

Les trajectoires mutualisées peuvent réduire encore le nombre de transitions
réellement effectuées. Le payoff central par chemin reste nécessaire aux
stencils unilatéraux et à leur erreur standard couplée : le prix moyen produit
par le premier lot ne le remplace pas. Ce rejeu local exceptionnel évite un
cache global de taille `nombre de prix × nombre de trajectoires` et une
synchronisation supplémentaire entre blocs.

Pour `B=1`, un gradient centré ordinaire hors premier lot simule donc exactement
ses deux extrémités. En l'absence de dépendances au central et de mutualisation,
`K=10, B=1` demande 21 évaluations logiques, contre 30 avec un central
systématique dans chacun des dix lots. Le modèle n'est jamais recopié dans le
moteur générique : ses transitions canoniques restent appelées.

Chaque lot régénère son flux d'innovations. Augmenter `B` amortit ce travail,
mais augmente les états/moments vivants et peut réduire l'occupation. Les
spécialisations de travail partagent actuellement un même kernel par largeur :
son allocation de registres reste déterminée par sa branche la plus exigeante.
Il faut donc mesurer le couple `(B, threads_per_block)` avec la charge visée.
La formule fermée conserve un thread par prix et gradients, avec des blocs de
256 threads par défaut (profil analytique configurable). Ce thread évalue le
central une seule fois, puis les deux extrémités de chaque coordonnée
séquentiellement et écrit leur différence finie. La sélection de `K`
coordonnées demande donc `1+2*K` évaluations de formule, sans trajectoire ni
flux aléatoire à conserver. Le découpage MC en lots ne s'applique pas.
La boucle étant déroulée par le compilateur, l'allocation effective de
registres doit néanmoins être mesurée pour chaque spécialisation.

Le planificateur dédié calcule la grille en fonction des lignes **et** du
nombre de lots. Il expose `sensitivity_batch_size`,
`sensitivity_batches_per_price`, `maximum_live_scenarios`,
`kernel_launches_per_price_batch`, ainsi que les grilles complète et résiduelle.
Les champs `price_moment_batches_per_price=1` et `central_work_policy` décrivent
également l'unicité du calcul du prix et la classification du central. Ces champs
entrent dans la provenance d'exécution et l'identité du checkpoint.
Pour la formule fermée, les deux champs de lots valent zéro. Le défaut MC est
`(B=1, threads_per_block=256)` ; il reste un **candidat non qualifié en performance**
sur l'ensemble des modèles, produits et GPU.

```bash
./build/inspect_pricing_launch_plan heston/european_option 1000 --price-gradients 10 --sensitivity-batch-size 2
cmake --build build --target benchmark_price_gradients_batching -j2
AI_FACTORY_CUDA_KERNEL_DIAGNOSTICS=1 ./build/benchmark_price_gradients_batching 1 256 64
```

Le benchmark ciblé `tests/performance/price_gradients/batching.cu` conserve
les sorties binaires, médiane, p95 et CV pour Black–Scholes et Heston ; il
utilise les cinq préchauffages et 21 répétitions du support de performance
commun. Les intervalles BS regroupent 64 appels identiques par répétition,
avec des temps normalisés par appel ; Heston utilise un appel. Le même
regroupement doit être utilisé pour la référence et le candidat. Les diagnostics exposent les ressources du kernel réellement
spécialisé. Une qualification doit suivre le
[protocole de performance](../performance-regression-protocol.md), avec trois
campagnes, comparaison sur le même matériel et mesure indépendante du binaire
compilé. Les résultats d'un point de mesure ne qualifient pas toute l'enveloppe
des maturités, nombres de prix et trajectoires.

Les tests CUDA ordinaires conservent leur volume numérique complet. Pour les
outils de concurrence/synchronisation, le même exécutable accepte
`--sanitizer` : 1 025 trajectoires BS et 513 Heston, avec la même matrice de
sélections, lots, frontières et blocs, plusieurs itérations de chemins et une
dernière itération partielle. Ce mode réduit le coût d'instrumentation ; il
complète les tests numériques complets et ne les remplace pas.

### Pipeline américaine gelée

Le moteur américain conserve les kernels centraux : préparation des lignes,
simulation forward, régressions backward, moments et finalisation du prix. La
politique ajoute une trace compacte `(observation, spot)` par chemin et une
décision initiale par ligne. Après la finalisation du prix, exactement deux
kernels supplémentaires sont lancés si `K>0` :

1. `frozen_gradient_moments`, sur une grille `(Bp, N*K)`, rejoue les deux
   scénarios d'une sensibilité avec les innovations Philox de la ligne et du
   chemin central, puis écrit somme et somme des carrés en FP64 ;
2. `finalize_frozen_gradients`, sur `N*K` blocs, réduit ces moments et écrit les
   sorties rangées `[ligne*K + sensibilité]`.

`B=1` est contractuel pour cette première intégration : une tâche de bloc porte
une sensibilité. `Bp=LaunchConfiguration::block_count` reste le nombre de
shards de chemins par prix, comme dans le LSM existant. Le workspace commun
réserve `2*K*Bp` doubles de moments par ligne, réutilisés après la finalisation
du prix. Le planner central continue de déterminer les lots de lignes selon la
VRAM disponible et les plafonne aussi à `maxGridSize[1]/K`, afin que la grille
aplatie `N*K` reste représentable. Les buffers de scénarios, stencils et sorties appartiennent
à l'appelant et sont déjà visibles dans la mémoire libre interrogée.

Une sensibilité spot ou strike homogène réutilise le spot central enregistré.
Les paramètres de dynamique Heston rejouent deux états QE-M : un seul triplet
d'innovations est tiré par pas puis appliqué aux deux transitions. Un bump de
taux modifie aussi les facteurs d'actualisation du payoff gelé. À l'exercice
initial, le stencil est évalué directement sur le payoff à `t=0`, avec une
erreur standard nulle.

Sur SM89, le kernel de replay Heston compilé en 128 comme en 256 threads utilise
128 registres par thread, 32 octets locaux et 384 octets partagés statiques.
La mémoire partagée dynamique vaut respectivement 64 et 128 octets ;
l'occupation théorique est 33,3 % dans les deux cas. Ce diagnostic
ne qualifie pas encore les performances. Memcheck, racecheck, initcheck et
synccheck passent sur la matrice calls/puts réduite. Les prix et erreurs du
central sont bitwise identiques à `price_delta`. Sur la fixture publique, le
delta spot diffère d'au plus `5e-6` relatif et son erreur standard de `5e-5`
relatif entre les deux politiques compilées séparément ; cette borne remplace
une affirmation de parité bitwise non observée.


## Suite de la migration

Après stabilisation de ce périmètre : mesurer et mutualiser les scénarios
identiques ; ajouter les adaptateurs des autres dynamiques et produits ;
traiter les calendriers d'observation et les modèles rough ; qualifier les
profils américains suivant `Bp`, `K`, la taille des états et le GPU ; puis remplacer
les consommateurs `price_delta` et supprimer leurs adaptateurs devenus inutiles.
Chaque étape exige ses propres tests numériques et de non-régression.

Le test `price_gradients_central_work_cuda` instrumente une dynamique et un payoff
synthétiques composés avec le vrai moteur terminal : il compte les transitions
et évaluations centrales pour chaque catégorie, avec `B=1,2,4`, grilles complètes
et parcours par pas de grille. Les tests BS/Heston/CEV vérifient séparément la parité
numérique, les frontières et les erreurs standards.

## Études de précision et mesures CF

Les sondes opt-in `benchmark_price_gradients_closed_form`,
`study_price_gradients_bumps` et `study_price_gradients_heston_rates` vivent dans
`tests/{performance/price_gradients,price_gradients}/`. Elles conservent les
stencils représentés et séparent comparaison de formules FP64, arrondi FP32 et
erreur standard MC. La référence Heston réutilise le solveur Riccati/Fourier
indépendant existant avec un noyau constant ; elle ne simule pas QE-M.

Un bump de maturité d'un pas peut être une grande fraction de la maturité
résiduelle. Son admissibilité ne garantit pas une faible erreur de dérivée.
De même, des extrémités FP32 distinctes ne garantissent pas que les transitions
accumulées distinguent assez précisément un très petit bump de paramètre.
L'erreur standard ne couvre ni ce biais numérique ni celui du stencil. Aucun
rétrécissement ou élargissement automatique du bump n'est introduit.

Le [rapport de l'étape CEV/CF](../performance-reports/price-gradients-cev-cf-sm89-2026-09-15.md)
possède les mesures, leurs limites et les cas de petits bumps Heston restant à
traiter. Ces diagnostics ne constituent pas une certification de dataset.

L'[étude des bumps de taux](../performance-reports/price-gradients-rate-bumps-sm89-2026-09-16.md)
compare ensuite 1/2/5/10 pb sur `r`, en conservant les différences par
trajectoire FP32 et leurs moments FP64. L'exemple de sélection ci-dessus
utilise 5 pb par côté, valeur de départ configurable soutenue par cette étude
limitée. Aucun taux n'est sélectionné ni aucun bump n'est ajusté automatiquement.

La [politique de qualification versionnée](price-gradient-bump-qualification-policy.md)
industrialise la campagne commune BS/Heston/CEV. Elle pré-déclare le domaine,
les cas, références et seuils, conserve les échecs et interdit de déduire une
qualification de la seule stabilité entre plusieurs bumps. Son statut initial
reste candidat jusqu'à l'exécution complète et la revue des preuves.
