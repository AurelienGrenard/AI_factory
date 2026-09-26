# Prix et gradients equity sélectionnés

## Périmètre actif

Le manifeste contient 261 bindings equity markoviens `price_gradients` :

- 55 bindings Monte Carlo terminaux, soit cinq produits
  (`european_option`, `asset_or_nothing_option`, `digital_option`,
  `gap_option`, `straddle`) sur onze modèles ;
- 189 bindings Monte Carlo de chemin, selon les couples déjà exposés par le
  catalogue de prix ;
- huit bindings Black--Scholes en formule fermée, dont cinq terminaux et trois
  produits de chemin analytiques ;
- neuf bindings américains Longstaff--Schwartz à exercice central gelé.

Tous publient les dérivées premières sélectionnées et, sur demande, la
diagonale de Hessienne. Les dérivées mixtes restent hors de ce contrat. Les
moteurs Monte Carlo proposent `mono` et `node_graph` avec les mêmes
`SensitivitySpec`, tâches, stencils, nœuds et sorties. La formule fermée garde
un thread par ligne ; elle ne passe pas par le graphe Monte Carlo.

Les paramètres supportés sont déclarés dans les propriétaires modèle et
produit. Leur domaine admissible est défini une seule fois dans le
`parameter_domain.hpp` correspondant ; loaders et préparation des nœuds
appellent le même prédicat. SABR reconstruit notamment sa préparation quand le
spot change, car son alpha dimensionnel en dépend. Les modèles à sauts et à
subordination emploient leurs domaines Philox et leurs événements couplés ;
leur intégration ne vaut pas qualification numérique automatique du bump.

Les calls et puts américains Black--Scholes, CEV, Heston, Schöbel--Zhu,
Merton, Kou, Bates, Variance-Gamma et NIG calculent une seule stratégie LSM
centrale, puis rejouent les nœuds sous cette stratégie gelée. Les coordonnées
continues déclarées par le modèle et le strike sont disponibles aux ordres un
et deux diagonal. La maturité, les dates et l’intervalle d’exercice restent
exclus. Les moteurs `price_delta` restent les références de migration jusqu’à
la couverture rough et au retrait de leurs derniers consommateurs.

Le catalogue equity contient 1 456 recettes permanentes issues des recettes
de prix existantes, avec ordre un ou ordre un plus Hessienne diagonale. Cette
couverture de génération ne certifie ni la taille des bumps, ni le biais de
discrétisation ou de politique gelée.

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

Tous les moteurs européens terminaux conservent seulement les modèles, les
produits et les `K` spécifications de sensibilité. L'hôte résout chaque nom une
fois vers un identifiant typé ; aucun nom n'est interprété dans une boucle
CUDA. Il ne matérialise plus `N*(1+2*K)` scénarios.

Pour un moteur MC, le bloc `(row, sensitivity)` charge le modèle et le
produit centraux, puis le thread 0 applique la `SensitivitySpec` de sa colonne.
Il construit dans la mémoire partagée le `SensitivityTask` de cette ligne :
les nœuds centraux/perturbés, le stencil réellement représenté et le besoin du
central. La mémoire d'entrée passe ainsi de `O(N*K)` scénarios à
`O(nombre de modèles + nombre de produits + K)`. Les `N*K` stencils de sortie
restent conservés parce qu'ils décrivent exactement les points utilisés par le
dataset.

Le launcher vérifie dimensions, capacités, plages, géométrie et absence de
recouvrement entre entrées et sorties. Les sorties gradients et erreurs
standards sont rangées par ligne puis coordonnée. La formule fermée affecte
une ligne à chaque thread et évalue ses sensibilités successivement à partir
du même plan compact.

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
Pour les dynamiques à pas fixe CEV, Heston et SABR, les mêmes innovations de
chaque pas alimentent aussi le stencil de maturité d'ordre deux diagonal ;
un stencil unilatéral peut activer quatre dates. L'écart d'un seul pas est une
capacité de calcul, pas une qualification du biais ou de la variance de la
dérivée temporelle seconde.

## CRN et responsabilité des dynamiques

Chaque ligne conserve la clé Philox `base_seed + global_row`. Le mapping V2
adresse le compteur par `(path_bas, path_haut, groupe, domaine)`. Le domaine
zéro porte le flux unique des modèles à consommation fixe ; Merton, Kou,
Bates, Variance-Gamma et les autres consommateurs variables isolent seulement
les sources qui pourraient déplacer un autre tirage. La sélection, l’indice
de sensibilité et le découpage des lignes ne changent jamais l’adresse d’une
innovation. Les `coupled_dynamics` adaptent les transitions canoniques ; elles
ne recopient ni équations du modèle ni préparation de ses coefficients.

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
Chaque `CoupledDynamics` déclare aussi
`kDrawRequiresCentralPrepared`. Il vaut `true` pour Merton : même lorsqu'un
bloc de sensibilité n'évalue pas le payoff central, le sampler de Poisson doit
lire la transition centrale afin de conserver le compte commun. Il vaut
`false` pour les dynamiques dont le tirage ne dépend d'aucun coefficient
préparé.
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
| Préparation compacte modèle/produit/spécification | `src/common/equity/price_gradients/device_preparation.cuh` |
| Composition d'un scénario terminal | `src/common/equity/price_gradients/terminal_device_preparation.cuh` |
| Kernels terminaux à préparation device | `src/common/monte_carlo/price_gradients/device_prepared_terminal_{kernel,diagonal_kernel}.cuh` |
| Plan compact et composition des launchers terminaux | `src/common/equity/price_gradients/terminal_device_prepared_{plan.hpp,launcher.cuh}` |
| Kernel MC opaque et kernels formule fermée | `src/common/{monte_carlo,closed_form}/price_gradients/` |
| Domaines admissibles des paramètres | `src/model/**/parameter_domain.hpp`, `src/product/**/parameter_domain.hpp` |
| Accès aux paramètres et adaptation des transitions | `src/model/equity/markovian/<model>/price_gradients/{device_preparation,coupled_dynamics_impl}.cuh` |
| Adaptateurs du produit européen | `src/product/european_option/price_gradients/` |
| Trace d'exercice gelée | `src/common/longstaff_schwartz/frozen_exercise_trace.cuh` |
| Replay couplé, workspace, planner et kernels LSM gradients | `src/common/longstaff_schwartz/price_gradients/` |
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

Les kernels communs MC reçoivent la dynamique, la policy produit et
l'adaptateur de préparation comme paramètres de composition. Ils ne
dépendent d'aucun modèle concret. Le launcher terminal assemble ces briques
au dernier niveau avant le kernel. De même, le planner hôte des gradients ne
connaît aucun nom de modèle, de produit ou de coordonnée : le plan préparé
valide la sélection, puis le planner calcule seulement le nombre de tâches
et le cap de grille.

Chaque modèle américain disposant de datasets source génère ses recettes
permanentes calls/puts, alignées/cartésiennes, à l'ordre un et à l'ordre deux
diagonal. Black--Scholes expose le binding mais ne publie pas de recette sans
dataset américain source. Le runner découpe une campagne en tranches durables ;
chaque appel est ensuite replanifié par le LSM selon la VRAM libre.
`result_offset` décale ensemble les lignes centrales, stencils, clés Philox et indices
de sortie. Un checkpoint n'est validé qu'après la copie des prix, erreurs et
colonnes gradients de toute la tranche. Une interruption reprend donc au
dernier multiple de 16 confirmé, ou à la dernière tranche plus courte.

Le replay de l'exercice gelé est composé à partir d'une dynamique couplée et de
l'une des deux politiques communes
`DevicePreparedFixedStepFrozenExerciseReplay` ou
`DevicePreparedExactTransitionFrozenExerciseReplay`. La première rejoue le
stub initial et les intervalles réguliers avec le nombre canonique de pas. La
seconde prépare une loi exacte pour le stub et une pour l'intervalle régulier,
puis consomme un incrément par date contractuelle. Chaque modèle fournit sous
`price_gradients/` son adaptation des transitions et reste propriétaire du
couplage de ses événements.

## Géométrie et qualification

### Européennes terminales préparées sur le device

Les 55 bindings Monte Carlo terminaux utilisent le même contrat compact :
`TerminalDevicePreparedPlan<ModelPreparation, ProductPreparation>`. Le plan
contient les tableaux centraux de modèles et produits, les `K`
`SensitivitySpec`, la construction alignée ou cartésienne et la configuration
temporelle. Il ne contient aucun scénario bumpé par ligne.

Le moteur terminal propose deux ordonnancements du même contrat compact. Le
moteur **mono** utilise une grille bidimensionnelle : `blockIdx.x` parcourt les
lignes et `blockIdx.y` désigne l'une des `K` sensibilités. La colonne `y=0`
calcule le prix central et sa sensibilité ; les autres colonnes ne calculent
que leur sensibilité. Le prix moyen et son erreur standard sont donc écrits
une seule fois par ligne. Chaque colonne réemploie la clé Philox
`base_seed + row` et rejoue les mêmes innovations. `B=1` est la géométrie de
cette famille ; `sensitivity_batch_size` doit valoir un.

Le thread 0 construit le `SensitivityTask` de son couple ligne/sensibilité et
le place dans la mémoire partagée. Le reste du bloc ne reçoit que les nœuds
préparés, le stencil et le besoin du central. Cette préparation appelle les
adaptateurs permanents du modèle et du produit ; elle ne contient aucune
branche fondée sur le nom du modèle.

Les briques permanentes ont les responsabilités suivantes :

| Type | Contenu |
|---|---|
| `SensitivitySpec` | paramètre résolu et politique de bump |
| `SensitivityRequest` | ordre un, ordre deux diagonal, ou les deux |
| `SensitivityStencil<C>` | points représentés et coefficients, sans logique de reconstruction |
| `SensitivityNodes<Node,C>` | jeux de paramètres centraux et perturbés propres à la ligne |
| `SensitivityTask<Node,C>` | nœuds, stencil et `CentralRequirement` |
| `SensitivityValues<C>` | payoffs pathwise aux nœuds |
| `SensitivityResult` | dérivées reconstruites |

Le moteur **graphe de nœuds** conserve exactement ces types, les mêmes
`DevicePreparedInputs`, le même `DevicePreparedPlan`, les mêmes stencils et
les mêmes sorties. Il ajoute seulement une configuration de chunks et un
workspace typé appartenant à l'appelant. Ses trois phases ont chacune une
responsabilité unique :

1. `node_evaluation` construit la ligne sur le device, évalue le central et
   les nœuds perturbés, puis écrit leurs observations terminales ;
2. `moment_accumulation` reconstruit prix et dérivées chemin par chemin et
   conserve leurs moments FP64 par thread à travers les chunks de chemins ;
3. `moment_finalization` réduit ces moments et écrit les sorties publiques.

Un groupe de threads traite un chemin. Sa lane productrice initialise Philox,
puis diffuse l'innovation aux lanes qui évaluent les nœuds ; cette lane peut
également évaluer des nœuds. Les modèles dont les innovations dépendent du
nœud emploient leur `draw_coupled`. Le central est le nœud zéro et n'est
jamais resimulé pour chaque sensibilité. Le chunking porte à la fois sur les
lignes et les chemins ; l'affectation d'un chemin à son thread de réduction
reste inchangée entre chunks, ce qui préserve l'ordre de sommation du moteur
mono.

L'implémentation commune vit dans
`src/common/monte_carlo/price_gradients/terminal_node_graph/` : `workspace.cuh`
décrit la mémoire bornée, `evaluation.cuh` évalue les nœuds,
`reconstruction.cuh` reconstruit et réduit, et `launcher.cuh` orchestre les
chunks. `terminal_sensitivity_policy.cuh` contient la préparation et la
contraction du payoff partagées avec le moteur mono. Une dynamique qui possède
une composante agrégée en fin d'intervalle expose le contrat optionnel
`kHasDistributedTerminalAggregation`. Bates l'emploie : les innovations QE-M
continues sont diffusées pas à pas, puis les domaines Philox Poisson et marques
construisent un ajustement terminal couplé. Les 13 coordonnées Bates, dont
intensité, marques et maturité, restent bitwise avec le moteur mono.

L'ordre un est spécialisé à capacité trois. Une Hessienne diagonale emploie
une capacité quatre : trois nœuds pour un stencil centré, quatre pour le repli
unilatéral d'ordre deux. Le nombre actif est stocké dans le stencil ; il ne
provoque aucune allocation dynamique. `reconstruct_sensitivity` transforme
les payoffs pathwise et le stencil en dérivées d'ordre un et/ou deux. Les
dérivées mixtes restent hors de cette pipeline et auront un kernel séparé.

Une sélection MC vide délègue au moteur prix canonique ; aucune paire bumpée
n'est alors créée. Black--Scholes européen suit sa formule fermée et ne fait
pas partie des deux ordonnancements MC. Les produits terminaux MC exposent la
maturité aux ordres un et deux diagonal lorsque l'adaptateur du modèle la
déclare. Les produits à exercice anticipé continuent de l'exclure.

Le contrat d'artefact d'ordre deux écrit `sensitivity.orders =
["first", "diagonal_second"]`, `outputs.diagonal_hessians` et, en Monte
Carlo, `outputs.diagonal_hessian_standard_errors`. Le stencil conserve
`node_count`, le quatrième point unilatéral éventuel et tous les poids de
reconstruction d'ordre deux. Ces canaux font partie du checkpoint : une
reprise ne peut valider un préfixe qui ne contiendrait que le prix ou le
gradient.

La formule fermée Black--Scholes emploie la même préparation compacte et un
thread par ligne. Ce thread construit puis évalue successivement les nœuds des
`K` sensibilités. Il n'y a ni état stochastique ni flux Philox. Le nombre de
threads par bloc reste un paramètre de lancement, avec 256 par défaut.

Les variantes bornées et non bornées des kernels terminaux sont toutes deux
compilables. Le moteur mono n'emploie `__launch_bounds__` que lorsque le nombre
de threads demandé à l'exécution correspond au profil compilé ; toute autre
géométrie passe par son symbole non borné. Le graphe fixe la géométrie de
`node_evaluation` dans son type `Tuning` et choisit sa variante bornée seulement
si ce profil l'active. Les paramètres CMake
`AI_FACTORY_CUDA_PRICE_GRADIENT_{MONO,NODE}_THREADS_PER_BLOCK` et
`AI_FACTORY_CUDA_PRICE_GRADIENT_{MONO,NODE}_MIN_BLOCKS_PER_SM`, ainsi que
`AI_FACTORY_CUDA_PRICE_GRADIENT_LAUNCH_BOUNDS`, décrivent ce profil. Ils sont
publiés dans les métadonnées de lancement. Le paramètre template `Tuning`
permet une spécialisation ponctuelle par couple modèle/produit sans dupliquer
le moteur. Ces valeurs sont un profil de build mesuré, pas une hypothèse que
tous les couples ou GPU partagent le même optimum.

Le manifeste déclare la stratégie de préparation de chaque binding :
`device_prepared_closed_form_terminal`, `device_prepared_step_terminal` ou
`device_prepared_exact_terminal`. Le renderer sélectionne son template à
partir de ce champ et échoue pour toute stratégie inconnue. Ajouter un modèle
terminal demande donc son `device_preparation.cuh`, sa dynamique couplée, puis
une entrée explicite dans ce manifeste ; les templates et launchers communs ne
changent pas.

Chaque binding MC terminal d'ordre deux expose les deux launchers publics et
la fonction qui dimensionne le workspace du graphe. Les générateurs acceptent
`--sensitivity-strategy mono|node_graph`, avec `mono` par défaut. La recette
énumère les deux choix ; le reçu enregistre `sensitivity_strategy` et
`sensitivity_workspace_bytes`. Le runner alloue le workspace une seule fois à
la taille du batch maximal et le réemploie pour le warmup, les batches et la
reprise.

La matrice `terminal_binding_matrix_cuda_test.cpp` compile et exécute les 55
couples MC terminaux par les deux launchers, puis compare bit à bit prix,
erreurs standard, gradients, Hessiennes diagonales et stencils. Les tests
spécialisés gardent les sélections complètes et les frontières de CEV, Heston,
Merton et Bates. Le test de dataset exécute les deux stratégies de bout en bout
et exige des lignes JSON identiques.

La reprise de génération ne rejoue pas les chemins déjà confirmés. Le runner
appelle le callback de préparation de stencils du binding pour reconstruire
seulement le préfixe déjà validé, puis reprend la simulation à la première
ligne absente du checkpoint. Les prix, erreurs et gradients du préfixe sont
restaurés depuis les canaux durables.

Le benchmark ciblé `tests/performance/price_gradients/terminal.cu` conserve
les sorties binaires, médiane, p95 et CV pour Black--Scholes, Heston ordre un,
Heston ordre un plus Hessienne diagonale et Merton. Il utilise cinq
préchauffages et 21 répétitions. Une qualification doit suivre le
[protocole de performance](../performance-regression-protocol.md), avec le
même binaire, le même GPU, les mêmes données et les mêmes opérations regroupées.

```bash
cmake --build build --target price_gradients_performance_benchmarks -j2
python3 tools/performance/run_price_gradient_strategies.py \
  --build-dir build \
  --output-dir artifacts/audit/price-gradients-strategies-sm89-<run> \
  --profile
```

La campagne Heston de migration a mesuré le temps mur complet, incluant
préparation hôte, allocations, transferts, kernels et copies de sortie. Sur
100 000 lignes, deux chemins et sept gradients, la préparation device termine
en 955,621 ms contre 1 052,274 ms pour la matérialisation hôte, soit `-9,19 %`,
avec sorties bitwise identiques. Ces mesures qualifient cette charge Heston sur
le SM89 utilisé ; elles ne prédisent pas le résultat d'un autre modèle, GPU,
nombre de chemins ou nombre de sensibilités.

Le manifeste permanent couvre aussi les neuf phases Heston American et les
deux stratégies Jamshidian CIR. Le
[rapport SM89 du 19 septembre 2026](../performance-reports/price-gradients-strategies-sm89-2026-09-19.md)
conserve leurs temps agrégés, ressources, accès mémoire, branches et limites.

### Pipeline américaine gelée

Le moteur américain conserve la pipeline centrale du prix : préparation,
simulation forward, régressions backward, décisions d’exercice, moments et
finalisation. Elle écrit le prix et son erreur une seule fois, ainsi qu’une
trace compacte de l’exercice central par chemin.

Deux ordonnancements évaluent ensuite les sensibilités sous cette politique
gelée :

1. `mono` lance `frozen_sensitivity_moments` sur `(path shard, row*sensitivity)`
   puis `finalize_frozen_sensitivities`. Un bloc possède une sensibilité d’une
   ligne et rejoue successivement ses trois ou quatre nœuds ;
2. `node_graph` lance `frozen_node_evaluation`,
   `frozen_node_moment_accumulation` et
   `finalize_frozen_node_sensitivities`. Le central est un nœud ordinaire du
   graphe, évalué une seule fois, et les observations communes ne sont pas
   recalculées par sensibilité.

Les deux voies utilisent la même préparation compacte, la même clé Philox par
`(row,path)`, les mêmes stencils et les mêmes sorties. Le graphe conserve une
réduction complète par ligne lorsque son workspace peut contenir tous les
chemins ; il refuse explicitement une configuration qui changerait l’ordre de
réduction au lieu de perdre silencieusement la parité. Le planner central
continue de choisir le batch de lignes d’après la VRAM disponible, en comptant
le workspace des nœuds, des moments et des sorties.

Les transitions exactes gardent la granularité du calendrier LSM ; les modèles
à pas fixe rejouent leurs intervalles canoniques. Un bump de taux modifie aussi
l’actualisation du payoff gelé. À l’exercice initial, le stencil est évalué
directement à `t=0`, avec erreur standard nulle.

Les tests permanents couvrent les familles exactes, à pas fixe et à sauts, et
comparent les sorties publiques des deux ordonnancements sur leur périmètre
commun. `memcheck`, `racecheck`, `initcheck` et `synccheck` passent sur les
représentants américain Bates et bermudéen. Ces preuves ferment l’intégration
et la sûreté mémoire ; elles ne qualifient pas le biais de la politique gelée
ni la performance de production sur tout matériel.

## Suite de la migration

Le [plan de migration prix-gradients](price-gradients-migration-plan.md)
ordonne l'extension à tout le markovien equity et fixed income, les
sensibilités de sauts et de temps admissibles, les modèles rough N-facteurs
et FFT, puis le retrait des consommateurs `price_delta`. Le présent contrat
décrit uniquement les capacités et invariants déjà actifs.

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
