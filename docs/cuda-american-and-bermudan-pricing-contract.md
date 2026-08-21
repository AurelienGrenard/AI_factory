# Contrat des pricers CUDA américains et bermudéens

## Objet

Ce document fixe l'ossature Longstaff–Schwartz commune aux produits à exercice
anticipé, indépendamment du modèle, de la courbe et du payoff. Chaque nouveau
fichier `src/model/<asset_class>/<model>/[<curve>/]<product>.cu` conserve les mêmes étapes et
les mêmes noms de fonctions. Les structures préparées et les champs d'état
restent spécifiques au modèle-produit.

L'algorithme utilise une grille temporelle régulière entre dates d'exercice,
avec un stub initial distinct, un workspace SoA contigu et des batchs adaptés à
la mémoire GPU disponible.

## Fonctions et types du fichier modèle-produit

### `product_name<Side>`

Attribut : fonction `constexpr` hôte templatisée lorsque plusieurs côtés sont
partagés. Elle fournit un nom stable destiné aux erreurs, allocations et
métriques. Elle ne participe pas au calcul numérique.

### `immediate_payoff<Side>`

Attributs : `__device__ __forceinline__` et template éventuel. Elle retourne la
valeur d'exercice immédiat à partir des états utiles et des paramètres du
produit. La spécialisation doit être résolue à la compilation.

### `PreparedRow`

Structure privée contenant tous les éléments invariants d'un prix pendant un
batch : un modèle préparé pour le pas élémentaire, clé aléatoire,
indices, offsets SoA, constantes produit, actualisations, nombre de dates et
nombres de pas.

Elle ne contient ni tableau dynamique ni pointeur propriétaire. Sa disposition
est déclarée au planner par `workspace_descriptor()`.

### `StateRegions`

Vue nommée des `WorkspaceRegion` propres au modèle. Elle traduit les positions
génériques de `WorkspaceLayout::state_fields` en noms lisibles tels que spot,
variance ou facteur de taux. Elle ne possède pas la mémoire.

Un modèle ne stocke que les états nécessaires au payoff et à la régression.

### `workspace_descriptor`

Fonction hôte privée qui retourne un `WorkspaceDescriptor` décrivant : taille
et alignement de `PreparedRow`, liste des champs d'état SoA, taille de la base
de régression et nombre de statistiques par équation normale.

### `state_regions`

Fonction hôte privée qui valide le nombre de champs d'état du layout générique
et construit la vue `StateRegions`. Une incohérence est une erreur de
logique, pas une condition récupérable du pricing.

### `make_row_plans`

Fonction hôte privée qui construit un `EarlyExerciseRowPlan` par résultat. Elle
mappe chaque résultat vers son produit, calcule le nombre de dates d'exercice et
le nombre total de valeurs d'état à conserver. Elle contrôle tous les produits
et débordements avant l'allocation GPU.

### `prepare_rows_kernel`

Attribut : `__global__`.

Un thread prépare une ligne : mapping modèle–courbe–produit, transition
élémentaire commune au stub et aux intervalles réguliers, seed, offsets,
constantes et actualisations.
Ce kernel ne simule aucun chemin.

### `simulate_paths_kernel<Side>`

Attributs : `__global__` et template de payoff éventuel.

La grille 2D affecte `blockIdx.y` au prix du batch et `blockIdx.x` à un bloc de
chemins. Le kernel simule chaque chemin sur la grille régulière, écrit les états
observés aux dates d'exercice dans les champs SoA et initialise le cashflow à
l'échéance.

### `regression_partials_kernel<Side>`

Attributs : `__global__` et template de payoff éventuel.

Pour un niveau backward donné, il sélectionne les chemins dans la monnaie,
évalue la base, accumule les équations normales en FP64, réduit les threads du
bloc et écrit un partial déterministe par bloc de chemins et par prix.

### `solve_regressions_kernel`

Attribut : `__global__`.

Wrapper volontairement mince : un bloc par prix appelle
`solve_regression_for_row`. Il réduit les partials, régularise la matrice de
Gram, résout le système et écrit les coefficients ainsi qu'un indicateur de
validité. Il ne dépend normalement ni du modèle ni du payoff.

### `update_cashflows_kernel<Side>`

Attributs : `__global__` et template de payoff éventuel.

Il actualise les cashflows d'un intervalle, reconstruit la continuation depuis
les coefficients et remplace le cashflow lorsque l'exercice immédiat est plus
favorable. Une régression invalide conserve la continuation actualisée.

### `moment_partials_kernel`

Attribut : `__global__`.

Il applique l'actualisation du stub initial, accumule somme et somme des carrés
en FP64, puis écrit un couple de partials par bloc de chemins et par prix. Il est
indépendant du modèle et du payoff lorsque le cashflow porte déjà toute leur
logique.

### `finalize_prices_kernel<Side>`

Attributs : `__global__` et template de payoff éventuel.

Un bloc par prix réduit les moments, calcule prix et erreur standard, compare la
continuation à l'exercice déterministe en temps zéro, puis écrit les sorties à
leur `result_index` global.

### `validate_<model>_<curve>_<product>_launch`

Fonction C++ hôte privée. Elle valide les pointeurs host/device, la construction
alignée ou cartésienne, les paramètres Monte Carlo, la taille de réduction, la
plage de seeds, `blocks_per_price` et les limites de grille du device. Aucun
planning ni kernel ne doit commencer avant son succès.

### `launch_<model>_<curve>_<product>_cuda<Side>`

Fonction C++ hôte publique déclarée dans le `.cuh`. Elle retourne un
`longstaff_schwartz::LaunchResult` et orchestre, dans cet ordre :

1. validation ;
2. calcul du nombre effectif de blocs par prix ;
3. budget mémoire, plans de lignes et batchs ;
4. allocation unique du workspace ;
5. construction du layout et des offsets de chaque batch ;
6. préparation des lignes et simulation ;
7. boucle backward `regression_partials`, `solve_regressions`,
   `update_cashflows` ;
8. moments et finalisation ;
9. chronométrage et agrégation des métriques.

Chaque lancement interne appelle
`report_cuda_kernel_launch_if_enabled` puis vérifie immédiatement
`cudaGetLastError()`. Les spécialisations publiques call/put sont instanciées
explicitement au bas du `.cu`.

Paramètres publics communs :

| Paramètre | Utilité |
|---|---|
| tableaux device modèle, courbe éventuelle et produit | données lues par les kernels |
| comptes modèle, courbe éventuelle et produit | dimensions de la construction |
| `host_products` | calendrier et stockage nécessaires au planning côté hôte |
| `cartesian_product`, `result_count` | mapping aligné ou cartésien des résultats |
| `monte_carlo_paths_per_price` | nombre de chemins et dimension des cashflows |
| `dt` | durée fixe de la transition élémentaire du stub et des intervalles réguliers |
| `threads_per_block` | taille commune des blocs CUDA |
| `blocks_per_price` | parallélisme demandé pour les chemins d'un prix |
| `base_seed` | origine de la clé `make_key(base_seed + result_index)` |
| `device_prices`, `device_standard_errors` | sorties FP32 sur le device |

## API commune `longstaff_schwartz`

### Calendrier

- `maturity_anchored_exercise_count` calcule le nombre de dates régulièrement
  espacées et ancrées à l'échéance, avec contrôles de validité et de capacité
  `uint32_t`.

### Workspace et batch planning

- `WorkspaceRegion` décrit l'offset en octets et le nombre de valeurs d'une
  région du buffer contigu.
- `StateFieldDescriptor` décrit taille et alignement d'un champ d'état SoA.
- `WorkspaceDescriptor` décrit les dimensions propres au pricer et à sa base.
- `WorkspaceLayout` contient toutes les régions communes et les champs d'état.
- `EarlyExerciseRowPlan` contient dates d'exercice et stockage d'état d'une
  ligne.
- `BatchPlan` décrit une plage consécutive de résultats tenant en mémoire.
- `ExecutionPlan` contient tous les batchs et les maxima d'allocation.
- `make_workspace_layout` aligne et dimensionne chaque région avec contrôle des
  débordements.
- `plan_batches` construit les batchs consécutifs respectant le budget mémoire.
- `workspace_pointer<Value>` retourne une vue typée non propriétaire sur une
  région.

### Base et régression

- `longstaff_schwartz/laguerre.cuh` contient toute la base inline ; il n'existe
  pas de paire `.cuh/.cu` distincte pour ces fonctions élémentaires.
- `laguerre_0`, `laguerre_1`, `laguerre_2` évaluent les polynômes élémentaires.
- `TwoFactorLaguerreBasis::evaluate` construit les valeurs de la base courante.
- `TwoFactorLaguerreBasis::kSize` fixe le nombre de coefficients ;
  `kGramValueCount` fixe le triangle supérieur de Gram ;
  `kRegressionValueCount` ajoute le second membre et le compteur de chemins ;
  `values` contient les valeurs FP32 de la base.
- `RegressionBasis` est l'alias de la base utilisée par les kernels.
- `accumulate_normal_equations` ajoute une observation aux statistiques FP64.
- `reduce_and_store_regression_partials` réduit un bloc et écrit ses partials.
- `solve_regression_for_row` agrège les blocs, ajoute le ridge, résout et écrit
  l'état de validité.
- `regression_shared_bytes` retourne la mémoire partagée dynamique requise par
  les réductions de régression.
- `cholesky_solve_normal_equations` effectue la factorisation et les deux
  substitutions ; il retourne `false` si le système n'est pas exploitable.

### Ressources et métriques de lancement

- `WorkspaceBudget` expose mémoire libre, totale, marge de sécurité et budget
  utilisable.
- `query_workspace_budget` interroge le device et réserve une marge de sécurité.
- `LaunchResources::LaunchResources` alloue le workspace et crée les événements
  CUDA ; `LaunchResources::~LaunchResources` les libère. La copie et
  l'affectation sont interdites.
- `LaunchResources::workspace` retourne le buffer non typé.
- `LaunchResources::start_batch` démarre la mesure du batch.
- `LaunchResources::finish_batch` synchronise l'événement final et retourne les
  secondes GPU du batch.
- `LaunchResult` retourne temps kernel, nombre de batchs et de kernels, taille
  maximale d'un batch, blocs effectifs par prix et octets du workspace.

## Invariants d'extension

- Conserver les sept kernels et leur ordre, sauf justification algorithmique
  documentée.
- Conserver le workspace SoA ; ne pas convertir les chemins en AoS.
- Stocker uniquement les états utiles au payoff et à la régression.
- Conserver le stub initial et la grille régulière entre exercices.
- Conserver le mapping déterministe des seeds et l'ordre des réductions.
- Faire commencer chaque chemin à `local_group_index = 0` dans son propre
  sous-espace Philox `(path_index, local_group_index)`.
- Ne pas généraliser les kernels par templates modèle-produit tant que ce
  chantier n'est pas décidé séparément ; reproduire le squelette lisible dans
  chaque pricer.
