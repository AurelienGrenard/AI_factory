# Validation des lancements et diagnostic des kernels CUDA

## Objet

Ce document décrit les contrôles communs de `check_cuda.cuh`, l'instrumentation
optionnelle de `cuda_kernel_diagnostics.cuh/.cpp` et le test qui garantit son
fonctionnement. Ces services sont côté hôte : ils ne changent ni les équations,
ni les trajectoires, ni les réductions des kernels.

## Contrôles communs de `check_cuda.cuh`

Toutes ces fonctions sont `inline`, exécutées sur l'hôte et lèvent une exception
C++ descriptive en cas d'échec.

Les cibles hôte qui incluent `cuda_runtime.h` doivent recevoir les include
directories du même CUDA Toolkit que celui utilisé par NVCC. Mélanger par
exemple un header système CUDA 11 avec le runtime CUDA 13 change la taille de
`cudaDeviceProp`; un simple `cudaGetDeviceProperties` peut alors écraser la
pile avant même le lancement. `ai_factory_configure_host_library` propage donc
`CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES` en `SYSTEM PUBLIC`.

### `check_cuda`

Paramètres : un `cudaError_t` et le nom de l'opération. Transforme toute erreur
CUDA Runtime en `std::runtime_error` contenant l'opération et le message CUDA.
Elle doit entourer les allocations, copies, événements, requêtes de propriétés
et appels de synchronisation. Après un lancement, elle reçoit immédiatement le
résultat de `cudaGetLastError()`.

### `checked_workspace_product`

Paramètres : deux dimensions `size_t` et un message. Multiplie les dimensions
d'une allocation seulement si le produit tient dans `size_t`; sinon lève
`std::overflow_error`.

Les planners de workspace appliquent la meme regle aux sommes d'offsets et
aux divisions plafond. Une division plafond s'ecrit `value / divisor` plus le
reste non nul; elle ne forme jamais `value + divisor - 1`. Chaque nombre de
blocs derive d'une cardinalite est controle avant conversion vers le type de
`dim3`, puis contre la limite `gridDim.x` du device. Un rejet de cardinalite
doit donc preceder allocation, arithmetique de pointeur et lancement.

### `bounded_block_count`

Paramètres : nombre de résultats et nombre de blocs configuré. Exige deux
valeurs positives et retourne le minimum, afin qu'une grille persistante ne
lance pas plus de blocs qu'elle n'a de lignes disponibles.

### `validate_block_count`

Exige des nombres de résultats et de blocs positifs et interdit plus de blocs
que de lignes de résultat.

### `validate_device_pointer`

Paramètres : pointeur et nom logique. Refuse un pointeur nul ou un pointeur qui
n'est pas identifié par CUDA comme mémoire device. Cette vérification concerne
les tableaux directement passés au kernel.

### `validate_model_product_construction`

Valide les comptes d'une construction modèle-produit. En mode aligné, les trois
comptes doivent être égaux. En mode cartésien, `result_count` doit être le
produit exact, sans débordement de `size_t`.

### `validate_model_curve_product_construction`

Extension du contrôle précédent aux modèles, courbes et produits. Le mode
aligné impose quatre comptes égaux ; le mode cartésien contrôle les deux
multiplications puis leur résultat exact.

### `validate_monte_carlo_path_count`

Exige au moins deux chemins par résultat. Les pricers à incréments exacts
l'utilisent directement, puisqu'ils ne reçoivent aucun pas de temps
numérique.

### `validate_monte_carlo_parameters`

Exige au moins deux chemins par résultat et un `dt` de transition strictement
positif et fini. Elle appelle d'abord `validate_monte_carlo_path_count`, puis
ajoute la validation propre aux simulations discrétisées.

### Calendriers et nombres de transitions

Un calendrier utilisé par un schedule discret doit être validé sur l'hôte
avec sa `FixedStepTimeConfiguration` avant toute inspection CUDA. La fonction
`checked_fixed_step_transition_count` forme
`simulation_steps_per_day * day_count` en `uint64_t`, refuse un résultat qui
ne tient pas dans le `uint32_t` stocké par le schedule et vérifie que la
fraction d'année FP32 correspondante reste finie. Les surcharges
`validate_calendar(calendar, time_configuration)` appliquent ce contrôle à
chaque intervalle des calendriers terminal, régulier, régulier avec stub et
statique ; elles contrôlent aussi la maturité totale. Les calendriers
d'exercice appliquent le même contrat à leur stub initial et à leur intervalle.

Les launchers Monte Carlo reçoivent donc, en plus des tableaux device, un
miroir hôte des produits. Les policies construisent leurs calendriers avec la
même fonction `calendar(...)` côté hôte et côté device. Les sources de
calendriers du sampler valident de même la constante ou la borne maximale ;
une source de calendriers déjà matérialisés sur device doit fournir son miroir
hôte. Il est interdit de remplacer ce contrat par une copie device-vers-hôte
dans le launcher ou par une multiplication non vérifiée dans le kernel.

Les transitions exactes ne multiplient pas un nombre de pas, mais leur
calendrier est tout de même validé avec sa configuration afin de refuser une
fraction d'année non finie.

### `validate_cuda_block_size`

Interroge le device courant et exige une taille de bloc positive ne dépassant
pas `maxThreadsPerBlock`.

### `validate_reduction_block_size`

Applique `validate_cuda_block_size` puis exige un nombre entier de warps, donc
un multiple de 32 threads.

### `validate_grid_x_size`

Interroge le device courant et vérifie que le nombre de blocs tient dans la
limite `maxGridSize[0]`.

### `validate_row_seed_range`

Valide le mapping reproductible `base_seed + result_index` et interdit son
débordement au-delà de `uint64_t`.

## Données de diagnostic

`CudaKernelLaunchDiagnostics` décrit une spécialisation et une géométrie de
lancement précises.

### Device et code

| Champ | Signification |
|---|---|
| `device_index`, `device_name` | device CUDA courant |
| `compute_capability_major/minor` | compute capability matérielle |
| `binary_version` | architecture du binaire chargé |
| `ptx_version` | version PTX annoncée par CUDA |

### Géométrie

| Champ | Signification |
|---|---|
| `grid_block_count` | produit des trois dimensions de grille |
| `grid_x/y/z` | dimensions exactes de la grille |
| `threads_per_block` | produit des trois dimensions du bloc |
| `block_x/y/z` | dimensions exactes du bloc |

### Ressources

| Champ | Signification |
|---|---|
| `registers_per_thread` | registres alloués à chaque thread |
| `static_shared_bytes_per_block` | mémoire partagée statique du kernel |
| `dynamic_shared_bytes_per_block` | mémoire partagée fournie au lancement |
| `local_bytes_per_thread` | mémoire locale déclarée par CUDA par thread |
| `maximum_threads_per_block` | limite propre à cette fonction compilée |
| `maximum_dynamic_shared_bytes_per_block` | plafond dynamique de cette fonction |

### Occupation théorique

| Champ | Signification |
|---|---|
| `active_blocks_per_multiprocessor` | blocs simultanément résidents par SM |
| `active_warps_per_multiprocessor` | warps résidents déduits de la taille du bloc |
| `maximum_warps_per_multiprocessor` | capacité maximale du device |
| `theoretical_occupancy` | rapport warps actifs / warps maximaux |

L'occupation est une capacité théorique, pas un temps d'exécution ni une mesure
d'utilisation effective. Une occupation plus élevée n'implique pas
automatiquement un kernel plus rapide.

## Fonctions de diagnostic

### `cuda_kernel_diagnostics_enabled`

Lit une seule fois `AI_FACTORY_CUDA_KERNEL_DIAGNOSTICS`. Les valeurs `1`,
`true` et `on`, sans distinction de casse, activent les rapports. Toute autre
valeur laisse le chemin normal désactivé.

### `reserve_cuda_kernel_launch_diagnostics`

Construit une clé avec nom, variante, grille, bloc et mémoire partagée
dynamique. Elle retourne `true` seulement pour la première occurrence du
processus. Un mutex protège la déduplication concurrente.

### `inspect_cuda_kernel_launch`

Template hôte recevant le pointeur vers le kernel exact, sa grille, son bloc et
sa mémoire partagée dynamique. Il valide la géométrie, appelle
`cudaFuncGetAttributes`, calcule les blocs actifs avec
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, interroge le device puis
retourne un `CudaKernelLaunchDiagnostics` complet.

Cette fonction inspecte le kernel mais ne le lance pas.

### `emit_cuda_kernel_launch_diagnostics`

Sérialise une structure en un objet JSON ordonné et l'écrit sur `stderr`. Un
mutex protège chaque ligne contre l'entrelacement. Aucun rapport n'est écrit
dans les datasets ou sur `stdout`.

### `report_cuda_kernel_launch_if_enabled`

Template utilisé par tous les launchers. Il reçoit nom logique, variante,
pointeur de kernel, grille, bloc et mémoire partagée dynamique. Si le diagnostic
est désactivé ou déjà réservé, il retourne immédiatement. Sinon il enchaîne
inspection et émission.

Le pointeur doit désigner la spécialisation réellement lancée, et la géométrie
doit être exactement celle utilisée par l'expression `<<<...>>>` suivante.

## Ordre obligatoire dans un launcher

1. valider les entrées avec `validate_..._launch` ;
2. calculer la géométrie et la mémoire partagée ;
3. vérifier la résidence minimale lorsque l'algorithme l'exige ;
4. appeler `report_cuda_kernel_launch_if_enabled` ;
5. lancer le même kernel avec la même géométrie ;
6. appeler `check_cuda(cudaGetLastError(), ...)`.

Le launcher analytique scalaire commun applique cet ordre à deux spécialisations
du même template. Il inspecte et lance
`closed_form_price_kernel<PricingPolicy, false>`
si la grille couvre tout le batch ; sinon il inspecte et lance
`closed_form_price_kernel<PricingPolicy, true>`, dont la boucle est grid-stride.
Le pointeur transmis au diagnostic reste donc toujours celui de la
spécialisation effectivement exécutée.

Le launcher analytique coopératif suit le même ordre avec un bloc par prix et
une boucle block-stride. Il calcule la mémoire partagée dynamique à partir de la
capacité hôte, vérifie `maxDynamicSharedSizeBytes` puis exige au moins un bloc
résident avec `cudaOccupancyMaxActiveBlocksPerMultiprocessor`. Si l'une de ces
deux contraintes matérielles échoue, il ne lance rien et rend la main au
launcher produit, qui sélectionne explicitement son chemin scalaire de repli.

Les entrées modèle-produit et modèle-courbe-produit sont validées par leur
`DeviceInputs::validate(result_count)`. Un contexte supplémentaire, comme un
pool de schedule explicite, ajoute son propre `validate_device_context` avant
toute inspection ou exécution du kernel.

Le diagnostic n'ajoute ni synchronisation du kernel ni mesure de durée. Le
chronométrage reste la responsabilité des événements CUDA du générateur ou de
`longstaff_schwartz::LaunchResources`.

Le moteur Longstaff–Schwartz possède en parallèle un diagnostic numérique de
régression, distinct des ressources CUDA. Les statuts device sont agrégés dans
`LaunchResult::regression_diagnostics`, puis copiés une seule fois par batch
après la synchronisation de chronométrage. Les générateurs appellent
`validate_regression_diagnostics` avant d'écrire un dataset. L'absence ou
l'insuffisance de candidats reste observable mais non fatale ; statistiques
non finies, échec de Cholesky et coefficients non finis interdisent la
publication.

## Test `cuda_kernel_diagnostics`

Le test dédié utilise un kernel sonde et vérifie : disponibilité du device,
restitution exacte des géométries, occupation dans l'intervalle `(0, 1]`,
limites de threads, présence du nom du device et déduplication d'une géométrie
répétée. Il retourne le code CTest `77` lorsque CUDA n'est pas disponible.

Ce test garantit le mécanisme d'inspection. Il n'impose aucun seuil de
registres, de mémoire locale ou d'occupation aux pricers. Les tests CUDA de
pricing vérifient séparément les résultats numériques ; les comparaisons SASS,
ressources et performances restent des contrôles avant/après sur le même GPU et
la même toolchain.

## Lecture pratique

L'ordre de contrôle recommandé est : mémoire locale nouvelle, évolution des
registres, mémoire partagée totale, blocs actifs et occupation théorique, puis
temps GPU réellement mesuré. Toute comparaison doit conserver architecture,
toolchain, géométrie et paramètres identiques.
