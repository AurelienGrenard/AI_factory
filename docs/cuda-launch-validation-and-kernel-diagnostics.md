# Validation des lancements et diagnostic des kernels CUDA

## Objet

Ce document décrit les contrôles communs de `check_cuda.cuh`, l'instrumentation
optionnelle de `cuda_kernel_diagnostics.cuh/.cpp` et le test qui garantit son
fonctionnement. Ces services sont côté hôte : ils ne changent ni les équations,
ni les trajectoires, ni les réductions des kernels.

## Contrôles communs de `check_cuda.cuh`

Toutes ces fonctions sont `inline`, exécutées sur l'hôte et lèvent une exception
C++ descriptive en cas d'échec.

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

Le diagnostic n'ajoute ni synchronisation du kernel ni mesure de durée. Le
chronométrage reste la responsabilité des événements CUDA du générateur ou de
`longstaff_schwartz::LaunchResources`.

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
