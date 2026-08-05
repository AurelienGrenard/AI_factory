# API des pricers CUDA : Monte Carlo et formule fermée

## Objet

Ce document fixe l'ossature des pricers sans exercice anticipé placés dans
`src/model/<model>/[<curve>/]<product>.cuh/.cu`. Un nouveau pricer doit reprendre
les mêmes couches, les mêmes responsabilités et la même convention de nommage.

Les implémentations Monte Carlo et en formule fermée partagent l'architecture
suivante :

1. préparer une ligne de calcul compacte ;
2. évaluer cette ligne ;
3. distribuer les lignes dans un kernel CUDA ;
4. valider le lancement côté hôte ;
5. lancer la spécialisation CUDA publique.

La différence volontaire porte sur l'évaluation et la stratégie du kernel :
Monte Carlo utilise `evaluate_path` et une réduction par prix ; une formule
fermée utilise `evaluate_price` et généralement un thread par prix.

## Conventions de nommage

- `<model>` désigne le modèle equity ou de taux.
- `<curve>` est présent uniquement lorsque le modèle reçoit une courbe externe.
- `<product>` désigne la famille de produit commune aux variantes call/put.
- Le préfixe complet est `<model>_<curve>_<product>` lorsqu'une courbe est
  nécessaire, sinon `<model>_<product>`.
- Le côté du payoff est un paramètre de template `OptionSide Side`. Il ne doit
  pas être stocké dans les lignes et ne doit pas produire de branche runtime
  dans le chemin chaud.

## Attributs CUDA utilisés

| Attribut | Domaine | Contrat |
|---|---|---|
| fonction C++ sans attribut CUDA | hôte | validation, planning et lancement |
| `__device__ __forceinline__` | device | primitive appelée dans un kernel et destinée à être inlinée |
| `__global__` | device, appelée par l'hôte | point d'entrée d'un kernel CUDA |
| `template<OptionSide Side>` | compilation | spécialisation call/put sans branche runtime |
| `__restrict__` | paramètres de kernel | absence d'alias entre les tableaux concernés |

## Types et fonctions obligatoires

### `PreparedRow`

`PreparedRow` est une structure privée au `.cu`. Elle contient uniquement les
coefficients de modèle, constantes de payoff, informations de simulation et
index nécessaires à l'évaluation répétée d'un prix.

Pour un pricer Monte Carlo persistant, une instance est généralement partagée
par les threads du bloc. Pour une formule fermée, elle peut rester locale au
thread. Le type ne fait pas partie de l'API publique et peut être spécifique au
couple modèle-produit.

### `prepare_row`

Attribut : `__device__ __forceinline__`.

Responsabilités :

- charger une ligne de modèle, de courbe éventuelle et de produit ;
- pré-calculer les coefficients invariants pendant l'évaluation ;
- préparer la clé aléatoire et la discrétisation en Monte Carlo ;
- retourner un `PreparedRow` sans allocation dynamique.

Cette fonction ne doit ni écrire les résultats ni effectuer de synchronisation.

### `evaluate_path<Side>`

Attributs : `template<OptionSide Side>` et `__device__ __forceinline__`.

Fonction obligatoire pour un pricer Monte Carlo. Elle simule exactement un
chemin à partir d'un `PreparedRow`, évalue le payoff spécialisé et retourne la
valeur actualisée de ce chemin. Elle ne doit pas écrire d'état de chemin en
mémoire globale sauf lorsque le produit exige explicitement une architecture
multi-kernel.

Pour un produit sans côté call/put, la fonction n'est pas templatisée.

### `evaluate_price<Side>`

Attributs : `template<OptionSide Side>` et `__device__ __forceinline__`.

Fonction obligatoire pour un pricer en formule fermée. Elle évalue un
`PreparedRow` et retourne son prix. Elle ne lance aucun kernel, n'alloue aucune
mémoire et ne modifie pas les tableaux d'entrée.

### `<model>_<curve>_<product>_kernel<Side>`

Attributs : `template<OptionSide Side>` et `__global__`. Le segment `<curve>`
est omis lorsqu'il n'existe pas.

Responsabilités communes :

- convertir un `result_index` en indices modèle, courbe et produit ;
- respecter la construction alignée ou cartésienne ;
- appeler `prepare_row` puis la primitive d'évaluation ;
- écrire chaque résultat exactement une fois.

Stratégie Monte Carlo : grille persistante bornée, un bloc responsable d'un
prix à la fois, distribution des chemins entre les threads, accumulation FP64
de la somme et de la somme des carrés, puis écriture FP32 du prix et de son
erreur standard.

Stratégie formule fermée : un thread responsable d'un prix, sans réduction de
moments ni erreur standard Monte Carlo.

### `validate_<model>_<curve>_<product>_launch`

Attribut : fonction C++ hôte privée au `.cu`.

Cette fonction centralise toutes les préconditions avant le lancement :

- pointeurs device valides ;
- nombres de modèles, courbes, produits et résultats cohérents ;
- construction alignée ou cartésienne valide ;
- intervalle de batch inclus dans le tableau de résultats ;
- nombre de chemins et `target_dt` valides en Monte Carlo ;
- taille de bloc compatible avec le device et les réductions ;
- dimensions de grille compatibles avec le device ;
- absence de débordement de `size_t` et de la plage de seeds.

La fonction lève une exception C++ descriptive à la première violation. Elle
ne modifie aucune entrée et ne lance aucun kernel.

### `launch_<model>_<curve>_<product>_cuda<Side>`

Attribut : fonction C++ hôte publique déclarée dans le `.cuh`. Le segment
`<curve>` est omis lorsqu'il n'existe pas.

Responsabilités :

- appeler exclusivement la fonction `validate_..._launch` correspondante ;
- calculer la mémoire partagée dynamique et la géométrie finale ;
- vérifier qu'au moins un bloc peut résider sur un SM lorsque le kernel utilise
  une réduction ou une grille persistante ;
- appeler `report_cuda_kernel_launch_if_enabled` avec la spécialisation et la
  géométrie exactes ;
- lancer le kernel ;
- contrôler immédiatement `cudaGetLastError()` avec `check_cuda`.

Le launcher n'effectue pas de `cudaDeviceSynchronize`. La synchronisation et le
chronométrage appartiennent au générateur ou à l'orchestrateur de plus haut
niveau.

Les paramètres publics suivent cet ordre logique : tableaux et comptes modèle,
courbe éventuelle et produit ; mode de construction ; plage de résultats ;
paramètres numériques et géométrie ; seed Monte Carlo éventuelle ; tableaux de
sortie.

### Instanciations explicites

Le bas du `.cu` doit instancier les versions publiques
`OptionSide::call` et `OptionSide::put`. Les générateurs C++ peuvent ainsi lier
directement `launch_..._cuda<OptionSide::call/put>` sans inclure
l'implémentation CUDA et sans wrapper de dispatch runtime.

Un produit sans côté n'ajoute ni template artificiel ni instanciation double.

## Paramètres communs des launchers

| Paramètre | Utilité |
|---|---|
| `device_models`, `model_count` | lignes de modèle présentes sur le device |
| `device_curves`, `curve_count` | courbes présentes sur le device, si requises |
| `device_products`, `product_count` | lignes de produit présentes sur le device |
| `cartesian_product` | sélection entre construction alignée et produit cartésien |
| `result_count` | taille totale des tableaux de résultats |
| `result_offset`, `launch_result_count` | sous-plage traitée par un batch Monte Carlo |
| `monte_carlo_paths_per_price` | chemins indépendants par prix |
| `target_dt` | pas cible utilisé pour construire la grille régulière |
| `threads_per_block` | nombre de threads CUDA par bloc |
| `block_count` | nombre de blocs de la grille persistante ou analytique |
| `base_seed` | origine du mapping déterministe `seed = base_seed + result_index` |
| `device_prices` | prix FP32 écrits sur le device |
| `device_standard_errors` | erreurs standards FP32, uniquement en Monte Carlo |

## Invariants d'implémentation

- Conserver l'ordre des lignes, le mapping des seeds et l'ordre des réductions.
- Ne jamais activer `--use_fast_math`.
- Conserver l'accumulation FP64 des moments Monte Carlo.
- Ne pas introduire de dispatch runtime call/put dans le kernel.
- Ne pas déplacer `PreparedRow` vers une représentation AoS globale des chemins.
- Conserver des fonctions courtes, privées au `.cu`, et un seul launcher public.
- Mettre à jour ce document si le contrat commun évolue.
