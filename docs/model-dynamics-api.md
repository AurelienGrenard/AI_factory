# API des dynamiques de modèles CUDA

## Objet

Ce document fixe l'interface et les responsabilités des dynamiques placées dans
`src/model/<asset_class>/<model>/dynamics.cuh/.cu`. Une nouvelle dynamique doit conserver les
mêmes couches, les mêmes noms et le même ordre de fonctions lorsque sa
mathématique les rend applicables.

Le fichier `.cuh` déclare les types et fonctions device réutilisables. Le
fichier `.cu` adjacent contient leurs définitions force-inlinées et est inclus
par les kernels consommateurs ; il n'est pas compilé comme une unité CMake
indépendante.

## Attributs et dépendances

Les fonctions de dynamique sont `__device__ __forceinline__`. Elles ne lancent
aucun kernel, n'allouent aucune mémoire et ne dépendent d'aucun paramètre de
produit. Les données brutes proviennent de `dataset.hpp`; les primitives
aléatoires proviennent de `common/philox.cuh`.

Les paramètres d'un produit, son payoff, ses barrières et ses règles d'arrêt
restent dans le pricer. Une dynamique expose seulement le processus et les
résumés de chemin réutilisables indépendamment d'un produit.

## Types communs

### Paramètres préparés

Un type tel que `HestonQeParameters`, `BatesQeParameters` ou
`OrnsteinUhlenbeckExactTransition` contient les coefficients invariants d'une
transition. Il est construit une fois par ligne et par intervalle temporel,
puis partagé par les chemins concernés.

Il ne contient ni paramètre produit, ni pointeur propriétaire, ni allocation
dynamique.

### État de chemin

Un type `<Model>State` contient uniquement les variables mutables d'un chemin.
Il reste dans les registres lorsque cela est possible. Un modèle dérivé peut
réutiliser l'état de son modèle de base lorsque cela reflète sa construction
mathématique, comme Bates réutilise l'état log-spot/variance de Heston.

### Résultats de chemin

Un type spécialisé tel que `<Model>MeanPathResult`,
`<Model>TwoTimePathResult` ou `<Model>MaximumPathResult` regroupe l'état terminal
et un résumé de chemin. Il n'est ajouté que si plusieurs produits peuvent le
réutiliser sans introduire de logique produit dans la dynamique.

## Fonctions fondamentales

Les déclarations publiques apparaissent dans cet ordre dans `dynamics.cuh`.

### `prepare_model`

Attribut : `__device__ __forceinline__`.

Pour un schéma discrétisé dont les coefficients dépendent du nombre de pas :

```cpp
PreparedParameters prepare_model(
    const ModelParameters& parameters,
    float maturity,
    std::size_t num_steps
);
```

Pour une transition exacte sur un intervalle fixé :

```cpp
ExactTransition prepare_model(
    const ProcessParameters& parameters,
    float time_interval
);
```

La fonction pré-calcule tous les coefficients réutilisés par les transitions.
Elle ne génère aucun aléa et ne construit aucun état de chemin. Les valeurs de
temps et le nombre de pas sont validés par le launcher avant son appel.

### `initial_state`

Attribut : `__device__ __forceinline__`.

```cpp
State initial_state(const PreparedParameters& model);
```

La fonction construit l'état en temps zéro lorsque celui-ci est porté par les
paramètres du modèle. Elle est omise lorsque l'état initial est naturellement
un argument explicite des fonctions de simulation.

### `one_step_transition`

Attribut : `__device__ __forceinline__`.

```cpp
void one_step_transition(
    const PreparedParameters& model,
    /* variates explicites propres au schéma */,
    State& state
);
```

Cette fonction est la transformation numérique déterministe d'un état. Tous
les uniformes, normales, comptes de Poisson ou autres variates apparaissent
explicitement dans sa signature. Elle ne connaît ni Philox, ni la seed, ni
l'indice du chemin.

### `simulate_one_step`

Attribut : `__device__ __forceinline__`, fonction privée au `.cu`.

```cpp
void simulate_one_step(
    const PreparedParameters& model,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,  // si nécessaire
    State& state
);
```

Cette fonction est utilisée lorsqu'une transition doit transformer des
uniformes, effectuer des tirages conditionnels ou partager la même logique
aléatoire entre plusieurs simulateurs de chemins. Elle consomme les uniformes
dans un ordre explicite, construit les variates nécessaires, puis appelle
`one_step_transition`.

Une transition exacte qui reçoit déjà ses normales peut appeler directement
`one_step_transition` sans ajouter un wrapper artificiel.

### `simulate_terminal_state`

Attribut : `__device__ __forceinline__`.

Pour une simulation complète qui possède sa suite aléatoire :

```cpp
State simulate_terminal_state(
    const PreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
);
```

Pour une transition exacte alimentée par des variates externes :

```cpp
State simulate_terminal_state(
    const ExactTransition& model,
    State initial_state,
    /* normales explicites */
);
```

La fonction retourne l'état terminal sans écriture en mémoire globale. La
version multi-pas construit exactement une `UniformSequence` pour tout le
chemin. Un processus de Lévy à incréments indépendants peut sommer exactement
la loi des sous-pas et ne tirer qu'un incrément terminal ; un schéma réellement
pathwise transmet la suite à chaque transition.

### `simulate_on_regular_grid`

Attribut : `__device__ __forceinline__`.

La signature contient, dans cet ordre logique : modèle du stub initial, modèle
de l'intervalle régulier, état ou clé de départ, indice du chemin, dimensions
de la grille, `path_count`, puis tableaux d'états observés.

La fonction :

1. construit une seule suite aléatoire pour le chemin ;
2. simule le stub initial ;
3. écrit uniquement les états pré-terminaux ;
4. stocke les observations en SoA date-major ;
5. retourne directement l'état terminal.

Lorsque seuls les points de la grille d'exercice sont observés, une dynamique
de Lévy exacte agrège les sous-pas de chaque intervalle avant le tirage. Cette
optimisation ne s'applique jamais à un résumé qui observe chacun des sous-pas,
comme une moyenne, un maximum ou une barrière quotidienne.

Pour une date donnée, deux chemins consécutifs écrivent à des adresses
consécutives. La maturité n'est pas écrite si elle peut être consommée
directement par le payoff.

## Simulateurs de résumés optionnels

Les fonctions suivantes reprennent le préfixe `simulate_`, construisent une
seule suite aléatoire par chemin et réutilisent `simulate_one_step` :

- `simulate_mean_state` : moyenne arithmétique des états observables ;
- `simulate_geometric_mean_state` : moyenne des logarithmes puis exponentielle ;
- `simulate_at_two_times` : états aux frontières de deux intervalles successifs ;
- `simulate_maximum_state` : maximum observé sur la grille.

Les accumulations de moyennes sont effectuées en FP64. Un résumé dépendant
d'une barrière, d'un coupon ou d'une règle d'exercice reste dans le pricer.

## Suite aléatoire Philox

Une ligne de résultat utilise la clé
`make_key(base_seed + result_index)`. Sous cette clé, chaque chemin possède le
sous-espace de compteur :

```text
(path_index: uint64, local_group_index: uint64)
```

Le code de simulation construit `UniformSequence(key, path)` une seule fois.
Chaque uniforme scalaire est obtenu par `uniforms.next()`, y compris les
uniformes transformés ensuite en normales ou en comptes de Poisson. La suite
cache les groupes de quatre produits par Philox et masque entièrement
`local_group_index`; aucune dynamique et aucun pricer ne manipule directement
les groupes.

Un processus exclusivement gaussien utilise la même `UniformSequence` avec
`next_normal(uniforms, normal_cache)`. Le `NormalPairCache` réutilise la seconde
normale de chaque transformation de Box-Muller, mais ne possède aucune suite
aléatoire et ne modifie pas le mapping Philox.

Un tirage conditionnel ou une méthode de rejet avance simplement la suite du
chemin courant. Les composantes restantes sont réutilisées par les transitions
suivantes. Seul le dernier groupe partiellement consommé peut laisser entre
zéro et trois uniformes inutilisés à la fin du chemin.

Les appels successifs à `uniforms.next()` sont affectés à des variables locales
avant toute transformation. Il ne faut pas placer plusieurs appels mutables
dans les arguments d'une même fonction, car leur ordre d'évaluation ne doit pas
dépendre du compilateur.

### Transformations de lois réutilisables

Les transformations indépendantes d'un modèle restent dans
`src/common/philox.cuh` et consomment la `UniformSequence` reçue par référence.
Elles ne créent ni clé ni sous-suite : une méthode de rejet continue donc le
flux du chemin courant.

```cpp
float marsaglia_tsang_gamma(
    UniformSequence& uniforms,
    NormalPairCache& normal_cache,
    float shape,
    float scale
);

float michael_schucany_haas_inverse_gaussian(
    UniformSequence& uniforms,
    NormalPairCache& normal_cache,
    float mean,
    float shape
);
```

`marsaglia_tsang_gamma` génère une Gamma paramétrée par forme et échelle. Pour
une forme inférieure à un, elle applique d'abord l'augmentation exacte de la
forme, puis la transformation en puissance. La boucle de rejet est locale au
chemin.

`michael_schucany_haas_inverse_gaussian` génère une inverse gaussienne
paramétrée par moyenne et forme. Son calcul du petit candidat utilise la forme
réciproque stable afin d'éviter la soustraction de deux nombres proches.

Les formes, échelles et moyennes strictement positives sont des préconditions
validées avant l'entrée dans les kernels.

## Invariants d'extension

- Conserver la séparation paramètres bruts, paramètres préparés et état mutable.
- Préparer les coefficients hors de la boucle de pas.
- Garder `one_step_transition` déterministe et indépendant de Philox.
- Construire une seule suite aléatoire continue par chemin.
- Conserver le mapping `(key, path_index, local_group_index)`.
- Ne pas réserver un nombre fixe de groupes par chemin.
- Ne pas introduire de paramètres produit dans la dynamique.
- Conserver les sorties multi-dates en SoA date-major.
- Ne jamais activer `--use_fast_math`.
- Mettre à jour ce document lorsque le contrat commun évolue.
