# Contrat des dynamiques de modèles CUDA

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

## Couches d'implémentation

Les responsabilités restent séparées entre les couches suivantes :

- `src/model/<asset_class>/<model>/dataset.hpp/.cpp` porte la ligne brute et
  son chargement ;
- `src/model/<asset_class>/<model>/dynamics.cuh/.cu` implémente le processus
  autonome ;
- `src/model/<asset_class>/<model>/analytics.cuh/.cu` expose ses formules
  réutilisables ;
- `src/curve/<curve>/term_structure.cuh/.cu` expose la courbe derrière les noms
  communs tels que `discount_factor` et `forward_rate` ;
- `src/model/<asset_class>/<model>/<curve>/analytics.cuh/.cu` compose, lorsque
  nécessaire, le processus et la courbe calibrée.

Les moments numériquement stables d'un facteur gaussien mean-reverting sont
centralisés dans
`src/model/fixed_income/common/mean_reverting_gaussian.cuh`. OU et Vasicek les
exposent derrière leurs interfaces propres ; G2 les applique séparément à ses
deux facteurs et conserve localement ses covariances croisées. Ce helper ne
contient aucun état de modèle, aucune logique de courbe et aucune simulation de
chemin.

Les formules de courbe, payoffs, règles produit et kernels de pricing restent
dans leurs couches respectives. Une dynamique ne les réimporte pas pour
faciliter ponctuellement un pricer.

## Analytics obligataires affines

Un modèle de taux affine adopte la convention multiplicative

```text
P(t,T) = A(t,T) * exp(-B(t,T)' * X_t).
```

Son `analytics.cuh` expose `log_A`, `A`, `B`, `log_zero_coupon_bond` et
`zero_coupon_bond`. Un modèle à un facteur retourne un `float` depuis `B`; G2
et G2++ retournent le type partagé `G2BondLoadings`. Le `.cu` calcule `log_A`
et tous les loadings dans un helper privé `affine_bond_coefficients`, afin que
le chemin chaud d'un ZCB partage les moments et transcendantes. `A` reste un
wrapper de lisibilité; le pricing travaille en espace logarithmique et ne fait
pas un aller-retour `exp` puis `log`.

`log_discount_factor` et `discount_factor` ne désignent pas ces coefficients :
ils reçoivent uniquement la valeur scalaire `integral_0^t r_s ds` et retournent
respectivement son opposé et son exponentielle. Le code appelant extrait donc
`state_integral` d'un éventuel état joint sans transmettre les autres facteurs
inutiles. Pour un modèle `++`, les paramètres de courbe et le temps complètent
cette entrée scalaire afin d'ajouter l'intégrale du shift déterministe.

## Types communs

### Compteurs et indices

Les nombres de pas, d'observations, d'exercices, de paiements et les indices
des boucles device correspondantes utilisent `std::uint32_t`. Ils décrivent
des calendriers ou des discrétisations bornés, pas des dimensions mémoire.

`std::size_t` reste réservé aux tailles d'allocation et de workspace, aux
dimensions globales, aux strides, aux offsets et aux indices de tableaux
globaux. Lorsqu'un planning hôte en `std::size_t` produit un compteur consommé
comme tel sur le device, le launcher valide d'abord sa plage puis le convertit
explicitement en `std::uint32_t`.

Les composantes de géométrie CUDA effectivement transmises à `dim3`, ainsi que
les indices ou dimensions directement issus de `threadIdx` et `blockDim`,
utilisent `unsigned int`.

### Modèle, transition et état

Dans un namespace propre au modèle, les types ne répètent pas son nom : la
ligne brute est `ModelParameters` ou `ProcessParameters`, le modèle préparé est
`PreparedModel`, la transition exacte préparée est `PreparedTransition` et
l'état mutable est `State`. Leur qualification (`kou::PreparedTransition`,
`model::g2::State`) apporte l'information du modèle sans produire des noms tels
que `kou::KouPreparedTransition`.

Pour un processus à transition directe, `PreparedModel` contient exactement
l'information du modèle nécessaire pour préparer n'importe quelle transition ;
`PreparedTransition` contient exactement les coefficients nécessaires pour
faire avancer ce modèle sur un intervalle `delta_t` donné. Le premier est
invariant par rapport au temps, construit une fois par ligne et partagé par les
chemins. Le second est construit une fois par intervalle distinct.

Pour un schéma numérique à pas fixe, `PreparedModel` contient directement les
coefficients de la transition élémentaire de durée `delta_t`. Ajouter un
`PreparedTransition` identique n'apporterait aucune séparation réelle : tous
les pas du chemin utilisent le même intervalle numérique.

Ces structures ne contiennent ni paramètre produit, ni pointeur propriétaire,
ni allocation dynamique. Une transition exacte de taux state-only et sa
transition jointe état-intégrale peuvent partager le même `PreparedModel` tout
en définissant chacune leur propre `PreparedTransition`.

### État de chemin

Le type `State` contient uniquement les variables mutables d'un chemin.
Il reste dans les registres lorsque cela est possible. Un modèle dérivé peut
réutiliser l'état de son modèle de base lorsque cela reflète sa construction
mathématique, comme Bates réutilise l'état log-spot/variance de Heston.

### Résultats de chemin

Un type spécialisé tel que `MeanPathResult`, `GeometricMeanPathResult` ou
`MaximumPathResult` ne contient que les observables demandés. Il n'est ajouté
que si plusieurs produits peuvent le réutiliser sans introduire de logique
produit dans la dynamique. Un résultat à deux dates n'est pas exposé : il est
le cas particulier d'un calendrier de deux observations.

## Fonctions fondamentales

Les déclarations publiques apparaissent dans cet ordre dans `dynamics.cuh`.
Les noms communs sont conservés lorsqu'ils désignent la même responsabilité :
`prepare_model`, `prepare_transition`, `prepare_calendar`, `initial_state`,
`one_step_transition`, `simulate_terminal_state`, `simulate_on_calendar` et
`simulate_on_regular_grid`. Une signature de variates n'est jamais artificiellement
uniformisée lorsqu'une loi exige une consommation différente.

### `prepare_model`

Attribut : `__device__ __forceinline__`.

Pour une transition exacte ou directe :

```cpp
PreparedModel prepare_model(
    const ModelParameters& parameters
);
```

Elle prépare uniquement les coefficients invariants par rapport à la durée de
transition. Elle ne reçoit ni maturité, ni `delta_t`, ni nombre de pas.

Pour un schéma discrétisé :

```cpp
PreparedModel prepare_model(
    const ModelParameters& parameters,
    float delta_t
);
```

Elle prépare exactement la transition élémentaire de durée `delta_t` sous les
paramètres fournis. CEV, Heston, Bates et Schöbel-Zhu suivent cette convention ;
`num_steps` reste uniquement le nombre d'appels successifs à cette transition.

Dans les deux cas, `prepare_model` ne génère aucun aléa et ne construit aucun
état de chemin.

### `prepare_transition`

Attribut : `__device__ __forceinline__`. Cette fonction appartient aux modèles
à transition exacte ou directe :

```cpp
PreparedTransition prepare_transition(
    const PreparedModel& prepared_model,
    float delta_t
);
```

Elle combine le modèle invariant avec une durée strictement positive et produit
les seuls coefficients nécessaires à un saut exact sur cet intervalle. Un
payoff terminal prépare directement sa durée contractuelle. Une barrière ou une
moyenne prépare au contraire une transition fine uniquement parce qu'elle
observe réellement le chemin à cette fréquence.

Black-Scholes, Merton, Kou, Variance-Gamma et NIG suivent ce découpage pour le
log-spot. OU, Vasicek, CIR et G2 le suivent pour leurs facteurs de taux. Les
dynamiques jointes OU, Vasicek et G2 réutilisent le `PreparedModel` state-only,
mais préparent une transition plus riche qui contient aussi les moments de
l'intégrale.

### `prepare_calendar`

Pour un modèle exact, le helper optionnel

```cpp
void prepare_calendar(
    const PreparedModel& prepared_model,
    const std::uint32_t* interval_steps,
    std::uint32_t interval_count,
    float delta_t,
    PreparedTransition* transitions
);
```

convertit les écarts entiers entre observations en transitions exactes. Chaque
entrée vaut `prepare_transition(prepared_model, interval_steps[i] * delta_t)`.
Le tableau de transitions est préparé une fois par ligne, jamais une fois par
chemin. Un schéma à pas fixe conserve au contraire le tableau des nombres de
pas et réutilise son unique `PreparedModel`.

### `initial_state`

Attribut : `__device__ __forceinline__`.

```cpp
State initial_state(const PreparedModel& prepared_model);
```

La fonction construit l'état en temps zéro lorsque celui-ci est porté par les
paramètres du modèle. Elle est omise lorsque l'état initial est naturellement
un argument explicite des fonctions de simulation.

### `one_step_transition`

Attribut : `__device__ __forceinline__`.

```cpp
void one_step_transition(
    /* PreparedModel si la loi en a besoin */,
    const PreparedTransition& prepared_transition,
    /* variates explicites ou suite adaptative propres au modèle */,
    State& state
);
```

Pour une consommation fixe, cette fonction est la transformation déterministe
d'un état : les variates apparaissent explicitement et elle ne connaît ni la
seed, ni l'indice du chemin. Son arité est volontairement propre au modèle : une
normale sous Black-Scholes, une normale et une somme de sauts sous Kou, deux
innovations corrélées sous G2, etc.

Une loi exacte à consommation adaptative peut recevoir directement
`philox::UniformSequence&` et `philox::NormalPairCache&`. C'est le cas de CIR :
`one_step_transition` effectue le mélange Poisson-Gamma de la chi-deux
décentrée. La fonction utilise uniquement la suite continue du chemin courant ;
elle ne construit ni clé, ni sous-suite.

### `simulate_one_step`

Attribut : `__device__ __forceinline__`, fonction privée au `.cu`.

```cpp
void simulate_one_step(
    const PreparedModel& prepared_model,
    const PreparedTransition& prepared_transition,
    philox::UniformSequence& uniforms,
    /* caches aléatoires optionnels */,
    State& state
);
```

Cette fonction privée est la frontière commune entre les simulateurs de chemin
et la consommation propre au modèle. Sa signature ne contient que le modèle,
la transition, la suite Philox, les éventuels caches et l'état. Elle consomme
les uniformes dans un ordre explicite, construit les variates nécessaires, puis
appelle `one_step_transition`. Un argument `PreparedModel` peut être inutilisé
pour une loi simple ; il est conservé ici parce que d'autres lois, comme VG,
NIG ou CIR, en ont réellement besoin.

### `simulate_terminal_state`

Attribut : `__device__ __forceinline__`.

Pour un schéma à pas fixe :

```cpp
State simulate_terminal_state(
    const PreparedModel& prepared_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t num_steps
);
```

Pour une transition directe :

```cpp
State simulate_terminal_state(
    const PreparedModel& prepared_model,
    const PreparedTransition& prepared_transition,
    philox::PhiloxKey key,
    std::size_t path
);
```

La fonction retourne l'état terminal sans écriture en mémoire globale. La
version à schéma construit une seule `UniformSequence` pour tout le chemin et
répète sa transition `num_steps` fois. La version directe appelle une seule fois
la transition préparée. Merton et Kou consomment ainsi un seul incrément
compound-Poisson sur la durée préparée ; VG et NIG consomment directement leur
incrément de Lévy terminal.

CEV ne possède pas ici de transition trajectorielle exacte. Sa loi marginale
peut se ramener à une loi du chi carré non centrale, mais, sur le domaine
`beta >= 0.5` retenu par le catalogue, la frontière absorbante atteignable
introduit une masse en zéro et un échantillonnage spécifique. La V1 utilise
donc un pas de Milstein uniforme pour tous les payoffs, avec absorption à zéro.
La puissance `S^beta` est calculée une fois par pas et réutilisée pour former
`S^(2 beta - 1)`, afin d'éviter une seconde évaluation coûteuse de `powf` sans
ajouter de registre persistant.

Schöbel-Zhu tire exactement l'extrémité OU de la volatilité. L'innovation OU est
couplée au Brownien de volatilité intégré sur le pas, puis au Brownien spot. Le
log-spot reste discrétisé par Euler: « endpoint OU exact » ne signifie donc pas
« transition jointe spot-volatilité exacte ».

### `simulate_on_regular_grid`

Attribut : `__device__ __forceinline__`.

Pour un modèle à schéma, la signature contient un unique `PreparedModel`, la
clé et l'indice du chemin, `initial_stub_steps`, `steps_per_observation`,
`observation_count`, `observation_stride`, puis les tableaux d'états observés.
Le stub et les intervalles réguliers utilisent strictement le même `delta_t`
préparé.

Pour un modèle exact, les deux compteurs de pas sont remplacés par
`initial_stub_transition` et `regular_transition`. Un intervalle d'observation
consomme ainsi un seul incrément exact, quelle que soit sa longueur.

La fonction :

1. construit une seule suite aléatoire pour le chemin ;
2. simule le stub initial ;
3. écrit uniquement les états pré-terminaux ;
4. stocke les observations en SoA date-major ;
5. retourne directement l'état terminal.

Cette agrégation exacte ne s'applique jamais à un résumé qui observe chacun des
pas, comme une moyenne, un maximum ou une barrière quotidienne. Dans ce cas,
le pricer prépare explicitement une transition fine et la répète.

Pour une date donnée, deux chemins consécutifs écrivent à des adresses
consécutives. La maturité n'est pas écrite si elle peut être consommée
directement par le payoff.

`observation_stride` est la distance entre deux observations successives du
même chemin. Il vaut le nombre de chemins pour une sortie SoA date-major et
`1` pour une sortie contiguë propre à un sample.
Chaque pointeur d'observation doit déjà viser le premier emplacement du chemin
courant : `base + path` en date-major, ou le début du tableau local pour un
sample. L'indice `path` reste ainsi réservé à la suite Philox et n'est jamais
réutilisé implicitement comme offset de sortie.

### `simulate_on_calendar`

Attribut : `__device__ __forceinline__`.

```cpp
State simulate_on_calendar(
    const PreparedModel& prepared_model,
    const PreparedTransition* prepared_transitions,
    std::uint32_t observation_count,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t observation_stride,
    /* tableaux d'observations */
);
```

Sous transition exacte, le tableau contient une transition préparée entre zéro
et `t1`, puis entre chaque paire `t_i` et `t_{i+1}`. Sous schéma, ce même rôle
est tenu par `steps_between_observations`, qui indique combien de fois appeler
la transition élémentaire. Un pointeur CUDA brut ne portant pas sa taille,
`observation_count` reste explicite. La fonction conserve une unique suite
Philox, écrit les états pré-terminaux selon `observation_stride` et retourne
l'état terminal. Les pointeurs suivent la même convention d'ancrage que la
grille régulière. `simulate_at_two_times` n'est pas exposé : un calendrier de
deux observations couvre ce cas sans dupliquer la logique aléatoire.

## Limites d'uniformisation

Rough Bergomi n'est pas forcé dans ce contrat Markovien : son historique de
Volterra, son workspace et sa consommation aléatoire exigent une interface
spécifique qui sera refondue avec son propre contrat. De même, CIR n'expose pas
une fausse transition jointe état-intégrale tant qu'une méthode justifiée n'est
pas implémentée. L'uniformité porte sur les responsabilités réellement
communes, pas sur le nombre de champs, de normales ou de caches.

## Simulateurs de résumés optionnels

Les fonctions suivantes reprennent le préfixe `simulate_`, construisent une
seule suite aléatoire par chemin et réutilisent `simulate_one_step` :

- `simulate_mean_state` : moyenne arithmétique des états observables ;
- `simulate_geometric_mean_state` : moyenne des logarithmes puis exponentielle ;
- `simulate_maximum_state` : maximum observé sur la grille.

Les accumulations de moyennes sont effectuées en FP64. Un résumé dépendant
d'une barrière, d'un coupon ou d'une règle d'exercice reste dans le pricer.

Sous Black-Scholes, `simulate_geometric_mean_state` exploite directement la loi
gaussienne de la moyenne discrète des log-spots. Elle inclut le spot initial et
la maturité, comme les autres dynamiques, et ne conserve ni état terminal ni
point intermédiaire lorsque seul le résumé géométrique est requis.

## Modèles ajustés à une courbe

Un modèle dérivé réutilise les dynamiques de son processus de base lorsque cela
reflète sa construction mathématique. Hull-White ne répète donc pas les
transitions Ornstein-Uhlenbeck : ses analytics composent le facteur OU et la
courbe puis reconstruisent `r(t) = x(t) + phi(t)` uniquement lorsqu'un payoff
a besoin du taux, d'un facteur d'actualisation ou d'un prix de zéro-coupon.

G2++ suit la même règle. Ses analytics réutilisent les moments exacts et l'état
à deux facteurs de G2, puis ajoutent seulement le décalage déterministe de la
courbe. Les versions autonomes et ajustées conservent ainsi les mêmes noms de
formules sans dupliquer leur processus stochastique.

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

std::uint32_t poisson_from_uniform_sequence(
    UniformSequence& uniforms,
    float mean
);

float scaled_noncentral_chi_square(
    UniformSequence& uniforms,
    NormalPairCache& normal_cache,
    float degrees_of_freedom,
    float noncentrality,
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

`poisson_from_uniform_sequence` conserve l'inversion exacte pour les petites
intensités et utilise le rejet transformé PTRS de Hoermann pour les grandes.
La seconde branche ne prépare jamais `exp(-mean)` et reste donc utilisable
lorsque cette probabilité sous-déborde en FP32. Les rejets continuent la suite
locale du chemin comme pour les autres transformations.

`scaled_noncentral_chi_square` compose cette loi de Poisson avec
`marsaglia_tsang_gamma` selon la représentation exacte de la loi du chi-deux
non centrée. Le facteur d'échelle est transmis à Gamma, sans allocation ni
multiplication séparée après le tirage. CIR utilise cette primitive avec une
suite uniforme et un cache de normales conservés pendant tout le chemin.

`michael_schucany_haas_inverse_gaussian` génère une inverse gaussienne
paramétrée par moyenne et forme. Son calcul du petit candidat utilise la forme
réciproque stable afin d'éviter la soustraction de deux nombres proches.

Les formes, échelles et moyennes strictement positives sont des préconditions
validées avant l'entrée dans les kernels.

## Invariants d'extension

- Conserver la séparation paramètres bruts, paramètres préparés et état mutable.
- Préparer les coefficients hors de la boucle de pas.
- Garder `one_step_transition` déterministe et indépendant de Philox pour les
  schémas à consommation fixe ; une loi exacte adaptative consomme directement
  la suite du chemin sans créer de clé ou de sous-suite.
- Construire une seule suite aléatoire continue par chemin.
- Conserver le mapping `(key, path_index, local_group_index)`.
- Ne pas réserver un nombre fixe de groupes par chemin.
- Ne pas introduire de paramètres produit dans la dynamique.
- Conserver les sorties multi-dates en SoA date-major.
- Mettre à jour ce document lorsque le contrat commun évolue.

## Nommage et structure des boucles

Les namespaces portent les noms du processus, du modèle et de la courbe. Les
fonctions réutilisables ne les répètent donc pas : `heston::prepare_model`,
`model::ornstein_uhlenbeck::prepare_model` ou
`model::hull_white::nelson_siegel::compose_model` sont les formes attendues.

Employer les mêmes noms de contrôle lorsque leur sens est identique : `path`,
`step_index`, `observation`, `output_index`, `uniforms`, `normals` et `state`.
Pour un schéma à pas fixe, employer `prepared_model`, `initial_stub_steps`,
`steps_per_observation`, `observation_count` et `observation_stride`. Les noms
spécifiques tels que
`observed_variances` ou `observed_integrated_states` sont réservés à des données
réellement différentes.

L'uniformité reste la règle, mais une fonction, une structure ou un template ne
doit pas être ajouté uniquement pour forcer une syntaxe identique. Toute
abstraction supplémentaire doit correspondre à une différence ou une
réutilisation réelle de mathématiques, de données ou de stratégie d'exécution.
