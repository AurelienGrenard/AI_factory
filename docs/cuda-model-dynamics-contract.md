# Contrat des dynamiques de modèles CUDA

## Objet

Ce document fixe l'interface et les responsabilités des dynamiques placées dans
`src/model/<asset_class>/<model>/dynamics.cuh` et `dynamics_impl.cuh`. Une
nouvelle dynamique doit conserver les mêmes couches, les mêmes noms et le même
ordre de fonctions lorsque sa mathématique les rend applicables.

Le fichier `dynamics.cuh` déclare les types et fonctions device réutilisables.
`dynamics_impl.cuh` contient leurs définitions force-inlinées et est inclus par
les kernels consommateurs. Les launchers autonomes restent des unités `.cu`
enregistrées dans CMake.

## Attributs et dépendances

Les fonctions de dynamique sont `__device__ __forceinline__`. Elles ne lancent
aucun kernel, n'allouent aucune mémoire et ne dépendent d'aucun paramètre de
produit. Les paramètres bruts proviennent de `parameters.hpp`; les primitives
aléatoires proviennent de `common/philox.cuh`.

Les paramètres d'un produit, son payoff, ses barrières, ses accumulateurs et
ses règles d'arrêt restent dans le pricer. Une dynamique expose seulement le
processus et les accès élémentaires à son état.

## Couches d'implémentation

Les responsabilités restent séparées entre les couches suivantes :

- `src/model/<asset_class>/<model>/parameters.hpp` porte la ligne brute du
  modèle, sans dépendance CUDA ni logique de sérialisation ;
- `src/model/<asset_class>/<model>/dataset.hpp/.cpp` expose et implémente son
  chargement hôte ;
- `src/model/<asset_class>/<model>/dynamics.cuh` et `dynamics_impl.cuh`
  déclarent et implémentent le processus autonome ;
- `src/model/<asset_class>/<model>/analytics.cuh` et `analytics_impl.cuh`
  exposent ses formules réutilisables ;
- `src/curve/<curve>/term_structure.cuh` et `term_structure_impl.cuh` exposent
  la courbe derrière les noms communs tels que `discount_factor` et
  `forward_rate` ;
- `src/model/<asset_class>/<model>/<curve>/analytics.cuh` et
  `analytics_impl.cuh` composent, lorsque nécessaire, le processus et la
  courbe calibrée.

Les moments numériquement stables d'un facteur gaussien mean-reverting sont
centralisés dans
`src/common/fixed_income/mean_reverting_gaussian.cuh`. OU et Vasicek les
exposent derrière leurs interfaces propres ; G2 les applique séparément à ses
deux facteurs et conserve localement ses covariances croisées. Ce helper ne
contient aucun état de modèle, aucune logique de courbe et aucune simulation de
chemin.

Les formules de courbe, payoffs, règles produit et kernels de pricing restent
dans leurs couches respectives. Une dynamique ne les réimporte pas pour
faciliter ponctuellement un pricer.

Pour les modèles factorisés, `common/simulation/path_simulation.cuh` porte les
boucles de chemin et `common/simulation/schedule.cuh` porte les calendriers.
Le marché ajoute ensuite ses observables et le produit sa `pricing_policy.cuh`.
Le fichier de dynamique ne conserve que les primitives du processus et leur
adaptateur statique `DynamicsPolicy`.

## Analytics obligataires affines

Un modèle de taux affine adopte la convention multiplicative

```text
P(t,T) = A(t,T) * exp(-B(t,T)' * X_t).
```

Son `analytics.cuh` expose `log_A`, `A`, `B`, `log_zero_coupon_bond` et
`zero_coupon_bond`. Un modèle à un facteur retourne un `float` depuis `B`; G2
et G2++ retournent le type partagé `TwoFactorAffineBondLoadings`. Le calcul de
`log_A` et des loadings est groupé dans `affine_bond_coefficients`, afin que le
chemin chaud d'un ZCB partage les moments et transcendantes. `A` reste un
wrapper de lisibilité; le pricing travaille en espace logarithmique et ne fait
pas un aller-retour `exp` puis `log`.

`log_discount_factor(parameters, state_integral, time)` et
`discount_factor(parameters, state_integral, time)` ne désignent pas ces
coefficients. Le code appelant extrait `state_integral` d'un éventuel état
joint sans transmettre les autres facteurs stochastiques. Un modèle standalone
ignore les arguments supplémentaires lorsque son état est directement le short
rate ; un modèle ajusté les utilise pour ajouter l'intégrale du shift
déterministe.

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
l'état mutable est `State`. Leur qualification
(`model::equity::kou::PreparedTransition`,
`model::fixed_income::g2::State`) apporte l'information du modèle sans produire des noms tels
que `kou::KouPreparedTransition`.

Pour un processus à transition directe, `PreparedModel` contient exactement
l'information du modèle nécessaire pour préparer n'importe quelle transition ;
`PreparedTransition` contient exactement les coefficients nécessaires pour
faire avancer ce modèle sur un intervalle `delta_t` donné. Le premier est
invariant par rapport au temps, construit une fois par ligne et partagé par les
chemins. Le second est construit une fois par intervalle distinct.

Pour les processus de taux réutilisés par un modèle ajusté à une courbe, la
primitive libre `prepare_model(ProcessParameters)` prépare uniquement le
processus et son état centré. La méthode
`DynamicsPolicy::prepare_model(ModelParameters)` complète ce résultat avec
l'état initial du modèle autonome. `initial_state(prepared_model)` n'a ainsi
besoin d'aucun argument extérieur. Hull-White et G2++ réutilisent directement
la primitive du processus centré sans introduire une surcharge ambiguë.

Pour un schéma numérique à pas fixe, `PreparedModel` contient directement les
coefficients de la transition élémentaire de durée `delta_t`. Ajouter un
`PreparedTransition` identique n'apporterait aucune séparation réelle : tous
les pas du chemin utilisent le même intervalle numérique.

Un modèle exact peut aussi satisfaire le contrat à pas fixe avec
`PreparedDynamics = {model, transition(delta_t)}` puis répéter cette transition.
Cette composition sert aux observations discrètes homogènes ; elle ne transforme
pas la loi exacte en schéma d'Euler.

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

Les modèles Markoviens ne déclarent aucun résultat de chemin dans leur
dynamique. La moyenne, le maximum, les barrières, les coupons et les écritures
de samples appartiennent aux handlers appelés par les simulateurs communs. Un
résultat à deux dates est le cas particulier d'un calendrier de deux
observations. Un exécuteur Volterra peut réutiliser le même contrat de handler
via `StatePolicy` sans prétendre exposer une transition markovienne `t -> t+dt`.

### `DynamicsPolicy`

Chaque modèle factorisé expose à la fin de `dynamics.cuh` une structure sans
donnée membre :

```cpp
struct DynamicsPolicy {
    using Parameters = ModelParameters;
    using PreparedDynamics = model_namespace::PreparedDynamics;
    using RandomContext = philox::NormalRandomContext;
    using State = model_namespace::State;

    static constexpr bool kPartitionInvariantAdvance = true;

    static PreparedDynamics prepare_dynamics(
        const Parameters& parameters,
        float delta_t
    );
    static State initial_state(const PreparedDynamics& dynamics);
    static void simulate_one_step(
        const PreparedDynamics& dynamics,
        RandomContext& random,
        State& state
    );
    static void advance(
        const PreparedDynamics& dynamics,
        std::uint32_t transition_count,
        RandomContext& random,
        State& state
    );
};
```

`advance(dynamics, n, random, state)` retourne l'état à la fin d'un intervalle
non observé de `n` pas homogènes. `advance(..., 0, ...)` ne modifie ni l'état ni
la suite aléatoire. La propriété `kPartitionInvariantAdvance` indique si cet
appel consomme exactement la même suite et produit les mêmes bits que `n`
appels successifs avec `step_count == 1`.

Bates fixe cette propriété à `false` : il simule les pas QE-M de Heston, puis
agrège le processus de sauts indépendant sur l'intervalle non observé. La loi à
la frontière est correcte, mais le partitionnement des appels change la
consommation Philox. Un calendrier dense appelle `advance(..., 1, ...)`, donc
les sauts restent appliqués à chaque date effectivement observée.

Les `using` sont des alias de types et ne stockent rien. Les méthodes statiques
redirigent vers les primitives propres au modèle et sont toutes
`__device__ __forceinline__`; il n'existe ni instance de policy, ni vtable, ni
dispatch runtime.

Le contrat de base n'impose aucun observable de marché. Une dynamique equity
ajoute `spot(state)` et, lorsque disponible, `log_spot(state)` ; ces capacités
sont vérifiées par `equity::SpotDynamicsPolicy` et
`equity::LogSpotDynamicsPolicy`. Une future dynamique de taux peut exposer ses
propres observables sans faux alias `spot`.

Pour un schéma à pas fixe, `PreparedDynamics` est le modèle déjà préparé pour
`delta_t`. `PreparedFixedStepDynamicsPolicy` vérifie l'état, le contexte
aléatoire et `advance` lorsque ces coefficients sont fournis extérieurement ;
`FixedStepDynamicsPolicy` ajoute la capacité de les construire directement sur
le device avec `prepare_dynamics(parameters, delta_t)`. Cette séparation permet
à rough Heston de préparer une approximation de noyau et une exponentielle de
matrice une seule fois côté hôte sans introduire une fausse préparation device.
Pour une simulation exacte, il agrège `PreparedModel` et
`PreparedTransition`; la policy expose aussi ces deux types et les surcharges
`prepare_model`, `prepare_transition`, `initial_state(model)` et
`simulate_one_step(model, transition, random, state)`.

`RandomContext` est défini dans `common/philox.cuh`. Il possède la suite
uniforme continue du chemin et seulement les caches dont la dynamique a besoin.
Le concept n'impose donc ni `NormalPairCache` séparé, ni nombre fixe de lois
aléatoires.

Le `static_assert` placé après la déclaration contrôle
`simulation::DynamicsPolicy<DynamicsPolicy>`. Un modèle exact contrôle en plus
`ExactTransitionDynamicsPolicy<DynamicsPolicy>`. Les concepts vérifient les
types et signatures utilisés par les templates communs ; ils n'ajoutent aucun
coût d'exécution.

## Primitives fondamentales

Les déclarations publiques apparaissent dans cet ordre dans `dynamics.cuh`.
Les primitives communes sont `prepare_model`, l'éventuel
`prepare_transition`, `initial_state` et `one_step_transition`. Une signature
de variates n'est jamais artificiellement uniformisée lorsqu'une loi exige une
consommation différente. `DynamicsPolicy` adapte ces primitives à l'interface
commune consommée par les templates de simulation.

Les simulations terminales, régulières et irrégulières sont exclusivement
portées par `common/simulation/path_simulation.cuh`. Elles ne sont pas dupliquées
dans les dynamiques factorisées.

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

Pour OU, Vasicek, CIR et G2, la primitive mathématique reçoit les seuls
paramètres du processus :

```cpp
PreparedModel prepare_model(
    const ProcessParameters& parameters
);
```

La policy autonome reçoit néanmoins `ModelParameters`, appelle cette primitive
et copie l'état initial dans `PreparedModel`. Cette frontière permet aux modèles
ajustés à une courbe de partager exactement la même dynamique centrée.

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
mais préparent une transition plus riche qui contient aussi les moments exacts
de l'intégrale.

La dynamique `cir::joint` réutilise également le `PreparedModel` et la
transition exacte de l'état CIR. Elle ajoute le coefficient du trapèze local
pour accumuler
`integral_t^(t + delta_t) r_s ds ~= delta_t (r_t + r_(t + delta_t)) / 2`.
Sa policy expose uniquement le contrat à pas fixe : elle ne satisfait pas
`ExactTransitionDynamicsPolicy`, car l'intégrale est approchée même si chaque
extrémité CIR est tirée exactement.

### `initial_state`

Attribut : `__device__ __forceinline__`.

```cpp
State initial_state(const PreparedModel& prepared_model);
```

La fonction construit l'état en temps zéro à partir du modèle préparé. L'état
initial n'est jamais un argument parallèle des simulateurs communs : il est
porté par `PreparedModel` ou `PreparedDynamics`.

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

### Simulations terminales

`common/simulation/path_simulation.cuh` expose deux fonctions
`__device__ __forceinline__` :

```cpp
simulate_fixed_step_terminal<Dynamics>(...);
simulate_exact_transition_terminal<Dynamics>(...);
```

Elles retournent l'état terminal sans écriture en mémoire globale. La première
construit un unique `RandomContext` et appelle `Dynamics::advance` pour le
nombre de pas demandé. La seconde appelle une fois la transition exacte
préparée. Merton et Kou consomment ainsi un seul incrément compound-Poisson sur
la durée préparée ; VG et NIG consomment directement leur incrément de Lévy
terminal.

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
« transition jointe spot-volatilité exacte ». Les termes `1-exp(-x)` de la
variance d'extrémité et de la corrélation utilisent `-expm1f(-x)` afin de rester
stables lorsque `kappa * delta_t` est petit. Une transition consomme exactement
trois normales, dans l'ordre innovation OU, résidu de l'incrément de volatilité,
puis résidu spot ; cette affectation Philox fait partie du contrat.

### Calendriers réguliers

Les deux algorithmes génériques sont :

```cpp
simulate_fixed_step_regular_schedule<Dynamics>(...);
simulate_exact_transition_regular_schedule<Dynamics>(...);
```

Pour un modèle à schéma, la forme homogène reçoit
`transitions_per_observation`, `observation_count`, la clé, le chemin et un
handler. La forme `stubbed` ajoute `initial_transition_count`. Le stub et les
intervalles réguliers utilisent le même `delta_t` préparé.

Pour un modèle exact, la forme homogène reçoit une transition préparée unique ;
la forme `stubbed` reçoit `initial_transition` et `regular_transition`. Un
intervalle d'observation consomme un seul incrément exact, quelle que soit sa
longueur.

La fonction :

1. construit une seule suite aléatoire pour le chemin ;
2. simule le premier intervalle, distinct seulement dans la forme `stubbed` ;
3. notifie le handler à chaque date contractuelle ;
4. retourne directement l'état terminal.

Cette agrégation exacte ne s'applique jamais à un résumé qui observe chacun des
pas, comme une moyenne, un maximum ou une barrière quotidienne. Dans ce cas,
le pricer prépare explicitement une transition fine et la répète.

Les handlers `SpotObservationWriter` et `SpotAndStateObservationWriter`
centralisent les écritures SoA nécessaires aux samples et à
Longstaff–Schwartz. Le pricer fixe leur `write_count` afin d'inclure ou non la
maturité sans modifier la boucle de simulation.

### Calendriers irréguliers

Les dynamiques factorisées n'exposent pas ces boucles. Les algorithmes
communs de `common/simulation/path_simulation.cuh` sont :

```cpp
simulate_fixed_step_regular_schedule<Dynamics>(...);
simulate_fixed_step_calendar<Dynamics>(...);
simulate_exact_transition_regular_schedule<Dynamics>(...);
simulate_exact_transition_calendar<Dynamics>(...);
```

Ils conservent une unique suite Philox par chemin et notifient un
`ObservationHandlerFor<Dynamics>` à chaque date contractuelle. Le handler
accumule uniquement les quantités requises par le payoff. Un calendrier de deux
observations couvre naturellement les produits à deux dates.

## Limites d'uniformisation

Rough Bergomi, log-modulated rough Bergomi, rough SABR et rough Stein--Stein ne
sont pas forcés dans ce contrat Markovien. Ils composent le moteur commun
`hybrid_fft_pricer` avec quatre politiques : driver Volterra gaussien,
transformation de chemin propre au modèle, calendrier et produit. Le
`PathPolicy` reçoit successivement `Y_i`, sa variance déterministe lorsqu'elle
est requise, et les normales corrélées ; il possède seul les transformations
du driver vers la variance ou la volatilité, puis l'évolution de `S_i`. Les
handlers reçoivent des observations spot scalaires et restent donc
indépendants du layout d'état.

Cette composition s'applique quand l'intégrale de Volterra est une convolution
linéaire d'un bruit gaussien par un kernel stationnaire. Rough Heston et
quadratic rough Heston ne rentrent pas dans cette classe car leur intégrande
dépend de l'état simulé. Après approximation exponentielle, ils satisfont en revanche
`PreparedFixedStepDynamicsPolicy` et réutilise les schedules et kernels Monte
Carlo communs. L'état joint CIR reste
volontairement limité au contrat à pas fixe afin que son intégration
trapézoïdale ne puisse pas être utilisée comme une transition exacte sur un
grand intervalle. L'uniformité porte sur les responsabilités réellement
communes, pas sur le nombre de champs, de normales ou de caches.

## Observations et résumés equity

Chaque `pricing_policy.cuh` définit le handler minimal de son payoff. Une option
asiatique accumule sa moyenne, une barrière conserve son indicateur de survie
et un lookback son extremum. Les templates de chemin restent identiques pour
tous les modèles et le dispatch statique permet au compilateur d'inliner le
handler sans coût virtuel.

Les accumulations nécessitant une meilleure stabilité peuvent rester en FP64
dans le handler, tandis que l'état simulé demeure en FP32. Black-Scholes suit
ce contrat pour ses pricers Monte Carlo. Rough Bergomi expose un `StatePolicy`
spot/log-spot aux mêmes handlers, mais conserve son exécution FFT spécialisée.

Le test générique `tests/common/dynamics_contract.cuh` vérifie pour chaque
policy concernée la reproductibilité, l'isolation des chemins, la neutralité
de `advance(0)`, l'accord entre simulation terminale et calendrier à une
observation, ainsi que la parité entre transition exacte et un unique pas
préparé. L'égalité bit-à-bit `advance(n) == n * advance(1)` est vérifiée
uniquement lorsque `kPartitionInvariantAdvance` vaut `true`.
`equity_dynamics_policy_cuda_test.cu` instancie ce contrat sur tous les modèles
equity non-rough et inclut toutes leurs implémentations dans une même unité de
traduction. `fixed_income_dynamics_policy_cuda_test.cu` l'applique aux modèles
de taux et à leurs variantes jointes état-intégrale.

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

Les modèles suivent uniformément
`workbench::model::<asset_class>::<model>`. Les namespaces portent ensuite les
noms du processus et de la courbe. Les fonctions réutilisables ne les répètent
donc pas : `model::equity::heston::prepare_model`,
`model::fixed_income::ornstein_uhlenbeck::prepare_model` ou
`model::fixed_income::hull_white::nelson_siegel::compose_fitted_model` sont les
formes attendues.

Employer les mêmes noms de contrôle lorsque leur sens est identique : `path`,
`step`, `observation`, `output_index`, `uniforms`, `normal_cache` et `state`.
Employer `prepared_model`, `prepared_transition`, `delta_t` et `step_count`
pour les primitives et les policies. Pour un schéma à pas fixe, employer
`prepared_model`, `initial_stub_steps`,
`steps_per_observation`, `observation_count` et `observation_stride`. Les noms
spécifiques tels que
`observed_variances` ou `observed_integrated_states` sont réservés à des données
réellement différentes.

L'uniformité reste la règle, mais une fonction, une structure ou un template ne
doit pas être ajouté uniquement pour forcer une syntaxe identique. Toute
abstraction supplémentaire doit correspondre à une différence ou une
réutilisation réelle de mathématiques, de données ou de stratégie d'exécution.
