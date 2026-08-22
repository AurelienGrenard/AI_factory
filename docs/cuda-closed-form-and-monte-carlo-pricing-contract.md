# Contrat des pricers CUDA : formule fermée et Monte Carlo

## Objet

Ce document fixe l'ossature des pricers sans exercice anticipé placés dans
`src/model/<asset_class>/<model>/[<curve>/]<product>.cuh/.cu`. Un nouveau pricer doit reprendre
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
- Le côté call/put du payoff est un paramètre de template `OptionSide Side`.
  Une swaption utilise le type dédié `SwaptionSide` pour éviter toute
  correspondance implicite entre call/put et payer/receiver. Le côté ne doit
  pas être stocké dans les lignes et ne doit pas produire de branche runtime
  dans le chemin chaud.

## Attributs CUDA utilisés

| Attribut | Domaine | Contrat |
|---|---|---|
| fonction C++ sans attribut CUDA | hôte | validation, planning et lancement |
| `__device__ __forceinline__` | device | primitive appelée dans un kernel et destinée à être inlinée |
| `__global__` | device, appelée par l'hôte | point d'entrée d'un kernel CUDA |
| `template<OptionSide Side>` | compilation | spécialisation call/put sans branche runtime |
| `template<SwaptionSide Side>` | compilation | spécialisation payer/receiver sans branche runtime |
| `__restrict__` | paramètres de kernel | absence d'alias entre les tableaux concernés |

## Composition Monte Carlo equity

Les pricers equity standards hors Black-Scholes, rough Bergomi et exercice
anticipé utilisent quatre politiques statiques. Aucun héritage, allocation ou
appel virtuel n'entre dans le kernel : les concepts contrôlent les interfaces,
puis le compilateur inline la composition complète.

```text
DynamicsPolicy<Model>
        |
        v
SchedulePolicy<DynamicsPolicy>
        |
        v
PricingPolicy<SchedulePolicy, DiscountPolicy, Side>
        |
        v
monte_carlo_price_kernel<PricingPolicy>
```

### `DynamicsPolicy`

Chaque `dynamics.cuh/.cu` expose `DynamicsPolicy`. Ses alias sont
`Parameters`, `PreparedDynamics`, `RandomContext` et `State`. Un modèle à
transition exacte expose en plus `PreparedModel` et `PreparedTransition`.

Son interface commune est :

```cpp
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
static float spot(const State& state);
```

Les modèles exacts ajoutent les surcharges qui séparent les coefficients
invariants de ceux d'un intervalle. `EquityDynamicsPolicy` et
`ExactTransitionDynamicsPolicy` vérifient ces contrats à la compilation.

### `SchedulePolicy`

Un schedule convertit les jours contractuels, prépare les transitions et fait
avancer un chemin. Il ne connaît aucun payoff. Les compositions disponibles
sont :

| Besoin | Schéma fixe | Transition exacte |
|---|---|---|
| payoff terminal | `FixedStepTerminalSchedule` | `ExactTransitionTerminalSchedule` |
| observations régulières | `FixedStepRegularSchedule` | `ExactTransitionRegularSchedule` |
| monitoring à chaque pas | `FixedStepDenseSchedule` | transition fine régulière |
| calendrier de taille statique | `FixedStepCalendarSchedule<N>` | `ExactTransitionCalendarSchedule<N>` |

Les produits terminaux appellent `simulate_terminal`. Les produits de chemin
fournissent un observation handler possédant :

```cpp
bool on_initial_state(const State& state);
bool on_observation(std::uint32_t observation, const State& state);
```

Retourner `false` arrête immédiatement le chemin, notamment après une barrière
knock-out ou un autocall.

### `PricingPolicy`

Une politique située dans `src/product/<product>/pricing_policy.cuh` reçoit le
schedule et la politique d'actualisation. Elle expose les alias
`Dynamics`, `ModelParameters`, `ProductParameters`, `PricingConfiguration`,
`DeviceInputs`, puis les éléments suivants :

```cpp
struct PreparedRow;

static PreparedRow prepare_row(
    const ModelParameters& model,
    const ProductParameters& product,
    const PricingConfiguration& configuration,
    const DeviceInputs& inputs,
    std::uint64_t seed
);

static float evaluate_path(
    const PreparedRow& row,
    std::size_t path
);

static void validate_configuration(
    const PricingConfiguration& configuration,
    const DeviceInputs& inputs,
    std::size_t monte_carlo_paths_per_price
);
```

`PreparedRow` contient uniquement les états préparés et scalaires nécessaires
à tous les chemins d'un prix. Il reste trivially copyable, borné à 256 octets
par `ScalarMonteCarloPricingPolicy`, puis stocké une seule fois en mémoire
partagée par bloc. `evaluate_path` ne reçoit donc aucun argument propre au
modèle ou au produit hors de cette ligne compacte.

Le côté call/put est un paramètre de template lorsque le payoff le demande. Un
produit sans côté ne crée ni template artificiel ni branche runtime.

### `monte_carlo_price_kernel<PricingPolicy>`

Le kernel générique de `common/equity/monte_carlo_kernel.cuh` :

1. décode la ligne modèle-produit ;
2. construit un unique `PreparedRow` partagé par le bloc ;
3. distribue les chemins entre les threads ;
4. accumule somme et somme des carrés en FP64 ;
5. réduit les moments et écrit prix et erreur standard en FP32.

`validate_monte_carlo_launch<PricingPolicy>` valide les pointeurs, la
construction alignée ou cartésienne, le batch, le nombre de chemins, la
configuration numérique, la géométrie et la plage de seeds.
`launch_monte_carlo_cuda<PricingPolicy>` vérifie ensuite l'occupation, émet les
diagnostics optionnels, lance le kernel et contrôle `cudaGetLastError()`.

### Launcher modèle-produit

Le `.cu` du couple modèle-produit ne réimplémente plus le kernel. Il compose
ses types :

```cpp
using Schedule = /* fixed-step ou exact, terminal/régulier/dense/calendrier */;
using Discount = equity::ConstantRateDiscountPolicy<DynamicsPolicy>;
using PricingPolicy = product::EuropeanOptionPricingPolicy<
    Schedule,
    Discount,
    Side
>;
```

Le launcher public conserve sa signature historique, construit les petites
structures `configuration` et `inputs`, puis appelle uniquement
`launch_monte_carlo_cuda<PricingPolicy>`. Les variantes call et put sont
explicitement instanciées au bas du fichier.

## Formules fermées et fixed income

Les pricers en formule fermée conservent un `PreparedRow` local au thread,
`prepare_row`, `evaluate_price` et un thread par prix. Les launchers de
swaptions instancient `SwaptionSide::payer` et `SwaptionSide::receiver`. Cette
couche n'est pas forcée dans le contrat Monte Carlo equity : elle ne possède ni
schedule de chemin, ni contexte aléatoire, ni réduction de moments.

Dans toutes les familles, les compteurs contractuels bornés (`num_steps`,
`observation_count`, `payment_count`) utilisent `std::uint32_t`. Les tailles de
workspaces, dimensions globales, offsets, strides et indices mémoire utilisent
`std::size_t`; les dimensions effectivement transmises à `dim3` utilisent
`unsigned int`.

## Paramètres communs des launchers

| Paramètre | Utilité |
|---|---|
| `device_models`, `model_count` | lignes de modèle présentes sur le device |
| `device_curves`, `curve_count` | courbes présentes sur le device, si requises |
| `device_products`, `product_count` | lignes de produit présentes sur le device |
| `device_payment_times`, `device_accrual_fractions` | pools parallèles réservés aux schedules explicitement datés de longueur variable |
| `schedule_size` | nombre d'éléments de chacun des deux pools explicites ; absent du fast path régulier |
| `cartesian_product` | sélection entre construction alignée et produit cartésien |
| `result_count` | taille totale des tableaux de résultats |
| `result_offset`, `launch_result_count` | sous-plage traitée par un batch Monte Carlo |
| `monte_carlo_paths_per_price` | chemins indépendants par prix |
| `day_fraction` | fraction d'année représentée par un jour contractuel, par exemple `1 / 252` |
| `time_day_fraction` | même conversion, nommée explicitement lorsqu'elle s'applique à l'horloge du modèle et jamais aux accruals contractuels |
| `dt` | durée fixe d'une transition élémentaire d'un schéma ; omise pour les incréments exacts aux dates du payoff |
| `simulation_steps_per_day` | nombre de transitions numériques ou de points de monitoring par jour contractuel |
| `threads_per_block` | nombre de threads CUDA par bloc |
| `block_count` | nombre de blocs de la grille persistante ou analytique |
| `base_seed` | origine de la clé déterministe `key = make_key(base_seed + result_index)` |
| `device_prices` | prix FP32 écrits sur le device |
| `device_standard_errors` | erreurs standards FP32, uniquement en Monte Carlo |

Une swaption européenne régulière porte directement `payment_interval`,
`payment_count` et `accrual_fraction` dans sa ligne produit. Son kernel ne
reçoit aucun pool de schedule. La surcharge explicite conserve un
`schedule_offset` et reçoit les deux pools au niveau du dataset ; les deux
chemins spécialisent le même corps de kernel et leurs vues sont résolues à la
compilation.

## Temps, grilles et convention

Un dataset produit porte ses dates contractuelles comme des nombres entiers de
jours. Sa racine JSON et son YAML déclarent une seule fois :

```yaml
time_convention:
  unit: "business_day"
  days_per_year: 252
```

Cette section n'est pas une grille de simulation. Le launcher d'une formule ou
d'une transition exacte reçoit `day_fraction = 1 / days_per_year`, convertit
une date absolue ou un écart exprimé sur cette même horloge en fraction d'année,
puis prépare directement l'intervalle contractuel. Il ne reçoit aucun
`num_steps` artificiel.

Une fraction d'accrual calculée sous une convention de coupon indépendante ne
se déduit pas de cette horloge. Elle est calculée en amont puis stockée
directement en FP32. Pour un schedule explicite, `payment_times` contient les
dates entières du modèle et `accrual_fractions` contient les $`\delta_i`$
contractuels dans deux pools parallèles. Chaque ligne produit porte
`schedule_offset` en `size_t` et `payment_count` en `uint32_t`; aucune taille
maximale statique n'est réservée par ligne. Le loader hôte impose des dates
strictement croissantes, des accruals finis et positifs, l'égalité des deux
longueurs, puis le launcher reçoit la taille commune des pools.

Un dataset de prix réellement discrétisé déclare en plus, dans son propre YAML
et jamais dans le produit :

```yaml
time_grid:
  simulation_steps_per_day: 2
  steps_per_year: 504
  delta_t: "1 / 504"
```

Le générateur transmet alors `dt = 1 / steps_per_year` et
`simulation_steps_per_day`. Pour une échéance de `maturity` jours, le kernel
effectue exactement `simulation_steps_per_day * maturity` transitions. Les
calendriers à schéma stockent de même les écarts comme des nombres entiers de
pas. Changer la convention 252, 360 ou 365 revient à changer la métadonnée et
les constantes du générateur, pas les signatures de dynamique.

Une liste de dates contractuelles simulées par incréments exacts est décrite
comme `observation_schedule`, pas comme `time_grid`. La convention de décompte,
le pas numérique et le calendrier contractuel restent ainsi trois
responsabilités distinctes.

## Invariants d'implémentation

- Conserver l'ordre des lignes, le mapping des seeds et l'ordre des réductions.
- Adresser chaque groupe Philox par `(path_index, local_group_index)` sans
  réservation `groups_per_path`.
- Conserver en FP32 les états, paramètres, primitives analytiques et fonctions
  spéciales du chemin device. Le FP64 device reste réservé aux longues
  réductions Monte Carlo et aux équations de régression explicitement
  documentées ; une série locale de distribution utilise une sommation
  compensée FP32.
- Conserver l'accumulation FP64 des moments Monte Carlo.
- Ne pas introduire de dispatch runtime call/put dans le kernel.
- Ne pas déplacer `PreparedRow` vers une représentation AoS globale des chemins.
- Conserver des fonctions courtes, privées au `.cu`, et un seul launcher public.
- Mettre à jour ce document si le contrat commun évolue.
