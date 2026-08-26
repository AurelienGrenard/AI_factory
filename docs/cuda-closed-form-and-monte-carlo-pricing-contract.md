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
fermée utilise `evaluate_price`. Les deux chemins reçoivent cependant le même
type de vue `DeviceInputs`, qui centralise les tableaux device et le décodage
aligné ou cartésien.

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

## Composition Monte Carlo commune

Les pricers Monte Carlo standards, quel que soit leur marché, utilisent trois
politiques statiques. Rough Bergomi et l'exercice anticipé conservent leurs
kernels spécialisés. Aucun héritage, allocation ou appel virtuel n'entre dans
le kernel : les concepts contrôlent les interfaces, puis le compilateur inline
la composition complète.

```text
DynamicsPolicy<Model>
        |
        v
SchedulePolicy<DynamicsPolicy>
        |
        v
PricingPolicy<SchedulePolicy, Side>
        |
        v
DeviceInputs<Model, Product[, Curve]>
        |
        v
monte_carlo_price_kernel<PricingPolicy>
```

### `DynamicsPolicy`

Chaque `dynamics.cuh/.cu` expose `DynamicsPolicy`. Ses alias minimaux sont
`Parameters`, `RandomContext` et `State`. Un schéma à pas fixe expose en plus
`PreparedDynamics`; un modèle à transition exacte expose `PreparedModel` et
`PreparedTransition`.

Le concept `simulation::DynamicsPolicy` vérifie uniquement les types
path-local. Il n'impose ni spot, ni taux, ni facteur de marché. Les schémas à
pas fixes ajoutent l'interface suivante :

```cpp
static PreparedDynamics prepare_dynamics(
    const Parameters& parameters,
    float delta_t
);
static State initial_state(const PreparedDynamics& dynamics);
static void advance(
    const PreparedDynamics& dynamics,
    std::uint32_t transition_count,
    RandomContext& random,
    State& state
);
```

Les modèles exacts ajoutent les surcharges qui séparent les coefficients
invariants de ceux d'un intervalle, ainsi que `simulate_one_step`. Les concepts
`DynamicsPolicy`, `FixedStepDynamicsPolicy` et
`ExactTransitionDynamicsPolicy` ne vérifient que les capacités réellement
consommées par chaque algorithme. Les accès de marché sont ajoutés séparément :
`equity::SpotDynamicsPolicy` et `equity::LogSpotDynamicsPolicy` imposent les
observables du spot sans contaminer le socle de simulation.

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

Chaque schedule expose `Calendar`, qui décrit les dates contractuelles en
jours, et `TimeConfiguration`, qui décrit leur représentation numérique. Sa
fonction `prepare(parameters, calendar, time_configuration)` produit le
`PreparedSchedule` compact utilisé sur le GPU, tandis que
`validate_time_configuration(time_configuration)` contrôle uniquement les
paramètres temporels propres à cette famille de simulation.

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
schedule. Elle expose `Schedule`, `DeviceInputs`, `ProductParameters`,
`PreparedRow` et les deux points d'extension consommés par le kernel :

```cpp
struct PreparedRow;

static PreparedRow prepare_row(
    const ModelParameters& model,
    const ProductParameters& product,
    const TimeConfiguration& time_configuration
);

static float evaluate_path(
    const PreparedRow& row,
    philox::PhiloxKey key,
    std::size_t path
);
```

`DeviceInputs::prepare_row<Pricing>(result_index, time_configuration)` décode
les indices, charge les lignes de modèle, produit et éventuellement courbe,
puis appelle le `prepare_row` ci-dessus. Le kernel ne connaît donc aucune
forme particulière d'entrée.

`PreparedRow` contient uniquement les états préparés et scalaires nécessaires
à tous les chemins d'un prix. Il reste trivially copyable, puis est stocké une
seule fois en mémoire partagée par bloc. Le kernel impose un budget explicite
`monte_carlo::kMaximumSharedPreparedRowBytes = 2048`; son diagnostic de
compilation demande une `ScheduleView` compacte lorsque la ligne dépasse cette
limite. La clé Philox de la ligne est également préparée une seule fois en
mémoire partagée. Cette disposition évite de conserver `base_seed` et le
calcul de clé dans les registres de chaque thread pendant toute la trajectoire.

Le côté call/put est un paramètre de template lorsque le payoff le demande. Un
produit sans côté ne crée ni template artificiel ni branche runtime.

### `monte_carlo_price_kernel<PricingPolicy>`

Le kernel générique de `common/monte_carlo/monte_carlo_kernel.cuh` :

1. demande à `DeviceInputs` de construire la ligne correspondant au résultat ;
2. stocke cet unique `PreparedRow` dans le bloc ;
3. construit une seule clé `make_key(base_seed + result_index)` partagée par
   le bloc ;
4. distribue les chemins entre les threads ;
5. accumule somme et somme des carrés en FP64 ;
6. réduit les moments et écrit prix et erreur standard en FP32.

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
using PricingPolicy = product::EuropeanOptionPricingPolicy<
    Schedule,
    Side
>;
```

Le launcher public conserve sa signature historique, construit
`ModelProductDeviceInputs`, puis la petite structure de configuration
temporelle, et appelle uniquement `launch_monte_carlo_cuda<PricingPolicy>`.
Les variantes call et put sont explicitement instanciées au bas du fichier.

## Entrées device communes

`common/device_inputs.cuh` fournit trois vues trivially copyable et sans
allocation :

| Vue | Contenu |
|---|---|
| `ModelProductDeviceInputs<Model, Product>` | deux tableaux, leurs comptes et le mode aligné/cartésien |
| `ModelCurveProductDeviceInputs<Model, Curve, Product>` | ajoute un tableau de courbes indépendantes |
| `DeviceInputsWithContext<Inputs, Context>` | ajoute un contexte device, par exemple les pools d'un schedule explicite |

Chaque vue expose `validate(result_count)` sur l'hôte et
`prepare_row<Pricing>(result_index, time_configuration)` sur le device. Cette
couche décrit uniquement où se trouvent les données d'une ligne. Elle ne
connaît ni simulation, ni payoff, ni réduction, ni formule analytique.

Les deux vues primaires acceptent en plus des arguments de préparation
additionnels et restent seules responsables du décodage de leurs tableaux.
`DeviceInputsWithContext` leur délègue donc la préparation avec son contexte ;
il n'inspecte ni alias `CurveParameters`, ni membres internes, et n'ajoute
aucune branche propre à la forme modèle-produit ou modèle-courbe-produit.

## Formules fermées

Les formules fermées suivent une voie parallèle au Monte Carlo : elles ne
possèdent ni `DynamicsPolicy`, ni `SchedulePolicy`, ni contexte aléatoire. Une
politique minimale expose :

```cpp
using DeviceInputs = /* vue commune */;
using TimeConfiguration = /* convention temporelle */;

struct PreparedRow;

static PreparedRow prepare_row(
    const ModelParameters& model,
    const ProductParameters& product,
    const TimeConfiguration& time_configuration
);

static float evaluate_price(const PreparedRow& row);
```

Une politique ajustée à une courbe ajoute simplement le paramètre
`CurveParameters` à `prepare_row`. Une entrée enrichie, comme le pool d'un
schedule de swaption, reçoit en plus le `Context` fourni par
`DeviceInputsWithContext`.

Le concept `closed_form::ClosedFormPricingPolicy` exige des entrées, une
configuration temporelle et une ligne préparée trivially copyable, puis
vérifie la préparation et l'évaluation scalaire. `price_one` impose séparément
le budget d'exécution
`closed_form::kMaximumThreadPreparedRowBytes = 256`, car cette ligne est locale
à chaque thread et peut sinon augmenter les registres ou les spills.

`closed_form_price_kernel<Pricing, GridStride>` possède un seul corps source
et deux spécialisations de compilation :

- `GridStride == false` traite directement un prix par thread, sans boucle ;
- `GridStride == true` parcourt les prix avec un stride égal à la taille de la
  grille lorsque le lancement contient moins de threads que de résultats.

Le launcher choisit la spécialisation directe dès que la grille couvre le
batch, et la spécialisation grid-stride sinon. Ce choix conserve les ressources
du kernel historique dans le cas usuel tout en autorisant une petite grille
persistante. Les deux spécialisations partagent `price_one<Pricing>`, la même
validation et la même signature de lancement.

Une formule fermée dont un prix contient beaucoup de termes indépendants peut
ajouter la capacité `CooperativeClosedFormPricingPolicy`. Elle conserve les
mêmes `DeviceInputs`, `TimeConfiguration` et `PreparedRow`, puis expose :

```cpp
static std::size_t required_shared_memory_bytes(
    std::uint32_t workspace_capacity
);

static float evaluate_price(
    const PreparedRow& row,
    std::byte* workspace,
    std::uint32_t workspace_capacity
);
```

`cooperative_closed_form_price_kernel<Pricing>` attribue un bloc à un prix,
prépare la ligne une seule fois en mémoire partagée, puis fournit le scratch
dynamique à tous les threads. Sa grille est block-stride lorsque le nombre de
blocs est inférieur au batch. Le launcher vérifie la limite de shared memory
du kernel et au moins un bloc résident par SM ; il retourne au chemin scalaire
si cette topologie n'est pas supportée. Le budget statique de la ligne préparée
reste `kMaximumSharedPreparedRowBytes = 256`.

Jamshidian coopératif utilise trois tableaux FP32 de capacité $`N_{\max}`$ :
$`\log A_i`$, $`B_i`$ et les valeurs des options sur zéro-coupon. Les threads
préparent les coefficients et les options indépendantes ; un thread résout la
frontière de Newton à partir des coefficients partagés, puis accumule les
cashflows dans l'ordre contractuel. Cet ordre déterministe évite une réduction
arborescente différente selon la géométrie. Pour un paiement unique,
$`K_1^\star=1/c_1`$ supprime exactement la recherche de frontière.

La capacité maximale est calculée côté hôte lors du chargement du dataset et
transmise une seule fois au launcher. Elle n'est ni répétée dans chaque ligne
produit, ni remplacée par une constante de taille maximale. Une ligne dont le
`payment_count` dépasse la capacité transmise est rejetée sur le device ; une
capacité qui dépasse la shared memory disponible sélectionne le fallback
scalaire.

Les transformations caplet/floorlet et option sur zéro-coupon sont définies
une seule fois dans
`common/fixed_income/bond_option_pricing_policies.cuh`. Les analytics propres
au modèle fournissent les prix call/put sur zéro-coupon ; les modèles ajustés
fournissent seulement la composition modèle-courbe et leur état analytique
initial.

Les launchers de swaptions instancient `SwaptionSide::payer` et
`SwaptionSide::receiver`, préparent leur vue de schedule dans
`DeviceInputsWithContext`, puis utilisent le même kernel fermé. Le chemin
standalone conserve directement la ligne de modèle et son `initial_state` ; le
chemin ajusté appelle la `FittedModelComposition` déjà utilisée par les options
sur taux et zéro-coupon. Aucun adaptateur modèle-produit local n'est nécessaire.
Le côté reste un paramètre de compilation.

Lorsqu'un launcher public est un template dont la définition reste dans le
`.cu`, chaque côté supporté est émis au bas du fichier avec une instanciation
explicite standard `template void launch_...<Side>(...)`. Ne pas forcer cette
instanciation par une variable pointeur, un alias `LaunchSignature` ou une prise
d'adresse artificielle.

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
| `maximum_payment_count` | maximum hôte des longueurs de jambes fixes d'un batch coopératif ; absent des lignes produit et des launchers scalaires |
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
