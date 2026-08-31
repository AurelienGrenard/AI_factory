# Contrat des pricers CUDA : formule fermée et Monte Carlo

## Objet

Ce document fixe l'ossature des pricers sans exercice anticipé. Les modèles
equity sont rangés par famille mathématique dans
`src/model/equity/<markovian|rough>/<model>/product/<product>.cuh/.cu`; les
modèles de taux restent dans
`src/model/fixed_income/<model>/product/[<curve>/]<product>.cuh/.cu`.
Ces répertoires sont une organisation de sources : ils ne changent ni les
namespaces publics, ni les targets CMake, ni les identifiants de catalogue.
Un nouveau pricer doit reprendre les mêmes couches, les mêmes responsabilités
et la même convention de nommage.

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

La frontière publique conserve des noms de launchers descriptifs et stables :
un symbole linkable appartenant à un modèle répète le modèle, la courbe
éventuelle et l'opération (`launch_heston_european_option_cuda`,
`launch_hull_white_nelson_siegel_rate_option_cuda`). À l'inverse, une primitive
method-neutral déjà qualifiée par son namespace porte seulement son rôle
(`closed_form::launch_closed_form_cuda`,
`monte_carlo::launch_monte_carlo_cuda`). Aucun alias court permanent ne doit
doubler un launcher public de modèle.

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
politiques statiques. Rough Heston et quadratic rough Heston rejoignent cette
composition après la préparation hôte de leur lift markovien. Les modèles à
noyau Volterra gaussien linéaire, actuellement rough Bergomi, log-modulated
rough Bergomi, rough SABR et rough Stein--Stein, utilisent une composition
parallèle à quatre politiques : `HybridKernelPolicy`, `ModelPathPolicy`,
`SchedulePolicy` et `ProductPolicy`. La FFT et la réduction sont uniques ; une
nouvelle dynamique rough n'ajoute que sa transformation de chemin, et un
nouveau produit européen n'ajoute que calendrier, handler et finalisation.
L'exercice anticipé suit une voie parallèle composée d'une schedule,
d'une pricing policy et d'un petit régresseur sur sept kernels
Longstaff–Schwartz partagés. Aucun héritage, allocation ou appel virtuel
n'entre dans le kernel : les concepts contrôlent les interfaces, puis le
compilateur inline la composition complète.

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

Chaque paire `dynamics.cuh` / `dynamics_impl.cuh` expose `DynamicsPolicy`. Ses
alias minimaux sont `Parameters`, `RandomContext` et `State`. Un schéma à pas
fixe expose en plus
`PreparedDynamics`; un modèle à transition exacte expose `PreparedModel` et
`PreparedTransition`.

Le concept `simulation::DynamicsPolicy` vérifie uniquement les types
path-local. Il n'impose ni spot, ni taux, ni facteur de marché. Les schémas à
pas fixes déjà préparés ajoutent `PreparedDynamics`, `initial_state` et
`advance`. `FixedStepDynamicsPolicy` ajoute l'interface de préparation device
suivante :

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

Lorsqu'une préparation est trop coûteuse ou non pertinente sur le device, le
schedule expose aussi `prepare_from_dynamics(prepared, calendar,
time_configuration)`. `PreparedModelProductDeviceInputs` sélectionne alors une
ligne préparée par indice de modèle avant d'appeler la même pricing policy.
Cette voie ne duplique pas les coefficients pour chaque produit d'une
construction cartésienne.

Les produits terminaux appellent `simulate_terminal`. Les produits de chemin
fournissent un observation handler possédant :

```cpp
bool on_initial_state(const State& state);
bool on_observation(std::uint32_t observation, const State& state);
```

Retourner `false` arrête immédiatement le chemin, notamment après une barrière
knock-out ou un autocall.

Les moteurs rough réutilisent le même contrat produit sans imposer un état de
dynamique particulier. Une `PathProductPolicy` canonique fournit son
calendrier, prépare les paramètres du produit, construit son handler et
finalise le payoff. L'adaptateur commun lui présente soit le spot, soit le
log-spot selon `ObservationCoordinate`. Rough Heston et quadratic rough Heston
consomment cette politique par le Monte Carlo N-facteurs préparé ; les quatre
modèles Gaussian-Volterra la consomment dans le kernel hybride FFT. Le payoff
et son calendrier ne sont donc
pas dupliqués entre ces moteurs.

#### Corps produit unique

Chaque produit equity Monte Carlo ordinaire définit exactement un corps
sémantique : sa `PathProductPolicy`. Elle possède le calendrier contractuel,
la préparation des seuls paramètres produit, le handler d'observation et la
finalisation du payoff. La surface publique `*PricingPolicy` n'est qu'un alias
vers `equity::PathProductMonteCarloPricingPolicy<Schedule, PathPolicy>`.
Le moteur markovien, le lift rough N-facteurs et le moteur Volterra FFT
consomment ainsi le même payoff et le même calendrier ; aucun produit ne doit
réintroduire une seconde implémentation de `prepare_row` ou `evaluate_path`.

Une exception n'est recevable que si une spécialisation mesurée démontre un
gain end-to-end sur une architecture cible sans divergence pathwise, sans
modifier le mapping Philox et sans dégrader les budgets de registres, spills,
mémoire locale/shared ou taille de code. L'exception reste bornée au moteur et
au produit mesurés, avec un test de parité face à la `PathProductPolicy`.
`path_product_factorization_cuda` impose l'identité de type des 21 surfaces
publiques avec leur composition canonique et rejoue toutes les catégories de
calendrier/handler sur des graines identiques.

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
La frontière n'est publiée que si son résidu satisfait la tolérance après le
dernier candidat. Une stagnation FP32 ou l'épuisement du budget d'itérations
avec un résidu trop grand retourne `NaN`, qui invalide ensuite la ligne.

La capacité maximale est calculée côté hôte lors du chargement du dataset et
transmise une seule fois au launcher. Elle n'est ni répétée dans chaque ligne
produit, ni remplacée par une constante de taille maximale. Une ligne dont le
`payment_count` dépasse la capacité transmise est rejetée sur le device ; une
capacité qui dépasse la shared memory disponible sélectionne le fallback
scalaire.

Les transformations caplet/floorlet et option sur zéro-coupon sont définies
une seule fois dans `product/rate_option/pricing_policy.cuh` et
`product/zero_coupon_bond_option/pricing_policy.cuh`. Les analytics propres
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
| `device_payment_times_days`, `device_accrual_fractions` | pools parallèles réservés aux schedules explicitement datés de longueur variable |
| `schedule_size` | nombre d'éléments de chacun des deux pools explicites ; absent du fast path régulier |
| `construction` | mode typé commun `PriceConstruction::Aligned` ou `PriceConstruction::CartesianProduct`, partagé par cardinalité host et indexation device |
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

Une swaption européenne régulière porte directement `payment_interval_days`,
`payment_count` et `accrual_fraction` dans sa ligne produit. Son kernel ne
reçoit aucun pool de schedule. La surcharge explicite conserve un
`schedule_offset` et reçoit les deux pools au niveau du dataset. Le loader
transpose les schedules en ELLPACK payment-major : pour la ligne `r`, le
paiement `p` vit à `p * product_count + r`; les cellules de queue inutilisées
sont du padding. Le launcher transmet donc `product_count` comme stride et
`maximum_payment_count` comme capacité coopérative. Tous les modèles à un
facteur évaluent ce chemin explicite avec Jamshidian coopératif; le fast path
régulier reste scalaire. Ce layout conserve l'ordre des lignes et résultats,
coalesce les lectures de threads voisins et borne le scratch par la longueur
maximale réellement chargée.

## Budgets de stockage des policies

Les plafonds sont des budgets de placement, pas des limites métier :

| Objet | Budget | Placement et alternative au dépassement |
|---|---:|---|
| observation handler | 128 octets | état local d'un chemin; déplacer le calendrier variable dans une vue device compacte |
| closed-form `PreparedRow` | 256 octets | état local d'un thread; utiliser une vue ou le kernel coopératif |
| Monte Carlo `PreparedRow` | 2 048 octets | une copie en shared memory par bloc; conserver les données variables dans un pool device |
| Longstaff-Schwartz `PreparedRow` | 2 048 octets | une copie en shared memory par bloc; employer une continuation state et des vues compactes |
| sample prepared input | 2 048 octets | une copie en shared memory par bloc; externaliser les tableaux dans un pool device |

`test_policy_size_budgets_cuda` compile un objet exactement à chaque plafond,
vérifie qu'un handler de 129 octets est rejeté et inspecte les ressources. Les
builds offline CUDA 13.3 donnent, pour les probes locales 128/256 octets puis
le probe shared 2 048 octets, respectivement 64/63/5 registres sur SM75,
40/40/8 sur SM86 et 40/40/8 sur SM89, sans allocation locale supplémentaire
au-delà du payload et avec exactement 2 048 octets de shared memory. Sur le
SM89 mesuré, les cinq probes atteignent 100 % d'occupation théorique. Ces
probes justifient le plafond de stockage; les ressources d'une policy réelle
restent baselinées séparément, car son calcul peut dominer son payload.

Les `static_assert` nomment le budget exact et l'alternative attendue. Toute
augmentation exige de régénérer les probes SM75/86/89 et la baseline runtime
de l'architecture déployée selon
[`performance-regression-protocol.md`](../performance-regression-protocol.md).

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
`simulation_steps_per_day`. Pour une échéance de `maturity_days`, le kernel
effectue exactement `simulation_steps_per_day * maturity_days` transitions. Les
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
- Refuser moins de deux échantillons et des moments non finis. Une variance
  centrée négative n'est ramenée à zéro que dans le budget de cancellation
  FP64 documenté par `reductions::compute_statistics`; sinon prix et erreur
  standard sont tous deux invalidés.
- Ne pas introduire de dispatch runtime call/put dans le kernel.
- Ne pas déplacer `PreparedRow` vers une représentation AoS globale des chemins.
- Conserver des fonctions courtes, privées au `.cu`, et un seul launcher public.
- Mettre à jour ce document si le contrat commun évolue.

### Qualification des moments et de leur finalisation

`reductions::MomentSums` conserve la formation, l'accumulation et la
finalisation des deux moments en FP64. Ce choix couvre les trois consommateurs
du même contrat : Monte Carlo markovien, Volterra FFT et Longstaff--Schwartz.
Il ne constitue pas un réglage propre au SM89 : le test
`monte_carlo_statistics_precision_cuda` compare sur le GPU cible le produit
FP64 direct, la FMA FP64, le carré FP32 promu, le carré FP32 mis à l'échelle et
la somme FP32 compensée, puis les finalisations FP64, mixte, FP32 et host.

La référence indépendante accumule les payoffs FP32 en `long double` sur des
distributions non négatives core, dispersée/stress et à faible variance aux
échelles 100 et 2 048. Le test impose au chemin FP64 une erreur relative de
prix inférieure à `1e-11` et une erreur relative d'erreur standard inférieure
à `2e-4`; il exige aussi que le cas de cancellation rende visible l'échec de
la finalisation FP32. Les timings restent informatifs, car leur ordre dépend de
l'architecture. Ils doivent être relus, avec les registres, spills, shared et
local memory, avant toute spécialisation de précision sur un autre GPU.

Le carré FP64 d'une valeur FP32 est exact avant addition. Descendre ce produit
en FP32 n'est donc permis que pour une famille bornée qui démontre son propre
budget de prix et d'erreur standard sur tout son domaine. La finalisation FP64
n'est exécutée qu'une fois par prix; une accélération isolée de sa racine ne
justifie pas de modifier l'arrondi publié sans gain end-to-end mesurable.

### Résolvante fractionnaire Rough Stein--Stein

`FractionalResolventHybridKernelPolicy` évalue le noyau, les poids de cellules
lointaines et les intégrales de puissance en FP32 compensé. La série de
Mittag--Leffler est utilisée pour `x <= 2`; au-delà, la représentation par
densité de Laplace positive évite la cancellation FP32. Les séries,
quadratures et poids de Simpson accumulent avec `CompensatedFloatSum`. Ne pas
relever ce seuil ni réintroduire du FP64 device sans requalifier ensemble
noyau, intégrales, poids, sorties de trajectoires et coût du kernel complet.

`fractional_resolvent_precision_cuda` balaie
`H in {0.01, 0.03, 0.10, 0.25, 0.45}`, mean reversion dans
`{0, 0.2, 1, 4, 8}`, temps de `1/504` à 7 ans et lags de 2 à 1 008. Une
quadrature `long double` plus fine sert de référence. L'erreur relative
maximale admise est `5e-4` séparément pour le noyau, les deux intégrales et les
poids; la mesure de qualification reste sous `8.3e-5`. Les timings et
ressources du test sont informatifs et doivent être rejoués sur chaque
architecture cible.

La policy Rough Stein--Stein courante déclare
`kUsesVolterraVariance = false`. `prepare` calcule donc exactement deux
intégrales pour les loadings singuliers; `volterra_variance` n'est pas appelée
à chaque pas par les kernels de pricing ou de sampling actuels. Toute nouvelle
path policy qui active cette variance transforme l'intégrale en charge par pas
et exige une nouvelle qualification end-to-end.

### Moyennes de chemin Asian

Les produits Asian arithmétiques et géométriques utilisent
`CompensatedFloatSum` dans leur `PathProductPolicy` unique. Une observation ajoute
un spot FP32 ou un log-spot FP32 avec compensation de Kahan; la division et,
pour la moyenne géométrique, `expf`, restent en FP32. Ne pas réintroduire une
somme FP64 générique ou une somme FP32 simple sans refaire la qualification.

`asian_mean_precision_cuda` balaie 17, 253, 1 765 et 4 097 observations,
faible variance, forte dispersion, grande échelle et cancellation, contre une
référence `long double`. Il compare FP64, FP32 simple, FP32 compensé et chunks
FP32 et borne séparément la coordonnée moyenne et sa valeur publiée. Le payoff
vanille est 1-Lipschitz en cette moyenne : la borne absolue se propage à chaque
payoff puis au prix Monte Carlo, multipliée seulement par le facteur de
discount borné du domaine modèle.

Les mesures de débit et ressources de ce test sont propres au GPU courant et
informatives. Toute nouvelle architecture doit rejouer le test et inspecter
les kernels complets Markov, N-factor et Volterra : registres, spills,
stack/local, shared et occupation priment sur le seul débit du microkernel.

Le closed form Black--Scholes du range accrual applique la même somme
compensée FP32 aux probabilités d'intervalle. Son domaine publié est borné à
1 764 observations et à des probabilités dans `[0,1]`.
`range_accrual_sum_precision_cuda` balaie bornes étroites/larges,
volatilités/taux core et stress, fréquences 1/5/21 jours et maturités jusqu'à
1 764 jours. Il compare FP64, FP32 simple, FP32 compensé, chunks FP32 et une
référence analytique host. Étendre le calendrier ou les bornes exige de
rejouer ce budget et de contrôler le prix final, pas seulement la somme.
