# Contrat CUDA des pricers américains et bermudéens

## Objet

Les produits à exercice anticipé utilisent un moteur Longstaff–Schwartz
multi-bloc commun aux marchés et aux modèles markoviens. Le fichier
modèle-produit ne redéveloppe ni les kernels, ni la régression, ni le planning
mémoire. Il compose quatre politiques statiques :

```text
Dynamics -> Schedule -> PricingPolicy <- ContinuationState
                           |
                           +---- Regressor<Basis>
```

Les concepts vérifient cette composition à la compilation. Il n'existe ni
classe de base, ni appel virtuel, ni allocation dans une trajectoire CUDA.

Les implémentations actuelles couvrent Heston, Bates, Variance Gamma, Normal
Inverse Gaussian, ainsi que les swaptions bermudéennes OU, Vasicek, CIR, G2,
Hull-White et G2++. Elles partagent les mêmes kernels et spécialisent seulement
schedule, état de continuation, payoff et base de régression.

## Calendrier d'exercice

Le produit américain courant contient une maturité `M` et un intervalle
d'exercice `d`, exprimés en jours. Les dates sont ancrées à la maturité. Leur
nombre et le premier stub valent

```math
N = 1 + \left\lfloor\frac{M-1}{d}\right\rfloor,
\qquad
d_0 = M-(N-1)d.
```

Le moteur conserve les états aux `R=N-1` dates précédant la maturité, puis
initialise le cashflow avec le payoff terminal. À l'étape backward `b`,

```math
b=0,\ldots,R-1,
\qquad
j=R-1-b.
```

L'indice `j` parcourt donc les états stockés de la dernière date d'exercice
vers la première. Chaque étape applique l'actualisation sur `d`; la réduction
finale applique l'actualisation du stub `d_0`, puis compare la continuation à
l'exercice immédiat en temps zéro.

`MaturityAlignedExerciseCalendar` représente ce calendrier.
`FixedStepMaturityAlignedExerciseSchedule<Dynamics>` l'exécute avec un schéma
à petit pas ; `ExactTransitionMaturityAlignedExerciseSchedule<Dynamics>`
prépare une transition pour le stub et une transition régulière. Un calendrier
bermudéen co-terminal régulier utilise
`RegularExerciseCalendar {first_exercise_days, exercise_interval_days,
exercise_count}` avec `FixedStepRegularExerciseSchedule` ou
`ExactTransitionRegularExerciseSchedule`. Un calendrier irrégulier pourra
fournir une autre schedule policy sans modifier le moteur Longstaff–Schwartz.

## Politiques statiques

### `EarlyExerciseSchedulePolicy`

Une schedule valide prolonge `simulation::SchedulePolicy` et expose :

```cpp
static std::uint32_t exercise_count(const PreparedSchedule&);

template<typename Handler>
static Dynamics::State simulate(
    const PreparedSchedule&,
    philox::PhiloxKey,
    std::size_t path,
    Handler&
);
```

La schedule décide uniquement quand et comment avancer la dynamique. Elle ne
connaît ni le payoff, ni la base de régression.

### `ContinuationState`

Cette politique définit la projection minimale de l'état à conserver en SoA :

- `SpotAndScaledStateContinuationState` stocke spot et un membre d'état mis à
  l'échelle, utilisé par Heston et Bates pour spot–variance ;
- `SpotLogMoneynessContinuationState` ne stocke que le spot et reconstruit le
  log-moneyness, utilisé par Variance Gamma et NIG.

Elle fournit les descripteurs de champs, la vue du workspace, l'observateur de
simulation et l'entrée du régresseur. Ajouter une variable de continuation ne
modifie pas les kernels.

Les swaptions utilisent `OneFactorRateContinuationState` ou
`TwoFactorRateContinuationState`. Ces projections stockent les facteurs requis
par la régression et l'intégrale du taux requise par l'actualisation pathwise.

### `EarlyExercisePricingPolicy`

La pricing policy relie modèle, produit, schedule et continuation. Elle expose
notamment :

```cpp
static EarlyExerciseRowPlan plan_row(...);
static PreparedRow prepare_row(...);
static float simulate_path(...);
static std::size_t state_index(...);
static float immediate_value(...);
static RegressionInput regression_input(...);
static bool regression_candidate(float immediate);
static double regression_target(...);
static float continued_cashflow(...);
static float initial_continuation_value(...);
static float initial_exercise_value(...);
```

`AmericanOptionPricingPolicy<Schedule, Side, ContinuationState>` implémente ce
contrat pour les calls et puts vanille. `HostInputs` sert au planning du
calendrier avant allocation ; `DeviceInputs` fournit les vues contiguës modèle
et produit au kernel de préparation.

`StandaloneBermudanSwaptionPricingPolicy` et
`FittedBermudanSwaptionPricingPolicy` appliquent le même contrat à un swap
co-terminal. Le sens payer/receiver reste un paramètre de template ; le produit
ne contient aucun booléen de sens.

### `SmallLinearRegressor`

Le régresseur est indépendant du modèle et du produit. Il évalue une base,
accumule les équations normales, réduit les partials, résout les coefficients
et prédit la continuation. La production utilise

```cpp
NormalEquationRegressor<LaguerrePolynomialTwoFactorBasis>
```

afin de reproduire exactement les datasets existants. Les familles disponibles
sont :

- `OneFactorLaguerreBasis<Degree>` ;
- `OneFactorHermiteBasis<Degree>` ;
- `CenteredHingeBasis<KnotSet>` ;
- `LaguerrePolynomialTwoFactorBasis`.

La petite stratégie linéaire accepte au plus huit features. Une base plus
grande ou un réseau de neurones relève d'une autre stratégie d'exécution et ne
doit pas gonfler ce kernel.

## Régression numérique

Pour les chemins sélectionnés, avec vecteur de features
`phi(X_i)` et cible actualisée `Y_i`, le moteur accumule

```math
G=\sum_i \phi(X_i)\phi(X_i)^{\mathsf T},
\qquad
h=\sum_i \phi(X_i)Y_i.
```

Il résout

```math
\left(G+\lambda I\right)\beta=h,
\qquad
\lambda=10^{-10}\frac{\mathrm{tr}(G)}{p},
```

par Cholesky, où `p` est la taille de la base. Les trajectoires, états,
cashflows et features restent en FP32. Les équations normales, réductions,
coefficients et moments finaux sont en FP64. Ce partage de précision est un
invariant du moteur. La prédiction promeut les features vers les coefficients
FP64 et la comparaison exercice/continuation reste en FP64.
`select_exercise_cashflow` porte cette dernière décision : aucun cast FP32 de
la continuation n'est autorisé avant le choix du cashflow.

Chaque résolution produit un `RegressionStatus` typé :

| Statut | Politique backward | Publication |
|---|---|---|
| `success` | utiliser la régression | autorisée |
| `no_candidates` | conserver la continuation | autorisée et diagnostiquée |
| `insufficient_candidates` | conserver la continuation | autorisée et diagnostiquée |
| `non_finite_statistics` | invalider la ligne | interdite |
| `factorization_failure` | invalider la ligne | interdite |
| `non_finite_coefficients` | invalider la ligne | interdite |

`insufficient_candidates` signifie que le nombre de candidats est inférieur
ou égal à `p` : le système empirique n'est pas suffisamment déterminé pour
justifier une décision issue de la régression. Ce cas n'est pas assimilé à une
factorisation réussie. Les coefficients sont remis à zéro et la date est
ignorée explicitement.

Les compteurs par ligne sont agrégés dans
`LaunchResult::regression_diagnostics`. Un statut fatal écrit `NaN` dans le
prix et l'erreur standard. Les générateurs appellent en plus
`validate_regression_diagnostics` avant toute publication afin de rendre la
cause et l'indice de la première ligne fautive immédiatement visibles.

## Kernels partagés

`longstaff_schwartz_kernels.cuh` contient les sept kernels communs :

1. `prepare_rows_kernel` prépare une ligne par résultat ;
2. `simulate_paths_kernel` simule les trajectoires et écrit les états SoA ;
3. `regression_partials_kernel` accumule un partial FP64 par bloc ;
4. `solve_regressions_kernel` résout une régression par prix ;
5. `update_cashflows_kernel` choisit exercice ou continuation ;
6. `moment_partials_kernel` réduit les moments des cashflows actualisés ;
7. `finalize_prices_kernel` écrit prix et erreur standard.

Une grille 2D affecte `blockIdx.y` au prix et `blockIdx.x` à un bloc de chemins.
La boucle backward est orchestrée sur l'hôte afin de synchroniser globalement
les trois kernels de chaque date. `PreparedRow` est partagé par bloc et reste
sous la limite commune de 2 048 octets.

## Workspace et planning

Le workspace est un buffer device contigu découpé en :

- lignes préparées et offsets d'état ;
- un tableau SoA par champ de continuation ;
- cashflows FP32 ;
- partials, coefficients et moments FP64 ;
- statuts typés et diagnostics des régressions.

`make_execution_plan<PricingPolicy, Regressor>` construit un
`EarlyExerciseRowPlan` par résultat, puis délègue à `plan_batches`. Le planner
forme des groupes consécutifs qui tiennent dans la mémoire libre après une
marge `max(1 GiB, 10 %)`. Une ligne qui ne tient pas seule produit une erreur
avant tout lancement.

Le workspace est alloué une seule fois au maximum requis et réutilisé entre
les batchs. Les états restent en VRAM ; les équations normales seules sont
réduites en shared memory. Les diagnostics compacts sont copiés une fois par
batch, après la synchronisation déjà requise par le chronométrage ; aucune
copie hôte n'est ajoutée dans la boucle backward.

## Fichier modèle-produit

Un nouveau couple modèle-produit ne conserve que la composition et le launcher
public :

```cpp
using Schedule = ...<model::DynamicsPolicy>;
using ContinuationState = ...<model::DynamicsPolicy>;

template<OptionSide Side>
using PricingPolicy = product::AmericanOptionPricingPolicy<
    Schedule,
    Side,
    ContinuationState
>;

using Regressor = longstaff_schwartz::NormalEquationRegressor<Basis>;

static_assert(longstaff_schwartz::LongstaffSchwartzPolicy<
    PricingPolicy<OptionSide::call>,
    Regressor
>);
```

Le launcher construit la time configuration et appelle uniquement

```cpp
launch_longstaff_schwartz_cuda<PricingPolicy<Side>, Regressor>(...);
```

Les spécialisations publiques call/put sont instanciées explicitement au bas
du `.cu`.

## Invariants d'extension

- Ne pas recopier les kernels dans un fichier modèle-produit.
- Conserver le workspace SoA et stocker uniquement les états nécessaires.
- Conserver l'ordre Philox `(base_seed + result_index, path)`.
- Conserver FP32 pour les chemins et FP64 pour les sommes/régressions.
- Ajouter une schedule policy pour un nouveau calendrier, sans condition
  runtime dans les kernels existants.
- Ajouter une continuation policy pour une nouvelle projection d'état.
- Ajouter une basis policy pour une autre petite base linéaire.
- Créer une stratégie séparée pour un régresseur volumineux ou entraîné hors
  kernel.
- Vérifier chaque nouvelle composition par concept, test de reproductibilité,
  `compute-sanitizer`, diagnostics de registres et comparaison numérique avec
  une référence.

La méthode suit [Longstaff et Schwartz
(2001)](https://doi.org/10.1093/rfs/14.1.113).
