# Taux stochastique corrélé dans le moteur multi-actifs

`execute_cross_asset_prices<EquityPack,RoughPack,RatePolicy,Product>` diffuse un
panier equity et un taux court sous une seule matrice de corrélation. Le payoff
terminal est multiplié, chemin par chemin, par `exp(-∫r dt)`. Le même kernel
sert aux paniers markoviens, rough et mixtes. Les actifs utilisent
`RateAdjustedModelGroup` et les rough conservent la première passe FFT qui
construit leur volatilité.

## Politiques de taux

| Famille | `CorrelatedRatePolicy` | Facteurs dans R | Transition |
| --- | --- | ---: | --- |
| Taux constant | `model/fixed_income/constant_rate/correlated_dynamics.cuh` | 0 | Intégrale déterministe `rate × Δt` |
| Vasicek | `model/fixed_income/vasicek/correlated_dynamics.cuh` | 1 | Gaussienne exacte conditionnelle à ΔW |
| Ornstein–Uhlenbeck | `model/fixed_income/ornstein_uhlenbeck/correlated_dynamics.cuh` | 1 | Gaussienne exacte conditionnelle à ΔW |
| CIR | `model/fixed_income/cir/correlated_dynamics.cuh` | 1 | Euler à troncature positive, intégrale trapézoïdale |
| Hull–White | `model/fixed_income/hull_white/correlated_dynamics.cuh` | 1 | Facteur OU exact et décalage de courbe |
| CIR++ | `model/fixed_income/cir_plus_plus/correlated_dynamics.cuh` | 1 | Facteur CIR à troncature positive et décalage de courbe |
| G2 | `model/fixed_income/g2/correlated_dynamics.cuh` | 2 | Loi jointe gaussienne exacte conditionnelle aux deux ΔW |
| G2++ | `model/fixed_income/g2_plus_plus/correlated_dynamics.cuh` | 2 | Facteurs G2 exacts et décalage de courbe |

Pour CIR et CIR++, la transition chi-deux exacte du moteur de taux seul ne
représente pas un incrément brownien donné. Le couplage avec les actifs utilise
donc le brownien de R dans un schéma explicite ; ses prix ont une erreur de
pas de temps. Pour les modèles gaussiens, des normales Philox auxiliaires
complètent les ponts browniens conditionnels. Pour G2, la covariance des ponts
croisés est évaluée numériquement par une quadrature de Gauss à huit points,
puis factorisée en 4×4. Leurs domaines Philox ne changent pas avec le nombre
de threads.

Les modèles ajustés à une courbe acceptent les fournisseurs `flat`,
`nelson_siegel` et `svensson`. Le vecteur `curves` est l'argument final du
runner ; `MultiAssetPriceRow::curve_index` y choisit la courbe. Les autres
modèles exigent `curve_index == 0`. `rate_indices[scenario_index]` choisit le
modèle de taux. Le writer `write_multi_asset_price_dataset` accepte un
`rate_model_dataset_path` et les `scenario_rate_indices` : chaque résultat
contient alors `rate_model_id`, en plus des identifiants equity, produit,
corrélation et courbe éventuelle. Chaque ligne exige `discount_factor == 1` pour éviter une
deuxième actualisation.

## Ordre des facteurs

L'ordre externe de R est : facteurs markoviens, facteurs rough (spot puis
volatilité pour chaque actif), puis tous les facteurs du taux. Le runner la
réordonne avant Cholesky dans l'ordre interne : volatilités rough, facteurs
markoviens, facteurs du taux, spots rough restants. Les corrélations spot/vol
propres aux modèles rough et la corrélation x/y de G2 sont vérifiées contre
les paramètres. Toutes les autres entrées de R, notamment spot/taux et
volatilité/taux, sont libres sous la contrainte que R soit une matrice de
corrélation valide. Le Cholesky réordonné est chargé en shared dynamique ; le
lancement vérifie que la matrice et les états préparés tiennent dans la
capacité du GPU et active la capacité étendue si nécessaire.

`RatePolicy` fournit `kBrownianDimension`, `Parameters`, `CurveParameters`,
`prepare`, `initial_state` et `one_step_transition`. Le kernel lui passe ses
normales corrélées et reçoit l'intégrale du taux sur le pas. Il transmet cette
même intégrale aux transitions equity pour assurer un drift et une
actualisation cohérents. La politique ajustée aux courbes partage son calcul
du décalage déterministe entre les trajectoires actives d'un warp.


## Paramètres factorisés pour les nouvelles recettes

Les recettes cross asset peuvent choisir leurs modèles equity dans les bases
`factor_parameters` : elles contiennent les paramètres propres au modèle,
sans `risk_free_rate`, `rho` ni `correlation`. Les bases mono-actif complètes
restent disponibles pour les recettes historiques. Le loader commun insère
uniquement en mémoire des valeurs neutres pour conserver les structures et
les transitions déjà partagées ; le JSON publié ne contient pas ces champs.

`assemble_factorized_scenarios` reçoit les indices des lignes modèle, taux
et corrélation. Une fois par couple `(scénario, matrice R)` utilisé, il impose
`risk_free_rate = 0` aux paramètres equity et déduit les corrélations internes
spot/vol ou G2 depuis le Cholesky de R. Il réutilise ensuite
`execute_cross_asset_prices` et ses kernels. Le taux simulé fournit le drift
commun de tous les actifs et l'intégrale pour l'actualisation. Avec le modèle
`constant_rate`, le même chemin donne un taux déterministe sans facteur
brownien supplémentaire. L'assembleur sépare les indices des taux préparés pour le GPU des indices
originaux nécessaires aux IDs du dataset de prix. Chaque résultat conserve
ainsi les IDs de tous ses modèles, du taux, du produit et de la matrice de
corrélation.

La [recette Black–Scholes + Heston + Vasicek](../../catalog/multi_asset/worst_of_call/bs_heston_vasicek_worst_of_calls_01/recipe.yaml)
illustre les trois sources de paramètres et une matrice à quatre facteurs.
Son ordre externe est `BS.spot, Heston.spot, Heston.variance, rate.factor`.
Les variantes rough passent par le même assembleur : avant la première passe
FFT, les facteurs de volatilité rough sont placés en tête de l'ordre interne ;
la seconde passe rejoue les innovations et avance tous les spots et le taux.

Le produit doit fournir un cashflow terminal (`kTerminalCashflow`). Un produit
à flux intermédiaires doit actualiser chaque flux au moment où il est payé.
