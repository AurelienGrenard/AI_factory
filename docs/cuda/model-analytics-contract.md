# Contrat des analytics CUDA

## Objet

Ce document fixe l'ossature des analytics déterministes de Black-Scholes et
des modèles de taux. Les fonctions publiques propres à un modèle restent dans
`src/model/<asset_class>/<model>/analytics.cuh`; leurs définitions device
incluses restent dans `analytics_impl.cuh`. Les primitives qui ne dépendent
d'aucun modèle vivent dans `src/common`, et les compositions ajustées à une
courbe réutilisent les analytics du processus stochastique de base.

Une API commune est définie par capacité. Aucun modèle ne doit implémenter une
fonction artificielle uniquement pour satisfaire un provider universel.

## Ordre canonique

Un header analytics présente, dans cet ordre :

1. les types publics spécifiques ;
2. la composition modèle-courbe éventuelle ;
3. le short rate et le shift éventuel ;
4. les coefficients affines ;
5. le zéro-coupon et le discounting de chemin ;
6. les options sur zéro-coupon ;
7. les taux forward, taux swap et valeurs de swap.

`analytics_impl.cuh` contient `#pragma once`, les includes, le namespace du
modèle, les primitives propres au modèle, les providers de capacité, leurs
contrôles de concept, puis les wrappers publics dans le même ordre que le
header. Les formules qui assemblent un schedule ou un payoff concret restent
dans le produit propriétaire.

## Signatures fixed income

Les modèles exposent les capacités mathématiquement applicables avec l'ordre
d'arguments suivant :

```cpp
short_rate(parameters, state, time);

log_A(parameters, valuation_time, maturity);
A(parameters, valuation_time, maturity);
B(parameters, valuation_time, maturity);

log_zero_coupon_bond(parameters, state, valuation_time, maturity);
zero_coupon_bond(parameters, state, valuation_time, maturity);

log_discount_factor(parameters, state_integral, time);
discount_factor(parameters, state_integral, time);

zero_coupon_bond_call_price(
    parameters, state, valuation_time, option_expiry, bond_maturity, strike
);
zero_coupon_bond_put_price(
    parameters, state, valuation_time, option_expiry, bond_maturity, strike
);

forward_rate(
    parameters, state, valuation_time, start_time, end_time, accrual_fraction
);
swap_rate(parameters, state, valuation_time, start_time, schedule);
payer_swap_value(
    parameters, state, valuation_time, start_time, fixed_rate, schedule
);
```

Un argument inutilisé par un modèle standalone reste présent afin de préserver
le contrat. Les temps sont des fractions d'année déjà converties. Une fraction
d'accrual est un poids contractuel indépendant de l'horloge du modèle.

Pour un modèle à un facteur, `B` retourne un `float`. Pour un modèle à deux
facteurs, il retourne `TwoFactorAffineBondLoadings`. Cette différence est une
capacité distincte, pas une exception cachée dans une interface universelle.

## Providers fixed income

Les concepts de `common/fixed_income/analytics_concepts.cuh` sont cumulatifs et
minimaux :

- `ZeroCouponBondProvider` fournit `zero_coupon_bond` ;
- `OneFactorAffineBondProvider` ajoute les coefficients `log_A` et `B` ;
- `BondOptionProvider` fournit le contexte et le prix d'une option sur bond ;
- `JamshidianAnalyticsProvider` combine les deux capacités nécessaires à la
  décomposition de Jamshidian.

Les formules de forward, swap et valeur de swap consomment uniquement un
`ZeroCouponBondProvider`. Jamshidian n'est exigé ni de G2 ni de G2++.

## Modèles ajustés à une courbe

Un modèle ajusté expose `compose_fitted_model(model, curve)`. Le type
`FittedModelComposition` est un adaptateur statique vers cette fonction pour
les politiques de pricing génériques.

Hull-White réutilise le processus OU. G2++ réutilise les loadings, covariances
conditionnelles et volatilités d'options sur bond de G2. Les variantes de
courbe ne fournissent que `log_discount_factor(curve, time)` et
`instantaneous_forward(curve, time)` au noyau fitted générique ; elles ne
recopient aucune formule de modèle.

Les formules canoniques d'instantaneous forward Nelson--Siegel et Svensson
exposent deux surcharges distinctes. La surcharge FP32 est `__host__
__device__` et constitue l'unique API atteignable par les analytics CUDA. La
surcharge FP64 est host-only et sert aux contrôles d'extrema et grilles des
générateurs de datasets. Ne pas réintroduire un template `__host__ __device__`
qui rendrait implicitement `double` disponible dans un kernel. Le test
`numerical_robustness_cuda` compare la surcharge device FP32 à son homologue
hôte et la surcharge hôte FP64 à une expression `long double`; les tests fitted
Hull--White et G2++ couvrent les consommateurs device.

## Primitive lognormale

`common/normal_distribution.cuh` contient l'unique CDF normale device.
`common/lognormal_option.cuh` contient le contexte de niveaux actualisés, les
calculs stables de `d1`/`d2`, les prix call/put et la limite à volatilité nulle.
Les options sur bond gaussiennes et les analytics Black-Scholes utilisent ce
même socle.

## Black-Scholes

`model/equity/markovian/black_scholes/analytics.cuh` et `analytics_impl.cuh` contiennent
les seules quantités propres au modèle : contexte Black-Scholes, `d1`/`d2`,
niveaux vanilla,
probabilités cash-or-nothing, termes asset-or-nothing et probabilités
d'intervalles lognormaux. Les politiques produit conservent `PreparedRow` et
réutilisent ces primitives sans redévelopper la distribution.

Les valeurs obtenues après conversion du calendrier portent le suffixe
`_years`, notamment `maturity_years` et `observation_interval_years`.

## Nommage

- `accrual_fraction` pour un poids contractuel déjà converti ; le champ brut
  `accrual_period` du dataset produit reste un nombre de jours et n'entre
  jamais directement dans une formule analytics ;
- `time_to_maturity`, `time_to_expiry` et `bond_tenor` pour une durée ;
- `compose_fitted_model` pour une composition modèle-courbe ;
- `TwoFactorAffineBondLoadings` pour les deux loadings de G2/G2++ ;
- paramètres, états et résultats device en FP32 ;
- pas de formule produit dans un provider de modèle lorsqu'une primitive plus
  basse suffit.

## Tests obligatoires

Chaque provider est contrôlé par `static_assert`. Les tests numériques couvrent
symétriquement les identités applicables : bond et log-bond, coefficient affine,
discounting de chemin, forward, swap, valeur payer, parité call/put, limites
dégénérées et capacités nécessaires à Jamshidian pour les seuls modèles un
facteur. Les schedules et la décomposition complète sont testés depuis le
produit swaption.

Black-Scholes vérifie en plus la parité put-call, les partitions digital et
asset-or-nothing, les payoffs gap/forward-start, l'Asian géométrique et le range
accrual avec plusieurs spots non unitaires.
