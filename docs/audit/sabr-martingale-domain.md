# SABR et rough SABR — domaine martingale

Revue du **2026-10-07**. La corrélation positive ne suffit pas à conclure
qu'un spot SABR perd sa propriété de martingale : l'exposant `β` compte.

## Modèle continu

Pour le rough SABR implémenté,
`dS_t = (r−q)S_t dt + α_t S_t^β dZ_t`, avec absorption en zéro,
`0,5≤β≤1` et
`α_t = √ξ₀ S₀^(1−β) exp(ηY_t/2 − η²t^(2H)/4)`.
Le processus gaussien de Volterra `Y` a variance `t^(2H)` ; `α_t` possède
donc des moments de tout ordre, uniformément sur chaque horizon fini.
La convention du modèle est décrite dans le
[README rough SABR](../../src/model/equity/rough/rough_sabr/README.md)
et dans [Fukasawa–Gatheral](https://arxiv.org/abs/2105.05359).

Pour `β<1`, le spot actualisé absorbé est une **vraie martingale pour tout
`ρ∈[−1,1]`**. Voici l'argument pour le modèle continu de ce dépôt : arrêter
le spot actualisé `X` à `τ_n`, choisir `q>1/(1−β)`, puis appliquer Hölder à
`E[α_t² X_{t∧τ_n}^(2β)]`. Comme
`2βq/(q−1)<2` et `sup_{t≤T}E[α_t^(2q)]<∞`, l'inégalité de Young donne
`E[α_t² X_{t∧τ_n}^(2β)]≤C_T(1+E[X_{t∧τ_n}²])`.
Itô et Gronwall bornent `sup_n sup_{t≤T} E[X_{t∧τ_n}²]` ; les martingales
arrêtées sont donc uniformément intégrables. Cette démonstration utilise
la volatilité exogène lognormale du modèle et ne dépend pas du signe de `ρ`.
Le SABR markovien à volatilité lognormale obéit au même argument ; voir aussi
la [démonstration de Jourdain](https://citeseerx.ist.psu.edu/document?doi=7eb0367e88b4b83734eb45cf1311fb783523f42c&repid=rep1&type=pdf).

À **`β=1`**, le rough SABR de ce dépôt coïncide avec son rough Bergomi :
`dS_t/S_t = α_t dZ_t`, `α_t` exponentielle du conducteur de Volterra.
L'application du [théorème de Gassiat](https://arxiv.org/html/1811.10935v3)
donne une martingale locale stricte pour `ρ>0` et une vraie martingale pour
`ρ≤0`, avec `η>0` et le noyau rough indiqué dans le README. Il s'agit
d'une **application** du théorème au noyau et à la paramétrisation du dépôt.
La même séparation `β<1` / `β=1` vaut pour le SABR markovien avec
volatilité lognormale non dégénérée.

## Lignes publiées et schéma numérique

Le [fichier de paramètres rough SABR](../../work/catalog/archives/parameter-catalog-20261007/frozen_inputs/model/equity/rough/rough_sabr/parameters/rough_sabr_01.json)
contient 1 000 lignes, toutes avec `β<1` (maximum `0,999943733`). Treize
lignes stress, aucune core, ont `ρ>0` ; leur `β` maximal est `0,987741113`.
Sur les [calls](../../work/datasets/releases/num029-lamperti-20261006-v1/model/equity/rough/rough_sabr/prices/european_calls/rough_sabr_01__european_calls_01__01.json)
et [puts](../../work/datasets/releases/num029-lamperti-20261006-v1/model/equity/rough/rough_sabr/prices/european_puts/rough_sabr_01__european_puts_01__01.json)
de la release NUM-029, les treize résidus de parité divisés par la SE
combinée ont une valeur absolue maximale de **2,109**. Cette observation
est compatible avec la parité ; elle ne certifie pas à elle seule chacun
des prix publiés.

Le [pas CUDA](../../src/model/equity/rough/rough_sabr/dynamics_impl.cuh)
prend sa branche lognormale lorsque `1−β<10⁻⁴`, même si `β<1`.
Une seule ligne publiée entre dans cette branche : **419**, avec
`β=0,999943733` et `ρ=−0,785906196`. Aucune des treize lignes à
`ρ>0` ne l'utilise. Cette approximation de grille ne doit pas être
confondue avec la propriété du modèle continu. Le
[gate de datasets](martingale-reference-datasets.md) laisse ces treize
lignes dans le domaine martingale et refuserait un futur `β=1, ρ>0` sans
qualification explicite.

Le déficit de premier moment du SABR markovien documenté dans
[NUM-037](unresolved-closures.md) ne peut donc pas être déclaré propriété
du modèle continu pour ses `β<1`. Il reste un problème d'estimation ou de
discrétisation à résoudre. À l'inverse, les lignes `ρ>0` du log-modulated
rough Bergomi dans [NUM-030](num030-martingale-domain.md) sont
lognormales en spot (`β=1`) : la parité forward ordinaire n'y est pas la
référence continue correcte.
