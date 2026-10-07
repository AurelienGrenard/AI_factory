# NUM-030 — domaine martingale et prix stress log-modulated rough Bergomi

Reprise du **2026-10-07**. Le constat reste **ouvert** : les prix stress exclus
ci-dessous n'ont pas de valeurs de remplacement indépendamment certifiées.

## Résultat sur le modèle continu

Le code pose `v_t = ξ₀ exp(ηY_t − η² Var(Y_t)/2)` avec
`Y_t = ∫₀ᵗ K(t−s)dW_s` et
`K(u) = C u^(H−1/2) max(ζ log(1/u), 1)^(−p)`.
Le spot satisfait `dS_t/S_t = (r−q)dt + √v_t dB_t`, avec corrélation `ρ`
entre `B` et `W`. Ce sont le noyau et la volatilité du modèle
[log modulé de Bayer, Harang et Pigato](https://arxiv.org/abs/2008.03204),
avec `ξ₀>0`, `η>0`, `0≤H<1/2`, `ζ>0`, `p>1` pour les 1 000 paramètres publiés.
La recette échantillonne `ρ` dans `[-0,95,−0,2]` sur les 900 lignes core
et dans `[-0,99,0,5]` sur les 100 lignes stress ; 42 stress ont `ρ>0`.

**Application du théorème de Gassiat, pas énoncé direct de l'article sur le
noyau log modulé.** Choisir `α` dans `(H+1/2, 1)`. Quand `u→0`,
`K(u)/(αu^(α−1))` est proportionnel à
`u^(H+1/2−α)/log(1/u)^p` et tend vers `+∞`. Il existe donc `T₀>0`
tel que `K(u)≥αu^(α−1)` pour `0<u≤T₀`. Sur cet intervalle,
`σ(t,y)=√ξ₀ exp(ηy/2−η² Var(Y_t)/4)` minore une constante positive
fois `exp(ηy/2)`. Cette fonction satisfait la condition de croissance de
[l'énoncé 1(2) de Gassiat](https://arxiv.org/html/1811.10935v3).
Il s'ensuit que `exp(−(r−q)t)S_t` est une **martingale locale stricte**
pour `ρ>0`. Pour `ρ≤0`, l'énoncé 1(1) donne une vraie martingale.
Cette conclusion concerne le modèle continu, et suppose le noyau et la
variance déterministe prévus par ces équations.

La parité de référence `C−P=S₀ exp(−qT)−K exp(−rT)` présuppose le premier
moment martingale. Avec `ρ>0`, le premier terme correct est
`exp(−rT) E[S_T]`, strictement inférieur à `S₀ exp(−qT)` dans le modèle
continu. Les huit écarts publiés ont tous un signe négatif et `ρ>0`.
Leur signe est cohérent avec ce résultat ; **leurs amplitudes ne mesurent pas
à elles seules le défaut martingale continu**.

Le rough SABR publié comporte aussi treize lignes `ρ>0`, mais elles ont toutes
`β<1`. Son spot absorbé reste une vraie martingale dans le modèle continu ;
la [preuve et les contrôles des lignes publiées](sabr-martingale-domain.md)
expliquent pourquoi le diagnostic NUM-030 ne s’y transpose pas. Le
[gate des domaines de référence](martingale-reference-datasets.md) contrôle
aussi les 15 lignes `ρ>0` du rough Bergomi standard.

À **grille finie**, le pas log Euler utilise une variance connue au début du
pas. En arithmétique exacte, son facteur exponentiel de spot a une espérance
conditionnelle égale à un : le schéma discret conserve donc le forward en
espérance exacte. Sa limite peut perdre cette propriété en l'absence
d'intégrabilité uniforme. Sur une grille finie, un écart empirique au forward
mesure aussi la difficulté d'échantillonner une queue très rare. On ne doit
ni annoncer un prix continu à partir du seul écart de parité ni utiliser la
SE des chemins observés pour certifier la masse non observée.

## Replays du worktree courant

Build isolé Release SM89/CUDA 12.9.41 sous
`/tmp/ai_factory_num030_20261007_build`. Les 20 exécutions, seeds,
hashes des binaires, sources et entrées sont dans la
[preuve brute](../../work/catalog/qualifications/num030-lmb-stress-20261007-v1/current_worktree_replays.json).
Les deux binaires et le cache CMake ont été archivés localement sous
`artifacts/audit/num030-2026-10-07/` avec empreintes dans le
[manifeste](../../work/catalog/qualifications/num030-lmb-stress-20261007-v1/manifest.json).

| Ligne | ρ | Forward actualisé | Moments conditionnels actualisés, deux graines | Contrôle ρ=0 | Écart FFT/direct maximal |
|---|---:|---:|---:|---:|---:|
| 948, 262 144 chemins, 2 pas/jour | 0,431 | 0,927773 | 0,920259 ± 0,000169 ; 0,920240 ± 0,000151 | 0,927772 | 2,86×10⁻⁶ |
| 993, 262 144 chemins, 1 pas/jour | 0,444 | 0,998961 | 0,942290 ± 0,001642 ; 0,943491 ± 0,002815 | 0,998958 | 3,34×10⁻⁶ |

Le contrôle FFT/direct porte sur huit chemins et huit dates par exécution ;
il ne certifie pas chaque chemin des jeux publiés. En complément, une
[quadrature FP64 indépendante](../../work/catalog/qualifications/num030-lmb-stress-20261007-v1/kernel_quadrature.json)
a comparé `scipy.integrate.quad` aux 96 points de la transformation rationnelle
du noyau sur les **huit** lignes signalées : erreur relative maximale
**4,32×10⁻⁶** sur normalisation, cellule proche et variance terminale.
Cela borne l'erreur de cette formule de quadrature en FP64 dans ces cas ;
ce contrôle ne mesure pas l'arrondi FP32 du kernel CUDA. La réduction de
variance conditionnelle intègre le bruit spot indépendant, mais utilise le
même générateur du bruit rough. Les deux contrôles `ρ=0` reviennent au forward à
1–4×10⁻⁶ près, dominés par l'arrondi FP32. Aucun chemin non fini n'a été
observé. Le scan historique a inspecté les 900 lignes core publiées sans
signal au seuil commun, et 20 lignes core ont été rejouées sur le binaire
du passage précédent.

Les six payoffs stress historiquement très incertains ont aussi été rejoués
sur le binaire courant avec deux graines à 262 144 chemins. L'Asian call 988
varie de **0,014933 à 0,098424**, le lookback 960 de **0,092252 à 0,191784**.
Les replays à 1 048 576 chemins et graine publiée reproduisent exactement
les outliers historiques : Asian call 988 **0,036931 ± 0,021093** et
up-and-in call 954 **0,068462 ± 0,017760**. La ligne 954 a `ρ<0` : sa
SE élevée relève de la queue du payoff, pas du défaut de martingale à
corrélation positive. Une campagne supplémentaire a exécuté **16 graines indépendantes à
1 048 576 chemins** pour chacun des six payoffs, soit 96 prix sur le même
binaire courant. Le [résultat brut](../../work/catalog/qualifications/num030-lmb-stress-20261007-v1/stress_multiseed_1m.json)
est hashé dans le manifeste de qualification.

| Payoff / ligne | Médiane des 16 prix | Minimum–maximum | Écart type entre graines | RMS des SE internes |
|---|---:|---:|---:|---:|
| Asian call 988 | 0,016983 | 0,015099–0,036931 | 0,006321 | 0,006568 |
| Gap call 997 | 0,041457 | 0,039118–0,063863 | 0,006111 | 0,006233 |
| Lookback 936 | 0,514896 | 0,483831–1,150338 | 0,191917 | 0,209017 |
| Lookback 960 | 0,097607 | 0,070270–3,747326 | 0,912468 | 0,889839 |
| Up-and-in call 954 (`ρ<0`) | 0,050543 | 0,050330–0,068462 | 0,004485 | 0,004442 |
| Up-and-in call 960 | 0,016904 | 0,015434–0,056313 | 0,009967 | 0,009414 |

Les SE internes deviennent grandes **quand** un événement extrême est
observé ; leur RMS concorde avec la dispersion entre ces seize graines.
Une graine qui ne voit pas l'événement peut pourtant publier une SE très
petite et un prix sensiblement différent. La moyenne des seize graines
reste dominée par quelques queues et ne certifie pas une limite stable.
Les quantiles historiques sont dans la
[preuve de 2026-10-06](../../artifacts/audit/rough-price-quality-2026-10-06/result.json).

## Décision de publication et preuve manquante

La [qualification versionnée](../../work/catalog/qualifications/num030-lmb-stress-20261007-v1/manifest.json)
marque `not_certified` les 42 lignes stress à `ρ>0` dans chacun des 29
jeux de prix, ainsi que l'up-and-in call 954 à `ρ<0` : **1 219 lignes de
prix/SE** au total. Les prix et reçus `generation.yaml` restent inchangés.
Le [vérificateur](../../maintainer/tools/datasets/verify_num030_qualification.py)
contrôle les 29 paires JSON/reçu, les 20 replays ciblés, les 96 replays
multi-graines, la quadrature indépendante, les hashes et le masque.
Une ligne non masquée ne reçoit aucune certification nouvelle par ce contrôle.

Pour clore NUM-030 comme problème de qualité des **prix**, il manque encore
une estimation indépendante et stable des queues et des prix sur les huit
lignes européennes stress, les cinq payoffs à `ρ>0` et l'up-and-in 954 à
`ρ<0`, ou une décision de domaine qui retire réellement ces sorties de la
capacité de référence. Si un estimateur ou le domaine de production change,
réexécuter uniquement les jeux touchés avec reçus authentiques. Ne pas
écrêter les payoffs extrêmes.
