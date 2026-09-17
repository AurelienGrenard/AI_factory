# Prix-gradients : extension CEV, formules fermées et précision des bumps

## Résultat et périmètre

L'extension CEV européenne utilise le moteur MC commun sans modifier ses
kernels ni recopier la transition Milstein. Les calls et puts exposent sept
coordonnées : spot, taux, dividende, sigma, beta, strike et maturité.
Contrairement à BS/Heston, le spot CEV n'est pas factorisé : ses scénarios
portent des états distincts sous les mêmes normales. Les quatre recettes
calls/puts alignées/cartésiennes sont générées depuis le manifeste.

Le défaut MC reste `(B=1, 256 threads)`. Les formules fermées conservent un
thread par prix et ensemble de gradients, avec 256 threads par bloc. La
certification indépendante des datasets et la qualification de toute
l'enveloppe de production restent en attente.

**L'étude de précision identifie des limites réelles aux petits bumps.**
La parité avec l'ancien delta et une faible erreur standard ne prouvent pas
l'exactitude des autres dérivées. En particulier, les petits bumps de taux
Heston appliqués dans chaque pas QE-M ne sont pas qualifiés par cette étude.

## Matériel et protocole

- NVIDIA GeForce RTX 4090 Laptop GPU, SM89 ; CUDA 13.3, GCC 14, Release,
  sans fast math ni limitation imposée des registres.
- Trois campagnes par sonde, cinq préchauffages, 21 répétitions ; médiane des
  trois médianes, p95 maximal et CV médian conservés.
- Préflight et postflight alimentation/concurrence du protocole commun.
- CF : 64 appels par échantillon, durées normalisées par appel ; CEV : un appel.
- Allocations/transferts hors des intervalles mesurés. Les temps de l'API et
  de l'horloge hôte sont conservés séparément.
- Registres, mémoire partagée et mémoire locale runtime contrôlés contre les
  symboles exacts de `cuobjdump`. Les empreintes de sorties CF sont identiques
  entre géométries ; les sorties CEV delta seul sont identiques bit à bit.

Les mesures ne modifient aucun profil historique ni dataset publié.

## Formules fermées Black–Scholes

La matrice comprend `K=0..6`, 1/1 000/65 536 lignes et
64/128/256/512 threads par bloc, soit 84 configurations par campagne.

| K | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Registres par thread | 25 | 34 | 34 | 36 | 36 | 38 | 38 |
| Mémoire locale par thread | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

La boucle est déroulée dans le binaire actuel. Cette montée modérée des
registres ne justifie pas de changer la stratégie pour cette composition.
Aucune variante de boucle non déroulée n'a été introduite.

Pour 65 536 lignes et six gradients :

| Threads | Médiane GPU (ms) | p95 maximal (ms) | CV médian |
|---|---:|---:|---:|
| 64 | 0,050032 | 0,051184 | 1,76 % |
| 128 | 0,050336 | 0,051054 | 0,98 % |
| 256 | 0,049435 | 0,049888 | 0,81 % |
| 512 | 0,050560 | 0,051423 | 0,53 % |

Les écarts de médiane entre ces géométries restent inférieurs à 5 %. Les
petites charges présentent plusieurs CV au-delà du budget ; toutes restent
archivées. Il n'y a pas de preuve d'un optimum universel à 256 threads.

## CEV : compatibilité et coût

Charge : 64 lignes identiques, 8 192 trajectoires par prix, 126 jours,
`dt=1/504`, call, seed 719, 256 threads, `B=1`.

| Moteur | K | Médiane GPU (ms) | p95 maximal (ms) | CV médian |
|---|---:|---:|---:|---:|
| Ancien `price_delta` | 1 | 5,799936 | 6,516736 | 1,31 % |
| `price_gradients` | 1 | 6,007808 | 6,214656 | 1,13 % |
| `price_gradients` | 4 | 13,248512 | 13,413376 | 1,24 % |
| `price_gradients` | 7 | 19,717953 | 20,146175 | 0,72 % |

Le coût médian delta seul augmente de 3,58 %, sous le seuil de 5 % du protocole.
Le kernel passe de 73 à 92 registres, de 80 à 144 octets de mémoire partagée
statique, et de zéro à 40 octets de pile locale par thread. Ces 40 octets sont
confirmés dans le binaire ; ils ne sont pas présentés comme une absence de
mémoire locale. L'extension reste une géométrie candidate, pas une certification
sur tous les paramètres, maturités ou GPU.

Contrôles réalisés :

- Calls/puts, `K=0..7`, lots 1/2/4 et blocs de 64/128/256 threads ;
  prix, delta et erreurs standards identiques à l'ancien moteur.
- Sélections isolées et réordonnées, lots résiduels, grille limitée à un bloc,
  offsets de lignes, construction cartésienne et expansion alignée équivalente.
- Régime absorbant, frontière beta à 0,5 et proche de 1 ; domaines invalides
  rejetés côté hôte.
- `memcheck`, `racecheck`, `initcheck`, `synccheck` : zéro erreur sur la matrice
  CEV réduite à 513 trajectoires, complétant les tests ordinaires à 4 097.
- Générateur natif sur deux lignes temporaires, quatre gradients, `2^20`
  trajectoires ; artefacts contrôlés et reprise complète du checkpoint identique.
- Tests BS/Heston, six contrôles hôte/codegen/layout, 30 tests Python de codegen
  et 46 tests Python de datasets réussis.

## Étude des bumps

La sonde utilise 262 144 trajectoires, trois seeds et cinq facteurs de bump
`0,125 / 0,5 / 1 / 2 / 4`. Elle conserve tous les scénarios et stencils
représentés. BS comprend trois lignes : cas ordinaire, maturité d'un jour et
volatilité faible. Heston comprend un cas ordinaire et un cas à `rho=-1`,
avec `dt=1/504` et `dt=1/1008`.

Les paramètres autres que la maturité utilisent ces facteurs sur les bumps
explicites de la sonde. La maturité utilise 1/1/2/4/8 pas de `1/504`, conservés
à temps physique identique lors du raffinement de la grille Heston.

Les 870 estimations individuelles sont agrégées par trois seeds en 290 cas.
L'erreur standard de cette moyenne est `sqrt(sum(SE_i²))/3`. Les comparaisons
entre pas de temps ne supposent pas un couplage brownien entre grilles : leurs
préfixes Philox ne constituent pas à eux seuls un tel couplage.

### Références indépendantes

- BS : formule et dérivées en FP64 côté hôte. Le biais du stencil est séparé de
  l'écart entre formule FP32 et même stencil FP64.
- Heston : solveur Riccati/Fourier existant dans
  `validation/volterra/rough_heston.py`, avec un facteur, nœud zéro, poids un,
  variance initiale explicite et dérive constante `kappa*theta`. Aucun pas QE-M
  ni moteur externe Premia/QuantLib n'est appelé.
- L'essai initial de coupures 160/320 échoue au seuil de convergence sur la
  ligne frontière ; il est conservé. Le raffinement 320/640, 3 201/6 401 points,
  respecte les seuils inchangés : écart de prix maximal `5,62e-8 < 2e-7` et
  écart de gradient maximal `3,57e-5 < 1e-3`.

Ces seuils définissent la résolution de cette étude, pas une certification
indépendante core/stress du catalogue.

### Observations

1. **Maturité très courte.** À un jour, `h=1/504` produit pour BS une dérivée
   de maturité de 0,655629 contre 0,633289 analytiquement : biais de **+3,53 %**.
   Les stencils unilatéraux utilisés pour des bumps plus grands restent
   admissibles mais ne suppriment pas ce biais. Le minimum `dt` n'est donc pas
   un critère suffisant de précision.
2. **Annulation FP32.** Pour la ligne BS à volatilité 0,005, le plus petit bump
   relatif de volatilité produit un écart de dérivée CF d'environ 0,01507
   vis-à-vis du même stencil en FP64. Réduire systématiquement epsilon est une
   mauvaise règle de précision. Les 90 cas MC BS sont tous à moins de six
   erreurs standards du stencil de référence, ce qui ne supprime pas son biais.
3. **Petits bumps Heston.** Vingt des 200 cas agrégés dépassent six erreurs
   standards du stencil de référence. Ils restent explicitement des écarts,
   sans élargissement rétrospectif du budget ni changement de référence.
   Raffiner `dt` ne garantit pas une amélioration des petites différences FP32.
4. **Diagnostic spécifique aux taux.** La sonde de contrôle réutilise exactement
   les trajectoires centrales canoniques, puis applique la variation
   déterministe de carry et d'actualisation au terminal, en FP64. Ses 40 cas
   sont à moins de **1,21 erreur standard** de la référence. Exemple à
   `rho=-1`, `dt=1/1008`, `h_r=1,25e-5` : le moteur actuel donne 0,318815,
   le contrôle 0,294133 et la référence 0,294123. Cela localise l'écart dans
   le calcul des différences de trajectoires/payoffs perturbés, plutôt que
   dans un défaut de la référence ou dans le bruit d'échantillonnage.

La sonde de contrôle reste un diagnostic ; elle n'est pas un nouveau moteur
production et ne modifie pas la parité historique. La préparation d'une
transformation terminale stable pour les coordonnées qui le permettent est
la prochaine correction à étudier. Les autres coordonnées, dont rho, exigent
leur propre analyse : ce résultat sur les taux ne les qualifie pas.

## Reproduction et preuves

```bash
cmake --build build --target price_gradient_generators \
  test_price_gradients_cev_cuda benchmark_price_gradients_closed_form \
  benchmark_price_gradients_cev study_price_gradients_bumps \
  study_price_gradients_heston_rates -j2
AI_FACTORY_CUDA_KERNEL_DIAGNOSTICS=1 ./build/benchmark_price_gradients_closed_form 256
AI_FACTORY_CUDA_KERNEL_DIAGNOSTICS=1 ./build/benchmark_price_gradients_cev
./build/test_price_gradients_cev_cuda
./build/study_price_gradients_bumps > /tmp/price-gradient-bumps.jsonl
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tests/price_gradients/analyze_bumps.py \
  /tmp/price-gradient-bumps.jsonl /tmp/price-gradient-bumps-analysis.json
./build/study_price_gradients_heston_rates > /tmp/price-gradient-heston-rates.jsonl
```

Les preuves locales sont dans
`artifacts/price_gradients/2026-09-15-cev-cf/` : trois campagnes de chaque sonde,
diagnostics et inventaires compilés, résultats/stencils bruts, références aux
deux résolutions, comparaison de contrôle des taux, logs de tests et de
sanitizers, checkpoint temporaire, scripts, binaires et empreintes de sources.
Le contrat durable reste celui des
[prix-gradients sélectionnés](../cuda/equity-price-gradients-contract.md).
