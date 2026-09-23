# Prix-gradients : bumps absolus de taux de 1 à 10 pb

## Résultat

À précision inchangée (trajectoires et différences par trajectoire FP32,
accumulations FP64), augmenter le bump de `model.risk_free_rate` améliore la
stabilité observée de Heston. **5 pb par côté est une valeur de départ
configurable raisonnable sur cette étude** ; 10 pb donne des écarts encore
plus faibles. Aucun optimum universel ni défaut automatique par modèle n'est
introduit. La configuration exige toujours une sélection et un bump explicites.

| Bump par côté | Écart relatif maximal BS | Écart relatif maximal Heston | Cas Heston au-delà de 6 SE |
|---|---:|---:|---:|
| 1 pb | 0,156 % | 1,429 % | 4/10 |
| 2 pb | 0,148 % | 0,826 % | 4/10 |
| 5 pb | 0,170 % | 0,457 % | 0/10 |
| 10 pb | 0,153 % | 0,225 % | 0/10 |

Les écarts comparent la moyenne de trois seeds au **même stencil représenté**
évalué indépendamment. Ils comprennent bruit d'échantillonnage et erreurs
numériques/discrétisation ; ils ne mesurent pas seuls le biais de ce stencil
vis-à-vis de la dérivée infinitésimale. À 5 pb, le maximum Heston atteint
5,97 SE : l'absence de dépassement de 6 SE ne prouve pas une absence de biais.
À 10 pb, le maximum est 2,65 SE ; pour BS il reste inférieur à 2 SE pour tous
les bumps. Toutes les observations sont conservées.

## Périmètre et méthode

- RTX 4090 Laptop SM89, CUDA 13.3, GCC 14, Release, sans fast math.
- Appel du moteur public : `B=1`, 256 threads, 262 144 trajectoires par prix
  et par seed ; seeds 719, 2719 et 4719, adresse Philox conservée entre bumps.
- Bumps absolus `0.0001 / 0.0002 / 0.0005 / 0.001` sur `r` uniquement.
- Six lignes BS : cas ordinaire, maturité d'un jour, volatilité faible,
  taux négatif, taux 10 % avec maturité deux ans, spot/strike multipliés par 100.
- Cinq lignes Heston : les deux lignes précédemment étudiées (dont `rho=-1`),
  taux négatif, taux 10 % avec maturité deux ans et spot/strike multipliés par 100.
- Deux grilles Heston : `dt=1/504` et `dt=1/1008`, à maturités physiques égales.
  Les grilles ne sont pas supposées partager un brownien couplé entre elles.
- 36 appels de configuration/seed, 192 estimations individuelles, 64 cas
  après agrégation des trois seeds. Le prix central et son erreur standard
  sont strictement identiques entre bumps à ligne, seed et grille fixées.
- Référence BS analytique FP64 ; Heston : solveur Riccati/Fourier existant,
  noyau constant, coupures 320/640 et 3 201/6 401 points. Tous les cas satisfont
  les seuils fixés avant exécution : convergence du prix `2e-7` et de la
  différence finie `1e-3`.
- L'erreur standard agrégée vaut `sqrt(sum(SE_i²))/3`. Les différences entre
  tailles de bump partagent les seeds ; aucune indépendance entre ces tailles
  n'est supposée pour les conclusions.

Cette étude est numérique ; elle ne modifie pas les kernels et ne prétend pas
mesurer un gain de performance. Les références indépendantes sont diagnostiques,
sans publication ou certification d'un dataset catalogue. Le dividende,
les autres modèles et les autres produits ne sont pas qualifiés par ces cas.

## Pourquoi absolu ou relatif

Le mode relatif signifie `h = epsilon * abs(x)` : il adapte le déplacement à
l'échelle du paramètre. Le mode absolu signifie `h = epsilon` dans l'unité du
paramètre. Un point de base est seulement une unité pour un déplacement absolu
de taux (`1 pb = 0.0001`).

Une échelle relative est pratique pour spot, strike ou volatilité positive.
Une échelle absolue reste définie à zéro et peut être préférable pour taux,
variance initiale ou corrélation. Les choix actuels sont des conventions de
recette configurables ; leur pertinence numérique doit être mesurée. Un taux
peut être perturbé relativement et une volatilité absolument si l'appelant le
demande, sous réserve de domaines et de points représentables admissibles.

Exemples :

- `r=0.03`, bump absolu 5 pb : extrémités proches de 0,0295 et 0,0305.
- Volatilité 0,20, bump relatif 0,5 % : extrémités proches de 0,199 et 0,201.
- `rho=0`, bump absolu 0,002 : extrémités -0,002 et +0,002. Un bump purement
  relatif serait nul et serait rejeté.

## Reproduction

```bash
cmake --build build --target study_price_gradients_bumps -j2
./build/study_price_gradients_bumps 262144 --rates > /tmp/gradient-rates.jsonl
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tests/price_gradients/analyze_bumps.py \
  /tmp/gradient-rates.jsonl /tmp/gradient-rates-analysis.json
python3 tests/price_gradients/summarize_rate_bumps.py \
  /tmp/gradient-rates.jsonl /tmp/gradient-rates-analysis.json /tmp/gradient-rates-summary.json
```

Les preuves locales sont conservées dans
`artifacts/price_gradients/2026-09-15-rate-bumps/` (campagne commencée le 15,
rapport finalisé le 16) : plan préalable, sorties brutes, références, résumé,
source et binaire exacts avec leurs empreintes. Le mode sans `--rates` conserve
l'étude précédente de toutes les coordonnées.

Voir le [contrat prix-gradients](../cuda/equity-price-gradients-contract.md) et
le [diagnostic initial](price-gradients-cev-cf-sm89-2026-09-15.md).
