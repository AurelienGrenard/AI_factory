# Qualification des stratégies CUDA `price_gradients` sur SM89

## Décision

Le profil représentatif conserve `B=1` et 256 threads pour le Monte Carlo
terminal et le replay américain. La stratégie coopérative Jamshidian conserve
128 threads et un bloc par prix. Les quinze symboles CUDA actifs de ce
périmètre ont un diagnostic de ressources et un profil Nsight Compute : aucun
spill compilateur local ou partagé et aucune phase avec des wavefronts shared
excessifs n'ont été observés.

Cette décision porte sur les implémentations présentes le 19 septembre 2026 :
Monte Carlo terminal exact et à pas fixe, closed form par ligne, LSM Heston à
exercice gelé, et Jamshidian CIR scalaire et coopératif. Elle ne qualifie ni un
autre GPU, ni les futurs gradients rough, FFT ou N-facteurs, ni toutes les
tailles de datasets.

## Matériel et protocole

- GPU : NVIDIA GeForce RTX 4090 Laptop, compute capability 8.9, 76 SM,
  17 170 956 288 octets de mémoire ;
- pilote CUDA 13.2, runtime 13.3, NVCC 13.3.73 ;
- build Release sans fast math ;
- trois campagnes complètes, 21 mesures chacune ;
- agrégation : médiane des trois médianes, maximum des p95 et médiane des CV ;
- seuil de bruit kernel : CV agrégé au plus égal à 5 %, donc au moins deux
  campagnes recevables sur trois ;
- les opérations très courtes sont regroupées dans une fenêtre déclarée et
  les durées restent normalisées par opération ;
- la pipeline LSM utilise 20 warmups et quatre pipelines par échantillon. Son
  appel public contient 57 lancements, ce regroupement évite de confondre le
  bruit d'ordonnancement avec son temps de calcul.

Les trois campagnes brutes sont conservées localement sous :

- `artifacts/audit/price-gradients-strategies-sm89-2026-09-19-qualified-exact-c1` ;
- `artifacts/audit/price-gradients-strategies-sm89-2026-09-19-qualified-exact-c2` ;
- `artifacts/audit/price-gradients-strategies-sm89-2026-09-19-qualified-exact-c3`.

Leurs `summary.json` portent respectivement les SHA-256
`87a44dc06d69654134dc2ad64f1d3d7de1abda6e1f44d5060720878afcc36e8d`,
`9a2417bf20be6944b9408e19eecfb85b789481fbb733d9914b6fca3bed4b9740` et
`556cf1ab89717d014efde8412c79c6295b2d8414f7ac9a028df08e9b1113e588`.
L'agrégat local porte le SHA-256
`2f126a35e39173bd28e86cbe9a1363fa2e78dd7755fd597ffbb2b20f800a4b67`.
Ces répertoires sont des preuves locales ignorées par Git ; le présent rapport
en conserve les résultats et les commandes de reproduction.

Les quatre binaires mesurés portent les SHA-256 `d629f3d5…31836` (MC),
`ae3e3bfb…6f439` (closed form), `6ff54391…2694b` (américain) et
`219570b3…a9e55` (Jamshidian). Le profil et les trois campagnes référencent
exactement ces empreintes.

## Couverture permanente

Le manifeste
`maintainer/tests/performance/price_gradients/strategy_manifest.json` possède le SHA-256
`04301666fc3e2405fde42a5a3082fe2148193a8d2d6996f7efb81e498d4094ee`.
Il exige les phases suivantes :

| Stratégie | Symboles profilés | Géométrie représentative |
|---|---:|---|
| MC terminal Black--Scholes, Heston, Merton | 3 | `B=1`, 256 threads |
| Closed form Black--Scholes | 1 | un thread par ligne, 256 threads/bloc |
| Heston américain gelé | 9 | géométrie propre à chaque phase, replay à 256 threads |
| CIR Jamshidian scalaire et coopératif | 2 | 256 threads/grille stridée et 128 threads/prix |

Le runner refuse un symbole requis absent, conserve les diagnostics de
lancement, lie le profil au SHA-256 du binaire, filtre le SASS aux symboles
effectivement lancés et exporte les métriques Nsight en CSV et JSON. Une phase
active ajoutée au manifeste ne peut donc pas disparaître silencieusement de la
qualification.

## Temps représentatifs

| Charge la plus lourde représentée | Médiane kernel | p95 maximal | CV agrégé | Campagnes sous 5 % |
|---|---:|---:|---:|---:|
| Black--Scholes closed form, 65 536 lignes, `K=6` | 0,0515 ms | 0,0531 ms | 3,43 % | 3/3 |
| Black--Scholes terminal MC, 64 lignes, `K=6` | 0,0910 ms | 0,1065 ms | 2,15 % | 3/3 |
| Heston terminal MC, 64 lignes, `K=10` | 60,6710 ms | 60,8635 ms | 0,60 % | 2/3 |
| Merton terminal MC, 64 lignes, `K=7` | 0,1734 ms | 0,1741 ms | 0,60 % | 3/3 |
| Heston américain gelé, 4 lignes, `K=9`, 262 144 chemins | 91,3311 ms | 91,7519 ms | 1,89 % | 3/3 |
| CIR Jamshidian scalaire, 1 000 lignes, `K=7` | 5,2941 ms | 6,8035 ms | 0,57 % | 2/3 |
| CIR Jamshidian coopératif, 1 000 lignes, `K=7` | 1,2114 ms | 1,6628 ms | 1,44 % | 3/3 |

Sur cette fixture Jamshidian, la version coopérative est 4,37 fois plus rapide
que la version scalaire. Cela justifie son maintien pour les schedules longs ;
la voie scalaire reste utile aux petites formules et comme comparaison
fonctionnelle.

## Registres, mémoire et accès

| Famille | Registres/thread maximaux | Local/thread maximal | Observation |
|---|---:|---:|---|
| MC terminal | 124 | 56 octets | maximum sur les spécialisations représentées |
| Closed form par ligne | 38 | 0 | aucune allocation locale |
| Pipeline américaine | 128 | 32 octets | maximum sur le replay gelé |
| Jamshidian | 63 | 0 | scalaire et coopératif sans local |

Le SASS contient des accès locaux dans certains kernels MC et cinq
instructions locales dans le replay gelé. Les compteurs Nsight rapportent
cependant zéro requête de spill compilateur local et partagé dans chacune des
quinze phases : ces allocations font partie du stockage local explicite du
kernel et ne constituent pas une preuve de débordement de registres.

Les lectures de paramètres du MC terminal affichent environ quatre octets par
secteur parce que chaque warp diffuse le même scénario et les mêmes paramètres.
Ce profil est attendu pour des données uniformes par bloc. Les phases
américaines qui parcourent l'état des chemins atteignent environ 13 à 31
octets par secteur selon la phase, ce qui confirme que les threads adjacents
accèdent aux chemins adjacents. Les quelques conflits shared relevés dans la
simulation et les réductions ne produisent aucun wavefront shared excessif.
Les branches divergentes des régressions et de la mise à jour des cashflows
suivent les décisions d'exercice ; le replay gelé atteint 99,88 % de cibles de
branche uniformes.

La fixture américaine réserve 155 386 004 octets suivis pour entrées,
workspace transitoire et sorties, avec une seule tranche de quatre prix. Le
planner de production continue de borner le nombre de lignes par la VRAM libre
et inclut les `2*K*Bp` moments FP64 du replay.

## Contrôle numérique

Toutes les sorties des benchmarks sont finies et stables pour une configuration
donnée. La comparaison Jamshidian utilise la même erreur de prix FP32 propagée
à travers le stencil réellement représenté : l'écart relatif maximal sur le
prix vaut `1,07009e-6` et l'écart de gradient maximal consomme 3,20 % de sa
borne. Ce contrôle distingue l'amplification normale de l'arrondi par la
différence finie d'une divergence entre stratégies.

## Reproduction

```bash
cmake --build build --target price_gradients_performance_benchmarks -j2
python3 maintainer/tools/performance/run_price_gradient_strategies.py \
  --build-dir build \
  --output-dir artifacts/audit/price-gradients-strategies-sm89-<run> \
  --profile
```

Répéter deux fois sans `--profile` fournit les trois campagnes temporelles.
Le premier passage produit en plus un rapport kernel-replay Nsight Compute et
son CSV pour chaque phase exigée par le manifeste.
