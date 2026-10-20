# Monte Carlo multi-actifs

## Contrat des facteurs

Une recette ordonne ses actifs une fois pour toutes. Chaque groupe
`simulation::ModelGroup<Dynamics, Count>` référence un dataset modèle et
`Count` positions ; plusieurs positions peuvent pointer vers la même ligne ou
vers le même dataset. `simulation::MultiAssetPack<Groups...>` fixe à la
compilation le nombre de positions, leurs types et l'ordre de leurs facteurs.
Les indices de lignes et le nombre de trajectoires restent des données runtime.

Chaque dynamique éligible expose `kBrownianDimension`, `BrownianStep` et
`DynamicsPolicy::one_step_transition(prepared, brownian, auxiliary, state)`.
Le premier facteur d'un actif est son spot. Les modèles de volatilité
stochastique ont ensuite un facteur de volatilité. L'overload reçoit les
innovations corrélées, tandis que `simulate_one_step` reste disponible pour le
moteur unidimensionnel existant. Black-Scholes, CEV, Heston, SABR, Bates,
Merton, Kou, Heston 3/2, Schöbel-Zhu, Stein-Stein, Variance-Gamma et NIG
suivent ce contrat. Les deux derniers reçoivent une normale corrélée pour
leur composante gaussienne et tirent leur horloge Gamma ou inverse gaussienne
dans le domaine Philox auxiliaire. La corrélation de ces normales n'est pas,
en général, la corrélation des rendements des modèles à horloge aléatoire.
Les sauts idiosyncratiques de Merton, Kou et Bates peuvent également modifier
la corrélation des rendements complets. Les modèles à volatilité OU exacte
tirent leur normale de pont dans
`auxiliary`, conditionnellement au Brownien de volatilité partagé.

La ligne de corrélation porte les mêmes noms de facteurs, dans le même ordre.
Le loader vérifie diagonale unité, symétrie et définition positive, puis
calcule Cholesky en double précision et stocke le triangle inférieur en FP32.
Pour les recettes à modèles complets, le runner vérifie aussi que la corrélation
spot-vol de la matrice est égale au paramètre `rho` ou `correlation`. Les
recettes factorisées déduisent ce paramètre de la matrice avant le lancement.
Une matrice semi-définie positive singulière ne passe pas ce Cholesky strict.

## Trajectoires et GPU

Un bloc calcule un prix. Le bloc copie le facteur triangulaire dans la mémoire
partagée et prépare chaque modèle une fois par ligne de prix. À chaque pas,
les threads d'une trajectoire forment le vecteur normal indépendant, calculent
`LZ`, puis chaque position avance avec la ou les composantes browniennes qui lui sont attribuées. Un thread
suit le produit worst-of à ses dates d'observation. La réduction du minimum
est collective pour une trajectoire répartie sur plusieurs threads.

`TeamSize` vaut 1, 2, 4, 8, 16 ou 32 à la compilation. Avec un seul thread,
l'état de la trajectoire est local au thread. Avec plusieurs threads, les
positions se répartissent entre les lanes et l'état réside en mémoire
partagée. Le triangle de Cholesky est toujours en mémoire partagée. Le bon
`TeamSize` dépend de la dimension et du GPU ; la recette l'accepte comme
argument pour les mesures de performance.

Le Brownien a le domaine Philox 1. Son compteur dépend de la ligne de prix,
de l'identifiant de trajectoire, du pas et du groupe de quatre normales ; il
ne dépend pas de `TeamSize`. Le domaine 0 conserve l'adressage scalaire.
Chaque position possède un domaine auxiliaire distinct `2 + position` pour
QE, pont OU et sauts. Un nombre variable de sauts ne décale donc ni le
Brownien, ni une autre position, ni le pas suivant. Le nombre de positions est
limité à 254 par cet adressage et le pas à la capacité de 24 bits du domaine.

Les trajectoires sont rangées dans des sous-totaux logiques indexés par
`path % threads_per_block`, quel que soit `TeamSize`. Les sommes FP64 et
l'arbre final gardent ainsi le même ordre : à géométrie, seed et binaire
identiques, le prix et l'erreur standard sont identiques bit à bit entre les
six valeurs de `TeamSize` sur le panier Black-Scholes de référence.

Sur la RTX 4090 Laptop SM89, la recette pilote à deux facteurs, 32 prix,
16 384 trajectoires et 21 pas a donné environ 0,73 ms de kernel avec
`TeamSize=1`, 2,19 ms avec 2 et 4,38 ms avec 4 (deux passages par taille).
Un panier Black-Scholes de 30 facteurs, 8 192 trajectoires et 21 pas a donné
17,0 ms avec 1 et 26,6 ms avec 4 dans un passage exploratoire. Les prix
étaient identiques. Le défaut de la recette est donc 1 ; une sélection pour
d'autres dimensions et mélanges de modèles demande des mesures dédiées.
Ces mesures exploratoires utilisaient un pas par jour. Les recettes de prix
du catalogue utilisent deux pas par jour (`dt = 1 / 504`) ; les temps ci-dessus
ne mesurent donc pas leur grille actuelle.

Black-Scholes avance aussi sur la grille lorsqu'un autre modèle utilise un
schéma. Son pas utilise la formule exacte GBM avec l'incrément brownien
corrélé de ce pas ; la composition des pas donne la loi terminale exacte sur
la grille. Le produit reçoit ensuite les observations de la trajectoire.

## Recette pilote

La recette
[`bs2_worst_of_calls_01`](../../catalog/multi_asset/worst_of_call/bs2_worst_of_calls_01/recipe.yaml)
utilise deux positions Black-Scholes, le produit worst-of et une matrice de
corrélation 2 × 2. Sa sortie porte les identifiants ordonnés `model_ids`,
`product_id`, `correlation_id`, le prix et son erreur standard. Une courbe peut
être référencée dans le contrat de ligne ; le pricer reçoit alors le facteur
d'actualisation calculé pour cette ligne.
La diffusion utilise deux incréments browniens par jour et par facteur, avec
`dt = 1 / 504` ; le produit reste observé aux dates prévues par son calendrier.

Depuis la racine du projet :

```bash
cmake --preset local-sm89
python3 tools/run_generator.py local-sm89 generate_bs2_worst_of_calls_01
cmake --build builds/local-cuda12-9-0-sm89 --target test_multi_asset_cuda
ctest --test-dir builds/local-cuda12-9-0-sm89 -R '^multi_asset_cuda$' --output-on-failure
```

Le lanceur génère au besoin les entrées produit et corrélation dans son dossier
isolé, puis enrichit les trois reçus de provenance. Ces recettes sont
inventoriées comme non publiées dans `catalog/manifest.json`.

## Mélange markovien et rough causal

Le moteur mixte est dans
`src/common/monte_carlo/multi_asset_rough_kernel.cuh`, avec la préparation
FFT causale dans `src/common/volterra/causal_fft_volatility.cuh` et
`causal_fft_core.cuh`. Il accepte un `MultiAssetPack` markovien et un nombre
fixé à la compilation de rough Heston ou de quadratic rough Heston. Les
lignes de modèles et de produits, le nombre de trajectoires et la taille des
chunks restent des données runtime.

La recette nomme les facteurs dans l'ordre des actifs : facteurs markoviens,
puis spot et volatilité de chaque actif rough. Le runner reconstruit la
corrélation depuis le Cholesky chargé et la refactorise dans l'ordre interne :
toutes les volatilités rough, les facteurs markoviens, puis les spots rough.
Pour quadratic rough Heston, le même facteur commande sa volatilité et son
spot ; il n'apparaît qu'une fois dans le préfixe. Le runner vérifie le `rho`
spot–vol de chaque rough Heston.

Pour chaque chunk de trajectoires, la première passe calcule les volatilités
rough et écrit leurs valeurs par pas dans un tampon GPU. Elle ne génère que
les normales nécessaires aux facteurs de volatilité : le Cholesky du bloc
principal rough est le bloc supérieur gauche du Cholesky complet, après
réordonnancement. La deuxième passe rejoue les mêmes normales Philox,
calcule le vecteur brownien complet, avance ensemble tous les spots et
observe le produit. Les normales rough déjà utilisées pour la volatilité
restent dans le vecteur corrélé, mais ne sont pas consommées une seconde fois
par les spots rough à deux facteurs. Le spot rough utilise son propre facteur
brownien. Un modèle rough à un facteur réutilise ce facteur pour son spot.

Le scan conjoint reçoit un pointeur vers les volatilités pré-calculées. Le
moteur purement markovien possède un kernel distinct qui n'a pas besoin de ce
tampon. Les deux partagent `MultiAssetPack`, la génération Philox, le calcul
`LZ` et les politiques de produit. Le worst-of call et la barrière basse sur
panier sont observés dans le scan conjoint. La barrière croisée dépend des
spots **à la même date**, par exemple `mean(S_i(t)/S_i(0)) <= B` : les seuls
minima individuels ne suffisent pas.

Le workspace FFT est réutilisé pour chaque modèle rough. Avec `R` facteurs
rough, `C` trajectoires par chunk et `N` pas, le tampon principal de
volatilités contient `R×C×N` valeurs FP32. Le flux auxiliaire du rough numéro
`r` est `2 + nombre_d_actifs_markoviens + r` ; les flux QE, sauts et ponts des
actifs markoviens conservent leurs adresses.

Les tests `mixed_rough_cuda`, `mixed_quadratic_rough_cuda`,
`mixed_two_rough_cuda`, `mixed_heston_rough_cuda`,
`mixed_large_dimension_cuda` et `mixed_basket_barrier_cuda` couvrent le prix
joint, des marginales avancées successivement, une barrière croisée, plusieurs
facteurs rough et le découpage en chunks. Sur une RTX 4090 Laptop SM89, avec
un BS, un rough Heston, 4 096 trajectoires et 128 pas, le scan joint a pris
environ 0,078 ms contre 0,095 ms pour les scans marginaux successifs. Le
pipeline complet a pris environ 0,257 ms. Avec 30 BS, un rough Heston,
1 024 trajectoires et 64 pas, le scan joint a pris environ 0,895 ms contre
7,44 ms pour les 31 marginales avancées successivement ; le pipeline complet
a pris environ 1,00 ms.

Pour la barrière basse sur panier, les médianes de sept exécutions du pipeline
complet, avec 4 096 trajectoires et 128 pas, sont :

| Panier | Monitoring | Temps |
|---|---:|---:|
| 2 actifs | chaque pas | 0,298 ms |
| 2 actifs | tous les 4 pas | 0,276 ms |
| 2 actifs | tous les 16 pas | 0,276 ms |
| 31 actifs | tous les 4 pas | 1,897 ms |

La recette pilote
[`bs_rough_heston_worst_of_calls_01`](../../catalog/multi_asset/worst_of_call/bs_rough_heston_worst_of_calls_01/recipe.yaml)
utilise huit matrices dont la corrélation spot–vol est prise sur la ligne
rough correspondante. Pour la générer en dossier isolé :

```bash
python3 tools/run_generator.py local-sm89 generate_bs_rough_heston_worst_of_calls_01
ctest --test-dir builds/local-cuda12-9-0-sm89 -R '^(multi_asset_cuda|mixed_.*rough_cuda|causal_fft_.*cuda)$' --output-on-failure
```

## Rough gaussien à FFT hybride

Les quatre politiques rough à FFT hybride (`rough_bergomi`,
`log_modulated_rough_bergomi`, `rough_sabr`, `rough_stein_stein`) utilisent
maintenant `src/common/volterra/hybrid_fft_volatility.cuh` pour convoluer les
facteurs de volatilité corrélés, puis le même scan conjoint de spots que le
moteur causal. Les adaptateurs modèle sont réunis dans
`src/model/equity/rough/hybrid_multi_asset.cuh`. La stratégie
La même stratégie de préparation des volatilités est utilisée pour ces
quatre politiques. Le test
`mixed_hybrid_rough_cuda` compare chaque volatilité à une convolution directe
et chaque payoff à une référence séquentielle ; il couvre aussi deux actifs
rough Bergomi avec un préfixe de corrélation 2 × 2 et une matrice complète
5 × 5. Sur 512 trajectoires et 64 pas, sept lancements après échauffement
ont donné des médianes de 0,23 ms pour deux rough ensemble et 0,30 ms pour
deux exécutions séparées du même moteur. Cette comparaison porte sur le coût
de simulation ; les produits des exécutions séparées ne donnent pas le prix
du panier joint. Ces temps varient avec l'état du GPU.

Pour plusieurs positions du même type rough, le runner homogène conserve son
kernel optimisé. Le pack `simulation::MultiRoughPack<Policies...>` et
`tools/pricing/heterogeneous_rough_price_generation.cuh` composent aussi des
types rough différents dans un même prix. Ils réutilisent les mêmes moteurs
de volatilité causal et hybride, puis un scan conjoint. L'ordre interne
reste : toutes les volatilités rough, les facteurs markoviens, puis les
spots rough qui disposent d'un second Brownien. `mixed_heterogeneous_rough_cuda`
vérifie rough Heston + rough Bergomi, quadratic rough Heston + rough Bergomi,
le chunking, une barrière croisée et l'identité avec le moteur homogène
lorsque les deux modèles rough sont des rough Heston. Le pack hétérogène utilise la même préparation des volatilités avant le
scan conjoint.

## Extension taux et FX

Une future composition equity/taux/FX utilisera le même vecteur d'innovations
et la même matrice de corrélation, avec un groupe d'états par classe d'actifs.
Le produit recevra à chaque observation les états nécessaires pour calculer
son panier, sa conversion de devise et le facteur d'actualisation de la
trajectoire. La ligne de prix portera alors les références vers les datasets
des modèles equity, de taux, de FX, de produit et de corrélation. Le payoff
actualisé par un taux stochastique exigera l'intégrale du taux simulé ; un
simple facteur d'actualisation déterministe par ligne ne suffit pas. La
mesure de simulation et les drifts equity/FX devront rester cohérents avec la
devise et le numéraire du payoff.
