# LSM multi-modèles — investigation du 5 septembre 2026

Suite du [diagnostic CIR/OU/G2](fixed-income-lsm-2026-09-05.md).
Cette campagne identifie les coûts et teste des géométries ; elle ne modifie
aucun kernel, schéma, générateur, tuning de production ou dataset de prix.
Ce n'est ni une certification numérique indépendante ni une qualification
de performance au protocole v3.

## Résultat et limites

- 98 processus de mesure terminés : 88 sans profileur, 10 profils Nsight
  Systems ; 362 répétitions chronométrées sans profileur, 10 avec profileur.
- Sorties finies et non négatives, aucun diagnostic de régression fatal,
  aucune différence de prix ou d'erreur standard entre répétitions. Les
  15 groupes comparables de géométries/découpages ont des sorties FP32
  identiques bit à bit. Ce n'est pas une comparaison à un moteur indépendant.
- Heston/Bates : la simulation domine sur les puts à un an et 12 exercices.
  Hull–White/G2++ : les passes backward dominent, avec une forte dépendance
  au nombre de coupons du swap. Le petit solveur Cholesky n'est pas le coût
  principal dans les profils mesurés.
- Les confirmations disponibles n'indiquent qu'environ 5–8 % de gain GPU
  de géométrie sur les cas G2++/Bates retenus, et moins côté host. La série
  prévue n'a pas été achevée : aucun nouveau réglage de production n'est adopté.
- Deux limites matérielles ont interrompu les essais : 86 °C lors du premier
  balayage, puis un changement de plafond électrique pendant la confirmation.
  Le premier balayage a été repris après refroidissement. La confirmation
  finale reste incomplète et les calculs GPU ont été arrêtés.

## Périmètre et méthode

GPU RTX 4090 Laptop, SM89, 16 Gio ; build Release, CUDA 13.3.73, driver CUDA
13.2. Révision `872a986b1f0947a1a832af0615ffc6d80dbedb81`, worktree sale conservé.
L'empreinte exacte du binaire et les sources des sondes sont dans chaque
manifeste/snapshot ; le [résumé structuré](../../tests/performance/reports/cross_model_lsm_20260905.json)
rattache les mesures à leurs fichiers bruts et empreintes.

Neuf modèles, onze compositions : OU, Vasicek, G2, Hull–White/NS et /Svensson,
G2++/NS et /Svensson, Black–Scholes, Heston, Bates, Variance Gamma. Côtés mesurés :
swaptions bermudéennes payer et puts américains. CIR est couvert par le rapport
précédent ; les rough, autres produits et samples ne sont pas mesurés ici.

Toutes les mesures de cette extension utilisent **2²⁰ trajectoires par prix**.
Convention 1/252 ; Heston et Bates gardent `dt = 1/504`, deux pas par jour ouvré.
Les autres compositions mesurées emploient leurs transitions exactes aux dates
contractuelles, sans sous-pas artificiels. Les générateurs ne sont jamais invoqués.

Le découpage host porte seulement sur des prix indépendants. Chaque régression
conserve ses 2²⁰ trajectoires. Le seed du morceau est le seed initial augmenté
de son offset ; aucun tri ou changement d'identité de ligne n'est introduit.
Les courbes fitted sont découpées avec les modèles et produits correspondants.

Les temps « GPU » des sondes sont les événements CUDA retournés par les launchers.
Les temps « host » sont mesurés par horloge monotone autour des appels, allocations
de workspace comprises, mais hors chargement JSON et transfert des entrées/sorties
persistantes. Nsight fournit séparément la somme des durées des kernels. Ces trois
périmètres ne sont pas mélangés : dans cet environnement, événements CUDA et
durées Nsight peuvent différer d'environ 10 %, et certains événements dépassent
même le temps host brut. Aucun gain n'est calculé entre ces deux horloges.

## Changement de puissance : ne pas confondre avec un effet du code

Les XML NVIDIA montrent une enveloppe initiale variable d'environ 161–175 W,
puis **55 W**, avec horloge mémoire passant de 9 001 à 6 001 MHz. Le premier
changement se situe pendant le cas `other-chunks/g2_plus_plus_catalog_chunk16` :
ses répétitions passent de 1,84 à 2,35 puis 3,96 s GPU, CV 33 %. Les mesures
suivantes commencent à 55 W. L'enveloppe revient autour de 165 W avant
`other-profiles` et `other-confirmation`, puis retombe à 55 W avant la huitième
sonde de confirmation.

L'origine de ces changements (secteur, batterie, profil système ou autre)
n'est pas établie. Aucun réglage de puissance/fréquence n'a été changé par la
sonde. Cela n'explique pas rétrospectivement l'ancien incident de 92 minutes,
qui ne dispose pas d'une trace comparable.

Le runner conserve maintenant un XML après chaque processus, vérifie le plafond
avant et après, et arrête la série si l'écart à l'enveloppe initiale dépasse 20 %.
Il refroidit au-dessus de 78 °C jusqu'à 72 °C, par attentes de 10 s, au plus 180 s,
et refuse un nouveau lancement au-dessus de 85 °C. Ce contrôle aux frontières
ne garantit pas l'absence de variations transitoires pendant un processus : il
ne remplace pas le conditionnement et le contrôle de stabilité du protocole v3.
Les mesures anciennes, antérieures à cette vérification de puissance, restent
identifiées avec un contrôle de fin non renseigné dans le résumé.

## Effet des coupons : même nombre de trajectoires et même simulation

Cas synthétique : 16 prix, 10 exercices aux jours 21, 42, …, 210,
128 threads/bloc, 64 blocs/prix. Seul le nombre de paiements du swap change.
Médianes de trois répétitions après un échauffement, avant le passage à 55 W :

| Modèle | 10 coupons | 200 coupons |
| --- | ---: | ---: |
| OU | 26,9 ms | 396 ms |
| Vasicek | 24,6 ms | 408 ms |
| G2 | 54,4 ms | 1 391 ms |
| Hull–White / NS | 57,9 ms | 1 495 ms |
| Hull–White / Svensson | 77,3 ms | 2 197 ms |
| G2++ / NS | 245,7 ms | 5 971 ms |
| G2++ / Svensson | 268,6 ms | 6 859 ms |

Ce tableau donne des ordres de grandeur exploratoires ; certaines cases ont un
CV supérieur à 5 %. Il ne justifie pas de modifier les paramètres numériques.

La lecture du code explique le mécanisme :

- [régression et update](../../src/common/longstaff_schwartz/longstaff_schwartz_kernels.cuh)
  appellent chacun `PricingPolicy::immediate_value` à chaque date et trajectoire ;
- la [policy Bermudan](../../src/product/bermudan_swaption/pricing_policy.cuh)
  calcule alors la valeur du swap restant ;
- [fixed_leg_terms](../../src/common/fixed_income/cashflows.cuh) reboucle sur
  chaque coupon restant et revalorise son zéro-coupon ;
- les [analytics Hull–White](../../src/model/fixed_income/hull_white/fitted_analytics.cuh)
  et [G2++](../../src/model/fixed_income/g2_plus_plus/fitted_analytics.cuh)
  réévaluent dans ce chemin les coefficients affines, courbes et corrections
  déterministes qui dépendent du modèle/calendrier, pas de la trajectoire.

Pour E exercices et P paiements, le nombre de valorisations de coupons du
chemin actuel, terminal compris, est par trajectoire :

`(P − E + 1) + 2 Σ[j=0…E−2](P − j) = (2E − 1)P − (E − 1)²`.

Avec E = 10, il passe de 109 à 3 719 entre P = 10 et P = 200. Si P augmente
avec E jusqu'à P = E, ce terme vaut `E² + E − 1`. Mettre à jour un cashflow
est bien une opération simple ; **calculer sa valeur d'exercice actuelle ne
l'est pas dans cette implémentation**. Le stockage et le solveur ne suffisent
donc pas à décrire le coût. Ce décompte n'est pas une mesure d'instructions ni
un temps proportionnel garanti : le compilateur et les branches comptent aussi.

## Où part le temps : profils des launchers réels

Quatre prix par profil, 2²⁰ trajectoires, 128 threads, 64 blocs/prix.
Les swaptions ont 10 exercices ; les puts ont une maturité de 252 jours,
avec intervalle de 21 jours sauf la dernière ligne. Enveloppe revenue autour
de 165 W, vérifiée avant/après les dix processus.

| Cas | Simulation, terminal inclus | Régression : accumulation | Update cashflows | Solveur |
| --- | ---: | ---: | ---: | ---: |
| Vasicek, 200 coupons | 5,1 % | 47,7 % | 47,1 % | 0,1 % |
| Hull–White/NS, 10 coupons | 4,1 % | 47,9 % | 47,0 % | 0,7 % |
| Hull–White/Svensson, 200 coupons | 4,7 % | 47,6 % | 47,7 % | < 0,1 % |
| G2++/NS, 10 coupons | 2,1 % | 59,7 % | 37,7 % | 0,3 % |
| G2++/Svensson, 200 coupons | 3,8 % | 58,1 % | 38,0 % | < 0,1 % |
| Black–Scholes, 12 exercices | 3,7 % | 68,1 % | 25,1 % | 2,5 % |
| Heston, 12 exercices | 84,2 % | 11,2 % | 4,1 % | 0,4 % |
| Bates, 12 exercices | 89,0 % | 7,8 % | 2,8 % | 0,3 % |
| Variance Gamma, 12 exercices | 21,7 % | 55,3 % | 20,4 % | 2,0 % |
| Black–Scholes, 252 exercices | 4,6 % | 67,9 % | 24,9 % | 2,5 % |

Le nom « regression_partials » n'isole pas l'algèbre linéaire : ce kernel calcule
aussi le payoff immédiat, les candidats et les actualisations. Les profils
n'isolent pas la part exacte des coefficients affines dans ce kernel ; le rôle
des coupons est étayé séparément par le balayage de P et le code.

Sur Black–Scholes, passer de 12 à 252 exercices à un an fait passer la somme
Nsight de 9,63 à 196,45 ms sur quatre prix, soit environ ×20,4 pour ×21 dates.
C'est cohérent avec un coût proche du linéaire en dates lorsque le payoff est
simple ; ce n'est pas la même charge qu'un swap dont la jambe restante s'allonge.

Les diagnostics de lancement trouvent 148 registres/thread et 25 % d'occupation
théorique pour les régressions G2++ ; Hull–White/NS : 96 et 41,7 % ; Heston/Bates :
88 et 41,7 %. Leurs kernels de régression ont un attribut mémoire locale nul.
Ce sont des attributs/occupations théoriques, pas des compteurs Nsight Compute
de débit ou de spills exécutés. Les essais historiques de séparation des
statistiques FP64 ont déjà été rejetés par mesure dans
[PERF-003](../audit/closed.md) ; ils ne sont pas reproposés ici.

## Blocs/threads : gains modestes après confirmation

Balayage de cinq configurations sur sept modèles, toujours 16 prix et 2²⁰
trajectoires. Les résultats les plus favorables du premier balayage étaient
souvent associés à une référence bruitée. La confirmation alterne les ordres,
avec trois échauffements et sept répétitions par processus :

| Cas et série | Référence 128 threads / 64 blocs | Candidat | Gain GPU | Gain host |
| --- | ---: | ---: | ---: | ---: |
| G2++/NS, série 0 | 190,52 ms | 180,63 ms, 128 / 256 | 5,2 % | 3,6 % |
| Bates, série 0 | 288,56 ms | 270,50 ms, 256 / 256 | 6,3 % | 4,4 % |
| Bates, série 1, ordre inverse | 293,44 ms | 269,45 ms, 256 / 256 | 8,2 % | 6,3 % |

La seconde paire G2++ et la troisième série n'ont pas été achevées, car le
plafond est retombé à 55 W. Le candidat Bates de série 1 a un CV de 7,7 %.
Ces gains restent donc des candidats, pas des réglages qualifiés. Pour
Black–Scholes synthétique, la configuration 256 / 256 monte à 36,97 ms contre
32,66 ms en 128 / 64 : augmenter partout n'est pas une optimisation générique.

## Paquets du catalogue et mémoire

128 lignes réelles de taux, indices 400–527 inclus (base zéro), même seed et
mêmes entrées, 2²⁰ trajectoires, géométrie 128 / 64. Médianes host/GPU :

| Modèle | Paquets de 16 : GPU / host | De 64 : GPU / host | De 128 : GPU / host |
| --- | ---: | ---: | ---: |
| Vasicek | 181 / 355 ms | 154 / 329 ms | 161 / 323 ms |
| Hull–White/NS | 496 / 651 ms | 446 / 584 ms | 437 / 571 ms |
| G2 | 507 / 738 ms | 458 / 684 ms | 440 / 655 ms |

La taille de workspace maximale passe de 1,31 à 3,52 puis 6,79 Gio en un
facteur ; de 1,94 à 5,15 puis 9,94 Gio en deux facteurs. Les petites tranches
réduisent la mémoire simultanée, pas le nombre de trajectoires par régression.
Le coût host comprend les réallocations des launchers à chaque morceau.
La comparaison G2++ est exclue à cause du changement de puissance décrit plus
haut. Ces essais ne sont pas une campagne complète de 1 000 prix.

Les puts du catalogue, indices 425–440, ont également été testés par morceaux
de 1, 4 ou 16 prix, **dans l'enveloppe 55 W**, avec 3,02 Gio maximum en un
facteur et 5,98 Gio en deux facteurs. Leurs mesures sont conservées dans le
résumé, mais ne sont pas mélangées aux timings à ~165 W ni utilisées pour
retenir un réglage de campagne. Les prix et erreurs restent bit-identiques.

## Suite technique proposée, non implémentée

1. Sur les swaptions, prototyper la préparation des coefficients de zéro-coupon
   et des termes d'actualisation déterministes par prix/calendrier. Garder les
   tables dans un workspace dédié, pas une grande matrice recopiée en shared
   pour chaque bloc. Conserver les mêmes conventions et qualifier les arrondis.
2. Mesurer un cache du payoff immédiat entre accumulation et update : un
   tampon réutilisable d'un float par trajectoire et prix actif, soit 4 Mio
   à 2²⁰, pas un nouveau cube trajectoires × dates. La lecture/écriture
   supplémentaire doit être incluse dans la comparaison ; aucun speedup de
   ce prototype n'a encore été mesuré.
3. Pour Heston/Bates, cibler la simulation et l'occupation plutôt que le solveur.
   Refaire les confirmations et profils matériels sous alimentation stabilisée,
   sans réduire le nombre de trajectoires, le pas ou la précision FP64.
4. Garder la géométrie de production actuelle en attendant une qualification
   multi-modèles. Aucun nouveau tri de lignes ou découpage pathwise des
   régressions n'est proposé.

## Reproduction et artefacts

La [sonde](../../tests/performance/fixed_income_lsm_probe.cu), malgré son nom
historique, couvre maintenant les modèles ci-dessus. Compilation limitée :

```sh
cmake --build build-dev --target ai_factory_fixed_income_lsm_probe -j 1
python3 -m unittest tests.performance.test_lsm_probe_tools
```

Les cinq tests CPU du harnais passent ; tous les launchers mesurés ont compilé
et exécuté leurs cas. Pas de validation Premia/QuantLib lancée : aucun calcul
de production n'a été changé.

Les suites et fixtures réutilisables sont `--suite other-models` et
`tests/performance/fixtures/lsm_probe_other_{geometry,chunks,profiles,confirmation}.json`.
Le runner accepte `--start-index` uniquement pour reprendre dans un **nouveau**
répertoire de preuves ; `--profile` capture un échauffement puis une mesure.
Chaque commande réellement exécutée est dans son `results.ndjson`.

Les preuves brutes sont sous `build-dev/lsm-probe-20260905/other-*` : manifestes,
empreintes des entrées, snapshots, prix/erreurs de chaque répétition, diagnostics
de ressources, XML NVIDIA et dix rapports Nsight/CSV. Ces fichiers de build
sont locaux ; le résumé JSON et ce rapport sont des fichiers source à versionner.

Le résumé se reconstruit sans GPU avec :

```sh
python3 tools/performance/summarize_lsm_probe.py \
  build-dev/lsm-probe-20260905/other-models \
  build-dev/lsm-probe-20260905/other-geometry \
  build-dev/lsm-probe-20260905/other-geometry-resumed \
  build-dev/lsm-probe-20260905/other-chunks \
  build-dev/lsm-probe-20260905/other-profiles \
  build-dev/lsm-probe-20260905/other-confirmation \
  --output tests/performance/reports/cross_model_lsm_20260905.json
```
