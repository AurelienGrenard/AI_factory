# Diagnostic GPU des Bermudans fixed income — 5 septembre 2026

Les mesures identifient la simulation CIR comme coût dominant sur les
calendriers ordinaires. Le stockage Longstaff–Schwartz reste praticable à
200 exercices et un million de trajectoires. Augmenter systématiquement les
blocs n'est pas une solution générale : le gain dépend du nombre de prix
simultanés. L'incident historique de 92 minutes n'a pas été reproduit et sa
cause précise reste indéterminée.

## Périmètre et traçabilité

- RTX 4090 Laptop, SM89, 76 SM, 16 Gio ; Windows/WSL, pilote 596.08,
  CUDA Runtime/NVCC 13.3.73, build Release GCC 14, sans fast math.
- Révision `872a986b1f0947a1a832af0615ffc6d80dbedb81`, worktree déjà modifié.
  Chaque série conserve le statut Git, le diff, les sources de la sonde,
  l'empreinte du binaire et celles des entrées JSON.
- 153 processus de benchmark terminés, 683 répétitions mesurées, plus les
  traces Nsight Systems de six configurations et un profil Nsight Compute.
- CIR, Ornstein–Uhlenbeck et G2, swaptions payer. Les autres modèles et les
  receivers n'ont pas été profilés dans cette campagne.
- Convention 252 jours ; CIR à `dt = 1/504`. OU/G2 emploient leurs transitions
  jointes exactes facteur–intégrale entre dates d'exercice.
- Appels aux launchers publics de production, sans remplacement des kernels.
  Aucun dataset de production n'est écrit et aucun réglage de production
  n'est modifié. La sonde écrit uniquement ses résultats de test sur stdout.
- Essais GPU séquentiels, watchdog de 25 à 60 secondes par processus,
  plafond mémoire explicite. La limite par défaut est 2 Gio ; seuls les
  essais mémoire montent explicitement jusqu'à 14 Gio de budget.

Les [mesures synthétisées](../../tests/performance/reports/fixed_income_lsm_20260905.json)
référencent les sorties brutes sous
`build-dev/lsm-probe-20260905/`. Chaque sous-dossier contient `manifest.json`,
`results.ndjson`, les sorties individuelles et les snapshots NVIDIA avant/après.
Les dernières séries possèdent aussi un snapshot avant chaque essai, avec
arrêt au-dessus de 85 °C. Le GPU était libre au démarrage et à la fin.

Il s'agit d'une campagne exploratoire, pas d'une nouvelle baseline de
qualification selon le [protocole v3](../performance-regression-protocol.md).
Le screening utilise une chauffe et trois mesures, ou une seule mesure pour
les grandes tranches catalogue. Les six géométries finalistes utilisent trois
séries de cinq chauffes et 21 mesures. La température/fréquence a dérivé entre
séries ; aucune stabilisation thermique formelle n'a été imposée. Les écarts
de quelques pour cent ne justifient donc aucun réglage définitif.

Les durées ci-dessous sont les intervalles d'événements CUDA, sauf mention
contraire. Le temps hôte brut est conservé séparément : les deux horloges
diffèrent sur cette machine. Les durées Nsight ne sont pas mélangées aux
répétitions non profilées. Les six profils retenus viennent de `profiles/`
et `profiles-extra-final/` ; `profiles-extra/` reste exploratoire, car la
reconstruction de la sonde a chevauché cette première capture complémentaire.
Un résumé a aussi été manqué par le parseur initial, à cause d'une barre de
progression Nsight sur la même ligne ; la trace et les sorties complètes sont
conservées, et le parseur a été corrigé.

## 1. La simulation CIR domine le cas ordinaire

Quatre prix synthétiques, chacun à `2^20` trajectoires, dix exercices
semestriels de 0,5 à 5 ans, dix paiements, 128 threads/bloc et 64 blocs/prix :

| Phase, profil Nsight Systems | Temps GPU | Part du temps des kernels |
|---|---:|---:|
| Simulation des trajectoires, stockage, payoff terminal | 745,23 ms | 99,17 % |
| Statistiques de régression, neuf dates | 3,61 ms | 0,48 % |
| Mises à jour des cashflows, neuf dates | 2,43 ms | 0,32 % |
| Résolutions, préparation et finalisation | 0,16 ms | 0,02 % |

Chaque pas CIR appelle le tirage exact non central chi-deux, implémenté par
Poisson–Gamma avec rejet. La représentation jointe ajoute l'intégrale du taux
par trapèzes. Un pas ne se réduit donc pas à tirer un seul brownien. L'état
final du pas précédent conditionne le suivant : ces transitions restent
séquentielles à l'intérieur d'une trajectoire.

Le profil matériel du même kernel, sur quatre prix à 65 536 trajectoires et
256 blocs/prix, mesure 70 registres/thread, aucun spill local, 48,55 %
d'occupation effective, environ 72,14 % du débit SM de référence et une
activité DRAM très faible. En moyenne, 13,48 threads par warp exécutent une
instruction, sur 32 : le flot d'exécution divergent des tirages est visible.
Ces compteurs étayent un coût de calcul et de contrôle, sans saturation du
débit DRAM dans ce cas. Ils ne constituent pas une comparaison de temps sous
profiler. Fichiers : `cir-simulation.ncu-rep` et
`cir-simulation.metrics.csv` dans le répertoire brut.

## 2. Le passage à un million de trajectoires reste proche du linéaire

Même cas synthétique, quatre prix, 128 threads/bloc, médianes du screening :

| Trajectoires par prix | CIR, 64 blocs/prix | CIR, 1 024 blocs/prix |
|---:|---:|---:|
| 16 384 | 15,46 ms | 12,70 ms |
| 65 536 | 58,72 ms | 38,19 ms |
| 262 144 | 206,19 ms | 150,21 ms |
| 1 048 576 | 819,66 ms | 639,00 ms |

Entre `2^16` et `2^20`, le facteur mesuré est 13,96 pour 64 blocs et 16,73
pour 1 024 blocs. Aucun effondrement d'un ordre de grandeur n'apparaît.
À un million de trajectoires, les témoins OU/G2 du même calendrier mesurent
respectivement 8,21/17,68 ms avec 64 blocs. Ce sont des modèles différents,
avec transitions jointes exactes : ces nombres ne proposent pas une
substitution numérique au CIR.

## 3. Plus de blocs aide surtout les petits paquets

Screening de 64/128/256/512 threads et 16/64/256/1 024 blocs par prix, à
`2^20` trajectoires. Le calendrier va de 21 à 210 jours, avec dix exercices
et dix paiements. Confirmation de six configurations : médiane des trois
médianes de campagne ; p95 = maximum des trois p95.

| Prix simultanés | Threads/bloc | Blocs/prix | Médiane | p95 | CV médian |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 64 | 101,56 ms | 112,19 ms | 0,3 % |
| 1 | 128 | 1 024 | 36,42 ms | 38,42 ms | 1,5 % |
| 1 | 256 | 1 024 | 37,95 ms | 38,05 ms | 1,1 % |
| 16 | 128 | 64 | 575,24 ms | 586,27 ms | 4,2 % |
| 16 | 128 | 1 024 | 525,49 ms | 534,43 ms | 3,1 % |
| 16 | 256 | 1 024 | 549,49 ms | 559,43 ms | 2,6 % |

Le gain d'environ 2,8 fois sur un prix isolé est robuste. Sur 16 prix, il
devient modeste et dépend de la série. À 64 blocs/prix, 16 prix lancent déjà
1 024 blocs au total. Le nombre de trajectoires par thread ne suffit donc
pas à diagnostiquer une sous-utilisation du GPU. Augmenter les blocs accroît
aussi le nombre de partials de régression à produire et à réduire.

Toutes les sorties prix du screening des 32 géométries sont identiques bit à
bit. Tous les replays mesurés d'une même configuration donnent également
les mêmes prix et erreurs standards. Cette reproductibilité n'est pas une
certification indépendante de la précision financière.

## 4. Horizon, exercices et coupons ont des effets distincts

Un prix, `2^20` trajectoires, 128 threads/bloc, 256 blocs/prix, dix exercices
semestriels et dix paiements ; seule la première date varie :

| Première date | Dernier exercice | CIR | OU |
|---:|---:|---:|---:|
| 0,5 an | 5 ans | 205,74 ms | 2,10 ms |
| 5 ans | 9,5 ans | 402,97 ms | 2,04 ms |
| 30 ans | 34,5 ans | 1 605,43 ms | 2,02 ms |

La mémoire est identique : le CIR paie les transitions intermédiaires, même
si seuls dix états sont sauvegardés. L'OU saute directement entre exercices.

À huit exercices fixes, de 21 à 168 jours, passer de dix à 200 coupons fait
passer le CIR de 27,75 à 48,63 ms et l'OU de 1,79 à 23,93 ms. Dans le profil
CIR à 200 coupons et quatre prix, la simulation occupe 55,2 % des kernels ;
les statistiques de régression et mises à jour occupent ensemble 44,7 %.
Ces deux passes recalculent le payoff immédiat en parcourant les coupons
restants. Leur temps ne mesure donc pas la seule algèbre de régression.

À horizon total fixé à un an, 200 paiements et 2/3/6/11 exercices, l'OU
mesure 4,77/8,01/17,60/33,31 ms. C'est compatible avec le coût attendu des
passes backward ; il n'y a pas de rupture liée au nombre de dates.

Le test demandé avec **200 exercices, 200 paiements et `2^20` trajectoires**
termine également. Les dates vont de 53 à 252 jours, espacées d'un jour :

| Modèle | Temps GPU | Workspace |
|---|---:|---:|
| CIR | 492,71 ms | 1,566 Gio |
| OU | 455,70 ms | 1,566 Gio |
| G2 | 1 441,14 ms | 2,348 Gio |

Un seul cashflow est maintenu par trajectoire. CIR/OU conservent deux floats
par date (facteur et intégrale), G2 trois. Les coupons ne sont pas stockés
comme des dates de simulation supplémentaires.

## 5. La pression mémoire seule ne reproduit pas le blocage

Témoin OU, dix exercices, `2^20` trajectoires par prix, 128 threads et
64 blocs/prix. Ces mesures uniques sont des vérifications de capacité,
pas des estimations de p95 :

| Prix simultanés | Workspace alloué | Temps GPU | Appel hôte brut |
|---:|---:|---:|---:|
| 64 | 5,251 Gio | 91,36 ms | 232,13 ms |
| 96 | 7,876 Gio | 136,92 ms | 334,57 ms |
| 128 | 10,501 Gio | 187,28 ms | 450,81 ms |
| 144 | 11,814 Gio | 208,55 ms | 503,65 ms |
| 160 | 13,126 Gio | 245,13 ms | 585,49 ms |

Chaque cas tient dans un batch ; les états de chaque prix restent disponibles
pendant toute sa régression. L'allocation/libération pèse dans l'appel hôte,
mais il n'y a pas de blocage au voisinage du volume mémoire de l'incident.
Cela ne teste pas toutes les interactions possibles avec un très long kernel
CIR : aucun lancement monolithique de 1 000 prix à `2^20` n'a été relancé.

## 6. Catalogue réel et incident historique

Les 1 000 couples alignés ont été parcourus par tranches de 25 à 4 096
trajectoires : tous terminent, toutes les sorties sont finies. La somme des
intervalles GPU est 2,880 s. Une seconde exécution traite toute la base à
65 536 trajectoires : 44,19 s d'événements CUDA, 40,60 s d'appel hôte brut,
3,241 Gio de workspace, un batch.

Quatre tranches réelles de quatre prix à `2^20`, 256 blocs/prix, terminent
respectivement en 0,242 / 1,870 / 7,843 / 4,698 s, aux offsets 0 / 425 /
825 / 975. Les longues dates et les paramètres propres au CIR rendent les
coûts très hétérogènes ; une base ne doit pas être extrapolée depuis ses
premières lignes seulement.

Une extrapolation linéaire du nouveau passage complet à `2^16` donne
environ 11,8 minutes à `2^20`. C'est une estimation, pas une mesure de cette
campagne complète. Le calendrier et les kernels ne montrent pas de raison
intrinsèque d'exiger 92 minutes. L'ancien artefact à 29,6 s hôte / 31,1 s GPU
utilise en outre une autre seed (`2130000001`) que la recette actuelle
(`11668829039698640896`) et n'a pas un environnement thermique contrôlé
commun avec cette campagne.

L'affirmation précédente « les 64 blocs causent la falaise WSL » était donc
trop catégorique. Le lancement interrompu n'avait pas de trace par kernel :
on ne peut pas lui attribuer rétrospectivement une cause précise. La mémoire
cumulée de 55,5 Go sur tous les prix n'est pas une allocation nécessaire pour
un seul prix et ne suffit pas davantage à l'expliquer.

## Suites justifiées par les mesures

1. Garder `2^20` et `dt=1/504` pour les références ; les tests ne justifient
   pas de sacrifier cette configuration pour résoudre un défaut de stockage.
2. Pour le CIR, cibler d'abord le coût du tirage et des transitions ; évaluer
   une géométrie adaptée au nombre de prix réellement présents dans le batch.
   Les chiffres ne justifient pas d'imposer 1 024 blocs à tous les cas.
3. Pour les longues jambes de coupons, étudier la préparation des coefficients
   obligataires invariants et la réutilisation du payoff entre les passes
   régression/update. Un buffer d'un float par trajectoire, réutilisé à chaque
   date, suffirait à éviter cette seconde évaluation ; le gain reste à mesurer.
4. Pour reprendre la campagne, rendre visibles les temps de chaque phase et
   de chaque paquet. Cela permettra d'identifier précisément une récidive du
   blocage sans attendre une heure et demie.

Ces pistes n'ont pas été implémentées dans le runtime. Les ajouts sont la
sonde, son runner borné, ses fixtures et ce compte rendu.

## Reproduction

Depuis la racine `AI_factory` :

```sh
cmake --build build-dev --target ai_factory_fixed_income_lsm_probe -j1
python3 tools/performance/run_fixed_income_lsm_probe.py \
  --suite geometry --output build-dev/lsm-probe-replay/geometry --timeout 30
python3 tools/performance/run_fixed_income_lsm_probe.py \
  --jobs tests/performance/fixtures/lsm_probe_confirmation.json \
  --output build-dev/lsm-probe-replay/confirmation --timeout 30
```

Les autres suites sont `catalog`, `scaling`, `calendar` et `batch` ; les
fixtures `lsm_probe_catalog_memory.json` et `lsm_probe_memory_pressure.json`
décrivent les cas réels et mémoire. Le dossier de sortie doit être nouveau.
`--profile --diagnostics`, avec les fixtures `lsm_probe_profiles*.json`,
capture les phases Nsight Systems. Les commandes complètes effectivement
exécutées sont conservées dans chaque journal brut.
