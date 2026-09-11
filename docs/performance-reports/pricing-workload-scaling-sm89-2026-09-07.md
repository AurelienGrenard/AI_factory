# Pricing : scaling par charge et géométrie — SM89, 7 septembre 2026

État : **mesures en cours**, pas une qualification globale ni un nouveau
profil de production. Suivi : `PERF-016`, `PERF-017`, `PERF-019` dans
[les constats ouverts](../audit/response.md).

Pour préparer la génération à **1 000 prix × 2²⁰ trajectoires**, consulter le
[notebook de temps par dataset](pricing-dataset-runtime-sm89.ipynb).
Il utilise les 1 000 lignes originales du catalogue, distingue les mesures
terminées des cas encore manquants et ne remplace aucun résultat par une
extrapolation. Le présent rapport conserve les comparaisons historiques sur
tuile répétée : les deux profils d'entrée ne sont pas interchangeables.

## Périmètre et méthode

Même HEAD `872a986b1f0947a1a832af0615ffc6d80dbedb81`, branche `main`, worktree
partagé modifié. Chaque répertoire d'exécution ci-dessous conserve un snapshot
source complet, les SHA-256 des binaires/données, jobs, résultats bruts,
diagnostics compilés et télémétrie. Machine : RTX 4090 Laptop, SM89,
driver 596.08. Aucun réglage matériel ni seuil thermique applicatif.

- Matrice : 100 / 1 000 / 10 000 prix × 65 536 / 262 144 / 1 048 576
  trajectoires **par prix**; pas d'axe trajectoires pour les formules fermées.
- Une tuile temporaire de 100 lignes stratifiées (90 core, 10 stress) garde
  paramètres et calendriers comparables entre tailles. Les répétitions de la
  tuile portent des identifiants/graines distincts; elles ne constituent pas
  10 000 paramètres indépendants. Aucun dataset publié n'est écrasé.
- Launchers publics inchangés. Géométries distinctes par modèle, moteur et
  taille : threads, grille, batches, chunks FFT; blocs/prix en plus pour LSM.
  Le profil natif MC à batches/grille plafonnés à 4 096 doit aussi être comparé.
- Préparation hôte, enveloppe CUDA, API synchronisée, copie et publication
  native sont séparées. L'enveloppe CUDA contient les intervalles entre appels
  hôte : elle ne remplace pas un profil des durées de chaque kernel.
- Présélection : un warmup exclu, une mesure. Géométrie répétée : au moins
  trois mesures et CV ≤ 5 % pour un point statistiquement recevable. La
  sélection d'un minimum entre géométries reste diagnostique et demande une
  confirmation fraîche avant remplacement du profil de production.

## Preuves disponibles

Tous les chemins ci-dessous sont relatifs à `build-dev/`.

| Campagne | Résultat acquis | Limite |
|---|---|---|
| `pricing-scaling-20260907-mc-calibration-01` | 35/35 cas MC à 100 × 64k | une mesure, aucun choix de réglage |
| `pricing-scaling-20260907-mc-screen-100-01` | 33/35 séries complètes; deux séries rough Heston arrêtées au candidat 512 threads | géométrie interdite par les ressources; pas de crash GPU ni de défaut numérique démontré |
| `pricing-scaling-20260907-heston-screen-100-01` | 18 jobs terminal/barrière, trois N, trois nombres de threads | présélection; compilation CPU concurrente |
| `pricing-scaling-20260907-closed-form-geometry-01` | 81/81 jobs, neuf compositions, jusqu'à 10 000 prix; 81 comparaisons numériques passent | bruit excessif sur plusieurs mesures très courtes; seulement CIR et G2 ont un candidat stable aux trois tailles |
| `pricing-scaling-20260907-exact-terminal-geometry-01` | 324/324 jobs Kou, Merton, NIG, VG; 1 440 comparaisons numériques passent; neuf formes couvertes par modèle | alertes de coût normalisé à confirmer, pas de réglage accepté |
| `pricing-scaling-20260907-remaining-mc-calibration-1000-01` | 25/25 autres cas à 1 000 × 64k | exploration sous enveloppe de puissance variable, non qualifiante |
| `pricing-scaling-20260907-single-price-information-01` | 114/114 jobs, 44 cas MC/analytiques | un prix identifié, sans représentativité moyenne imposée |

La présélection à 1 000 × 64k pour Heston et les deux lifts N-facteurs termine
ses 62/62 jobs dans `pricing-scaling-20260907-heston-nfactor-screen-1000-01`.
Les candidats respectent les limites compilées; aucune série n'échoue.
La confirmation des quatre terminaux exacts est distincte :
`pricing-scaling-20260907-exact-terminal-confirmation-01`, 71 jobs préplanifiés
avec trois warmups exclus, sept répétitions et regroupement d'appels visant
une seconde par échantillon. Kou et Merton terminent 18/18 jobs chacun.
NIG est interrompu après dix jobs : la limite de puissance passe de 175 à
150 W, au-delà du seuil de comparaison stricte de 10 %, avec alimentation
secteur toujours déclarée. NIG reste non qualifié; VG n'a pas démarré.
Ce n'est ni un arrêt sur température ni une erreur CUDA. Aucun de ces cas
interrompus n'est automatiquement relancé.

`pricing-scaling-20260907-consolidated-summary-01.json` regroupe 800 résultats
bruts, y compris les séries partielles/non qualifiantes, et 2 559 comparaisons
numériques sans écart hors tolérance. Il conserve les enveloppes par campagne;
il ne mélange pas un minimum de présélection avec une confirmation plus chaude.
Les rapports `T(1000)/(1000*T(1))` sont renseignés mais ne testent pas le scaling :
le prix isolé européen sélectionné a une maturité de 21 jours, contre 438,11
jours en moyenne dans la tuile. Ils n'isolent donc pas le seul remplissage GPU.

## Reprise sur cinq représentants — premier lot interrompu, rough repris

Périmètre demandé : Heston (schéma), Kou (transition terminale exacte et
barrière), rough Bergomi (FFT), rough Heston (sept facteurs) et CIR (analytique
dans cette phase, LSM ensuite). Ce sous-ensemble ne qualifie pas les autres
modèles par analogie.

`pricing-scaling-20260907-representative-geometry-1000-01` termine 18/18 jobs
à 1 000 prix : CIR analytique, Kou barrière et rough Bergomi terminal/barrière;
MC à 65 536 trajectoires. Les 60 comparaisons numériques passent. Kou barrière
a un candidat stable à 896 threads (708 ms GPU); le candidat à 256 threads
dépasse le budget de CV. Rough Bergomi privilégie ici le chunk 65 536 :
3 843 ms barrière et 4 008 ms terminal. Il s'agit de comparaisons locales,
pas de configurations acceptées pour toutes les tailles.

`pricing-scaling-20260907-representative-small-matrix-01` a été lancé avec
93 jobs, prédéclarés dans le répertoire adjacent `*-small-matrix-plan-01` :

- sept couples MC × 100/1 000 prix × trois nombres de trajectoires × deux
  configurations, un warmup exclu et trois répétitions;
- CIR analytique aux trois nombres de prix, trois tailles de blocs et
  1 024 appels regroupés par échantillon;
- blocs 512/896 pour Heston et Kou barrière, 256/384 pour rough Heston,
  chunks 16 384/65 536 pour rough Bergomi; ordre des alternatives alterné.

Kou terminal conserve sa confirmation complète antérieure. Chaque nouvelle
comparaison garde ses propres médianes et conditions; aucun minimum entre
périodes ne constitue une preuve de gain. Les blocs internes FFT ne changent
pas dans cette expérience sur le chunking.

Le contrôleur séquentiel `strict`, sans veto thermique ni relance automatique,
interrompt le lot à 08:32 UTC : la limite de puissance passe de 170,54 à 150 W,
au-delà des 10 % admis, secteur déclaré avant/après. Ce n'est pas une erreur
CUDA. L'estimation initiale de 57 minutes concernait le lot entier et ne doit
pas être confondue avec sa durée d'exécution avant cet arrêt.

Au point de pause demandé à 08:57 UTC :

- 45/93 jobs dans des cas complets : CIR 9, Heston barrière 12, Heston terminal
  12, Kou barrière 12. Les 27 comparaisons numériques passent; chaque forme a
  au moins un candidat stable. Aucune comparaison de scaling de ces cas ne
  dépasse le seuil d'alerte prédéclaré, sans acceptation de profil.
- Sept résultats rough Bergomi barrière supplémentaires sont conservés mais
  non qualifiants, comme le cas entier interrompu. Trois comparaisons
  numériques passent. Rough Bergomi terminal et les deux cas rough Heston
  n'ont pas démarré.
- `summary-pause-01.json` conserve la synthèse des 52 résultats bruts et
  30 comparaisons numériques. `pause-handoff-01.json` identifie les quatre cas
  rough à exécuter plus tard (48 jobs), sans refaire les 45 jobs complets ni
  fusionner les fragments interrompus avec les mesures de la reprise.
- GPU au repos, aucune application compute; aucun nouveau lancement.
  Les grandes charges MC à 10 000 prix et le LSM restent à faire.

La reprise autorisée à 16:11 UTC lance les quatre cas rough (48 jobs) dans
`pricing-scaling-20260907-representative-rough-resume-01`. Empreintes des
binaires, données et jobs vérifiées identiques; GPU libre sur secteur et
limite courante 175 W. Les 45 jobs complets ne sont pas rejoués. Le protocole
`strict` et le watchdog restent inchangés; les sept anciens résultats rough
Bergomi ne qualifient pas la nouvelle période. Le lot s'arrête à 16:18:20 UTC
après 380,9 s sur la variation 175 → 150 W : dix résultats barrière sur douze,
cas entier non qualifiant; les trois autres cas n'ont pas démarré.
`summary-completion-01.json` conserve les résultats sans les requalifier.

`PERF-021` corrige ensuite les pilotes : une variation de limite de puissance
exclut désormais les timings sans interrompre le calcul. Télémétrie, warmups,
critères statistiques, watchdog et protections matérielles restent distincts.
La dernière priorité utilisateur est le LSM Heston/Kou/CIR, avant le solde MC
rough. Les constats de scaling restent ouverts; aucun réglage de production
n'est accepté par la correction du contrôleur.

## Début de la campagne LSM représentative

Le calibrage `pricing-scaling-20260907-representative-lsm-calibration-01`
achève les trois cas à 100 prix × 65 536 trajectoires, 128 threads et 64
blocs/prix : CIR 3 472,6 ms, Heston 193,9 ms, Kou 90,8 ms GPU. Ce sont des
mesures uniques après un warmup, sans qualification statistique ni publication.
Les ressources compilées des sept kernels sont conservées; aucune régression
fatale ni sortie non finie. Les quelques dates sans candidats ou à candidats
insuffisants restent visibles dans les diagnostics, sans être effacées.

La mémoire transitoire atteint 346,9 Mo pour CIR, 2 688,5 Mo pour Heston et
1 358,2 Mo pour Kou. Augmenter les trajectoires peut donc changer le batching
natif; l'étude ne supposera pas le même nombre de prix résidents. Les données
restent la tuile stratifiée 90/10, avec les calendriers du catalogue, pas des
échéances raccourcies pour accélérer la sonde. CIR utilise l'intégrale de taux
discrétisée; Kou des transitions exactes entre dates d'exercice.

Le screening `*-representative-lsm-screening-02` achève ensuite 18 jobs :
six géométries par modèle, 45 parités numériques toutes bit à bit identiques.
À ce seul point, CIR passe de 3 817,6 ms (128 × 32) à 3 137,9 ms
(128 × 1 024), tandis que Heston/Kou préfèrent environ 128 × 64; augmenter
les blocs pénalise ces derniers à 64k. Ce sont des observations à confirmer,
pas des réglages globaux acceptés. L'essai `*-screening-01`, refusé avant
calcul pour un champ `path_chunk` nul dans le plan, reste conservé.

À 16:49 UTC, `*-representative-lsm-path-screening-01` lance 18 jobs à
100 prix × 262k/1M trajectoires, trois candidats par forme, avec ordre inversé
au palier supérieur. Ce screening ne couvre ni les confirmations répétées,
ni les grands nombres de prix, ni le profilage par phase.
Ce lot est maintenant achevé : 18 jobs, mais cinq comparaisons numériques Kou
hors tolérance, toutes les parités CIR/Heston passant. Les mesures sont uniques.
À 1M, Heston utilise quatre batches et environ 13 389 MiB de workspace de pointe;
Kou deux batches et environ 13 533 MiB; CIR un batch et environ 5 293 MiB.

`pricing-scaling-20260907-kou-lsm-numerics-01` isole les lignes d'indices
37, 55, 57 et 77 de la tuile. Les 16 jobs rejouent bit à bit sur trois
répétitions. À géométrie identique, les 12 comparaisons avec les valeurs des
batches complets sont identiques; en changeant de géométrie, 17/24 paires
échouent. L'écart initial maximal de prix est d'environ `2,05e-5`, supérieur
au budget prédéclaré.

`pricing-scaling-20260907-kou-regression-trace-01` établit ensuite le
mécanisme sur ces quatre lignes, toutes issues du core : 20 configurations,
avec le moteur LSM de production et des policies instrumentées temporaires.
Chaque sortie instrumentée reproduit bit à bit le prix et l'erreur standard
du launcher public. Entre géométries, les tableaux complets d'états forward
et de payoffs terminaux sont identiques bit à bit : ni les trajectoires ni
le découpage en prix n'expliquent les écarts observés.

Les matrices de Gram diffèrent relativement d'environ `1e-16`, mais leur
conditionnement après ridge atteint `6e10`. Les coefficients diffèrent dès
le premier niveau backward; les différences se propagent ensuite aux
cashflows et au second membre. Le résidu backward normalisé maximal reste
`1,15e-16` : c'est une sensibilité aux arrondis de problèmes mal conditionnés,
pas une preuve que le solveur calcule mal le système fourni. Les sources,
commandes et empreintes sont conservées dans `*-kou-regression-trace-plan-01`
et `*-kou-regression-trace-build-01`; `analysis-01.json` documente les niveaux
où les écarts apparaissent. La sonde dépend volontairement du layout interne
de réduction pour ce diagnostic, sans modifier le contrat du moteur partagé.
Elle n'est pas une mesure de performance ni une certification indépendante
des prix. Kou reste ouvert; aucune base, précision ni tolérance n'est modifiée.

La synthèse a été corrigée pour que ces conflits interdisent de qualifier les
mesures concernées, même à faible CV. Les anciennes synthèses sont conservées
et les réanalyses sont nommées `summary-numerical-gate-02.json`.

`pricing-scaling-20260907-lsm-confirmation-01` achève Heston/CIR : 20 jobs,
deux warmups exclus et trois mesures, les trois tailles de trajectoires à
100 prix puis 1 000 prix × 64k avec deux géométries et batches 100/1 000.
Les réglages de référence sont ceux des recettes compilées (equity 128 × 128,
CIR 128 × 64), pas le point fixe 128 × 32 de la baseline officielle.
Les 18 comparaisons sont bitwise identiques et tous les CV GPU inférieurs
à 5 %. Durée des processus : CIR 1 669,7 s, Heston 112,8 s.

Les meilleures médianes GPU parmi les configurations testées donnent
l'enveloppe exploratoire suivante, sans acceptation de réglages :

| Prix × trajectoires/prix | Heston : secondes, threads × blocs/prix | CIR : secondes, threads × blocs/prix |
| --- | --- | --- |
| 100 × 65 536 | 0,208 ; 128 × 64 | 3,128 ; 128 × 1 024 |
| 100 × 262 144 | 0,754 ; 256 × 64 | 13,002 ; 128 × 256 |
| 100 × 1 048 576 | 3,358 ; 128 × 256 | 69,488 ; 128 × 1 024 |
| 1 000 × 65 536 | 2,388 ; 128 × 64 | 43,668 ; 128 × 64 |

Heston ne déclenche aucune alerte de coût normalisé sur les trois paires
couvertes. CIR présente deux alertes : 1,336 pour 256k → 1M à 100 prix,
1,396 pour 100 → 1 000 prix à 64k. La fréquence descend d'environ 2 325
à 1 590 MHz durant le lot : ces ratios ne prouvent pas une non-linéarité
algorithmique. À 1 000 × 64k, 1 024 blocs ne battent plus les 64 blocs
natifs CIR (44,2 contre 43,7 s). Un réglage global issu du petit lot serait
donc prématuré.

Le batching hôte est distinct du découpage VRAM natif. À 1 000 × 64k,
Heston 128 × 64 donne 2,388 s GPU et 2,773 s API en un appel hôte, contre
2,403 s GPU et 3,191 s API en dix appels de 100 prix : pas de gain GPU,
mais un surcoût API. CIR natif passe de 43,668 à 46,997 s. Le champ API est
l'enveloppe `max(host, CUDA)`, pas le temps hôte brut, que ces captures ne
conservent pas séparément. L'égalité API/GPU ne prouve donc pas un surcoût hôte
nul. La sonde a été corrigée ensuite, sans reconstruire les anciennes valeurs.
Ces mesures n'incluent pas une publication effective de base complète.

Le contrôle CIR `pricing-scaling-20260907-cir-lsm-order-control-01` a été lancé
à 18:27 UTC : mêmes données et géométrie native 128 × 64, ordre inverse
(100 prix à 1M, 256k, 64k, puis 1 000 prix à 64k), deux warmups et trois
répétitions fixes. Budget estimé 12 minutes, watchdog 1 800 s, aucune attente
thermique ni relance automatique. Il teste l'effet de l'ordre/du régime de
fonctionnement, sans fusionner les périodes ni choisir la plus favorable.
Il termine ses 4/4 jobs en 654,1 s, tous les CV GPU sous 2,2 % :

- 100 prix à 64k / 256k / 1M : 4,745 / 18,734 / 72,984 s;
- 1 000 prix à 64k : 43,825 s;
- rapports de temps pour ×4 trajectoires : 3,948 puis 3,896;
- rapport de temps pour ×10 prix : 9,235.

Les coûts normalisés (0,987 / 0,974 / 0,924) ne déclenchent aucune alerte.
Les quatre sorties prix/erreurs égalent bit à bit celles des configurations
natives correspondantes du premier lot.
Ce résultat appuie l'explication par le régime de fonctionnement; il ne prouve
pas une linéarité universelle. Les alertes et mesures du premier lot restent
visibles, sans changement de kernel ni sélection entre périodes.

`pricing-scaling-20260907-heston-lsm-large-01` achève ensuite six jobs :
1 000 prix à 256k/1M et 10 000 prix à 64k, deux géométries par forme, deux
warmups et trois répétitions. Durée 655,5 s, watchdog 1 800 s.
Son plan adjacent fixe les six jobs et alterne l'ordre
natif/candidat. Même binaire et mêmes données sources, un appel hôte par
opération avec le découpage VRAM natif; aucun calcul concurrent ni retry.

| Charge Heston | Natif 128 threads × 128 blocs/prix | Candidat | Batches natifs |
| --- | ---: | --- | ---: |
| 1 000 × 262 144 | 8,734 s | 8,913 s ; 256 × 64 | 8 |
| 1 000 × 1 048 576 | 33,147 s | 33,821 s ; 128 × 256 | 32 |
| 10 000 × 65 536 | 27,035 s | 23,374 s ; 128 × 64 | 19 |

Les trois comparaisons prix/erreurs sont bitwise identiques, tous les CV GPU
sous 2 %, workspace de pointe voisin de 14,2 Go. La limite de puissance
175 → 150 W rend cependant le lot non qualifiant pour une décision de retuning;
aucun arrêt applicatif ni relance n'a eu lieu. Le ratio 256k → 1M à 1 000
prix vaut 3,795 (coût normalisé 0,949), informatif seulement. Il manque un
point 1 000 × 64k dans cette même période pour qualifier l'axe prix à 10 000;
le point d'une autre période ne le remplace pas. Le candidat 64 blocs à
10 000 prix mérite une confirmation, sans devenir un défaut global.

Les ressources compilées de ces mêmes binaires restent disponibles, sans
profilage GPU concurrent. Registres par thread des trois phases principales :

| Phase | Heston | CIR | Kou |
| --- | ---: | ---: | ---: |
| Simulation | 69 | 70 | 70 |
| Statistiques de régression | 88 | 76 | 86 |
| Résolution | 82 | 62 | 82 |

Pour les sept kernels de chaque cas, stack/local et instructions SASS
`LDL`/`STL` sont nuls sur ces binaires. À 128 threads, l'occupation théorique
des statistiques Heston/Kou est de 41,7 %, contre 50 % pour CIR. Ce n'est
pas un diagnostic de goulot à lui seul : il manque les temps et compteurs
par phase. Aucun plafond de registres, changement de précision ou duplication
de moteur n'est justifié par ces seules valeurs.

Après la fin du lot et une compilation séquentielle, la sonde conserve le
temps hôte brut (`raw_host_clock`, `raw_host_samples_ms`) indépendamment
de l'enveloppe API. `pricing-scaling-20260907-raw-host-clock-check-01`
vérifie les deux branches du harness : CIR LSM, Heston LSM et terminal MC,
trois jobs courts en observation seule. Prix/erreurs identiques bit à bit
aux références préchangement, ressources compilées inchangées, statistiques
et relation API = max(host, GPU) vérifiées. Les anciens bruts restent intacts;
leur temps hôte manquant est `null` dans une nouvelle synthèse, jamais estimé.
61 tests Python et deux CTest ciblés passent.

`PERF-019` reste ouvert : sensibilité numérique Kou non corrigée, grands lots
CIR, qualification temporelle Heston, 10 000 prix à 256k/1M et profilage
par phase/pipeline encore à couvrir. Aucun changement de production
ni rebaseline n'est accepté à partir de ces seuls lots.

## Observations, sans extrapolation

1. **La géométrie dépend de la charge.** Kou terminal à 1M trajectoires/prix
   donne environ 23 ms pour 100 prix, 64 ms pour 1 000, puis 620 ms pour
   10 000. Les gros blocs aident le petit lot; à 10 000 prix, l'écart entre
   plusieurs tailles de blocs devient faible. Le rapport 100 → 1 000 n'a
   aucune raison d'être ×10 tant que le remplissage du GPU change.
2. **Une grille de candidats doit respecter les registres.** Rough Heston
   terminal utilise 151 registres/thread et autorise au plus 384 threads/bloc
   sur ce binaire. 512 est rejeté avant lancement; 128 et 256 fonctionnent.
   Le pilote sait désormais reprendre cette limite depuis la calibration du
   même binaire et GPU, sans plafonner les registres ni changer les kernels.
3. **Le pipeline n'est pas seulement le GPU.** La calibration rough Heston
   terminal prépare 100 lignes en 4 761 ms, pour 694 ms GPU. Cette préparation
   n'est pas une régression CUDA et ne se corrige pas avec des blocs plus gros.
   Sur 1 000 lignes répétées, rough Heston réutilise les fits exacts par Hurst
   et reste proche de 5 s; QRH refait les fits et prend environ 44 s. La tuile
   ne qualifie donc pas le coût de préparation de 1M paramètres distincts.
   L'interpolation déjà qualifiée pour les samples (`NUM-007`) ne remplace
   pas automatiquement les fits exacts contractuels des pricers.
4. **Les alertes ne prouvent pas encore une cause dans le code.** À 10 000
   prix, l'enveloppe des géométries donne un coût normalisé 256k → 1M de
   1,23 pour Kou, 1,32 pour Merton et 1,30 pour VG. Les fréquences varient
   durant ces séries. Dans la confirmation achevée, les ratios 256k → 1M
   deviennent **1,037 pour Kou et 1,032 pour Merton**; 64k → 256k donne
   respectivement 1,011 et 0,985. Les deux alertes initiales ne sont donc
   pas reproduites. VG attend sa confirmation; les constats globaux restent
   ouverts faute de couverture complète des autres modèles et configurations.

La campagne effective 1M prix × 1M trajectoires, les samples, la validation
indépendante et les autres GPU physiques sont exclus. Les mesures actuelles
ne certifient ni le temps à cette échelle ni une géométrie portable.

## Durée de la suite et vérifications

Le budget indicatif `pricing-scaling-20260907-budget-01.json` part des 35
mesures à 1 000 × 65 536, soit 66,89 s GPU cumulées. Sous hypothèse de
linéarité des deux axes, une matrice complète avec **une géométrie, un warmup
et trois mesures** représente environ **17,3 h GPU**. Ce n'est ni un temps
certifié ni une borne : préparations hôte, géométries supplémentaires,
profilage par phase et LSM s'ajoutent. Il faut réserver des sessions longues,
pas annoncer la clôture après les seuls calibrages.

Vérifications hôte : 55 tests scaling/protocole et cinq tests de l'outillage
LSM passent; deux CTest ciblés passent; `git diff --check` passe. Les 61
sondes ont compilé. La correction ultérieure du champ `time_kind` du manifeste
de sondes (exact/fixe/non applicable selon la signature publique) ne change
aucun des 61 fichiers CUDA générés. Aucun kernel ni défaut de lancement de
production n'a été modifié par ce passage.
