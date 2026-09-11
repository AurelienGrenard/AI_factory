# CIR Bermudan sous mesure forward — 8 septembre 2026

**Intégration : le pricer CIR Bermudan utilise désormais la mesure forward.**
La préparation sans tableau fixe reproduit bit à bit les 2 000 prix et erreurs
standards du prototype qualifié, à mêmes graines. Le filtre PDE est donc
inchangé. Les mesures initiales et leurs écarts avec Q sont conservés ci-dessous ;
elles ne constituent pas une certification de publication.
Le pricer réutilise les kernels LSM, la base d'Hermite cubique
et les régressions/réductions FP64 existants. Seuls le calendrier de simulation,
le stockage du taux et l'expression du payoff dans le nouveau numéraire changent.

## Méthode et périmètre

Le numéraire est le zéro-coupon de maturité `T*`, dernière date d'exercice.
Sous sa mesure forward, le drift CIR devient
`kappa*theta - (kappa + sigma²*B(t,T*))*r`. La transition reste un chi-deux
non central redimensionné. Chaque cashflow est exprimé comme
`H(t)*P(0,T*)/P(t,T*)` : l'intégrale trajectorielle d'actualisation disparaît.
Voir [Brigo–Mercurio, équations 17–18 et 21](https://www.ma.imperial.ac.uk/~dbrigo/detshiftrep.pdf).

Il ne s'agit pas de simuler sous la mesure risque-neutre en supprimant
l'actualisation. Le changement de loi et celui du payoff sont indissociables.
Les obligations du swap sont calculées analytiquement à chaque exercice ;
leurs dates de paiement ne nécessitent pas de nouvelles simulations du taux.
L'identité de changement de mesure est exacte, pas la régression LSM de degré
fini : son biais peut changer et doit être contrôlé séparément.

- Même catalogue aligné : 900 lignes ordinaires et 100 lignes de stress,
  `2^20 = 1 048 576` trajectoires par prix, côtés payeur et receveur.
- Ancienne méthode : launcher de production, transitions du taux à `1/504`,
  intégrale par trapèzes. Nouvelle méthode : transitions aux exercices seuls.
- 128 threads/bloc, 64 blocs/prix, tranches appelantes de 32 prix pour les deux
  méthodes. Les clés Philox gardent l'indice original, même sur une sélection
  non contiguë. Les graines des deux mesures sont indépendantes.
- Selon la ligne, 4 à 19 656 transitions anciennes contre 2 à 12 nouvelles ;
  moyennes respectives de 6 492,4 et 6,12 par trajectoire.
- RTX 4090 Laptop, SM89 ; NVCC/CUDA 13.3, GCC 14, Release, sans fast math.
  Révision `872a986b1f0947a1a832af0615ffc6d80dbedb81`, worktree déjà modifié.
  Les sources, empreintes binaires/entrées, état Git et télémétries sont
  conservés dans les répertoires bruts référencés plus bas.

## Prix ligne par ligne

Le filtre annoncé avant les grandes séries est
`abs(nouveau-ancien) <= 5*hypot(SE_ancien, SE_nouveau) + 2e-6`.
C'est un détecteur d'écarts, **pas une certification financière**, ni une
borne du biais LSM. Deux prix nuls avec erreur standard nulle ne démontrent
pas une bonne précision relative sur un événement rare.

Les [1 000 lignes payeuses](../../tests/performance/reports/cir-forward-measure-sm89-2026-09-08/payer/rows.csv)
et les [1 000 lignes receveuses](../../tests/performance/reports/cir-forward-measure-sm89-2026-09-08/receiver/rows.csv)
conservent les deux méthodes, les erreurs standards, une deuxième graine
forward et l'EDP. La synthèse correspondante contient les empreintes des sources.

| Comparaison au filtre annoncé | Payeur | Receveur |
|---|---:|---:|
| Ancien contre nouveau, lignes compatibles | 997/1 000 | 999/1 000 |
| Nouveau contre EDP, chacune des deux graines | 1 000/1 000 | 1 000/1 000 |
| Ancien contre EDP | 995/1 000 | 995/1 000 |
| Nouveau : première contre deuxième graine | 1 000/1 000 | 1 000/1 000 |

Sur le côté payeur, 997/1 000 lignes passent le filtre ancien/nouveau.
Les 1 000 anciens prix **et erreurs standards** reproduisent exactement la
publication de la campagne précédente. Les nouveaux prix et erreurs standards
sont également identiques lors d'une relance à graine inchangée ; la seconde
graine forward passe le filtre sur les 1 000 lignes.

| Ligne payeuse | Ancien prix | Nouveau prix | EDP affinée |
|---|---:|---:|---:|
| 000363 | 0,0220505465 | 0,0219629500 | 0,0219802484 |
| 000591 | 0,0014652873 | 0,0014005030 | 0,0013974727 |
| 000785 | 0,0190393366 | 0,0187793039 | 0,0188048920 |

L'EDP indépendante travaille sous la mesure risque-neutre originale, avec
terme d'actualisation `-r*V`, sans transition forward ni régression. Les grilles
384/768 et 64/128 pas annuels couvrent toutes les lignes payeuses. Le filtre
par méthode ajoute `2*écart_de_maillage` à `5*SE + 2e-6` : aucun échec forward,
pour chacune des deux graines ; cinq échecs anciens, lignes 363, 571, 591, 677
et 785. Cette marge empirique de maillage n'est pas une borne rigoureuse.

Sur ces cinq lignes : affinement jusqu'à 3 072 nœuds/512 pas annuels,
LSM CPU indépendant avec apprentissage et évaluation disjoints de `2^20`
trajectoires chacun, nouvelle graine GPU et essais anciens à 252/504/1 008/2 016
pas annuels. Le nouvel estimateur est compatible avec ces références.
L'ancien prix ne converge pas monotoniquement lorsque son pas diminue :
attribuer les écarts aux seuls trapèzes n'est pas justifié.
Un doublement de la frontière haute de l'EDP, à résolution locale comparable,
change les cinq références de moins de `4e-9`.

Côté receveur, seul le couple 000591 échoue à la comparaison ancien/nouveau :
`0,0743303001` contre `0,0749551356`, pour une EDP affinée à `0,0749434780`.
Les cinq échecs anciens contre EDP sont 396, 591, 700, 785 et 892.
L'affinement 3 072/512 et une évaluation CPU disjointe de l'apprentissage
(`2^18` trajectoires par partition) sont compatibles avec la nouvelle méthode.

### Diagnostic de l'ancien calcul, sans LSM

Une sonde séparée simule cinq modèles sur dix ans, avec 252/504/2 016 pas
annuels et `2^18` trajectoires. Elle compare les moments du taux aux formules
exactes et les obligations actualisées à leurs prix analytiques. Les tirages
et taux restent ceux de production ; seul un accumulateur FP64 parallèle
sert de témoin de l'intégrale FP32. Aucun code runtime n'est corrigé ici.

Les biais existent déjà **avant la régression**. À `dt=1/504`, sur la ligne
591, le taux moyen terminal vaut `0,0366427208` contre `0,0365148811`
attendu, soit 6,86 erreurs standards. Le zéro-coupon vaut `0,6232032787`
contre `0,6238086383`, soit un écart de 7,99 erreurs standards.
L'accumulation FP64 ne corrige pas ce résultat ; sur toute la sonde, son
effet moyen apparié sur l'actualisation reste inférieur à `1,71e-8`.
Les moments obtenus avec une seule transition longue passent les contrôles.

Cela localise un problème dans la chaîne de simulation à petits pas et
écarte l'explication « seulement l'accumulation de l'intégrale ».
La responsabilité précise — coefficients FP32, tirage Poisson–Gamma,
propagation des arrondis — **n'est pas isolée**. Le diagnostic donne 27
échecs sur 100 contrôles corrélés, pas 27 expériences indépendantes.
La nouvelle méthode réutilise le tirage partagé : ce point mérite aussi une
investigation pour ses intervalles très courts avant généralisation.

## Temps et ressources

| Côté payeur, 1 000 prix × 2²⁰ trajectoires | Ancien | Nouveau |
|---|---:|---:|
| Temps GPU, passage principal | 783,73 s | 3,126 s |
| Temps hôte brut autour des appels | 785,90 s | 4,388 s |
| Workspace maximal d'une tranche | 2 349 095 424 octets | 1 241 815 040 octets |

Pour les 1 000 receveurs : **788,95 s contre 2,990 s GPU**,
791,29 s contre 4,527 s hôte brut, mêmes maxima de workspace. La deuxième
graine forward donne 3,045 s GPU et 4,668 s hôte brut.

Deux autres passages forward donnent 2,535 et 2,574 s GPU. Le gain principal
observé est d'environ 251× GPU, 179× sur l'horloge hôte. Ce n'est pas une
qualification de performance : répétitions limitées, travail CPU de référence
parfois simultané, fréquences variables et géométrie non optimisée pour le
nouvel algorithme. Le temps hôte exclut le chargement JSON initial, les copies
d'entrée et la copie finale des résultats ; aucune publication n'est chronométrée.
Les mesures ne remplacent pas celles du générateur à découpage VRAM natif et
ne garantissent pas un débit sur un autre GPU ou sur un million de prix.
La température est une télémétrie, sans seuil d'arrêt applicatif.

Le kernel de trajectoires passe de 70 à 64 registres/thread, sans pile locale
compilée. En revanche la préparation du prototype utilise 162 registres/thread
et 536 octets de pile/thread, contre 38 et 0 auparavant. Sa limite compilée
est de 384 threads/bloc : **512 threads ne sont pas admissibles en l'état**.
La pile observée ne permet pas, seule, d'affirmer un nombre de spills.

## Qualification initiale et intégration

La décision initiale demandait les quatre contrôles suivants :

1. Les écarts avec l'ancien calcul : préférer une référence indépendante
   étayée, sans ajuster artificiellement le nouveau prix pour reproduire un biais.
2. La qualification numérique : budgets absolus/relatifs contractuels,
   événements rares, biais d'exercice hors apprentissage du GPU, références
   indépendantes selon le workflow du projet. L'EDP expérimentale et les
   contrôles QuantLib analytiques ne remplacent pas cette certification.
3. L'intégration : sortir les tables de coefficients des tableaux fixes de
   32 exercices, conserver les kernels LSM communs, ajouter une policy de
   numéraire explicite et des guards/test de calendrier. Ne pas modifier la
   loi risque-neutre des samples CIR en changeant celle du pricer Bermudan.
4. La qualification des ressources et du débit du générateur réel, puis la
   sélection de géométrie par charge et architecture. Aucun nouveau défaut
   codegen/catalogue n'est fixé par cette exploration.

La bascule ultérieure remplace le tableau fixe par une région de coefficients
par exercice dans le workspace LSM. Les sources Q et samples ne changent pas.
`build-dev/cir-forward-production-20260908` contient le contrôle des 1 000
payers et 1 000 receivers à `2^20` chemins : prix et SE identiques bit à bit
aux exports précédents. Mesures à 128 threads, 64 blocs/prix, chunks de 32 :
2,6114 s GPU / 3,9341 s hôte pour payer, 2,6156 s / 3,8306 s pour receiver.
Workspace maximal : 1 241 803 136 octets. La préparation passe de 162 à 40
registres, de 536 à 0 octet local déclaré ; la simulation reste à 64 registres
et 0 octet local. Le lancement 512 threads / 16 blocs passe sur les huit
lignes de contrôle. Ces timings excluent la publication et ne sont pas une
nouvelle baseline portable.

Les 96 contrôles de loi de production contre SciPy/QuantLib passent
(`cir-forward-production-transitions-checked-20260908.json`). Les contrôles
memcheck et racecheck ciblés passent sans erreur/hazard. La certification
indépendante persistante core/stress reste une étape distincte avant toute
publication marquée `verified: true` ; elle n'est pas déduite du seul filtre PDE.

### Recettes de production intégrées

Les deux recettes ont ensuite généré et relu leurs 1 000 lignes à `2^20`
trajectoires, avec leurs graines de catalogue inchangées. Leurs JSON/YAML
décrivent désormais la transition forward exacte, sans `dt`, et restent
`pending` / `verified: false` avant certification persistante.

| Recette | GPU | Runner hôte | Processus complet |
|---|---:|---:|---:|
| Payer | 2,707 s | 5,636 s | 5,73 s |
| Receiver | 2,867 s | 3,668 s | 3,77 s |

Ce passage isolé emploie 128 threads/bloc et 64 blocs/prix. Le planner VRAM
natif choisit trois batches et un workspace maximal de 14 160 229 560 octets,
contrairement aux tranches de 32 prix de la comparaison initiale. Ce maximum
est une allocation adaptée à la mémoire libre, pas une mémoire minimale
requise. Le runner inclut initialisation, warmup et copies ; le timer GPU les
exclut, le processus complet inclut chargement et écriture/relecture JSON/YAML.
Le premier processus subit davantage de coût d'initialisation : les temps
hôtes ne servent pas à comparer payer et receiver. Les logs portent le nom
`build-dev/generate_cir_bermudan_<side>_swaptions_01-20260908.log`.
Ces mesures ne constituent ni un tuning optimal ni une rebaseline.

## Reproduction et preuves

La [notice de qualification](../../tests/performance/cir_forward_measure/README.md)
décrit les cibles CMake opt-in, scripts et fixtures. Les essais GPU sont
séquentiels et bornés, sans nouvelle tentative automatique. Les mesures
initiales n'ont remplacé aucun dataset ou cache de validation. La notice
courante pointe vers le launcher intégré ; les anciens corps expérimentaux
ne sont pas conservés en parallèle dans les sources.

Sources brutes sous `build-dev/` :

- `cir-forward-catalogue-20260908-02` : ancienne/nouvelle méthode payeuse.
- `cir-forward-receiver-20260908` : répétitions payeuses et comparaison receveuse.
- `cir-forward-pde-20260908`, `cir-forward-receiver-pde-20260908` : EDP catalogue.
- `cir-forward-discrepancies-20260908` : graines et pas anciens sur cinq lignes.
- `cir-forward-pde-discrepancies-20260908` : EDP affinée et LSM CPU hors apprentissage.
- `cir-forward-receiver-pde-discrepancies-20260908` : mêmes contrôles sur les
  cinq receveurs signalés par l'EDP.
- `cir-forward-discount-chains-20260908.ndjson` et
  `cir-forward-discount-chains-checked-20260908.json` : diagnostic sans LSM.
- `cir-forward-transitions-20260908.json` et
  `cir-forward-transitions-checked-20260908.json` : 24 cas, 96 contrôles réussis
  de moments, zéro-coupons actualisés et options sur obligations.

Les contrôles Compute Sanitizer `memcheck` et `racecheck` passent sur huit
lignes par côté à 4 096 trajectoires : zéro erreur, zéro hazard, zéro warning.
Logs : `build-dev/cir-forward-memcheck-20260908.log` et
`build-dev/cir-forward-racecheck-20260908.log`. Ce petit workload ne certifie
pas toutes les tailles de calendrier ni toutes les architectures.

Les dix tests CPU couvrent moments contre ODE indépendante, obligations et
options contre QuantLib, et rejet des comparaisons à indices/côtés/volumes
incohérents. Les contrôles QuantLib sont explicitement analytiques ; ils
n'ont régénéré aucun cache de certification.
