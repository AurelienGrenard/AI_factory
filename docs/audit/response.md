# Constats d'audit non résolus

Référentiel actif : **v10.1 — 2026-10-02**.

Les anciens chemins de preuves `build-*` se retrouvent via le
[plan des artefacts locaux](../local-artifacts.md).

## Report vers le référentiel v10

Ce registre contient **quatre constats ouverts** : `PERF-025` et
`PERF-026` pour la performance, `NUM-028` et `NUM-030` pour les prix rough.
`PERF-024` est réglé par acceptation explicite du compromis mémoire/temps,
sans passage du gate de performance. `NUM-029` a été corrigé et publié ;
`NUM-036` a défini la portée de la SE LSM ; ces constats sont dans
[closed.md](closed.md), avec leurs limites de périmètre et leurs preuves.
`NUM-037` et `PERF-027` sont clos administrativement mais leur problème
numérique reste non résolu : les 26 paires SABR markoviennes sont exclues des
références, et la Hessienne CIR–Jamshidian n'est plus exposée. Leurs preuves et
conditions de résolution figurent dans
[unresolved-closures.md](unresolved-closures.md). Ce registre ne contient que les constats ouverts. Les cinq constats
`BOUNDARY-001`, `STRUCT-001`, `FACTOR-004`, `FACTOR-005` et `NAME-014`
ont été corrigés et transférés dans [closed.md](closed.md) lors de la reprise
ciblée du 2026-10-06. Les identifiants `BOUNDARY-001` et `STRUCT-001`
reprenaient des clôtures historiques archivées dans ce même fichier. `BOUNDARY-005` a été
refermé pendant le passage hygiène du 2026-10-06. Les constats
`NUM-035`, `FACTOR-003` et `BOUNDARY-008` du passage ciblé du
2026-10-05 ont été corrigés et transférés dans [closed.md](closed.md) le
2026-10-06. Les six constats
`DELTA-001`, `NUM-031` à `NUM-034` et `PERF-016` ont été transférés dans
[closed.md](closed.md) par décision de l'utilisateur du 2026-10-02.

Plusieurs descriptions sont antérieures aux migrations récentes du code. Certaines
signatures encore non rejouées sur le binaire courant doivent être confrontées
au snapshot, puis :

- mettre à jour le constat si le risque subsiste sous une forme différente ;
- transférer dans [closed.md](closed.md) une correction vérifiée, un contrat
  explicitement borné ou un compromis accepté, en conservant les résultats
  défavorables et les limites du périmètre ; si le problème numérique reste
  ambigu malgré une exclusion ou qualification des données, transférer dans
  [unresolved-closures.md](unresolved-closures.md) jusqu'à sa résolution ;
- ouvrir un nouvel identifiant seulement si le défaut observé est distinct.

| Constat | Axe principal v10 | Axes secondaires |
|---|---|---|
| `NUM-028` | I. Véracité | IV. Performance |
| `NUM-030` | I. Véracité | IV. Performance |
| `PERF-025` | IV. Performance | — |
| `PERF-026` | IV. Performance | V. CMake |

L'absence de constat reporté ayant V ou VI comme axe principal ne constitue
pas un verdict de conformité. La couverture courante est publiée dans
[status.md](status.md).

## Sélection du catalogue source — 2026-10-07 (étape fixed income)

Après réorganisation, le catalogue livrable contient 1 195 couples
`generator.cpp`/`recipe.yaml` : 25 modèles, 28 produits, 3 courbes, 655 prix
alignés et 484 prix avec gradients alignés. Les recettes cartésiennes et les
samples restent locales. Les 24 prix fixed income sur courbe flat ont été
réintégrés après un test d'exhaustivité ; le catalogue fixed income contient
117 prix et 104 recettes prix + gradients. Les 26 gradients de swaption
bermudéenne sélectionnés utilisent `frozen_regression_policy`, tandis que CIR
standalone/Jamshidian reste limité à l'ordre un. Le call up-and-out a 13
recettes de prix mais pas de binding de gradients. Le codegen comparé au
dépôt, 49 tests dataset, le plan de 221 jobs sur un paquet public sans `work/`,
et la compilation de cinq générateurs représentatifs passent. Le générateur
de paramètres du produit barrière a été exécuté en staging (1 000 lignes et
reçu authentique). La campagne des 221 prix et gradients fixed income est
prête mais n'a pas été exécutée. Les 618 anciens prix physiquement présents
ne deviennent pas des références des nouveaux paramètres par ce déplacement.
`NUM-028` et `NUM-030` restent ouverts ; aucune clôture numérique nouvelle
n'est revendiquée.

## Reprise complète des paramètres — 2026-10-07

Les [25 modèles et 27 produits canoniques](parameter-catalog-refresh-2026-10-07.md) ont été régénérés à partir de leurs `generator.cpp` avec de nouvelles seeds pour les tirages aléatoires, puis publiés avec recettes et reçus authentiques. Les cœurs ont été conservés sauf les corrélations positives de Schöbel–Zhu, rough Stein–Stein et G2/G2++, le domaine cœur SABR à forte variance MC et le calendrier cœur Bermudan trop long. Les stress ont été resserrés sans supprimer ni écrêter de lignes ; les 52 jeux conservent 900 lignes cœur et 100 stress. Les quatre paires européennes rough testées et la paire SABR affinée ont zéro signal de parité >5 SE et zéro prix matériel à SE relative >25 %. Les anciens prix canoniques n'utilisent pas ces nouvelles entrées : `NUM-028` et `NUM-030` restent ouverts jusqu'au repricing, aux contrôles de tous les payoffs et à la décision sur l'estimand continu QRH. Les [entrées historiques figées](../../work/catalog/archives/parameter-catalog-20261007/manifest.json) préservent la vérification des releases et des masques antérieurs : NUM-029, NUM-030, rough Bergomi et le gate de domaine passent sur leurs anciens prix sans attribuer ces prix aux nouveaux paramètres.

## Qualité des prix rough alignés publiés

Le contrôle ciblé des 174 datasets de prix seuls compare les calls/puts
européens de même modèle et de mêmes lignes d'entrée. L'écart diagnostique est
`C - P - (S0 exp(-qT) - K exp(-rT))`, avec `T = maturity_days / 252`.
Une ligne est signalée si sa valeur absolue dépasse à la fois cinq erreurs
standards combinées et 0,5 % de `max(S0, K)`. Ce n'est pas une référence de
prix indépendante : la cause d'un écart peut être une queue rare, le biais de
discrétisation, le statut martingale du modèle ou une erreur de code.
[Le script et son résultat local](../../artifacts/audit/rough-price-quality-2026-09-14/result.json)
figent la règle et les nombres. Le compte rendu détaillé appartient aux preuves
historiques antérieures à v10 ; sa compatibilité avec le prochain snapshot doit
être réétablie avant réemploi. La reprise du 2026-10-06 utilise deux builds
isolés du worktree courant ; [son manifeste de preuves](../../artifacts/audit/rough-price-quality-2026-10-06/result.json)
fige les hashes des sources, binaires et replays.

### NUM-028 — Qualifier les prix quadratic rough Heston dominés par les queues extrêmes

- **État / date / propriétaire :** ouvert le 2026-09-14 ; propriétaire :
  moteur quadratic rough Heston et qualification des prix MC.
- **Sévérité / priorité / confiance :** haute / haute / élevée sur les écarts
  et l'incertitude observés, cause encore indéterminée.
- **Contrat / localisation :** [contrat MC](../cuda/closed-form-and-monte-carlo-pricing-contract.md),
  `src/model/equity/rough/quadratic_rough_heston/`,
  `datasets/model/equity/rough/quadratic_rough_heston/prices/`.
- **Preuve reproductible :** les 29 datasets de ce modèle ont 1 000 prix
  finis et 2²⁰ chemins par prix. Le lookback ligne 770 affiche
  `1053.8477783 ± 1051.7395020` en erreur standard, pour `S0=1` et
  `K=0.9663277`. Sur les prix `>0.01`, 59 lignes core et 27 stress ont
  `SE/prix >25 %`. La parité diagnostique échoue sur 107 lignes core et
  62 stress, dont la ligne européenne 984 avec un résidu `-0.77448`
  pour une erreur combinée `3.35e-6`.
- **Revalidation du 2026-10-02 :** les JSON présents à `HEAD`
  `55093962278a50c82c1c1f5f1b5a34e981b3cba5` conservent, à l'id
  `000984`, `C=0.00000579437665`, `P=0.56904321909`, `S0=1`,
  `K=0.66028785706`, `T=1569/252` et un résidu
  `-0.77447979196` pour `SE=0.00000335031842`, calculé avec la formule de
  parité ci-dessus sur les JSON `european_calls` et `european_puts` du modèle.
  Les tests CUDA recompilés passent sur une autre configuration ; la ligne
  `000984` n'a pas encore été rejouée sur ce binaire.
- **Reprise sur le build isolé du 2026-10-06 :** les 169 lignes de parité
  signalées (107 core, 62 stress) ont été rejouées avec trois couples
  chemins/graines chacune : 507 évaluations, dont 99 lignes persistent au
  seuil sur les trois replays. La ligne 984 donne un premier moment actualisé
  du spot de 4,12e-5 à 262 144 chemins contre 0,774518 attendu ; le
  résidu vaut −0,774477 ± 7,34e-6. À 1, 2, 4 pas/jour, ce moment descend de
  2,70e-4 à 4,12e-5 puis 3,14e-6. Avec 2, 3, 7 facteurs, il reste
  respectivement près de zéro, près de zéro et 4,12e-5
  ([replays facteurs](../../artifacts/audit/rough-price-quality-2026-10-06/quadratic_factor_replays.jsonl)).
  Sur un million de chemins, les dix plus grands payoffs call de cette ligne portent 98,4 %
  de leur moyenne ; 44 550 spots terminaux sont nuls en FP32
  ([queue européenne 984](../../artifacts/audit/rough-price-quality-2026-10-06/quadratic_european_tail_replay.jsonl)).
  Pour le lookback 770, quatre replays de 262 144 à 1 048 576 chemins donnent des
  moyennes de payoff de 1,78 à 2,42, des maxima de 1,30e4 à 1,62e5 et une
  contribution des cent plus grands de 25 % à 44 %. Les seeds, quantiles
  et configurations figurent dans [les 169 replays](../../artifacts/audit/rough-price-quality-2026-10-06/quadratic_flag_replays.jsonl),
  [les queues lookback](../../artifacts/audit/rough-price-quality-2026-10-06/quadratic_lookback_tail_replays.jsonl)
  et [le raffinement](../../artifacts/audit/rough-price-quality-2026-10-06/temporal_refinement_replays.jsonl).
  [Résumé des 507 replays](../../artifacts/audit/rough-price-quality-2026-10-06/quadratic_sweep_summary.json).
- **Domaine candidat du 2026-10-07 :** le générateur de paramètres borne
  `a`, `λ` et `η` pour les longues maturités du catalogue sans écrêter la variance ou les prix. Les 1 000 paires européennes
  exécutées dans le build courant ont 0 signal de parité et 0 prix >0,01 à
  `SE/prix >25 %`. La ligne 984 reste entre 99,958 % et 100,073 % du
  forward sur 18 replays (2/3/7 facteurs, 1/2/4 pas/jour, deux graines) ;
  le lookback 770 donne 0,2171–0,2202 sur neuf replays et ses 100 plus grands
  payoffs pèsent au plus 0,196 % de la moyenne.
  [Preuves de cette campagne antérieure](aligned-mc-model-domain-2026-10-07.md). Le nouveau catalogue modèles/produits est documenté dans la [reprise complète](parameter-catalog-refresh-2026-10-07.md). Le modèle
  continu quadratique non borné n’a pas de preuve de martingalité ici ; le
  candidat n’est pas encore publié pour les 29 payoffs.
- **Preuve manquante pour clore :** qualifier le domaine et l'estimand du
  premier moment, obtenir une estimation de queue stable ou une référence
  indépendante, puis publier ou écarter explicitement les jeux concernés.
  Le pas log-Euler du code satisfait formellement l'identité de premier
  moment conditionnel en arithmétique exacte pour une variance connue en
  début de pas ; la chute empirique se localise donc dans la queue non
  observée, le domaine ou la précision, sans preuve suffisante pour les
  départager. L'erreur standard des chemins observés ne couvre pas la queue
  manquante. Aucune correction de dynamique ni régénération n'est fondée à
  ce stade ; les prix historiques restent non qualifiés comme références
  sur ces lignes.
- **Conséquence / portée :** la finitude et les hashes corrects ne suffisent
  pas à qualifier ces étiquettes pour l'apprentissage ou la comparaison de
  méthodes. Les deux symptômes peuvent avoir des causes différentes ; ce
  constat ne tranche ni biais du modèle ni défaut de l'estimateur.
- **Correction minimale proposée :** rejouer des lignes core et stress ciblées
  avec graines indépendantes ; mesurer la convergence en chemins, temps et
  facteurs, le premier moment actualisé du spot et les quantiles des payoffs.
  Corriger la dynamique, le domaine ou la méthode d'estimation seulement
  après localisation de la cause ; régénérer les sorties affectées avec
  provenance. Ne pas simplement écrêter les prix extrêmes.
- **Clôture vérifiable :** domaine admissible et estimation MC qualifiés sur
  les lignes signalées, incertitude reproductible par répétitions indépendantes,
  écart de premier moment expliqué ou corrigé, et datasets concernés
  régénérés/écartés explicitement. `NUM-007` demeure clos pour son ancienne
  signature (samples non finis) ; ce constat vise la qualité de prix de la
  campagne publiée.

### NUM-030 — Qualifier la queue stress des prix log-modulated rough Bergomi

- **État / date / propriétaire :** ouvert le 2026-09-14 ; propriétaire :
  moteur log-modulated rough Bergomi FFT et qualification des prix MC.
- **Sévérité / priorité / confiance :** moyenne / moyenne / élevée sur le
  signal local ; domaine martingale continu établi, prix des queues encore incertains.
- **Contrat / localisation :** [contrat MC](../cuda/closed-form-and-monte-carlo-pricing-contract.md),
  `src/model/equity/rough/log_modulated_rough_bergomi/`,
  `datasets/model/equity/rough/log_modulated_rough_bergomi/prices/`.
- **Preuve reproductible :** huit écarts de parité sur 100 lignes stress,
  zéro sur les 900 lignes core au seuil commun. Ligne 948 : résidu
  `-0.0090962`, `SE` combinée `0.0004228`, soit environ 21,5 fois celle-ci.
  Six prix stress `>0.01` ont aussi `SE/prix >25 %` ; aucun core dans ce
  critère. Les fichiers et leurs empreintes de publication sont intègres.
- **Revalidation du 2026-10-02 :** les JSON courants conservent à l'id
  `000948` un résidu `-0.00909622949` et une erreur combinée
  `0.00042279761`, soit `21.5` fois cette erreur. Le diagnostic est
  reproductible avec la formule ci-dessus sur les JSON alignés
  `european_calls` et `european_puts` ; le binaire courant n'a pas encore
  repricé cette ligne.
- **Reprise sur le build isolé du 2026-10-06 :** le balayage hashé des 900
  lignes core publiées ne retrouve aucun signal au seuil ; vingt lignes core
  réparties sur cette plage, rejouées à 262 144 chemins, n'en montrent aucun.
  Les huit lignes stress signalées ont été rejouées avec trois couples
  chemins/graines : à un million de chemins, les huit écarts persistent. La
  ligne 948 donne −0,00900 ± 0,000423 contre le premier moment attendu
  0,927773. Ses résidus à 1, 2, 4 pas/jour restent −0,00716, −0,00900,
  −0,00816 à un million de chemins : pas de convergence nette. L'intégration
  conditionnelle du bruit spot indépendant donne 0,920259 ± 0,000169
  de premier moment contre 0,919778 ± 0,001109 brut ; cette réduction de
  variance ne résout pas le déficit. Pour la ligne 993, elle donne
  0,942389 ± 0,001789 contre 0,998961 attendu. Avec rho=0, le même
  estimateur revient à 1,4e-6 et 5,4e-6 des forwards des lignes 948 et
  993, écarts de précision FP32 plutôt que d'échantillonnage.
  [Erreurs conditionnelles](../../artifacts/audit/rough-price-quality-2026-10-06/log_modulated_conditional_moment_se_replays.jsonl).
  Sur huit chemins et huit pas effectivement utilisés par le pricing,
  convolution directe et FFT diffèrent au plus de 2,86e-6 ligne 948 et
  8,58e-6 ligne 993. Mettre rho=0 réduit le déficit à deux pas/jour ;
  à quatre pas/jour la ligne 948 revient dans une erreur standard, sans tendance temporelle monotone.
  La ligne 993 garde un déficit à un et deux pas/jour ; son échéance dépasse
  la limite FFT de 4 096 pas à quatre pas/jour. Aucune erreur FFT n'est
  démontrée. [Contrôle rho=0](../../artifacts/audit/rough-price-quality-2026-10-06/log_modulated_rho_zero_refinement.jsonl).
  [Scan core/stress](../../artifacts/audit/rough-price-quality-2026-10-06/log_modulated_published_parity_scan.json),
  [replays stress](../../artifacts/audit/rough-price-quality-2026-10-06/log_modulated_flag_replays.jsonl),
  [moment conditionnel](../../artifacts/audit/rough-price-quality-2026-10-06/log_modulated_conditional_moment_se_replays.jsonl),
  [queues et FFT](../../artifacts/audit/rough-price-quality-2026-10-06/fft_tail_diagnostics.jsonl).
- **Six prix stress à forte incertitude :** les six lignes historiques ont
  été rejouées sur trois couples chemins/graines. L'Asian call 988 passe de
  0,03693 ± 0,02109 à 0,01658 ± 0,00078 et 0,01488 ± 0,00074 sur deux
  graines indépendantes ; le lookback 936 et le gap call 997 présentent
  aussi des sauts de queue. Le call européen 948 a un payoff maximum 170,69
  et ses cent plus grands payoffs portent 19,8 % de la moyenne sur
  262 144 chemins. La ligne européenne 993 atteint un payoff 736,20,
  avec un quantile 99,99 % de 67,39. [Replays des six lignes](../../artifacts/audit/rough-price-quality-2026-10-06/log_modulated_high_se_replays.jsonl).
- **Reprise sur le build isolé du 2026-10-07 :** les 1 000 paramètres
  satisfont les hypothèses du noyau log modulé ; les 900 core ont `ρ≤0`,
  42 stress ont `ρ>0`, et les huit écarts européens sont tous dans ce
  dernier domaine, avec résidu négatif. Par application explicite du
  théorème de Gassiat au noyau du dépôt, le spot total-return du modèle
  **continu** est une martingale locale stricte pour `ρ>0` : la parité
  avec le forward usuel n'est donc pas la bonne référence dans ce domaine.
  Sur une grille finie, le schéma log Euler conserve pourtant le forward en
  espérance exacte ; le déficit MC observé ne chiffre pas à lui seul le
  défaut du modèle continu. Le nouveau build CUDA 12.9/SM89 a exécuté
  20 replays : deux graines sur 948 et 993 confirment le déficit de moment
  conditionnel, le contrôle `ρ=0` retrouve le forward à 1–4×10⁻⁶,
  et FFT/direct diffèrent au plus de 3,34×10⁻⁶ sur les chemins contrôlés.
  La quadrature FP64 indépendante des huit noyaux signalés diffère au plus
  de 4,32×10⁻⁶ en relatif de la formule à 96 points ; elle ne couvre pas
  l'arrondi CUDA FP32. Deux outliers historiques de payoff à 2²⁰ chemins sont reproduits bit à bit.
  Les treize lignes rough SABR à `ρ>0` ont `β<1` et restent martingales
  dans leur modèle continu ; elles ne relèvent pas de ce diagnostic
  ([distinction démontrée](sabr-martingale-domain.md)).
  [Dérivation, replays et limites](num030-martingale-domain.md).
- **Qualification publiée, sans clôture :** le
  [masque versionné](../../work/catalog/qualifications/num030-lmb-stress-20261007-v1/manifest.json)
  exclut des références prix/SE les 42 lignes `ρ>0` de chacun des 29 jeux de
  prix, plus l'up-and-in call 954 à `ρ<0` dont la queue reste instable :
  1 219 lignes. Les prix et reçus originaux sont inchangés ; le
  [vérificateur](../../maintainer/tools/datasets/verify_num030_qualification.py)
  contrôle les 29 datasets, les empreintes, 20 replays ciblés et 96 replays
  supplémentaires à 2²⁰ chemins (16 graines par payoff). Le lookback 960
  varie de 0,07027 à 3,74733 selon la graine : l'incertitude de queue reste
  incompatible avec une référence de prix stable. En complément, le
  [gate de domaine des datasets](martingale-reference-datasets.md) impose
  une décision `ρ≤0` propre aux deux Bergomi pour les références ordinaires :
  15 lignes stress rough Bergomi supplémentaires sont masquées dans leurs
  29 jeux (435 prix/SE), avec reçus originaux inchangés. Le gate CTest passe
  sur 58 jeux Bergomi et les deux jeux de paramètres SABR. Cette mise en
  cohérence des références ne certifie pas les prix NUM-030 restants.
- **Reprise des domaines modèles du 2026-10-07 :** la
  [campagne alignée](aligned-mc-model-domain-2026-10-07.md) resserre les
  générateurs SABR, rough Bergomi, log-modulated rough Bergomi et quadratic
  rough Heston sans changer les produits **dans cette campagne antérieure**. La release SABR complète
  (1 modèle + 29 prix, 30 000 lignes) est exécutée, publiée et revérifiée :
  zéro signal de parité et zéro prix significatif à SE relative >25 %. Les
  essais européens des deux Bergomi et du QRH ont également zéro signal ;
  les 18 replays temps/facteurs QRH et neuf replays lookback montrent une
  forte amélioration du domaine candidat. **NUM-028 et NUM-030 restent ouverts**
  tant que les autres payoffs et la portée continue des prix QRH ne sont pas
  qualifiés. Les anciens datasets restent historiques avec leurs masques.
- **Preuve manquante pour clore :** une estimation indépendante et stable des
  prix/queues stress, particulièrement pour les huit lignes européennes
  `ρ>0` et l'up-and-in 954 à `ρ<0`, ou une décision de domaine qui retire
  réellement ces sorties de la capacité de référence. La théorie établit le
  signe du défaut continu, pas sa taille ni les prix des payoffs. Ne pas
  écrêter les extrêmes ni annoncer que la SE empirique couvre la queue rare.
- **Conséquence / portée :** la queue stress ne peut être considérée
  qualifiée par le seul contrôle de finitude. Le signal ne justifie pas
  d'invalider sans examen les 900 lignes core ou les autres modèles FFT.
- **Suite proposée :** construire un estimateur indépendant de la queue et
  des prix dans le domaine `ρ>0`, ainsi qu’un contrôle distinct du payoff
  up-and-in 954 ; confronter leurs intervalles à plusieurs grilles et graines.
  Ne régénérer que les données pour lesquelles un prix de remplacement est
  justifié, avec provenance authentique.
- **Clôture vérifiable :** les huit écarts et les six lignes à forte
  incertitude sont expliqués ou corrigés, avec contrôle des 900 lignes core
  et résultats indépendants sur la queue stress ; toute sortie modifiée est
  régénérée et liée à sa recette.

## Audit de l’axe IV — performance, 2026-10-06

Périmètre : source et binaires du worktree courant, Release SM89, CUDA 12.9.41,
RTX 4090 Laptop 16 Gio. Snapshot initial, commandes, sorties et profils sont
conservés sous `artifacts/audit/axis-performance-2026-10-06/` ; le fichier
`receipt.json` fixe leurs empreintes. Les chronos de ce passage sont
**diagnostiques** : ils ne constituent pas les trois campagnes admissibles du
protocole v3. Les alternatives ci-dessous restent à comparer avec les mêmes
modèles, produits, chemins, calendriers, sorties et précision.

### PERF-025 — L’évaluateur de Hessienne mixte Bates charge fortement la mémoire locale

- **État / propriétaire :** ouvert le 2026-10-06 ; propriétaire :
  `path_node_graph/mixed_evaluation.cuh`, compositions de Hessienne complète
  et politique d’inlining des dynamiques.
- **Axes / sévérité / priorité / confiance :** IV ; haute / haute / élevée
  pour la pression mémoire mesurée, gain d’une variante encore inconnu.
- **Contrat :** l’évaluation du graphe complet doit rester viable en
  registres/shared/local et être qualifiée à sa capacité de nœuds effective.
- **Preuve :** le test `mixed_path_node_graph_cuda_test.cu` passe avec quatre
  axes et six paires Bates range accrual. L’instanciation admet 15 axes et
  105 paires, soit 466 nœuds compilables. Sur le **noyau lancé** du binaire
  isolé, diagnostics et Nsight Compute relèvent 255 registres, 5 680 octets
  de stack/mémoire locale par thread, 33,60 Kio de shared statique,
  3,73 Kio dynamique et 16,65 % d’occupation mesurée. La mémoire locale
  représente 81,82 % des secteurs demandés en L1TEX et 84,85 % des secteurs
  demandés en L2 ; 81,60 % de ses lectures et 98,91 % de ses écritures
  atteignent L2 (`bates-mixed-path-current.ncu-rep`). Le test maximal
  Bates cliquet montre aussi 4 800 octets locaux par thread ; les graphes
  mixtes Heston, Merton et path ont des profils distincts.
- **Reprise du 2026-10-06 :** les chiffres Nsight historiques ont été
  confrontés à l'instanciation actuelle : la source générée admet toujours
  15 axes et 105 paires pour le cas de quatre axes observé. Aucune
  spécialisation n'est retenue sans profil des capacités courante et maximale,
  égalité des sorties et campagne A/B ; ce constat demeure ouvert.
- **Conséquence :** l’évaluation dépense du cache et du trafic sur la mémoire
  privée, en plus du workspace explicite borné à 512 Mio. Une mesure du
  workspace seul sous-estime le coût mémoire du graphe complet. Les règles
  de gain proposées par Nsight sont heuristiques, pas une accélération prouvée.
- **Alternative examinée :** la Hessienne complète à 529 nœuds de la fixture
  cliquet exige une vraie grande capacité ; réduire partout les tableaux
  casserait cette couverture. Le cas courant à quatre axes peut toutefois
  utiliser une instanciation moins large ou des durées de vie plus courtes.
- **Correction minimale :** profiler SASS `LDL/STL`, tableaux indexés et
  frontières d’inlining ; expérimenter une ou plusieurs classes de capacité
  réelle et le stockage temporaire par équipe, sans modifier l’estimateur.
  Budgéter le pic VRAM du processus en plus des octets du workspace.
- **Clôture :** ressources, trafic L1/L2/DRAM, pic VRAM, temps de chaque phase
  et appel public à capacité courante et maximale ; parité numérique complète
  et campagne A/B admissible avant de retenir une spécialisation.

### PERF-026 — Le profil CUDA courant et les nouveaux graphes n’ont pas de gate performance

- **État / propriétaire :** ouvert le 2026-10-06 ; propriétaires : manifeste,
  runner et CMake de performance.
- **Axes / sévérité / priorité / confiance :** IV principal, V secondaire ;
  moyenne / haute / prouvée sur la configuration, sans régression runtime
  déduite de la seule absence de gate.
- **Contrat :** chaque stratégie active doit avoir phases, ressources,
  budget mémoire, sorties numériques et scopes temporels attachés à un
  binaire et à un profil admissibles.
- **Preuve :** `CMakePresets.json` sélectionne
  `sm89_cuda12_unqualified_v1`, mais `maintainer/cmake/AIFactoryPerformance.cmake`
  crée `performance_regression_gate` seulement pour
  `sm89_reference_v1`. Le baseline v3 est CUDA 13.3 ; il ne qualifie pas
  CUDA 12.9. Le manifeste `price_gradients/strategy_manifest.json` couvre
  mono terminal, closed form, LSM et Jamshidian, mais aucun graphe terminal,
  path, mixte, rough FFT/N-facteurs ni nouveau pricer causal FFT. Le banc
  causal FFT ne fournit que cinq répétitions CUDA-event et aucune campagne
  de publication, de pic mémoire ou de ressources par phase. La commande
  `cmake --preset dev` du protocole ne correspond plus aux presets présents.
- **Reprise du 2026-10-06 :** le protocole a été corrigé pour nommer le
  preset actuel et interdire l'usage du baseline CUDA 13.3 comme référence
  du profil CUDA 12.9. Aucun baseline v4 ni trois campagnes admissibles
  n'existent encore ; le gate reste désactivé pour le profil courant.
- **Conséquence :** les bons résultats ponctuels de ce passage ne bloquent
  aucune régression sur le build courant ; un gain local peut masquer un coût
  de préparation, de mémoire ou de sérialisation.
- **Alternative examinée :** désactiver le gate est correct tant que la
  toolchain diffère ; comparer le baseline CUDA 13.3 au build CUDA 12.9
  produirait un verdict trompeur. Il faut qualifier le nouveau profil.
- **Correction minimale :** créer un baseline v4 propre à CUDA 12.9/SM89,
  étendre la matrice aux graphes diagonal/mixte, produits de chemin,
  Bermudan, rough FFT et causal, samples un/multiples chemins ; mettre à
  jour la reproduction et activer le gate uniquement après trois campagnes
  admissibles avec contrôles numériques et ressources des phases lancées.
- **Clôture :** CMake expose le gate sur le profil qualifié ; le runner
  échoue si une phase active, un hash, un budget ou une sortie manque ;
  médiane, p95, CV, VRAM et publication sont enregistrés sur toutes les
  familles annoncées.
