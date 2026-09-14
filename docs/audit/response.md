# Constats d'audit non résolus

Les anciens chemins de preuves `build-*` se retrouvent via le
[plan des artefacts locaux](../local-artifacts.md).

## État courant — prix rough, prix-delta et produits de taux — 2026-09-14

**Six constats ouverts, 106 fermés, 112 identifiants.** Le lot 1 puis
STRUCT-025/026/027/028 sont corrigés et clôturés avec leurs preuves et limites dans
[closed.md](closed.md). Restent PERF-016, DELTA-001, PRODUCT-001 et les trois
constats de qualité des prix rough NUM-028/029/030. Aucune clôture de performance
globale ou de validation indépendante n'est revendiquée.

Le passage indépendant portait sur le worktree réel de main, HEAD
`872a986b1f0947a1a832af0615ffc6d80dbedb81`, query v9, pas sur le seul commit.
Son snapshot et sa couverture partielle restent historiques dans
[status.md](status.md), qui décrit séparément la remédiation actuelle.
Aucune conformité globale ni certification des datasets n'est déduite des
tests de non-régression. Les anciennes campagnes PERF-017/019 restent fusionnées
dans PERF-016; aucune campagne longue n'est lancée dans ce lot.

## Extension prix et delta equity

### DELTA-001 — Déployer et qualifier la voie prix-delta sans dupliquer les moteurs métier

- **État :** ouvert le 2026-09-10, extensions markovienne et rough implémentées; chantier d'extension
  demandé, pas défaut attribué aux anciens prix.
- **Sévérité / priorité / confiance :** moyenne / haute / périmètre manquant prouvé.
- **Signature :** les launchers prix seuls n'exposent pas encore toute la voie
  prix + delta S0, ses recettes et ses garanties numériques. L'absence de cette
  nouvelle capacité n'invalide pas les bases de prix existantes.
- **Acquis du premier lot :** deux stratégies de chemins, kernel MC apparié,
  bumping closed form, sept façades générées BS/Heston/CEV et enregistrement
  CMake. Prix central/erreur standard bitwise sur six cas MC bornés; BS fermé
  call/put contre delta analytique avec trois bumps. Les équations CEV/Heston
  et les payoffs sont réutilisés, pas recopiés. FP64 limité aux moments MC.
- **Preuves :** `build-price-delta/price-delta-final-tests.log`, deux CTests
  réussis en 1,94 s sur SM89; diagnostics des spécialisations présentes dans
  le même log. 29 tests du manifeste, layout et codegen 1 563 sorties zéro
  diff. Le dossier ignoré est à conserver/exporter; aucune campagne longue.
  Compute Sanitizer memcheck : zéro erreur; archive source et hashes binaires
  consignés dans status.
- **Acquis LSM :** façades American BS/Heston/CEV call/put générées;
  une régression centrale, trace de dates/spots, exercice initial global gelé,
  deux passages delta et planner mémoire commun. Six CTests ciblés passent
  (6,25 s), deux memchecks zéro erreur, 29 tests du manifeste et codegen
  1 569 sorties zéro diff. Parité centrale bitwise et replay CEV pathwise
  exact dans le pilote; preuves/ressources dans le nouveau bloc de status.
  Les écarts au refit complet atteignent environ 0,0322 sur une ligne CEV :
  ce diagnostic ne qualifie pas le biais global de l'estimateur gelé.
- **Acquis markoviens / recettes :** 261 bindings compilés (244 MC, huit
  formules fermées, neuf LSM) et 364 recettes du catalogue existant générées,
  avec recipe.yaml prévisionnel, publication JSON/dataset.yaml et provenance.
  Aliases CRN explicites, 2^20 chemins MC/LSM; profils prix seuls inchangés,
  candidats MC delta bornés à 256 threads pour les états de payoff riches.
  34 cas publics : prix/erreur centraux bitwise, dont BS/Heston/CEV LSM à
  2^20 chemins. Sept générations natives sur deux lignes temporaires passent;
  aucune base existante remplacée. Correction ciblée du garde de cancellation
  des moments MC : sa borne tient compte de la longueur de sommation, sans
  changer les sommes ni les trajectoires. Reproduction constante avant/après,
  rejet d'incohérences matérielles et preuves consignés dans status.
- **Acquis rough :** 126 bindings et 174 recettes ajoutés pour six modèles.
  Heston/QRH réutilisent les préparations à 2, 3 et 7 facteurs. Les quatre
  modèles FFT partagent la convolution. Rough SABR prépare et avance trois
  états avec les transitions d'origine. Les observateurs de payoff sont
  communs aux voies MC et FFT; leurs arrêts restent indépendants.
  Les tests bornés et leur périmètre compilé sont consignés dans status.
- **Reste :** qualification des paramètres singuliers, des biais de bump et
  de discrétisation rough, du biais de policy LSM, et des ressources/temps sur
  des maturités de production. Les recettes ajoutées ne constituent pas une
  génération exhaustive ni une certification de leurs deltas. L'exercice
  anticipé rough n'est pas une surface implémentée par ce lot.
- **Prochaine étape :** qualifier des cas de production ciblés. Varier la
  largeur du bump et mesurer les franchissements de seuil pour les produits
  discontinus. Conserver séparément l'erreur d'échantillonnage et les biais
  d'approximation. Ne pas transformer les tests courts en garantie globale.
- **Clôture :** surfaces prévues générées et compilées, petits tests de
  parité prix seul/prix-delta et références delta adaptés à chaque moteur;
  dates LSM réellement gelées et comparaison bornée au refit complet;
  recettes à 2^20 chemins publiant bump, méthode, seed et géométrie truthful;
  précision des singularités et coût des nouvelles ressources documentés.
  Une limitation non qualifiée reste explicite et empêche une déclaration
  globale de disponibilité. Aucun test ou certificat Premia/QuantLib inventé.
- **Contrat :** [prix-delta equity](../cuda/equity-price-delta-contract.md).
  PERF-016 reste indépendant et reporté; aucune campagne de scaling lancée.

## Produits fixed income à trajectoire fine

### PRODUCT-001 — Ajouter trois produits de taux dépendant du chemin

- **État / date / propriétaire :** ouvert le 2026-09-14 à la demande de
  l'utilisateur ; extension fonctionnelle à réaliser, pas défaut des prix
  fixed income existants ; propriétaire : chantier produits fixed income.
- **Sévérité / priorité / confiance :** moyenne / haute / élevée sur l'absence
  de capacité, choix et coût des produits encore à qualifier.
- **Contrat et localisation :** [workflow d'extension](../catalog-extension-and-validation-workflow.md),
  `src/product/`, `src/model/fixed_income/**/product/`,
  `tools/codegen/pricing_bindings/capability_manifest.py`, `catalog/` et `datasets/`.
- **Preuve reproductible :** dans `AVAILABLE_DATASET_SPECS`, filtrer
  `asset_class == "fixed_income"`, `dataset_kind == "prices"` et
  `construction == "aligned"` : les 80 recettes couvrent `rate_option`,
  `zero_coupon_bond_option`, `european_swaption` et `bermudan_swaption`, mais
  aucun range accrual de taux ni coupon overnight composé. Le range accrual
  equity existant ne remplit pas ce contrat de taux.
- **Conséquence :** la famille fixed income n'exerce pas encore le suivi fin
  d'un taux le long du chemin, l'accumulation de coupons et l'actualisation
  correspondante. Les Bermudans testent des dates d'exercice, pas une
  observation quotidienne de coupon.
- **Ordre demandé :** (1) coupon range accrual non callable sur un indice de
  taux défini par le modèle, avec observations quotidiennes ; (2)
  caplet/floorlet sur taux overnight composé en fin de période ; (3) note
  callable range accrual à coupons multiples, avec décision de rappel de
  l'émetteur. Les conventions d'indice, calendrier, jours d'accumulation,
  paiement et actualisation seront déclarées avant implémentation. Ne pas
  appeler le taux court simulé « SOFR » sans correspondance explicite.
- **Correction minimale :** déclarer chaque produit et les compositions
  modèle-produit réellement supportées dans le manifeste ; réutiliser un
  observateur de chemin et un accumulateur d'actualisation communs, sans
  recopier les dynamiques par produit. Étendre ensuite les recettes codegen,
  les générateurs, la validation et les métadonnées selon le contrat existant.
  La note callable vient après les deux coupons non callable et réemploie le
  moteur LSM lorsque ses états de continuation sont définis.
- **Clôture vérifiable :** paramètres et payoff documentés, générateurs et
  cibles CMake disponibles pour la matrice déclarée, limites financières et
  cas dégénérés testés, contrôle de raffinement temporel et de ressources sur
  les observations fines, datasets de prix générés avec provenance, puis
  validation indépendante appropriée. Une couverture partielle ou une
  recette créée sans prix exécuté reste explicitement ouverte.

## Qualité des prix rough alignés publiés

Le contrôle ciblé des 174 datasets de prix seuls compare les calls/puts
européens de même modèle et de mêmes lignes d'entrée. L'écart diagnostique est
`C - P - (S0 exp(-qT) - K exp(-rT))`, avec `T = maturity_days / 252`.
Une ligne est signalée si sa valeur absolue dépasse à la fois cinq erreurs
standards combinées et 0,5 % de `max(S0, K)`. Ce n'est pas une référence de
prix indépendante : la cause d'un écart peut être une queue rare, le biais de
discrétisation, le statut martingale du modèle ou une erreur de code.
[Le script et son résultat local](../../artifacts/audit/rough-price-quality-2026-09-14/result.json)
figent la règle et les nombres ; le passage est détaillé dans
[status.md](status.md#qualité-ciblée-des-prix-rough-alignés--2026-09-14).

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

### NUM-029 — Expliquer les écarts de parité des prix rough SABR

- **État / date / propriétaire :** ouvert le 2026-09-14 ; propriétaire :
  moteur rough SABR FFT et qualification des prix MC.
- **Sévérité / priorité / confiance :** moyenne / haute / élevée sur les 15
  écarts, cause indéterminée.
- **Contrat / localisation :** [modèle rough SABR](../../src/model/equity/rough/rough_sabr/README.md),
  `src/model/equity/rough/rough_sabr/`,
  `datasets/model/equity/rough/rough_sabr/prices/european_{calls,puts}/`.
- **Preuve reproductible :** 13 lignes core et deux stress sur 1 000
  dépassent le seuil commun. Ligne 394 : résidu de parité `+0.0628550`
  pour `SE` combinée `0.0003255`, environ 193 fois celle-ci. Les datasets
  sont finis, alignés et publiés ; les prix nominaux proches de 10 sont
  compatibles avec certains spots d'entrée proches de 10 et ne fondent pas
  ce constat à eux seuls.
- **Conséquence / portée :** la précision déclarée de ces lignes ne suffit
  pas à justifier leur usage comme référence. La frontière absorbante et le
  pas Lamperti sont des pistes à mesurer, pas une cause démontrée ; la
  correction historique `NUM-006` n'est pas rouverte sans cette preuve.
- **Correction minimale proposée :** rejouer les 15 lignes avec d'autres
  graines et pas, contrôler `E[e^{-rT}S_T]`, l'incidence de l'absorption et
  les moments des payoffs. Corriger seulement le mécanisme identifié, puis
  republier les lignes concernées si leurs prix changent.
- **Clôture vérifiable :** les écarts sont expliqués par une limite documentée
  et admissible ou disparaissent après correction ; la qualification inclut
  les 13 lignes core, des répétitions indépendantes et les nouvelles preuves
  de provenance des datasets corrigés.

### NUM-030 — Qualifier la queue stress des prix log-modulated rough Bergomi

- **État / date / propriétaire :** ouvert le 2026-09-14 ; propriétaire :
  moteur log-modulated rough Bergomi FFT et qualification des prix MC.
- **Sévérité / priorité / confiance :** moyenne / moyenne / élevée sur le
  signal local, cause indéterminée.
- **Contrat / localisation :** [contrat MC](../cuda/closed-form-and-monte-carlo-pricing-contract.md),
  `src/model/equity/rough/log_modulated_rough_bergomi/`,
  `datasets/model/equity/rough/log_modulated_rough_bergomi/prices/`.
- **Preuve reproductible :** huit écarts de parité sur 100 lignes stress,
  zéro sur les 900 lignes core au seuil commun. Ligne 948 : résidu
  `-0.0090962`, `SE` combinée `0.0004228`, soit environ 21,5 fois celle-ci.
  Six prix stress `>0.01` ont aussi `SE/prix >25 %` ; aucun core dans ce
  critère. Les fichiers et leurs empreintes de publication sont intègres.
- **Conséquence / portée :** la queue stress ne peut être considérée
  qualifiée par le seul contrôle de finitude. Le signal ne justifie pas
  d'invalider sans examen les 900 lignes core ou les autres modèles FFT.
- **Correction minimale proposée :** reproduire les huit lignes avec
  répétitions indépendantes, isoler effet des paramètres stress, premier
  moment du spot, convolution FFT, discrétisation et erreur d'échantillonnage ;
  corriger ou documenter le domaine affecté avec provenance.
- **Clôture vérifiable :** les huit écarts et les six lignes à forte
  incertitude sont expliqués ou corrigés, avec contrôle des 900 lignes core
  et résultats indépendants sur la queue stress ; toute sortie modifiée est
  régénérée et liée à sa recette.

## Performance

### PERF-016 — À 2²⁰ trajectoires par prix, un million de prix prend-il environ 1 000 fois le temps de 1 000 prix ?

- **État :** ouvert depuis le 2026-09-06, recentré le 2026-09-09.
- **Sévérité / priorité / confiance :** moyenne / haute / à mesurer.
- **Propriétaire :** agent référent, à la demande de l'utilisateur.
- **Signature / impact :** le temps à 1 000 prix est mesuré, mais la stabilité
  du débit permettant de prévoir le temps à un million de prix n'est pas
  établie pour toutes les familles représentatives.
- **Objectif :** vérifier si `T(1 000 000) ≈ 1 000 × T(1 000)`, avec
  **2²⁰ = 1 048 576 trajectoires par prix**, en MC terminal, barrière et LSM,
  sur les représentants déjà retenus, y compris FFT et N-facteurs.
- **Acquis :** [29 cas à 1 000 prix](../performance-reports/pricing-dataset-runtime-sm89.ipynb),
  dont les cas MC/LSM à 2²⁰ trajectoires; les formules fermées n'ont pas de
  trajectoires. [Mesures historiques et limites](../performance-reports/pricing-workload-scaling-sm89-2026-09-07.md).
  Les réglages sont des références mesurées SM89, pas des optima universels.
- **Vérification minimale :** comparer 1 000 et 10 000 prix de difficulté
  comparable, avec le code courant et son découpage de production borné par
  la VRAM. Garder des warmups exclus, des répétitions et des conditions
  comparables; distinguer temps GPU et génération complète, préparation et
  écriture comprises. Ne pas confondre paramètres distincts et tuile répétée.
- **Clôture :** publier, par représentant, les temps, le rapport
  `T(10 000) / (10 × T(1 000))` et l'estimation à un million avec ses
  hypothèses et son incertitude. Si le débit n'est pas stable, expliquer
  l'écart et mesurer un palier supplémentaire si nécessaire avant de conclure.
  Le résultat peut réfuter le facteur ×1 000; une extrapolation étayée ne doit
  jamais être présentée comme une mesure directe à un million.
- **Hors mandat :** refaire le balayage 65k/262k/1M trajectoires, retuner chaque
  couple, modifier un moteur sans dégradation démontrée, relancer les études
  closed form ou samples, ou effectuer la validation indépendante.
- **Dernière vérification :** audit indépendant du 2026-09-09 : hashes des
  binaires, inputs et artefacts des 29 cas concordants; provenance détaillée
  dans status. Pas de nouvelle mesure du scaling.
- **Prochaine expérience proposée, en attente d'accord :** onze représentants
  Heston/Kou terminal, barrière et LSM; rough Bergomi/Heston terminal et
  barrière; CIR LSM. Deux warmups et trois répétitions à chaque taille,
  toujours 2²⁰ chemins/prix. Budget conditionnel ≈6 h 02 GPU / ≈6 h 40
  somme des phases host, sous hypothèse de débit linéaire et préparation
  comparable (`cuda/scaling-cost-estimate.json` dans le dossier de preuves).
  L'utilité et le coût ont été présentés à l'utilisateur; aucun accord reçu
  à la consolidation, donc campagne non lancée. L'estimation ne prouve pas
  le rapport demandé et ne couvre pas tous les 25 cas stochastiques.
