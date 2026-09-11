# Constats d'audit non résolus

## État courant — extension prix-delta en cours — 2026-09-11

**Deux constats ouverts, 106 fermés, 108 identifiants.** Le lot 1 puis
STRUCT-025/026/027/028 sont corrigés et clôturés avec leurs preuves et limites dans
[closed.md](closed.md). Restent PERF-016 et le chantier demandé DELTA-001; aucune clôture de
performance globale ou de validation.

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
