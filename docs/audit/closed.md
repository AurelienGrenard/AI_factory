# Constats d'audit fermes

## Provenance et conservation des datasets — 2026-09-10

### STRUCT-028 — Distinguer provenance de génération et compatibilité des bases

- **État / qualification :** ouvert et fermé le 2026-09-10; sévérité moyenne,
  priorité haute, confiance prouvée. Propriétaire : agent référent. Chantier
  demandé avant PERF-016, secondaire reproductibilité; distinct de STRUCT-023
  (reprise) et STRUCT-025/026 (description/replay samples).
- **Signature :** le contrôleur figeait binaires, entrées et recettes dans la
  campagne, sans provenance attachée au YAML publié ni diagnostic permettant
  de conserver une base après refactoring. Les hashes codegen et validation
  ne prouvent pas l'équivalence de deux implémentations.
- **Correction :** `dataset_provenance.py` possède un enregistrement versionné
  des futures publications prix/samples : intégrité JSON/recette/enregistrement,
  identité logique, forme, nombre de trajectoires MC, seeds par stream, entrées par rôle,
  binaire, recette, méthode déclarée, plan compilé et build. Le contrôleur v2
  conserve les sources réelles (dont non suivies), les recettes sélectionnées
  et les observations GPU; enrichit uniquement le YAML en staging avant le
  hash du journal; refuse les modifications de preuves figées en reprise.
- **Décision :** `check_dataset_compatibility.py` inspecte sans générer ni
  publier. Chemins et présentation des entrées ne déterminent pas leur
  identité; ordre et champs numériques restent significatifs. Les réponses
  distinguent compatible, revue, contrat nécessitant une nouvelle base et
  corruption. Code/binaire/build différent signifie revue, jamais preuve
  automatique de changement mathématique. Le résultat identifie le candidat
  et l'ancien enregistrement pour conserver une décision étayée.
- **Preuves :** 35 tests Python datasets/publication, dont 18 tests dédiés
  de provenance; 21 CTests hôte réussis, comprenant codegen zéro diff et
  contrôles d'architecture. Gel réel Heston samples et inspection du candidat
  réussis, **sans exécution** du générateur. Dossier local ignoré
  `build-dataset-provenance-20260910-z8KotZ/`, à conserver/exporter : snapshot,
  commandes/retours/hashes, `python-tests`, `host-ctests`, `freeze-probe`,
  `scope`. Comparaison au snapshot : 1 304 fichiers `src`, 1 132 fichiers
  catalogue et 96 fichiers codegen préservés; `git diff --check` propre.
- **Contrat durable :** [provenance et réutilisation](../dataset-provenance-contract.md),
  lié au workflow de génération et aux guides docs/outils/tests.
- **Limites assumées :** enregistrement automatique via le contrôleur, pas
  les invocations natives directes ni les générateurs autonomes de paramètres.
  Les paramètres sont identifiés comme entrées des prix. Pas de backfill
  historique, d'allowlist d'équivalence, de modification de certification ni
  de preuve bitwise multi-GPU. Les SDK et bibliothèques externes ne sont pas
  archivés. Une campagne v1 se termine avec son contrôleur d'origine.
  Aucune base existante, aucun kernel, aucun réglage ni audit de validation
  modifié; aucune campagne PERF-016 lancée.
- **Réouverture :** provenance inventée pour une ancienne base, classement
  compatible malgré une entrée/seed/calcul non couvert, disparition des
  guards de reprise, publication avant hash du YAML enrichi, ou assimilation
  d'un hash de source à une preuve mathématique/certification indépendante.

## Remédiation de l'organisation des tests — 2026-09-10

### STRUCT-027 — Regrouper les tests par responsabilité et clarifier leurs noms

- **État / qualification :** ouvert et corrigé le 2026-09-10; sévérité
  moyenne, priorité moyenne, confiance prouvée. Axe 1, secondaire 4.
- **Signature originale :** 74 sources C++/CUDA dispersées à la racine de
  `tests`, malgré les dossiers datasets, sampling et LSM; compositions
  American/Bermudan éloignées du solveur et noms énumérant les modèles
  plutôt que le contrat partagé. Aucun défaut numérique déduit du rangement.
- **Correction :** les 74 sources rejoignent leur propriétaire : datasets,
  sampling, modèles equity/fixed income, produits, LSM, Volterra, numérique
  ou infrastructure CUDA. La fixture Jamshidian rejoint le fixed income.
  La racine ne conserve que son README. Les tests transversaux restent
  groupés; aucun miroir mécanique des modèles ni nouveau moteur de test.
  Sept noms de fichiers sont clarifiés, notamment le tirage de paramètres,
  la composition LSM equity et les contrats Gaussian-Volterra. Quatre
  en-têtes précisent les modèles ou la courbe couverts. CMake accepte le
  chemin explicite des tests de stages; noms publics et labels restent stables.
- **Contrat durable :** [guide des tests](../../tests/README.md), carte des
  responsabilités, convention de noms/en-têtes, distinction entre tirage
  de paramètres et trajectoires, fixtures locales et helpers partagés.
- **Preuves :** `build-remediation-test-layout-20260910-kHS8Sz/` (ignoré,
  à conserver/exporter) contient snapshot, `mapping.json`, commandes,
  codes retour et hashes de logs. `qualified-layout-invariants` vérifie les
  contenus contre le snapshot, les 83 sources natives enregistrées, les
  346 contrats CTest et l'absence d'anciens chemins actifs. Les rapports
  historiques conservent leurs chemins d'époque. Build complet
  `ai_factory_tests` Release SM89 réussi; 20 CTests hôte et 11 CUDA réussis,
  aucun sauté. Codegen 1 549 sorties zéro diff et `git diff --check` propre.
- **Limites :** aucun changement d'assertion, oracle, algorithme ou paramètre
  de campagne; seuls des en-têtes et un include changent dans les sources
  déplacées. `src`, `tools`, catalogue, query et validation inchangés.
  Pas de campagne de performance, validation externe ni qualification
  multiarchitecture. PERF-016 reste ouvert.
- **Réouverture :** sources sans propriétaire prévisible, tests apparentés
  à nouveau dispersés, noms opaques, ancien chemin actif, ou perte de
  source enregistrée, nom public, label ou couverture après déplacement.
  Ne pas dupliquer STRUCT-016, NAME-012, NAME-007 ou STRUCT-019, dont les
  signatures historiques portent sur d'autres responsabilités.

## Remédiation des recettes samples — 2026-09-10

Preuves locales : `build-remediation-samples-20260910-Z9uPBq/`, ignoré,
à conserver/exporter. Commandes, retours, durées et hashes de logs conservés;
snapshot et exclusions dans [status.md](status.md).

### STRUCT-025 — Décrire toutes les lois dérivées des paramètres samples

- **État / qualification :** ouvert le 2026-09-09, corrigé et fermé le
  2026-09-10; sévérité moyenne, priorité moyenne, confiance prouvée.
- **Signature :** descriptions incomplètes des transformations Heston/Bates,
  NIG, VG, rough Heston, Hull-White/OU/Vasicek dans les deux layouts samples.
- **Correction :** lois conditionnelles/déterministes et intermédiaires
  d'acceptation renseignés dans `sample_manifest.py`; constantes NIG spot et
  Stein–Stein rho explicites. Ordre des propositions sérialisé, rejet et
  ordre des lignes décrits. Aucun changement de fabrique ou d'acceptation.
  Le validateur refuse les champs publiés ou scalaires dérivés sans loi,
  les doublons et les descriptions vides. Le contrat samples possède la règle.
- **Preuves :** 28 tests Python du manifeste; mutations retirant chaque loi
  rejetées, contrôle positif CIR/CIR++ conservé, descriptions présentes dans
  les helpers partagés des deux recettes. Codegen 1 549 sorties zéro diff;
  générateurs Heston des deux layouts compilés, assembly JSON/YAML testé.
  `unchanged-numerics` vérifie les 25 fabriques octet pour octet et les
  1 132 fichiers du catalogue inchangés par rapport au snapshot d'entrée.
- **Portée :** métadonnées des futures générations corrigées, aucune base
  existante ni aucun YAML publié réétiqueté. Pas de nouvelle certification
  financière ou de changement du tirage RNG.
- **Réouvrir si :** une transformation exécutée, constante ou intermédiaire
  d'acceptation est absente/inexacte dans une recette, ou si une nouvelle
  fabrique contourne la vérification de complétude.

### STRUCT-026 — Faire correspondre replay et métadonnées à la géométrie FFT exécutée

- **État / qualification :** ouvert le 2026-09-09, corrigé et fermé le
  2026-09-10; sévérité moyenne, priorité haute, confiance prouvée SM89.
- **Signature :** argument de threads ignoré par les adaptateurs FFT,
  fausse variation de géométrie au preflight et stratégie inconditionnelle
  décrite comme thread grid-stride alors que le kernel est parameter-block.
- **Correction :** requête hôte mince dans les bindings générés, déléguant
  au descripteur `Forward::block_dim` du profil cuFFTDx compilé. Le runner
  sérialise dimensions réelles, nombre de blocs, lignes exécutées et stratégie
  parameter-block pour les deux layouts. Le replay FFT diminue réellement
  la grille d'un bloc; le cas un seul bloc annonce explicitement une répétition
  à géométrie identique. Le replay Markovien continue de varier les threads.
  `MODEL_SAMPLE_PREFLIGHT` porte un objet JSON et un `replay_scope` explicite;
  le smoke ne publie plus une grille hypothétique de production.
- **Preuves :** `tests/sampling/recipe_replay_cuda_test.cpp` utilise les vrais
  helpers des quatre modèles FFT, deux layouts, 1 000 lignes chacun : finitude
  et replay identique. `matched-native-replay` compare automatiquement les
  huit recettes aux diagnostics des kernels effectivement lancés : blocs
  `[128,1,1]`, grille 4→3 sur la fixture. Plans de production sans allocation :
  grille 4 096→4 095 pour `12 000×250` et `3 000 000×1`, cas un seul bloc
  vérifié; le descripteur n'est pas déduit du réglage générique 256.
  `fft-regression` : deux CTests passants, dont replay bitwise aux frontières
  de batches du lot précédent. Contrôles codegen/architecture et assembly
  samples passants.
- **Portée :** aucune modification des kernels, du tuning, des seeds ou des
  équations; le diff du code device FFT est nul (`unchanged-numerics`).
  Le descripteur et les en-têtes/bindings ajoutés sont hôte uniquement.
  Les replays complets à trois millions de lignes et les autres architectures
  ne sont pas exécutés dans ce lot; le preflight de la machine reste requis
  avant sa grande génération. Aucune campagne performance ni base publiée.
- **Réouvrir si :** métadonnées divergentes du lancement compilé, replay
  prétendument variable mais natif identique, ou défaut de replay sur une
  autre taille/architecture. NUM-026 reste la correction numérique distincte
  des partenaires FFT aux frontières de batches.

## Remédiation ciblée — lot 1 — 2026-09-10

Preuves locales : `build-remediation-v9-lot1-20260910-xdQwSa/`, avec
commandes, sorties, codes retour et SHA-256 des logs. Ce dossier ignoré
doit être conservé/exporté; il n'est pas fourni par un clone.

### NUM-012 — Préserver le put Asian géométrique après absorption du spot à zéro

- **État / qualification :** corrigé et fermé le 2026-09-10;
  sévérité moyenne, priorité haute, confiance prouvée sur le périmètre ci-dessous.
- **Signature :** après absorption, `-inf` contaminait Kahan puis le put
  devenait zéro au lieu de K×discount.
- **Correction :** indicateur explicite d'observation nulle; Kahan FP32 conservé
  pour les valeurs finies, NaN/+inf non masqués. Aucun FP64 ajouté.
- **Preuves :** `geometric_asian_boundary_cuda` : zéros initiaux,
  intermédiaires et finaux, call/put, vraie absorption CEV et poursuite d'un
  état SABR absorbé donnent le put .97; contrôles finis et invalides.
  `asian_mean_precision_cuda`, factorisation produit et frontière SABR passent.
  Logs `test-consumers-final`, `test-retained-final`.
  Kernel CEV Asian : 59→60 registres SM89, zéro spill.
- **Réouvrir si :** une observation nulle/invalidité est masquée ou si le
  budget des chemins finis régresse; les limites historiques ci-dessous restent applicables.

#### Qualification historique du 2026-08-30, avant réouverture

- **Nature :** FP64 chaud elimine, qualifie et verifie le 2026-08-30.
- **Signature originale :** les deux facades geometric Asian additionnaient
  chaque log-spot et divisaient en FP64 avant `expf`, sans distinguer erreur de
  somme, exponentielle et cout sur les trois moteurs.
- **Cloture :** les deux variantes `GeometricMeanObservationHandler` et la
  facade `GeometricAsianOptionPathPolicy` reutilisent `CompensatedFloatSum`;
  la politique explicite de spot non positif est preservee. Le meme sweep
  compare la coordonnee log et la moyenne publiee a `long double`.
- **Preuve numerique et prix :** l'erreur relative Kahan maximale vaut
  `3.58e-8` sur la coordonnee et `1.04e-7` apres exponentielle. Le test
  Black--Scholes conserve sa comparaison analytique FP64; les tests Heston,
  QRH et Volterra sont finis, QRH rejoue bitwise et sa call geometrique reste
  sous la call arithmetique dans l'incertitude Monte Carlo.
- **Cout et ressources SM89 :** pour 14 458 880 log-observations, FP64 mesure
  `0.3747 ms`, FP32 simple `0.1907 ms`, Kahan FP32 `0.1149 ms`, chunks
  `0.3812 ms`. Heston reste a 68 registres; QRH N=7 passe de 108 a 91 et
  Volterra de 71 a 69, sans stack/local/spill.
- **Reouvrir seulement si :** domaine de log-spots, nombre d'observations ou
  fonction aval sort du sweep/budget, ou si une architecture cible invalide le
  gain end-to-end ou les ressources.

### NUM-027 — Employer les barrières relatives dans le range accrual Black-Scholes

- **État / qualification :** corrigé et fermé le 2026-09-10;
  sévérité moyenne, priorité haute, confiance prouvée sur le périmètre ci-dessous.
- **Signature :** bornes relatives S(t)/S(0) traitées comme niveaux absolus
  dans le seul closed form Black-Scholes.
- **Correction :** ajout de log S0 aux bornes dans le template canonique;
  façade et manifeste régénérés, oracle du test corrigé sans changer le
  helper de probabilités absolues.
- **Preuves :** `black_scholes_cuda`, S0=.75/1/1.4 et cas S0=1.2 :
  prix attendu 1.1, tolérance 2e-6. Logs `test-g2-conditioned`,
  `test-consumers-final`; `codegen-check` zéro diff et
  `manifest-contract` 26 tests passants.
- **Réouvrir si :** divergence de convention entre chemin et closed form,
  dépendance indue au spot normalisé ou perte de la correction après codegen.

### NUM-022 — Tirer une vraie loi de Poisson dans les intervalles Bates de grande moyenne

- **État / qualification :** corrigé et fermé le 2026-09-10;
  sévérité moyenne, priorité haute, confiance prouvée sur le périmètre ci-dessous.
- **Signature :** l'inversion depuis exp(-mean) sous-débordait à grande
  moyenne agrégée, alors que le compensateur conservait l'intensité réelle.
- **Correction :** inversion inchangée sous 10, PTRS commun à partir de 10;
  indices Philox inchangés, consommation de la petite branche préservée.
- **Preuves :** `bates_dynamics_cuda`, 32 768 chemins, intervalles de
  1/252 pas, moyennes 0/.1/9.99/10/110/1000 : comptes entiers, moyenne,
  variance, trois points de CDF Poisson et moment exponentiel compensé
  contre des oracles hôte, budgets de six erreurs standard. Replay
  128/256 threads identique par branche. Les contrôles existants de
  simulation terminale et calendrier passent (`test-consumers-final`).
  Le sweep de grande moyenne cible la primitive commune appelée par ces
  consommateurs; ce n'est pas une certification de prix du catalogue.
- **Ressources :** vrai kernel européen Bates 76→78 registres SM89,
  stack/spills nuls; aucun FP64 ajouté, pas de campagne prix longue.
- **Réouvrir si :** loi, compensation ou replay échoue sur le domaine accepté
  ou si un consommateur contourne la primitive corrigée.

### NUM-023 — Préserver la correction de martingale VG quand nu tend vers zéro

- **État / qualification :** corrigé et fermé le 2026-09-10;
  sévérité moyenne, priorité haute, confiance prouvée sur le périmètre ci-dessous.
- **Signature :** l'arrondi de 1+incrément avant log effaçait la correction
  de martingale lorsque nu tendait vers zéro.
- **Correction :** `log1pf(-nu*(theta+.5*sigma²))/nu`, sans FP64 ajouté.
- **Preuves :** `levy_dynamics_cuda`, 21 couples theta/nu
  (nu=1e-9 à .25), dérive et transition contre log1p long double hôte,
  limite brownienne et contrôles usuels conservés. Budget absolu 3e-8;
  `test-consumers-final` passe. Kernel européen VG : 55 registres
  avant/après, aucun spill.
- **Réouvrir si :** perte de la limite nu→0, moment compensé incorrect
  ou régression du domaine usuel.

### NUM-024 — Stabiliser les covariances G2 lorsque les deux vitesses diffèrent fortement

- **État / qualification :** corrigé et fermé le 2026-09-10;
  sévérité moyenne, priorité haute, confiance prouvée sur le périmètre ci-dessous.
- **Signature :** covariance intégrée et Cholesky incohérents lorsque seule
  une vitesse est petite; variance reconstruite du contre-exemple +48.7 %.
- **Correction :** identité stable divisant par a+b, série petits temps,
  formule directe réservée au domaine bien conditionné; série FP32 de degré
  huit du helper gaussien partagé. Résidu Cholesky matériellement négatif
  signalé par NaN, pas projeté silencieusement. Aucun FP64 ajouté.
- **Preuves :** `g2_covariance_cuda`, 2 904 cas, deux ordres de vitesses,
  rho=-.9/0/.9, seuils encadrés, comparaison aux noyaux intégrés par Simpson
  long double hôte : erreur normalisée maximale 2.6431e-5, budget 3e-5.
  G2/G2++, OU/Vasicek et pricers G2/G2++ Nelson–Siegel/Svensson passent;
  smoke pricing à 4 096 chemins, calendriers réguliers/explicites,
  constructions alignée/cartésienne et replay de géométrie.
  `test-retained-final`, `test-g2-pricing-retained`,
  `memcheck-g2-retained` : zéro erreur mémoire.
- **Coût / choix :** seule la branche sensible reste un appel device direct,
  non inliné; inliner toute la formule faisait monter G2++ à 157 registres
  et interdisait 512 threads. Version retenue : G2 régulier 72→72,
  explicite 80→78; G2++ Nelson–Siegel 109→112 et 110→114 registres.
  `resources-*-retained` et diagnostics du vrai lancement documentent
  stack/shared/spills. Sur 32 prix × 32 768 chemins, 256 threads/32 blocs,
  quatre warmups/onze répétitions : G2 .896–.897→.957 ms (~+7 %),
  G2++ 2.777→2.851 ms (~+3 %). Logs `timing-complete-before-*`
  et `timing-retained-1`; variantes intermédiaires non retenues conservées.
  Ce coût de justesse n'est ni un optimum universel ni une campagne scaling.
- **Réouvrir si :** dépassement du budget, covariance non PSD masquée,
  géométrie contractuelle impossible ou coût non acceptable sur une autre
  architecture/charge. Contrat permanent dans `model-dynamics-contract.md`.

### NUM-025 — Ne pas perdre les observations Volterra sur une grille acceptée

- **État / qualification :** corrigé et fermé le 2026-09-10;
  sévérité moyenne, priorité haute, confiance prouvée sur le périmètre ci-dessous.
- **Signature :** arrondis séparés du premier jour et de l'intervalle
  perdaient des callbacks; une première date arrondie à zéro bloquait le curseur.
- **Correction :** projection des dates cumulées, observation contractuelle
  positive affectée au moins au premier pas, coïncidences livrées dans l'ordre.
- **Preuves :** `volterra_hybrid_schedule_cuda`, 12 calendriers
  réguliers/stubbed/statiques, dt=1/504,1/378,1/63 : tous les callbacks et
  paiement Athena final présents. Projection 2/3/5/6/8 vérifiée,
  grille canonique exacte (`test-retained-final`).
- **Portée :** une grille non alignée reste une approximation temporelle
  documentée; livrer les événements ne prouve pas l'exactitude du prix
  sur une grille grossière. Pas de nouveau kernel ni de FP64.
- **Réouvrir si :** événement manquant/désordonné, paiement final perdu
  ou approximation présentée comme calendrier exact.

### NUM-026 — Préserver les samples FFT lorsqu'une paire traverse une frontière de batch

- **État / qualification :** corrigé et fermé le 2026-09-10;
  sévérité faible, priorité moyenne, confiance prouvée sur le périmètre ci-dessous.
- **Signature :** mettre à zéro le partenaire FFT valide hors batch changeait
  l'arrondi du chemin conservé malgré des indices RNG identiques.
- **Correction :** calculer les deux partenaires existant dans le paramètre;
  seule l'écriture reste bornée à la tranche. Aucun remapping Philox.
- **Preuves :** vrai launcher rough Bergomi et rough SABR, deux paramètres,
  P=1/249/250/251, T=4/32/128/504 jours, coupures 1/3/125/249/250/251
  lorsqu'applicables : comparaison bitwise; sentinelles hors première tranche
  inchangées. `test-retained-final` et `memcheck-fft-retained`,
  zéro erreur mémoire.
- **Coût :** 21 spécialisations compilées rough Bergomi, aucun nouveau spill;
  registres majoritairement identiques, variations -1 à +2.
  Sur 2×250 samples à 128 jours, les mesures complètes fluctuent de
  ~9.9 à 11.3 ms avant comme après; aucune accélération revendiquée.
  Le split 1+499 reste ~10.03→10.05 ms. Ce contrôle local ne qualifie pas
  toutes les charges/architectures.
- **Réouvrir si :** replay non bitwise à une coupure, écriture hors tranche
  ou coût matériel non qualifié. STRUCT-026 reste distinct et ouvert.

### TEST-001 — Faire échouer les comparaisons du runner sur un résultat NaN

- **État / qualification :** ouvert le 2026-09-09, corrigé et fermé le
  2026-09-10; sévérité moyenne, priorité moyenne, confiance prouvée.
- **Signature :** `fabs(actual-expected)>tol` acceptait NaN; le mutant
  du faux noyau analytique passait malgré des attentes finies.
- **Correction :** `require_close` rejette les deux opérandes non finis;
  auto-test NaN et deux infinis, côté attendu et observé. Prix et erreurs
  standard utilisent le même helper.
- **Preuve :** `test-guards.json/.log` : tests natifs passants.
  `test-runner-mutant.json/.log` : copie isolée du runner corrigé,
  unique sortie analytique remplacée par NaN, échec explicite
  `offline CUDA runner returned wrong data` (SIGABRT/code -6).
  Source et compilation du mutant conservées; runner de production inchangé.
- **Réouvrir si :** un non-fini contourne le helper ou le mutant repasse.

### TEST-002 — Initialiser le padding copié par le test de robustesse numérique

- **État / qualification :** ouvert le 2026-09-09, corrigé et fermé le
  2026-09-10; sévérité faible, priorité moyenne, confiance prouvée.
- **Signature :** huit octets de padding indéfinis dans la fixture
  `NumericalResults` de 672 octets étaient copiés vers l'hôte.
- **Correction :** stockage device initialisé à `0xff` avant le kernel.
  Les champs flottants restent NaN tant qu'ils ne sont pas écrits;
  aucune assertion supprimée ni assouplie.
- **Preuve :** test natif passant; `initcheck-robustness.json/.log`,
  Compute Sanitizer 13.3 Initcheck SM89, code 0, **zéro erreur**.
- **Réouvrir si :** lecture indéfinie de cette fixture ou initialisation
  masquant un champ numérique manquant.

### BUILD-010 — Reconfigurer les dépendances inférées lorsqu'une source change

- **État / qualification :** ouvert le 2026-09-09, corrigé et fermé le
  2026-09-10; sévérité moyenne, priorité moyenne, confiance prouvée.
- **Signature :** trois inférences CMake lisaient des sources non suivies
  comme dépendances de configuration; includes modifiés, lien/labels périmés.
- **Correction :** `CMAKE_CONFIGURE_DEPENDS` chez les trois propriétaires,
  règle dans `cmake/README.md`, aucune portée PUBLIC ajoutée.
- **Preuve :** `cmake-inference.json/.log` : le test
  `tests/build/inferred_dependencies_test.py` exerce les fonctions
  de production dans un projet hôte isolé. Ajout **et retrait** d'includes,
  dépendances de test et générateur, compilation/lien et labels common/equity
  recalculés par le seul `cmake --build`.
  Enregistrement CTest `cmake_inferred_dependencies`.
- **Réouvrir si :** source d'inférence non suivie, lien incrémental périmé
  ou labels ne suivant plus les includes.

## Contre-revue indépendante — query v9 — 2026-09-09/10

Les 93 identifiants fermés ont été consultés avant ouverture des nouveaux
constats. **Aucune clôture ni fusion supplémentaire; NUM-012 est réouvert.**
Onze nouvelles signatures sont dans [response.md](response.md);
la couverture, le snapshot et le dossier local de preuves sont décrits dans
[status.md](status.md). Les 92 autres entrées fermées restent inchangées;
la qualification historique de NUM-012 accompagne sa réouverture dans response.

| Hypothèse ou rapprochement examiné | Décision et preuve contradictoire | Condition d'un nouvel examen |
|---|---|---|
| Réouvrir ANALYTICS-001 pour les méthodes Jamshidian CIR++ | Non retenu : elles délèguent au solveur commun avec un calendrier neutre, sans dépendance à un produit concret; mêmes surfaces présentes dans les autres modèles affines. `structure/report.md` suit les includes et appels. | Dépendance produit concrète ou formule dupliquée démontrée; le seul nom d'une méthode ne suffit pas. |
| Réouvrir BUILD-004/005 pour le lien incrémental périmé | Non : leurs signatures sont portée des dépendances et ownership. L'inférence non recalculée après modification de source est reproduite séparément sous BUILD-010. | Les anciennes conditions restent celles de leurs entrées; ne pas fusionner des causes distinctes. |
| Réouvrir STRUCT-013/017 ou PERF-015 pour les samples | Non : les recettes existent, le C++ n'est pas caché dans Python et le tuning reste central. STRUCT-025 vise la description des lois; STRUCT-026 vise des arguments de replay abandonnés. | Réapparition effective de l'ancienne absence, de l'ancien code caché ou d'une table de tuning locale. |
| Réouvrir NUM-008 pour les quelques bits FFT différents | Non : les indices Philox du chemin sont stables; le partenaire complexe mis à zéro change l'arrondi. Nouvelle cause NUM-026, sonde publique bitwise à l'appui. | Collision ou réallocation non maîtrisée des anciens domaines RNG, selon l'entrée historique. |
| Réouvrir CUDA-001 pour l'échec Initcheck | Non : le padding fautif appartient à NumericalResults dans un test, pas au PreparedRow Bermudan de production. Layout de 672 octets et contre-épreuve avec stockage initialisé sous TEST-002. | Réapparition d'une lecture non initialisée du PreparedRow/workspace de l'ancienne signature. |
| Cache statique FFT invalidé par cudaDeviceReset | Hypothèse non retenue sur le cas testé : deux lancements publics réussissent avec buffers recréés, 65 536 octets de shared et même prix affiché. `cuda/context_reset_probe.static.result.txt`. Cela ne qualifie pas plusieurs GPU. | Échec attribuable au cache sur autre contexte/device, à reproduire avec binaire et contrat précis. |
| SIGSEGV des deux sondes FFT initiales | Pas un défaut de production retenu : GDB situe l'échec dans le JIT driver avant lancement. Les mêmes unités liées statiquement comme CMake fonctionnent; sources, deux liens et logs conservés. | Reproduction dans la chaîne de liaison supportée ou preuve indépendante d'une erreur du dépôt. |
| Cliquet au spot absorbé zéro | Pas de constat : le calcul 0/0 suivi du floor appelle une convention financière qui n'est pas suffisamment définie pour affirmer un paiement incorrect dans ce passage. Aucune correction arbitraire prescrite. | Convention opposable et contre-exemple produit établis. |
| Étendre PERF-016 aux anciennes matrices PERF-017/019 | Non : leur fusion et la réduction du mandat restent applicables. Seule la comparaison 1 000/10 000 à 2²⁰ chemins puis extrapolation est proposée; aucune campagne longue sans accord. | Mandat utilisateur explicitement distinct, comme prévu dans leurs conditions historiques. |

Les preuves historiques ne sont pas automatiquement reconduites : provenance
des 29 cas à 1 000 prix vérifiée et copiée dans le dossier isolé, mais trois
exécutables actuellement aux anciens chemins de profils Nsight diffèrent du
hash du profil. Ces profils conservent leur portée historique; aucune nouvelle
régression ni clôture n'est déduite de cette seule différence.


## Fusion du suivi de scaling — 2026-09-09

L'utilisateur remplace la matrice multi-tailles par une seule question :
à 2²⁰ trajectoires par prix, le passage de 1 000 à un million de prix
multiplie-t-il approximativement le temps par 1 000 ? `PERF-016` reste ouvert
et porte le suivi commun MC terminal, barrière et LSM. Les deux entrées
ci-dessous sont fermées **par fusion et réduction explicite du mandat**,
pas parce que leurs anciens critères auraient été entièrement vérifiés.
Les mentions historiques de leur état dans les passages antérieurs restent
datées; [response.md](response.md) porte seul l'état courant.

### PERF-017 — Qualifier le scaling des barrières avec leur monitoring inchangé

- **Nature :** ouvert le 2026-09-06, fusionné dans `PERF-016` le 2026-09-09,
  sur décision de l'utilisateur; sévérité originale moyenne, priorité haute,
  confiance prouvée pour la couverture manquante, coût et géométrie à mesurer.
  Propriétaire : agent référent.
- **Signature originale :** la baseline de géométrie privilégie les policies
  Phoenix Bates à 4 096 trajectoires; elle ne mesure pas un produit barrière
  pour chaque modèle/méthode jusqu'à 1 000 prix × 1 048 576 trajectoires.
  Arrêts anticipés et monitoring changent la charge par rapport au terminal.
- **Preuve initiale :** `tests/performance/generic_kernel_benchmark.cu`,
  `tests/performance/baseline_sm89_v3.json`,
  `src/common/equity/barrier_pricing_policy.cuh`.
- **Décision :** les barrières restent des cas distincts dans le constat
  commun; même monitoring et charge comparable entre nombres de prix.
  Le balayage 65 536/262 144/1 048 576 trajectoires n'est plus une condition
  de clôture. Aucune performance barrière n'est déduite du seul terminal.
- **Preuves conservées :** calibrage à 100 × 64k; screening Heston aux trois
  nombres de trajectoires et de threads; 62 jobs Heston/N-facteurs à
  1 000 × 64k; matrice Heston/Kou à 100/1 000 prix. Les reprises rough
  interrompues restent non qualifiantes. La campagne du 2026-09-08 mesure
  huit barrières à 1 000 lignes réelles × 2²⁰ trajectoires.
  [Historique et campagnes brutes](../performance-reports/pricing-workload-scaling-sm89-2026-09-07.md),
  [confirmation courante](../performance-reports/catalogue-generation-readiness-sm89-2026-09-08.md).
- **Réouvrir seulement si :** un mandat distinct rétablit la qualification
  spécifique des barrières hors du suivi commun. Une case manquante de
  l'ancienne matrice ne suffit pas; les questions de débit restent rattachées
  à `PERF-016`, sans doublon.

### PERF-019 — Qualifier LSM à un million de trajectoires par prix

- **Nature :** ouvert le 2026-09-06, fusionné dans `PERF-016` le 2026-09-09,
  sur décision de l'utilisateur; sévérité originale moyenne, priorité haute,
  confiance prouvée pour la couverture officielle insuffisante, cause du
  ralentissement à mesurer. Propriétaire : agent référent.
- **Signature originale :** le benchmark LSM officiel fixe 64 prix × 16 384
  trajectoires, 128 threads et 32 blocs/prix. Il ne qualifie pas chaque modèle
  à 1 000 prix × 1 048 576 trajectoires, ni les changements de concurrence
  induits par le workspace. Les sondes du 2026-09-05 restent exploratoires.
- **Preuve initiale :** `tests/performance/early_exercise_benchmark.cu`,
  `src/common/longstaff_schwartz/longstaff_schwartz_kernels.cuh`,
  rapports LSM sous `docs/performance-reports`.
- **Décision :** LSM equity et fixed income restent dans le suivi commun,
  avec leurs batches VRAM natifs et sans partitionner les trajectoires en
  régressions indépendantes. Le nombre de trajectoires reste fixé à 2²⁰;
  l'ancienne matrice trajectoires × prix × calendriers n'est plus exigée.
- **Preuves conservées :** 17 adaptateurs compilés; calibrages et screenings
  Heston/Kou/CIR; confirmation Heston/CIR 20 jobs et 18 parités bitwise;
  contrôle CIR en ordre inverse, quatre jobs; extension Heston à
  1 000 × 256k/1M et 10 000 × 64k, six jobs. Les réserves de puissance
  et les frontières de temps historiques sont conservées. Les timings de
  l'ancien CIR à intégrale discrétisée ne qualifient pas le CIR actuel.
  La sensibilité Kou est corrigée séparément sous `NUM-017`; la campagne
  courante mesure neuf LSM à 1 000 lignes réelles × 2²⁰ trajectoires.
  [Historique, diagnostics et campagnes brutes](../performance-reports/pricing-workload-scaling-sm89-2026-09-07.md),
  [correction Kou et confirmation courante](../performance-reports/catalogue-generation-readiness-sm89-2026-09-08.md).
- **Réouvrir seulement si :** un mandat distinct rétablit une qualification
  LSM hors du suivi commun. Les écarts numériques relèvent de leur signature
  propre (`NUM-017` pour Kou); les questions de débit/batching à nombre de
  trajectoires fixé restent sous `PERF-016`.

## Préparation des générations — 2026-09-08

### STRUCT-024 — Distinguer la RAM disponible de la RAM libre dans les samples

- **Nature :** ouvert, corrigé et fermé le 2026-09-08; sévérité moyenne,
  priorité haute, confiance prouvée. Propriétaire : agent référent.
- **Signature originale :** la garde samples utilise `_SC_AVPHYS_PAGES`,
  excluant le cache récupérable, après allocation des paramètres/entrées.
  Le pilote Kou inconditionnel est refusé pour environ 114 Mio utiles,
  avec environ 7.2 Gio disponibles mais 288 Mio libres au diagnostic effectué
  après la sortie du processus, pas au moment précis de sa garde.
- **Correction :** helper hôte `tools/sampling/host_memory.hpp`, lecture
  de `MemAvailable` Linux et repli conservateur sur les pages libres.
  Contrôle avant les allocations; seuils 70% RAM / 85% VRAM conservés.
  Aucun cache système vidé, aucun kernel/seed/FP64 device modifié.
- **Preuve :** test CPU cache récupérable, zéro mesuré, valeurs manquantes,
  négatives, unités invalides et overflow. Deux pilotes de 3M lignes réussis;
  corps de `samples_01` identique avant/après. `samples_02 --preflight`
  retrouve les 3M maturités/observables bit à bit entre 256 et 128 threads.
  Première tentative refusée conservée; les nouveaux binaires sont figés
  dans une campagne distincte, pas substitués dans l'ancienne.
- **Preuves :** [pilotes et empreintes](../../tests/performance/reports/generation-readiness-sm89-2026-09-08/native-generation-pilot.json),
  logs et archive `sample-host-memory-sources.tar.gz` sous
  `build-dev/kou-lsm-launch-confirmation.zpDtZU`, SHA-256
  `5742ced6541138cbd417974604063aa83173e93e23a6e2686ac3e06e4341da8d`.
  Contrat durable : `model-sample-dataset-generation.md`.
- **Réouvrir seulement si :** le cache récupérable redevient un faux veto,
  une télémétrie invalide est acceptée, ou les allocations précèdent la garde.

### STRUCT-023 — Rendre la campagne de génération reprenable sans écrasement prématuré

- **Nature :** ouvert, corrigé et fermé le 2026-09-08; sévérité moyenne,
  priorité haute, confiance prouvée. Propriétaire : agent référent.
- **Signature originale :** les exécutables natifs écrivent directement
  JSON puis YAML; leur invocation en série ne fournit ni snapshot commun
  des entrées/binaires, ni journal par dataset, ni reprise de publication
  avec conservation de la paire précédente.
- **Correction :** contrôleur séquentiel dérivé du manifeste, entrées et
  binaires figés, staging, contrôle des sorties, marge disque, journal et
  sauvegarde JSON/YAML. Reprise explicite, sans recalcul des datasets achevés.
  Contrat durable dans `dataset-generation-workflow.md`.
- **Preuve :** seize tests Python (interruption entre renommages, corruption,
  édition concurrente, verrou et reprise). Sept targets natifs courants
  vérifiés : Kou call/put, swaptions européennes CIR/CIR++/G2++, deux shapes
  Kou samples de 3M lignes. Les cinq prix ont 1 000 lignes, `2^20` chemins
  si MC/LSM et `pending / verified: false`; aucune publication canonique.
  Le put Kou retrouve les 1 000 prix/erreurs de la résolution binary128.
  La reprise des deux samples complets vérifie leurs hashes, conserve une
  seule tentative et ne relance aucun GPU. La tentative refusée par la garde
  mémoire reste enregistrée, corrigée séparément sous `STRUCT-024`.
- **Preuves :** [artefacts, hashes, temps de processus et de contrôle distincts](../../tests/performance/reports/generation-readiness-sm89-2026-09-08/native-generation-pilot.json).
  Les quatorze destinations JSON/YAML canoniques gardent leurs empreintes.
- **Limites :** renommage atomique par fichier, pas par paire; pas de checkpoint
  à l'intérieur d'un dataset ni de garantie de temps pour 1M × `2^20`.
  Pas de veto thermique, changement de kernel ou certification indépendante.
  Cette clôture ne ferme pas `PERF-016/017/019`.
- **Réouvrir seulement si :** la reprise recalcule un dataset complet, masque
  une corruption/édition concurrente, perd la paire précédente, ou publie
  des sorties incomplètes ou faussement certifiées.

### NUM-017 — Qualifier la resolution FP64 des equations normales LSM

- **Nature :** clos le 2026-08-30, réouvert selon sa condition historique puis
  corrigé et refermé le 2026-09-08; sévérité moyenne, priorité haute, confiance
  prouvée. Propriétaire : agent référent.
- **Signature originale :** assemblage, ridge, Cholesky, substitutions et
  coefficients FP64 sans grille de conditionnement, référence haute précision
  ni coût comparé à FP32. La clôture initiale couvre un SPD proche de `1e8`
  après ridge `1e-10`: erreur coefficients FP64 `7.47e-10`, arrondi FP32
  singulier; 65 536 résolutions SM89 en `0.0870 ms` FP64 contre `0.0403 ms`
  FP32, 80/48 registres; kernels Heston/G2 à 82, sans local. Ces preuves restent
  valables dans leur domaine, pas à tout conditionnement.
- **Réouverture :** des lignes core Kou atteignent `6e10`. Des arrondis du
  Gram de l'ordre de `1e-16` changent les décisions malgré des chemins
  forward identiques. Sur 1 000 lignes à `2^20`, huit prix dépassent le budget
  `1e-6 + 1e-6*abs(reference)`, écart maximal `8.2850456e-6`.
- **Correction :** une seule correction du résidu normal, activée à la
  compilation pour Kou uniquement; deux spécialisations des kernels existants,
  RHS/workspace réutilisés. Base, ridge, mapping Philox, tolérances et stockage
  FP32 des chemins/cashflows inchangés; aucun transfert CPU par date ni boucle
  de convergence. Contrat durable dans `cuda/american-and-bermudan-pricing-contract.md`.
- **Preuve numérique :** référence de résolution pivotée CPU binary128 sur
  statistiques GPU hi/lo; six géométries du prototype, launcher intégré et
  trois nouveaux processus corrigés retrouvent les 1 000 prix/erreurs bit à
  bit. Les trois processus anciens reproduisent les huit écarts. Tests permanents
  des huit lignes call/put à trois géométries, régresseur contre `long double`,
  cas sans/avec trop peu de candidats et échec fatal de correction; CTests
  réussis, memcheck zéro erreur, racecheck zéro hazard. La référence partage
  chemins/features : ce n'est pas une certification financière indépendante.
- **Coût / FP64 :** six processus sans compilation concurrente, ordre figé,
  un warmup exclu et trois mesures chacun. Rapports de coût GPU par paire
  1.672 / 1.355 / 1.461; premier passage corrigé CV 5.50%, non qualifiant.
  Fréquences variables et toutes les mesures conservées : aucun pourcentage
  universel ni optimum revendiqué. Profilage séparé : résidu 4.242 s,
  correction du petit système 0.069 s. Nouveaux kernels 56/104 registres;
  les sept anciens restent à 40/70/86/82/39/23/35. Tous sans stack/local ni
  instruction SASS LDL/STL; pas de buffer de workspace ajouté, mêmes 16 batches.
- **Preuves et limites :** [rapport compact et artefacts](../performance-reports/catalogue-generation-readiness-sm89-2026-09-08.md).
  Portée SM89/CUDA 13.3, correction numérique ciblée; ni validation indépendante,
  ni rebaseline, ni clôture de la matrice de scaling `PERF-019`.
- **Réouvrir seulement si :** base, ridge, domaine/conditionnement, méthode,
  compilateur ou architecture changent et invalident ces preuves, ou si une
  autre composition présente cette sensibilité sans correction qualifiée.

### BUILD-009 — Adapter le test de swaption OU à la signature publique complète

- **Nature :** ouvert puis corrigé et fermé le 2026-09-08; sévérité faible,
  priorité moyenne, confiance prouvée. Propriétaire : agent référent.
- **Signature originale :** les pointeurs `RegularLauncher` et
  `ExplicitLauncher` omettaient les arguments de capacité/distribution du
  launcher OU. Les arguments par défaut ne font pas partie du type du pointeur;
  le build agrégé échouait avec six erreurs NVCC de résolution de surcharge.
- **Correction :** types complets et arguments explicites dans le seul test,
  conservant le mode scalaire régulier et coopératif explicite. Aucune formule,
  tolérance, recette ni interface de production modifiée.
- **Preuve :** rebuild et CTest `ornstein_uhlenbeck_european_swaptions_cuda`
  réussis sur SM89; logs `focused-build.log` et `focused-ctest.log` sous
  `build-dev/kou-lsm-launch-confirmation.zpDtZU`. Cas alignés, cartésiens,
  offsets, deux côtés et jambe de 600 paiements contre référence CPU conservés.
- **Réouvrir seulement si :** une signature publique n'est plus propagée
  aux pointeurs de fonctions des tests ou si ce raccordement ne compile plus.

### STRUCT-022 — Centraliser les métadonnées de prix non encore certifiés

- **Nature :** ouvert puis corrigé et fermé le 2026-09-08; sévérité moyenne,
  priorité haute, confiance prouvée. Propriétaire : agent référent.
- **Signature originale :** le writer générique publiait une référence à un
  ancien `validation.ipynb`; deux pipelines taux reconstruisaient un chemin
  contenant à tort `prices`. Les sections supplémentaires du writer MC
  pouvaient remplacer le bloc de certification.
- **Correction :** `price_validation_metadata(dataset_path)` est l'unique
  propriétaire : statut `pending`, `verified: false`, chemin prévu
  `validation/datasets/price/<taxonomie modèle>/<produit>/<dataset>.json`,
  avec la courbe intermédiaire si applicable. Une sortie temporaire hors
  taxonomie n'invente pas de cache. Les reconstructions des pipelines taux
  disparaissent et une surcharge `validation` est refusée avant toute écriture.
- **Preuve :** tests host `artifact_io_stage` et `price_dataset_stage` passés;
  le second a été recompilé séparément contre les archives courantes pendant
  le build agrégé, incluant le cas anti-écrasement ajouté. Famille rough,
  modèle ajusté/courbe, sortie temporaire et tentative `verified: true`
  couverts. Contrat durable dans `catalog-extension-and-validation-workflow.md`
  et `dataset-generation-workflow.md`.
- **Portée :** génération uniquement, pas audit des validateurs; aucun ancien
  YAML, prix ou cache modifié. La vérification des sorties par le contrôleur
  ne certifie pas les prix. La reprise/publication reste suivie sous STRUCT-023.
- **Réouvrir seulement si :** un writer invente une autre référence, omet
  pending/false, accepte de surcharger ce statut, ou écrit avant de refuser
  une tentative de certification non autorisée.

### NUM-021 — Résoudre les rejets Jamshidian sur les lignes de stress CIR et Vasicek

- **Nature :** ouvert puis corrigé et fermé le 2026-09-08; sévérité haute,
  priorité haute, confiance prouvée. Propriétaire : agent référent.
- **Signature originale :** sur les 1 000 lignes alignées acceptées par les
  loaders, les swaptions CIR `000917`, `000949`, `000953`, `000958`,
  `000989`, `000993` et la référence scalaire Vasicek `000997` donnent `NaN`.
- **Cause / correction :** la somme des coupons et la maille d'une racine
  absolue FP32 empêchent de certifier certains résidus à `2e-7`. La primitive
  partagée utilise une somme compensée FP32 et, si nécessaire, représente la
  frontière par une ancre et un petit déplacement FP32. Le même déplacement
  entre dans les strikes obligataires. Le bracket déplacé est revérifié et
  le raffinement borné à 48 itérations; un débordement exploratoire garde son
  signe sans contaminer la compensation. Aucun FP64 device ajouté, aucune
  tolérance augmentée ni ligne écartée. Les rejets légitimes de `NUM-001`
  restent obligatoires, notamment avec zéro itération autorisée.
- **Preuve numérique :** 12 lignes figées (8 CIR, 4 Vasicek) et leurs deux
  voisins de volatilité FP32, soit 36 cas, comparés à des formules CPU
  indépendantes locales, payer/receiver, scalaire/coopératif, 128/256/512
  threads et plusieurs nombres de blocs. Budget prix `2e-6 + 2e-5 * |référence|`.
  Les CTests `one_factor_european_swaptions_cuda` et
  `numerical_robustness_cuda` passent. La matrice du catalogue passe 48/48
  configurations sur les 1 000 lignes originales, sans prix invalide; deux
  warmups puis cinq répétitions, grilles denses et persistantes.
- **Ressources SM89 / portée :** registres scalaire CIR 59 → 60, Vasicek
  46 → 44; coopératif CIR 56 → 56, Vasicek 39 → 40. Aucun local ni
  instruction de spill LDL/STL observé; aucun calcul FP64 trouvé dans le
  SASS du banc. Pas de gain avant/après revendiqué : l'ancien chemin rejetait
  des lignes et sautait une partie du travail. Ni retuning, rebaseline,
  certification Premia/QuantLib ni qualification d'un autre GPU.
- **Preuves conservées :** [synthèse, hashes, matrices et ressources](../../tests/performance/reports/generation-readiness-sm89-2026-09-08/jamshidian-correction.json),
  fixture `tests/fixtures/jamshidian_stress_rows.hpp`; contrat permanent dans
  `cuda/closed-form-and-monte-carlo-pricing-contract.md`. Les mesures
  historiquement incomplètes de `PERF-022` ne sont pas requalifiées.
- **Réouvrir seulement si :** une ligne ou son voisinage perd sa référence
  locale, un résidu non certifié devient accepté, une géométrie produit des
  rejets, ou un changement de domaine/compilateur/GPU invalide ces preuves.

## Expériences Jamshidian one-factor — 2026-09-08

### PERF-022 — Mesurer le choix scalaire/cooperatif Jamshidian selon batch et calendrier

- **Nature :** ouvert puis fermé par expérience et décision bornée le
  2026-09-08; sévérité moyenne, priorité haute, confiance prouvée.
  Propriétaire : agent référent, mandat explicite de l'utilisateur.
- **Signature initiale :** les recettes européennes régulières mélangent un
  prix par thread et un prix par bloc sans matrice couvrant tous les modèles
  one-factor, calendriers courts/longs et volumes jusqu'à 2²⁰ prix.
- **Clôture :** 1 624 configurations mesurées sur cinq modèles/sept
  compositions, 100 à 1 048 576 prix, 64/128/256/512 threads et plusieurs
  grilles; profils calendaires à 16 384 prix. Screening séparé de trois
  confirmations payer indépendantes (cinq warmups, 21 mesures), plus contrôle
  receiver. Bruts, comparaisons, ressources, mémoire et exclusions conservés.
- **Décision :** conserver les deux modes; l'hypothèse d'un choix universel
  est rejetée. Coopératif favorisé à 1 000 prix; les plus grands lots rendent
  le scalaire compétitif pour OU/Vasicek. Les couples threads/blocs et les
  réserves statistiques figurent dans le rapport, sans transfert automatique
  aux recettes ni à un autre GPU. Aucun kernel mathématique n'est refactorisé.
- **Preuve :** [rapport, tableaux et bruts portables](../performance-reports/jamshidian-strategy-scaling-sm89-2026-09-08.md).
  Binaire mesuré SHA-256
  `154f173bf24d2a744463e2de4d929b1d9839b703fc672146815be27cefb5bce3`;
  archive source/inputs SHA-256
  `4261bb32f353aca4249358ae13739d660f6c47a6e5bb0c7abbd33d6b695b1481`.
  Ressources exactes : 37–85 registres, zéro stack/local et zéro instruction
  SASS LDL/STL; buffers globaux 40–68 Mio à 2²⁰ prix, hors contexte.
- **Limites / coordination :** clôture de l'étude, pas de tous ses candidats.
  346 mesures ont une référence numérique incomplète; `NUM-021` reste ouvert.
  51 des 84 géométries payer confirmées passent les gates locaux numérique et
  CV; aucune répétition favorable ne remplace les autres. Comparaison entre
  modes, pas certification indépendante, publication ou rebaseline. Portée
  distincte de `PERF-006` (ELLPACK), `PERF-018` (caplets/Black–Scholes) et des
  constats MC/LSM `PERF-016/017/019`, qui restent ouverts.
- **Réouvrir seulement si :** la composition, le calendrier, le domaine, le
  kernel, le compilateur ou le GPU change le classement, un retuning est
  proposé hors couverture, ou une preuve manquante/inéligible est réutilisée
  comme qualification positive.

### BUILD-008 — Permettre la composition de plusieurs modèles ajustés dans une unité CUDA

- **Nature :** ouvert puis corrigé et fermé le 2026-09-08; sévérité moyenne,
  priorité moyenne, confiance prouvée. Propriétaire : agent référent.
- **Signature initiale :** les deux `term_structure_impl.cuh` de courbes
  n'avaient pas de garde d'inclusion; composer CIR++ et Hull–White dans une
  même unité CUDA redéfinissait 14 fonctions Nelson–Siegel/Svensson.
- **Correction :** `#pragma once` ajouté aux deux headers propriétaires, sans
  changement mathématique ni garde recopiée chez leurs consommateurs.
  Règle durable dans le contrat `cuda/model-analytics-contract.md`.
- **Preuve :** le banc composant les sept variantes compile en Release SM89
  et exécute l'intégralité de `PERF-022`; build final réussi et 56 contrôles
  GPU terminés, avec les sept références identiques au pilote mesuré, rejets
  de `NUM-021` compris. Les headers originaux sont reconstructibles depuis Git;
  le log du build réussi est archivé avec le binaire.
- **Réouvrir seulement si :** l'inclusion répétée d'un header propriétaire
  redevient non idempotente ou plusieurs compositions partagées ne peuvent
  plus coexister dans une unité CUDA.

## Objet

Ce document est le registre compact des constats issus de `query.md` qui sont
effectivement clos. Avant de creer un constat, rechercher ici une cause et un
perimetre equivalents : une autre formulation ou une autre correction proposee
ne justifie pas un nouvel identifiant.

Chaque entree conserve la signature du probleme initial, la nature de la
cloture, la decision prise, sa preuve et la condition minimale de reouverture.
Les regles durables restent dans les contrats d'implementation. Un constat qui
regresse reprend son identifiant et retourne dans `response.md` avec son
historique de cloture.

## Analytics

### ANALYTICS-001 — Ne pas faire dependre les analytics modele du schedule ou des formules d'un produit concret

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** des analytics fixed income situes sous `src/model` incluaient le schedule et des types de formule appartenant a un produit concret.
- **Cloture :** `BusinessDayFixedLegScheduleView` porte la vue partagee sous `src/common`; les produits adaptent leurs parametres sans devenir une dependance des providers modele.
- **Preuve :** recherche finale sur les 18 fichiers analytics modele sans aucun include `src/product`; tests du contrat analytics fixed income et matrice CUDA passes.
- **Reouvrir seulement si :** un provider ou une formule analytique sous `src/model` inclut de nouveau un produit concret.

## Numerical robustness

### NUM-001 — Interdire a Jamshidian de publier une racine apres stagnation ou epuisement des iterations

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** les solveurs Jamshidian scalaire et cooperatif pouvaient rendre leur dernier milieu de bracket sans certifier le residu apres stagnation FP32 ou nombre maximal d'iterations.
- **Cloture :** le residu final est certifie; un resultat non convergent devient `NaN` au lieu d'un prix silencieusement plausible.
- **Preuve :** `numerical_robustness_cuda` couvre iteration maximale nulle et bracket FP32 effondre pour les deux solveurs; 48/48 tests CUDA passent.
- **Reouvrir seulement si :** un chemin peut de nouveau produire une racine finie sans satisfaire la tolerance documentee.

### NUM-002 — Distinguer les causes d'echec des regressions Longstaff-Schwartz

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** un booleen unique confondait absence de candidats, echantillon sous-determine et echec numerique, puis conservait silencieusement la continuation.
- **Cloture :** `RegressionStatus` distingue succes, zero candidat, `candidate_count <= basis_size`, statistiques non finies, echec de Cholesky et coefficients non finis. Les deux premiers cas non resolubles conservent explicitement la continuation et restent comptes; les trois echecs numeriques invalident prix et erreur standard, remontent dans `LaunchResult` et bloquent les generateurs avant publication.
- **Preuve :** cinq causes synthetiques classees sur GPU, test d'invalidation `NaN`, nouveau test OU/G2 payer/receiver reproductible bit a bit, sept tests LSM cibles passes et generateurs OU/G2 executes avec diagnostics non fatals explicites. Les anciens echecs Heston ligne 2 et Levy ligne 3 sont identifies comme `insufficient_candidates`, pas comme echecs de factorisation.
- **Reouvrir seulement si :** une cause de regression redevient indifferenciee, un echec numerique fatal peut produire un prix fini publiable, ou un cas vide/sous-determine est traite sans politique ni diagnostic explicites.

### NUM-003 — Preserver les coefficients et la decision Longstaff-Schwartz en FP64

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** les coefficients de regression FP64 et la prediction de continuation etaient retrecis en FP32 avant la comparaison exercice/continuation.
- **Cloture :** coefficients, prediction et comparaison restent en `double`; `exercise_decision.cuh` centralise la selection sans cast intermediaire.
- **Preuve :** test d'une frontiere sensible ou le chemin FP64 choisit `1.0`, contre `0.25` pour l'ancien retrecissement; tests LSM et matrice CUDA complets passent.
- **Reouvrir seulement si :** un cast FP32 reapparait entre regression, prediction et decision d'exercice.

### NUM-004 — Stabiliser les endpoints Schobel-Zhu quand la mean reversion est petite

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** les expressions `1 - exp(-x)` des variances et correlations d'endpoint perdaient leur precision lorsque `mean_reversion * dt` etait petit.
- **Cloture :** les differences exponentielles utilisent `expm1f` et conservent les limites analytiques attendues.
- **Preuve :** grille de 10 points comparee a une reference host FP64, incluant mean reversion 0.03 et 10 ainsi que de petits pas; `numerical_robustness_cuda` passe.
- **Reouvrir seulement si :** une nouvelle formule d'endpoint reutilise une soustraction exponentielle instable ou echoue sur cette grille.

### NUM-005 — Refuser les moments Monte Carlo invalides au lieu de masquer leur variance

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** la reduction acceptait moins de deux echantillons ou des moments non finis et rabattait toute variance negative vers zero.
- **Cloture :** cardinalite et finitude sont validees; une variance materiellement negative produit `NaN`, seul l'arrondi negatif borne est ramene a zero.
- **Preuve :** cas valides, 0/1 echantillon, NaN, Inf, variance negative et erreur d'arrondi couverts dans `numerical_robustness_cuda`.
- **Reouvrir seulement si :** un prix ou une erreur standard peut etre publie depuis des moments non finis, insuffisants ou incoherents.

## Naming

### NAME-004 — Ne plus inclure textuellement des fichiers `.cu` comme headers d'implementation

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** 27 fragments `dynamics.cu`, `analytics.cu` ou `term_structure.cu` etaient inclus par `#include` au lieu d'etre des unites de traduction CUDA.
- **Cloture :** les fragments inclus portent `*_impl.cuh`; les `.cu` restants sont des unites autonomes enregistrees dans CMake.
- **Preuve :** build `all_models` en 315 etapes et recherche statique sans aucun `#include "*.cu"`.
- **Reouvrir seulement si :** un fichier `.cu` est de nouveau inclus textuellement ou echappe au graphe CMake.

### NAME-005 — Remplacer les noms affines symboliques et les types redondants

- **Nature :** devenu inapplicable apres meta-revue le 2026-08-27; aucun changement de code.
- **Signature :** les analytics fixed income exposent `log_A`, `A` et `B`, certains types ajustes repetent le modele dans leur nom et plusieurs implementations emploient `a`/`b` localement.
- **Cloture :** [`cuda/model-analytics-contract.md`](../cuda/model-analytics-contract.md) definit explicitement `log_A`, `A` et `B` comme la surface canonique des coefficients affines et distingue le loading `B` a un ou deux facteurs. Cette notation mathematique publique est donc une exception documentee, pas une incoherence. La seule repetition d'un nom de modele dans un type ne prouve ni ambiguite, ni collision, ni impact technique et reste une preference de style exclue de l'audit.
- **Preuve :** section `Signatures fixed income`, contrat des providers et section `Nommage` du meme contrat; meta-revue E13.
- **Reouvrir seulement si :** une ambiguite, collision, erreur d'usage ou divergence entre modeles est attribuee a ces noms, ou si le contrat canonique abandonne explicitement cette notation.

## Project structure

### STRUCT-001 — Garder les pricing policies produit hors de `src/common`

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** des policies fixed income et equity concretes vivaient dans `common` et incluaient directement les parametres d'un produit.
- **Cloture :** les policies sont dans `src/product/<product>/pricing_policy.cuh`; `common` ne conserve que les primitives independantes des produits.
- **Preuve :** recherche statique sans include produit dans les analytics modele; builds host/CUDA complets passes.
- **Reouvrir seulement si :** une policy liee a un produit concret ou son header de parametres retourne sous `src/common`.

### STRUCT-009 — Aligner chemins, namespaces et responsabilites des modules communs

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** le helper mean-reverting etait sous `src/model/fixed_income/common` sans namespace `common`, deux abstractions de grille temporelle divergeaient et le sampling restait regroupe historiquement.
- **Cloture :** helper deplace sous `src/common/fixed_income`, `time_grid.cuh` inutilise supprime et sampling/LSM repartis en modules nommes par responsabilite.
- **Preuve :** anciens chemins absents, documentation `src/common` alignee et builds agreges passes.
- **Reouvrir seulement si :** chemins et namespaces divergent de nouveau ou un module commun redevient un regroupement sans responsabilite coherente.

## Tools and src ownership

### BOUNDARY-001 — Supprimer toute dependance de `src` vers les composants de `tools`

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** les loaders de `src` incluaient `tools/datasets/dataset_validation.hpp` et leurs cibles liaient la serialisation/generation offline de `tools`.
- **Cloture :** validation de lecture deplacee dans `src/common/dataset_validation.*`; `tools` depend de cette cible et non l'inverse.
- **Preuve :** aucun include `tools` sous `src`; 3/3 tests host et build de 1 193 etapes des `price_generators` sans mathDx passent.
- **Reouvrir seulement si :** un fichier ou une cible runtime de `src` depend de nouveau de `tools`.

### BOUNDARY-003 — Centraliser dans `src` les primitives mathematiques reutilisees par `tools`

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** les generateurs recopiaient les forwards instantanes Nelson-Siegel/Svensson et la conversion mean-reversion/dispersion stationnaire appartenant au modele.
- **Cloture :** primitives pures host/device exposees sous `src/curve` et `src/common/fixed_income`, puis consommees par les generateurs.
- **Preuve :** une seule definition par primitive, tests de parite FP32/FP64 et builds generateurs passes.
- **Reouvrir seulement si :** un generateur reimplemente une identite mathematique deja possedee par `src`.

### BOUNDARY-005 — Supprimer les facades sans consommateur et les dossiers placeholders

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** la facade CMake `ai_factory_dataset_tools`, un `.gitkeep` sous `src/generative` et des dependances tools de la facade runtime n'avaient pas de consommateur ou de contrat reel.
- **Cloture :** facade et placeholder supprimes; la facade runtime ne publie plus de composant offline.
- **Preuve :** anciens noms absents des sources et du graphe CMake; builds agreges passes.
- **Reouvrir seulement si :** une facade de compatibilite ou un dossier vide est ajoute sans consommateur, export ou contrat documente.

## Build and CUDA instantiations

### BUILD-001 — Ne pas publier sans mathDx une cible rough Bergomi qui ne peut pas etre liee

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** CMake enregistrait un generateur de prix rough Bergomi sans `AI_FACTORY_MATHDX_ROOT`, puis le link echouait faute de CUDA runtime et d'instanciation du launcher FFT.
- **Cloture :** l'enregistrement des generateurs dependants de mathDx est garde par `AI_FACTORY_MATHDX_ROOT`; les cibles publiees sans mathDx sont toutes constructibles.
- **Preuve :** configuration fraiche sans mathDx et construction reelle des 1 193 etapes de `price_generators`.
- **Reouvrir seulement si :** une configuration supportee publie de nouveau une cible dont une dependance ou instanciation requise est absente.

## Performance

### PERF-003 — Reduire la pression registre des reductions LSM

- **Nature :** hypothese de performance refutee par mesure le 2026-08-27; aucun prototype conserve.
- **Signature :** remplacer l'accumulation combinee de 28 statistiques FP64 par thread soit par deux passages Gram/second membre, soit par une distribution des statistiques entre lanes d'un warp.
- **Cloture :** le kernel combine d'origine est conserve. La separation Gram/second membre reduit les registres G2 SM89 de 118 a 103 et 78, mais ne change pas l'occupation du passage Gram (33,3 %), ajoute un kernel et une lecture des chemins par date, et augmente le temps profile de 435 ms a 534 ms (+22,6 %). La distribution par warp conserve elle aussi les prix G2 bit a bit, mais reduit de 32 fois le parallelisme entre trajectoires et fait passer le temps kernel du generateur de 0,415 s a 4,965 s, soit environ x12.
- **Preuve :** profils Nsight Systems du generateur G2 1 000 lignes, diagnostics exacts SM89 sur RTX 4090 Laptop, comparaison des JSON hors metadonnees de timing bit a bit et sept tests LSM GPU passes pour chaque prototype fonctionnel. La baseline finale G2 retrouve 118 registres, aucune memoire locale, 33,3 % d'occupation theorique et 0,413 s kernel.
- **Reouvrir seulement si :** une autre reduction conserve un chemin par thread et demontre un gain reproductible sur les familles American et Bermudan representatives sans modifier prix, erreurs standards, FP64 ou diagnostic type.

### PERF-011 — Trier les lignes LSM American/Bermudan par nombre de dates d'exercice pour supprimer les blocs backward inactifs

- **Nature :** hypothese de performance refutee par mesure le 2026-08-27.
- **Signature :** tri intra-batch ou global par `exercise_count`, reduction de `grid.y` au prefixe encore actif, puis restauration de l'ordre original des resultats.
- **Cloture :** 354 executions CUDA et 42 configurations donnent environ 1,3 % de gain median global; le tri global peut regresser, notamment Heston de 2,1 %, et la complexite de permutation n'est pas justifiee. Une cle secondaire par nombre de pas ne supprime aucune divergence intra-warp et aucun profil ne lui attribue un cout significatif.
- **Preuve :** resultats bit-identiques; sur le catalogue 1 000 lignes a 1 048 576 paths, le tri intra-batch gagne 5,9 % en Variance Gamma, 4,4 % en NIG, 2,0 % en Heston et 3,6 % en Bates, sans gain robuste sur les distributions homogenes ou etroites.
- **Limite de provenance :** le rapport brut, le manifeste des 42 configurations et l'empreinte des binaires ne sont pas versionnes dans ce quadriptyque. Cette entree conserve la decision historique, mais ses chiffres ne constituent pas une baseline courante sous le referentiel v2 tant que cette provenance n'est pas rattachee a un artefact durable.
- **Reouvrir seulement si :** un nouveau materiel ou une distribution d'exercices differente montre un gain reproductible suffisamment important pour justifier indices, permutation et scatter, ou si la provenance historique ne peut pas etre produite au moment ou cette decision doit etre reutilisee.

## Naming — second passage de remediation

### NAME-002 — Clarifier le namespace public des produits

- **Nature :** clarifie et verifie le 2026-08-27.
- **Signature :** le namespace public plat `ai_factory::workbench::product` pouvait etre lu comme une taxonomie incomplete ou entrer en contradiction avec les sous-repertoires du catalogue.
- **Cloture :** le namespace plat est confirme comme surface publique volontaire; les types portent le nom descriptif du produit et les helpers internes restent dans `detail`. Cette regle et l'absence de namespaces miroir par dossier sont documentees dans le README racine.
- **Preuve :** inventaire des declarations produit, compilation agregee de tous les modeles et tests host/CUDA; aucun symbole public ambigu n'a ete introduit.
- **Reouvrir seulement si :** deux produits exposent des symboles publics en collision, si une API exige une taxonomie imbriquee stable, ou si le contrat de namespace est abandonne.

### NAME-003 — Distinguer les launchers de modele des primitives generiques

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** les launchers rough publics `launch_european_option_cuda` et `launch_hybrid_fft_cuda` ne nommaient pas le modele alors que leurs signatures et implementations lui etaient propres.
- **Cloture :** les launchers externes rough Bergomi et rough SABR portent des noms qualifies par modele; seuls les composants reellement communs sous `src/common/volterra` conservent des noms methodes-neutres.
- **Preuve :** sources, tests, catalogues, probe de validation et templates de bindings migrent atomiquement; `all_models`, la matrice CUDA et la compilation manuelle du probe rough SABR passent.
- **Reouvrir seulement si :** un launcher public propre a un modele reprend un nom generique, ou si deux implementations distinctes revendiquent le meme symbole non qualifie.

## Project structure — second passage de remediation

### STRUCT-006 — Unifier les 46 loaders de datasets de parametres

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** chaque produit, modele et courbe repetait ouverture JSON, verification de famille, cardinalite, enveloppe `parameters` et parcours des lignes, avec des diagnostics susceptibles de diverger.
- **Cloture :** `src/common/dataset_validation.*` possede la lecture et la validation communes via `ParameterDatasetFamily`, `read_parameter_dataset` et `load_parameter_rows`; les 46 loaders les reutilisent. Le loader de swaptions garde seulement son assemblage specialise du pool de schedules.
- **Preuve :** aucune ouverture directe `std::ifstream stream(dataset_path)` ne subsiste dans les loaders; cas d'erreurs communs, familles et schedules multi-lignes sont testes; 3/3 tests host passent.
- **Reouvrir seulement si :** un loader reimplemente l'enveloppe commune, produit un diagnostic incompatible, ou si une nouvelle famille ne peut pas exprimer sa validation specialisee apres la lecture commune.

## Pricing policies and concepts — second passage de remediation

### POLICY-001 — Rendre explicites et testables les budgets de stockage des policies

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** les concepts imposaient des caps de taille a 128, 256 ou 2 048 octets sans contrat central, message actionnable ni preuve de ressources sur les architectures ciblees.
- **Cloture :** le contrat publie les cinq budgets, les raisons et les alternatives `view`/pool device; les `static_assert` indiquent le cap exact et la remediation. Un test CUDA compile chaque limite et rejette conceptuellement la premiere taille hors budget.
- **Preuve :** le test `policy_size_budgets_cuda` passe sur SM89; les probes offline SM75/86/89 conservent les tailles demandees sans spill observe dans les objets inspectes. Les cinq geometries atteignent 100 % d'occupation theorique sur le SM89 mesure.
- **Reouvrir seulement si :** un cap change sans mesure, une policy valide depasse le budget, ou un materiel cible montre une regression de ressources qui justifie un budget architecture-dependant.

## Tools and src ownership — second passage de remediation

### BOUNDARY-002 — Remplacer le booleen public de construction des prix par un type metier

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** le booleen `cartesian_product` traversait launchers, outils et recettes sans rendre lisibles les deux modes d'indexation.
- **Cloture :** `PriceConstruction::{aligned,cartesian}` est la representation canonique dans `src/common/price_construction.cuh`; la migration est atomique jusque dans les templates de bindings et aucun wrapper booleen transitoire n'est conserve.
- **Preuve :** recherche statique sans ancien parametre public; tests aligned/cartesian, cardinalites et debordements; compilation des generateurs, de tous les modeles et des tests CUDA.
- **Reouvrir seulement si :** un booleen ou un entier brut encode de nouveau le mode de construction a une frontiere publique.

## Performance — second passage de remediation

### PERF-001 — Reduire le cout du decodage d'indices de resultats

- **Nature :** corrige et mesure le 2026-08-27.
- **Signature :** chaque thread recalculait avec des divisions `size_t` les indices parametre/produit alors que les grilles CUDA publiees sont bornees.
- **Cloture :** `result_index.cuh` centralise les decodeurs et fournit un chemin device `uint32_t` apres validation host de la cardinalite; toute cardinalite superieure a `UINT32_MAX` est refusee avec instruction de decoupage. La specialisation compile-time du mode, sans le passage 32 bits, est rejetee faute de gain.
- **Preuve :** mappings et checksums identiques sur les modes 1D, aligned, cartesian et trois dimensions; sur SM89, la mediane passe de 0,270 ms a 0,124 ms, soit environ 54 %, sans changer les identifiants. Les CV kernel de 0,7 % et 1,3 % restent sous le seuil de 5 % du protocole.
- **Reouvrir seulement si :** une cardinalite valide exige plus de 32 bits dans un lancement unique, si un nouveau mapping contourne le decodeur canonique, ou si une architecture cible invalide le gain.

### PERF-002 — Mesurer la geometrie des kernels aux policies riches

- **Nature :** hypothese tranchee par mesure le 2026-08-27; geometrie de production conservee.
- **Signature :** les policies Phoenix et Phoenix-memory Bates riches en registres pouvaient sembler favoriser 128 ou 256 threads plutot que les 512 utilises par les generateurs.
- **Cloture :** 512 threads restent la geometrie par defaut : une occupation theorique plus faible ne predit pas le debit de ce workload et les variantes plus petites sont plus lentes.
- **Preuve :** a 32 prix et 4 096 trajectoires, Phoenix mesure 11,10/6,28/4,04 ms et Phoenix-memory 11,14/6,30/4,05 ms pour 128/256/512 threads. Prix identiques, 101 registres, 112 octets de shared statique et aucun spill local pour les trois variantes sur SM89.
- **Reouvrir seulement si :** la policy, la distribution des calendriers, l'architecture ou le nombre de trajectoires change assez pour inverser reproductiblement ce classement.

### PERF-004 — Borner l'inlining des grandes primitives noncentral-chi-square

- **Nature :** corrige et mesure le 2026-08-27.
- **Signature :** plusieurs grandes routines numeriques etaient force-inlinees dans les kernels CIR, augmentant registres, taille de code et temps de compilation sans preuve de debit.
- **Cloture :** les series gamma, fraction continue, paire gamma, melange de Poisson et saddlepoint sont des helpers internes `static __noinline__`; la variante force-inline ne vit que dans une cible experimentale separee.
- **Preuve :** sur 16 384 resultats, le chemin de production mesure 1,063 ms contre 1,158 ms force-inline (-8,2 %), avec des CV kernel de 0,5 % et 1,2 %. Il utilise 56 contre 64 registres, 75 % contre 66,7 % d'occupation theorique, aucun spill; l'archive passe d'environ 3,40 Mo a 1,64 Mo. Les compteurs de stalls Nsight Compute n'etaient pas accessibles (`ERR_NVGPUCTRPERM`) et ne sont pas revendiques.
- **Reouvrir seulement si :** un compilateur ou GPU cible mesure un avantage reproductible de l'inlining sans inflation disproportionnee des registres, cubins ou temps de build.

### PERF-006 — Coalescer et cooperer sur les schedules explicites de swaptions

- **Nature :** corrige et mesure le 2026-08-27.
- **Signature :** le pool row-major des schedules explicites donnait des lectures stridees et un seul thread effectuait l'evaluation Jamshidian d'une ligne heterogene.
- **Cloture :** le loader produit un ELLPACK payment-major, les vues transportent le stride de produits et les surcharges explicites CIR, OU, Vasicek et Hull-White evaluent les paiements cooperativement. Le chemin regulier scalaire reste intact.
- **Preuve :** le cas heterogene 2–30 paiements passe d'environ 6,99 ms a 0,292 ms (environ x24); l'homogene explicite reste comparable au regulier, 0,296 contre 0,286 ms. Le kernel explicite descend a 46 registres, 83,3 % d'occupation theorique et aucun spill; un test deux lignes verifie exactement la transposition du pool.
- **Reouvrir seulement si :** une nouvelle representation de calendrier recree des lectures stridees, si le chemin explicite diverge numeriquement du contrat, ou si une autre architecture inverse le gain.

### PERF-008 — Conserver FP64 pour les accumulations numeriquement sensibles

- **Nature :** hypothese de precision mixte refutee pour le contrat generique le 2026-08-27.
- **Signature :** remplacer les sommes FP64 de moments et statistiques par FP32 compense ou par des chunks FP32 pouvait accelerer les boucles chaudes.
- **Cloture :** FP64 reste le contrat commun; aucune approximation globale n'est introduite sans budget d'erreur propre a une famille de prix et a son erreur standard.
- **Preuve :** sur un flux non negatif a echelles mixtes, FP64 mesure 11,77 ms avec reference exacte, FP32 compense 3,10 ms mais `1,47e-2` d'erreur absolue, et chunks FP32 5,04 ms mais `1,43e-1`. Les erreurs relatives sont petites mais non nulles et ne satisfont pas un contrat universel non specifie.
- **Reouvrir seulement si :** une famille documente une tolerance prix/erreur standard, une distribution representative et une validation independante qui autorisent explicitement une accumulation mixte specialisee.

### PERF-009 — Quantifier et amortir le cout fixe des launchers courts

- **Nature :** mesure et decision documentee le 2026-08-27; validation fail-fast conservee.
- **Signature :** allocations, inspections de pointeurs et synchronisations des petits launchers closed-form pouvaient dominer les prix tres courts et suggerer un cache de metadata device.
- **Cloture :** les appels doivent etre groupes par les lanceurs grid-stride existants; aucun cache de pointeur n'est ajoute car `cudaFree` et la reutilisation d'adresse rendraient sa duree de vie incorrecte. La mutualisation ulterieure des runners a ete traitee independamment sous `STRUCT-007`/`BOUNDARY-004`, sans introduire ce cache non possede.
- **Preuve :** 256 appels par echantillon mesurent environ 5–7 microsecondes par lancement pour 1, 32 et 1 024 resultats. La variance dynamique est elevee (CV 22–28 %), donc le protocole classe les comparaisons fines comme inconclusives, mais confirme l'ordre de grandeur; les generateurs catalogue amortissent deja sur 1 000 resultats.
- **Reouvrir seulement si :** un appel public unitaire devient un workload catalogue significatif, ou si une API de contexte possede explicitement et surement la duree de vie des allocations.

### PERF-012 — Mesurer le crossover convolution directe/FFT pour Volterra

- **Nature :** hypothese de chemin direct refutee par mesure le 2026-08-27.
- **Signature :** une convolution directe bornee pouvait eviter le cout fixe cuFFTDx aux tres petites grilles Volterra.
- **Cloture :** le chemin de production reste exclusivement FFT. L'implementation directe reproductible est exclue par preprocesseur de la bibliotheque normale et ne vit que dans une archive/benchmark experimental borne a 32 pas.
- **Preuve :** a 8 pas, FFT mesure 0,050 ms contre 0,068 ms direct (+34,7 %); a 16 pas 0,071 contre 0,216 ms; a 32 pas 0,099 contre 0,803 ms. Prix identiques a 8/16 pas et ecart de `1,86e-9` a 32 pas avec le meme mapping Philox.
- **Reouvrir seulement si :** un nouveau GPU, compilateur ou primitive de convolution directe bat reproductiblement FFT a une longueur supportee tout en conservant mapping aleatoire, prix et taille de production.

### PERF-013 — Retuner les instanciations FFT longues

- **Nature :** corrige et mesure le 2026-08-27.
- **Signature :** l'instanciation FFT 8 192 points a 32 elements par thread utilisait 139 registres, limitait l'occupation et gonflait les archives rough.
- **Cloture :** cette longueur utilise 16 elements par thread; le reste de la courbe 16–4 096 conserve ses specialisations mesurees et le dispatch deterministe.
- **Preuve :** la mediane 8 192 tombe de 22,45 a 17,87 ms (-20,4 %), les registres de 139 a 72 et l'occupation theorique de 16,7 % a 33,3 %, sans spill. Les archives Bergomi/SABR diminuent d'environ 10,3 % et la compilation mesuree d'environ 6,6 %; prix et erreur standard restent dans le contrat et tous les tests rough passent.
- **Reouvrir seulement si :** une architecture ou version cuFFTDx change le classement, si une longueur devient dominante sans baseline, ou si registres/spills regressent.

### PERF-014 — Borner le workspace Volterra et evaluer le recouvrement multi-stream

- **Nature :** mesure et decision documentee le 2026-08-27; strategie de production conservee.
- **Signature :** le chunking sequentiel et un seul stream pouvaient sous-utiliser le GPU ou retenir un workspace excessif sur les jeux Volterra de production.
- **Cloture :** un workspace reutilisable de 65 536 trajectoires et un stream restent la strategie canonique. Il borne le cas 252 pas/1 048 576 trajectoires a environ 63,14 MiB, sature mieux le GPU que les petits chunks et preserve l'ordre des reductions et les seeds.
- **Preuve :** medianes SM89 de 37,68, 14,07 et 11,17 ms pour des chunks 4 096, 16 384 et 65 536, sorties strictement identiques; huit prix sequentiels mesurent 91,83 ms, soit 11,48 ms/prix, sans gain de batching concurrent visible. Les quatre configurations sont conservees dans la baseline v1.
- **Reouvrir seulement si :** le budget memoire cible devient inferieur a 63,14 MiB, si un nouveau GPU montre un recouvrement reproductible superieur a 5 %, ou si le workspace peut etre partage sans modifier ordre deterministe ni duree de vie.

## Project structure — troisieme passage de remediation

### STRUCT-007 — Centraliser l'orchestration CUDA des generateurs de catalogue

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** les recettes ordinaires recopiaient allocations, transferts, warmup, evenements, chronometrage, copie des resultats et nettoyage CUDA.
- **Cloture :** `tools/cuda/pricing_runner.cuh` fournit buffers et evenements RAII ainsi que les chemins types analytique et Monte Carlo. Les 293 recettes ordinaires et les helpers de swaptions reutilisent cette execution; huit pipelines Longstaff-Schwartz et deux anciens pipelines cuFFTDx conservent leur workspace algorithmique comme echappatoires explicites et bornees.
- **Preuve :** le checker `catalog_generator_boundaries` inspecte 382 recettes, accepte exactement les 10 echappatoires revues et refuse toute nouvelle gestion CUDA brute; les 1 156 etapes de `price_generators` passent. Le test CUDA autonome du runner verifie sur GPU allocations, transferts, resultats analytique/Monte Carlo, erreurs standards, evenements et chronometrage. Les tests de schema et de catalogue passent. Preuve consolidee E21.
- **Reouvrir seulement si :** une recette ordinaire gere de nouveau une ressource CUDA, si une echappatoire apparait sans revue explicite, si le runner perd sa propriete RAII, ou si la publication change le schema des artefacts.

## Tools and src ownership — troisieme passage de remediation

### BOUNDARY-004 — Separer les responsabilites offline de `tools/datasets`

- **Nature :** corrige et verifie le 2026-08-27.
- **Signature :** `tools/datasets/dataset.*` melangeait sampling, grilles, assemblage des datasets, JSON, YAML et orchestration de pricing.
- **Cloture :** le monolithe est supprime. `sampling.*`, `artifact_io.*`, `parameter_dataset.*` et `price_dataset.*` possedent des API et bibliotheques distinctes; l'execution CUDA vit sous `tools/cuda`, les orchestrations produit reutilisables sous `tools/pricing`, et les generateurs lient seulement les etapes qu'ils emploient. La facade `ai_factory_dataset_core` restante est une cible `INTERFACE` de compatibilite et ne reintroduit aucune implementation agregee.
- **Preuve :** quatre tests de stage independants, les tests `dataset_catalog`, `dataset_loaders` et `rough_sabr_dataset_loader`, le test GPU autonome du runner et les deux matrices de generateurs passent. Le checker interdit le retour du monolithe et des anciens helpers. Preuve consolidee E21.
- **Reouvrir seulement si :** sampling, serialisation, assemblage ou execution sont de nouveau fusionnes dans une implementation commune, si un generateur depend d'une etape inutilisee, ou si une etape perd son test independant.

## Project structure — passage de remediation du 2026-08-28

### STRUCT-012 — Declarer et generer les recettes American/LSM

- **Nature :** corrige et verifie le 2026-08-28.
- **Signature :** huit recettes American recopiaient manuellement le workspace,
  l'execution, les diagnostics et la publication Longstaff-Schwartz derriere
  une escape hatch du checker.
- **Cloture :** `AmericanRecipeSpec`, un template commun et
  `american_option_price_generation.cuh` generent les huit corps minces; le
  checker ne conserve aucune escape hatch CUDA American.
- **Preuve :** les huit recettes sont zero-diff dans la generation complete,
  les huit executables compilent, les quatre tests CUDA American/LSM passent et
  le checker refuse toute recette possedant de nouveau ses ressources CUDA;
  preuve ciblee E23.
- **Reouvrir seulement si :** une recette American redevient manuelle, sort du
  manifeste, gere directement CUDA ou peut publier apres un diagnostic de
  regression fatal.

### STRUCT-013 — Publier les deux recettes de samples contractuelles par modele

- **Nature :** corrige structurellement et verifie le 2026-08-28; le defaut
  numerique isole est transfere sous `NUM-007`.
- **Signature :** les 24 modeles ne possedaient pas tous leurs bindings
  `sample.cuh/.cu` et leurs deux recettes contractuelles `samples_01` et
  `samples_02`, imposant une extension manuelle incomplete.
- **Cloture :** chaque modele possede ses deux corps minces generes, son helper
  type et son binding; l'agregat CMake `sample_generators` est derive de
  l'arborescence et les dependances Volterra restent conditionnees par mathDx.
- **Preuve :** les 48 executables compilent sur SM89 avec mathDx; un build frais
  sans mathDx compile les 40 recettes independantes en 168 etapes; 47/48 smoke
  tests produisent et rechargent leur JSON/YAML. Le seul echec n'est ni une
  absence ni un drift de codegen : Quadratic rough-Heston `samples_02` produit
  une valeur non finie et reste ouvert sous `NUM-007`; preuves E20 et E22.
- **Reouvrir seulement si :** un modele perd un des deux layouts, si un binding
  ou une recette doit etre ajoute manuellement hors specification, si le
  zero-diff ne couvre plus cette surface ou si un target disponible ne linke
  plus avec ses dependances declarees.

## Numerical robustness — passage de remediation du 2026-08-28

### NUM-006 — Definir la frontiere CEV des dynamiques SABR au lieu de la projeter silencieusement

- **Nature :** corrige et verifie le 2026-08-28.
- **Signature :** SABR et rough SABR projetaient la coordonnee de Lamperti sur
  `1e-12` apres un franchissement de zero, permettant a une trajectoire CEV de
  repartir sans condition de frontiere documentee.
- **Cloture :** une primitive commune impose la frontiere absorbante : un spot
  traverse devient `log_spot = -inf`, ne ressuscite pas, tandis que la
  volatilite et la consommation Philox poursuivent leur contrat.
- **Preuve :** contrat dynamics mis a jour; tests forces `beta=0` et `beta=0.5`;
  comparaison SABR beta-zero a la loi brownienne tuee, raffinement de pas
  rough-SABR, reproductibilite et trois tests CUDA cibles passes sur SM89;
  preuve E23.
- **Reouvrir seulement si :** une projection positive reapparait, une
  trajectoire absorbee peut repartir, le mapping Philox change ou un domaine
  publie montre un biais hors de la borne/convergence documentee.

## CUDA safety — passage de remediation du 2026-08-28

### CUDA-001 — Eliminer les lectures non initialisees du `PreparedRow` Bermudan

- **Nature :** corrige et verifie le 2026-08-28.
- **Signature :** les copies structurelles du `PreparedRow` Bermudan lisaient
  le padding non initialise du workspace, produisant 1 792 erreurs initcheck.
- **Cloture :** le workspace `PreparedRow` est initialise deterministement
  avant le kernel de preparation, sans changer sa representation ni les copies
  device.
- **Preuve :** `compute-sanitizer --tool initcheck` sur le test Bermudan OU/G2
  passe de 1 792 a zero erreur; memcheck, racecheck et synccheck restent propres
  et les sorties fonctionnelles restent reproductibles; preuve E23.
- **Reouvrir seulement si :** initcheck signale de nouveau une lecture de row,
  si l'initialisation ajoute un cout materiel mesure ou si un champ semantique
  peut rester non initialise.

## Remediation portabilite et samples du 2026-08-28

### BUILD-002 — Ne pas confondre tuning SM89 et compatibilite cuFFTDx

- **Nature :** corrige par matrice explicite et builds representatifs le
  2026-08-28.
- **Signature originale :** activer mathDx imposait exactement `sm_89`, alors
  que le projet annoncait `75;86;89`; une provenance de tuning bloquait ainsi
  fonctionnellement les engines Gaussian-Volterra sur les autres GPU.
- **Cloture :** cuFFTDx 26.06 accepte les descripteurs
  `75,80,86,87,89,90,100,103,110,120,121`. Un build mono-architecture choisit
  son descripteur exact; un fatbin choisit le plus ancien descripteur demande
  comme profil d'implementation et emet chaque architecture nvcc. Includes et
  definitions mathDx sont possedes par l'interface CMake unique
  `ai_factory_cufftdx`.
- **Preuve :** configurations et compilations fraiches pricing plus sample
  passent avec mathDx pour SM75, SM86, SM89 et le fatbin `75;86;89`; le fatbin
  sample s'execute sur le SM89 disponible. Le preset local SM89 et le build
  sans mathDx restent utilisables; une architecture sans descripteur est
  refusee avec la matrice supportee. README et messages CMake distinguent
  compatibilite offline et runtime mesure.
- **Reouvrir seulement si :** une version mathDx change sa matrice, si un
  descripteur annonce ne compile plus un binding pricing ou sample, si un
  fatbin selectionne une implementation non executable sur une cible, ou si
  un profil de performance redevient une garde fonctionnelle.

## Naming et frontieres de policies — passage de remediation du 2026-08-28

### NAME-100 — Nommer une policy d'apres la responsabilite qu'elle implemente

- **Nature :** corrige et verifie le 2026-08-28.
- **Signature :** les types `*HybridDriverPolicy`, `PreparedDriver` et les
  methodes `driver_parameters`, `variance` et `value` etaient presentes comme
  un driver de chemin, alors que leur responsabilite etait exclusivement la
  discretisation d'un noyau Volterra, ses poids, sa variance et la
  reconstruction de la cellule singuliere. Le terme masquait la separation
  entre convolution et transformation propre au modele.
- **Cloture :** les trois implementations sont desormais des
  `*HybridKernelPolicy` avec `PreparedKernel`, `kernel_parameters`,
  `volterra_variance` et `reconstruct_volterra_value`. Le vocabulaire est
  propage dans le pricer FFT, le sampling, les quatre modeles Volterra, les
  tests, CMake, le manifeste et le template codegen. Le contrat et le schema de
  composition sont documentes dans
  [`cuda/pricing-policy-composition.md`](../cuda/pricing-policy-composition.md).
- **Preuve :** recherche statique sans ancien symbole ou ancien header sous
  `src`, `tools` et `tests`, hors signatures historiques de ce registre;
  generation samples `--compare-root .`
  zero-diff; compilation des quatre pricers europeens et des quatre samplers
  Volterra; cinq tests CUDA cibles passes sur SM89; preuve E27.
- **Reouvrir seulement si :** une policy est nommee comme une source aleatoire,
  un modele ou un produit alors qu'elle ne possede que le noyau mathematique,
  ou si des noms generiques comme `value`/`variance` rendent de nouveau
  ambigu le passage convolution -> valeur Volterra -> etat du modele.

### POLICY-002 — Ne pas faire dependre un moteur generique des champs internes d'une policy

- **Nature :** corrige et verifie le 2026-08-28.
- **Signature :** le moteur FFT lisait directement
  `PreparedDriver::sqrt_time_step`, bien que ce champ ne fasse pas partie du
  contrat annonce. Une nouvelle implementation conforme aux methodes
  documentees pouvait donc echouer seulement a l'instanciation profonde du
  kernel CUDA.
- **Cloture :** `volterra::HybridKernelPolicy` impose exactement `prepare`,
  `far_cell_weight`, `volterra_variance` et
  `reconstruct_volterra_value`; `HybridPathPolicyFor` impose le contrat de
  chemin et l'egalite du type retourne par `kernel_parameters`. Le moteur
  conserve lui-meme `sqrt_time_step` dans son `PreparedRow` et traite
  `PreparedKernel` comme opaque. Chaque noyau concret et chaque composition
  modele/noyau sont controles par `static_assert`.
- **Preuve :** les trois kernels satisfont le concept commun; les quatre
  compositions pricing et sampling compilent; `volterra_kernel_policy_cuda`,
  `rough_bergomi_dynamics_cuda`, `rough_bergomi_european_option_cuda`,
  `rough_volterra_product_policy_cuda` et `rough_volterra_samples_cuda`
  passent sur GPU; preuve E27.
- **Reouvrir seulement si :** un moteur generique accede a un membre concret
  d'une policy hors types explicitement contractuels, si une relation de types
  entre deux policies n'est verifiee qu'au fond d'un kernel, ou si une nouvelle
  policy exige de modifier le moteur malgre un contrat semantiquement
  identique.

## Structure et naming — remediation du 2026-08-28

### STRUCT-016 — Isoler les compositions modele-produit de l'infrastructure modele

- **Nature :** corrige et verifie le 2026-08-28.
- **Signature originale :** 832 fichiers de bindings produit partageaient la
  racine de chaque modele avec dynamics, analytics, parametres, datasets,
  sampling et helpers; une exploration ne permettait pas de separer le
  catalogue de produits de l'infrastructure du modele.
- **Cloture :** chaque paire modele-produit vit sous
  `src/model/equity/<family>/<model>/product/` ou
  `src/model/fixed_income/<model>/product/[<curve>/]`. Le dossier `product/`
  ne contient reciproquement aucune infrastructure, et les noms de targets
  CMake publics restent stables malgre le deplacement physique.
- **Preuve :** inventaire exhaustif de 832 fichiers et 416 paires; checker de
  profondeur/ownership/references, codegen zero-diff, configuration CMake,
  CTests architecture et builds representatifs passes; preuve E26.
- **Reouvrir seulement si :** un binding produit revient a la racine d'un
  modele, une infrastructure entre sous `product/`, un niveau non semantique
  apparait ou un target public derive a cause du chemin physique.

### NAME-011 — Rendre le role des fichiers d'infrastructure modele immediatement lisible

- **Nature :** corrige et verifie le 2026-08-28.
- **Signature originale :** des helpers comme `hybrid_pricing.cuh`,
  `pricing_workspace.cuh`, `markovian_pricing.cuh` et `numerics.hpp` ne
  nommaient pas leur engine; plusieurs fichiers canoniques hors produits,
  notamment des couples `dynamics.cuh`/`dynamics_impl.cuh`, n'expliquaient pas
  immediatement la difference entre contrat, preparation host et definitions
  device.
- **Cloture :** les helpers portent les qualificatifs
  `volterra_fft_*`/`markovian_n_factor_*`. Les 199 fichiers C++/CUDA
  d'infrastructure hors `product/` commencent par une phrase courte de contenu
  et d'utilite; les headers publics et leurs `*_impl.cuh` ont des roles
  explicitement distincts. Le checker refuse nom non revu, nom ambigu,
  en-tete generique, profondeur inattendue et paire publique/impl mal decrite.
- **Preuve :** inventaire exhaustif des 199 fichiers, zero ancien basename ou
  reference, checker `model_source_layout`, regeneration et builds
  representatifs passes; preuve E26.
- **Reouvrir seulement si :** le role d'un fichier ne peut plus etre deduit de
  son chemin et de son nom, si son en-tete n'en precise pas contenu et utilite,
  si deux engines partagent un helper non qualifie ou si le checker est
  contourne par une nouvelle exception non documentee.

## Remediation locale du 2026-08-30

### NAME-012 — Ajouter un en-tete de responsabilite aux fichiers handwritten restants

- **Nature :** corrige et verifie le 2026-08-30.
- **Signature originale :** sept fichiers handwritten commencaient par
  pragma, include ou code, et le checker ne couvrait exhaustivement que
  l'infrastructure modele.
- **Cloture :** les sept fichiers portent une phrase specifique de contenu et
  d'utilite. `check_model_layout.py` inventorie tout fichier C++/CUDA, Python,
  CMake ou template sous `src`, `tools`, `tests` et `cmake`, plus le
  `CMakeLists.txt` racine; les outputs codegen, manifests et preambules de
  format sont classes explicitement.
- **Preuve :** `model_source_layout` accepte 829 fichiers generes et 572
  handwritten, dont quatre manifests et trois shebangs; les 572/572 ont une
  phrase valide et une fixture sans phrase est refusee.
- **Reouvrir seulement si :** un fichier handwritten du perimetre perd son
  en-tete, si une nouvelle famille echappe a l'inventaire ou si une exception
  generated/manifest/format est deduite implicitement de son contenu.

### NUM-009 — Valider integralement la preparation N-factor QRH explicite

- **Nature :** corrige et verifie le 2026-08-30.
- **Signature originale :** la surcharge QRH explicite ne validait ni tous les
  parametres modele, ni horizon, nodes et weights; un node nul et des taux ou
  feedback non finis produisaient silencieusement un `PreparedDynamics` non
  fini.
- **Cloture :** tous les champs modele sont valides selon leur domaine, les
  nodes/weights explicites sont finis et strictement positifs, `dt` et
  l'horizon d'ajustement sont controles, puis les coefficients prepares sont
  certifies finis. L'horizon inutilise est retire de la surcharge recevant un
  noyau deja ajuste.
- **Preuve :** `quadratic_rough_heston_preparation_cuda` couvre NaN/Inf pour
  chaque champ, zero/negatif pour les domaines positifs, `H=0.5`, horizon et
  `dt` invalides, chaque node/weight invalide, overflow prepare, frontieres
  valides et le contrat de noyau rough-Heston; build et CTest passent.
- **Reouvrir seulement si :** un consommateur direct peut construire un
  prepare non fini, si un noyau non positif est accepte, ou si un argument
  public de preparation redevient inutilise et non valide.

### BUILD-003 — Donner un owner CMake aux quatre bindings American additionnels

- **Nature :** corrige et verifie le 2026-08-30.
- **Signature originale :** les quatre unites American CEV, Kou, Merton et
  Schobel-Zhu existaient et leurs huit launchers etaient appeles, mais elles
  etaient absentes du manifeste early, du fragment CMake et de la compile DB;
  l'agregat CUDA echouait au link avec huit references indefinies.
- **Cloture :** les quatre modeles sont declares dans la source typee des
  recettes American; le codegen produit leur owner CMake unique et leurs huit
  recettes call/put. Les archives, generateurs et test LSM portent tous la
  meme capacite declaree.
- **Preuve :** les quatre TU figurent dans la compile DB avec un owner unique;
  les huit generateurs et l'agregat `ai_factory_cuda_tests` compilent et
  lient; `black_scholes_cev_kou_merton_schobel_zhu_american_lsm_cuda` passe
  sur RTX 4090 Laptop.
- **Reouvrir seulement si :** une TU modele-produit n'a plus exactement un
  owner CMake, si une capacite declaree diverge des recettes generees, ou si
  l'agregat/test early ne linke ou ne s'execute plus.

### CUDA-002 — Verifier les plafonds avant de dimensionner le workspace Volterra

- **Nature :** corrige et verifie le 2026-08-30.
- **Signature originale :** la division plafond des moments partiels formait
  `path_count + threads - 1`; `SIZE_MAX` wrappait a zero, sous-dimensionnait le
  workspace puis exposait une ecriture device hors bornes.
- **Cloture :** les divisions plafond n'additionnent plus avant division;
  produits, sommes et offsets sont checked, les limites de grille sont
  refusees avant allocation et la progression par chunks utilise la taille du
  dernier chunk reel. Le test adresse sur device le dernier `double2` du
  dernier bloc partiel calcule par le planner.
- **Preuve :** `volterra_fft_workspace_bounds_cuda` couvre `SIZE_MAX`, le plus
  grand nombre de blocs representable, les overflows de convolution/somme,
  le dernier bloc partiel et sa frontiere device; il passe sur RTX 4090
  Laptop. Compute Sanitizer `memcheck`, `racecheck`, `initcheck` et `synccheck`
  rapportent zero erreur ou hazard.
- **Reouvrir seulement si :** une taille de workspace, un offset ou un nombre
  de blocs est de nouveau calcule sans arithmetique checked, si une conversion
  vers `dim3` precede sa validation, ou si un sanitizer detecte une erreur sur
  les frontieres Volterra.

### CUDA-003 — Refuser le débordement des nombres de transitions fixed-step

- **Nature :** corrige et verifie le 2026-08-30.
- **Signature originale :** les calendriers et schedules d'exercice
  multipliaient `steps_per_day` par un nombre de jours en `uint32_t` sans
  controle host; un overflow pouvait donc lancer un kernel avec zero ou un
  nombre tronque de transitions et publier un prix fini mais faux.
- **Cloture :** `checked_fixed_step_transition_count` centralise le produit en
  `uint64_t`, refuse le depassement de `UINT32_MAX` et les fractions d'annee
  FP32 non finies. Tous les calendriers terminal, regular, stubbed, static et
  d'exercice sont valides avant lancement. Les launchers Monte Carlo recoivent
  le miroir host des produits; les sources de calendriers sample et la formule
  geometric Asian suivent le meme contrat. L'arithmetique device sous les
  bornes et le mapping Philox sont inchanges.
- **Preuve :** `simulation_schedule_validation` couvre la borne exacte et la
  valeur suivante pour chaque forme de calendrier, les schedules Regular et
  MaturityAligned, les sources sample UniformMaturity et RandomIncreasing et
  les fractions fixed/exact non finies. Le codegen complet est zero-diff,
  l'agregat de 121 cibles CUDA compile et 63/64 CTests CUDA passent; l'unique
  timeout QRH passe directement en environ 61 secondes sans erreur. Les modes
  Compute Sanitizer memcheck, racecheck, initcheck et synccheck sur les 21
  compositions du test `path_product_factorization_cuda` rapportent zero
  erreur ou hazard.
- **Reouvrir seulement si :** une duree fixed-step atteignable par un launcher
  CUDA est de nouveau multipliee sans controle host, si un rejet peut survenir
  apres un lancement, si une fraction de maturite non finie est acceptee, ou
  si l'arithmetique device sous la borne change sans preuve de reproductibilite.

### STRUCT-003 — Deriver et borner tous les artefacts depuis la source de verite

- **Nature :** corrige une seconde fois et verifie contradictoirement le
  2026-08-30.
- **Signature originale :** le zero-diff ne parcourait que les sorties
  attendues; il acceptait fichiers orphelins, retraits incomplets et mauvais
  mappings, et levait un `FileNotFoundError` brut pour une sortie absente.
- **Cloture :** l'inventaire type enumere les 832 fichiers de bindings, les 697
  recettes, les 48 bindings samples et les 24 helpers samples. Le comparateur
  confronte ensembles reels et declares dans les deux sens et emet
  `CODEGEN_MISSING`, `CODEGEN_EXTRA`, `CODEGEN_RENDERER_*` ou
  `CODEGEN_MISMATCH` sans exception implicite.
- **Preuve :** generation complete de 1 500 fichiers zero-diff; fixtures
  negatives pour absence, contenu modifie, orphelin, rename, famille retiree,
  produit pluralise et renderer incomplet; CTests `pricing_binding_codegen`,
  `pricing_capability_manifest`, `model_source_layout` et
  `catalog_generator_boundaries` passes.
- **Reouvrir seulement si :** un artefact peut exister hors inventaire, un
  retrait laisse une sortie acceptee, un mapping incorrect passe le checker ou
  une absence produit de nouveau une exception non diagnostiquee.

### STRUCT-015 — Generer les bindings et recettes fixed-income closed form

- **Nature :** corrige apres report explicite et verifie le 2026-08-30.
- **Signature originale :** 21 paires de bindings et 42 recettes fixed-income
  closed form etaient inventoriees mais toutes handwritten, sans branche de
  templates possedant leurs signatures et corps.
- **Cloture :** les branches `affine_one_factor/{cir,gaussian}`,
  `affine_two_factor`, `curve_fitted_one_factor` et
  `curve_fitted_two_factor` generent les 21 paires et 42 recettes. CIR et les
  modeles gaussiens restent separes lorsque leur API differe; les huit paires
  et seize recettes Bermudan sont explicitement `hand_written`.
- **Preuve :** zero-diff complet; 74 templates nommes inventories; les 21 TU
  closed form, les 42 executables de recettes et l'agregat fixed-income
  compilent. Les 24 CTests fixed-income passent hors sandbox sur RTX 4090
  Laptop sans benchmark de performance.
- **Reouvrir seulement si :** une composition closed form compatible redevient
  manuelle, si un nouveau modele/courbe exige de copier un corps hors branche
  semantique, ou si bindings et recettes divergent du manifeste.

### STRUCT-018 — Aligner la taxonomie canonique de `src`, `catalog` et `datasets`

- **Nature :** corrige et verifie le 2026-08-30.
- **Signature originale :** les 18 modeles equity perdaient la famille
  `markovian`/`rough` dans `catalog` et `datasets`, et les 26 produits etaient
  ranges sous asset class avec un nom de dossier pluralise distinct de `src`.
- **Cloture :** `catalog` et `datasets` reutilisent exactement les prefixes
  `model/equity/<markovian|rough>/<model>`,
  `model/fixed_income/<model>`, `curve/<curve>` et `product/<product>` de
  `src`. Recettes, YAML, JSON, URLs, CMake, tests et documentation ont ete
  migres; toute transformation concurrente est refusee par les specs.
- **Preuve :** identite physique 24/24 modeles, 26/26 produits et 2/2 courbes;
  aucune ancienne taxonomie hors fixtures historiques; configuration CMake
  fraiche, test loader, 697 recettes, zero-diff et fixtures negatives passes.
- **Reouvrir seulement si :** une couche ajoute asset class, pluralisation ou
  retire la famille par rapport a `src`, si un chemin est reconstruit hors
  `source_prefix`, ou si catalog/dataset/URL cessent d'etre identiques.

### BUILD-004 — Reduire les dependances transitives publiques des unites CUDA

- **Nature :** corrige et mesure le 2026-08-30.
- **Signature originale :** chaque archive de launcher CUDA exportait les
  loaders modele, produit et courbe en `PUBLIC`; le test Heston tirait ainsi
  quinze archives produit, son loader modele et cinq facades offline alors
  qu'il n'appelait aucun loader.
- **Cloture :** les launchers lient `runtime`, Longstaff--Schwartz et cuFFTDx
  en `PRIVATE`. Les executables resolvent directement les seules archives
  correspondant aux headers dataset qu'ils incluent via
  `ai_factory_collect_source_dependencies`; les tests CUDA ne lient plus
  `ai_factory_dataset_core`. Les recettes Bermudan declarent explicitement le
  loader produit qu'elles utilisent. La configuration refuse toute archive
  CUDA qui reexporte un target `*_dataset`.
- **Preuve :** le lien Heston passe de 39 a 17 archives et conserve ses quinze
  launchers. Le generateur Heston European lie sept archives, dont exactement
  launcher, loaders modele/produit et pipeline prix; le generateur
  Hull-White/Nelson-Siegel lie aussi son loader courbe. Apres toucher
  `european_option/dataset.cpp`, Heston est no-op tandis que son generateur
  reconstruit seulement loader, archive et executable. Configuration, builds,
  CTests Heston et American LSM passent; un generateur CIR Bermudan relie son
  loader produit apres configuration fraiche.
- **Reouvrir seulement si :** un launcher exporte un loader ou une facade
  offline, si un consommateur obtient un loader sans l'inclure explicitement,
  ou si une modification de dataset invalide de nouveau un test de kernel pur.

### NUM-008 — Allouer des domaines Philox disjoints aux recettes et variantes

- **Nature :** corrige et verifie le 2026-08-30.
- **Signature originale :** les recettes samples espacaient leurs bases de 10
  ou 1 000 et les prix stochastic reutilisaient deux bases globales; Bates
  `samples_01` ligne 10 et `samples_02` ligne 0 partageaient ainsi la meme
  cle et le meme compteur Philox.
- **Cloture :** `RngDomainSpec` version 1 alloue a chaque chemin de recette
  stochastique canonique un domaine de `2^32` cles et des flux nommes de
  `2^30` cles. Samples separe `parameters`, `schedule` et `dynamics`; pricing
  reserve `dynamics`; l'allowlist CRN est explicitement vide. Codegen et les
  seize recettes Bermudan consomment ces seules graines.
- **Preuve :** les 14 tests du manifeste prouvent couverture exhaustive,
  intervalles disjoints, Bates distinct et rejet d'une fixture collision. Le
  checker fail-closed valide les 697 recettes et chaque literal declare; le
  codegen complet de 1 500 sorties est zero-diff. Le test CUDA
  `black_scholes_samples_cuda` rejoue exactement entre 128/512 threads et deux
  decoupages de batch. Les generateurs Black-Scholes samples, Heston European
  et CIR Bermudan compilent.
- **Reouvrir seulement si :** une recette Philox obtient une graine hors du
  manifeste, si deux intervalles se recouvrent hors allowlist CRN versionnee,
  si le volume d'une recette atteint `2^30` lignes, ou si batch/geometrie entre
  de nouveau dans la derivation de cle ou de compteur.

### NUM-007 — Stabiliser les samples Quadratic rough-Heston sur le domaine publie

- **Nature :** corrige, qualifie et verifie le 2026-08-30.
- **Signature originale :** `samples_02` produisait deterministement un spot
  non fini sur les bornes core; apres equilibrage de la recurrence, seuls la
  ligne historique et un smoke 1 000 lignes avaient ete qualifies. En outre,
  la recette ajustait un noyau L2 par `H`, soit environ 77 heures extrapolees
  de preparation hote pour trois millions de lignes.
- **Cloture :** la recurrence device equilibree est conservee. Le helper sample
  construit 257 fits L2 positifs exacts sur `H in [0.01, 0.20]`, puis
  interpole lineairement nodes et weights; preparation scalaire et pricers
  gardent leurs fits exacts. Le mode generique `--preflight` execute toute la
  forme production, rejoue sous une seconde geometrie et refuse maturites ou
  sorties non finies/differentes sans publier d'artefact.
- **Preuve numerique :** les 256 milieux de cellule restent sous 0,11 %
  d'erreur L2 relative contre le noyau fractionnaire analytique; sur sept fits
  exacts hors grille, la penalite maximale vaut `1.67e-6`. Sur
  `H={0.005,0.01,0.10,0.20,0.45}`, l'erreur contracte strictement de N=2 a
  N=3 puis N=7; N=7 varie de `3.23e-5` a `1.031e-3`. Les coins core/stress,
  1 024 trajectoires chacun et N=2/3/7 sont finis.
- **Preuve production et prix :** les deux layouts de 3 000 000 lignes sont
  finis et bitwise identiques entre 256 et 128 threads. `samples_01` mesure
  33,45 ms kernel / 19,53 s total et `samples_02` 28,04 ms / 20,66 s. Les
  48/48 recettes smoke passent. Le pricer European passe replay, parite,
  comparaison grille/fit exact et raffinements N=2/3/7 ainsi que
  `dt=1/252,1/504,1/1008` dans l'incertitude Monte Carlo.
- **Ressources et surete SM89 :** random-terminal N=7 utilise 82 registres,
  32 octets de stack, zero local; la variante parameter-block utilise 91
  registres, 160 octets shared, 32 octets stack et zero local. Le pricer N=7
  utilise 91 registres, 192 octets shared, zero stack/local. Memcheck,
  racecheck, initcheck et synccheck rapportent zero erreur/hazard; quatre
  CTests CUDA cibles et cinq checks architecture/codegen passent.
- **Reouvrir seulement si :** un output QRH redevient non fini, si un layout
  3M n'est plus reproductible entre geometries, si l'erreur grille depasse ses
  bornes, si le domaine sample sort de `[0.01,0.20]` sans nouvelle grille, ou
  si un raffinement temps/facteurs ou un pricer echoue a sa qualification.

### NUM-010 — Qualifier le FP64 de finalisation des statistiques Monte Carlo

- **Nature :** usage FP64 conserve, qualifie et verifie le 2026-08-30.
- **Signature originale :** moyenne, variance centree, seuil d'annulation,
  divisions et racine de `compute_statistics` restaient en FP64 sans
  comparaison propre FP32/mixte/host pour Monte Carlo, Volterra et LSM.
- **Cloture :** `monte_carlo_statistics_precision_cuda` compare finalisation
  FP64, racine FP32 mixte, FP32 complete et host contre une reference
  `long double` sur quatre distributions core/stress/faible variance. FP64
  reste le contrat commun : une seule finalisation est executee par prix, hors
  boucle de trajectoires, et preserve la semantique de `NUM-005`.
- **Preuve numerique et cout SM89 :** l'erreur relative maximale mesuree vaut
  `4.10e-12` sur l'erreur standard FP64. La finalisation FP32 l'annule sur les
  deux cas de faible variance. Pour 1 048 576 resultats, FP64 mesure
  `0.1847 ms`, mixte `0.1199 ms`, FP32 `0.0241 ms`; la seule copie device-host
  des moments mesure `10.217 ms`, puis le calcul host `3.859 ms`. Le gain mixte
  isole est donc sans effet end-to-end demontre aux volumes reels et changerait
  l'arrondi publie.
- **Ressources et consommateurs :** le kernel isole FP64 utilise 35 registres,
  contre 31 mixte et 19 FP32, sans stack/shared/local. Heston MC utilise 66
  registres et 80 octets shared; Heston LSM finalise avec 35 registres, 128
  octets shared dynamiques, zero spill et 100 % d'occupation theorique;
  Volterra conserve son finalizer 31 registres sans spill. Les trois CTests
  consommateurs et les quatre outils Compute Sanitizer passent.
- **Reouvrir seulement si :** une famille demontre un gain end-to-end de la
  finalisation mixte sur chaque architecture cible avec budget prix/erreur
  standard, ou si le contrat de sortie cesse d'etre FP32.

### NUM-020 — Qualifier la formation FP64 des seconds moments Monte Carlo

- **Nature :** usage FP64 chaud conserve, qualifie et verifie le 2026-08-30.
- **Signature originale :** chaque payoff FP32 etait promu puis carre en FP64
  dans Monte Carlo, Volterra et LSM sans mesure separee de la multiplication ni
  comparaison des alternatives.
- **Cloture :** le test commun compare produit FP64 direct, FMA FP64, carre
  FP32 promu, carre FP32 mis a l'echelle et somme FP32 compensee contre une
  accumulation `long double`. Le produit de deux valeurs FP32 est exact en
  FP64 avant addition; ce contrat commun est conserve dans les trois callsites.
- **Preuve numerique :** produit et FMA FP64 restent sous `4.10e-12` d'erreur
  relative d'erreur standard. Sur les cas de faible variance aux echelles 100
  et 2 048, le carre FP32 produit respectivement 100 % et 15,48 % d'erreur; la
  compensation atteint 45,87 % et 15,45 %. Prix et mapping stochastic restent
  inchanges.
- **Cout et ressources SM89 :** pour 256 lignes de 32 768 payoffs, produit
  FP64 `0.1558 ms`, FMA FP64 `0.1549 ms`, FP32 promu `0.1908 ms`, FP32 mis a
  l'echelle `0.1920 ms`, FP32 compense `0.1174 ms`. Les cinq variantes
  utilisent 21--22 registres, zero stack/local; l'alternative plus rapide ne
  respecte pas le budget numerique. Les kernels reels MC, Volterra et LSM
  passent leurs tests; `moment_partials` LSM utilise 23 registres, 128 octets
  shared dynamiques, zero spill et 100 % d'occupation theorique.
- **Reouvrir seulement si :** une famille bornee prouve sur son domaine complet
  qu'une formation mixte respecte prix et erreur standard et gagne end-to-end
  sur toutes ses architectures cibles.

### NUM-011 — Qualifier l'accumulation FP64 des moyennes arithmetiques de chemin

- **Nature :** FP64 chaud elimine, qualifie et verifie le 2026-08-30.
- **Signature originale :** les deux facades Asian additionnaient chaque spot
  FP32 et divisaient en FP64 dans les 18 compositions Markov, N-factor et
  Volterra, sans budget ni cout marginal mesure.
- **Cloture :** `CompensatedFloatSum` implemente une somme de Kahan FP32 compacte
  et devient l'unique accumulateur des deux facades. Division et valeur publiee
  restent en FP32. `asian_mean_precision_cuda` compare FP64, FP32 simple, Kahan
  FP32 et chunks FP32 contre une reference `long double` sur 17, 253, 1 765 et
  4 097 observations, faible variance, forte dispersion et echelles extremes.
- **Preuve numerique et prix :** l'erreur relative maximale de coordonnee Kahan
  vaut `5.29e-8` et les quatre moyennes arithmetiques publiees sont identiques a
  la reference arrondie FP32. Le payoff vanille est 1-Lipschitz, donc cette
  borne, multipliee par le discount borne, se propage au payoff et au prix. La
  somme FP32 simple atteint `3.39e-6` sur la valeur publiee.
- **Cout et ressources SM89 :** 8 192 trajectoires x 1 765 observations mesurent
  FP64 `0.4169 ms`, FP32 simple `0.3108 ms`, Kahan FP32 `0.2817 ms`, chunks
  `0.4188 ms`. Heston reste a 70 registres/80 octets shared; QRH N=7 passe de
  108 a 93 registres; le kernel de payoff Volterra passe de 71 a 69 registres.
  Aucun stack/local/spill n'apparait. Markov, QRH N-factor et Volterra passent,
  ainsi que memcheck sur le microbenchmark et QRH.
- **Reouvrir seulement si :** le domaine depasse 4 097 observations ou les
  echelles balayees sans nouveau budget, si un payoff non 1-Lipschitz reutilise
  la moyenne, ou si une architecture cible regresse en ressources/end-to-end.

### NUM-013 — Qualifier la somme FP64 du range accrual analytique Black-Scholes

- **Nature :** FP64 chaud elimine, qualifie et verifie le 2026-08-30.
- **Signature originale :** le closed form additionnait jusqu'a 1 764
  probabilites FP32 en FP64 sans budget de prix ni comparaison de cout propre.
- **Cloture :** `RangeAccrualClosedFormPricingPolicy` reutilise
  `CompensatedFloatSum`. `range_accrual_sum_precision_cuda` compare FP64, FP32
  simple, Kahan FP32, chunks FP32 et reference analytique host sur quatre coins
  core/stress : 12, 252 et 1 764 observations, barrieres etroites/larges,
  volatilites `0.01`--`1.0`, taux `-0.03`--`0.12` et coupons jusqu'a `0.25`.
- **Preuve numerique :** les quatre strategies device publient les memes prix
  FP32 sur le sweep; Kahan reste sous `9.63e-8` d'erreur relative contre la
  reference host. Le test Black--Scholes existant conserve ses controles
  analytiques sur trois spots.
- **Cout et ressources SM89 :** 1 024 prix x 1 764 observations mesurent FP64
  `0.4605 ms`, FP32 simple `0.4238 ms`, Kahan FP32 `0.4282 ms`, chunks
  `0.4905 ms`, contre `39.07 ms` host. Le kernel de production passe de 34 a
  35/36 registres selon sa specialisation, sans stack/shared/local/spill.
  Les deux CTests et memcheck passent.
- **Reouvrir seulement si :** plus de 1 764 observations, probabilites hors
  `[0,1]`, nouveaux extremes de barrieres/taux/volatilite/coupon, ou regression
  ressources/end-to-end sur une architecture cible.

### NUM-014 — Qualifier le noyau de resolvante fractionnaire FP64 sur device

- **Nature :** FP64 chaud elimine, qualifie et verifie le 2026-08-30.
- **Signature originale :** Rough Stein--Stein evaluait noyau, fonction de
  Mittag--Leffler et neuf points de chaque poids lointain en FP64, jusqu'a 96
  termes ou points et avec fonctions speciales, sans comparaison FP32/mixte ni
  cout device propre.
- **Cloture :** le chemin device est FP32 compense. La serie stable reste sous
  `x <= 2`; la representation positive par densite de Laplace prend le relais
  avant la zone de cancellation FP32. Series, quadratures et Simpson utilisent
  `CompensatedFloatSum`. Une variante FP32 gardant l'ancien crossover a ete
  rejetee : jusqu'a 11,94 % d'erreur noyau, 22,23 % sur les integrales et
  70,16 % sur les poids.
- **Preuve numerique et prix :** sur 100 coins couvrant `H=0.01--0.45`, mean
  reversion `0--8`, temps `1/504--7` ans et lags `2--1008`, les erreurs
  relatives maximales contre quadrature `long double` valent `8.23e-5` pour le
  noyau et `7.14e-5` pour les poids. Sur 1 000 sorties sample a parametres,
  calendriers et seeds identiques, l'ecart terminal relatif maximal vaut
  `1.41e-6`, le p99 `7.85e-7` et le decalage de moyenne `1.17e-8` ecart-type.
  Le payoff European est 1-Lipschitz, donc son ecart de prix couple est borne
  par l'ecart terminal absolu maximal `2.39e-6` avant discount.
- **Cout et ressources SM89 :** le microbenchmark apparie de 1 024 lignes x 16
  poids passe de `46.44 ms` FP64 a `5.64 ms` FP32. Le sampler `samples_02`
  passe de `3.2472 s` a `0.4145 s` kernel et de `9.2128 s` a `1.6393 s` wall.
  Son kernel terminal passe de 148 a 128 registres/thread et de 96 a 88 octets
  locaux/thread; l'occupation theorique reste 16,7 %, donc le kernel FFT reste
  contraint par d'autres ressources. Les 21 bibliotheques produit et les deux
  generateurs European compilent; cinq CTests cibles et memcheck passent.
- **Reouvrir seulement si :** le domaine `H`/mean reversion/temps/lag est
  etendu, le crossover ou la quadrature change, une sortie depasse le budget
  `5e-4`, ou une architecture cible invalide gain end-to-end ou ressources.

### NUM-015 — Qualifier les integrales de puissance FP64 de la resolvante

- **Nature :** FP64 chaud elimine et frequence originale rectifiee le
  2026-08-30.
- **Signature originale :** `power_integral` accumulait 96 contributions et
  appelait le noyau en FP64; le constat estimait `step_count + 2` quadratures
  par resultat prepare sans preuve propre de precision ou de cout.
- **Cloture :** les integrandes, fonctions speciales et sommes passent en FP32
  compense avec le noyau de `NUM-014`. L'inventaire du chemin compile montre
  surtout que Rough Stein--Stein declare `kUsesVolterraVariance = false` : la
  branche `volterra_variance` est eliminee a la compilation et seules les deux
  integrales de `prepare` sont executees, pas une integrale par pas.
- **Preuve numerique et cout SM89 :** le meme sweep compare les puissances 1 et
  2 a une quadrature `long double`; l'erreur relative maximale vaut
  `4.03e-5`. Le microbenchmark de 1 024 integrales passe de `8.379 ms` FP64 a
  `0.854 ms` FP32. La comparaison des 1 000 trajectoires de `NUM-014` couvre la
  propagation des deux loadings singuliers; les tests policy, samples,
  workspace et sanitizer passent.
- **Reouvrir seulement si :** une path policy active
  `kUsesVolterraVariance`, si la variance devient atteignable par pas, si les
  loadings sortent du domaine qualifie ou si le budget `5e-4` n'est plus tenu.

### NUM-016 — Qualifier les produits des statistiques de regression LSM en FP64

- **Nature :** usage FP64 chaud conserve, qualifie et verifie le 2026-08-30.
- **Signature originale :** cible actualisee, produits feature-feature du Gram
  et feature-cible du second membre etaient formes en FP64 a chaque candidat et
  date backward, sans comparaison propre des produits FP32/mixtes ni cout.
- **Cloture :** les operandes FP32 restent promus avant multiplication : leur
  produit est exact avant l'accumulation FP64 commune. Le test
  `longstaff_schwartz_precision_cuda` compare ces termes a des produits FP32
  ensuite promus sur 32 768 observations Laguerre deux facteurs, core et
  presque colineaires/stress, contre une reference `long double`.
- **Preuve numerique et cout SM89 :** l'erreur relative maximale des
  statistiques FP64 vaut `3.92e-13`, contre `2.22e-9` pour les produits FP32.
  Pour 16 384 lignes x 32 observations et 27 statistiques, le chemin FP64
  mesure `0.0853 ms`, contre `0.1258 ms` pour FP32 puis promotion :
  l'alternative est a la fois moins precise et plus lente avec les reductions
  FP64 contractuelles. Les kernels isoles utilisent 80 et 72 registres, sans
  stack/local; les kernels reels utilisent 88 registres Heston et 118 G2,
  sans local, aux occupations theoriques respectives 33,3 %.
- **Preuve consommateurs :** tests du regresseur, Heston American et Bermudan
  OU/G2 payer/receiver passent, ainsi que memcheck du nouveau sweep.
- **Reouvrir seulement si :** accumulation/reduction cesse d'etre FP64, la
  base ou sa taille change, un domaine de features/cashflows sort du sweep, ou
  une architecture cible demontre une alternative plus precise et plus rapide
  end-to-end.

### NUM-018 — Qualifier la prediction et la decision d'exercice LSM en FP64

- **Nature :** usage FP64 chaud conserve; variante selective mesuree et rejetee
  le 2026-08-30.
- **Signature originale :** les FMA de prediction et la comparaison
  exercice/continuation restaient en FP64 pour chaque candidat; `NUM-003`
  prouvait une frontiere sensible mais ni cout ni strategie selective.
- **Cloture :** coefficients, prediction et comparaison restent FP64. Le sweep
  de `2^20` cas contient 16 384 marges sous `1e-7` et compare FP32 a une
  variante selective qui calcule une borne conservative d'arrondi puis
  reevalue FP64 pres de la frontiere.
- **Preuve decisions et prix :** FP32 diverge sur 7 040 decisions et deplace la
  moyenne des cashflows de `0.59406694` a `0.59094747` (`-0.00311947`, environ
  `-0.525 %`). La variante selective declenche 16 384 fallbacks, ne diverge
  jamais et retrouve exactement la moyenne FP64.
- **Cout et ressources SM89 :** sur `2^20` predictions, FP64 mesure
  `0.0581 ms`, FP32 `0.0239 ms`, mais la variante selective `0.2324 ms`. Les
  kernels isoles utilisent 33, 30 et 40 registres respectivement, sans
  stack/local. En production, `update_cashflows` utilise 40 registres Heston
  et 64 G2, sans local, avec 100 % et 66,7 % d'occupation theorique. Heston
  call/put et OU/G2 payer/receiver passent avec replay bitwise.
- **Reouvrir seulement si :** une nouvelle base ou plage de coefficients/marges
  est introduite, ou si une selection bornee conserve zero divergence et prix
  tout en gagnant end-to-end et en ressources sur American et Bermudan de
  chaque architecture cible.

## Concepts, structure et naming — remediation du 2026-08-30

### STRUCT-020 — Supprimer ou integrer les headers runtime sans consommateur

- **Nature :** surface morte supprimee et verifiee le 2026-08-30.
- **Signature originale :** `equity/observables.cuh` et
  `simulation/barrier_handlers.cuh` exposaient quatre types sans include ni
  consommateur, en parallele des policies barriere actives.
- **Cloture :** les deux headers orphelins sont supprimes; la responsabilite
  barriere active reste uniquement dans `equity/barrier_pricing_policy.cuh`.
- **Preuve :** recherche sans ancien include ou symbole; le checker de layout,
  les builds Markov/N-facteurs/Volterra et les tests GPU representatifs passent.
- **Reouvrir seulement si :** un header runtime sans consommateur reapparait ou
  si deux implementations independantes revendiquent la meme responsabilite
  barriere sans contrat de composition.

### NAME-007 — Nommer les diagnostics du test d'apres ses modeles et son contrat

- **Nature :** reouverture corrigee et verifiee le 2026-08-30.
- **Signature originale :** cinq diagnostics employaient encore le repere
  historique « New equity dynamics test ».
- **Cloture :** chaque diagnostic nomme desormais le contrat dynamics
  Merton/Kou/CEV/Schobel-Zhu et l'operation CUDA concernee.
- **Preuve :** recherche sans `new`, `additional` ou `remaining` dans les noms
  et diagnostics de tests; le CTest dynamics correspondant passe sur GPU.
- **Reouvrir seulement si :** un nom de fichier, target, CTest ou diagnostic
  decrit l'anciennete plutot que le modele et le contrat testes.

### NAME-013 — Nommer et placer explicitement la policy Phoenix partagee

- **Nature :** corrige et verifie le 2026-08-30.
- **Signature originale :** `pricing_policy_core.cuh`, possede par Phoenix sans
  memoire, cachait une policy commune a deux produits et l'invariant de memoire
  coupon selectionne a la compilation.
- **Cloture :** le proprietaire neutre
  `src/product/phoenix_coupon_memory_path_policy.cuh` expose des types nommes
  `PhoenixCouponMemory*` et le parametre explicite `RememberMissedCoupons`; les
  deux facades produit choisissent respectivement `false` et `true`.
- **Preuve :** ancien chemin et anciens symboles absents; commentaire d'en-tete
  explicite; checker de layout, codegen zero-diff et test des 21 produits
  factorises passent.
- **Reouvrir seulement si :** le partage Phoenix redevient possede par un seul
  produit concret, si le nom masque l'invariant de memoire coupon ou si les
  deux facades cessent de composer la meme implementation.

### NAME-006 — Encoder les unites dans les coordonnees temporelles publiques et device

- **Nature :** reouverture corrigee et verifiee le 2026-08-30.
- **Signature originale :** les deux courbes exposaient encore les coordonnees
  annuelles de `forward_rate` sous les noms ambigus `start` et `end`, hors du
  checker ayant motive la premiere cloture.
- **Cloture :** declarations, definitions, commentaires et references de
  courbes utilisent `start_years` et `end_years`. Le checker couvre maintenant
  `start` et `end` en plus des anciens noms, avec trois fixtures negatives, et
  inspecte aussi les references README locales.
- **Preuve :** inventaire sans coordonnee flottante publique/device ambigue;
  `model_source_layout` passe; Hull--White et G2++ compilent et leurs CTests
  passent sur GPU. Aucune cle de serialisation ou valeur dataset n'est modifiee.
- **Reouvrir seulement si :** une coordonnee temporelle flottante publique,
  device ou documentee perd de nouveau son unite, ou si le checker cesse de
  rejeter l'un des noms ambigus couverts.

### STRUCT-010 — Supprimer les inventaires README locaux recopies et non maintenus

- **Nature :** reouverture corrigee et verifiee le 2026-08-30.
- **Signature originale :** 22 references locales n'avaient aucun index; deux
  pages de famille niaient la taxonomie physique active et les courbes
  recopiaient des signatures deja obsoletes.
- **Cloture :** `docs/model-and-curve-reference-index.md` indexe exhaustivement
  les 22 references mathematiques conservees pres du source et explicite
  qu'elles ne possedent ni arbre, ni signatures, ni matrice de capacites. Les
  taxonomies markovian/rough sont synchronisees; les inventaires de fichiers et
  blocs de signatures des courbes sont remplaces par des liens vers les headers
  proprietaires.
- **Preuve :** le checker impose la bijection entre index et README physiques,
  rejette lien stale/duplique, inventaire Files/signatures, ancienne taxonomie
  et coordonnee temporelle sans unite. Les trois CTests codegen/layout/catalogue
  passent.
- **Reouvrir seulement si :** une reference locale redevient non indexee,
  recopie une API/arborescence, publie une taxonomie fausse ou si le controle
  bidirectionnel des liens est retire.

### STRUCT-021 — Synchroniser les contrats CUDA normatifs avec l'architecture active

- **Nature :** corrige et verifie le 2026-08-30.
- **Signature originale :** le guide de composition annonçait N-facteurs et
  closed form comme futurs, decrivait deux corps produit et le contrat dynamics
  recommandait encore le FP64 des handlers remplaces par somme compensee FP32.
- **Cloture :** le guide couvre les douze engines actifs du manifeste et les
  compositions markovienne, Volterra FFT, N-facteurs, closed form, LSM et
  sampling. Les alias publics sont documentes comme identiques a l'unique
  `ProductPathPolicy`. Le contrat dynamics reporte les domaines mesures de
  `CompensatedFloatSum` et borne les usages FP64 encore contractuels.
- **Preuve :** le checker impose la presence des douze identifiants engine et
  rejette les deux formulations architecturales obsoletes; les symboles et
  branches cites correspondent aux templates requis et au codegen zero-diff;
  les CTests architecture passent.
- **Reouvrir seulement si :** un engine actif manque au guide, une branche
  supprimee y reste presentee comme active, un second corps produit est decrit
  ou une recommandation de precision contredit les decisions numeriques
  mesurees du registre.

### NUM-019 — Borner les surcharges device FP64 des courbes Nelson--Siegel et Svensson

- **Nature :** surface device FP64 supprimee et contrat host/device separe le
  2026-08-30.
- **Signature originale :** les deux headers d'instantaneous forward exposaient
  un template `__host__ __device__` et `forward_exponential(double)`, alors que
  seul le test les instanciait sur GPU et qu'aucune composition de production
  n'appelait FP64.
- **Cloture :** chaque courbe expose maintenant une surcharge FP32 explicite
  `__host__ __device__` et une surcharge FP64 explicite host-only. Le template
  et les helpers exponentiels surcharges disparaissent : un appel CUDA ne peut
  plus instancier `double` par inference. Les generateurs conservent le FP64
  pour leurs scans d'extrema et de domaine.
- **Preuve :** `numerical_robustness_cuda` compare device et host en FP32, puis
  les deux surcharges host FP64 a des expressions `long double` sous `1e-15`.
  Les bibliotheques de generation Nelson--Siegel/Svensson, les compositions
  fitted Svensson Hull--White/G2++ et leurs trois CTests representatifs
  compilent et passent. La recherche statique ne trouve plus aucun overload
  `double` marque `__device__` dans ces headers.
- **Reouvrir seulement si :** un calcul de courbe FP64 redevient atteignable
  depuis device, si une composition CUDA exige reellement FP64, ou si le
  contrat host-only n'est plus suffisant aux generateurs de domaine.

### FACTOR-001 — Eliminer ou justifier les deux facades produits equity

- **Nature :** duplication semantique eliminee et cout qualifie le 2026-08-30.
- **Signature originale :** 13 headers produit, les deux policies barriere et
  la policy Phoenix maintenaient un corps markovien direct en parallele de la
  `PathProductPolicy` reutilisee par N-facteurs et Volterra; les 21 produits
  ordinaires pouvaient diverger entre engines.
- **Cloture :** les 16 corps `*PricingPolicy` directs disparaissent. Chaque
  surface publique est un alias de
  `equity::PathProductMonteCarloPricingPolicy<Schedule, PathPolicy>` et le
  produit ne possede plus qu'un calendrier, une preparation, un handler et une
  finalisation. Le contrat interdit une nouvelle facade directe sans exception
  bornee, parite pathwise et gain end-to-end mesure.
- **Preuve fonctionnelle :** le temoin compile avant la bascule compare les
  anciennes facades et la composition canonique sur 21 produits Heston x 4 096
  trajectoires avec memes parametres, schedules, seeds et Philox; prix et
  erreur standard sont bitwise identiques et la taille `PreparedRow` maximale
  reste 84 octets. Apres bascule, `path_product_factorization_cuda` impose par
  `std::same_as` l'identite des 21 surfaces publiques avec leur composition et
  rejoue les 86 016 trajectoires. Les tests Heston terminal/path, Volterra FFT
  et QRH N=7 passent sur GPU.
- **Preuve matrice et build :** le codegen reproduit sans diff 12 engines,
  24 modeles, 416 compositions et 1 500 sorties. `all_models` compile les 872
  etapes de la matrice puis un second build est sans travail. Les archives
  Heston representative augmentent de 2,08 % a 3,25 %, effet accepte du nom de
  type generique; aucun second corps ni objet par engine n'est introduit.
- **Ressources SM89 :** European passe de 66 a 68 registres, Asian reste a 70,
  Athena de 75 a 78, range accrual de 72 a 75, down-and-out reste a 70 et
  Phoenix memory de 77 a 80. Les six restent a zero stack/local/spill, shared
  80 ou 96 octets et meme nombre de blocs residents; aucune classe
  d'occupation ne regresse. La faible hausse est acceptee contre la suppression
  de 16 implementations divergentes et reste couverte par la baseline globale.
- **Reouvrir seulement si :** une surface publique cesse d'etre identique a la
  composition canonique, un calendrier/payoff est duplique par engine, une
  architecture cible franchit un seuil d'occupation ou de spill, ou la
  campagne end-to-end montre une regression au-dela de son budget.

### BUILD-005 — Déléguer les targets CMake par domaine

- **Nature :** structure CMake corrigée et vérifiée le 2026-08-30; sévérité
  originale moyenne, priorité moyenne.
- **Signature originale :** le `CMakeLists.txt` racine enregistrait directement
  les targets runtime, catalogue, tests, performance et validation dans plus de
  1 800 lignes, sans propriétaire local par domaine.
- **Clôture :** la racine est réduite à 127 lignes de configuration globale,
  options et orchestration. Les enregistrements appartiennent désormais à
  `cmake/AIFactoryRuntime.cmake`, `AIFactoryCatalog.cmake`,
  `AIFactoryPerformance.cmake`, `AIFactoryTests.cmake` et
  `AIFactoryValidation.cmake`; les helpers partagés restent dans
  `AIFactoryTargets.cmake`. Le README rend ce découpage navigable.
- **Preuve :** l'inventaire des 573 targets exposés avant/après extraction est
  strictement identique. La configuration SM75 sans mathDx compile les 125
  étapes host puis les 39 étapes performance et passe ses 12 tests; la
  configuration fatbin SM75/86/89 avec mathDx compile un pricer rough
  représentatif. Le build principal compile host, CUDA et performance; le
  second build est sans travail en 0,12 s. Les cinq contrôles CTest
  codegen/layout/catalogue/QRH passent, le timeout QRH étant porté à 120 s pour
  sa durée observée de 36,54 s.
- **Propagation :** une régénération effective des sorties codegen a provoqué
  229 recompilations/relinks ciblés; aucune cible ni dépendance de domaine n'a
  disparu lors du déplacement. Le dry-run fondé sur le seul timestamp a été
  conservativement jugé non probant et n'est pas présenté comme preuve.
- **Reouvrir seulement si :** la racine réenregistre un target de domaine, si
  une configuration avec/sans mathDx diverge, si l'inventaire public change
  sans décision explicite, ou si une mutation réelle ne propage plus vers ses
  consommateurs attendus.

### STRUCT-019 — Donner au gate Performance un propriétaire dans le périmètre principal

- **Nature :** corrigé, qualifié et vérifié le 2026-09-03; sévérité originale
  moyenne, priorité haute, confiance prouvée.
- **Signature originale :** benchmarks, fixtures, manifeste, baseline, runners
  et checker de décision vivaient exclusivement sous `validation/**`; aucun
  propriétaire principal n'existait sous `tests/performance` et
  `tools/performance`, et le protocole complémentaire ne satisfaisait pas les
  quatre sous-audits Performance version 7.
- **Clôture :** `tests/performance` possède les benchmarks, fixtures, baseline
  et preuves par architecture; `tools/performance` possède exécution, checker,
  rebaseline et profilage; `docs/performance-regression-protocol.md` porte le
  contrat durable et `cmake/AIFactoryPerformance.cmake` le graphe de targets.
  Le protocole couvre 41 mesures, quatre frontières de temps et quatre rapports,
  interdit best-of-N et recomposition par clé, conserve chaque campagne brute,
  applique les budgets numériques, ressources compilées, VRAM et binaires, et
  distingue 5 % de bruit kernel de 10 % pour l'enveloppe host sans relâcher le
  seuil de régression médiane/p95 à 5 %.
- **Preuve :** trois campagnes SM89 sur trois sont admissibles sous
  `build-dev/performance_candidate_sm89_v3.ndjson.campaigns/20260903T184737.789921Z`;
  le checker passe 41 mesures avec zéro inconclusive bloquant et deux messages
  attachés à l'unique timing informatif. La baseline qualifiée porte le SHA-256
  `94b7370a2bf1ebed350a04a1037c42115d7a6946127399d7c4358f329786429f`;
  son prédécesseur conservé porte
  `26c4af09f90ffeb812d7ea906619981476d94f2c4e6b18dd07e5470c54381604`
  et le diff exhaustif d'initialisation
  `205cee367c9b0b152b37a02b1477252949fa9cf9753de41d429524cbd2323717`.
  Les quatre rapports contiennent 18/8/4/11 mesures. Les huit artefacts Nsight
  Compute 2026.2.1 sous `tests/performance/profiles/sm89` lient le symbole,
  l'exécutable, le candidat, la baseline, l'environnement et le CSV brut pour
  CIR, sample rough N-factor, LSM Heston et rough SABR FFT. Les 27 tests
  fail-closed, le CTest `performance_baseline_checker`, le build performance
  sans travail et `git diff --check` passent.
- **Portabilité :** cette baseline et ces profils qualifient uniquement le
  profil RTX 4090 Laptop SM89 et son toolchain déclarés. Chaque autre GPU ou
  toolchain doit publier son propre manifeste natif et ses propres profils;
  aucune géométrie SM89 n'est une valeur universelle.
- **Réouvrir seulement si :** un composant décisionnel principal retourne sous
  `validation/**`, une clé ou ressource peut échapper au manifeste, une
  campagne est sélectionnée ou recomposée opportunément, une rebaseline peut
  s'auto-valider sans prédécesseur/diff/raison/approbation, un scope de temps
  cesse de bloquer selon son budget, un profil ne correspond plus au binaire
  mesuré, ou une architecture réutilise les seuils observés d'un autre GPU.

## Remédiation du passage indépendant version 8 — 2026-09-04

### STRUCT-014 — Ne pas publier un lien interne vers un arbre ignoré

- **Nature :** réouverture corrigée; sévérité originale moyenne, priorité
  haute, confiance prouvée.
- **Signature originale :** `docs/README.md` publiait un lien local vers
  `AI_factory_website/README.md`, présent dans le checkout mais ignoré par Git;
  le checker validait seulement son existence physique.
- **Clôture :** les index suivis pointent vers la frontière suivie
  `docs/proposed-protected-dataset-download-design.md` et présentent le site
  comme un projet séparé. Le checker inventorie les documents maintenus depuis
  Git, rejette les cibles absentes, ignorées ou hors dépôt et possède une
  fixture négative visant l'ancien chemin ignoré.
- **Preuve :** `model_source_layout` passe sur 832 unités modèle-produit, 199
  fichiers d'infrastructure et 81 templates; tous les liens locaux maintenus
  sont contrôlés dans le snapshot versionné.
- **Réouvrir seulement si :** un document suivi dépend de nouveau d'une cible
  locale absente ou ignorée, ou si le contrôle Git des liens est retiré.

### DOC-001 — Aligner la carte d'ownership CMake sur les modules réels

- **Nature :** corrigé; sévérité originale faible, priorité haute, confiance
  prouvée.
- **Signature originale :** `cmake/README.md` inversait les responsabilités de
  `AIFactoryRuntime.cmake` et `AIFactoryTargets.cmake`.
- **Clôture :** la carte décrit les propriétaires réels. Le root mesure
  automatiquement les targets créés avant/après chaque include, leur affecte
  `AI_FACTORY_OWNER_MODULE` et refuse toute target sans propriétaire. Les
  targets CTest du root sont distingués des six modules de domaine; aucune
  seconde liste de targets n'est maintenue dans le README.
- **Preuve :** les configurations fraîches principale et sans mathDx passent;
  le checker compare dynamiquement la carte à tous les modules
  `cmake/AIFactory*.cmake` présents.
- **Réouvrir seulement si :** une target configurée perd son owner, un module
  n'est plus documenté, ou la carte contredit la propriété configurée.

### POLICY-003 — Faire vérifier les paramètres réellement lus par `prepare_product`

- **Nature :** réouverture corrigée; sévérité originale moyenne, priorité
  haute, confiance prouvée.
- **Signature originale :** le concept produit acceptait un modèle sans
  `risk_free_rate`, puis l'instanciation profonde de `prepare_product`
  échouait; range accrual lisait aussi `spot` sans l'exprimer.
- **Clôture :** les concepts minimaux
  `RiskFreeRateModelParameters` et
  `SpotAndRiskFreeRateModelParameters` sont appliqués aux policies avant la
  composition. Tous les produits concernés exigent le taux; range accrual
  exige en plus le spot.
- **Preuve :** le test CUDA contient les probes négatifs taux/spot, conserve
  les identités de factorisation et passe dans la suite principale 78/78.
- **Réouvrir seulement si :** un callback lit un membre modèle non exprimé par
  son concept, ou si une composition insuffisante atteint de nouveau un corps
  device avant d'être rejetée.

### STRUCT-011 — Rendre le manifeste canonique, total et résolvable jusqu'aux symboles

- **Nature :** réouverture corrigée; sévérité originale moyenne, priorité
  haute, confiance prouvée.
- **Signature originale :** alias modèles, types courbes, listes CMake et
  états de capacité vivaient dans des tables concurrentes; plusieurs concepts,
  launchers et runners publiés n'étaient pas des symboles réels.
- **Clôture :** les specs typées portent alias, displays et types. Chaque
  référence d'API est un couple chemin/symbole vérifiable et chaque cellule de
  pricing reçoit un état distinct `available`, `deferred`, `unsupported`,
  `ambiguous` ou `unclassified`. CMake et les renderers consomment la projection
  unique `CapabilityManifest.cmake`, y compris la nature et la condition de
  chaque recette.
- **Preuve :** les fixtures ajout modèle FI/courbe/produit atteignent les
  outputs et targets sans table auxiliaire; les symboles sont résolus dans les
  fichiers déclarés; 50 tests manifeste/performance passent et la génération
  des 1 500 sorties est zéro-diff.
- **Réouvrir seulement si :** une identité ou condition est recopiée hors des
  specs, une référence ne résout plus son chemin/symbole, ou un état absent est
  confondu avec unsupported.

### STRUCT-017 — Sortir les fonctions C++ complètes des chaînes du renderer

- **Nature :** réouverture corrigée; sévérité originale moyenne, priorité
  moyenne, confiance prouvée.
- **Signature originale :** le renderer Python possédait des fonctions et
  lambdas C++ complètes invisibles dans l'arbre de templates.
- **Clôture :** factories paramètres, sérialisation JSON, lancement sample,
  préparations N-facteurs/QRH et loader produit sided résident dans sept
  templates fragments nommés. Python ne fournit plus que substitutions et
  sélection de fragments.
- **Preuve :** le checker analyse l'AST des chaînes Python, rejette une fixture
  contenant une définition C++ complète et classe 81 templates nommés; les
  1 500 sorties restent zéro-diff.
- **Réouvrir seulement si :** un renderer reprend la propriété d'une fonction
  ou lambda C++ complète, ou si le contrôle redevient dépendant d'une bannière
  textuelle particulière.

### BUILD-006 — Conserver les générateurs de paramètres host-only sans mathDx

- **Nature :** corrigé; sévérité originale moyenne, priorité haute, confiance
  prouvée.
- **Signature originale :** une regex sur le chemin modèle supprimait quatre
  générateurs de paramètres host-only lorsque mathDx était absent.
- **Clôture :** CMake enregistre les recettes depuis leur kind, engine et
  condition sémantiques projetés par le manifeste. Seuls les générateurs
  pricing/sample réellement dépendants de mathDx sont omis.
- **Preuve :** un build frais SM89 sans mathDx expose 53 dépendances de
  `parameter_generators` et construit les quatre targets
  `generate_rough_bergomi_01`, `generate_rough_sabr_01`,
  `generate_log_modulated_rough_bergomi_01` et
  `generate_rough_stein_stein_01`; les tests de manifeste interdisent leur
  intersection avec la liste conditionnelle mathDx.
- **Réouvrir seulement si :** un générateur host de paramètres dépend de la
  présence de mathDx, ou si CMake réinfère une condition depuis un chemin ou le
  texte d'un source.

### BUILD-007 — Faire correspondre le preset de build `tests` à son preset CTest

- **Nature :** corrigé; sévérité originale moyenne, priorité haute, confiance
  prouvée.
- **Signature originale :** le build preset construisait 78 tests principaux,
  mais le test preset sans filtre sélectionnait aussi 254 validations
  indépendantes.
- **Clôture :** tout test non-validation reçoit le label exact `main` après
  enregistrement; le preset `tests` sélectionne `^main$` et le nouveau preset
  `validation` sélectionne `^validation$`. Les README publient cette
  séparation.
- **Preuve :** l'inventaire CTest configuré contient exactement 78 tests
  `main` et 254 tests `validation`, sans recouvrement; `ctest --preset tests`
  passe 78/78 sans exécuter la validation indépendante.
- **Réouvrir seulement si :** le preset standard sélectionne une validation,
  omet un test de son agrégat, ou si les deux labels cessent d'être disjoints.

### PERF-015 — Rendre toutes les géométries Volterra retunables par profil

- **Nature :** réouverture corrigée et qualifiée; sévérité originale moyenne,
  priorité haute, confiance prouvée pour SM89.
- **Signature originale :** threads path/finalizer et tables longueur/EPT/FFT
  par bloc étaient dispersés dans pricing, sampling et workspace; sept lignes
  étaient dupliquées et tout changement exigeait d'éditer les moteurs.
- **Clôture :** `hybrid_fft_tuning.cuh` est l'unique propriétaire des choix
  compile-time. Le descripteur garde des champs pricing et sampling distincts,
  même lorsqu'ils coïncident; path et finalizer sont deux variables CMake
  séparées. Les moteurs ne contiennent plus de table ni de constante de
  géométrie dupliquée. Les choix SM89 ont été transférés sans changement : il
  ne s'agit pas d'une revendication universelle ni d'une retune opportuniste.
- **Preuve :** pricing, sampling et workspace compilent contre le même profil;
  la campagne officielle conserve 41 mesures, dont les horizons Volterra du
  manifeste, et le checker passe sans régression bloquante. La baseline SM89
  porte le SHA-256
  `de48e3a11fa4ccd50d59d151005e7b30fed2070e022346c4a36364d3a964c327`.
- **Portabilité :** toute nouvelle architecture ou toute modification d'un
  choix EPT/FFT par bloc doit comparer les alternatives sur ses longueurs
  actives et publier son propre profil, sa baseline et ses ressources; elle ne
  peut pas réutiliser cette qualification SM89.
- **Réouvrir seulement si :** un knob revient dans un moteur, pricing et
  sampling sont artificiellement liés, un profil change sans mesure native,
  ou une autre architecture hérite silencieusement des seuils SM89.

### PERF-010 — Budgéter les ressources de chaque phase Volterra réellement lancée

- **Nature :** réouverture corrigée, profilée et rebaselinée; sévérité
  originale moyenne, priorité haute, confiance prouvée pour SM89.
- **Signature originale :** les workloads pricing Volterra ne diagnostiquaient
  que convolution ou evaluator direct; préparation, path evaluation et
  finalisation échappaient aux budgets et le checker acceptait une liste
  seulement non vide.
- **Clôture :** chaque diagnostic porte une phase dans sa clé de déduplication.
  Le manifeste impose quatre phases hybrides
  (`row_preparation`, `convolution`, `path_evaluation`, `finalization`) et trois
  phases directes; baseline et candidat doivent contenir exactement cet
  ensemble sans doublon. Une fixture négative retire une phase.
- **Preuve campagne :** trois campagnes complètes recevables ont capturé 41
  mesures sous
  `build-dev/performance_candidate_sm89_v3.ndjson.campaigns/20260904T071757.521796Z`;
  le checker passe avec zéro inconclusive bloquante et deux informations sur
  le microbenchmark closed-form. Le prédécesseur SHA-256
  `94b7370a2bf1ebed350a04a1037c42115d7a6946127399d7c4358f329786429f`
  et le diff exhaustif SHA-256
  `f06eb7691158e3865d5ce8d276900540e5cf2e01748966b61906c311068729ab`
  sont conservés.
- **Preuve ressources :** rough-SABR budgete préparation 40 registres,
  convolution 48, evaluator 64 avec 32 octets de stack et finalizer 31. Le
  profil Nsight Compute 2026.2.1 cible explicitement la phase
  `path_evaluation`; son CSV porte le SHA-256
  `3c5287140e69bbe4a2a8031527fc7fa9177b83db4982b0bcfa1b17e054f9dbdd`
  et sa provenance le SHA-256
  `79c185da5a085871970878121e2eeef9fa8765fd6611653c976339a47cfff2fb`.
  Le mode `resource_only` conserve les pré/postflights stricts mais ne sert à
  aucune décision de timing; celles-ci restent issues des campagnes
  thermiquement stabilisées.
- **Réouvrir seulement si :** une phase active disparaît du contrat, deux
  phases se dédupliquent, une ressource ou un symbole échappe aux budgets, le
  profil ne vise plus le binaire candidat, ou une baseline est remplacée sans
  prédécesseur et diff exhaustif.

### PERF-018 — Qualifier les formules fermées selon le nombre de prix et le batching

- **Nature :** corrigé par couverture et mesure; sévérité originale moyenne,
  priorité haute, confiance prouvée pour le profil SM89 observé.
- **Signature originale :** les sondes CIR, schedules et overhead ne
  constituaient pas une courbe de débit couvrant tous les modèles analytiques,
  notamment les deux compositions de courbe des taux ajustés et
  Black--Scholes.
- **Clôture :** le manifeste de capacités génère maintenant une sonde pour les
  neuf compositions analytiques. La campagne couvre 1/16/1 000 prix et
  128/256/512 threads, sépare GPU, API publique, préparation, copie,
  publication et mémoire, et refuse de transformer ce périmètre en prédiction
  à un million de prix. Les 81 comparaisons par forme sont bitwise identiques.
- **Décision :** conserver 256 threads. Les kernels utilisent 23--40 registres
  par thread, sans stack, local spill ni mémoire partagée. Le seul résultat
  initialement bruité, G2++/Nelson--Siegel, a été répété; l'ordre des
  géométries n'est pas robuste et leur écart reste négligeable devant les
  5.0--9.8 ms de publication native.
- **Preuve :** rapport versionné
  `docs/performance-reports/closed-form-price-count-scaling-sm89-2026-09-06.md`;
  summary principale SHA-256
  `1de1dc68ac2d71efc88d8489df2c06fa0a7e68e46c02767c7e68e1efa7021323`,
  summary combinée de confirmation SHA-256
  `13151584be6b2ad47dbd7756e62c08cb13a9238da661f445910cffdb29b0b0b8`.
- **Portabilité :** aucune géométrie n'est présentée comme optimale hors RTX
  4090 Laptop/SM89. Chaque profil cible doit rejouer la matrice avant de
  modifier un défaut de production.
- **Réouvrir seulement si :** une composition analytique disparaît de la
  matrice, la géométrie modifie les résultats, des spills apparaissent, la
  publication cesse d'être isolée, ou un gain end-to-end d'au moins 5% devient
  répétable.

### PERF-021 — Ne pas interrompre une expérience sur variation de limite de puissance

- **Nature :** ouvert puis corrigé et fermé le 2026-09-07 à la demande de
  l'utilisateur; sévérité moyenne, priorité haute, confiance prouvée.
  Propriétaire : agent référent.
- **Signature originale :** après retrait des veto thermiques (`PERF-020`), le
  pilote scaling tue encore le processus quand la limite GPU varie de plus
  de 10 %; la sonde LSM possède un veto analogue à 20 %. Les bornes de
  comparabilité temporelle deviennent ainsi des arrêts d'expériences longues.
- **Preuve initiale :** `pricing-scaling-20260907-representative-small-matrix-01`
  interrompu à 170,54 → 150 W et `*-representative-rough-resume-01` à 175 →
  150 W, sans erreur CUDA et secteur déclaré avant/après. Expériences et
  études de génération bloquées; cela ne prouve pas une panne d'alimentation.
- **Correction :** règle partagée dans `tools/performance/experiment_environment.py`,
  appliquée aux deux pilotes exploratoires. La puissance est observée avant,
  pendant et après; les excursions excluent les timings de façon persistante,
  mais ne tuent plus le calcul et n'empêchent plus le job suivant. Les
  synthèses conservent les motifs; résultat complet ne signifie pas qualifié.
  Protocole permanent et query mis à jour; anciennes preuves inchangées.
- **Preuve de clôture :** 58 tests Python scaling/protocole et six tests de
  l'outillage LSM passent; CTest `performance_baseline_checker` et
  `pricing_scaling_protocol` : 2/2. Processus simulés poursuivis malgré
  excursion/rétablissement, deux jobs LSM achevés malgré écarts avant/live/après,
  exclusion persistante et absence de faux succès à faible CV vérifiées.
  Admission officielle toujours refusée hors de ses bornes; contrôles
  secteur/concurrence/power-brake et watchdog conservés dans leur périmètre.
- **Portée :** aucun kernel, réglage matériel, catalogue ou baseline modifié.
  Cette clôture de contrôleur ne qualifie pas le scaling et ne ferme pas
  `PERF-016`, `PERF-017` ou `PERF-019`; leurs preuves GPU restent distinctes.
- **Réouvrir seulement si :** un veto de variation de limite de puissance
  revient dans un pilote exploratoire, la télémétrie ou l'exclusion persistante
  disparaît, ou la correction permet de qualifier artificiellement des
  conditions non comparables ou de désactiver une protection matérielle.

### PERF-020 — Supprimer les veto thermiques applicatifs des expériences longues

- **Nature :** ouvert puis corrigé et fermé le 2026-09-06, sur demande
  explicite de l'utilisateur; sévérité moyenne, priorité haute, confiance
  prouvée. Propriétaire : agent référent.
- **Signature originale :** les pilotes de campagne et de profilage imposaient
  85 °C, refusaient les signaux de ralentissement thermique ou attendaient
  une température cible/stabilisée. Ces veto bloquaient les expériences longues
  et les expériences de génération/publication de bases de données, sans
  constituer un invariant numérique. Heston barrière a été interrompu avant
  la fin de sa troisième répétition à 1 000 prix × 1 048 576 trajectoires.
- **Périmètre exact :** contrôleurs d'expériences et sondes de publication.
  Aucun veto thermique trouvé dans les générateurs natifs `catalog`,
  `tools/pricing`, `tools/sampling`, `tools/cuda` ou le runtime `src`.
- **Correction :** retrait des plafonds, veto de ralentissement thermique,
  attentes de refroidissement et boucles de convergence de `run_baseline.py`,
  `run_pricing_scaling.py`, `run_fixed_income_lsm_probe.py` et
  `profile_kernel.py`. Le manifeste, son checker, les tests, le protocole et
  la query portent désormais la règle `thermal_policy: telemetry_only`.
  Les warmups déclarés et les contrôles puissance/concurrence/watchdog
  applicables restent en place. Le signal power-brake est distingué des
  signaux thermiques; aucun réglage firmware/driver n'est modifié.
- **Preuve :** 47 tests Python protocole/scaling et cinq tests de l'outillage
  LSM passent; CTest `performance_baseline_checker` et
  `pricing_scaling_protocol` : 2/2. Les fixtures vérifient les températures
  élevées et les signaux thermiques sans veto, et l'analyse AST interdit une
  branche ou attente thermique dans les quatre pilotes. Recherche globale
  sans seuil actif résiduel; `git diff --check` passe. Aucun test GPU lancé.
- **Provenance :** le manifeste précédent complet est conservé sous
  `tests/performance/history/baseline_sm89_v3_pre_perf_020.json`, SHA-256
  `de48e3a11fa4ccd50d59d151005e7b30fed2070e022346c4a36364d3a964c327`.
  Les 41 mesures, workloads et budgets statistiques/numériques/ressources
  restent strictement identiques. L'amendement de politique n'est pas une
  rebaseline mesurée; les anciennes preuves conservent leurs conditions.
- **Limites :** la télémétrie reste nécessaire à l'interprétation des timings.
  Une température observée ne qualifie pas leur stabilité. Cette clôture ne
  lève pas les refus d'autorisation externes et ne ferme pas `PERF-016`,
  `PERF-017` ou `PERF-019`; leurs campagnes restent à réaliser séparément.
- **Réouvrir seulement si :** un veto/attente thermique applicatif revient
  dans une expérience ou une génération de base, la télémétrie disparaît,
  ou cette suppression sert à masquer une dérive statistique, à réécrire les
  anciennes preuves ou à prétendre lever une protection/autorisation externe.
