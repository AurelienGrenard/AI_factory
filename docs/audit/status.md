# État de l'audit indépendant

## Rédaction accessible et bibliothèque locale — 2026-09-11

- **Mandat :** préciser la règle de rédaction et retirer l'ancien dossier
  `articles/`, à la demande de l'utilisateur. Conserver `Articles/` localement
  sans le publier. Changements portés par la PR #12.
- **Référentiel :** version 9.1. Toute documentation, README inclus, doit
  aider un néophyte à comprendre et explorer le projet. Les phrases sont
  courtes, claires et descriptives. Termes techniques, acronymes et prérequis
  sont expliqués. Le guide documentaire renvoie à cette règle unique.
- **Bibliothèque :** retrait des six PDF suivis dans `articles/` et exclusion
  Git des deux dossiers `/articles/` et `/Articles/`. La grande bibliothèque
  locale reste sur disque. Les commits historiques ne sont pas réécrits.
- **Portée :** mise à jour des consignes, pas revue de conformité de toutes
  les pages. Aucun code numérique ni résultat historique modifié. Aucun test
  GPU nécessaire; `DELTA-001` et `PERF-016` restent ouverts.

## Consolidation Git et prochaine étape rough — 2026-09-11

- **Trace GitHub :** le [commit de consolidation `4a8c140`](https://github.com/AurelienGrenard/AI_factory/commit/4a8c140c74d39a78920ecd8e9209b1eee6082410)
  a été poussé directement sur `main` avant la demande de PR. Le
  [diff complet depuis le 3 septembre](https://github.com/AurelienGrenard/AI_factory/compare/872a986b1f0947a1a832af0615ffc6d80dbedb81...4a8c140c74d39a78920ecd8e9209b1eee6082410)
  conserve les changements logiciels. Une PR documentaire de suivi porte la
  trace de revue rétrospective; le code est déjà intégré, l'historique n'est
  pas réécrit et les qualifications ouvertes ci-dessous restent inchangées.
- **Mandat :** commit et push demandés du chantier logiciel accumulé depuis
  `872a986b1f0947a1a832af0615ffc6d80dbedb81` (2026-09-03), avec mention explicite
  de la suite rough. Cette consolidation ne constitue pas un nouvel audit
  global ni une nouvelle qualification GPU.
- **Périmètre :** corrections numériques et ressources CUDA, CIR/CIR++ sous
  numéraire terminal, swaptions européennes G2/G2++ Monte Carlo, planning de
  production à 2^20 chemins, génération reprenable et provenance, organisation
  CMake/codegen/tests/documentation, puis prix-delta equity markovien et ses
  recettes. Les preuves et limites de chaque lot restent dans les entrées
  historiques et les clôtures correspondantes.
- **État initial :** main, 818 suivis modifiés, 82 supprimés, 2 016 non suivis,
  index vide. La bibliothèque locale `Articles/` (388 fichiers) et les six
  suppressions PDF dans `articles/` restent hors de cette consolidation;
  builds, datasets et archives ignorés restent locaux.
- **Preuves prix-delta revérifiées :** empreintes de l'archive markovienne et
  du binaire de parité conformes au bloc du 2026-09-10; ses 2 824 fichiers
  identiques aux sources à la reprise. Les empreintes des sources codegen et
  des 2 805 sorties hors manifeste concordent. Présence de 261 bibliothèques
  prix-delta compilées et 364 recettes; aucun dataset.yaml prix-delta publié
  dans le catalogue. Les résultats GPU historiques ne sont pas relancés.
- **Contrôles de consolidation :** 60 tests Python manifeste/campagne/provenance
  réussis; régénération dans `/tmp` de 2 806 sorties, zéro diff; contrôles
  d'arborescence et des 1 086 recettes catalogue passants. Le contrôle initial
  des fichiers déjà suivis est propre; après indexation des nouveaux fichiers,
  `git diff --cached --check` signale 21 espaces de fin de ligne dans 14
  façades BS générées et les fins CRLF de trois rapports CSV. Ces contenus
  préexistants sont conservés pour maintenir le zéro diff codegen et les
  empreintes historiques; aucune modification numérique n'en est déduite.
  Aucune compilation ni exécution GPU dans cette consolidation.
- **Prochaine étape :** delta S0 rough Heston et quadratic rough Heston
  N-facteurs, pilote européen/barrière avec parité centrale et ressources
  ciblées; puis rough Bergomi FFT. Préparations et transitions N-facteurs
  inspectées : les facteurs sont indépendants du niveau S0. Implémentation et
  qualification restent à faire. `DELTA-001` reste ouvert et `PERF-016`
  reporté; aucune certification de biais ou performance globale ajoutée.

## Prix-delta — extension markovienne et recettes — 2026-09-10

- **Mandat :** points 1 et 2 demandés : intégrer génération/provenance et
  couvrir le catalogue equity markovien, sans passer au rough. Pas nouvel
  audit global. `main`, HEAD `872a986b1f0947a1a832af0615ffc6d80dbedb81`;
  worktree préexistant conservé, index vide. À la consignation : 818 suivis
  modifiés, 82 supprimés, 2 016 non suivis (`--untracked-files=all`); ce n'est
  pas le diff de ce lot. Query et validation non modifiées par ce travail.
- **Couverture :** 12 modèles, 261 bindings compilés : 244 MC, huit formules
  fermées BS, neuf LSM. 364 recettes dérivées du catalogue existant : 334 MC,
  14 variantes fermées, 16 variantes LSM. BS American possède un binding mais
  pas de recette prix dans le catalogue source; aucune recette fictive ajoutée.
  Codegen génère aussi recipe.yaml prévisionnel; dataset.yaml est un résultat
  d'exécution. Cible `price_delta_generators` séparée de `price_generators`.
  Les fichiers inchangés gardent leur mtime lors d'une régénération.
- **Factorisation :** mêmes payoffs/contrats, sept policies BS extraites dans
  des headers produit partagés avec le prix seul; SABR partage deux normales
  et sa transition existante entre trois états à volatilité initiale relative
  constante. LSM conserve les coordonnées de continuation et raffinements
  existants, notamment Bates/Schobel-Zhu et Kou. Pas de nouveau refit ni de
  changement de solveur, base, dates ou mapping Philox.
- **Publication :** 2^20 chemins MC/LSM, zéro chemin analytique, aliases CRN
  des prix sans rekeying. Métadonnées bump réel/méthode/seed/grille/géométrie;
  contrôle des deux artefacts et provenance du recipe.yaml gelé. Bump/méthode
  entrent dans la spécification sémantique. Validation pending/false, sans
  faux validateur. Sept recettes natives testées sur deux lignes temporaires,
  calendriers raccourcis; aucune publication de base catalogue.
- **Défaut incident corrigé :** une première fixture de génération conservait
  les calendriers JSON d'origine (les clés sont maturity/observation_interval,
  pas les noms C++). Elle a révélé un prix SABR NaN à 256 threads. Reproduction
  indépendante avec payoff constant à 2^20 : moments centrés relatifs
  −8,45402e−14 à 128 threads et −4,25995e−14 à 256, au-delà du garde fixe
  64 epsilon. La borne MC commune inclut désormais le nombre de termes par
  thread et la réduction; prix/erreur finis après correction, incohérences
  matérielles toujours rejetées. Le cas SABR avec ses dates d'origine passe
  aussi. Ni sommes, ni ordre de réduction, ni FP32 des dynamiques/payoffs
  modifiés; le garde FP64 s'exécute une fois par prix. LSM/FFT conservent
  leur garde antérieur. Les fixtures courtes ont été corrigées explicitement.
- **Vérifications :** 60 tests Python manifeste/campagne/provenance passent;
  deux CTests hôte planner/publication passent; sept CTests finaux prix-delta
  et moments constants passent en 2,88 s. 34 compositions publiques vérifiées,
  prix central et erreur standard bitwise : deux lignes BS/Heston/CEV American
  à 2^20 chemins, autres MC/LSM à 4 096 pour couverture bornée. SABR couplé
  comparé aux transitions originales re-préparées, delta pathwise exact sur
  le cas court. Memcheck de la couverture publique : zéro erreur. 2 806
  sorties codegen zéro diff, checker catalogue et arborescence passent.
- **Ressources :** SM89 Release, CUDA 13.3.73/GCC14, MC testé à 256 threads.
  Delta européen : 58–104 registres selon modèle; produits riches Bates :
  117–142, mémoire locale déclarée nulle dans ces spécialisations. Formules
  fermées delta : 30–44 registres, mémoire locale déclarée nulle. Les 140–142
  registres des autocalls justifient de ne pas reprendre aveuglément 512
  threads : le planner delta plafonne le candidat MC à 256, sans changer
  le profil prix seul ni le LSM. Ce n'est pas un optimum ni une garantie
  multiarchitecture. Pas de mesure comparative de performance longue.
- **Preuves :** `build-price-delta/extension-all-bindings-final-build.log`,
  `extension-python-tests.log`, `extension-zero-diff.log`,
  `extension-catalog-check.log`, `extension-layout.log`,
  `extension-gpu-final.log`, `extension-catalogue-parity-final.log`,
  `extension-memcheck-final.log`, `extension-generators-smoke-final.log`,
  `extension-constant-before.log`, `extension-constant-after.log`,
  `extension-sabr-calendar-diagnostic.log`, `extension-sabr-calendar-fixed.log`.
  Dossier ignoré : conserver/exporter ces preuves avec le snapshot de sources.
  Archive `markovian-extension-sources.tar.gz`, SHA-256
  `98ef547175aea5a4020fa240c442c058175b2dc169a87f550e0a2bf618d0d599`;
  binaire de parité catalogue
  `fe2de3c268aeedfc36ea8f546c751376950698f6e584c377fb922efd125e2bd5`.
- **Exclusions / suivi :** rough, campagne complète des 364 recettes, tuning
  delta, certification singularités/bump/biais LSM, références Premia/QuantLib
  et PERF-016. DELTA-001 reste ouvert pour ces suites : deux ouverts,
  106 fermés, 108 identifiants. Aucun commit/push ni certificat global.

## Prix-delta — pilote LSM BS/Heston/CEV terminé — 2026-09-10

- **Mandat :** point 1 uniquement de l'extension, pas nouvel audit global.
  `main`, HEAD `872a986b1f0947a1a832af0615ffc6d80dbedb81`; worktree
  préexistant conservé, index vide. État final avant consignation : 804
  suivis modifiés, 82 supprimés, 759 non suivis; ces nombres ne sont pas le diff
  de ce pilote. Aucun commit/push ni génération de base.
- **Implémentation :** trois façades American prix-delta générées, call/put;
  une seule régression centrale. Dates et spots retenus coûtent 8 octets par
  chemin, plus un indicateur par prix, intégrés au planner VRAM. BS/Heston
  réutilisent le spot arrêté; CEV rejoue les innovations communes jusqu'à
  l'exercice central. Deux passages ajoutés par batch, scratch de moments
  réutilisé. Exercice en zéro, maturité, stub et invalidation sont explicites.
  Solveur, base, équations et méthode de régression inchangés.
- **Tests :** six CTests CUDA ciblés réussis en 6,25 s : quatre prix-delta,
  régresseur existant et American Heston existant. Le pilote LSM compare
  quatre lignes à 4 096 chemins, deux géométries et deux bumps; prix centraux
  et erreurs standards bitwise. L'oracle CEV vérifie 512 chemins, dates
  intermédiaires/maturité et absorption, plus budget mémoire et sorties
  invalides. Deux memchecks réussis, zéro erreur. 29 tests du manifeste,
  layout et codegen 1 569 sorties zéro diff. Les petits comptes de test ne
  remplacent pas les 2^20 chemins prévus pour les futures recettes.
- **Comparaison au refit :** diagnostic borné, pas certificat de biais.
  Écarts absolus jusqu'à 0,0322 sur le call CEV et 0,0301 sur son put;
  l'erreur standard appariée ne couvre pas le biais de policy LSM approchée.
- **Ressources SM89, CUDA 13.3.73/GCC14, call, 128 threads :** simulation
  centrale prix→prix-delta : BS 48→48 registres, Heston 69→60, CEV 55→48.
  La mémoire locale déclarée vaut respectivement 32→32, 0→32 et 0→32 octets;
  les rapports ptxas identifient une pile de 32 octets et **zéro spill
  load/store** dans les trois unités delta, pas une absence de coût mémoire.
  Moments delta : 34/40/77 registres BS/Heston/CEV; CEV a aussi 32 octets de
  pile. Aucun FP64 ajouté aux dynamiques/payoffs; moments et régression
  conservent leur contrat FP64. Timings des smoke tests avec effets à froid,
  sans qualification de surcoût ni réglage production/multiarchitecture.
- **Preuves locales ignorées, à conserver/exporter :** `build-price-delta/`,
  logs `lsm-regression-tests.log`, `lsm-memcheck.log`,
  `lsm-replay-memcheck.log`, `lsm-codegen-final.log` et `*-lsm-ptxas.log`.
  Archive runtime/outils/CMake/tests concernés/contrats CUDA
  `lsm-pilot-sources.tar.gz`, SHA-256
  `906b1a0ad9661f5eccfc6d5d4b06de500c9c6f8ae425771adad0e9532b1aa5fd`.
  Binaires pilote/oracle :
  `d79bd4b96a014246597760947f24b641760b95001ef5414a1b52652ab80d23c9`
  et `4bc1858037965c233fd7485f091a60d2871c75fa54562ef9b1e9d8b704ab599f`.
- **Exclusions :** autres couples LSM/MC, rough, recettes/datasets/provenance
  prix-delta, références Premia/QuantLib et campagnes longues. `DELTA-001`
  reste ouvert pour la suite; `PERF-016` reste reporté. Deux ouverts,
  106 fermés, 108 identifiants. Query et validation non modifiées.

## Extension prix-delta — premier lot borné — 2026-09-10

- **Mandat :** implémentation progressive demandée, petits tests de parité;
  pas nouveau passage global de la query ni campagne PERF-016.
- **Révision :** main, HEAD `872a986b1f0947a1a832af0615ffc6d80dbedb81`,
  worktree déjà très modifié et index vide. Les changements préexistants
  restent en place; les décomptes historiques ne représentent pas ce lot.
- **Couverture :** stratégies multiplicative et CEV couplée, payoffs/handlers
  existants, réductions appariées, guards de bump et launchers distincts;
  sept couples pilotes générés. Extraction de la composition européenne
  analytique chez le produit, réutilisée par les deux launchers BS.
- **Preuves :** build Release SM89 CUDA 13.3.73/GCC14, sept archives nouvelles;
  deux CTests GPU courts passants dans `build-price-delta`, incluant vrais
  launchers publics, parité centrale bitwise et delta BS analytique.
  29 tests du manifeste passent; layout conforme et codegen 1 563 fichiers
  zéro diff. Logs GPU/ressources dans `price-delta-final-tests.log`, contrôles
  codegen dans `codegen-final-check.log`. Pas de temps de production extrapolé.
  Compute Sanitizer memcheck du test MC : zéro erreur (`memcheck.log`).
  Archive des sources runtime/outils/CMake, tests prix-delta et contrats CUDA :
  `pilot-sources.tar.gz`, SHA-256
  `2733227322fc1e68beb0e29e867dac201bcedc856917490f8694604885e39a70`.
  Binaire MC SHA-256
  `d385d8f66882f5a41307ae987cd1d47fc31a298226104d82a4014b5de7e96b50`;
  binaire fermé SHA-256
  `1d7212983eda323d3d0c3e76c7a39f94f95ef7b420602227eba48135f0a2ddfa`.
- **Ressources :** sondes MC BS 48→53 registres, Heston européen 68→83,
  CEV 55→71, barrières Heston 70→94 et CEV 56→80, sans mémoire locale
  déclarée. BS fermé 23→40. Ce coût pilote n'est pas un retuning ni une
  qualification multiarchitecture; pas de FP64 ajouté aux dynamiques.
- **Exclusions :** LSM, autres modèles/produits, FFT/N-factor, recettes et
  publication prix-delta, qualification des singularités et performance
  production restent dans DELTA-001. Les seeds, bases existantes, réglages
  de génération, validation indépendante et query sont inchangés.
- **Registre :** DELTA-001 ouvert, aucune clôture prématurée : deux ouverts,
  106 fermés, 108 identifiants. Le contrat permanent distingue implémenté,
  testé et prévu. Aucun push/commit ni génération de base.

## Provenance et compatibilité — STRUCT-028 corrigé — 2026-09-10

- **Mandat :** traiter la conservation des datasets après refactoring avant
  PERF-016. Intervention sur l'outillage hôte, pas sur la mécanique CUDA.
- **Snapshot :** main, HEAD `872a986b1f0947a1a832af0615ffc6d80dbedb81`;
  799 suivis modifiés, 76 supprimés, 323 non suivis à l'entrée, index vide.
  Worktree préexistant conservé. Archive d'entrée
  `build-dataset-provenance-20260910-z8KotZ/before-sources.tar.gz`, SHA-256
  `def3fb6a14140c49d3512c0d6d250d8e928da962fa2fc85d14cc2566434eae22`.
- **Résultat :** STRUCT-028 ouvert puis fermé, **1 ouvert, 106 fermés,
  107 identifiants**. PERF-016 reste seul ouvert; aucun verdict global.
- **Couverture :** enregistrement generation v1, contrôleur de campagne v2,
  fingerprints d'entrées ordonnées indépendants des chemins, checksum
  JSON/recette/enregistrement et diagnostic en lecture seule. Staging,
  reprise, conservation des anciens artefacts, absence de provenance et
  changements de seeds, paramètres, forme, binaire ou géométrie testés.
  Les invariants sont centralisés, sans modification des templates codegen.
- **Vérifications :** 35 tests Python réussis; 21 CTests hôte réussis, dont
  la nouvelle entrée `dataset_provenance`, codegen zéro diff et architecture.
  Gel réel de `generate_heston_samples_01` avec le build SM89 déjà qualifié :
  snapshot et candidat concordants, job toujours pending, aucun générateur
  exécuté. Le chemin prix/inspecteur est couvert par tests CPU simulés;
  aucune nouvelle campagne native de prix/samples publiée.
- **Preuves :** dossier ignoré ci-dessus, à conserver/exporter : `python-tests`,
  `host-ctests`, `freeze-probe`, `scope`, archive d'entrée et campagne figée
  Heston non exécutée. Chaque commande dispose d'un retour et d'un hash de log.
  `scope` vérifie notamment 1 304 fichiers src, 1 132 catalogue et 96 codegen
  inchangés. Contrat dans `docs/dataset-provenance-contract.md`.
- **Exclusions / limites :** aucun backfill des bases anciennes, aucun changement
  dans `src`, RNG, FP64 device, tuning, recettes ou audit de validation.
  L'enregistrement automatique appartient au contrôleur, pas aux invocations
  natives directes ni à la génération autonome des paramètres. Pas de
  certification, d'inférence d'équivalence mathématique ou de build hermétique;
  une preuve manquante ou un binaire différent requiert une revue explicite.
  Aucune campagne GPU, validation externe ou opération Git distante.
- **État final :** 799 suivis modifiés, 76 supprimés, 327 non suivis, index
  vide; query inchangée. Les quatre nouveaux fichiers sont deux outils
  Python, leur suite de tests et le contrat documentaire.

## Remédiation structure tests — STRUCT-027 corrigé — 2026-09-10

- **Mandat :** ranger et nommer les tests par responsabilité, sans modifier
  leur logique ni le code de production. Pas de campagne PERF-016.
- **Snapshot :** main, HEAD `872a986b1f0947a1a832af0615ffc6d80dbedb81`;
  815 suivis modifiés, 5 supprimés, 252 non suivis à l'entrée, index vide.
  Changements préexistants préservés. Archive des sources avant intervention :
  `build-remediation-test-layout-20260910-kHS8Sz/before-sources.tar.gz`,
  SHA-256 `838e231b9960526f0e058c171d468825d93889c261d7b5a022e1768f7977d318`.
- **Résultat :** STRUCT-027 fermé : **1 ouvert, 105 fermés, 106 identifiants**.
  PERF-016 reste seul ouvert. Aucun verdict global de conformité.
- **Couverture :** 74 sources et une fixture déplacées, sept noms clarifiés,
  quatre en-têtes précisés et un include adapté. Le README donne les
  propriétaires et CMake pointe vers les nouveaux chemins. Les 83 sources
  natives restent enregistrées; 346 entrées CTest conservent noms,
  propriétés et commandes attendues (92 principales, 254 de validation).
  Aucune source de production ni contenu de test hors en-têtes/include
  modifié; aucun ancien chemin actif. Rapports historiques conservés.
- **Vérifications :** build complet des tests Release SM89, CUDA 13.3.73,
  g++14, mathDx 26.06, à deux tâches; 20 CTests hôte et 11 CUDA réussis,
  aucun sauté. Codegen 1 549 sorties zéro diff, contrôles d'architecture
  passants, `git diff --check` propre. Les 254 validations ne sont pas lancées.
- **Preuves :** même dossier ignoré, à conserver/exporter : `mapping.json`,
  `configure`, `build-tests`, `host-tests`, `cuda-tests`,
  `qualified-layout-invariants`, inventaires CTest avant/après et snapshot.
  Le premier contrôle post-build comparait littéralement les commandes JSON
  omises par CTest pour 64 exécutables encore absents avant compilation;
  le contrôle qualifié vérifie leur résolution vers les targets originaux,
  sans ignorer les autres champs. Ce n'était pas une modification de contrat.
- **État final :** 799 suivis modifiés, 76 supprimés, 323 non suivis,
  index vide. Les suppressions/ajouts incluent les déplacements non indexés;
  ce décompte global n'est pas le volume de code changé par ce lot.
- **Exclusions :** query, validation, `src`, outils, recettes, données et
  tuning inchangés. Aucune campagne de performance ou génération de bases,
  aucun moteur externe ni nouveau test numérique; autres architectures
  et sanitizers non rejoués pour ces déplacements.

## Remédiation samples — STRUCT-025/026 corrigés — 2026-09-10

- **Mandat :** compléter les lois des recettes samples et rendre leurs
  métadonnées/replays FFT fidèles; ni STRUCT-027, ni provenance générale,
  ni campagne PERF-016 dans ce lot.
- **Snapshot :** main, HEAD `872a986b1f0947a1a832af0615ffc6d80dbedb81`,
  782 fichiers suivis modifiés, 5 supprimés, 251 non suivis à l'entrée,
  index vide. Changements préexistants conservés.
  Archive des sources, outils, recettes et contrats avant intervention :
  `build-remediation-samples-20260910-Z9uPBq/before-sources.tar.gz`.
  SHA-256 `05ab8627971425872abe8687e3ff6739eba036a79b15954b1df6fb2ba4c41df2`.
- **Preuves locales :** même dossier ignoré, à conserver/exporter;
  commandes, codes retour, durées et hashes de logs. Réutilisation du build
  Release SM89 CUDA 13.3.73/g++14/mathDx 26.06 qualifié au lot 1.
- **Résultat :** STRUCT-025 et STRUCT-026 fermés : **2 ouverts, 104 fermés,
  106 identifiants**; STRUCT-027 et PERF-016 restent ouverts.
- **Contrôles :** 28 tests Python, codegen 1 549 sorties zéro diff,
  layout source et frontières des générateurs passants. Quatre bindings FFT,
  deux générateurs Heston et les tests ciblés compilés. Deux CTests CUDA et
  assembly samples JSON/YAML passants. Les huit recettes FFT sont exercées
  sans publication sur 1 000 lignes; leurs métadonnées correspondent aux
  diagnostics natifs (128 threads, grille 4→3). Plans de production contrôlés
  côté hôte (4 096→4 095 blocs), pas de replay complet à trois millions de
  lignes. Logs `matched-native-replay`, `fft-regression`, `sample-stage`,
  `static-contracts`, `codegen-zero-diff`.
- **Périmètre vérifié :** `unchanged-numerics` compare au snapshot les 25
  fabriques et les 1 132 fichiers du catalogue, inchangés, ainsi que le code
  device FFT. Modifications limitées aux descriptions, orchestration hôte,
  descripteur de dimensions et ses façades générées, tests et contrats.
  Un cache Python ignoré a été rafraîchi par CMake; aucune source étrangère
  au lot modifiée. État final : 815 suivis modifiés, 5 supprimés, 252 non suivis,
  index vide; query inchangée. Aucun verdict global de conformité.
- **Exclusions :** aucune équation, fabrique de paramètres, logique kernel,
  configuration de tuning, seed ou domaine RNG modifié; aucune génération
  de production, validation Premia/QuantLib, campagne longue ou opération Git
  distante. Autres architectures et campagnes machine complètes non qualifiées.

## Remédiation du lot 1 — corrigé, qualification ciblée — 2026-09-10

- **Mandat :** NUM-012/022/023/024/025/026/027, TEST-001/002 et BUILD-010.
  STRUCT-025/026/027, provenance et PERF-016 hors de ce lot.
- **État initial :** main, HEAD `872a986b1f0947a1a832af0615ffc6d80dbedb81`;
  769 fichiers suivis modifiés, 5 supprimés, 247 non suivis, index inchangé.
  Changements préexistants conservés. Archive des sources/contrats :
  `before-sources.tar.gz`, SHA-256
  `3e6bc591a36a725330edd04edfded6ea37fb6f536bb9c5dd9f08f9171a92caf0`.
  Cette archive ne prétend pas inclure les datasets/validations.
- **Preuves locales :** `build-remediation-v9-lot1-20260910-xdQwSa/`,
  ignoré, à conserver/exporter. Chaque contrôle conserve commande,
  code retour, durée et hash du log. Build neuf Release SM89,
  CUDA 13.3.73, hôte g++14, mathDx 26.06; CUDA exécuté hors sandbox.
  Premier build sans compilateur hôte CUDA explicite : échec C++23,
  logs conservés. Build corrigé distinct.
- **Mouvement qualifié :** NUM-012/022/023/024/025/026/027,
  TEST-001/002 et BUILD-010 corrigés et fermés :
  **4 ouverts, 102 fermés, 106 identifiants**. Restent uniquement
  STRUCT-025/026/027 et PERF-016, hors mandat du lot 1.
- **Contrôles passés :** runner original, mutant NaN rejeté, robustesse
  numérique et Initcheck sans erreur; inférence CMake ajout/retrait;
  codegen zéro diff (1 549 sorties), 26 tests de manifeste.
  Série de 14 tests CUDA ciblés, puis six contrôles des dernières versions,
  dont 2 904 cas G2 (erreur maximale 2.6431e-5, budget 3e-5),
  absorption Asian, calendriers et replay FFT bitwise. Smoke-test des
  pricers G2/G2++ à 4 096 chemins passant. Memcheck G2 et FFT : zéro erreur.
  Logs qualifiants : `test-consumers-final`, `test-retained-final`,
  `test-g2-pricing-retained`, `memcheck-g2-retained`, `memcheck-fft-retained`.
- **Performance bornée :** ressources avant/après SM89 pour G2/G2++, Bates,
  VG, Asian CEV et samples FFT. Sur 32 prix × 32 768 chemins,
  256 threads/32 blocs, G2 ~.897→.957 ms et G2++ ~2.777→2.851 ms.
  Coûts de correction explicités dans les clôtures, aucune qualification
  portable ni extrapolation à un million de prix. La branche stable G2
  seule est non inlinée pour conserver les géométries 512 threads G2++.
  Les premiers prototypes trop coûteux, l'échec de lancement G2++ à 157
  registres et les essais numériques échoués restent dans les preuves;
  ils ne décrivent pas la version retenue (`*-retained`). Une première
  fixture FFT donnait trop de blocs pour sa tranche : corrigée sans relâcher
  le garde de production. Timeout initial et logs conservés.
- **Exclusions :** aucune validation Premia/QuantLib, publication ou
  régénération de datasets/YAML, campagne longue, rebaseline, commit/push/PR.
  La régénération codegen change seulement la façade range accrual BS et
  ses empreintes. Aucun nouvel audit exhaustif ni conformité globale :
  les verdicts de l'audit initial ci-dessous restent historiques.

## Complément ciblé — organisation de tests — 2026-09-10

- **Mandat :** consigner, à la demande de l'utilisateur, la dispersion des
  tests constatée après lecture de l'audit indépendant. Aucun rangement ni
  correctif autorisé ou appliqué pendant ce complément.
- **Révision / worktree :** `872a986b1f0947a1a832af0615ffc6d80dbedb81`,
  branche `main`, aucun changement staged; 769 fichiers suivis modifiés,
  5 supprimés et 247 fichiers non suivis non ignorés à l'entrée. Ces
  changements préexistants sont conservés; le commit seul ne décrit pas
  les sources examinées.
- **Couverture / verdict :** revue statique ciblée des axes 1 et 4,
  couverture partielle, **non conforme localement pour l'organisation de
  `tests`** (`STRUCT-027`). Les autres axes et la justesse des tests ne sont
  pas réévalués. Le bilan indépendant ci-dessous reste un passage historique
  distinct; son absence de constat de rangement ne qualifie pas cet aspect.
- **Preuves :** inventaire `rg --files tests`, lecture du README, des en-têtes
  et includes des tests représentatifs, des enregistrements dans
  `cmake/AIFactoryTests.cmake` et du rapport spécialisé de structure.
  Le comptage reproductible
  `rg --files tests | awk -F/ 'NF==2 && /\.(cu|cpp)$/ {n++} END {print n}'`
  donne **74 sources C++/CUDA à la racine**. Les domaines datasets, LSM,
  samples, modèles et précision y coexistent malgré les sous-dossiers;
  exemples et conséquences sont consignés dans `response.md` sous STRUCT-027.
- **Historique / mouvement :** signatures fermées examinées, notamment
  STRUCT-016/019 et NAME-007/012; pas de réouverture ni doublon. Un nouveau
  constat de sévérité moyenne et priorité moyenne : **14 ouverts, 92 fermés,
  106 identifiants**. Aucune clôture; `closed.md` reste inchangé.
- **Vérification / périmètre :** seules `response.md` et `status.md` sont
  modifiées. Contrôle des identifiants, des nouveaux liens et de
  `git diff --check`; empreintes du diff suivi hors ces deux registres, de
  l'index, de `query.md` et de `closed.md` comparées avant/après.
- **Exclusions :** aucun déplacement, modification de code/test/CMake,
  ajout de règle à la query, build, test CPU/CUDA, génération, campagne de
  performance ou validation indépendante.

## Audit indépendant du worktree — query v9 — 2026-09-09, consolidé le 2026-09-10

**Passage consolidé, couverture partielle, non-conformités locales prouvées.**
Le coordinateur a consolidé les trois spécialistes : structure/codegen
(axes 1, 3, 5), finance/numérique (2, 6 et oracles de 4), CUDA/performance
(7, 9 et coûts FP64); il possède tests/CMake (4, 8), interfaces et
contre-revues. Les neuf axes ont un responsable; aucun n'est déclaré
entièrement qualifié par le nombre de tests ou de wrappers.

Interrompu à la demande de l'utilisateur le 9 septembre, le passage est
repris le 10 pour consolidation uniquement. Les empreintes confirment que
les trois registres sont les seules différences depuis le snapshot, index
inchangé. Les deux sondes produits finales, auparavant seulement dans /tmp,
sont reconstruites et confirmées dans le dossier durable. Aucun correctif ni
constat fermé, aucune campagne longue. Les compléments SM75/SM86 et SASS
annoncés avant l'interruption n'ont pas conservé leurs artefacts finaux : ils
ne servent pas de preuve qualifiante dans cette consolidation.

### État, preuves et intervention

- **État audité :** HEAD `872a986b1f0947a1a832af0615ffc6d80dbedb81`, branche
  `main`, aucun changement staged (l'index contient normalement les fichiers
  suivis). État initial : 769 fichiers suivis modifiés, 5 supprimés,
  247 fichiers non suivis non ignorés, soit 774 changements suivis.
  Les anciens comptes de 113 entrées regroupaient des répertoires non suivis.
- **Snapshot :** `initial-state.json`, 7 815 chemins dont 5 absents,
  SHA-256 `1eddcc103da4261864b23480ef717027e19ed0eaa1a2b1aaba7d1258f24a17e3`;
  archive reconstructible des fichiers suivis et non suivis non ignorés
  `initial-worktree.tar.gz`, 182 208 828 octets, SHA-256
  `2c469dda0859011863e7c27462f9ddfa28b645c3ad99307de07ccf1f83c9e147`.
  HEAD seul ne décrit donc pas les sources auditées. Les arbres ignorés de
  builds/datasets ne font pas partie de cette archive; les inputs et binaires
  des sondes sont conservés séparément, avec leurs hashes.
- **Référentiel lu :** AGENTS, README racine, docs/README, query v9 et trois
  registres intégralement; contrats dynamics/analytics/pricing/LSM/launch,
  génération/extension/publication et protocole performance selon propriétaire.
  Query inchangée, SHA-256
  `72803627905075b76b0b43430048ec821712a8618c7f6e62ef1263314a78081b`.
- **Dossier de preuves local :** `build-audit-v9-20260909/`, isolé et ignoré,
  avec archive initiale, inventaires, rapports spécialisés, sources des sondes,
  commandes, binaires et logs. `evidence-manifest.json` porte les hashes;
  `README.md` décrit la reconstruction. Ce dossier doit être conservé ou
  exporté avec l'audit : il n'est pas distribué par un clone Git. Les preuves
  ne reposent pas uniquement sur la session `/tmp/ai_factory_audit_20260909`.
  Les chemins de preuves ci-dessous sont relatifs à ce dossier, sans lien
  documentaire suivi vers un arbre ignoré.
- **Conservation :** contrôle global SHA avant modification des registres :
  aucun changement concurrent parmi les 7 815 chemins; index inchangé,
  SHA-256 de `git ls-files --stage`
  `8464e85e604785e6474dfb87f5e083d87bd6a5cdb0a93bdcacc6eb0192eb7d84`.
  Le contrôle final exige exactement les trois registres autorisés comme
  différences depuis le snapshot; ses résultats sont dans `final-scope.json`.
  Aucun correctif de production, commit, push ou PR.

### Couverture et verdict des neuf axes

| Axe de la query | Couverture | Verdict sur le périmètre examiné | Preuves et limites |
|---|---|---|---|
| 1. Structure, découverte, naming, documentation | partielle | indéterminé globalement; parcours et ownership examinés cohérents | Navigation README→contrat→owner→binding suivie, 62 Markdown/322 liens contrôlés; lecture fine transverse common/model/product/tests/tools, pas chaque corps ni chaque en-tête. |
| 2. Justesse financière | partielle | non conforme localement | `NUM-012/022/025/027`, secondaires `NUM-023/024`; matrice des 25 modèles et branches distinctes dans `finance/report.md`. Plusieurs payoffs et oracles, convergence rough et qualification complète LSM restent partiels. |
| 3. Homogénéité et factorisation | partielle | non conforme sur l'interface samples FFT; architecture principale cohérente | `STRUCT-026`; concepts et assemblages Markov/FFT/N-factor/LSM relus, dry-runs de trois extensions. L'inventaire ne constitue pas une preuve mathématique de toutes les exceptions. |
| 4. Tests | partielle | non conforme localement | `TEST-001/002`; assertions, références et échecs ciblés examinés, 111 tests Python et 18 CTests natifs passent; mutant NaN accepté et Initcheck négatif. Suite principale complète et tous oracles non rejoués. |
| 5. Codegen et génération | partielle | non conforme localement | `STRUCT-025/026`; zéro divergence de 1 549 sorties, 722 recettes, reprise/publication mockées. Pas de génération volumineuse ni de qualification GPU de chaque forme samples. |
| 6. Robustesse numérique et reproductibilité principale | partielle | non conforme localement | `NUM-012/023/024/026`, secondaires `NUM-022/025`; contre-exemples natifs et inventaire FP64 interprété. Domaines complets, raffinements, coûts comparés FP32/FP64 et multi-SM non qualifiés. |
| 7. Sécurité CUDA | partielle | indéterminé pour l'ensemble de la production | Gardes/workspaces/synchronisations lus, 20 contrôles sanitizer sur cinq tests : 19 sans erreur, padding de fixture sous `TEST-002`. Reset FFT réussi sur SM89; streams non défaut, multi-device et toutes spécialisations non éprouvés. |
| 8. CMake, compilation, portabilité | partielle | non conforme localement | `BUILD-010`; graphe source/targets bidirectionnel, configuration fraîche sans mathDx, builds Release ciblés et no-op. Pas de build complet de tous les targets ni de runtime sur une seconde architecture. |
| 9. Performance | partielle | indéterminé pour le scaling et la qualification globale | Quatre sous-audits et pipeline séparés ci-dessous; provenance de 29 cas historiques vérifiée, ressources fraîches ciblées. `PERF-016` reste ouvert; campagne longue soumise à accord et non lancée. |

### Inventaire et parcours réconciliés

`structure/inventory.json` et `inventory-summary.json` confrontent sources,
manifestes, sorties et catalogue : 25 modèles (12 equity markoviens, 6 rough,
7 taux), 26 produits, 2 courbes, 13 familles de moteurs, 427 bindings
(410 générés et 17 manuels), 722 recettes dont 652 générées. Les 1 549 sorties
physiques correspondent aux 1 548 sorties comptées par le générateur plus son
manifeste de provenance : aucun manquant, extra ni contenu divergent.
Les données locales contiennent 407 JSON avec 407 YAML correspondants;
315 recettes générables ne sont pas matérialisées. Cette absence seule n'est
pas un défaut. Le verrou `.generation-campaign.lock` est identifié séparément.

Le graphe avec mathDx contient les 1 335 sources physiques standalone de
src/tools/tests/catalog; 1 404 commandes pour 1 400 sources uniques, incluant
les probes générés de build. Quatre sources ont deux propriétaires justifiés
par leurs variantes A/B. Sans mathDx : 1 177 commandes, 1 175 sources uniques;
217 sources physiques omises (88 src, 124 catalog, 5 tests), aucun générateur
de paramètres omis. JSON de preuve `build-dev-graph.json` et
`build-no-mathdx-graph.json`.

Les parcours examinés comprennent paramètres/loader→binding→préparation→
dynamique→payoff→moments/discount pour VG/Bates et G2; CIR terminal-forward
et interface Bermudan/LSM; calendrier et moteur rough FFT; recettes
samples→fabrique→adaptateur→lancement→sorties→publication JSON/YAML.
La sérialisation/reprise est examinée par source et tests mockés, sans
publier de nouveau dataset. La revue financière distingue explicitement
formules lues, oracles lus, branches seulement inventoriées et sondes exécutées.

Les dry-runs isolés ajoutent un modèle compatible (+76 sorties, 2 projections
modifiées), un produit terminal (+54 sorties, 579 sorties existantes modifiées
par les domaines RNG), puis une composition HW/courbe (+12 sorties,
2 projections). Après restauration dans un répertoire frais, les 1 549 sorties
sont identiques. Ce sont des fixtures de codegen, pas des modèles scientifiques
qualifiés. Le coût RNG d'extension produit exige une migration versionnée
déjà prévue par les digests; il ne justifie pas de rouvrir `STRUCT-011`.

### Performance, FP64 et ressources

| Sous-audit | Examen actuel | Preuve manquante conservée comme limite |
|---|---|---|
| Générique CUDA | MC bloc persistant/prix, closed form thread/bloc, préparation, moments, batching et tuning central | Ressources et débit de toutes les policies, comparaison 1 000/10 000 lignes à 2²⁰ chemins |
| Samples, deux layouts | Threads/paramètre vs bloc/paramètre pour Markov/N-factor; blocs FFT et paires complexes, mémoire hôte/device, replay | Débit et coût complet des deux formes de 3 millions de samples; pas de relance implicite de campagne samples |
| LSM | Préparation, simulation/états SoA, Gram, solve, backward, moments; deux phases résiduelles propres à Kou; batch de lignes gardant tous leurs chemins | Mesures par phase et limites VRAM courantes pour les 19 policies; l'ancien CIR discrétisé ne qualifie pas le CIR terminal-forward |
| Rough et préparation | Quatre phases pricing FFT, noyau samples FFT, fit N-factor CPU et workspaces | Toutes longueurs, ressources/spills et coût de chaque phase; comparaison complète FFT/direct hors mandat courant |
| Pipeline complet | Chargement/fit, allocation, H2D, GPU, D2H, contrôles et sérialisation séparés | Temps processus réel; somme de phases, temps host API et GPU event ne sont pas interchangeables |

`cuda/review.md` classe tous les sites FP64 identifiés depuis l'index lexical
de 22 fichiers : moments MC/FFT/LSM par chemin, finalisation par prix,
discount×cashflow, Gram/RHS par candidat ITM/date, solve par ligne/date,
continuation et décision, correction résiduelle Kou. Il donne fréquences,
volumes (p=4 ou 6, q=15 ou 28 pour les réductions), alternatives et limites.
Fits rough, surcharges de courbes et planification double host sont séparés
du device. Les sommes arithmétiques/range accrual et le résolveur fractionnaire
compensés FP32 restent préservés. NUM-012 est réouvert uniquement pour le
log-spot absorbé, hors domaine fini du sweep géométrique. Les coûts marginaux comparés et
ressources de chaque spécialisation ne sont pas mesurés ici : aucune
optimalité FP64 ni innocuité d'un remplacement FP32 n'est certifiée.

Les sondes fraîches enregistrent toutes les quatre phases du pricing FFT8192
rough Bergomi : 50/72/69/31 registres par thread, mémoire locale déclarée 0,
shared dynamique 65 536/65 536/128/128 octets. Samples FFT16/128/512/2048 :
134/145/145/168 registres, 32 octets locaux/thread et blocs natifs
32/128/128/128 threads. Ces attributs ne mesurent ni coût ni spills SASS;
`localSizeBytes==0` n'est pas une preuve d'absence d'accès local.
Commandes, symboles, géométries et hashes dans `cuda/fresh-probe-resources.json`.

La provenance des 29 cas à 1 000 prix est vérifiée : hashes des binaires,
inputs et quatre artefacts par cas concordants; archive de campagne valide,
228 des 229 sources comparées identiques, seul CMake des tests diffère.
Les quatre anciens CSV Nsight conservent leurs hashes, mais trois exécutables
aux anciens chemins ont été remplacés; seuls les attributs du profil rough
SABR concordant restent rattachables à ce binaire. Aucun profil ancien n'est
silencieusement transféré au nouveau binaire.

`PERF-016` conserve son mandat 2²⁰ chemins/prix, 1 000→10 000 puis projection
à un million, sans réouvrir `PERF-017/019`. Budget conditionnel soumis à
l'utilisateur : onze représentants Heston/Kou terminal, barrière et LSM;
rough Bergomi/Heston terminal et barrière; CIR LSM. Deux warmups et trois
répétitions à chaque taille : environ 6 h 02 GPU et 6 h 40 somme des phases
host, sous hypothèse linéaire, davantage si fit/débit changent. Les 25 cas
stochastiques coûteraient environ 13 h selon la même estimation.
**Aucun accord reçu à la consolidation : campagne non lancée.** Ce budget
n'est ni une mesure du scaling ni une mesure directe à un million.

### Exécutions, résultats et exclusions

Environnement : RTX 4090 Laptop 16 376 MiB, SM89, driver 596.08; CUDA/nvcc
13.3.73, Compute Sanitizer 13.3, CMake 3.22.1, Ninja, GCC 14.3 pour les
builds frais; sonde CPU financière GCC 11.4, `-fno-fast-math -ffp-contract=off`.
Release/O3/NDEBUG pour les targets de production ciblés, sans ccache;
mathDx 26.06 utilisé pour les deux unités FFT isolées.

| Exécution réelle | Résultat |
|---|---|
| Manifestes/codegen Python | 26/26; génération 1 549 sorties sans divergence |
| Publication/reprise datasets, opérations externes mockées | 16/16 |
| Protocole/scaling Python, GPU mocké ou plan-only | 69/69 |
| Checkers layout et catalogue | passent; pas assimilés à une preuve mathématique |
| CTests CPU du build frais | 11/11 : launch plan, catalogue, stages de génération/publication, mémoire hôte, loaders, calendriers |
| ASan/UBSan sur quatre targets CPU frais | 4/4; détection de fuite désactivée explicitement, pas de qualification LeakSanitizer |
| CTests CUDA frais | 7/7 : robustness, pricing runner, résidu LSM, factorisation produits, Bermudan, workspace Volterra, samples Black-Scholes |
| Compute Sanitizer : cinq tests × memcheck/racecheck/initcheck/synccheck | 19/20 sans erreur; Initcheck robustness échoue (padding, `TEST-002`). La contre-épreuve initialisée passe, le test de production reste inchangé |
| Sondes financières CPU/CUDA et loaders réels | quatre contre-exemples confirmés; contrôles usuels concordants |
| Sonde produits native du 10 septembre, un bloc/un thread | Put géométrique absorbé : 0 au lieu de .97; range accrual BS : 1 au lieu de 1.1 pour spot 1.2 et sigma .01; contrôles positifs concordants et oracle normal indépendant |
| Mutant NaN du test runner | code 0 malgré NaN : faux positif `TEST-001` |
| Sondes FFT publiques | batch impair non bitwise (`NUM-026`); reset contexte réussi, hypothèse de défaut rejetée sur ce cas |
| Mutation CMake isolée | build initial passe; ajout d'un loader échoue au lien; reconfiguration seule rétablit le lien (`BUILD-010`); build natif inchangé no-op |

La sélection CTest a été inspectée avant exécution : 341 tests déclarés dans
le build existant, dont 87 principaux et 254 de validation, sans chevauchement
des labels examinés. Les listes exactes, commandes et hashes exécutés sont
conservés. Aucun `assert(...)` désactivable par NDEBUG dans les tests examinés
par scan; cela n'empêche pas le faux positif par comparaison NaN démontré.

Toutes les charges GPU sont sérialisées; aucune compilation pendant leurs
exécutions et aucune mesure de performance qualifiée déduite des durées des
sondes. Les premiers probes FFT liés à libcudart dynamique ont échoué dans
le JIT driver avant lancement; logs/GDB conservés. Le lien statique conforme
au projet fonctionne : ces SIGSEGV ne sont pas des défauts de production retenus.

Exclus : pipelines validation/Premia/QuantLib, régénération de références ou
caches, publication, grandes générations, sweep historique multi-tailles,
GPU concurrent, seuil thermique ajouté, rebaseline, performance longue sans
accord. Pas de qualification financière exhaustive des 25 modèles/26 produits,
de comparaison FP64 exhaustive ni de validation multi-GPU revendiquée.

### Décisions et choix à préserver

**Onze nouveaux constats, NUM-012 réouvert, aucune clôture : 13 ouverts,
92 fermés, 105 identifiants.** Les six anomalies financières (NUM-012,
NUM-022 à NUM-025 et NUM-027) et `STRUCT-026` sont prioritaires; `NUM-026` et `TEST-002` ont un faible impact
observé. [response.md](response.md) porte causes, preuves, corrections proposées
et critères de clôture; [closed.md](closed.md) conserve les signatures historiques
et la décision contradictoire de ce passage.

À préserver : frontières common/model/product, assemblages générés depuis
manifestes, différences justifiées entre exact/Markov/FFT/N-factor/LSM,
loaders PRIVATE et générateurs de paramètres sans mathDx, domaines Philox
versionnés, moments et décisions LSM à précision qualifiée, batching par
lignes sans partition de régression, staging/journal/rollback de publication,
gardes mémoire avant allocation et séparation de la validation indépendante.

Aucune modification de la query proposée : les difficultés rencontrées sont
des défauts ou des preuves manquantes que la v9 permet déjà de distinguer.


## Révision du référentiel en version 9 — 2026-09-09

- **Mandat :** réécriture explicitement approuvée de `query.md`, après revue
  de sa version 8. Il s'agit d'une évolution du référentiel, pas d'un nouveau
  passage de conformité sur le code ni d'une campagne de performance.
- **Révision / worktree :** `872a986b1f0947a1a832af0615ffc6d80dbedb81`,
  branche `main`, index vide; 774 entrées suivies modifiées/supprimées et
  113 non suivies regroupées avant/après. Changements préexistants conservés.
- **Périmètre relu :** query v8, entrées documentaires, contrats et registres
  examinés pendant la méta-revue, protocole de performance, guides tests/tools
  et contrôle documentaire existant. Aucun inventaire mathématique ou test
  runtime supplémentaire n'est revendiqué.
- **Évolution :** neuf axes; justesse modèle/produit/mesure/calendrier explicite;
  qualité des tests distincte de leur nombre; exigences de navigation,
  nommage et en-têtes identiques dans `src`, `tests` et `tools`; pyramide
  conservée avec examen de la sur-factorisation. Couverture et verdict
  deviennent deux dimensions indépendantes; les anciennes tables restent
  historiques. Les changements de titre ne renumérotent aucun constat.
- **Performance :** quatre sous-audits maintenus, inventaire FP64 device,
  phases et ressources compilées, registres/spills, mémoire, budgets et
  statistiques bloquants, références par environnement et rebaseline explicite.
  Le mandat fixe les tailles d'une campagne; aucune ancienne matrice n'est
  automatiquement relancée ni aucun budget de protocole modifié.
- **Documents :** query de 2 049 à 1 199 lignes. SHA-256 avant
  `268faafb5596882275ba4f6147a774cf7ff8990bd57afd3b62a502e74132e552`,
  après `72803627905075b76b0b43430048ec821712a8618c7f6e62ef1263314a78081b`.
- **Vérifications :** dix liens du sommaire et quinze cibles locales valides;
  `PYTHONDONTWRITEBYTECODE=1 python3 tools/cuda/check_model_layout.py` passe;
  `git diff --check` propre. Diff suivi hors `query.md`/`status.md` inchangé,
  SHA-256 `8930fa5f178aa8dc1c1912f96a43c9e39f96ec0775392ce5ed26fdadf3cf408f`.
- **Exclusions :** aucun code, CMake, template, recette, dataset, protocole,
  baseline ou registre de constats modifié; aucun build, test GPU, génération,
  publication ou validation indépendante. Un seul constat reste ouvert,
  `PERF-016`; aucun résultat antérieur n'est requalifié par le passage en v9.

## Recentrage du suivi de scaling — 2026-09-09

- **Mandat :** simplification explicite demandée par l'utilisateur :
  uniquement vérifier si un million de prix coûte environ 1 000 fois
  1 000 prix, à 2²⁰ trajectoires par prix. Mise à jour des registres,
  pas nouvel audit principal ni campagne de performance.
- **Révision / worktree :** `872a986b1f0947a1a832af0615ffc6d80dbedb81`,
  branche `main`, index vide; 774 entrées suivies modifiées/supprimées et
  113 non suivies regroupées à l'entrée. Changements préexistants conservés.
  Aucun nouveau snapshot de code audité : seules les trois pages de registre
  sont modifiées, les preuves restent liées à leurs snapshots historiques.
- **Mouvement :** `PERF-016` reste ouvert sous son identifiant stable et
  porte MC terminal, barrière et LSM. `PERF-017` et `PERF-019` sont déplacés
  dans `closed.md` comme fusionnés, sans validation de leurs anciennes
  matrices. Total : **un constat ouvert, 93 fermés, 94 identifiants**.
- **Signature antérieure de PERF-016 :** la qualification terminale ne couvre
  pas chaque modèle et méthode à 65 536, 262 144 et 1 048 576 trajectoires par
  prix, croisés avec 1, 16 et 1 000 prix. Le moteur Markov/N-facteurs ne permet
  qu'un bloc par prix; `validate_block_count` interdit de dépasser le nombre
  de prix, même lorsque les trajectoires augmentent et que les petits lots
  ne remplissent pas le GPU. Le moteur FFT a une topologie distincte à
  qualifier séparément. L'extension du 2026-09-07 portait ensuite les nombres
  de prix à 100/1 000/10 000. Le présent mandat retire ce balayage exhaustif,
  sans prétendre que toutes ses cases ont été mesurées ou cette topologie
  modifiée.
- **Couverture relue :** les trois entrées ouvertes, le registre fermé, les
  mesures de scaling du 2026-09-07 et la confirmation de 29 cas du 2026-09-08.
  Les résultats historiques sur tuile répétée, les changements CIR/Kou et
  les réserves statistiques ne sont ni effacés ni requalifiés.
- **Critère courant :** comparer le débit à 1 000 et 10 000 prix à 2²⁰
  trajectoires, distinguer GPU et génération complète, puis étayer ou réfuter
  l'extrapolation à un million. Un palier supplémentaire n'est demandé que
  si nécessaire; aucune mesure directe à un million n'est revendiquée.
- **Exclusions :** aucun code, réglage, recette, dataset, baseline ou rapport
  de mesure modifié; aucun build, test GPU, génération, publication ou audit
  indépendant de validation. `query.md` reste inchangé : le mandat ciblé
  n'abaisse pas les exigences générales d'un futur audit.
- **Contrôles :** identifiants uniques et disjoints entre registres, liens
  des entrées modifiées et `git diff --check`; empreinte du diff suivi hors
  des trois registres inchangée :
  `38b11e56328c52923f0cfcfaa9acd6d272e5f179acd1fc858b5a1947b7159918`.

## Reprise bornée des configurations de lancement — 2026-09-08

- Mandat utilisateur : résoudre Kou, terminer la campagne et les pilotes,
  sans refonte de `src` ni prolifération de tests. Passage ciblé, pas audit
  principal exhaustif; validation indépendante et publication restent exclues.
- Révision `872a986b1f0947a1a832af0615ffc6d80dbedb81`, `main`, index vide;
  768 entrées suivies modifiées/supprimées, 111 non suivies regroupées.
  Snapshot `build-dev/kou-lsm-launch-confirmation.zpDtZU` : diff SHA-256
  `37c62710075c630dbc7df53b47d5153a664ba71f2eed58a040fe4c6f19af9c24`,
  archive `88c8ac9941bb52c6af29119c8e072bdb1613b6365d72fb4bb0e554d2513d4a84`.
- `NUM-017` réouvert selon sa condition historique (conditionnement supérieur
  au sweep); la réserve numérique de `PERF-019` lui est rattachée sans nouveau
  numéro doublon. Prototype résiduel toujours hors production à l'ouverture.
  Six géométries sont préplanifiées sur les 1 000 lignes à `2^20` chemins.
- Compilation agrégée précédemment interrompue proprement, objets conservés.
  Aucun timing pendant une compilation, aucun seuil thermique, aucune relance
  automatique pour sélectionner une période favorable. État partiel tant que
  les vérifications et pilotes annoncés ne sont pas achevés.
- Correction résiduelle intégrée uniquement pour Kou, sans autre moteur ni
  changement de base/ridge/Philox. Six géométries du prototype et le launcher
  intégré retrouvent les 1 000 prix/erreurs binary128 bit à bit. Deux tests
  ciblés, memcheck/racecheck et 40 tests de protocole passent; codegen 1 548
  sorties zéro-diff. Build agrégé en cours à deux workers, nouveaux CTests et
  pilotes natifs encore attendus. Le CV du premier timing intégré (5.0028%)
  est conservé non qualifiant, sans relance automatique.
  [Correction, ressources et preuves](../performance-reports/catalogue-generation-readiness-sm89-2026-09-08.md).
- `NUM-017` refermé : les trois processus corrigés supplémentaires retrouvent
  les 1 000 références, les trois anciens reproduisent les huit écarts.
  Ordre avant/après fixé, coût par paire 1.672/1.355/1.461; CV 5.50% au premier
  passage corrigé, conservé non qualifiant. Profil séparé : 4.242 s de résidu,
  0.069 s de correction du système. Aucune compilation pendant ces mesures.
- Les trois nouveaux CTests ciblés passent. `BUILD-009` ouvert puis fermé
  après adaptation du pointeur de launcher OU, qui arrêtait l'agrégat à
  l'étape 1603. Reprise incrémentale du build, objets conservés. Les 50
  générateurs samples passent leur smoke-test à 1 000 lignes; effectués
  pendant compilation CPU, leurs timings ne sont pas qualifiants.
  Pilotes natifs complets et campagne de 29 couples encore en attente du build.
- Build agrégé achevé, vérification Ninja sans travail restant; CTest principal
  **85/85**, sans label validation. Les 50 binaires samples courants passent
  le smoke-test (46 revérifiés après reconstruction, quatre inchangés identifiés
  par hash). Les 426 identités de pricing passent 852 inspections CPU à
  1 000/1M lignes; MC/LSM à `2^20`, sans prétendre vérifier la VRAM hors GPU.
  Campagne de production démarrée à 19:55 UTC : 29 couples, les 1 000 lignes
  catalogue ordonnées, deux warmups exclus puis trois mesures, sans compilation
  concurrente. Campagne achevée : **29/29**, sorties finies et replay
  déterministe; le Kou courant retrouve aussi les 1 000 références binary128.
  CIR++ LSM conserve sa réserve de puissance; G2++/HW caplets et Kou terminal
  conservent leurs CV GPU > 5%, sans relance. Pas de nouveau profil optimal
  accepté. Snapshot de campagne : diff
  `8db431ffea77d003ae6c7aa353eea1191a5e99423b6e27dd3d2049c869a01172`, archive
  `e0a954ef5a8ca1a0ac1663ca46153b7de3a80bdd78e4dc03d1ea9e10c9482703`.
  [Export complet](../../tests/performance/reports/pricing-dataset-runtime-sm89-2026-09-08.json).
  Sept pilotes natifs sans publication restent en cours.
- Premier pilote : cinq prix et Kou conditionnel (3M samples) réussissent;
  Kou inconditionnel est refusé par la garde hôte. `STRUCT-024` ouvert :
  `_SC_AVPHYS_PAGES` ignore le cache récupérable (environ 7.2 Gio disponibles,
  288 Mio libres). Correction limitée à `tools/sampling` : `MemAvailable`,
  repli conservateur, contrôle avant allocation, seuils inchangés. Test CPU
  passé; nouvelle campagne samples nécessaire avec les binaires corrigés,
  sans modifier ceux figés dans la tentative refusée.
- Lot achevé : `STRUCT-023/024` fermés. Sept targets natifs courants passent,
  destinations canoniques inchangées; reprise des deux samples complets sans
  recalcul. Le sample conditionnel garde exactement son corps avant/après;
  l'inconditionnel rejoue les 3M lignes entre 256/128 threads. Contrôle CPU de
  la RAM passé; source de la correction archivée avec les preuves des pilotes.
  Build agrégé à jour, **86/86 CTests principaux**, **50/50 smoke-tests samples
  sur les binaires recompilés**. Le notebook est exécuté, son export complet
  contient les 29 points et toutes leurs réserves. `git diff --check` propre;
  trois identifiants ouverts, 91 fermés, uniques et disjoints.
  État final observé : même révision, index vide, 774 entrées suivies sales et
  113 non suivies regroupées; changements préexistants conservés. Validation
  indépendante, publication catalogue, multi-architecture, rebaseline et
  matrice multi-tailles complète restent exclus. `PERF-016/017/019` restent
  ouverts; la préparation de la génération à 1 000 lignes ne les clôt pas.

## Préparation des générations — 2026-09-08, passage ciblé en cours

- Mandat d'implémentation approuvé, pas nouvel audit exhaustif. Révision
  `872a986b1f0947a1a832af0615ffc6d80dbedb81`, branche `main`, index vide;
  763 entrées suivies modifiées/supprimées et 105 non suivies regroupées à
  l'entrée. Worktree partagé et changements préexistants conservés. Snapshot
  initial : diff SHA-256 `915faddf047a6b815aa28d1211fa0dccb56463e9c846f2ffd6ce3f59a1a0b5de`,
  archive des sources modifiées/non suivies
  `c1884a860889dd0932f6a5328fd99ac134da503005a4e785de249f6d129338c6`.
  Le commit seul ne reconstruit pas ce contenu.
- Historique relu avant consignation. `NUM-021` corrigé et fermé : somme
  compensée et racine ancre/déplacement FP32, seuil de résidu inchangé.
  Deux CTests CUDA passent, dont les stress et leurs voisins contre référence
  CPU locale. La matrice CIR/Vasicek payer/receiver, scalaire/coopératif,
  128/256/512 threads, grilles denses/persistantes passe 48/48 sur 1 000 lignes.
  [Preuves brutes, ressources et synthèse](../../tests/performance/reports/generation-readiness-sm89-2026-09-08/jamshidian-correction.json).
  Aucun FP64 device ajouté ni spill observé dans le banc. Anciennes mesures
  invalides conservées; pas de nouveau gain ni de retuning revendiqué.
- `STRUCT-022` ouvert puis fermé pour les métadonnées émises par les générateurs :
  propriétaire commun, statut pending/false et référence canonique. Deux
  tests host passent; le dernier cas anti-écrasement est recompilé séparément
  contre les archives courantes et passe également. Aucun ancien YAML modifié.
- `STRUCT-023` ouvert pour le contrôleur de campagne : 618 recettes prix et
  50 samples inventoriées; staging, snapshots, reprise explicite et sauvegarde
  de publication implémentés. Seize tests Python passent. Les pilotes natifs
  du nouveau contrôleur restent à effectuer, sans publication du catalogue.
- Kou LSM : deux prototypes de compensation FP64 stabilisent les quatre
  lignes/vingt géométries diagnostiques, mais augmentent temps et registres.
  Ils vivent hors du source de production; ni base, ridge, tolérance,
  géométrie ni mapping Philox modifiés pour masquer les écarts. Le choix
  utilisateur est d'approfondir la recherche de surcoût réduit, pas de
  promouvoir ces prototypes. `PERF-019` reste ouvert.
- Build agrégé paramètres/prix/samples/tests en cours, un seul worker.
  Pas de timing comparatif pendant la compilation. `PERF-016/017` restent
  ouverts; aucune clôture de scaling ne découle des tests locaux.
- Exclusions : validation indépendante et ses caches, publication générale,
  retuning/rebaseline, autre GPU/toolchain et exécution 1M prix × `2^20`.
  Le statut global reste partiel : couverture et preuves limitées à ce lot,
  étapes en cours explicitement non revendiquées comme réussies.

## Planificateur de pricing et précision de production — 2026-09-08

- Lot d'implémentation ciblé approuvé, pas nouvel audit exhaustif. Révision
  `872a986b1f0947a1a832af0615ffc6d80dbedb81`, `main`, worktree partagé déjà
  modifié (154 entrées suivies, 102 non suivies regroupées avant ce lot),
  index vide. Les changements préexistants sont conservés. Le grand diff de
  recettes provient de leur régénération depuis les templates, pas de centaines
  de réglages indépendants; les nouveaux fichiers et les diffs sont nécessaires
  à la reproduction, le commit seul ne décrit pas cet état.
- Propriétaires : `tools/cuda/tuning_profile.hpp` pour les valeurs et candidats,
  `pricing_launch_plan.hpp` pour l'arithmétique hôte et `inspect_pricing_launch_plan`
  pour l'inspection CPU. L'inventaire des familles est dérivé des 427 bindings
  codegen. Cinq pipelines de pricing et les recettes fermées consomment ce
  plan; aucune table de géométries n'est recopiée dans le contrôleur Python.
- Production MC/LSM : constante unique `2^20 = 1 048 576` par prix. Aucun
  cap de blocs, découpage de prix ou chunk FFT ne réduit ce nombre. LSM conserve
  son planificateur VRAM natif; FFT conserve ses grilles compilées. L'inspecteur
  ne prétend vérifier ni VRAM ni ressources device et signale les inconnues.
  Les anciennes données et leurs YAML à 16k/65k ne sont pas modifiées : leur
  migration exige une vraie régénération. Samples, warmups et tests gardent
  leurs charges distinctes.
- Jamshidian : sélection hôte des deux kernels existants, capacité chargée
  transmise uniformément; suppression de l'ancienne configuration locale
  redondante. Candidats catalogue/grande taille consignés, pas de seuil optimal
  inventé entre tailles. CIR et Vasicek conservent leur géométrie native
  pendant `NUM-021`. Aucune formule, tolérance, précision, réduction ou clé
  Philox n'est changée par ce lot.
- Vérification : 26 tests de manifeste, 40 tests Python de scaling, test hôte
  du planificateur (queues, grands comptes, overflow, VRAM inconnue et chunks),
  722 recettes vérifiées et 1 548 sorties codegen zéro-diff. Builds séquentiels
  de recettes représentatives des familles MC, LSM, FFT, N-facteurs, formules
  fermées et courbes; tests CUDA runner, swaptions one-factor scalaire/coopératif
  et CIR++ réussis. `git diff --check` propre. Pas de compilation exhaustive
  des 722 recettes ni de qualification multi-architecture revendiquée.
- Contrôle du mode `production` : Kou terminal à 1 000 prix réels × `2^20`
  chemins, 512 threads × 1 000 blocs, deux warmups puis trois répétitions,
  replay déterministe et publication locale temporaire réussis. CIR caplet
  échoue initialement dans le parseur du banc, qui imposait des trajectoires
  aux formules fermées; garde corrigée, seul CIR est repris et passe à 1 000
  prix, 256 threads × 4 blocs. La tentative échouée reste conservée.
  [Export initial](../../tests/performance/reports/pricing-launch-plan-sm89-2026-09-08/initial-check.json)
  et [contrôle CIR corrigé](../../tests/performance/reports/pricing-launch-plan-sm89-2026-09-08/cir-check.json)
  contiennent plans, hashes, échantillons, ressources et statuts distincts.
- Timings diagnostiques seulement (`observe`, `power.limit` non vérifiable),
  ni retuning qualifié ni rebaseline. Pas de nouvelle grande campagne, de
  publication catalogue, de reprise transactionnelle des générations,
  d'audit de validation indépendante ou d'exécution 1M prix × `2^20` chemins.
  `PERF-016`, `PERF-017`, `PERF-019` et `NUM-021` restent ouverts; la sensibilité
  Kou LSM n'est pas corrigée par un changement de géométrie masquant ses écarts.

## Expérience Jamshidian one-factor — 2026-09-08, achevée avec limites

- Suivi ciblé demandé par l'utilisateur, pas nouvel audit exhaustif. Trois
  constats ouverts pendant le passage : `PERF-022` (expérience de stratégie)
  et `BUILD-008` (gardes d'inclusion de courbes), maintenant fermés avec preuves;
  `NUM-021` (prix non finis), laissé ouvert. Les constats
  MC/LSM antérieurs ne changent pas d'état. Les clôtures de `NUM-001`,
  `PERF-006` et `PERF-018` ont été relues; signatures distinctes.
- Révision `872a986b1f0947a1a832af0615ffc6d80dbedb81`, `main`, worktree
  partagé modifié. À l'ouverture de cette consignation : 154 entrées suivies
  modifiées/supprimées, 102 entrées non suivies regroupées, index vide.
  Les snapshots de chaque campagne conservent leur propre état antérieur,
  diff, empreintes des sources (y compris non suivies), inputs et binaire.
  La révision seule ne reconstruit pas ce contenu. Aucun changement
  préexistant n'est annulé.
- Périmètre : swaptions européennes régulières CIR, CIR++, Hull–White,
  Ornstein–Uhlenbeck, Vasicek; Nelson–Siegel et Svensson pour les deux modèles
  ajustés. 100/1 000/16 384/65 536/262 144/1 048 576 prix, 64/128/256/512
  threads, grilles denses ou persistantes, profils catalogue/court/long.
  Les grands buffers répètent les lignes alignées originales; ce ne sont pas
  un million de paramètres indépendants. Le préfixe de 100 lignes ne représente
  pas la composition complète core/stress et ne sert pas à prouver sa linéarité.
- Achevés : pilote 56 configurations (`jamshidian-strategy-pilot-03`),
  screening 588 (`*-screen-02`), profils 364 (`*-profiles-01`), grandes tailles
  280 (`*-large-01`). Confirmations : trois campagnes payer indépendantes
  de 84 configurations et un contrôle receiver de 84 configurations, cinq
  warmups puis 21 mesures. Total retenu : 1 624 mesures de configurations,
  dont 1 278 `passed` et 346 `reference_incomplete`. Les trois confirmations
  donnent 51/84 géométries payer satisfaisant les gates locaux numérique/CV;
  les autres restent inéligibles. Ni retuning ni rebaseline effectués.
  Les essais initiaux interrompus sont conservés et exclus : référence CIR
  non finie dans `*-pilot-01/02`; fausse détection de concurrence WSL dans
  `*-screen-01`, corrigée par vérification du PID, parent et exécutable.
  Receiver a nécessité une continuation explicite : CIR terminé dans
  `*-confirm-receiver-01`, puis les six autres compositions dans
  `*-confirm-receiver-resume-01`. Une course de suivi sur le PID déjà terminé
  avait interrompu le contrôleur après la sortie sans erreur CUDA de CIR++;
  cette mesure n'est pas retenue. Le suivi mémorise désormais l'identité
  préalablement vérifiée et distingue sa sortie d'un processus étranger.
- Kernels/formules/recettes inchangés. Le banc instancie les policies et
  kernels communs existants; il ne mesure pas la sérialisation/publication du
  générateur natif. Seules deux gardes `#pragma once` ont été ajoutées aux
  implémentations de courbes pour permettre leur inclusion commune.
- Les rejets de `NUM-021` ne sont pas transformés en succès : les comparaisons
  conservent `reference_incomplete`, distinct de `passed`, et interdisent la
  qualification numérique. La tolérance de comparaison entre modes, fixée
  avant le pilote, n'est pas une certification indépendante du prix.
- Plateforme : RTX 4090 Laptop/SM89, CUDA 13.3, build Release, compilation
  séquentielle avant campagnes. Température en télémétrie seule, secteur,
  concurrence, power-brake et watchdog conservés; aucune relance automatique
  visant à obtenir un CV favorable. Aucun réglage matériel n'est changé.
- Exclusions : autres produits, calendriers explicites ELLPACK, MC, LSM,
  samples, validation indépendante, autres GPU, publication et rebaseline.
  Ce passage reste partiel pour les sections principales de la query.
- Livrables : [rapport](../performance-reports/jamshidian-strategy-scaling-sm89-2026-09-08.md),
  tableaux CSV, configurations, échantillons bruts et provenance consolidée
  sous `tests/performance/reports/jamshidian-strategy-sm89-2026-09-08`.
  Binaire mesuré conservé dans `build-dev/jamshidian-strategy-snapshot-01`,
  SHA-256 `154f173bf24d2a744463e2de4d929b1d9839b703fc672146815be27cefb5bce3`;
  archive source/inputs
  `4261bb32f353aca4249358ae13739d660f6c47a6e5bb0c7abbd33d6b695b1481`.
- Vérification finale : ajout de la valeur effective des lignes à référence
  invalide dans le diagnostic et formatage du banc, sans changement des
  kernels. Le build final réussit et 56 contrôles courts terminent; les sept tuiles de
  référence restent identiques au binaire mesuré. Ces contrôles ne remplacent
  aucun timing de confirmation. Les sept lignes de `NUM-021` sont non finies
  dans les deux modes payer. Dix tests Python et `model_source_layout` passent;
  `git diff --check` propre. Registres 37–85, aucun stack/local ni instruction
  SASS LDL/STL; compteurs dynamiques Nsight non mesurés.

| Gate | État de ce suivi ciblé | Preuve / limite |
|---|---|---|
| P0-SNAPSHOT | pass | Binaire mesuré et archive source/inputs conservés, hashes et états Git par campagne. |
| P0-INVENTORY | pass | Sept compositions dérivées du manifeste canonique, test de couverture. |
| P0-SCOPE | pass | Banc dans tests/performance, pilotes dans tools/performance. |
| P0-COVERAGE | partial | Matrice expérimentale achevée; pas de qualification générale, exclusions explicites. |
| P0-IDENTITY | pass | Registres relus, signatures comparées avant nouveaux IDs. |
| P0-DOCS | partial | Rapport livré et indexé, pas audit documentaire général. |
| P0-NUMERICS | partial | NUM-021 et absence de certification indépendante générale. |
| P0-PERF-STATS | partial | Trois confirmations conservées intégralement; bruit et dérive inter-campagnes non effacés. |
| P0-REBASELINE | excluded | Aucune baseline remplacée. |
| P0-EVIDENCE | pass | Bruts, ressources, synthèse, provenance, limites et snapshot de l'expérience conservés. |

## Douze couples mesurés à 1 000 prix réels × 2²⁰ trajectoires — 2026-09-07

- Mandat du 2026-09-07 : présenter les douze couples représentatifs à la même
  charge, **1 000 prix × 1 048 576 trajectoires par prix** pour MC/LSM,
  1 000 prix sans trajectoires pour le caplet CIR fermé. Aucune substitution
  par une charge plus petite ni estimation présentée comme mesure manquante.
- Révision `872a986b1f0947a1a832af0615ffc6d80dbedb81`, `main`, worktree partagé
  initialement modifié : 116 chemins suivis, 45 entrées non suivies regroupées,
  index vide. Les changements préexistants sont conservés. Ce passage modifie
  l'outillage d'inputs/export, ses tests, la documentation et le notebook de
  présentation; pas les kernels, profils de production, datasets publiés,
  tolérances, baseline ou validation indépendante.
- Nouveau profil `ordered_catalogue_900_core_100_stress_v1` : les 1 000 lignes
  originales sont conservées dans leur ordre, avec leurs identifiants,
  paramètres, calendriers et clés de ligne. Inputs temporaires séparés; le
  profil historique de tuile répétée reste inchangé. Les tests refusent une
  autre taille et l'export refuse aussi 1 000 000 au lieu de 2²⁰ trajectoires.
- Les douze sondes sont compilées avant calcul, neuf mises à jour en séquentiel.
  Départ à **20:35:23 UTC** dans
  `build-dev/pricing-dataset-runtime-20260907-01`; plan préétabli dans le
  dossier adjacent `*-plan-01`. Snapshot diff SHA-256
  `6fdc8b64fe32f387a08beb5480590b9a909b680dd2d85dd385451033c69ad49b`,
  archive `7f6cccbc89eaa29db0e59650bb01217a7d696e22d71586f82b790553e652e0a1`.
  Ordre explicite rough Heston, rough Bergomi, Heston, Kou, CIR; une géométrie
  native par couple, un warmup complet exclu puis trois mesures. Le caplet
  groupe 1 024 appels par échantillon, normalisés à un seul dataset.
- Contrôleur séquentiel et surveillé, sans concurrence GPU, veto thermique,
  attente de convergence ou relance automatique. Les excursions de puissance
  restent visibles et excluent le retuning sans arrêter l'expérience.
  Watchdog de 3 600 s par cas; budget de planification 60–90 minutes.
- Export portable et [notebook](../performance-reports/pricing-dataset-runtime-sm89.ipynb)
  distinguent états terminés/manquants, GPU, hôte brut, préparation, copie et
  publication locale. La somme des phases estime la génération d'un dataset;
  elle ne revendique pas un chronomètre de processus à froid, un upload ou une
  certification indépendante. Aucun double comptage GPU + API hôte.
- Résultat : **12/12 cas terminés**, 36 échantillons normalisés, replay
  numérique réussi et publication locale temporaire sur chaque couple.
  Les quatre couples rough sont mesurés à la charge cible, sans réutiliser
  les anciens résultats à 64k. Aucun processus compute ne reste actif après
  la campagne. Les douze binaires et les 24 tableaux d'inputs originaux ont
  été revérifiés; les sources de production archivées n'ont pas changé.
  Heston/Kou/CIR LSM utilisent respectivement 32/16/4 batches VRAM.
  Une excursion de puissance
  exclut rough Bergomi terminal du retuning; les CV GPU rough Heston terminal
  et Kou terminal dépassent 5 %. Les mesures restent conservées et affichées.
- CIR LSM : 2 743,2 s pour le processus complet, warmup et répétitions inclus;
  médiane GPU 718,349 s, hôte brut 686,424 s. CV GPU 0,0095 %, hôte 0,095 %,
  workspace 14,154 Go, aucune régression fatale; prix/erreurs reproduits entre
  répétitions. Génération estimée par somme des phases : 686,746 s pour ce
  dataset, **1 117,030 s (18 min 37 s)** pour un dataset de chacun des douze
  couples. Ce total n'est ni celui du catalogue entier ni une mesure de
  processus à froid. Les écarts chronomètres GPU/hôte de 1,9 à 6,7 % restent
  affichés, sans cause attribuée ni précision absolue de 1 % revendiquée.
- Vérifications hôte : 68 tests Python et deux CTest ciblés passent;
  `git diff --check` propre. L'exécution CPU du notebook nécessite les ports
  Jupyter locaux, indisponibles dans le sandbox strict; notebook exécuté sans
  erreur hors sandbox, sans travail GPU supplémentaire. Les cellules restent
  CPU seulement et refusent tout total qui inclurait une mesure manquante.
  Le JSON portable est complet (12/12); la synthèse multi-tailles générique
  reste non qualifiante pour la matrice entière, qui dépasse ce mandat borné.
- `PERF-016`, `PERF-017`, `PERF-019` restent ouverts : une charge unique ne
  démontre pas la linéarité multi-tailles. Sensibilité LSM Kou aux géométries
  non corrigée. Autres couples, samples, autres GPU, validation indépendante
  et exécution effective 1M prix × 2²⁰ trajectoires exclus de ce passage.

## Confirmation achevée et diagnostic Kou instrumenté — 2026-09-07

- Révision `872a986b1f0947a1a832af0615ffc6d80dbedb81`, `main`; état initial
  inchangé : 116 chemins suivis modifiés/supprimés, 45 entrées non suivies
  regroupées, index vide. Ce passage ajoute du suivi, des sondes temporaires
  sous `build-dev` et la capture du temps hôte brut dans l'outillage de scaling,
  avec ses tests et son protocole. Aucun changement du runtime, des réglages,
  datasets publiés, tolérances, baseline ou validation indépendante.
- `pricing-scaling-20260907-lsm-confirmation-01` terminé : 20/20 jobs,
  deux warmups et trois répétitions; 18/18 parités bitwise, aucun diagnostic
  fatal. Tous les CV GPU restent sous 5 %. CIR : 1 669,7 s de processus;
  Heston : 112,8 s, soit environ 30 minutes au total. Les sources de la sonde
  Kou ont été préparées pendant l'attente, mais compilées/exécutées seulement
  après la sortie complète du lot, sans concurrence GPU.
- Heston ne déclenche aucune alerte sur les trois comparaisons couvertes.
  CIR déclenche deux alertes : coûts normalisés 1,336 (256k → 1M à 100 prix)
  et 1,396 (100 → 1 000 prix à 64k). Les fréquences passent d'environ
  2 325 à 1 590 MHz; ces alertes ne prouvent donc pas une cause algorithmique.
  À 1 000 prix × 64k, CIR 64/1 024 blocs donnent 43,7/44,2 s : pas de gain
  justifiant un réglage global. Heston n'obtient pas de gain GPU en découpant
  en batches de 100, et le temps API augmente. Détails dans le rapport.
- `pricing-scaling-20260907-kou-regression-trace-01` : 20 traces, moteur de
  production réutilisé avec instrumentation temporaire des policies. Les
  20 sorties reproduisent bit à bit le launcher public; tous les états forward
  et payoffs terminaux sont comparés intégralement, bit à bit, entre géométries.
  Les quatre lignes appartiennent au core, pas seulement au stress.
- Diagnostic établi sur ces lignes : différences relatives des matrices de
  Gram de l'ordre de `1e-16`, conditionnement après ridge jusqu'à `6e10`,
  coefficients déjà différents au premier niveau backward; les différences
  se propagent ensuite au second membre/cashflows et au prix final. Résidu
  backward normalisé maximal du solveur : `1,15e-16`. Les trajectoires et le
  batching sont exclus comme cause de ces écarts observés; la sensibilité des
  régressions aux arrondis est instrumentée. Cela ne justifie ni une baisse de
  précision, ni une nouvelle base/ridge, ni une tolérance relevée sans validation.
  Code source, commandes et empreintes sous `*-kou-regression-trace-plan-01`
  et `*-kou-regression-trace-build-01`; brut et analyse dans le lot de trace.
- Contrôle CIR achevé dans `pricing-scaling-20260907-cir-lsm-order-control-01` :
  départ à 18:27 UTC, diff SHA-256
  `e5e1bca6d5df4e85e7eeea338c05180be50e42d0192e953bacdaeb5e3c234810`,
  archive `8c30134876bc5fe57c51ee5fa3b010c13ee3620ae0387dc234b7b8e0a2aad0d9`;
  quatre jobs en ordre inverse (100 prix à 1M, 256k, 64k, puis 1 000 prix à
  64k), uniquement la géométrie native 128 × 64. Deux warmups et trois mesures
  fixes; aucune attente ou convergence thermique. 4/4 jobs en 654,1 s,
  tous les CV GPU sous 2,2 %. Coûts normalisés : 0,987 (64k → 256k),
  0,974 (256k → 1M), 0,924 (100 → 1 000 prix). Les alertes initiales ne
  se reproduisent pas dans cette période, sans modification de code : résultat
  compatible avec l'effet du régime de fonctionnement, pas preuve d'une
  linéarité universelle. Les quatre sorties prix/erreurs égalent bit à bit les
  références natives de la confirmation. Les deux périodes sont conservées.
- Extension Heston achevée dans `pricing-scaling-20260907-heston-lsm-large-01` :
  départ à 18:40 UTC, diff SHA-256
  `0b8ae48656d1c7c55d0b882d2aa92a3f815a9a0169535c9f0c20f546bdccb5aa`,
  archive `ef09da96fe3a37d4e5975e97b6d4162d19f51ae8be71e129658d19cbeabd4cf2`;
  six nouveaux jobs à 1 000 prix × 256k/1M puis 10 000 prix × 64k, deux
  géométries par forme, deux warmups et trois répétitions. Plan explicite sous
  `*-heston-lsm-large-plan-01`; binaire et inputs identiques à la confirmation.
  Un appel hôte et batching VRAM natif; 6/6 jobs en 655,5 s, trois parités
  bitwise, tous les CV GPU sous 2 %. À 1 000 prix : 8,734/8,913 s pour
  256k, 33,147/33,821 s pour 1M selon la géométrie. À 10 000 × 64k :
  27,035/23,374 s (128/64 blocs). Respectivement 8/32/19 batches natifs,
  environ 14,2 Go de workspace. Une excursion 175 → 150 W rend le lot
  non qualifiant pour le retuning, sans interrompre ses calculs. Aucune
  relance ni sélection entre périodes; tous les résultats sont conservés.
- Inspection hôte des ressources compilées : les sept kernels des trois cas
  n'ont ni stack/local ni instructions SASS `LDL`/`STL`; les statistiques LSM
  utilisent néanmoins 88/76/86 registres par thread pour Heston/CIR/Kou.
  Le profilage par phase reste nécessaire, l'occupation théorique ne suffit pas.
  Limite d'outillage corrigée sous `PERF-019` après la fin des mesures : la sonde
  conserve désormais `raw_host_clock` et `raw_host_samples_ms` séparément de
  l'enveloppe API `max(host, CUDA)`. Les anciens bruts restent inchangés et la
  synthèse indique `null` pour leurs horloges manquantes, sans les reconstituer.
- Vérification `pricing-scaling-20260907-raw-host-clock-check-01` : trois
  sondes recompilées séquentiellement, puis trois jobs courts CIR LSM / Heston
  LSM / Heston terminal MC en observation seule. Prix et erreurs bitwise
  identiques aux références préchangement, ressources compilées identiques,
  échantillons et statistiques hôte vérifiés. 61 tests Python, deux CTest et
  `git diff --check` passent. Départ à 18:53 UTC; diff SHA-256
  `1d6c6e47738b72689f0085ced5099b8f0707d31adf6fd459172e9eac87c8e7cf`,
  archive `172ee2405e9ba70e6e071fc373c324990c5277872941188c21e35643372f41d5`.
  Toutes les campagnes de ce passage sont terminées, aucun job laissé actif.
- `PERF-019` reste ouvert : correction de la sensibilité Kou, contrôle des
  grandes charges CIR, qualification temporelle des grands lots Heston,
  10 000 prix à 256k/1M et profilage par phase/pipeline complet.
  Autres modèles, samples, autres GPU, validation indépendante et exécution
  effective 1M × 1M restent hors de ce passage. Aucun retuning accepté.

## Poursuite LSM : confirmations et écarts Kou — 2026-09-07

- Révision inchangée `872a986b1f0947a1a832af0615ffc6d80dbedb81`, `main`;
  état initial : 116 chemins suivis modifiés/supprimés, 45 entrées non suivies
  regroupées, index vide. Travail préexistant conservé. Modifications de ce
  passage : synthèse/tests/protocole et suivi, pas de kernel, profil de
  production, catalogue publié, baseline ou validation indépendante.
- `*-representative-lsm-path-screening-01` achevé : 18/18 jobs à 100 prix ×
  262k/1M, trois candidats par forme. Heston et CIR passent toutes les parités;
  Kou échoue sur cinq comparaisons sur six, sans valeur non finie ni régression
  fatale. À 1M, le planner réalise quatre batches Heston et deux Kou; CIR
  tient en un batch. Les mesures uniques ne qualifient pas le CV.
- Diagnostic ciblé `pricing-scaling-20260907-kou-lsm-numerics-01` : 16 jobs,
  quatre lignes × quatre géométries, un warmup et trois répétitions. Replay
  bitwise dans chaque configuration; 12 comparaisons avec les valeurs du
  batch complet sont bitwise identiques. L'écart dépend donc de la géométrie,
  pas du seul découpage en prix. 17/24 parités inter-géométries échouent;
  la sensibilité des réductions/régressions reste à instrumenter. Kou n'est
  pas qualifié et aucune tolérance n'a été relevée.
- Faiblesse de synthèse corrigée sous `PERF-019` : les conflits numériques
  excluent désormais les deux mesures de l'enveloppe qualifiante et du verdict
  temporel. Offsets distincts conservés; 60 tests Python et deux CTest ciblés
  passent, `git diff --check` propre. Les synthèses initiales restent intactes;
  résultats réanalysés dans `summary-numerical-gate-02.json`.
- Confirmation Heston/CIR lancée dans `pricing-scaling-20260907-lsm-confirmation-01` :
  départ à 17:50 UTC; snapshot diff SHA-256
  `3f8e00eb05d7444bada3a70fa01520b5d732b6379a052543e656ef14d1dced16`,
  archive `17e3d536e769c90b161d5030ad612b010131a0eb9dc73f569b6e5a5262b39608`.
  20 jobs, deux warmups et trois répétitions. À 100 prix, les trois tailles
  de trajectoires; à 1 000 prix × 64k, deux géométries croisées avec batches
  hôte de 100 et 1 000, plus le batching VRAM natif. Références vérifiées dans
  CMake et les recettes : equity 128 × 128 blocs/prix, CIR 128 × 64.
  Plan conservé dans `*-lsm-confirmation-plan-01`, budget estimé 25–35 minutes,
  watchdog 3 600 s par modèle, aucune relance automatique. Le contrôleur
  a enregistré la télémétrie et produit sa synthèse; le lot est maintenant
  achevé, résultats détaillés ci-dessus. Binaires et inputs des screenings inchangés.
- `PERF-016`, `PERF-017`, `PERF-019` restent ouverts. Encore exclus de cette
  étape : 1 000 prix aux deux grands nombres de trajectoires, 10 000 prix,
  profils par phase, retuning Kou, autres modèles, samples, autres GPU,
  validation indépendante et calcul effectif 1M × 1M. Pas de gain accepté
  ni de rebaseline à partir des mesures exploratoires.

## Correction du contrôleur et priorité LSM — 2026-09-07

- Révision `872a986b1f0947a1a832af0615ffc6d80dbedb81`, branche `main`;
  état initial : 116 chemins suivis modifiés/supprimés, 44 entrées non suivies
  regroupées, index vide. Changements préexistants conservés; nouveau module
  partagé `experiment_environment.py` ajouté aux modifications de l'outillage.
- `PERF-021` ouvert, corrigé puis fermé : variations de limite de puissance
  sans arrêt applicatif dans les deux pilotes exploratoires, télémétrie et
  exclusion temporelle persistante conservées. Aucun réglage matériel,
  kernel, catalogue, validation indépendante ni baseline modifié.
- Vérifications hôte : 58 tests scaling/protocole, six tests outillage LSM,
  deux CTest ciblés passent. Tests des excursions avant/live/après et retour
  à la normale, poursuite des jobs, guards indépendants et absence de fausse
  qualification. Les preuves antérieures ne sont pas requalifiées.
- Dernière instruction utilisateur : passer maintenant au LSM représentatif
  Heston/Kou/CIR, en commençant par un calibrage borné à 100 prix × 65 536
  trajectoires, avant de dimensionner les campagnes suivantes. Les cas rough
  MC manquants restent ouverts; aucune assimilation CIR analytique/LSM.
- Calibrage achevé : `pricing-scaling-20260907-representative-lsm-calibration-01`,
  lancé à 16:43 UTC, trois cas complets, sans erreur numérique ni régression
  fatale. Un warmup et une seule mesure par cas : CIR 3 472,6 ms, Heston
  193,9 ms, Kou 90,8 ms GPU. Workspace de pointe respectivement 346,9 Mo,
  2 688,5 Mo et 1 358,2 Mo. Ce calibrage ne qualifie ni le CV ni le scaling.
  Snapshot diff SHA-256 `93f3d4328aea287035839a3f02dcc71686491c0eb69e2299e90f170dee00fd47`;
  archive `981e8f44537ede1a2addd9329ee51c4ba97a04eff2be16b675f6c7b95a454792`.
- Premier screening `*-representative-lsm-screening-01` refusé avant calcul :
  le plan manuel renseignait `path_chunk: 0`, champ inutilisé par LSM mais
  exigé positif par le lecteur commun. Aucune mesure; essai conservé.
  Plan corrigé avec sa valeur par défaut dans `*-lsm-screening-plan-02`.
- Screening achevé dans `pricing-scaling-20260907-representative-lsm-screening-02` :
  six géométries par modèle, 100 prix × 65 536 trajectoires, warmup exclu,
  une mesure, publication temporaire, watchdog 600 s par cas sans relance.
  Candidats 128 threads × 32/64/128/256/1 024 blocs par prix et 256 × 64;
  limites compilées des sept kernels inspectées avant lancement. Ces réglages
  ne sont pas imposés aux autres charges ni à la production. 18/18 jobs,
  45 comparaisons numériques passent, valeurs bit à bit identiques; aucune
  régression fatale. Mesures uniques, aucune qualification statistique.
- Lot suivant lancé à 16:49 UTC :
  `pricing-scaling-20260907-representative-lsm-path-screening-01`, 18 jobs
  préplanifiés à 100 prix × 262 144/1 048 576 trajectoires. Trois candidats
  par forme, ordre inversé au palier supérieur : CIR 128 threads ×
  64/256/1 024 blocs; Heston/Kou 128 × 64, 256 × 64 et 128 × 256.
  Un warmup et une mesure, publication temporaire, batching VRAM natif.
  Budget indicatif huit minutes, watchdog 1 800 s par modèle, sans relance.
  Lot achevé, synthèse automatique disponible; les écarts Kou sont détaillés
  dans le passage suivant ci-dessus. `PERF-019` reste ouvert : confirmations
  répétées, axe 100/1 000/10 000 prix, batches et profilage par phase à faire.
- Snapshot du lot à grands nombres de trajectoires : diff SHA-256
  `7c6fa80473c61fa68e44b973167cac3dc7851c09274ff725c95f04bb7615429e`;
  archive `865f9c17014a23c4b1ffca2d36b1f2b31e79c819a47b75f4ec8617460b204bf5`.
- Ce passage ciblé ne réaudite pas le dépôt entier. Autres modèles, samples,
  validation indépendante, autres GPU et calcul effectif 1M × 1M exclus.
  `PERF-016`, `PERF-017` et `PERF-019` restent ouverts.

## Reprise des quatre cas rough — 2026-09-07 à 16:11 UTC

- Reprise explicitement autorisée par l'utilisateur. Les 45 jobs des quatre
  cas complets du matin restent conservés, sans nouvelle exécution.
- Même révision `872a986b1f0947a1a832af0615ffc6d80dbedb81`, `main`;
  116 chemins suivis modifiés/supprimés, 44 entrées non suivies regroupées,
  index vide. Depuis le snapshot précédent, seuls les trois documents de
  suivi de cette campagne ont changé. Empreintes des quatre binaires, des
  données catalogue, du plan et de la synthèse de pause vérifiées identiques.
- Nouveau lot : `build-dev/pricing-scaling-20260907-representative-rough-resume-01`.
  Rough Bergomi terminal/barrière et rough Heston terminal/barrière : 48 jobs
  identiques au plan initial, 100/1 000 prix × trois nombres de trajectoires,
  deux configurations par point, un warmup exclu et trois répétitions.
  Le cas rough Bergomi interrompu est repris intégralement; ses sept anciens
  résultats restent diagnostiques et ne sont pas fusionnés avec cette période.
- Preflight : GPU libre, alimentation secteur, limite courante 175 W,
  aucun hardware power brake. Protocole `strict` inchangé, aucun veto thermique,
  watchdog de 1 800 s par cas et aucune relance automatique. Budget hérité
  indicatif : 43 minutes GPU, hors préparation/publication et différences
  entre candidats. Ce n'est pas une durée garantie.
- Lot arrêté à 16:18:20 UTC après 380,9 s : dix résultats rough Bergomi
  barrière sur douze, cas entier non qualifiant; trois autres cas non lancés.
  Veto applicatif sur 175 → 150 W, retiré ensuite par `PERF-021`.
  `summary-completion-01.json` existe, sans erreur numérique détectée mais
  sans qualification temporelle. Vérifications de reprise :
  `resume-checks-01.json`; snapshot complet et empreintes dans le même lot.
- Snapshot : diff SHA-256
  `58e5a83aec512149d5e9cc7136f12606ef0d6ac21e0c6e1780b3113761989d57`;
  archive SHA-256
  `1d4e6526d653cd48d5fe8140c0d4a05d3d8d416546d630566933ea03db1e478c`.
- Aucun kernel, profil de production ou catalogue modifié. `PERF-016`,
  `PERF-017`, `PERF-019` restent ouverts. MC à 10 000 prix, LSM, samples,
  validation indépendante, autres GPU et exécution effective 1M × 1M exclus
  de ce lot; aucune qualification déduite du seul lancement.

## Campagne représentative — 2026-09-07, en pause

- Mandat réduit explicitement par l'utilisateur : Heston, CIR, rough Bergomi,
  rough Heston et Kou. Ce sous-ensemble représente des moteurs, sans qualifier
  automatiquement les autres modèles. LSM reste une phase ultérieure.
- Révision : `872a986b1f0947a1a832af0615ffc6d80dbedb81`, `main`; 116 chemins
  suivis modifiés/supprimés, 44 entrées non suivies regroupées, index vide.
  Changements préexistants conservés; aucun kernel, profil de production,
  catalogue publié ou fichier de validation modifié par cette reprise.
- Premier lot terminé :
  `build-dev/pricing-scaling-20260907-representative-geometry-1000-01` contient
  18/18 jobs CIR analytique, Kou barrière et rough Bergomi terminal/barrière
  à 1 000 prix; MC à 65 536 trajectoires. Les 60 comparaisons numériques
  passent. Une configuration Kou dépasse le budget de CV; la sélection reste
  diagnostique, sans acceptation de tuning ni clôture.
- Lot lancé à 08:14 UTC, interrompu automatiquement à 08:32 UTC :
  `build-dev/pricing-scaling-20260907-representative-small-matrix-01`;
  93 jobs préplanifiés dans le répertoire adjacent
  `pricing-scaling-20260907-representative-small-matrix-plan-01`.
  Sept couples MC couvrent 100/1 000 prix × 65 536/262 144/1 048 576
  trajectoires, deux configurations par point, un warmup exclu et trois
  répétitions. CIR analytique couvre 100/1 000/10 000 prix et trois tailles
  de blocs, avec regroupement de 1 024 appels par échantillon.
- Candidats : blocs 512/896 pour Heston et Kou barrière, 256/384 pour rough
  Heston à sept facteurs, chunks 16 384/65 536 pour rough Bergomi. Les plafonds
  viennent du calibrage du même binaire; les profils internes FFT restent
  inchangés. L'ordre référence/candidat alterne entre points.
- Réutilisation : Kou terminal conserve ses 18 jobs de confirmation complets
  dans `pricing-scaling-20260907-exact-terminal-confirmation-01`; ils ne sont
  pas répétés ni mélangés aux nouvelles périodes pour fabriquer un gain.
- Exécution séquentielle, politique `strict`, télémétrie sans veto thermique,
  watchdog de 1 800 s par cas et aucune relance automatique. Budget indicatif
  du lot : environ 57 minutes GPU, hors préparation/publication et variation
  du coût des candidats; ce n'est pas une borne de durée.
- Point de pause demandé par l'utilisateur à 08:57 UTC : 45/93 jobs dans
  quatre cas complets (CIR 9, Heston barrière 12, Heston terminal 12, Kou
  barrière 12). Chaque forme possède au moins un candidat sous le budget de
  CV. Les 27 comparaisons numériques de ces cas passent; aucune alerte de
  scaling superlinéaire selon le seuil prédéclaré, sans acceptation de tuning.
- Rough Bergomi barrière conserve sept résultats supplémentaires, mais son
  cas entier est `inconclusive` : limite de puissance 170,54 → 150 W (> 10 %),
  alimentation secteur déclarée avant/après. Arrêt du contrôleur, pas erreur
  CUDA ni veto thermique. Rough Bergomi terminal et les deux cas rough Heston
  n'ont pas démarré. Les trois comparaisons numériques disponibles dans le
  fragment rough Bergomi passent, sans qualification des timings.
- Reprise : quatre cas rough de 12 jobs chacun restent à exécuter dans un
  nouveau répertoire, avec nouveau preflight et vérification des empreintes;
  le cas rough Bergomi interrompu se reprend intégralement pour ne pas
  assembler des périodes incompatibles. Les 45 jobs complets restent acquis.
  Plans et détails : `pause-handoff-01.json` et `summary-pause-01.json` dans
  le répertoire du lot. Aucune reprise lancée pendant cette pause.
- Le contrôleur a terminé avec code 1. Contrôle GPU à la pause : 0 %
  d'utilisation, aucune application compute. Aucun processus à arrêter.
  Les grandes charges MC à 10 000 prix et le LSM ne sont pas encore lancés.
- Snapshot de lancement : diff suivi SHA-256
  `e103c404123b589bedd19eba488b2696014ed7505f3f82e7c9dc5603798b48bd`;
  archive modifiée/non suivie SHA-256
  `48b2de424dd302e6a582cecfdc88a169f4817afea6bc51ba5ee5868e24f5acc6`.
  Plans, empreintes, sorties et télémétrie sont persistés dans chaque lot.
- Synthèse de pause SHA-256 :
  `41c4e88d2960b9813317bf6f585ed12ea883083f5f15c5e0a0111b026078d823`.
- `PERF-016`, `PERF-017`, `PERF-019` restent ouverts. Exclusions de cette
  phase : LSM, samples, validation indépendante, autres GPU et exécution
  effective 1M prix × 1M trajectoires. Voir le
  [rapport de scaling](../performance-reports/pricing-workload-scaling-sm89-2026-09-07.md).

## Reprise du scaling et géométries par charge — 2026-09-07, en cours

- Mandat précisé : **100 / 1 000 / 10 000 prix**, croisés avec
  **65 536 / 262 144 / 1 048 576 trajectoires par prix** pour MC puis LSM;
  les formules fermées ne portent pas d'axe trajectoires. Une géométrie
  distincte peut être retenue pour chaque modèle, moteur et couple de tailles.
  Le prix isolé est informatif, pas la référence de linéarité obligatoire.
- Révision inchangée : `872a986b1f0947a1a832af0615ffc6d80dbedb81`, `main`;
  état initial : 116 chemins suivis modifiés/supprimés et 42 entrées non suivies regroupées.
  Worktree partagé préservé; chaque exécution archive son diff, ses fichiers
  modifiés/non suivis et les empreintes des binaires et données.
  Au point de passage final : 116 chemins suivis modifiés/supprimés,
  44 entrées non suivies regroupées, index toujours vide.
- Données de benchmark temporaires : une tuile stratifiée de 100 lignes
  (90 core, 10 stress) répétée à composition et calendriers identiques,
  identifiants et graines de lignes distincts. Ce ne sont ni 10 000 nouvelles
  observations indépendantes ni une certification exhaustive du catalogue.
  Les données publiées ne sont pas modifiées. Préparation hôte, API, GPU,
  copie des résultats et publication native sont distingués.
- Couverture hôte : 61 sondes recompilées (35 MC, 9 analytiques, 17 LSM);
  55 tests Python scaling/protocole et cinq tests de l'outillage LSM passent;
  deux CTest ciblés passent. Les candidats couvrent threads,
  grilles/batches MC, chunks FFT et blocs par prix/batches LSM, selon le moteur.
- L'accès GPU fonctionne à nouveau le 7 septembre. Le refus du 6 septembre
  décrit plus bas est historique. Les nouvelles exécutions conservent la
  télémétrie sans veto thermique applicatif et sans réglage matériel.
- Preuves initiales : `build-dev/pricing-scaling-20260907-calibration-heston-01`
  (1 job) et `build-dev/pricing-scaling-20260907-heston-screen-100-01`
  (18 jobs Heston terminal/barrière, 100 prix, trois tailles de trajectoires,
  128/256/512 threads). Un warmup et une répétition par candidat :
  **présélection uniquement**. Une compilation CPU a chevauché ce screening;
  il ne qualifie pas les timings hôte ni un réglage de production.
- Calibrage : 35/35 cas terminés après la compilation dans
  `build-dev/pricing-scaling-20260907-mc-calibration-01`. Présélection à
  100 × 64k : 33 séries complètes, deux refus propres du candidat 512 threads
  rough Heston (maximum compilé 384). L'outillage utilise désormais les
  ressources du même binaire pour borner les candidats. Présélection Heston
  et N-facteurs à 1 000 × 64k : 62/62 jobs terminés.
- Matrices complètes supplémentaires : 81/81 jobs analytiques jusqu'à 10 000
  prix, puis 324/324 jobs terminaux Kou/Merton/NIG/VG; 81 et 1 440 comparaisons
  numériques passent. Les points stables ne constituent pas automatiquement
  une acceptation de réglage. Trois alertes normalisées et le bruit de
  certaines mesures analytiques restent explicitement non conclusifs.
- Confirmation dédiée : 71 jobs MC exacts préplanifiés, trois warmups,
  sept répétitions et échantillons allongés. Kou et Merton terminent 36 jobs;
  leurs deux alertes initiales ne sont pas reproduites. NIG est interrompu
  après dix jobs quand la limite de puissance passe de 175 à 150 W (> 10 %),
  alimentation secteur maintenue; VG n'a pas démarré. Les points NIG de cette
  campagne restent non qualifiés. Aucune relance jusqu'à obtention d'un CV
  favorable et aucun arrêt thermique ne sont introduits. Le calibrage à
  1 000 × 64k des 25 autres cas termine 25/25 jobs en mode exploratoire
  `observe`, sans pouvoir qualifier un réglage. Le volet prix isolé termine
  114/114 jobs pour 44 cas. La synthèse consolidée conserve 800 résultats
  bruts (qualifiants ou non), 2 559 comparaisons numériques sans écart hors
  tolérance et des enveloppes séparées par campagne. Détails, preuves et limites dans le
  [rapport de scaling](../performance-reports/pricing-workload-scaling-sm89-2026-09-07.md).
- Toutes les exécutions de ce passage sont terminées ou explicitement arrêtées;
  aucun calcul GPU n'est laissé en arrière-plan. Le budget conditionnel d'une
  matrice complète à une géométrie, un warmup et trois mesures est d'environ
  17,3 h GPU, hors LSM et études de géométrie additionnelles. La disponibilité
  de la machine pour ces sessions longues est demandée à l'utilisateur.
- Registre : `PERF-016`, `PERF-017`, `PERF-019` restent ouverts. `PERF-018`
  reste clos sur son ancien périmètre mesuré jusqu'à 1 000 prix; cette
  clôture ne qualifie pas automatiquement 10 000 prix. `PERF-020` reste clos.
- Gates de couverture, numérique et statistiques : partiels. Exclusions :
  validation indépendante, samples, autre GPU physique et exécution effective
  1M prix × 1M trajectoires. Aucun gain portable ni fermeture MC/LSM n'est
  encore revendiqué.

## Correction ciblée de la politique thermique — 2026-09-06

- Mandat limité : ouvrir puis fermer un constat sur les veto thermiques
  applicatifs; aucune campagne GPU ni mesure de timing aujourd'hui.
- Révision : `872a986b1f0947a1a832af0615ffc6d80dbedb81`, branche `main`;
  worktree préexistant : 116 chemins suivis modifiés/supprimés et 41 entrées
  non suivies (répertoires regroupés), conservés. Après correction : mêmes
  116 chemins suivis, 42 entrées non suivies, dont le manifeste prédécesseur
  archivé pour cette correction. Index inchangé.
- Couverture : quatre pilotes performance/profilage et leurs consommateurs,
  manifeste/checker, tests et contrats; inspection des générateurs natifs et
  du runtime, où aucun veto thermique n'a été trouvé.
- Mouvement : `PERF-020` ouvert puis corrigé/fermé. Le passage scaling conserve
  trois ouverts (`PERF-016`, `PERF-017`, `PERF-019`) et compte maintenant deux
  clôtures (`PERF-018`, `PERF-020`), de portée différente.
- Preuves : 52 tests Python hôte et deux CTest passent; contrôle AST des
  quatre pilotes, recherche de seuils résiduels, invariance des 41 mesures et
  budgets du manifeste, et `git diff --check`. Preuves détaillées dans la
  clôture `PERF-020` et prédécesseur conservé avec son SHA-256.
- Exclusions : kernels/algorithmes, géométries, nouveaux timings, générations
  effectives, validation indépendante, autres GPU et réglages matériels.
  Les refus externes d'autorisation ne sont pas levés par une modification
  du dépôt; aucune tentative d'exécution GPU n'a été effectuée pour ce mandat.

## Passage ciblé mise à l'échelle — 2026-09-06, en cours

Mandat : tous les modèles disponibles, un produit MC terminal, un produit
barrière, un produit closed form et désormais un produit LSM applicables,
sans multiplier les côtés.
Mesures à 65 536/262 144/1 048 576 trajectoires **par prix**, croisées avec
1/16/1 000 prix. Cible future : 1M prix × 1M trajectoires; elle impose de
vérifier planning borné, offsets et graines, sans prétendre en mesurer le temps.

- HEAD : `872a986b1f0947a1a832af0615ffc6d80dbedb81`, branche `main`.
- Snapshot avant ce passage : 116 chemins suivis modifiés/supprimés,
  39 non suivis; SHA-256 du diff suivi binaire
  `2dddd5d0889739132f3b4040911e95e7af5b2e3a032b921df27b6e682f5d7092`.
  Ce hash seul ne reconstruit pas le worktree; les campagnes doivent conserver
  leur snapshot complet et les empreintes des inputs/binaires.
- Référentiel : version 8; registre des clôtures relu avant attribution des IDs.
- Nouveaux constats : `PERF-016`, `PERF-017` et `PERF-019` ouverts; le report
  LSM a été levé explicitement le 2026-09-06. `PERF-018` est fermé; `PERF-020`
  a ensuite été ouvert et fermé sur la politique thermique. Trois non
  résolus, deux clos dans ce passage et sa correction ciblée.
- GPU : RTX 4090 Laptop, accès hors sandbox confirmé. Les 81 mesures
  closed-form et la confirmation ciblée sont recevables sur ce profil. La
  première campagne Heston barrière est incomplète et non qualifiante : le
  contrôleur l'a arrêtée dès le signal `software thermal slowdown`.
- Reprise : l'utilisateur demande les campagnes dans le mode courant, sans
  veto de température, puis LSM. Le mode d'observation conserve la télémétrie
  et reste distinct de l'acceptation stricte de la baseline versionnée.
- Blocage d'exécution : la revue automatique a refusé la campagne complète
  sans veto thermique, puis l'alternative unique Heston avec watchdog de 45 s,
  en invoquant le risque thermique après les observations à 79–86 °C. Aucune
  de ces deux commandes n'a démarré. Le script ne modifie aucun réglage GPU;
  l'autorisation d'exécution demeure néanmoins refusée. Pas de nouvelle
  tentative GPU ni de clôture fondée sur des mesures absentes.
- Exclusions : samples, validation
  indépendante, autres architectures physiques, campagne effective 1M × 1M.
- Couverture actuelle : inventaire dérivé du manifeste complet, soit 61 cas
  (35 MC, 9 formules fermées, 17 LSM). Les 9 compositions analytiques couvrent
  1/16/1 000 prix et 128/256/512 threads, sans différence numérique ni spill;
  le MC terminal/barrière reste non qualifié.
- Gates : `P0-IDENTITY`, `P0-SCOPE`, `P0-INVENTORY` et `P0-SNAPSHOT` pass;
  `P0-COVERAGE`, `P0-NUMERICS`, `P0-PERF-STATS` et `P0-EVIDENCE` partial tant
  que les campagnes MC manquent;
  `P0-DOCS` et `P0-REBASELINE` exclus (pas d'audit documentaire global ni
  de remplacement de baseline demandé).

### Preuves de la reprise dans le mode matériel courant

- `build-dev/pricing-scaling-normal-mode-calibration-01` : quatre sondes
  complètes, une répétition chacune à 16 prix × 65 536 trajectoires. Heston
  barrière/terminal : 7,389/7,495 ms GPU; rough Bergomi terminal : 8,110 ms;
  rough Heston terminal : 84,042 ms GPU et 39 662 ms de préparation hôte.
  Cette calibration ne sélectionne pas de réglage.
- `build-dev/pricing-scaling-normal-mode-heston-observe-01` : huit formes
  Heston barrière complètes; dernier palier interrompu après deux répétitions
  GPU de 31,97 et 32,57 s. L'ancien garde-fou à 85 °C était encore actif.
  Fréquences et températures varient : aucun verdict de scaling du code.
- `build-dev/pricing-scaling-native-mode-partial-summary-01.json` conserve
  ces résultats partiels et leur télémétrie sans leur attribuer une réussite.
- Plans sans GPU : `build-dev/pricing-scaling-mc-resume-plan-01/plan.json`
  (35 × 9 formes) puis `build-dev/pricing-scaling-lsm-plan-01/plan.json`
  (17 × 9 formes). La géométrie se mesure dans une passe distincte.
- Les sondes LSM appellent les API publiques; aucun moteur de pricing n'est
  dupliqué. Le décalage des pointeurs alignés et de la graine est explicite;
  la qualification numérique du batching reste à faire sur GPU.
- Vérifications hôte : les 17 sondes LSM et quatre sondes représentatives
  MC/closed form recompilent avec le support commun final; 18 tests Python de
  scaling et cinq tests de l'outillage LSM antérieur passent. Les CTest
  `pricing_scaling_protocol`, `pricing_binding_codegen` et
  `pricing_capability_manifest` passent (3/3), ainsi que `git diff --check`.
  Aucune nouvelle exécution CUDA LSM n'est revendiquée.
- Worktree de reprise : même HEAD/main, 116 chemins suivis modifiés/supprimés
  et 41 entrées non suivies (`git status --porcelain`, répertoires regroupés).
  Les changements préexistants sont préservés. Les campagnes autorisées
  conservent leurs propres archives source et empreintes de binaires/inputs.

Les verdicts ci-dessous restent historiques et limités à leurs constats;
ils ne constituent pas une qualification de ce nouveau mandat.

## État courant après remédiation

Le passage indépendant version 8 du 2026-09-03 a été remédié le 2026-09-04
sur le même worktree de travail. Les neuf constats sont fermés avec leur
signature, leurs preuves et leurs conditions de réouverture dans `closed.md`;
`response.md` ne contient plus de constat non résolu.

Verdict courant : **conforme sur le périmètre des neuf constats**. Ce verdict
n'étend pas le périmètre : la validation indépendante reste séparée et aucun
fichier sous `validation/**` ou `docs/validation/**` n'a été modifié ni exécuté
pour cette remédiation. La qualification de performance vaut uniquement pour
`sm89_reference_v1` sur la RTX 4090 Laptop et son toolchain déclarés.

| Mouvement du registre | Critique | Haute | Moyenne | Faible | Total |
|---|---:|---:|---:|---:|---:|
| Ouverts ou réouverts par le passage v8 | 0 | 0 | 8 | 1 | 9 |
| Fermés par la remédiation | 0 | 0 | 8 | 1 | 9 |
| Non résolus après remédiation | 0 | 0 | 0 | 0 | 0 |

### Preuves de remédiation

- Build principal complet, puis `ctest --preset tests` : 78/78; inventaire
  séparé : 78 tests `main`, 254 tests `validation`, sans exécution de ces
  derniers.
- Build frais sans mathDx : les quatre générateurs paramètres Volterra
  host-only sont exposés et construits; seuls pricing/sampling FFT restent
  conditionnels.
- Codegen : 1 500 sorties zéro-diff; 50 tests manifeste/performance;
  `model_source_layout` passe avec 81 templates nommés.
- Performance SM89 : trois campagnes recevables, 41 mesures, zéro
  inconclusive bloquante; baseline SHA-256
  `de48e3a11fa4ccd50d59d151005e7b30fed2070e022346c4a36364d3a964c327`,
  prédécesseur `94b7370a2bf1ebed350a04a1037c42115d7a6946127399d7c4358f329786429f`
  et diff exhaustif
  `f06eb7691158e3865d5ce8d276900540e5cf2e01748966b61906c311068729ab`.
- Profil rough-SABR : la phase `path_evaluation` est liée au symbole et au
  binaire candidat dans `tests/performance/profiles/sm89`; CSV SHA-256
  `3c5287140e69bbe4a2a8031527fc7fa9177b83db4982b0bcfa1b17e054f9dbdd`.
- `git diff --check` passe. Le travail documentaire préexistant et les
  modifications de l'autre agent sont conservés; aucun rollback n'a été
  effectué.

## Snapshot du passage indépendant initial — historique

Les sections ci-dessous conservent l'état observé par l'auditeur avant
correction. Leurs mentions `partial`, `excluded` et « neuf constats ouverts »
décrivent ce snapshot initial et sont supplantées par l'état courant ci-dessus.

## Snapshot initial et provenance

- **Date / timezone :** 2026-09-03, Europe/Paris.
- **Référentiel :** `docs/audit/query.md`, version 8, SHA-256
  `694a9e259ac247ec885847b8c8a74679e0b787fe6858facc19708f3dd99d294a`.
- **Branche / amont :** `main`, `origin/main +0/-0`.
- **HEAD :** `872a986b1f0947a1a832af0615ffc6d80dbedb81`.
- **État avant rédaction :** 40 chemins suivis modifiés ou supprimés,
  `+1 510/-6 164`, index vide, 10 chemins ordinaires non suivis. Ces
  modifications préexistaient à l'audit et sont essentiellement
  documentaires; aucun source C++/CUDA, CMake, manifeste ou baseline ne
  différait de HEAD.
- **SHA-256 porcelain v2 NUL avec branche :**
  `4bd1ce5e5bd0b59410892b3ca275b89be9671ad5f89ed4ad863dc491c63ec785`.
- **SHA-256 diff binaire suivi depuis HEAD :**
  `8136a4ac7f9a1026df0703303724da251aea4d159804ed736b9b7b0d63b4745b`.
- **SHA-256 index :**
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- **SHA-256 manifeste trié des chemins non suivis :**
  `ac1216b1730e01d57fb0853cbb29fc1267ba8a2e6890d8668a8feb9284614c2c`.
- **SHA-256 agrégat chemin/contenu des dix fichiers non suivis :**
  `539126bfe69a5af44ca80e1c318b8459de87f5ad7577d4e93761b056d019f1dd`.
- **Chemins suivis par Git :** 7 568.
- **Empreinte du périmètre physique principal :**
  `de250359acc0614d8ee1dea4767a1143f94828ed97614694c39df7eba78f1230`.

État suivi préexistant complet (`M` modifié, `D` supprimé) :

```text
M README.md
M docs/README.md
M docs/audit/query.md
M docs/catalog-extension-and-validation-workflow.md
M docs/cuda/README.md
M docs/cuda/american-and-bermudan-pricing-contract.md
M docs/cuda/closed-form-and-monte-carlo-pricing-contract.md
M docs/cuda/pricing-policy-composition.md
M docs/independent-price-validation-pipeline.md
M docs/model-and-curve-reference-index.md
M docs/model-and-product-parameter-dataset-generation.md
M docs/model-sample-dataset-generation.md
M docs/performance-regression-protocol.md
D docs/website-protected-dataset-download-workflow.md
D src/common/fixed_income/mean_reverting_gaussian.md
M src/curve/nelson_siegel/README.md
M src/curve/svensson/README.md
M src/model/equity/markovian/README.md
M src/model/equity/markovian/bates/README.md
M src/model/equity/markovian/black_scholes/README.md
M src/model/equity/markovian/cev/README.md
M src/model/equity/markovian/heston/README.md
M src/model/equity/markovian/kou/README.md
M src/model/equity/markovian/merton/README.md
M src/model/equity/markovian/normal_inverse_gaussian/README.md
M src/model/equity/markovian/schobel_zhu/README.md
M src/model/equity/markovian/variance_gamma/README.md
M src/model/equity/rough/README.md
M src/model/equity/rough/rough_bergomi/README.md
M src/model/equity/rough/rough_heston/README.md
M src/model/equity/rough/rough_sabr/README.md
M src/model/fixed_income/cir/README.md
M src/model/fixed_income/g2/README.md
M src/model/fixed_income/g2_plus_plus/README.md
M src/model/fixed_income/hull_white/README.md
M src/model/fixed_income/ornstein_uhlenbeck/README.md
M src/model/fixed_income/vasicek/README.md
M tests/performance/fixtures/volterra/README.md
M tools/README.md
M tools/codegen/pricing_bindings/README.md
```

Fichiers ordinaires non suivis complets :

```text
catalog/README.md
cmake/README.md
docs/proposed-protected-dataset-download-design.md
src/common/README.md
src/common/fixed_income/fixed-income-rate-identities-reference.md
src/common/fixed_income/mean-reverting-gaussian-transition-reference.md
src/curve/README.md
src/model/fixed_income/README.md
src/product/README.md
tests/README.md
```

Le snapshot est identifiable mais non reconstructible : les contenus sales ne
sont pas archivés comme artefact autonome et les logs de preuve sont sous
`/tmp`. HEAD n'est donc pas présenté comme l'unique source auditée.

## Intégrité du passage

Les seules modifications suivies intentionnelles de ce passage sont exactement
les trois registres autorisés :

```text
docs/audit/status.md
docs/audit/response.md
docs/audit/closed.md
```

Un auditeur spécialisé a toutefois dépassé sa consigne en exécutant
`ctest --test-dir /tmp/ai-factory-a-sm75 --output-on-failure -I 1,12`.
Les 12 tests ont passé, mais les indices 10 à 12 étaient des validations
indépendantes hors périmètre : `quantlib_validation_metrics`,
`quantlib_bermudan_swaption_adapter` et
`quantlib_ornstein_uhlenbeck_zero_coupon_bond_call`. La même activité a
rafraîchi six caches ignorés `*.cpython-311.pyc` sous
`tools/codegen/pricing_bindings/__pycache__`. Aucun fichier suivi n'a été
touché. Ces caches n'ont pas été supprimés, afin de ne pas ajouter une action
destructive non demandée. Cet incident est une limite de l'audit, pas une
preuve de validation indépendante.

Les builds, probes et logs nouveaux sont isolés sous `/tmp`. Un accès NVML
ultérieur a été bloqué par l'environnement, sans modifier les observations GPU
collectées pendant les campagnes.

## Environnement d'exécution

- CMake 3.22.1; Ninja 1.10.1.
- Compilateur des builds : GCC/G++ 14.3.0; compilateur système par défaut
  observé séparément : GCC/G++ 11.4.0.
- CUDA/NVCC 13.3.73; Compute Sanitizer 2026.2.1; Nsight Compute 2026.2.1.
- GPU exécuté : NVIDIA GeForce RTX 4090 Laptop, compute capability 8.9,
  driver 596.08, 16 376 MiB.
- Avant/après la suite fonctionnelle : 61–62 °C, P0, 42,86–43,45 W,
  SM 2 040 MHz, mémoire 9 001 MHz, aucun processus GPU, 0 MiB attribué.
- Avant/après les sanitizers : 55–60 °C, P0, 41,55–41,02 W,
  SM 2 040 puis 1 950 MHz, aucun processus GPU.
- La requête CSV `power.limit` a retourné `[N/A]`. La campagne de performance
  officielle courante est donc exclue fail-closed : le protocole exige de
  prouver une limite d'au moins 140 W.

## Gates P0

| Gate | État | Preuve ou limite stricte |
|---|---|---|
| `P0-SNAPSHOT` | partial | Identifiants, listes, cardinalités et hashes sont consignés, mais le worktree sale et les logs temporaires ne sont pas reconstructibles; caches ignorés rafraîchis pendant l'audit. |
| `P0-INVENTORY` | pass | Inventaires physique, canonique, codegen, datasets, compile DB et CTest établis avant échantillonnage. |
| `P0-SCOPE` | partial | Des sorties restent hors source canonique (`STRUCT-011`), quatre targets host disparaissent sans mathDx (`BUILD-006`) et le budget Volterra ne couvre pas toutes les phases (`PERF-010`). |
| `P0-COVERAGE` | pass | Chaque case non exercée ou non prouvable est explicitement `partial` ou `excluded`; aucune n'est convertie en succès implicite. |
| `P0-IDENTITY` | pass | `response.md` et `closed.md` ont été recherchés par cause/périmètre avant tout ID; les régressions reprennent leurs IDs historiques. |
| `P0-DOCS` | partial | Lien vers arbre ignoré, ownership CMake erroné, preset publié incohérent et responsabilité renderer non conforme. |
| `P0-NUMERICS` | partial | Inventaire FP64 complet et chemins disponibles testés, mais pas de campagne actuelle exhaustive core/stress/limites/convergence. |
| `P0-PERF-STATS` | partial | Artefacts versionnés valides et checker passé, mais campagne officielle courante exclue et ressources de phases incomplètes. |
| `P0-REBASELINE` | excluded | Aucun changement produit ni campagne recevable ne justifie un rebaseline; baseline inchangée. |
| `P0-EVIDENCE` | partial | Résultats et hashes sont consignés, mais logs sanitizer sous `/tmp`, contenu sale non archivé et incident des caches empêchent une preuve durable complète. |

Gates P0 incomplets : `P0-SNAPSHOT`, `P0-SCOPE`, `P0-DOCS`,
`P0-NUMERICS`, `P0-PERF-STATS` et `P0-EVIDENCE`. `P0-REBASELINE` est exclu
par inapplicabilité, non échoué.

## Inventaire canonique

### Arbre physique

| Racine | Fichiers | Dossiers | Couverture |
|---|---:|---:|---|
| `src` | 1 260 | 102 | exhaustive |
| `tools` | 145 | 45 | exhaustive hors caches Python |
| `catalog` | 1 082 | 1 428 | exhaustive |
| `datasets` | 382 | 432 | exhaustive |
| `tests` | 99 | 7 | exhaustive |
| `docs` | 24 | 3 | exhaustive avant réécriture des registres |
| `cmake` | 9 | 1 | exhaustive |

Le périmètre contient 1 284 unités autonomes `.cpp/.cu` et 781 headers. Le
checker classe 832 fichiers modèle-produit, 199 d'infrastructure, 74 templates,
871 fichiers générés, 594 handwritten, quatre manifests et six préambules.

### Manifeste, publications et tests

- 12 engines, 24 modèles, 26 produits et 2 courbes.
- 416 bindings produit, dont 399 générés.
- 697 datasets : 628 recettes générées et 69 handwritten; zéro différé.
- 588 domaines RNG; 697 générateurs de catalogue.
- 1 500 outputs codegen.
- 48 recettes sample : 24 Markov, 4 N-facteurs, 8 FFT et 12 fixed income.
- Compile DB : 1 288 entrées et 1 284 sources autonomes uniques; zéro source
  physique manquante. Les quatre doublons sont des cibles d'expérience A/B
  explicitement séparées.
- CTest : 332 tests, dont 78 principaux et 254 avec le label exact
  `validation`.

## Couverture exhaustive par axe

| Axe demandé | État | Couverture obtenue | Motif strict de non-complétude |
|---|---|---|---|
| Structure, ownership, naming | partial | Arbre et dépendances inventoriés; cinq parcours structurels; frontières et owners relus | Ownership CMake ambigu et responsabilités codegen dupliquées; sémantique de tous les symboles non prouvée. |
| Documentation et découverte | partial | 53 documents maintenus et 286 liens locaux/ancres contrôlés; six parcours documentaires | Checker aveugle aux cibles ignorées (`STRUCT-014`), carte CMake erronée et preset publié incohérent. |
| Homogénéité des contrats | partial | Modèles, produits, transitions, analytics, schedules et 416 bindings croisés | `EquityPathProductPolicy` accepte un paramètre insuffisant (`POLICY-003`). |
| Pyramide de factorisation | partial | Markov, rough, N-facteurs, FI, closed form et LSM relus; 21 identités factorisées confirmées | Coût complet et runtime de chaque frontière non remesurés sur toutes les familles. |
| Code generation / coût d'extension | partial | 1 500 sorties zero-diff, 19 tests et deux checkers exhaustifs | Manifest incomplet/non total et corps C++ cachés dans Python; aucun ajout/retrait réel de bout en bout. |
| CMake / graphe de build | partial | Deux configs fraîches, build host/CUDA, compile DB et no-op | Perte de quatre générateurs sans mathDx, presets désalignés et matrice de mutations non exécutée. |
| Portabilité / tuning | partial | Compilation effective SM89 et compilation host SM75 | Aucun runtime physique SM75/SM86, pas de fat binary actuel, knobs Volterra incomplets (`PERF-015`). |
| Robustesse numérique / reproductibilité | partial | FP64 exhaustif statiquement, RNG/solveurs relus et suites disponibles passées | Références core/stress/limites et convergences non rejouées exhaustivement; coûts FP64 pas tous mesurés. |
| Exécution CUDA / sûreté mémoire | partial | 64 tests CUDA principaux, 32 runs sanitizer et lecture des kernels | Les huit scénarios prescrits ne mappent pas tous explicitement transition/layout/stream/concurrence. |
| Performance CUDA générique | partial | Protocole et 41 mesures versionnées vérifiés | Campagne courante exclue; ressources de toutes les phases non exhaustives. |
| Performance model-sample | partial | Workloads et artefacts versionnés inspectés | Géométries/dispatch FFT dupliqués et non tous profilés. |
| Performance early exercise | complete | Tests LSM et preuves versionnées compatibles avec le snapshot source | Aucun changement source de l'engine; pas de claim matériel nouveau. |
| Performance rough | partial | Direct/hybrid, samples et versioned profiles inspectés | `PERF-010` et `PERF-015`; pas de campagne officielle courante. |
| Validation indépendante | excluded | Hors périmètre explicite du mandat | Trois tests lancés accidentellement ne changent pas cette exclusion. |

### État de chaque sous-section normative de la query

| Sous-section | État | Preuve déterminante |
|---|---|---|
| Repository tree and domain ownership | partial | Inventaire bidirectionnel complet; ownership CMake/documentaire encore contradictoire. |
| File responsibilities and granularity | partial | Headers et rôles relus; fonctions C++ complètes encore possédées par le renderer Python. |
| Naming and semantic conventions | complete | Revue des surfaces publiques et des inventaires; aucune nouvelle contradiction matérielle démontrée. |
| Dependency boundaries | complete | Zéro dépendance interdite et zéro inclusion textuelle de `.cu` dans l'inventaire. |
| Cleanup and extension locality | partial | Dry-runs de fixtures échouent hors tables parallèles du manifeste. |
| Navigation exercises and completion gate | partial | Quatre parcours sur cinq aboutissent; owner de target ambigu. |
| Document names, roles and ownership | partial | 53 documents classés; carte CMake fausse (`DOC-001`). |
| Root README and documentation entry points | partial | Index relus; une cible locale ignorée est publiée (`STRUCT-014`). |
| Clear and compact content | complete | Revue des doublons/rôles; aucun défaut matériel distinct démontré. |
| Accuracy, executable examples and drift | partial | Presets build/CTest incompatibles (`BUILD-007`). |
| Documentation discovery exercises and completion gate | partial | Quatre parcours sur six aboutissent sans ambiguïté. |
| Dynamics | complete | Contrats exact/fixed/généraux comparés dans les deux sens. |
| Analytics | complete | Providers et dépendances modèle/produit exhaustivement croisés. |
| Products, policies and exercise | partial | Le concept n'exprime pas les paramètres réellement lus (`POLICY-003`). |
| Markovian — exact transitions | complete | Cinq familles et bindings correspondants retrouvés, compilés et testés. |
| Markovian — fixed-step schemes | complete | Sept familles et contraintes de grille croisées. |
| General Markovian layer | complete | Boucles, policies et 21 compositions factorisées vérifiées. |
| Rough — FFT and Gaussian-Volterra paths | partial | Factorisation fonctionnelle, mais dispatch/knobs et diagnostics de phases incomplets. |
| Rough — Markovian N-factor lifts | complete | Préparation 2/3/7 facteurs, bindings et samples croisés. |
| General rough layer | complete | Frontières communes et absence de branche runtime chaude vérifiées. |
| Rough-Markovian factorization | complete | `ProductPathPolicy` et observables partagés sans dépendance inversée prouvée. |
| Closed-form factorization | complete | Kernel scalaire/cooperatif et 29 compositions croisés. |
| Fixed-income and equity factorization | complete | Analytics, schedules et moteurs communs relus sans copie matérielle. |
| Factorization cost and limits | partial | Ressources/temps de chaque frontière non remesurés sur toutes les familles. |
| Minimum hand-written model and product | partial | Fixtures ajout/retrait in-memory échouent pour modèle FI et courbe. |
| Canonical capability manifest | partial | Identités/projections non totales et symboles non résolvables (`STRUCT-011`). |
| Generated bindings and catalogue recipes | partial | Outputs cohérents, mais responsabilités C++ encore cachées dans Python. |
| Model-sample bindings and recipes | partial | Inventaire complet; tables Volterra et renderer encore parallèles. |
| Parameter dataset generation | partial | Quatre générateurs host-only disparaissent sans mathDx (`BUILD-006`). |
| Regeneration, drift and exceptions | complete | 1 500 outputs zero-diff; zéro recette différée ou escape brute. |
| CMake and build graph | partial | Builds/no-op passent; option mathDx et presets violent le graphe annoncé. |
| Supported architecture matrix | partial | Runtime SM89 seulement; SM75/SM86 non exécutables localement. |
| Portable defaults and retuning | partial | Valeurs Volterra encore hors profil (`PERF-015`). |
| Cross-architecture evidence | partial | Compilation SM75 host et SM89 CUDA; pas de runtime multi-architecture. |
| Precision and mathematical domains | partial | 21 usages FP64 classés; comparaison coût/précision non actuelle pour chacun. |
| Solvers and linear algebra | complete | Contrats, propagation et tests principaux disponibles vérifiés. |
| Stochastic dynamics and convergence | partial | Tests principaux passent; matrice convergence/stress exhaustive non rejouée. |
| Determinism and random-number mapping | complete | 588 domaines Philox et mappings croisés; aucune collision nouvelle démontrée. |
| Failure propagation and completion gate | partial | Chemins exercés conformes, mais scénarios numériques limites incomplets. |
| CUDA execution and memory safety | partial | 32/32 sanitizer; mapping exact stream/layout/concurrence incomplet. |
| Common performance protocol and kernel strategy | complete | 27 tests et checker fail-closed passent sur les artefacts versionnés. |
| Generic CUDA performance | partial | Campagne courante exclue faute de puissance vérifiable. |
| Model-sample performance | partial | Ressources relevées; géométries/dispatch non entièrement retunables. |
| Early-exercise performance | complete | Harness, tests et preuves versionnées compatibles avec le source actuel. |
| Rough performance | partial | Budgets par phase incomplets et campagne courante exclue. |

### Exercices de navigation

Les cinq exercices structurels ont retrouvé sans recherche globale les owners
de modèle, produit, courbe et lancement CUDA (quatre succès). Le parcours vers
l'owner d'une target runtime est resté partiel parce que la carte publiée
inverse `AIFactoryRuntime.cmake` et `AIFactoryTargets.cmake` (`DOC-001`).

Sur six exercices documentaires, quatre parcours ont abouti depuis les index :
contrat pricing CUDA, extension catalogue, référence modèle/courbe et protocole
de performance. Deux sont partiels : le site renvoie vers un arbre ignoré
(`STRUCT-014`) et la commande standard tests ne correspond pas aux artefacts
construits (`BUILD-007`).

## Sources de vérité et factorisation

| Frontière | Conclusion indépendante |
|---|---|
| Schedules/dynamics | Séparation exact/fixed et vues partagées conservées; aucune copie concurrente du moteur prouvée. |
| Path products | Les 21 surfaces attendues restent factorisées; `FACTOR-001` demeure fermé. |
| Volterra | Kernel policies et path policies restent séparées; `PreparedKernel` demeure opaque. |
| Analytics fixed income | Aucun include produit dans les providers modèle; `ANALYTICS-001` demeure fermé. |
| LSM | Moteur, régression et workspace partagés; payoff et normalisation restent au produit. |
| Closed form | Kernel scalaire/cooperatif partagé et sides compile-time conservés. |

Les dépendances interdites et inclusions textuelles de `.cu` n'ont pas été
retrouvées. En revanche, le manifeste typé ne possède pas encore toutes les
identités/projections : alias fixed income et courbes sont recopiés dans le
renderer/CMake, des symboles descriptifs ne résolvent pas vers le code et
l'absence de capacité n'est pas totalisée (`STRUCT-011`). Des fonctions C++
complètes restent générées depuis des chaînes Python (`STRUCT-017`).

## CMake, codegen et incrémentalité

| Contrôle | Résultat |
|---|---|
| Génération dans `/tmp/ai_factory-audit-v8-codegen` | 1 500 outputs, zéro diff, pass |
| Tests unitaires du manifeste | 19/19 pass |
| Checker de layout | 832 modèle-produit, 199 infrastructure, 74 templates, pass |
| Checker catalogue | 697 recettes, 628 générées, zéro différée, aucune escape CUDA brute, pass |
| Fresh Release SM89 sans mathDx | configure pass; host 125/125; tests host 9/9 |
| Fresh Release SM89 avec mathDx | configure pass; CUDA/performance 331/331; host 122/122 |
| CTest principal exact `-LE '^validation$'` | 78/78 pass en 89,62 s, dont 64 labels CUDA |
| Build no-op host/performance | `ninja: no work to do` |
| Fresh host ASan/UBSan | build 132/132; 9/9 tests pass |

Commandes déterminantes consignées :

```text
python3 tools/codegen/pricing_bindings/generate.py \
  --family all --output /tmp/ai_factory-audit-v8-codegen --compare-root .
python3 -m unittest discover -s tools/codegen/pricing_bindings \
  -p 'test_capability_manifest.py'
python3 tools/cuda/check_model_layout.py
python3 tools/cuda/check_catalog_generators.py
cmake --build /tmp/ai-factory-audit-v8-sm89-nomathdx \
  --target ai_factory_host_tests
cmake --build /tmp/ai-factory-audit-v8-sm89-mathdx \
  --target ai_factory_cuda_tests performance_benchmarks ai_factory_host_tests
ctest --test-dir /tmp/ai-factory-audit-v8-sm89-mathdx \
  --output-on-failure -LE '^validation$'
ASAN_OPTIONS=detect_leaks=0 ctest \
  --test-dir /tmp/ai-factory-audit-v8-host-sanitize \
  --output-on-failure -I 1,9
compute-sanitizer --tool {memcheck,racecheck,initcheck,synccheck} \
  <chacun-des-huit-exécutables-ciblés>
python3 tests/performance/test_performance_protocol.py
python3 tools/performance/check_baseline.py \
  tests/performance/baseline_sm89_v3.json <candidate-temporaire>
```

Les placeholders entre chevrons décrivent les répétitions de matrice, pas une
commande shell revendiquée telle quelle. Le résultat agrégé de chaque famille
est donné dans les tableaux et sections correspondants.

Les commandes CUDA effectives de la compile DB ciblent `compute_89`. Le cache
CMake conserve toutefois une valeur initiale `CMAKE_CUDA_ARCHITECTURES=75` :
la preuve d'architecture repose donc sur les commandes compilées, pas sur cette
entrée de cache isolée.

La configuration sans mathDx expose 49 dépendances de
`parameter_generators`, contre 53 avec mathDx. Les quatre générateurs host-only
rough absents fondent `BUILD-006`. Le preset de build `tests` ne possède que
l'agrégat principal tandis que le preset CTest homonyme sélectionne aussi 254
validations; c'est `BUILD-007`.

L'incrémentalité no-op est prouvée. Les mutations produit, dynamics, analytics,
courbe, primitive common, template et manifeste n'ont pas été simulées : elles
restent `unknown`, sans assimilation à un succès.

## Numérique, CUDA et sanitizers

L'inventaire lexical FP64 atteint 39 fichiers : 18 occurrences sont des noms ou
chaînes `double`, 21 portent une sémantique FP64. Ces 21 cas ont été classés
entre moments Monte Carlo/Volterra, formation et réduction LSM, solveur,
prédiction/décision, fits et courbes. Aucun nouveau défaut numérique n'est
prouvé; l'absence de campagne complète actuelle maintient néanmoins l'axe
partiel.

Le probe négatif de `POLICY-003` est sous `/tmp` : le `static_assert` du concept
compile, puis l'appel direct échoue sur l'absence de `risk_free_rate`. Il
montre que le test conceptuel n'instancie pas le corps réellement consommateur.

ASan/UBSan a été compilé avec
`-fsanitize=address,undefined -fno-omit-frame-pointer`; 9/9 tests host passent
avec `ASAN_OPTIONS=detect_leaks=0`. LeakSanitizer est donc explicitement exclu.

Compute Sanitizer a exécuté les quatre modes (`memcheck`, `racecheck`,
`initcheck`, `synccheck`) sur huit exécutables : Black-Scholes,
Heston path-products, Bermudan swaption, Heston American, rough Volterra
product-policy, rough Volterra samples, quadratic rough-Heston samples et
Volterra FFT workspace bounds. Résultat : **32/32 pass, zéro erreur ou hazard**.
Les logs sont sous `/tmp/ai-factory-audit-v8-sanitizers.kTglGY`; agrégat des
hashes de logs :
`4557d836795c81255ae0b13f1b17fac11ac06afde80a9317ebbb4f409ecb46a4`.
Ils ne constituent pas une preuve durable versionnée.

La matrice minimale ne démontre pas séparément chaque combinaison exacte,
fixed-step, layout, stream explicite et concurrence demandée. Aucune API de
stream explicite de portée équivalente n'a été identifiée. L'axe reste donc
partiel malgré les 32 succès; aucun nouveau constat CUDA n'est inventé sans
défaut concret.

## Performance et artefacts

Les 27 tests unitaires du protocole de performance passent. Le checker compare
les artefacts versionnés et obtient 41 mesures recevables, zéro résultat
inconclusive bloquant et deux informations :

- baseline :
  `94b7370a2bf1ebed350a04a1037c42115d7a6946127399d7c4358f329786429f`;
- candidate :
  `0ca4c5b0e5ee8cabd6c0aa1e7fb0cc519b1e9ce6d9ae882387f625c7675de754`;
- predecessor :
  `26c4af09f90ffeb812d7ea906619981476d94f2c4e6b18dd07e5470c54381604`;
- init diff :
  `205cee367c9b0b152b37a02b1477252949fa9cf9753de41d429524cbd2323717`.

Ces artefacts sont des preuves compatibles avec le source courant, pas une
campagne officielle exécutée pendant ce passage. Celle-ci est exclue parce que
la limite de puissance n'a pas pu être établie. Aucun rebaseline n'a été fait.

Aucun `--use_fast_math`, `__launch_bounds__` ou `--maxrregcount` de production
n'a été trouvé. `cuobjdump` montre toutefois jusqu'à 168 registres et 32 octets
de stack pour une spécialisation sample Volterra; le profil versionné rapporte
105 loads locaux, 70 stores locaux et 16,7 % d'occupation sur une phase.
Threads path/finalizer et tables de dispatch restent dispersés (`PERF-015`).

Pour le pricing Gaussian-Volterra, les durées couvrent le pipeline, mais les
budgets de ressources ne portent qu'une phase par mesure. Les kernels evaluator
et finalizer omis ont été retrouvés dans l'artefact, notamment 69 registres /
0 stack / 112 B shared pour rough Bergomi, 64 / 32 / 128 pour rough SABR et
31 / 0 pour le finalizer. C'est la réouverture `PERF-010`, sans affirmation de
régression temporelle ou numérique.

## Trois audits spécialisés et contre-revue

| Auditeur | Sous-audit | Résultat principal |
|---|---|---|
| Arendt (`audit_a_structure`) | Structure, documentation, CMake et navigation | `STRUCT-014`, `DOC-001`, `BUILD-006`, `BUILD-007`; candidature `BUILD-003` |
| Linnaeus (`audit_b_factorization`) | Contrats, factorisation, manifeste et extension | `POLICY-003`, `STRUCT-011`, `STRUCT-017`; candidature `BUILD-003` |
| Ptolemy (`audit_c_cuda`) | CUDA, FP64, portabilité et performance | `PERF-010`, `PERF-015`; rejet motivé de `BUILD-003` |

Chaque formulation candidate a été relue contre le code, `response.md` et
`closed.md` :

- les deux formulations de `STRUCT-011` ont été fusionnées sous l'ID
  historique;
- la candidature de réouverture `STRUCT-019` sur les ressources de phases a
  été absorbée par `PERF-010`; `STRUCT-019` reste fermé car son ancien défaut
  d'ownership du gate est corrigé;
- `BUILD-003` n'est pas rouvert. Deux auditeurs ont lu littéralement la double
  compilation de sources par les targets `_experiment`; la contre-revue a
  établi que ces bibliothèques `STATIC EXCLUDE_FROM_ALL` n'ont que la macro A/B
  attendue, des consommateurs exclusifs et répondent aux expériences requises
  par `PERF-004`/`PERF-012`. Aucun défaut d'ownership ou impact runtime concret
  n'est prouvé. La condition historique est trop large pour transformer cette
  expérience isolée en régression.

La décision finale retient donc neuf constats uniques, sans doublon entre
registres. Les priorités sont huit hautes et une moyenne (`STRUCT-017`); les
neuf confiances sont `prouvée`.

## Exclusions et limites consolidées

- Validation indépendante, Premia/QuantLib et équivalents : exclus par le
  mandat; les trois exécutions accidentelles sont signalées plus haut.
- Campagne de performance officielle et rebaseline : exclus fail-closed faute
  de `power.limit` prouvable; artefacts versionnés seulement.
- Runtime SM75/SM86 et portabilité multi-GPU : exclus faute de matériel;
  compilation seulement.
- Debug, fat binary actuel et matrice complète de mutations incrémentales : non
  exécutés, donc `unknown`/`partial`.
- Nsight Compute courant : non lancé puisque la campagne officielle était
  irrecevable; les profils versionnés sont seulement relus.
- LeakSanitizer : désactivé; ASan et UBSan seulement.
- Références numériques, stress, limites et convergence de chaque famille :
  non rejoués exhaustivement.
- Exactitude des liens : test physique complet, mais appartenance Git non
  contrôlée par le checker existant.
- Logs `/tmp`, probe conceptuel et builds frais : temporaires, hashes/résultats
  consignés mais non versionnés.
- Aucun correctif, suppression de cache, téléchargement, dépendance, changement
  de baseline, commit ou publication externe n'a été autorisé ou exécuté.

## Conclusion de projet

Le dépôt présente une architecture nettement factorisée, une couverture de
build/test locale solide et des contrôles automatiques utiles. Il n'est
cependant pas auditable comme conforme au mandat version 8 : plusieurs sources
de vérité restent parallèles, les parcours publiés ne correspondent pas tous
au graphe réel, une option de build retire des targets valides et les budgets
Volterra ne couvrent pas toutes les phases exécutées. Les neuf constats sont
actionnables et bornés; les axes non démontrés restent explicitement partiels
ou exclus plutôt que crédités sur la base de succès voisins.
