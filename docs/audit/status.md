# Etat des audits

## Objet

Ce document conserve la provenance, la couverture, les exclusions et les
preuves du dernier passage de chaque audit defini dans `query.md`. Les
problemes non resolus sont exclusivement decrits dans `response.md`; les
constats fermes sont dans `closed.md`. L'audit de validation possede son propre
registre sous `../validation/` et ne figure pas dans cette matrice.

La couverture historique initiale a ete produite avec la version 1 du
referentiel. La meta-revue documentaire du 2026-08-27 a introduit la version 2
de `query.md`; elle n'etait pas un nouveau passage technique. Le second passage
de remediation du meme jour est une campagne ciblee sur les 16 constats encore
autorises, pas une recertification exhaustive de toutes les sections v2.

## Provenance du passage

- Date : 2026-08-26.
- Branche : `refactor/unify-cuda-model-contracts`.
- Revision auditee : `3bfb6a56449f60ce856f7e0734e2b72d60da1b7a`.
- Worktree deja modifie avant l'audit : oui, 133 fichiers modifies, 9 supprimes
  et 77 non suivis. Les documents `query` et `response`, avant leur deplacement
  sous `docs/audit/`, etaient deja modifies; leur `status` etait non suivi.
- Version du referentiel utilisee : version 1, anterieure a la meta-revue du
  2026-08-27.
- Limite de provenance : ni l'empreinte du diff suivi ni l'empreinte du contenu
  non suivi du snapshot initial n'ont ete conservees. La revision Git et les
  seuls comptes 133/9/77 ne suffisent donc pas a le reconstruire exactement.
- Outils : CUDA 13.3, CMake 3.22.1, GCC 14.3.0, build `dev` cible SM89.
- Mutation effectuee pendant le passage initial : uniquement les anciens
  documents `query`, `response` et `status` de l'audit.
- Separation documentaire effectuee le 2026-08-27 : quadriptyque principal
  sous `docs/audit/` et quadriptyque de validation sous `docs/validation/`.
  Aucun probleme de code ou artefact n'a ete corrige pendant cette separation.

## Premier passage de remediation

- Date : 2026-08-27.
- Perimetre : uniquement `AI_factory`, sur le worktree modifie decrit ci-dessus.
- Constats corriges, verifies et deplaces de `response.md` vers `closed.md` :
  `ANALYTICS-001`, `NUM-001`, `NUM-002`, `NUM-003`, `NUM-004`, `NUM-005`, `NAME-004`,
  `NAME-007`, `STRUCT-001`, `STRUCT-009`, `STRUCT-010`, `BOUNDARY-001`,
  `BOUNDARY-003`, `BOUNDARY-005` et `BUILD-001`.
- La construction reelle de `price_generators` sans mathDx a aussi expose deux
  restes mecaniques de migrations deja engagees : 350 recettes incluaient
  encore l'ancien chemin de validation sous `tools`, et 265 recettes equity
  n'avaient pas encore qualifie leur namespace `model::equity`. Ces appels ont
  ete alignes sur les API canoniques avant la validation finale.
- `NUM-002` est ferme apres remplacement du booleen par six causes typees,
  propagation d'un resume jusqu'a l'hote et blocage explicite des trois causes
  numeriques fatales avant publication.
- Chantiers explicitement reportes a l'issue de ce premier passage :
  `STRUCT-003`, `STRUCT-006`, `STRUCT-007` et `BOUNDARY-004`; a ce stade ils
  restaient dans `response.md` et n'etaient pas comptes comme resolus.
- `PERF-011` est ferme parce que l'experience demandee existe deja sur 354
  executions CUDA et conclut au rejet; sa signature, ses resultats et sa
  condition de reouverture sont conserves dans `closed.md`.
- `PERF-003` est ferme comme hypothese refutee : les deux reductions demandees
  ont conserve les sorties G2 bit a bit, mais la separation Gram/second membre
  regresse de 22,6 % et la distribution par warp d'environ un facteur 12.
- A l'issue de ce premier passage, quinze constats restaient ouverts : trois
  sujets de nommage, une migration de layout, un budget de policies et dix
  hypotheses de performance. Les
  migrations exigent un choix d'API non
  univoque et les constats de performance un protocole de mesure dedie; aucune
  transformation speculative n'a ete appliquee.
- En particulier, `BOUNDARY-002` traverse 471 fichiers qui consomment encore
  le booleen public `cartesian_product`. Ajouter seulement un enum canonique
  sous `src/common` laisserait deux representations runtime; le constat reste
  donc ouvert jusqu'au choix d'une migration atomique ou d'une compatibilite
  publique transitoire explicite.
- Comptabilite historique apres ce premier passage : sur 37 constats initiaux,
  19 etaient non resolus dans `response.md` (15 ouverts et 4 reportes) et 18
  fermes dans `closed.md` (15 corriges, 2 hypotheses refutees par mesure et 1
  constat de nommage classe inapplicable faute d'impact technique etabli).

## Second passage de remediation ciblee

- Date : 2026-08-27.
- Revision de base : `3bfb6a56449f60ce856f7e0734e2b72d60da1b7a`;
  worktree volontairement modifie et partage avec les extensions American/LSM.
  Au controle final avant mise a jour de ce registre, le porcelain contenait
  1 032 entrees suivies modifiees, 46 supprimees et 120 entrees non suivies.
- Autorisation utilisateur : traiter les 16 constats restants, y compris
  `STRUCT-006`, en excluant exactement `STRUCT-003`, `STRUCT-007` et
  `BOUNDARY-004`, pris en charge plus tard avec la generation automatique des
  `.cu/.cuh`, des cibles et des runners.
- Constats fermes pendant cette passe : `NAME-002`, `NAME-003`, `NAME-006`,
  `STRUCT-006`, `POLICY-001`, `BOUNDARY-002`, `PERF-001`, `PERF-002`,
  `PERF-004`, `PERF-006`, `PERF-008`, `PERF-009`, `PERF-010`, `PERF-012`,
  `PERF-013` et `PERF-014`.
- Comptabilite a l'issue de cette passe : 37 identifiants, dont 34 fermes dans
  `closed.md` et 3 reportes dans `response.md`. Aucun constat ouvert non
  reporte ne subsiste.
- Les migrations d'API (`PriceConstruction`, noms rough et unites temporelles)
  ont ete atomiques dans le worktree : aucune couche booleenne ou ancien nom
  public n'est conserve. Les templates de bindings ont seulement ete alignes
  sur ces API; la generation structurelle reportee n'a pas ete commencee.
- La qualification performance runtime porte sur le RTX 4090 Laptop SM89
  disponible. SM75 et SM86 ont recu les builds/probes de ressources de
  `POLICY-001`, mais aucune baseline runtime n'est inferee sans materiel
  correspondant.
- La validation independante cache-only CIR a ete tentee mais s'est fermee
  avant comparaison : les six caches existants portent des empreintes de
  source/policy devenues obsoletes dans ce grand worktree. Aucun cache externe
  n'a ete regenere et aucun resultat de prix independant n'est revendique par
  cette passe; les tests fonctionnels et numeriques CUDA constituent la preuve
  disponible.

## Troisieme passage de remediation structurelle ciblee

- Date : 2026-08-27.
- Revision de base : `3bfb6a56449f60ce856f7e0734e2b72d60da1b7a`;
  worktree volontairement modifie et partage avec les extensions de modeles,
  produits, datasets et validations rough.
- Autorisation utilisateur : terminer les trois constats reportes
  `STRUCT-003`, `STRUCT-007` et `BOUNDARY-004`, puis aller au bout de leur
  verification.
- Le manifeste de bindings genere maintenant toute la matrice equity non
  American et le fragment CMake consomme par les cibles. Le catalogue est
  inscrit automatiquement depuis les recettes presentes dans l'arbre.
- L'execution CUDA offline ordinaire est possedee par un runner RAII commun;
  293 recettes ont ete migrees et 10 pipelines algorithmiquement atypiques
  restent des echappatoires explicites controlees par test d'architecture.
- Le monolithe `tools/datasets/dataset.*` est remplace par quatre etapes
  compilees et testees independamment; les runners CUDA et les orchestrations
  de pricing ont leur propre frontiere sous `tools`.
- Constats fermes pendant cette passe : `STRUCT-003`, `STRUCT-007` et
  `BOUNDARY-004`. Comptabilite courante exhaustive : 37 identifiants fermes
  dans `closed.md`, aucun dans `response.md`.
- Cette campagne ferme les trois signatures ciblees et construit leurs graphes
  complets. Elle ne constitue pas une recertification exhaustive de toutes les
  sections du referentiel v2; les exclusions historiques sans constat actif
  restent decrites dans la couverture.

## Meta-revue documentaire et snapshot post-remediation

- Date du snapshot : 2026-08-27T11:53:56+02:00.
- Nature : audit des quatre documents d'audit, sans nouveau passage technique et
  sans modification du code.
- Revision de base toujours courante :
  `3bfb6a56449f60ce856f7e0734e2b72d60da1b7a`.
- Etat porcelain au moment du snapshot : 759 entrees suivies modifiees ou
  ajoutees, 46 supprimees, aucune renommee et 117 entrees non suivies. Ces
  nombres decrivent le worktree post-remediation, pas le snapshot initial.
- Empreinte SHA-256 du diff binaire suivi contre `HEAD`, hors `docs/audit/` :
  `e3088de5f0b000d1d05dc90c8e8e3ef91b7d32719c7ce4c6628e61eb8a91d7d6`.
- Fichiers non suivis hors `docs/audit/` : 212 fichiers; empreinte SHA-256 du
  manifeste trie :
  `4ca73e994ff542321420d8ef545374121af97c0bd7b7ca388a17c08a99f9d87b`;
  empreinte SHA-256 de la liste triee de leurs checksums de contenu :
  `a0b0d7882a7dd009cb236710da6327fec5078a2f252934aebd80f1537d3e759f`.
- `docs/audit/` est exclu de ces empreintes pour eviter une auto-reference; son
  contenu exact devra etre capture par le commit qui versionnera le
  quadriptyque.
- `NAME-005` a ete retire des constats actionnables et transfere vers
  `closed.md` : la notation affine et les types concernes sont deja codifies par
  le contrat, et aucun impact technique distinct d'une preference de nommage
  n'etait etabli.
- Les valeurs de ressources E6 n'ont pas ete regenerees apres les corrections.
  Elles restent des observations du snapshot initial et sont marquees a
  rafraichir dans les hypotheses de performance qui les reutilisent.

## Registre des preuves

| Ref. | Preuve executee ou inspectee | Resultat utile |
|---|---|---|
| E1 | `rg`, `rg --files`, `find`, `wc`, lecture des contrats, sources, CMake, catalogues, datasets et tests | Inventaire complet des familles et dependances; 376 generateurs, 46 loaders `src`, 27 `.cu` inclus a 304 endroits. Le journal exhaustif des commandes n'a pas ete conserve : sous le referentiel v2, E1 reste un inventaire historique et non une preuve exactement rejouable. |
| E2 | `cmake --preset dev`; build `ai_factory_host_tests`; build `ai_factory_cuda_tests`; no-op `cmake --build build-dev --target ai_factory_cuda_tests -j2` | Configuration reussie avec mathDx; cibles host et CUDA compilees; no-op 0,03 s. |
| E3 | `ctest --test-dir build-dev` cible sur les trois tests de datasets host | 3/3 tests host passes. |
| E4 | `ctest --test-dir build-dev -L cuda --output-on-failure` apres build de la matrice CUDA | 47/47 tests retour 77 (`Skipped`) faute de GPU accessible; aucun resultat runtime n'est compte comme passe. |
| E5 | Configuration fraiche dans `/tmp` avec `BUILD_TESTING=OFF`, sans `AI_FACTORY_MATHDX_ROOT`, puis build `generate_rough_bergomi_european_calls_01` | Configure reussi, link echoue sur CUDA runtime et `launch_european_option_cuda<call>` absents. |
| E6 | `cuobjdump --dump-resource-usage`, `stat`, `ar` sur les archives SM89 du passage initial | LSM 88/80 registres; Bates Phoenix 93-94 registres et 32 octets stack; rough FFT8192 139 registres; archives rough 9,73 Mo; CIR swaption 8,3 fois Vasicek. Artefacts et commandes detaillees non conserves; preuve pre-remediation a rafraichir avant toute baseline ou decision. |
| E7 | Comparaison reexecutee le 2026-08-27 : `rg -l -0 'wall_seconds:' catalog`, puis extraction AWK des deux flottants quotes et filtre `kernel_seconds > wall_seconds` pour chaque fichier | 9 catalogues : Bates American call/put, Heston American call/put, NIG American put, Variance Gamma American call/put et CIR Bermudan payer/receiver. Exces de 0,612 % a 6,469 %; le maximum est CIR Bermudan receiver. Les YAML appartiennent au snapshot post-remediation. |
| E8 | Build `all_models` SM89 apres renommage des fragments inclus | 315 etapes terminees; toutes les familles equity, rough et fixed income compilees. |
| E9 | Build agrege `ai_factory_cuda_tests ai_factory_host_tests` | Matrice complete compilee, y compris les nouveaux tests numeriques et les 48 executables CUDA. |
| E10 | CTest cible sur les trois tests de datasets host, puis `ctest -L cuda` hors sandbox | 3/3 host et 48/48 CUDA passes sur GPU. Le test numerique couvre epuisement et stagnation FP32 Jamshidian scalaire/cooperatif, frontiere LSM FP64 avec ecart de cashflow borne, grille Schobel-Zhu FP64 de 10 points et mapping explicite de trois normales Philox. Un premier essai fail-fast LSM avait produit 2 echecs sur 48; il a ete retire parce que le booleen ne distinguait pas les causes, preuve conservee dans `NUM-002`. |
| E11 | Configuration fraiche sans `AI_FACTORY_MATHDX_ROOT` dans `/tmp`, puis construction reelle de `price_generators` | Les 1 193 etapes de la cible passent; les 350 recettes utilisent la validation sous `common`, les 269 recettes equity markoviennes ont un namespace modele qualifie, et aucune cible rough dependante de mathDx n'est publiee. |
| E12 | Recherches statiques finales et correspondance des `.cu` de `src`/`tests` avec `build-dev/build.ninja` | Aucun include `.cu`, include `tools` sous `src`, include produit dans les analytics modele, inventaire README recopie ou `.cu` de ce perimetre hors graphe CMake. |
| E13 | Meta-revue : inventaire des identifiants, controles croises des champs, `git rev-parse HEAD`, `git status --porcelain=v1`, empreinte de `git diff --binary HEAD` et empreintes des fichiers non suivis hors `docs/audit/` | 37 identifiants avec un seul etat courant apres reclassement de `NAME-005`; 21 non resolus et 16 fermes; snapshot post-remediation consigne ci-dessus. |
| E14 | Build SM89 agrege, sept tests LSM cibles puis matrice `ctest -L cuda`; `compute-sanitizer` memcheck sur regresseur et Bermudan, racecheck sur Bermudan; executions des generateurs OU/G2 sous `/tmp` | Toutes les causes `RegressionStatus` sont classees; seuls statistiques/coefficients non finis et Cholesky sont fatals. 49/49 tests CUDA et 5/5 tests host cibles passent, memcheck rend 0 erreur, racecheck 0 hazard, les prix OU/G2 payer/receiver sont reproductibles bit a bit et les anciens cas Heston/Levy sont diagnostiques `insufficient_candidates`. |
| E15 | Diagnostics de kernels, profils Nsight Systems et comparaisons JSON sur les generateurs Heston American, OU et G2 Bermudan; deux prototypes temporaires mesures puis retires | Baseline G2 : regression 118 registres, aucun local, occupation theorique 33,3 %, environ 0,415 s kernel. Separation Gram/RHS : 103/78 registres, occupation Gram inchangee et +22,6 %. Distribution par warp : sorties bit-identiques mais environ x12. Le source final revient a la baseline combinee. |
| E16 | Migration atomique et recherches statiques des APIs de nommage/indexation; tests `catalog_contract`, `dataset_loaders`, `rough_sabr_dataset_loader`; compilation manuelle du probe rough SABR | `PriceConstruction` remplace le booleen public; les launchers rough sont qualifies par modele; les noms temporels portent leur unite; 46 loaders utilisent la lecture commune. Les tests d'unites 252/360/365, erreurs de loader et transposition de schedules passent. |
| E17 | `policy_size_budgets_cuda`, builds offline SM75/SM86, `cuobjdump --dump-resource-usage` et diagnostics runtime SM89 | Caps 128/256/2 048 octets verifies. SM75 : 64/63/5 registres; SM86 et SM89 : 40/40/8 pour les trois probes; shared exacte 2 048 octets, occupation theorique SM89 100 %, aucune taille hors cap acceptee. |
| E18 | Protocole v1 et benchmarks generique, CIR, schedules et Volterra sur RTX 4090 Laptop SM89; JSON versionne et `check_baseline.py` | Decisions chiffrees de `PERF-001/002/004/006/008/009/012/013/014` conservees dans `validation/performance`. Gains retenus : decodeur 32 bits, noinline CIR, ELLPACK cooperatif, FFT8192 a 16 elements/thread et chunk Volterra 65 536. Hypotheses rejetees : geometries Phoenix plus petites, FP32 generique, cache de pointeurs, convolution directe et multi-stream. Les compteurs Nsight Compute de PERF-004 restent indisponibles (`ERR_NVGPUCTRPERM`). |
| E19 | Build agrege `all_models ai_factory_host_tests ai_factory_cuda_tests performance_benchmarks`; CTest host puis `ctest -L cuda --output-on-failure`; `git diff --check`; parse JSON et compilation/aide du checker Python | Build SM89 termine en 563 etapes; 3/3 tests host et 50/50 tests CUDA passent sur GPU. Baseline JSON valide et verifier executable. |
| E20 | Execution cache-only des six validations independantes CIR apres modification numerique de l'inlining | 0 comparaison executee : les six pipelines refusent fail-closed les empreintes source/policy obsoletes. Cette limite est explicite; aucune regeneration Premia/QuantLib ni modification de l'audit de validation. |
| E21 | Configuration fraiche GCC 14/CUDA 13.3 SM89 sans mathDx; builds `price_generators`, `parameter_generators` et tests host; build `all_models` dans la configuration mathDx; generation temporaire avec `--compare-root`; checker d'architecture; test CUDA du runner hors sandbox; `git diff --check` | `price_generators` termine 1 156/1 156, tous les generateurs de parametres sont lies, et `all_models` mathDx termine 784/784. Neuf tests host/architecture passent, puis `cuda_pricing_runner` passe sur GPU en 2,41 s. Les 757 sorties codegen concordent, 382 recettes sont controlees, 293 utilisent le runner et exactement 10 echappatoires revues restent; aucun ancien monolithe/helper ni erreur de whitespace ne subsiste. |

Les chemins sous `/tmp`, les rapports Nsight et les objets de build sont des preuves locales non
versionnees : leurs resultats chiffres sont recopies ci-dessus, mais aucun lien
durable ne pretend les exposer. Cette limite interdit de les traiter comme une
baseline courante sous le referentiel v2. E15 constitue une campagne ciblee de
remediation, pas une baseline generale du projet. Aucun passage
`compute-sanitizer` complet n'est deduit de ces profils.

## Couverture courante

Le statut `complet` de Naming reprend la conclusion historique du referentiel
v1, completee par la verification ciblee E16. Les sections non recertifiees
selon toute la matrice du referentiel v2 restent `partiel` ou `non execute`,
meme lorsque tous leurs constats connus sont fermes.

| Audit | Statut | Date | Revision | Perimetre et exclusions | Preuves |
|---|---|---|---|---|---|
| Dynamics | partiel | 2026-08-26; verification ciblee 2026-08-27 | `3bfb6a5`; worktree modifie | Passage initial : tous les couples parameters/state/dynamics des modeles equity, rough et fixed income; concepts, mapping aleatoire et symetrie des tests inspectes; execution GPU d'audit exclue. Remediation : matrice fonctionnelle GPU executee, sans campagne de moments complete. | E1, E2, E4, E10 |
| Analytics | partiel | 2026-08-26; verification ciblee 2026-08-27 | `3bfb6a5`; worktree modifie | Passage initial : Black-Scholes, familles affines un/deux facteurs, compositions de courbes, concepts et call graph; references numeriques runtime GPU exclues. Remediation : tests fonctionnels GPU executes, sans campagne de reference exhaustive. | E1, E2, E4, E10 |
| Numerical robustness | partiel | 2026-08-26; verification ciblee 2026-08-27 | `3bfb6a5`; worktree modifie | Passage initial : dynamics, Jamshidian, noncentral-chi-square, LSM, reductions et rough inspectes; campagnes FP32/FP64 et GPU exclues. Remediation : tests limites cibles, matrice GPU et causes LSM typees executes, sans campagne de sensibilite exhaustive. | E1, E2, E4, E10, E14 |
| CUDA execution and memory safety | non execute | 2026-08-27 | referentiel v2; snapshot post-remediation | Nouvelle section ajoutee par la meta-revue. Aucun passage ASan/UBSan, compute-sanitizer memcheck, racecheck, initcheck ou synccheck; aucune conclusion de surete ne doit etre deduite des seuls tests fonctionnels E10. | E10, E13 |
| Naming | complet | 2026-08-26; verification ciblee 2026-08-27 | `3bfb6a5`; worktree modifie | Namespaces, symboles publics, extensions, unites et noms temporels sur `src`, `tools`, `tests` et CMake; launchers rough, namespace produit et suffixes temporels recertifies. | E1, E16, E19 |
| Project structure | partiel | 2026-08-26; verifications ciblees 2026-08-27 | `3bfb6a5`; worktree modifie | Loaders communs, matrice de bindings generee, inscriptions CMake et runners de generateurs corriges et verifies. Les constats connus de cette section sont fermes; aucun passage v2 complet de l'arbre n'est infere de cette campagne ciblee. | E1, E2, E16, E19, E21 |
| Exercise and dynamics-family boundaries | partiel | 2026-08-26; verification ciblee 2026-08-27 | `3bfb6a5`; worktree modifie | Passage initial : European/LSM, American/Bermudan, schedules, continuation states et separation Markov/rough inspectes; prix runtime GPU exclus. Remediation : tests fonctionnels GPU executes sans validation independante exhaustive des prix. | E1, E2, E4, E10 |
| Pricing policies and concepts | partiel | 2026-08-26; verification ciblee 2026-08-27 | `3bfb6a5`; worktree modifie | Concepts closed form, Monte Carlo, LSM, sample et rough; caps documentes et verifies par compilation SM75/86/89, ressources runtime mesurees seulement sur SM89. | E1, E2, E17, E19 |
| Tools and src ownership | partiel | 2026-08-26; verifications ciblees 2026-08-27 | `3bfb6a5`; worktree modifie | Frontiere runtime/offline, indexation typee et decomposition sampling/assemblage/serialisation/execution verifiees. Les constats connus de cette section sont fermes; aucun passage v2 complet de tout `tools` n'est infere. | E1, E2, E16, E19, E21 |
| Build and CUDA instantiations | partiel | 2026-08-26; verification ciblee 2026-08-27 | `3bfb6a5`; worktree modifie | Configuration dev, build host/CUDA/performance, frontiere mathDx et generateurs sans mathDx; probes policy SM75/86. Build propre de tous les presets et runtime hors SM89 exclus. | E2, E5, E8, E9, E11, E17, E19 |
| Generic CUDA performance | partiel | 2026-08-26; campagne ciblee 2026-08-27 | `3bfb6a5`; worktree modifie | Indexation, geometries, FP64, couts fixes, schedules, taille de code et baseline runtime SM89 mesures selon protocole v1. Baselines runtime SM75/86 et compteurs hardware detailles exclus. | E1, E7, E17, E18, E19 |
| Early-exercise performance | partiel | 2026-08-26; verification ciblee 2026-08-27 | `3bfb6a5`; worktree modifie | Workspace, batching, boucle backward et reductions inspectes. Ressources et temps SM89 mesures sur Heston American, OU et G2 Bermudan; les deux reductions candidates de PERF-003 sont rejetees. La campagne ne couvre pas toutes les geometries ni architectures. | E1, E6, E14, E15 |
| Rough FFT performance | partiel | 2026-08-26; campagne ciblee 2026-08-27 | `3bfb6a5`; worktree modifie | Crossover direct/FFT, courbe FFT, instanciation 8 192, workspace et chunking mesures sur SM89 avec baseline v1; runtime SM75/86 et profils de compteurs detailles exclus. | E1, E2, E18, E19 |
