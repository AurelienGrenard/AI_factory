# Etat des audits principaux

## Objet

Ce document consigne le snapshot, la couverture, les exclusions et les preuves
du dernier passage du referentiel principal `query.md`. Les constats non
resolus sont exclusivement dans `response.md`; les constats fermes sont dans
`closed.md`. L'audit lent des references independantes reste dans
`docs/validation/` et n'est pas inclus ici.

## Snapshot du passage structure, naming et codegen version 4

- **Date :** 2026-08-28.
- **Referentiel :** `docs/audit/query.md`, version 4 du 2026-08-28, durci sur
  la navigabilite de l'arborescence, les noms de fichiers et les en-tetes.
- **Branche :** `refactor/model-layout-codegen-volterra`, empilee sur
  `refactor/unify-cuda-model-contracts`.
- **Revision de base :** `b85097e58edb1d038402230515786ecd5ce51b6d`.
- **Portee :** passage partiel exhaustif sur les 1 031 fichiers C++/CUDA sous
  `src/model` (832 bindings modele-produit et 199 fichiers
  d'infrastructure), leurs references CMake/tests/catalogue/validation, et les
  35 templates du codegen pricing/sampling/catalogue. La query Structure est
  durcie pour les futurs passages sur le reste du depot, mais ce passage ne
  recertifie pas tous les autres dossiers.
- **Etat audite :** worktree deja sale et modifie concurremment. Le snapshot
  initial exact de ce passage n'est pas reconstructible. Apres remediation et
  hors les quatre registres `docs/audit/{query,response,status,closed}.md`,
  l'etat contient 1 639 lignes porcelain apres detection des renommages.
  Empreinte de cette liste :
  `45e8936c0d42e006fc512e51c35ae64c7a0d2de06b9359f8ce6a1e6d705a5277`;
  empreinte du diff binaire suivi :
  `b9d273dcb8052d478d76334172decfd90ea6f5f6c337af1687d0f53fa324124a`.
  Aucun fichier non suivi ne subsiste dans ce snapshot.
- **Mutations :** oui. Les bindings modele-produit ont ete isoles, les noms
  de helpers ambigus et les en-tetes d'infrastructure ont ete corriges, les
  templates ont ete classes par artefact et engine, et un checker bloquant a
  ete ajoute. Le constat fixed-income closed form `STRUCT-015` est seulement
  documente et reste reporte.

### Registre des preuves structure, naming et codegen

- **E26 — Navigation modele et codegen.** Les 832 fichiers modele-produit
  forment 416 paires `.cuh/.cu` exclusivement sous `product/`; aucun fichier
  d'infrastructure n'y subsiste. Les 199 fichiers d'infrastructure hors
  `product/` utilisent un vocabulaire de noms explicitement revu et commencent
  par une phrase specifique; les paires publiques/`*_impl.cuh` distinguent
  contrat et definitions. Les helpers rough portent maintenant
  `volterra_fft_*` ou `markovian_n_factor_*`. Les 35 templates sont classes
  sous `pricing/`, `sampling/` et `catalog/`, puis par engine, et les artefacts
  C++ complets ne vivent plus en chaines multilignes dans `generate.py`.
  `check_model_layout.py` couvre chemins, profondeur, ownership, noms,
  en-tetes, paires et references obsoletes. La regeneration compare bit a bit
  1 407 sorties; les quatre tests unitaires du manifeste, le checker des 689
  recettes, la configuration CMake et les quatre CTests architecture/codegen
  passent. Les builds representatifs precedemment compiles sur ce worktree
  passent 84/84. Ce sous-passage n'execute aucune mesure de performance ni
  nouveau test CUDA runtime; E27 consigne separement les executions GPU de la
  remediation Volterra parallele.
- **E27 — Contrats Gaussian-Volterra FFT.** Le vocabulaire de driver est
  remplace atomiquement par `HybridKernelPolicy`, `PreparedKernel`,
  `kernel_parameters`, `volterra_variance` et
  `reconstruct_volterra_value`. `src/common/volterra/concepts.cuh` formalise
  les contrats kernel/path et leur relation de types. Le moteur possede
  `sqrt(dt)` dans son `PreparedRow` et traite `PreparedKernel` comme opaque;
  equations, ordre numerique et mapping Philox restent inchanges. Les quatre
  pricers europeens et quatre samplers compilent; cinq tests CUDA cibles
  passent sur SM89. La verification finale execute aussi 35 tests Python
  Volterra, la generation complete zero-diff, les quatre CTests
  architecture/codegen et la recherche sans ancien symbole sous code,
  codegen, tests, validation et CMake.

### Matrice de couverture structure, naming et codegen

| Axe | Statut | Couverture | Exclusions | Preuves |
|---|---|---|---|---|
| Structure et ownership de `src/model` | complet sur le perimetre | 1 031 fichiers, profondeur, paires produit, courbes fixed income, references et CMake | Autres arbres `src/common`, `src/product`, `tools` hors codegen | E26; fermeture `STRUCT-016` |
| Naming et en-tetes de `src/model` | complet sur le perimetre | 199 fichiers hors `product/`, noms canoniques ou method-specific, phrase de contenu/utilite, distinction public/impl | Noms de symboles et variables non reaudites exhaustivement | E26; fermeture `NAME-011` |
| Navigabilite du codegen | complet hors fixed income closed form | 35 templates classes par artefact/engine, renderer sans artefact complet inline, zero-diff 1 407 sorties | Templates fixed-income closed form reportes | E26; fermeture `STRUCT-017`; `STRUCT-015` |
| Naming et frontieres des policies Volterra | complet | Kernel/path concepts, relation de types, `PreparedKernel` opaque, propagation pricing/sampling/modeles | Aucun benchmark de performance rejoue | E27; fermetures `NAME-100`, `POLICY-002` |

### Exclusions du passage structure, naming et codegen

- L'audit exhaustif des noms, en-tetes et profondeurs dans les autres arbres du
  depot; la query durcie les exigera lors d'un passage complet ulterieur.
- Les templates et recettes fixed-income closed form, reportes sous
  `STRUCT-015`; aucun debut de remediation n'est revendique.
- Les axes homogeneity, factorization, numerique, CUDA safety, performance et
  validation independante non directement affectes.
- Toute campagne de performance, conformement a l'indisponibilite actuelle de
  la machine pour une mesure stable; aucun resultat SM89 ou autre GPU n'est
  infere.

### Comptabilite apres ce passage

- **Ouverts : 3** — `STRUCT-015`, `NUM-007` et `PERF-010`.
- **Fermes : 49** dans `closed.md`; `STRUCT-016`, `STRUCT-017` et `NAME-011`
  y sont transferes apres remediation et preuves E26.
- **Correction de code :** oui, limitee aux frontieres, noms, en-tetes,
  codegen, references et gardes structurelles de ce passage.

## Snapshot du passage partiel version 4

- **Date :** 2026-08-28.
- **Referentiel :** `docs/audit/query.md`, version 4 du 2026-08-28.
- **Branche :** `refactor/unify-cuda-model-contracts`.
- **Revision de base :** `1889f7fbbd0f81ce2b06cfb16645e48a43971aa0`.
- **Portee :** passage partiel limite aux nouveaux axes portabilite/tuning GPU
  et model samples, plus verification ciblee des remediations presentes. Les
  autres axes ne sont pas recertifies sur ce worktree par ce passage.
- **Etat audite initial :** worktree deja tres sale et modifie concurremment par
  plusieurs agents. Avant la remediation, l'etat technique hors quatre fichiers
  `docs/audit/{query,response,status,closed}.md` contient 300 lignes porcelain,
  dont 101 fichiers non suivis. Empreinte de la liste d'etat :
  `2d2ed3177c28a53e9c1b4322d94cada983a5d3af0f8bb5b6882224d63253dec4`;
  empreinte du diff binaire suivi :
  `9a5b03c5ad6b1927276f7cd02c00804da9fdf046f677aa09aaf32f3eae821867`;
  empreinte agregee des noms/contenus non suivis :
  `1d7bf75628c09183b5d80874d090465ef8d38b501dce816914696df75c7ae800`.
- **Mutations :** le passage d'audit initial n'a modifie que son registre. La
  remediation ulterieure a modifie code, CMake, tests, codegen, recettes et
  contrats dans le meme worktree; E25 distingue les preuves acquises des deux
  campagnes lourdes encore exclues. L'ensemble reste sale jusqu'au commit de
  la pull request.

### Environnement d'execution version 4

- CMake 3.22.1, GCC/G++ 14.3.0, nvcc CUDA 13.3.73, builds `Release`, sans
  `--use_fast_math`; runtime RTX 4090 Laptop, compute capability 8.9.
- mathDx/cuFFTDx 26.06.1 pour CUDA 13 disponible sous le chemin deja consigne
  dans le snapshot version 3.
- Les executions CUDA ont ete faites explicitement hors sandbox. Les essais
  sandboxes refusent l'acces au driver et ne sont pas comptes comme echecs du
  code.

## Registre des preuves version 4

- **E19 — Referentiel et contrats etendus.** Lecture des instructions, du
  registre ferme et des contrats samples/dynamics/CMake/performance. La query
  version 4 integre les samples dans structure, homogeneite, factorisation,
  codegen, CMake, numerique, CUDA et performance. Elle ajoute un axe explicite
  de portabilite : SM89 est un profil mesure de reference, pas la seule cible;
  invariants et parametres de tuning sont separes et chaque autre GPU doit etre
  rebenchmarke avec sa propre provenance.
- **E20 — Matrice codegen et samples finale.** Le manifeste genere declare 12
  engines, 24 modeles, 26 produits, 689 datasets disponibles, zero report et
  1 407 sorties. Le tree contient 24 `sample.cu`, 24 `sample.cuh`, 24 helpers
  types et 48 recettes. `generate.py --family all --compare-root .`, le checker
  des 689 recettes et les quatre tests unitaires du manifeste passent; CMake
  `dev` se configure et la generation est zero-diff.
- **E21 — Portabilite de build.** Le preset local SM89 reste utilisable. Un
  build frais sans mathDx compile les 40 recettes samples independantes en 168
  etapes. A l'inverse, une configuration fraiche avec mathDx et
  `CUDA_WORKBENCH_ARCHITECTURES=86` echoue avant compilation parce que
  `AIFactoryTargets.cmake` exige exactement `89`, bien que le projet annonce
  `75;86;89`; preuve de `BUILD-002`, pas preuve d'incompatibilite cuFFTDx.
- **E22 — Execution des model samples.** L'agregat avec mathDx compile les 48
  generateurs. Une campagne GPU execute les deux layouts des 24 modeles :
  47/48 smoke tests ecrivent et rechargent 1 000 lignes conformes. Quadratic
  rough-Heston `samples_02` echoue deux fois au meme endroit avec un spot non
  fini ligne 77; ses bornes sont les bornes core du generateur de parametres.
  L'orchestration memoire, les offsets Philox et les profils communs
  `{256,4096}` ont aussi ete inspectes. Les harness/baselines performance ne
  contiennent aucun workload sample. Les quatre tests communs
  `sample_dataset_stage`, rough Volterra, rough Heston et Black-Scholes passent.
- **E23 — Remediations ciblees presentes.** Les preuves deja obtenues sur ce
  snapshot ferment `STRUCT-010`, `STRUCT-011`, `STRUCT-012`, `STRUCT-013`,
  `NUM-006` et `CUDA-001` : zero-diff codegen, builds American/samples, tests
  CUDA SABR/rough-SABR, smoke tests samples sauf le defaut transfere sous
  `NUM-007`, et initcheck Bermudan passe de 1 792 a zero erreur. Le checker de
  baseline est maintenant fail-closed sur ses 22 cles v3, mais `PERF-010`
  reste ouvert pour la nouvelle surface samples v4.
- **E24 — Retrait volontaire et reference morte.** Le proprietaire confirme
  avoir demande la suppression de `docs/deferred-work.md`; le fichier et son
  lien ne doivent donc pas etre restaures. `AGENTS.md` exige toutefois toujours
  sa lecture. Cette seule instruction devenue impossible ouvre `STRUCT-014`.
- **E25 — Remediation portabilite, QRH et baseline samples.** L'instruction
  morte est retiree. cuFFTDx 26.06 utilise une matrice explicite et une cible
  CMake d'interface; pricing et sample compilent fraichement pour SM75, SM86,
  SM89 et le fatbin `75;86;89`, execute sur le SM89 disponible. Les geometries
  pricing/LSM/samples sont centralisees sous le profil
  `sm89_reference_v1`, surchargeables par CMake et publiees avec provenance;
  codegen complet et checker des 689 recettes passent. QRH equilibre sa cellule
  Volterra : regression CUDA ligne 77, smoke `samples_02` 1 000 lignes et
  reference Python avec raffinement temporel passent. Le manifeste performance
  contient 30 cles dont huit samples executes isolement; ses tests unitaires et
  l'agregat de build passent. Le preflight QRH 3M/stress/facteurs et la campagne
  performance complete de 30 cles n'ont pas ete acheves et restent requis sous
  `NUM-007` et `PERF-010`.

## Matrice de couverture du passage partiel version 4

| Axe | Statut | Couverture | Exclusions | Preuves |
|---|---|---|---|---|
| Structure et nettoyage lies aux samples | partiel | Arborescence, bindings, helpers, recettes et retrait volontaire du registre differe; instruction morte retiree | Revue generale de tous les renommages du worktree non reprise | E19, E20, E24, E25; fermeture `STRUCT-014` |
| Codegen model samples | complet structurel | 24 modeles, deux layouts, bornes core, observables, CMake, provenance, zero-diff et builds avec/sans mathDx | Validite numerique du modele separee sous NUM | E20--E23; fermeture `STRUCT-011`, `STRUCT-013` |
| CMake et matrice d'architectures | complet pour la compatibilite declaree | Builds mathDx SM75/86/89 et fatbin, runtime fatbin SM89, build sans mathDx, refus cible des descripteurs absents | Runtime SM75/SM86 et architectures futures faute de materiel | E21, E25; fermeture `BUILD-002` |
| Portabilite et tuning materiel | complet structurel | Profil central, provenance, surcharges CMake, metadonnees et workflow pricing/samples/LSM/rough | Aucune mesure runtime hors RTX 4090 Laptop/SM89 | E19, E21, E22, E25; fermeture `PERF-015` |
| Robustesse numerique des samples | partiel | Correction QRH, ligne 77, smoke 1 000 lignes, ligne extreme 680 et raffinement temporel | Preflight 3M, stress, facteurs et validations de prix independantes | E20, E22, E25; `NUM-007` |
| Execution et memoire CUDA samples | partiel | Geometries, offsets, strategie persistent grid/block, garde memoire et deux layouts executes | Sanitizers representatifs non rejoues sur les 24 bindings; aucun autre GPU | E19, E22 |
| Performance model samples | partiel | Huit workloads, deux layouts, kernel/publication wall, baseline SM89 et checker 30 cles | Campagne appariee complete interrompue; aucun runtime hors SM89 | E19, E22, E25; `PERF-010` |

## Exclusions du passage version 4

- Les axes version 3 non directement affectes par portabilite ou samples; leur
  couverture historique reste ci-dessous mais n'est pas promue au snapshot
  sale courant.
- Les mesures runtime sur SM75, SM86 ou toute autre architecture, faute de
  materiel correspondant. Aucune performance n'est inferee depuis le SM89.
- L'execution complete des 48 datasets de 3 000 000 lignes, le preflight QRH
  core/stress/multi-facteurs et les validations de prix independantes affectees.
- La campagne performance appariee des 30 cles; son lancement a ete interrompu
  et aucun resultat partiel n'est compte comme succes du gate.
- L'audit lent `docs/validation/`, le contenu ignore du website et les compteurs
  Nsight Compute deja inaccessibles dans le passage version 3.

## Comptabilite des constats du passage version 4

- **Ouverts : 3** — `STRUCT-015`, `NUM-007` et `PERF-010`.
- **Fermes : 44** dans `closed.md`; `STRUCT-014`, `BUILD-002` et `PERF-015`
  quittent le registre apres remediation et preuves E25.
- **Correction de code :** oui, dans la phase de remediation E25; le passage
  d'audit qui a precede cette phase est reste read-only hors registre.

## Snapshot du passage version 3

- **Date :** 2026-08-27.
- **Referentiel :** `docs/audit/query.md`, version 3 du 2026-08-27.
- **Branche :** `refactor/unify-cuda-model-contracts`.
- **Revision auditee :** `1889f7fbbd0f81ce2b06cfb16645e48a43971aa0`.
- **Etat initial :** worktree propre; `git status --porcelain=v1` et
  `git ls-files --others --exclude-standard` ne produisaient aucune ligne. La
  revision suffit donc a reconstruire le snapshot technique initial.
- **Etat apres audit :** seuls `docs/audit/status.md` et
  `docs/audit/response.md` ont ete reecrits; `docs/audit/closed.md` a perdu les
  deux entrees reellement rouvertes `STRUCT-010` et `PERF-010`. Aucun code,
  CMake, test, generateur, recette ou contrat d'implementation n'a ete corrige.
- **Site :** `AI_factory_website/` est ignore par `.gitignore`, contient
  `equations/`, `index.html` et `static/`, et resout vers le depot parent plutot
  que vers une revision propre. Son contenu exact n'est pas reconstructible par
  le snapshot Git et est exclu de la certification; seule sa frontiere avec le
  depot suivi a ete inspectee.

### Environnement d'execution

- CMake 3.22.1, GCC/G++ 14.3.0, nvcc CUDA 13.3.73, build `Release`, sans
  `--use_fast_math`, architecture runtime SM89.
- NVIDIA GeForce RTX 4090 Laptop GPU, compute capability 8.9, 76 SM,
  17 170 956 288 octets, driver API 13020, runtime 13030.
- mathDx/cuFFTDx disponible sous
  `/home/aurelieng/opt/nvidia-mathdx-26.06.1-cuda13/nvidia/mathdx/26.06`.
- Nsight Systems 2026.1.3; Nsight Compute 2026.2.1 present. Un profilage d'un
  kernel reel execute explicitement hors sandbox reste bloque par le pilote avec
  `ERR_NVGPUCTRPERM`; un executable sans kernel se connecte mais ne collecte
  naturellement aucun compteur.

## Registre des preuves

- **E01 — Instructions et contrats.** Lecture complete de `AGENTS.md`,
  `README.md`, `docs/README.md`, `docs/audit/{query,closed,status,response}.md`,
  puis des contrats d'extension, dynamics, analytics,
  pricing CUDA, early exercise, diagnostics de launch, generation de
  parametres et samples, et protocole performance.
- **E02 — Inventaire et ownership.** Parcours des arbres suivis, tailles,
  extensions, includes, namespaces et consommateurs. Environ 7 376 fichiers
  hors builds/datasets, dont 1 243 sous `src`, 1 025 sous `catalog`, 60 sous
  `tests`, 59 sous `tools` et 4 956 sous `validation`; 478 `.cu`, 626 `.cuh`,
  729 `.cpp` et 121 `.hpp`. Aucun include textuel de `.cu`, aucun header source
  sans consommateur et aucune TU `src`/`tests` absente de `compile_commands.json`.
- **E03 — Matrices de contrats.** Comparaison des dynamics exactes, pas fixe,
  rough FFT et lifts N-facteurs, des analytics equity/fixed income et des
  policies/schedules/continuation states avec leurs concepts et tests. Les
  differences observees hors `NUM-006` sont justifiees par capacite
  mathematique ou engine.
- **E04 — Pyramide de factorisation.** Inspection des moteurs
  `simulation`, `monte_carlo`, `closed_form`, `sample`, `volterra` et
  `longstaff_schwartz`, de leurs consommateurs modeles/produits et des
  specialisations fixed income/equity. Aucun adaptateur vide ni branche runtime
  dans la boucle chaude n'a ete trouve.
- **E05 — Codegen.** Le manifeste couvre 11 modeles markoviens plus le sous-
  ensemble Black-Scholes, 6 modeles rough, 21 produits et 29 variantes. La
  commande `python3 tools/codegen/pricing_bindings/generate.py --family all
  --output /tmp/... --compare-root .` donne zero diff sur 1 279 sorties: 756
  bindings `.cu/.cuh`, 522 recettes equity ordinaires et un fragment CMake.
- **E06 — Recettes et checkers.** Inventaire de 641
  `catalog/**/generator.cpp`: 530 prix equity, 58 prix fixed income et 53
  parametres/courbes/produits. `python3 tools/cuda/check_catalog_generators.py`
  passe: 641 recettes controlees, 522 recettes equity generees, 8 escape
  hatches algorithmiques American. Les tests `pricing_binding_codegen` et
  `catalog_generator_boundaries` passent.
- **E07 — Parametres et samples.** Les 24 recettes de parametres modele et 27
  recettes produit ont toutes 1 000 lignes, 900 core/100 stress et un YAML
  adjacent complet. Dix-huit `sample.cu` existent pour 24 modeles catalogues;
  aucun fichier n'existe sous `catalog/model/**/samples/`.
- **E08 — Configuration et build courant.** `cmake --preset dev` reussit. Le
  build agrege `all_models ai_factory_host_tests ai_factory_cuda_tests
  performance_benchmarks parameter_generators price_generators -j2` reussit en
  212 etapes sur le build existant; un second passage est no-op en 0,08 s.
- **E09 — Architectures et ressources statiques.** Builds offline SM75, SM86
  et SM89 de `test_policy_size_budgets_cuda` reussis. `cuobjdump` mesure pour
  les probes payload thread 64/63 registres et 128/256 octets de stack en SM75,
  40/40 et 128/256 en SM86/89; le probe shared utilise 2 048 octets et 5
  registres en SM75, 8 en SM86/89. Le test runtime annonce 100 % d'occupation
  theorique pour ses caps.
- **E10 — Sanitizers host.** Build frais GCC 14 ASan+UBSan+LSan sous
  `/tmp/ai-factory-audit-host-sanitize-gcc14`; 7/7 tests loaders, catalogues,
  stages, codegen et frontieres passent. Le premier run sandboxe avait seulement
  l'echec LSan `ptrace`; le meme binaire hors sandbox passe.
- **E11 — Tests CUDA hors sandbox.** Les 52 tests du label CTest `cuda` ont ete
  reexecutes explicitement hors sandbox: 52/52 passent en 22,27 s. Les 9 tests
  host cibles passent en 0,23 s. Les cas couvrent closed form, simulations
  exactes/pas fixe, samples, early exercise, rough FFT et rough N-facteurs.
- **E12 — Compute Sanitizer hors sandbox.** La matrice a ete reexecutee
  explicitement hors sandbox. Memcheck: 7/7 cas sans erreur; racecheck: 4/4
  sans hazard; synccheck: 4/4 sans erreur. Initcheck: 4 cas equity/rough sans
  erreur; `test_bermudan_swaption_cuda` reproduit exactement 1 792 lectures
  globales non initialisees dans les copies de `PreparedRow`. Logs
  `/tmp/ai-factory-outside-{memcheck,racecheck,initcheck,synccheck}-*.log`.
- **E13 — Frontiere SABR.** Revue des deux `dynamics_impl.cuh`, des domaines
  core/stress et des tests. Les deux engines appliquent le meme plancher
  Lamperti `1e-12`; les tests couvrent finitude et limite `beta=1`, sans cas de
  franchissement ni contrat de frontiere.
- **E14 — Baseline performance SM89.** Reproduction des 18 cles exactes avec
  5 warmups et 21 repetitions. Le checker retourne `PASS`, avec 4 resultats
  inconclusifs pour CV > 5 %. Sur les comparaisons stables, aucune regression
  ne depasse +5 %; delta maximal +3,52 %, CIR noinline -11,27 % et un petit cas
  Volterra -8,83 %. Le +43,57 % du launcher court a des CV 36,0 %/22,6 % et
  reste inconclusif, pas un constat.
- **E15 — Profils et diagnostics.** Diagnostics runtime sans spill: OU
  Bermudan regression 77 registres/50 % d'occupation; G2 118/33,3 %; Heston
  American 88/33,3 %; Volterra 48/83,3 % avec 4 096 octets shared dynamiques;
  rough Heston N7 77/50 %. Profils Nsight Systems conserves sous `/tmp` pour
  Levy American, Heston American, Bermudan OU/G2 et Volterra. Les phases solve
  et partials dominent les traces LSM; les compteurs Nsight Compute sont
  exclus par permission.
- **E16 — Documentation et nettoyage.** `src/common/README.md` compte 985
  lignes et n'a aucune reference entrante; il enumere l'arbre et l'API. Le
  README racine cite encore des `.cu` inclus. Le contrat samples impose deux
  recettes par modele tandis que le README annonce explicitement leur absence.
- **E17 — Exhaustivite de baseline.** La baseline versionnee contient 18
  mesures. Le candidat exact de 18 cles passe avec 4 inconclusifs; le meme
  checker execute sur une seule de ces cles retourne
  `PASS: 1 measurement(s), 0 inconclusive`. Aucune cible CTest n'appelle
  `check_baseline.py` et aucune des 18 cles ne couvre Longstaff-Schwartz.
- **E18 — Dependances optionnelles.** Build frais avec mathDx du test et du
  benchmark Volterra en 34 etapes, puis test CUDA passe. Configuration fraiche
  sans mathDx et build complet `price_generators -j2` reussissent en 1 705
  etapes (`real 1296,92 s`). Aucun fallback rough FFT n'est pretendu sans
  mathDx; les targets independantes restent constructibles.

## Matrice de couverture version 3

Le snapshot de toutes les lignes ci-dessous est celui defini plus haut. Une
ligne `partiel` indique precisement la preuve inaccessible ou le protocole
manquant; elle n'est pas promue a `complet` par des tests voisins.

### Structure, conventions and ownership

| Sous-section | Statut | Couverture | Exclusions | Preuves |
|---|---|---|---|---|
| Repository tree and domain ownership | complet | Arbres `src/common/model/curve/product/generative`, `tools`, `catalog`, `validation`, `tests`, `docs`; classification markovian/rough/fixed income; frontiere du site | Contenu interne du site ignore non reconstructible | E01, E02 |
| File responsibilities and granularity | complet | Headers publics/impl, TUs `.cu/.cpp`, rows device, gros fichiers, wrappers generes, consommateurs | Mesure avant/apres d'une refactorisation non proposee | E02, E04, E08, E09 |
| Naming and semantic conventions | complet | Chemins/namespaces, noms canoniques, unites, sides, tests/targets/catalogue; recherche legacy/new/old | Preferences stylistiques sans impact | E02, E03 |
| Dependency boundaries | complet | Graphes d'includes et liens: aucun `src -> tools/catalog/validation`, aucun common -> type concret, aucune contamination equity/fixed income; mathDx cible | References externes de l'audit de validation | E02, E04, E08, E18 |
| Cleanup and extension locality | complet | Headers/TUs orphelins, doubles enregistrements, codegen/manuel, README, localite d'un ajout modele/produit/couple | Site ignore | E02, E05, E06, E16; `STRUCT-010`, `STRUCT-011` |

### Contract homogeneity

| Sous-section | Statut | Couverture | Exclusions | Preuves |
|---|---|---|---|---|
| Dynamics | complet | Exact, fixed-step, Levy/sauts, rough FFT, rough N-facteurs; parametres/preparation/etat/advance/observables/Philox et tests | References de prix independantes | E03, E11, E13; `NUM-006` |
| Analytics | complet | Black-Scholes, affines un/deux facteurs, compositions courbe-modele, providers/concepts, signes payer/receiver et tests | Certification QuantLib/Premia | E03, E04, E11 |
| Products, policies and exercise | complet | Concepts, schedules, handlers, policies, continuation states, regressors, sides compile-time, American/Bermudan | Mesure de biais LSM par reference externe | E03, E04, E11, E12 |

### Factorization pyramid

| Sous-section | Statut | Couverture | Exclusions | Preuves |
|---|---|---|---|---|
| Markovian factorization | complet | Schedules exact/fixed-step, simulation de paths, Monte Carlo, sampling et 11 familles equity plus 6 fixed income | Aucun | E03, E04, E11 |
| Rough factorization | complet | Driver fractional, hybrid schedule, sampling/pricing Volterra, rough Bergomi/SABR et log-modulated variant | Compteurs materiels n'affectant que Performance | E04, E11, E15 |
| Rough-Markovian factorization | complet | Rough Heston/quadratic/rough Stein-Stein, preparations host, factor counts 2/3/7, reutilisation du moteur markovien | Reference d'erreur N-facteurs de validation | E03, E04, E11 |
| Closed-form factorization | complet | Kernels scalaires/cooperatifs, Black-Scholes, options de taux, Jamshidian, providers et sides | Aucun | E03, E04, E11 |
| Fixed-income and equity factorization | complet | Primitives neutres, reductions, Philox, sample, LSM; specialisations financieres laissees aux domaines | Aucun | E02, E04 |
| Factorization cost and limits | complet | Branches runtime, tailles/ressources probes, TUs et rebuilds; aucune nouvelle factorisation proposee sans consommateurs | Comparaison avant/apres sans changement, inapplicable | E04, E08, E09, E15 |

### Code generation and extension cost

| Sous-section | Statut | Couverture | Exclusions | Preuves |
|---|---|---|---|---|
| Minimum hand-written model and product | complet | Inventaire de ce qui reste manuel pour modele, produit et couple; launchers, recettes, parametres, tests et CMake | Squelette de reference independante | E05, E06, E07; `STRUCT-011`--`STRUCT-013` |
| Canonical capability manifest | complet | Champs actuels, matrice resolue, doubles listes fixed income/samples/parametres/LSM et incompatibilites | Aucun | E05, E06, E07; `STRUCT-011` |
| Generated bindings and catalogue recipes | complet | 756 bindings, 522 recettes ordinaires, 8 American, 58 fixed income, 0 sample; sides, policies, engines, fragments CMake | Execution 3M impossible faute de recettes, objet du constat | E05, E06, E07; `STRUCT-012`, `STRUCT-013` |
| Parameter dataset generation | complet | 24 modeles et 27 produits, schemas, ordre 90/10, YAML adjacent, contraintes et tests loaders | Generation des references de prix | E06, E07, E10, E11 |
| Regeneration, drift and exceptions | complet | Zero-diff, outputs attendus, checker, escapes, builds avec/sans mathDx, architectures, provenance des headers generes | Execution runtime SM75/SM86 faute de GPU; compile/resources couverts | E05, E06, E08, E09, E18 |

### CMake and build graph

| Section | Statut | Couverture | Exclusions | Preuves |
|---|---|---|---|---|
| CMake and build graph | partiel | Racine/modules/fragments, toutes TUs, doubles inscriptions, globs/checkers, targets/agregats, includes/liens, mathDx, SM75/86/89, clean frais sans mathDx, build courant mathDx et no-op | Rebuild incremental apres les cinq mutations representatives et comparaison detaillee des tailles objet/archive/cubin non executes; depfiles et no-op controles | E02, E05, E06, E08, E09, E18 |

### Numerical robustness and reproducibility

| Section | Statut | Couverture | Exclusions | Preuves |
|---|---|---|---|---|
| Numerical robustness and reproducibility | complet | Domaines log/sqrt/pow/division, positivite/frontieres, limites, precision FP32/FP64, solveurs, LSM, reductions, temps, erreurs, convergence et mapping Philox dans toutes les familles | Certification de prix independante et notebooks, audit separe | E01, E03, E10, E11, E13, E14; `NUM-006` |

### CUDA execution and memory safety

| Section | Statut | Couverture | Exclusions | Preuves |
|---|---|---|---|---|
| CUDA execution and memory safety | complet | Launchers/workspaces, overflow/offsets, pointeurs, RAII, streams/sync, erreurs async; ASan/UBSan/LSan; memcheck/racecheck/initcheck/synccheck sur exact/fixed, aligned/cartesian, samples, LSM, rough FFT/N-facteurs | Toutes les specialisations catalogue ne sont pas rejouees sous sanitizer; matrice representative exigee couverte | E10, E11, E12; `CUDA-001` |

### Performance

| Sous-section | Statut | Couverture | Exclusions | Preuves |
|---|---|---|---|---|
| Common performance protocol and kernel strategy | partiel | Protocole v1, environnement SM89, 18 workloads, diagnostics de ressources, cubin SM75/86/89, Nsight Systems, flags et invariants | Compteurs Nsight Compute occupation atteinte/stalls/caches/branches/debit interdits par `ERR_NVGPUCTRPERM`; gate incomplet `PERF-010` | E09, E14, E15, E17; `PERF-010` |
| Generic CUDA performance | partiel | Closed form, index, accumulation, geometries riches, CIR inline/noinline, schedules regular/ragged, overhead et ressources | Compteurs materiels Nsight Compute; les quatre comparaisons bruitees doivent etre rejouees | E09, E14, E15 |
| Early-exercise performance | partiel | Profils descriptifs equity un etat NIG/VG, equity multi-etats Heston, taux un facteur OU et deux facteurs G2; workspace, phases, registres/spills/occupation | Pas de baseline versionnee multi-dimensions prix/paths/dates/blocs; pas de compteurs Nsight Compute. Toute geometrie alternative reste **a mesurer** | E11, E15, E17 |
| Rough performance | partiel | Six workloads baseline Volterra, crossover/tuning/chunking historiques, complements Heston/rough Heston N7/rough SABR, ressources et decomposition Nsight Systems | Compteurs Nsight Compute; runtime SM75/86; courbe d'erreur N-facteurs independante reservee a validation | E09, E11, E14, E15 |

## Exclusions transversales

- L'audit `docs/validation/query.md`: Premia, QuantLib, caches de 1 000 prix,
  fingerprints, notebooks et certification des 900/100 lignes. Aucun statut ni
  artefact de cet audit separe n'a ete modifie.
- Le contenu exact de `AI_factory_website/`, ignore et sans revision propre.
- Les mesures runtime SM75 et SM86, faute de materiel; leurs builds et
  ressources statiques sont couverts, aucune performance runtime n'en est
  inferee.
- Les compteurs Nsight Compute refuses par le pilote, y compris pour le kernel
  Black-Scholes lance explicitement hors sandbox. Leur absence limite les
  quatre sous-audits Performance et ne devient pas un constat generique.

## Decisions et hypotheses de performance

- **Decision mesuree :** aucune regression stable superieure au seuil v1 de
  5 % sur les 18 workloads appariees. Les quatre CV excessifs sont
  inconclusifs et ne justifient aucune correction ni mise a jour de baseline.
- **A mesurer :** une autre geometrie ou decomposition LSM pour G2 (118
  registres, 33,3 % d'occupation theorique) et Heston (88, 33,3 %). Les traces
  montrent que regression partials/solve dominent, mais aucun gain n'est
  presume sans matrice paths/dates/blocs et compteurs.
- **A mesurer :** le compromis facteur/temps/erreur des lifts rough au-dela du
  cas N7 observe. Le delta Heston 85,049 ms contre rough Heston N7 88,163 ms
  sur un complement local ne constitue ni baseline versionnee ni preuve
  d'erreur d'approximation.
- **A mesurer :** les quatre workloads bruités de E14 dans un environnement
  thermique/frequences mieux stabilise. Le bruit ne masque ni ne prouve une
  regression.

## Comptabilite des constats du passage version 3

- **Ouverts : 7** — `STRUCT-010`, `STRUCT-011`, `STRUCT-012`, `STRUCT-013`,
  `NUM-006`, `CUDA-001`, `PERF-010`.
- **Rouverts pendant ce passage : 2** — `STRUCT-010`, `PERF-010`, chacun selon
  le critere conserve dans son ancienne entree fermee.
- **Identifiants historiques encore fermes : 35** dans `closed.md`.
- **Correction de code pendant l'audit : aucune.**
