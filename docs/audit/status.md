# État des audits principaux

## Objet et verdict du passage version 7

Ce document consigne le snapshot, les inventaires exhaustifs, les matrices de
couverture, les exclusions et les preuves du passage indépendant demandé le
2026-08-30. Le code, les tests, CMake, les générateurs, manifests, baselines et
contrats d'implémentation n'ont pas été corrigés. Les constats ouverts sont
exclusivement dans `docs/audit/response.md`; les entrées encore fermées sont
dans `docs/audit/closed.md`.

Verdict global : **non conforme et partiel**. Onze constats sont ouverts, dont
sept nouveaux (`STRUCT-019`, `CUDA-003`, `POLICY-003`, `BUILD-005`,
`STRUCT-020`, `NAME-013`, `STRUCT-021`) et quatre réouvertures (`STRUCT-011`,
`NAME-006`, `STRUCT-010`, `NAME-007`). Les inventaires structurels, codegen,
FP64, RNG et launches CUDA sont exhaustifs statiquement. Les axes CMake,
portabilité, robustesse numérique, sûreté CUDA et Performance restent partiels
selon leurs conditions strictes de complétude version 7.

## Snapshot et limites de provenance

- **Date / timezone :** 2026-08-30, Europe/Paris.
- **Référentiel :** `docs/audit/query.md`, version 7, SHA-256
  `44446a1c1871e7b6591a21d56e5a0f300913a16f6938e5be921a8e95e2dd2b06`.
- **Branche :** `refactor/model-layout-codegen-volterra`, amont à `+0/-0`.
- **HEAD :** `5d65c23cf709173a65ab809baff0da2566b58acf`.
- **Worktree source avant rédaction de l'audit :** 1 303 chemins suivis
  différents de HEAD, soit 372 modifiés et 931 supprimés; 989 chemins non
  suivis; index vide; diff suivi `+7 771/-68 497` lignes.
- **SHA-256 porcelain v2 NUL avec branche :**
  `42bfc69b994512e6da49fe2f219847e6314e616a33c62e0c6d5f89767667d410`.
- **SHA-256 diff binaire suivi depuis HEAD :**
  `678a548fbdcb313b0ef7f5f9bf93f0e58389adf268ec27abde95c72c2ee8b128`.
- **SHA-256 index :**
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- **SHA-256 manifeste trié des chemins non suivis :**
  `f8d3770cfce3770d9b78a47600985792c4b975c0663edb6e3534b274b86ee256`.
- **SHA-256 agrégat initial du contenu non suivi :**
  `4419fd0978b6b0088be4f352d6ba3321bdcb05be287f6e20ab5c6f25e2422c42`.
- **SHA-256 agrégat du contenu non suivi après exécution et avant rédaction :**
  `3cac9bb5118120e9c36eb2c2ec67b38c683eb577cc0b0c8deef06b100321c12e`;
  le manifeste de chemins et son cardinal sont inchangés.
- **Mutation produite par ce passage :** seulement les trois registres
  `docs/audit/status.md`, `docs/audit/response.md` et
  `docs/audit/closed.md` sont édités intentionnellement. Les scripts Python et
  CTest ont aussi rafraîchi des caches `.pyc` non suivis préexistants sous
  `tools/**/__pycache__` et `validation/performance/__pycache__`; aucun source,
  manifest, baseline ou dataset n'a été modifié. Les artefacts de build et logs
  nouveaux sont sous `build-dev` déjà ignoré ou `/tmp`.

Le passage a observé le worktree réel, fichiers non suivis inclus. Les hashes
ci-dessus précèdent les trois sorties d'audit afin d'éviter l'auto-référence du
rapport et permettent de détecter une différence avec cet état. Ils ne
permettent toutefois pas de le reconstruire : ni le porcelain complet, ni le
manifeste des 989 chemins non suivis, ni le diff suivi correspondant n'ont été
conservés comme artefacts durables. La provenance du passage est donc
identifiée mais partielle, et HEAD seul n'est pas présenté comme sa source.

## Gates P0

| Gate | État | Preuve / limite |
|---|---|---|
| `P0-SNAPSHOT` | partial | Branche, HEAD, cardinalités, hashes, toolchain, build et GPU sont identifiés, mais porcelain, manifeste non suivi et diff ne sont pas conservés : le snapshot n'est pas reconstructible. |
| `P0-INVENTORY` | pass | Inventaires physiques et canoniques exhaustifs établis avant échantillonnage. |
| `P0-SCOPE` | partial | Le gate Performance principal n'a pas de propriétaire hors `validation/**`; `STRUCT-019`. |
| `P0-COVERAGE` | pass | Toute case non exécutée ou matériellement indisponible reste `partial`/`excluded`, jamais succès implicite. |
| `P0-IDENTITY` | pass | Les trois registres sous `docs/audit` ont été recherchés par cause et périmètre avant attribution des IDs. |
| `P0-NUMERICS` | partial | Matrice et FP64 classés, mais toutes les références core/stress/limites n'ont pas été rejouées sur le snapshot. |
| `P0-PERF-STATS` | partial | Aucune campagne principale v7 recevable; l'artefact v2 sous validation n'est que complémentaire. |
| `P0-REBASELINE` | excluded | Aucun rebaseline exécuté; la baseline existante n'a pas été modifiée. |
| `P0-EVIDENCE` | partial | Commandes/résultats et hash des logs sanitizer sont conservés ici, mais les logs bruts sous `/tmp` ne sont pas versionnés. |

## Inventaire canonique

### Arbre physique

| Racine | Fichiers | Dossiers | État d'inventaire |
|---|---:|---:|---|
| `src` | 1 257 | 105 | exhaustif |
| `tools` | 148 | 47 | exhaustif |
| `catalog` | 1 081 | 1 429 | exhaustif |
| `datasets` | 382 | 433 | exhaustif |
| `tests` | 73 | 2 | exhaustif |
| `docs` | 22 | 4 | exhaustif hors registres réécrits |
| `cmake` | 3 | 2 | exhaustif |

- 1 278 unités autonomes `.cpp/.cu` dans `src`, `tools`, `tests` et
  `catalog`; 783 headers sous `src`/`tools`; 74 templates codegen.
- Empreinte triée des chemins du périmètre principal :
  `17e650b03563d0bb432f3b4a61c27740727ccefbfd74a4704bd76bc93ce4e363`.
- Le checker classe 871 fichiers générés et 578 handwritten, plus quatre
  manifests et trois préambules de format.

### Matrice canonique et publications

- 12 engines, 24 modèles, 26 produits, 2 courbes.
- 416 `ProductBindingSpec`, dont 399 sorties générées et 17 handwritten.
- 697 `DatasetSpec`, tous disponibles : 628 recettes générées et 69
  handwritten; zéro sample différé.
- 588 domaines RNG; 697 générateurs de catalogue; 382 YAML et 382 JSON locaux
  matérialisés; 832 fichiers modèle-produit; 1 500 sorties codegen.

| Famille | Modèles | Transition / préparation | Produits / exercice | Sampling |
|---|---:|---|---|---|
| Equity Markov exact | 5 | exact, compatible fixed-step | 21 ordinaires; LSM optionnel | disponible |
| Equity Markov fixed | 7 | fixed-step | 21 ordinaires; 4 familles LSM | disponible |
| Rough Gaussian-Volterra | 4 | hybrid FFT, mathDx | 21 ordinaires; exercice non supporté | conditionnel mathDx |
| Rough N-facteurs | 2 | lift préparé 2/3/7 facteurs | 21 ordinaires; exercice non supporté | disponible |
| Fixed income standalone | 4 | état exact; joint CIR fixed, autres exacts | closed form et Bermudan selon modèle | disponible |
| Fixed income fitted | 2 × 2 courbes | adaptation exacte état/intégrale | closed form et Bermudan | disponible |

Produits equity ordinaires : 5 terminal, 10 dense, 5 regular et 1 calendrier à
deux dates; 19 observent `spot`, 2 `log_spot`; 12 sont call/put compile-time et
9 non sided. Décompte des compositions : 244 Markov, 84 Volterra, 42
N-facteurs, 21 FI closed form, 8 Black-Scholes closed form, 8 FI LSM, 5 equity
LSM exact et 4 equity LSM fixed.

## Couverture par axe

| Axe | État | Couverture obtenue | Cause de non-complétude |
|---|---|---|---|
| Structure, ownership, naming | partial | Arbre, 783 headers, dépendances et docs inspectés exhaustivement; exercices de navigation ciblés | Huit constats touchent structure/nommage; cinq exercices contrôlés sans recherche globale non formalisés. |
| Contract homogeneity | partial | 24 modèles, 26 produits, transitions, analytics, schedules et 416 bindings croisés | Concept produit et contrat CIR/manifeste non homogènes. |
| Factorization pyramid | partial | Markov, rough, N-facteurs, closed form, FI et LSM relus; 21 alias `same_as` confirmés | Coûts complets de chaque frontière non remesurés sous protocole v7. |
| Code generation / extension | partial | Zero-diff 1 500 sorties, 14 tests de manifeste, inventaire bidirectionnel 697/832 | Source de vérité fragmentée; aucun dry-run réel ajout/retrait complet. |
| CMake / build graph | partial | Unités/compile DB, configs avec/sans mathDx, builds clean ciblés et no-op | Root monolithique; matrice de mutations et revue exhaustive `PUBLIC/PRIVATE` absentes. |
| Portability / tuning | partial | Compilation SM75 mono et fat 75/86/89 représentative; runtime SM89 | Aucun GPU physique SM75/SM86, ni qualification/runtime/profil séparé hors SM89. |
| Numerical robustness / reproducibility | partial | FP64 device exhaustif, RNG, solveurs et contrats relus; tests numériques disponibles exécutés dans la suite GPU | Toutes les références et convergences core/stress/limites n'ont pas été isolément rejouées; `CUDA-003`. |
| CUDA execution / memory safety | partial | 23 sites de launch relus; ASan/UBSan host; 32 runs Compute Sanitizer; 77 CTests | La matrice obligatoire de huit scénarios n'est pas entièrement mappée aux layouts/streams/concurrence prescrits. |
| Performance CUDA générique | partial | Binaires construits; baseline de validation lue comme consommateur | Aucun propriétaire principal ni campagne v7 recevable; pas de profil Nsight. |
| Performance model-sample | partial | Harness complémentaire et shapes déclarées inspectés | Même défaut d'ownership; campagne v7 non exécutée. |
| Performance early exercise | partial | Binaire complémentaire et tests LSM exécutés | Même défaut d'ownership; campagne/ressources v7 non exécutées. |
| Performance rough | partial | Binaires Volterra/N-facteurs construits et testés | Même défaut d'ownership; crossover/profil v7 non exécuté. |

## Factorisation et frontières positives à préserver

| Frontière | Producteur / consommateurs | Conclusion |
|---|---|---|
| Schedules/dynamics | `common/simulation` vers 12 equity Markov, 6 FI et 2 N-facteurs | Réutilisation sans branche runtime; conserver la séparation exact/fixed. |
| Path products | 21 policies produit vers Markov, N-facteurs et Volterra | Les 16 anciens corps directs sont absents et les 21 surfaces sont des alias; `FACTOR-001` reste fermé. |
| Volterra | 3 kernel policies et 4 path policies vers pricing/sampling FFT | `PreparedKernel` reste opaque; `POLICY-002` reste fermé. |
| Analytics FI | providers standalone/fitted vers quatre produits FI | Zéro include produit dans 18 headers analytics; `ANALYTICS-001` reste fermé. |
| LSM | moteur, regression et workspace communs vers American/Bermudan | Payoff, continuation et normalisation restent au produit; aucune copie de moteur prouvée. |
| Closed form | kernel scalaire/cooperatif vers 8 BS et 21 FI | Sides compile-time et providers spécifiques restent factorisés. |

Les dépendances interdites sont absentes : `src -> tools/catalog/validation`,
`common -> model/curve/product`, `curve -> model/product`, `product ->
model/curve` et includes textuels de `.cu` valent tous zéro. L'homonyme
`src/common/sample.cuh` + `src/common/sample/` est une façade documentée avec
28 consommateurs, pas un constat.

## Codegen et coût d'extension

| Preuve | Résultat |
|---|---|
| `tools/codegen/pricing_bindings/generate.py --family all --compare-root .` | 1 500 sorties, zéro diff |
| `test_capability_manifest` | 14/14 tests passés, fixtures négatives incluses |
| `tools/cuda/check_model_layout.py` | 832 modèle-produit, 199 infrastructure, 74 templates, 871 générés/578 handwritten |
| `tools/cuda/check_catalog_generators.py` | 697 recettes, 628 générées, zéro différée, aucune escape hatch CUDA brute |

Le zero-diff et l'exhaustivité des outputs sont valides, mais ne compensent pas
`STRUCT-011`: l'identité modèle et les sémantiques pricing/sample/variant sont
encore réparties entre plusieurs specs. `STRUCT-003`, `STRUCT-015` et
`STRUCT-018` restent fermés sur leurs signatures propres.

## CMake, configurations et incrémentalité

### Matrice de configuration exécutée

| Configuration | Configure | Build | Run | Conclusion |
|---|---|---|---|---|
| Release SM89, CUDA 13.3, mathDx en cache, tests ON | pass | host + CUDA + performance pass | 77/77 hors validation | preuve principale locale |
| Release SM75, sans mathDx | pass | `ai_factory_host_tests`, 128 étapes | 12 CTests host/architecture/codegen pass | compilation portable ciblée |
| Release SM75, avec mathDx | pass | rough Bergomi représentatif pass | test host QRH pass | cuFFTDx compile SM75 |
| Release fat 75/86/89, avec mathDx | pass | rough Bergomi représentatif pass | non exécuté | compilation offline uniquement |
| Debug | non exécuté | unknown | unknown | exclusion de couverture |
| Runtime SM75 / SM86 | non exécuté | n/a | matériel absent | exclusion matérielle |

La compile DB mathDx contient 1 287 entrées / 1 283 sources uniques; le graphe
sans mathDx 1 064 entrées. Les 1 278 unités physiques du périmètre sont
enregistrées; les doublons restants sont des expériences explicitement
nommées. La racine CMake de 1 802 lignes motive `BUILD-005`.

### Matrice d'incrémentalité

| Mutation | Attendu | Mesuré |
|---|---|---|
| no-op host + CUDA + performance | aucune opération | pass, `ninja: no work to do`, 0,07 s |
| produit | binding/consommateurs ciblés | non exécuté |
| dynamics | bindings du modèle | non exécuté |
| analytics | spécialisation concernée | non exécuté |
| courbe | fitted consumers | non exécuté |
| primitive common | consommateurs réels | non exécuté |
| template | sorties possédées puis consommateurs | non exécuté |
| manifeste | régénération matrice exacte | non exécuté |

Aucun fichier source n'a été touché pour simuler ces mutations; cette matrice
manquante maintient l'axe partiel.

## Numérique, RNG et reproductibilité

### Inventaire FP64 device

L'inventaire lexical touche 39 fichiers, dont 18 faux positifs
`double-knock-out`; les 21 fichiers sémantiques sont tous classés.

| Classe | Propriétaires | État |
|---|---|---|
| Moments MC / Volterra | `monte_carlo_kernel`, `reductions`, `hybrid_fft_pricer` | qualifiés historiquement par `NUM-005`, `NUM-010`, `NUM-020` |
| Formation/réduction LSM | `longstaff_schwartz_kernels`, `small_linear_regressor` | `NUM-002`, `NUM-016` |
| Solveur LSM | `linear_solver`, Cholesky/substitutions | `NUM-003`, `NUM-017` |
| Prédiction/décision LSM | `exercise_decision`, policies American/Bermudan | `NUM-003`, `NUM-018` |
| Calculs FP64 host | fits rough, préparation, courbes, diagnostics/timings | pas d'opération device additionnelle; courbes `NUM-019` |

Aucun `long double`, `__float128`, `double2` applicatif ou atomic FP64 device
supplémentaire n'est atteint. Les clôtures FP64 restent donc supportées par
l'inventaire, sans prétendre que toute leur campagne numérique a été rejouée.

### Philox

Les 588 domaines déclarent un stride `2^32` et une capacité par flux `2^30`.
Le calcul indépendant des clés transformées trouve 96 clés fixed-source, 588
clés dynamics et zéro collision. Uniformes ouverts, rejet entier sans biais et
garde de seed de ligne ont été relus; `NUM-008` reste fermé.

Le débordement temporel `CUDA-003` est la seule nouvelle défaillance numérique
prouvée. L'absence d'une requalification isolée de chaque matrice historique
est une exclusion, pas un finding générique.

## CUDA, sanitizers et GPU

### Environnement

- NVIDIA GeForce RTX 4090 Laptop GPU, UUID
  `GPU-214a7d0d-e7b0-ded2-983c-0c960ec5bcd4`, compute capability 8.9,
  driver 596.08, 16 376 MiB.
- Toolchain du build : CMake 3.22.1, Ninja 1.10.1, GCC 14.3, NVCC
  13.3.73; Compute Sanitizer 2026.2.1.0.
- Preflight : 63 °C, P0, SM 2 040 MHz, mémoire 9 001 MHz, 42,24 W, aucun
  processus compute. Postflight : 67 °C, P0, mêmes clocks, 45,03 W, 0 MiB
  utilisé et aucun processus compute résiduel.

### Tests fonctionnels

- Build `ai_factory_host_tests`, `ai_factory_cuda_tests` et
  `performance_benchmarks` : 142 étapes recompilées, succès.
- CTest host/architecture/codegen hors CUDA/validation : 13/13.
- CTest principal hors `validation` : 77/77 en 87,56 s, dont 64 labels CUDA;
  equity, fixed income, LSM, Volterra, N-facteurs, samples, limites workspace,
  diagnostics kernel et budgets de policy passent.
- Une tentative complémentaire du CTest total a fait passer les 71 premiers
  tests QuantLib puis rencontré l'interdiction sandbox de binder un socket Wine
  dans les cas Premia. Elle a été arrêtée vers le test 136; `validation/**`
  étant hors périmètre, cette limite n'est ni un échec produit ni un finding.

### ASan/UBSan host

Configuration fraîche `/tmp/ai-factory-audit-v7-host-sanitize.2Uo98W`, GCC 14,
`-fsanitize=address,undefined -fno-omit-frame-pointer`; build host 128/128.
Les huit tests loaders/planners/offline passent 8/8 avec ASan/UBSan. LSan seul
est exclu : il termine fatalement sous le `ptrace` imposé par l'environnement;
le rerun documenté emploie `ASAN_OPTIONS=detect_leaks=0` sans désactiver ASan
ni UBSan.

### Compute Sanitizer

Les quatre modes `memcheck`, `racecheck`, `synccheck`, `initcheck` passent sur
chacun des huit exécutables suivants, soit 32/32 runs et zéro erreur :

| Exécutable | Famille principale |
|---|---|
| `test_black_scholes_cuda` | closed form |
| `test_heston_path_products_cuda` | Markov fixed/path products |
| `test_bermudan_swaption_cuda` | fixed income LSM |
| `test_heston_american_option_cuda` | equity LSM multi-state |
| `test_rough_volterra_product_policy_cuda` | Volterra FFT pricing/workspace |
| `test_rough_volterra_samples_cuda` | Volterra sampling, deux layouts |
| `test_quadratic_rough_heston_samples_cuda` | N-facteurs 7, sampling |
| `test_volterra_fft_workspace_bounds_cuda` | bornes/dernier lot/workspace |

Logs : `/tmp/ai-factory-audit-v7-sanitizers.wXD0dR`, agrégat SHA-256
`b29b8e8543853bdb243ce298f29660941c8b0248377ca9f3063eb94844f6a00c`.

La revue statique couvre 23 sites de launch : 3 closed form, 1 Monte Carlo,
4 sampling générique, 1 sampling Volterra, 7 LSM et 7 pricing Volterra; chacun
est suivi d'un `cudaGetLastError` à sa phase attribuable. Aucun nouvel OOB,
champ non initialisé ou défaut de workspace précis n'est prouvé après
`CUDA-001`/`CUDA-002`. La surface actuelle n'expose pas de stream non-default;
le scénario est non applicable, mais la couverture des combinaisons exactes
regular/aligned, ragged/cartesian, réutilisation et concurrence n'est pas
entièrement démontrée : l'axe reste partiel.

## Performance

La baseline complémentaire sous `validation/performance` déclare 22 commandes,
41 mesures et trois campagnes sur SM89; SHA-256
`2e93b4dc7d44db7174b7e168d2de1194c2bfe4ef636af9bfc5e01d501ebd0697`.
Elle n'a pas été modifiée ni rebaselinée. Conformément à la version 7, son
placement exclusif sous validation interdit de la traiter comme preuve
principale; son implémentation interne ne produit pas de finding dans ce
référentiel.

Sa lecture comme outil complémentaire montre en outre que
`best_stable_manifest` sélectionne encore, clé par clé, la plus petite médiane
kernel parmi les essais stables. Ce best-of-N n'est pas enregistré comme un
constat interne à `validation/**`, mais interdit de clore `STRUCT-019` par un
simple déplacement des fichiers : le futur propriétaire principal devra
adopter une agrégation complète non opportuniste conforme à la version 7.

Aucune campagne principale v7, aucun profil Nsight et aucune décision
`accept/reject/inconclusive/unavailable` par workload n'ont donc été produits.
Les quatre sous-audits Performance restent partiels pour `STRUCT-019`, et non
par assimilation silencieuse de la baseline historique à une preuve actuelle.

## Exclusions et limites explicites

- Pas de correction, refactor, rebaseline, régénération de dataset, commit ou
  push.
- `validation/**` et `docs/validation/**` lus/exécutés seulement comme
  consommateurs ou preuves complémentaires; l'échec Wine est exclu.
- Pas de GPU physique SM75/SM86, donc aucune conclusion runtime, numérique ou
  performance sur ces architectures.
- Pas de build Debug complet, build clean de toutes les 1 278 unités dans une
  même configuration, ni matrice réelle de mutations CMake.
- Pas de dry-run complet ajout/retrait modèle/produit/composition, de smoke des
  697 recettes ou de régénération des 382 datasets matérialisés.
- Pas de campagne numérique isolée complète pour chaque core/stress/limite et
  pas de profilage Nsight.
- Les logs sanitizer sous `/tmp` ne sont pas une preuve versionnée durable.

Ces exclusions maintiennent les axes concernés partiels; elles ne créent pas
de constat sans défaut précis.

## Commandes principales et résultats

```text
python3 tools/codegen/pricing_bindings/generate.py --family all --compare-root .
# 1 500 sorties, zéro diff

python3 -m unittest -v tools.codegen.pricing_bindings.test_capability_manifest
# 14/14

python3 tools/cuda/check_model_layout.py
# 832 model-product, 199 infrastructure, 74 templates,
# 871 generated / 578 handwritten

python3 tools/cuda/check_catalog_generators.py
# 697 recettes, 628 générées, zéro différée

cmake --preset dev
cmake --build build-dev --target \
  ai_factory_host_tests ai_factory_cuda_tests performance_benchmarks -j2
ctest --test-dir build-dev --output-on-failure -LE validation
# 77/77

env ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
  UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir /tmp/ai-factory-audit-v7-host-sanitize.2Uo98W \
  --output-on-failure -I 1,8
# 8/8

compute-sanitizer --tool {memcheck,racecheck,synccheck,initcheck} \
  --error-exitcode=99 ./build-dev/<8 executables>
# 32/32, zéro erreur

env CCACHE_DISABLE=1 /usr/bin/time cmake --build build-dev --target \
  ai_factory_host_tests ai_factory_cuda_tests performance_benchmarks -j2
# no-op, 0,07 s
```

## Remediation partielle du 2026-08-30 — `CUDA-003`

- **Revision :** `5d65c23cf709173a65ab809baff0da2566b58acf`, branche
  `refactor/model-layout-codegen-volterra`.
- **Worktree :** sale avant et apres la remediation; 1 908 chemins suivis
  modifies/supprimes et 990 chemins non suivis au releve final. Les changements
  sans rapport presents dans ce worktree partage n'ont pas ete nettoyes.
- **Couverture :** arithmetique host checked pour tous les calendriers
  fixed-step, schedules d'exercice, launchers pricing ordinaires/N-factor/LSM,
  formule geometric Asian et sources de calendriers sample; generation de la
  matrice complete et build de l'agregat CUDA.
- **Exclusions :** aucune mesure de performance ni changement de geometrie;
  l'arithmetique device et le mapping Philox restent inchanges. Le timeout
  CTest de 30 secondes du test QRH de preparation est traite avec `BUILD-005`.
- **Preuve :** `simulation_schedule_validation` passe; codegen 1 500 sorties
  zero-diff; agregat `ai_factory_cuda_tests` compile; 63/64 CTests CUDA passent
  et le seul timeout passe directement en environ 61 secondes; memcheck,
  racecheck, initcheck et synccheck passent sans erreur/hazard sur
  `path_product_factorization_cuda`.

## Remediation partielle du 2026-08-30 — `STRUCT-011`

- **Revision :** `5d65c23cf709173a65ab809baff0da2566b58acf`, branche
  `refactor/model-layout-codegen-volterra`.
- **Worktree :** sale avant et apres; les modifications paralleles de
  catalogue, validation et codegen ont ete preservees.
- **Couverture :** source canonique des 24 modeles, contrats minimaux des 12
  engines, 26 produits et 697 datasets, resolution des 416 compositions et cas
  FI LSM CIR.
- **Exclusions :** pas de smoke runtime des 697 recettes et pas de mesure GPU;
  ce passage porte sur la coherence statique et la regeneration exhaustive.
- **Preuve :** 19/19 tests du manifeste; ajout/retrait modele, produit et
  composition; fixture de table divergente; codegen 1 500 sorties zero-diff;
  provenance regeneree en schema 2.

## Remediation partielle du 2026-08-30 — `BUILD-005`

- **Revision :** `5d65c23cf709173a65ab809baff0da2566b58acf`, branche
  `refactor/model-layout-codegen-volterra`.
- **Worktree :** sale avant et apres; les changements paralleles de catalogue,
  validation, codegen et datasets ont ete preserves.
- **Couverture :** extraction des propriétaires runtime, catalogue,
  performance, tests et validation; configurations native SM89 avec mathDx,
  SM75 sans mathDx et fatbin SM75/86/89; inventaire, builds et no-op.
- **Exclusions :** la matrice de mutation n'est pas prétendue exhaustive : le
  dry-run par timestamp n'était pas discriminant. Une régénération codegen
  réelle et ses 229 consommateurs recompilés fournissent la preuve de
  propagation disponible.
- **Preuve :** racine 1 833 vers 127 lignes; 573 targets strictement identiques;
  builds host/CUDA/performance réussis; 12/12 tests SM75; cinq CTests principaux
  réussis, dont la préparation QRH en 36,54 s avec timeout explicite de 120 s;
  second build sans travail en 0,12 s.

## Remediation partielle du 2026-08-31 — `STRUCT-019`

- **Revision de depart :** `5d65c23cf709173a65ab809baff0da2566b58acf`,
  branche `refactor/model-layout-codegen-volterra`.
- **Worktree :** partage et deja massivement sale avant la remediation; les
  changements de structure, catalogue, codegen, numerique et validation des
  passages precedents sont consolides dans la meme branche sans nettoyage
  destructif.
- **Couverture realisee :** les sources, fixtures et baselines principales de
  performance appartiennent maintenant a `tests/performance`; runners et
  checker appartiennent a `tools/performance`; le protocole durable est dans
  `docs/performance-regression-protocol.md` et le graphe de targets dans
  `cmake/AIFactoryPerformance.cmake`. Le protocole version 3 impose 41 mesures,
  quatre rapports, quatre frontieres de temps, trois campagnes eligibles sans
  best-of-N, ressources compilees par symbole, budgets numeriques/VRAM/binaires,
  provenance et preflight fail-closed.
- **Durcissement environnemental :** alimentation secteur, processus compute,
  temperature, derive thermique, raisons de throttling et enveloppe de
  puissance NVIDIA sont verifies. Une limite courante inferieure a 140 W est
  refusee pour le profil SM89; le journal de campagnes est persiste
  incrementiellement et peut etre repris seulement apres verification de ses
  SHA-256 et nombres de lignes.
- **Preuve disponible :** 17/17 tests unitaires du protocole passent; le build
  des benchmarks passe; deux campagnes SM89 completes sont eligibles et
  conservees parmi quatre tentatives. Une campagne de chauffe a ete rejetee
  pour derive thermique et un preflight a ete refuse apres une breve bascule
  batterie/secteur.
- **Blocage volontaire :** le pilote expose actuellement une limite de 55 W
  contre 150 W par defaut, avec `software_thermal_slowdown` actif. Aucune mesure
  sous ce profil degrade n'est acceptee et la cinquieme tentative n'est pas
  consommee.
- **Suite obligatoire :** retablir le profil 150 W, reprendre l'unique tentative
  restante, produire l'agregat et les quatre rapports, conserver le
  predecesseur avec un diff exhaustif, enregistrer les profils Nsight
  representatifs, rejouer le checker puis seulement deplacer `STRUCT-019` vers
  `closed.md`. Le constat reste donc explicitement ouvert dans cette PR.

## Remediation partielle du 2026-09-03 — `STRUCT-019`

- **Revision de depart :** `03a70a4585967da9a51208c22f5c80a03a46a43c`,
  branche `refactor/model-layout-codegen-volterra`; worktree sale avec les
  changements `STRUCT-019` non encore commites. Aucun fichier sous
  `docs/validation` n'entre dans ce passage.
- **Protocole mesure :** les trois campagnes sont toutes agregees par la
  mediane predeclaree de leurs medianes et de leurs coefficients de variation;
  le p95 reste le maximum conservateur. Deux campagnes sur trois doivent donc
  respecter 5 % de CV kernel et 10 % de CV pour l'enveloppe host, sans
  best-of-N ni recomposition par cle. La regression mediane/p95 reste bornee a
  5 %. Les fenetres sont normalisees par operation et les 27 tests fail-closed
  du protocole passent.
- **Campagne qualifiante SM89 :** trois campagnes sur trois sont eligibles sous
  `build-dev/performance_candidate_sm89_v3.ndjson.campaigns/20260903T184737.789921Z`.
  Les 41 mesures passent avec zero inconclusive bloquant; les deux messages
  restants appartiennent a l'unique microbenchmark closed-form informatif.
  Le candidat porte le SHA-256
  `0ca4c5f3ccfa68ad1415c08ded3afb14e6570c08326ed0805c11e3ec19d9a754`.
- **Rapports :** les partitions `generic_cuda`, `model_sampling`,
  `early_exercise` et `rough` contiennent respectivement 18, 8, 4 et 11
  mesures. Le checker rejoue la baseline contre le candidat avec `PASS: 41
  measurement(s), 0 blocking inconclusive, 2 informational`.
- **Rebaseline :** le predecesseur conserve a le SHA-256
  `26c4af09f90ffeb812d7ea906619981476d94f2c4e6b18dd07e5470c54381604`;
  la nouvelle baseline a le SHA-256
  `94b7370a2bf1ebed350a04a1037c42115d7a6946127399d7c4358f329786429f`.
  Le diff exhaustif d'initialisation, raison et approbation incluses, a le
  SHA-256
  `205cee367c9b0b152b37a02b1477252949fa9cf9753de41d429524cbd2323717`.
- **Blocage restant :** Nsight Compute 2026.2.1 refuse le premier profil avec
  `ERR_NVGPUCTRPERM`; le pilote interdit a l'utilisateur courant l'acces aux
  compteurs GPU et `sudo -n` exige un mot de passe. Aucun profil partiel n'a
  ete publie. `STRUCT-019` reste ouvert tant que les quatre profils
  representatifs ne sont pas captures apres activation de cette permission.
- **Suite obligatoire :** autoriser les compteurs de performance NVIDIA,
  capturer CIR, sample rough N-factor, LSM Heston et rough SABR FFT, verifier
  leurs hashes et scopes, rejouer les tests/checker finaux, puis deplacer le
  constat vers `closed.md`.

## Clôture du 2026-09-03 — `STRUCT-019`

- **Périmètre final :** passage principal Performance uniquement; aucune
  modification de `docs/validation` ni prétention sur l'audit de validation
  séparé. La révision de départ reste
  `03a70a4585967da9a51208c22f5c80a03a46a43c`; les changements de clôture sont
  présents dans le worktree avant commit.
- **Profils :** les quatre captures Nsight Compute 2026.2.1 sont présentes sous
  `tests/performance/profiles/sm89`, avec un CSV brut et un document de
  provenance chacun. Leurs scopes sont respectivement `generic_cuda`,
  `model_sampling`, `early_exercise` et `rough`; leurs hashes internes, ceux de
  la baseline et du candidat, les symboles compilés et les exécutables mesurés
  concordent. Une première capture refusée par `ERR_NVGPUCTRPERM` et une
  tentative rough SABR refusée par `software_thermal_slowdown` n'ont produit
  aucun artefact partiel; la capture finale a été faite après permission et
  refroidissement.
- **Vérifications finales :** 27/27 tests unitaires du protocole, checker
  `PASS: 41 measurement(s), 0 blocking inconclusive, 2 informational`, CTest
  `performance_baseline_checker`, build `performance_benchmarks` sans travail,
  validation des huit hashes de profils et `git diff --check` passent.
- **Décision :** tous les critères de clôture sont satisfaits. `STRUCT-019` est
  retiré de `docs/audit/response.md` et enregistré avec signature, preuve et
  condition de réouverture dans `docs/audit/closed.md`.
