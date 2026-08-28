# Constats d'audit non resolus

## Objet

Ce document contient exclusivement les constats non resolus du referentiel
principal version 4. Le dernier passage du 2026-08-28 est partiel et cible la
structure/naming de `src/model`, la navigabilite du codegen, les frontieres des
policies Gaussian-Volterra et les remediations presentes dans le worktree. La
provenance, la couverture, les exclusions et les preuves sont consignees dans
`status.md`.

Trois constats restent ouverts au 2026-08-28. Quarante-neuf identifiants
historiques sont fermes dans `closed.md`.

## Code generation and extension locality

### STRUCT-015 — Generer les bindings et recettes fixed-income closed form

- **Etat :** ouvert; explicitement reporte.
- **Severite :** haute.
- **Priorite :** haute.
- **Confiance :** prouvee.
- **Fichiers et symboles :** les 42 fichiers closed form sous
  `src/model/fixed_income/<model>/product/[<curve>/]`; les recettes
  correspondantes sous `catalog/model/fixed_income/**/prices/`;
  `tools/codegen/pricing_bindings/{capability_manifest.py,generate.py,templates/}`.
- **Signature originale :** la source de verite typee inventorie les capacites
  fixed income et leurs unites CMake, mais codegen ne possede aucun template
  `.cuh/.cu` fixed-income closed form. Les launchers standalone affines et les
  compositions ajustees a Nelson--Siegel ou Svensson restent recopies a la
  main. Les recettes fixed income restent elles aussi une exception explicite
  du manifeste (`CAP-FI-RECIPE-BODY`) au lieu d'etre rendues depuis des
  templates canoniques.
- **Risque :** ajouter un modele, une courbe ou un produit fixed income exige
  des editions transversales difficiles a inventorier; des signatures, includes,
  sides, geometries ou metadonnees peuvent diverger silencieusement entre des
  compositions mathematiquement equivalentes. La pyramide de factorisation
  closed form n'est donc pas traduite dans l'architecture du generateur.
- **Correction requise :** ajouter un manifeste de bindings fixed income et
  une branche explicite `templates/pricing/fixed_income/closed_form/`, separee
  au minimum entre affine standalone un facteur, affine standalone deux
  facteurs et modele ajuste a une courbe. Generer egalement leurs recettes
  catalogue depuis une branche nommee symetriquement. Les Bermudans
  Longstaff--Schwartz appartiennent a une branche `early_exercise` distincte et
  sont hors de ce constat.
- **Test de cloture :** une generation dans un dossier temporaire reproduit
  bit a bit tous les `.cuh/.cu` et toutes les recettes fixed-income closed form
  suivies; aucun nouveau couple compatible ne demande de copier un launcher ou
  une recette; le checker de frontiere refuse un artefact manuel equivalent et
  les builds/tests fixed income cibles passent.
- **References de preuve :** E20, E26; inventaire structurel du 2026-08-28.
- **Derniere verification :** 2026-08-28.
- **Proprietaire :** non attribue; remediation reportee a une passe ulterieure.

## Numerical robustness and reproducibility

### NUM-007 — Stabiliser les samples Quadratic rough-Heston sur le domaine core publie

- **Etat :** ouvert; remediation implementee, qualification complete requise.
- **Severite :** haute.
- **Priorite :** haute.
- **Confiance :** prouvee pour l'echec initial et sa correction ciblee;
  incomplete pour tout le domaine de production.
- **Fichiers et symboles :**
  `src/model/equity/rough/quadratic_rough_heston/{markovian_n_factor_preparation.hpp,dynamics.cuh,dynamics_impl.cuh}`;
  `validation/volterra/quadratic_rough_heston.py`; tests Python et CUDA QRH;
  recette `catalog/model/equity/quadratic_rough_heston/samples/samples_02/`.
- **Signature originale :** les 48 generateurs compilaient sur SM89 avec
  mathDx, mais `quadratic_rough_heston/samples_02` echouait
  deterministement avec un spot non fini ligne 77, sur les bornes core
  publiees et avec les graines contractuelles.
- **Remediation presente :** la recurrence N-facteurs equilibre sans clamp ni
  branche la contribution de cellule Volterra complete,
  `F / hypot(1, c F)`. Le contrat dynamics et les references independantes
  dense/exponentielle decrivent le meme schema; le mapping Philox reste
  `(parameter row key, path_index, local_group_index)`.
- **Preuves acquises :** regression CUDA de la ligne 77 finie,
  reproductible bit a bit et invariante entre deux geometries; smoke test
  complet des 1 000 lignes publiees de `samples_02` passe; la ligne core 680,
  qui explosait encore avec le schema explicite FP64, reste finie et l'erreur
  terminale diminue entre 126, 252 et 504 pas face a la reference fine. Les
  tests Python QRH passent. Les 47 autres smoke tests avaient deja passe avant
  cette modification locale au modele QRH.
- **Risque restant :** ces preuves ne constituent pas encore le preflight des
  trois millions de lignes, ni un balayage core/stress multi-trajectoires et
  multi-facteurs. La dynamique etant partagee par les pricers, la validation
  de prix independante doit aussi etre rejouee avant publication.
- **Correction restante :** executer le preflight production, le balayage
  core/stress, le raffinement factoriel et les validations Premia/QuantLib
  pertinentes; conserver les sorties et ressources du kernel exact teste.
- **Test de cloture :** les 48 smoke tests passent avec les graines publiees;
  le preflight production ne trouve aucune valeur non finie; les raffinements
  temporel et factoriel et la reference CPU/haute precision bornent le
  comportement sur core et stress; les validations independantes affectees
  passent.
- **References de preuve :** E20, E22, E25.
- **Derniere verification :** 2026-08-28.
- **Proprietaire :** non attribue.

## Performance

### PERF-010 — Rendre le controle de baseline exhaustif et bloquant

- **Etat :** rouvert; implementation complete, campagne bloquante finale
  differee.
- **Severite :** moyenne.
- **Priorite :** moyenne.
- **Confiance :** prouvee.
- **Fichiers et symboles :**
  `validation/performance/{model_sample_benchmark.cu,benchmark_support.cuh,check_baseline.py,run_baseline.py,baseline_sm89_v1.json}`
  et leurs tests; `CMakeLists.txt`, cible `performance_benchmarks`.
- **Signature originale :** le checker v3 etait exhaustif sur ses 22 cles,
  mais son manifeste omettait entierement les model samples requis par la
  query v4 : deux layouts, publication wall et familles
  Markovian/N-facteurs/Volterra.
- **Remediation presente :** le manifeste contient maintenant 30 cles, dont
  huit workloads samples couvrant transition exacte, pas fixe, lift rough a
  sept facteurs et Volterra FFT sur `3 000 000 x 1` et `12 000 x 250`. Les
  shapes Markovian/N-facteurs sont completes; les reductions Volterra
  documentent leur saturation. Kernel et publication JSON/YAML sont gates
  separement. Le checker conserve 5 % de CV maximum pour CUDA et 10 % pour la
  publication host batchee, refuse candidat manquant, duplique, inconnu ou
  d'environnement incompatible, et selectionne seulement une campagne stable.
- **Preuves acquises :** 11 tests unitaires du checker passent; les huit
  workloads samples ont ete compiles et executes isolement sur le SM89 de
  reference avec sorties finies, kernels sous le seuil de bruit et provenance
  `sm89_reference_v1`. L'agregat `performance_benchmarks` compile en entier.
- **Risque restant :** la campagne complete de 30 cles a ete interrompue avant
  production d'un candidat apparie. Aucun succes du gate complet n'est donc
  revendique et la baseline ne doit pas etre presentee comme recertifiee.
- **Correction restante :** sur machine alimentee et thermiquement stable,
  executer les trois campagnes de `performance_regression_gate`, conserver le
  candidat NDJSON et resoudre toute cle en echec ou encore inconclusive.
- **Test de cloture :** le gate complet contient exactement une fois chaque
  workload generique, sample, LSM et rough requis; aucune cle bloquante n'est
  manquante ou inconclusive; toute regression superieure au seuil echoue et
  toute comparaison GPU/toolchain incompatible est refusee.
- **Historique de cloture :** ferme le 2026-08-27 apres creation du protocole
  v1, puis rouvert quand l'omission de references produisait tout de meme un
  succes. L'exhaustivite structurelle est maintenant corrigee; seule sa
  campagne de certification reste requise.
- **References de preuve :** E08, E14, E15, E17, E20, E22, E25.
- **Derniere verification :** 2026-08-28.
- **Proprietaire :** non attribue.
