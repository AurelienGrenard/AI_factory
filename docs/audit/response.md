# Constats d'audit non resolus

## Objet

Ce document contient exclusivement les constats non resolus produits par les
audits de `query.md`, y compris ceux dont le traitement est explicitement
reporte. La provenance, la couverture et les preuves sont consignees dans
`status.md`; les constats clos sont dans `closed.md`.

## Etat courant

Deux constats sont ouverts au 2026-08-27. Les 37 identifiants historiques
restent fermes dans `closed.md`; `STRUCT-011` a ete ouvert apres verification
de la portee effective du codegen des bindings equity et `STRUCT-012` isole les
recettes d'exercice anticipe qui ne satisfont pas encore le nouveau contrat.

Une regression doit reprendre son identifiant historique, etre retiree de
`closed.md` et etre reinscrite ici avec la nouvelle preuve.

## Project structure — generation des recettes equity

### STRUCT-011 — Deriver les recettes de prix equity du manifeste canonique

- **Nature :** ouvert le 2026-08-27 a la demande utilisateur.
- **Signature initiale :** `pricing_bindings` regenerait les paires `.cu/.cuh`
  et leur inscription CMake, mais les
  `catalog/model/equity/*/prices/*/generator.cpp` restaient materialises
  manuellement. CMake les decouvrait et le checker imposait le runner commun
  sans garantir leur completude, leur reproductibilite ou l'absence de derive
  de leurs profils numeriques.
- **Etat de remediation :** le manifeste type croise maintenant 18 modeles,
  21 produits et 29 variantes publiees. Il regenere 756 bindings `.cu/.cuh`,
  522 recettes de prix ordinaires et leur fragment CMake. Quatre templates de
  recette couvrent closed form, Markov fixe/exact, rough N-facteurs et
  Volterra FFT; l'orchestrateur commun possede chargement, workspace RAII,
  timings, JSON, YAML et validation. Les particularites gap loader,
  geometric-Asian analytique et geometrie N-facteurs sont des champs du
  manifeste, pas des sources manuelles.
- **Reste ouvert :** les huit recettes American/LSM sont les seules recettes
  de prix equity non reproductibles par cette generation. Conformement a la
  consigne utilisateur, le constat parent reste ouvert tant que l'exception
  `STRUCT-012` subsiste, meme si la matrice ordinaire est corrigee.
- **Correction attendue :** une description typee modele/produit/variante doit
  generer les recettes ordinaires pour les backends analytique, markovien a
  pas fixe, transition exacte, rough N-facteurs et Volterra FFT. L'execution
  de ces recettes doit rester proprietaire du JSON de prix et du YAML catalogue
  contenant les timings mesures. Les exceptions algorithmiques non exprimables
  par ces contrats doivent recevoir un constat distinct et rester explicitement
  controlees.
- **Preuve requise :** regeneration temporaire comparee au tree, matrice de
  recettes attendue complete, checker refusant toute recette ordinaire
  manuscrite, builds avec et sans mathDx, tests des orchestrateurs et execution
  CUDA representative de chaque backend disponible.
- **Preuve courante :** E22 couvre la comparaison exacte des 1 279 sorties,
  le checker des 522 recettes, le build exhaustif `price_generators` avec
  mathDx, un build sans mathDx et six executions CUDA temporaires incluant les
  cinq backends et le geometric-Asian analytique.
- **Condition de fermeture :** aucune recette equity non American ordinaire
  n'est ajoutee ou modifiee hors manifeste/templates et chaque exception
  restante possede un identifiant, une justification et une condition de
  reouverture propres.

### STRUCT-012 — Declarer et generer les recettes American/LSM

- **Nature :** ouvert le 2026-08-27 comme exception explicite de `STRUCT-011`.
- **Signature :** huit recettes American pour Bates, Heston,
  Normal-Inverse-Gaussian et Variance-Gamma possedent un workspace de paths,
  une regression backward, des diagnostics `RegressionStatus` et une
  publication LSM qui ne se reduisent pas au contrat d'une recette Monte Carlo
  scalaire. Elles restent ecrites a la main et constituent les seules recettes
  de prix equity hors codegen.
- **Correction attendue :** definir un `AmericanRecipeSpec`, un template de
  recette LSM et un orchestrateur RAII qui possedent explicitement paths,
  regression, exercice et diagnostics sans affaiblir les garanties FP64 et
  fail-closed. Les huit sources doivent alors etre reproductibles depuis le
  manifeste.
- **Preuve requise :** comparaison codegen des huit recettes, tests LSM CUDA,
  invariance des prix/diagnostics et checker ne contenant plus d'echappatoire
  equity.
- **Condition de fermeture :** aucune recette American ne gere directement
  ses ressources CUDA ou ne vit hors du manifeste de generation.
